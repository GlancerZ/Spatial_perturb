"""Resumable four-GPU V2 training; full selected-center epoch, independent eval."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler
import yaml
from spatial_pretrain.data.subcellular_v2 import V2Dataset, collate_v2
from spatial_pretrain.models.subcellular_v2 import SubcellularV2


class ResumeBatchSampler(Sampler):
    """Seek rank-local batches without loading already-consumed dataset items.

    ``total_batches`` remains the full epoch length, whereas ``len(self)`` is
    the number still to load. DistributedSampler retains its original shuffle,
    rank striding, padding repeats and final partial batch.
    """
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.total_batches = math.ceil(len(sampler) / self.batch_size)
        self.start_micro = 0

    def seek(self, start_micro):
        if not 0 <= start_micro <= self.total_batches:
            raise ValueError('Resume micro_step is outside the epoch sampler')
        self.start_micro = start_micro

    def __iter__(self):
        indices = list(self.sampler)
        for offset in range(self.start_micro * self.batch_size, len(indices), self.batch_size):
            yield indices[offset:offset + self.batch_size]

    def __len__(self):
        return self.total_batches - self.start_micro


def accumulation_window(micro, epoch_micro_batches, accumulation):
    """Return divisor and update boundary using absolute, full-epoch positions."""
    window_start = micro - micro % accumulation
    window_size = min(accumulation, epoch_micro_batches - window_start)
    boundary = (micro + 1) % accumulation == 0 or micro + 1 == epoch_micro_batches
    return window_size, boundary


def reconstruction_loss(pred, batch, mask=None):
    valid = batch['query_valid_mask'].bool()
    if mask is not None:
        valid = valid & mask
    losses = []
    for compartment in (0, 1):
        selected = valid & batch['query_compartment_ids'].eq(compartment)
        count = selected.sum(-1)
        bce = F.binary_cross_entropy_with_logits(pred[..., 0].float(),
            batch['target_nonzero'].float(), reduction='none')
        positive = selected & batch['target_nonzero'].bool()
        value = F.huber_loss(pred[..., 1].float(), batch['target_log_counts'].float(), reduction='none')
        per_cell = (bce*selected).sum(-1)/count.clamp_min(1)
        per_cell = per_cell + (value*positive).sum(-1)/positive.sum(-1).clamp_min(1)
        losses.append((per_cell, count.gt(0)))
    values = torch.stack([v for v, _ in losses], -1)
    present = torch.stack([v for _, v in losses], -1)
    per_cell = (values*present).sum(-1)/present.sum(-1).clamp_min(1)
    cells = present.any(-1)
    return (per_cell*cells).sum()/cells.sum().clamp_min(1)


def losses(output, batch, cfg, step):
    gene = reconstruction_loss(output['gene_pred'], batch)
    representation = 0.5*(reconstruction_loss(output['intrinsic_pred'], batch)
                           + reconstruction_loss(output['joint_pred'], batch))
    neighbor = reconstruction_loss(output['neighbor_pred'], batch, batch['neighbor_query_mask'])
    ramp = min(1., step/max(1, cfg['training']['spatial_ramp_steps']))
    total = cfg['loss']['gene']*gene + cfg['loss']['repr']*representation + ramp*cfg['loss']['neighbor']*neighbor
    return total, dict(gene=float(gene.detach()), representation=float(representation.detach()),
                       neighbor=float(neighbor.detach()), total=float(total.detach()))


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name+'.partial')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-update-limit', type=int, default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = yaml.safe_load(args.config.read_text())
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                          [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world != cfg['training']['expected_world_size']:
        raise ValueError(f'Expected {cfg["training"]["expected_world_size"]} GPUs, got {world}')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    seed = cfg['data']['seed'] + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    run = Path(cfg['output_dir']); run.mkdir(parents=True, exist_ok=True)
    (run/'checkpoints').mkdir(exist_ok=True)
    if (run/'STOP_REQUESTED').exists() or (run/'PAUSE_REQUESTED').exists():
        raise RuntimeError('Run stopped/paused for review; inspect the marker before resuming')
    dataset = V2Dataset(cfg, split='train')
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=cfg['data']['seed'])
    train = cfg['training']
    batch_sampler = ResumeBatchSampler(sampler, train['batch_size'])
    loader = DataLoader(dataset, batch_sampler=batch_sampler,
                        num_workers=train['num_workers'], collate_fn=collate_v2,
                        pin_memory=True, persistent_workers=False)
    epoch_micro_batches = batch_sampler.total_batches
    accumulation = train['grad_accum_steps']
    updates_per_epoch = math.ceil(epoch_micro_batches/accumulation)
    total_updates = updates_per_epoch*train['epochs']
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    model = SubcellularV2(cfg['assets_dir'], **cfg['model']).cuda()
    model.set_backbone_trainable(full=True)
    new, backbone = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith('backbone.') else new).append(parameter)
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': train['lr_backbone'], 'base_lr': train['lr_backbone']},
                                  {'params': new, 'lr': train['lr_new'], 'base_lr': train['lr_new']}],
                                 weight_decay=train['weight_decay'])
    step = start_epoch = start_micro = 0
    resumed_rng = None
    if (run/'resume.pt').exists():
        state = torch.load(run/'resume.pt', map_location='cpu', weights_only=False)
        if state['config_sha256'] != fingerprint or state['world_size'] != world:
            raise ValueError('Resume config/world size mismatch; use a separate experiment')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, start_epoch, start_micro = state['step'], state['epoch'], state['micro_step']
        resumed_rng = state['rng_by_rank'][rank]
        if state.get('complete'):
            if rank == 0:
                (run/'TRAIN_COMPLETE').write_text(f'step={step}\n')
                print('[complete] Saved epoch already complete', flush=True)
            dist.destroy_process_group(); return
        del state
    frozen = step < train['unfreeze_step']
    model.set_backbone_trainable(full=not frozen)
    wrapped = DDP(model, device_ids=[local], find_unused_parameters=True)
    if rank == 0:
        print('[schedule]', json.dumps(dict(cells_context=cfg['data']['context_cells'],
            selected_centers=len(dataset), updates_per_epoch=updates_per_epoch, total_updates=total_updates,
            world_size=world, batch_per_card=train['batch_size'], accumulation=accumulation,
            backbone_parameters=sum(p.numel() for p in backbone), new_parameters=sum(p.numel() for p in new))), flush=True)
        (run/'run_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        (run/'schedule.json').write_text(json.dumps(dict(selected_centers=len(dataset),
            updates_per_epoch=updates_per_epoch, total_updates=total_updates, world_size=world,
            batch_per_card=train['batch_size'], grad_accum_steps=accumulation,
            backbone_provenance=model.backbone.provenance), indent=2))
        if step == 0 and not (run/'checkpoints/step_00000000.pt').exists():
            atomic_save(dict(model=model.state_dict(), config=cfg, step=0, config_sha256=fingerprint,
                             backbone_provenance=model.backbone.provenance),
                        run/'checkpoints/step_00000000.pt')
    dist.barrier()
    deadline = float(os.environ.get('V2_TRAIN_DEADLINE', 'inf'))
    began = time.monotonic(); initial_step = step
    optimizer.zero_grad(set_to_none=True)

    def save(epoch, micro, complete=False, archive=False):
        states = [None]*world if rank == 0 else None
        dist.gather_object(rng_state(), states, dst=0)
        if rank == 0:
            payload = dict(model=model.state_dict(), config=cfg, config_sha256=fingerprint, step=step,
                           backbone_provenance=model.backbone.provenance)
            if archive:
                atomic_save(payload, run/f'checkpoints/step_{step:08d}.pt')
            atomic_save(dict(**payload, optimizer=optimizer.state_dict(), world_size=world,
                            epoch=epoch, micro_step=micro, complete=complete, rng_by_rank=states), run/'resume.pt')
            if complete:
                # Publish completion after both final checkpoint and resume state,
                # even when an allocation deadline coincides with the last update.
                (run/'TRAIN_COMPLETE').write_text(f'step={step}\n')
            print(f'[checkpoint] step={step} epoch={epoch} micro={micro} complete={complete}', flush=True)
        dist.barrier()

    for epoch in range(start_epoch, train['epochs']):
        dataset.set_epoch(epoch); sampler.set_epoch(epoch)
        epoch_start_micro = start_micro if epoch == start_epoch else 0
        batch_sampler.seek(epoch_start_micro)
        if epoch_start_micro % accumulation:
            raise ValueError('Resume micro_step must be at an optimizer-update boundary')
        wrapped.train()
        entries = {}
        for micro, batch in enumerate(loader, start=epoch_start_micro):
            if resumed_rng is not None:
                # Keep restore-after-fetch semantics: DataLoader iterator setup
                # consumes its usual worker-base-seed draw before this restore.
                # V2Dataset uses private index/epoch RNGs; skipped items need not
                # be materialized to recover the saved model RNG stream.
                restore_rng(resumed_rng); resumed_rng = None
            if frozen and step >= train['unfreeze_step']:
                dist.barrier(); del wrapped
                model.set_backbone_trainable(full=True)
                wrapped = DDP(model, device_ids=[local], find_unused_parameters=True)
                frozen = False
                if rank == 0: print(f'[unfreeze] All backbone layers trainable at step={step}', flush=True)
            batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            window_size, boundary = accumulation_window(micro, epoch_micro_batches, accumulation)
            context = contextlib.nullcontext() if boundary else wrapped.no_sync()
            with context, torch.autocast('cuda', dtype=torch.bfloat16):
                output = wrapped(batch)
                loss, parts = losses(output, batch, cfg, step+1)
                finite = torch.tensor(int(torch.isfinite(loss)), device='cuda')
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item(): raise FloatingPointError(f'Nonfinite V2 loss at step {step}')
                (loss/window_size).backward()
            for key, value in parts.items(): entries[key] = entries.get(key, 0.)+value/window_size
            if not boundary: continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train['grad_clip_norm'], error_if_nonfinite=True)
            warmup = max(1, round(total_updates*train['warmup_ratio']))
            if step+1 <= warmup:
                multiplier = (step+1)/warmup
            else:
                progress = min(1., (step+1-warmup)/max(1, total_updates-warmup))
                multiplier = train['min_lr_ratio']+(1-train['min_lr_ratio'])*.5*(1+math.cos(math.pi*progress))
            for group in optimizer.param_groups: group['lr'] = group['base_lr']*multiplier
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            summary = torch.tensor([entries[k] for k in ['total', 'gene', 'representation', 'neighbor']], device='cuda')
            dist.all_reduce(summary); summary /= world
            if rank == 0 and (step <= 3 or step%train['log_every'] == 0):
                elapsed = time.monotonic()-began
                seconds = elapsed/max(1, step-initial_step)
                record = dict(step=step, total_updates=total_updates, epoch=epoch,
                    loss=dict(zip(['total', 'gene', 'representation', 'neighbor'], summary.tolist())),
                    gradient_norm=float(norm), lr_backbone=optimizer.param_groups[0]['lr'],
                    lr_new=optimizer.param_groups[1]['lr'], seconds_per_update=seconds,
                    eta_hours=max(0, total_updates-step)*seconds/3600,
                    allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30, frozen=frozen)
                print('[train]', json.dumps(record), flush=True)
                with (run/'train_metrics.jsonl').open('a') as handle: handle.write(json.dumps(record)+'\n')
            entries = {}
            stop = torch.tensor(int(time.time() >= deadline or (run/'STOP_REQUESTED').exists()
                or (run/'PAUSE_REQUESTED').exists() or (args.run_update_limit and step-initial_step >= args.run_update_limit)
                or (train.get('max_steps', 0) and step >= train['max_steps'])), device='cuda')
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            epoch_end = micro+1 == epoch_micro_batches
            complete = epoch_end and epoch+1 == train['epochs']
            archive = step%train['checkpoint_every'] == 0 or complete
            if archive or step%train['resume_every'] == 0 or stop.item():
                save(epoch+1 if epoch_end else epoch, 0 if epoch_end else micro+1,
                     complete=complete, archive=archive)
            if stop.item():
                if rank == 0: print(f'[paused] step={step}; recoverable state saved', flush=True)
                dist.destroy_process_group(); return
        start_micro = 0
    if rank == 0:
        (run/'TRAIN_COMPLETE').write_text(f'step={step}\n')
        print(f'[complete] full selected-center epoch step={step}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__': main()
