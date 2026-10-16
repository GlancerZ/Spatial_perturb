"""Dual compartment CLS training with an allocation-independent global batch.

One/four physical GPUs consume identical shuffled global windows. A world-size
change resumes at a saved optimizer boundary, without repeating the old prefix.
The final window alone is padded from the epoch prefix to keep DDP reductions
well-defined. Its small repeat count is recorded in schedule.json.
"""
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
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
import yaml
from spatial_pretrain.data.subcellular_v3 import V3Dataset, collate_v3
from spatial_pretrain.models.subcellular_v3 import SubcellularV3
from spatial_pretrain.training.subcellular_v3 import losses


class GlobalBatchSampler(Sampler):
    """Partition fixed global windows into physical-rank minibatches."""
    def __init__(self, size, global_batch, micro_batch, world=1, rank=0, seed=42):
        if size < 1 or min(global_batch, micro_batch, world) < 1:
            raise ValueError('Positive dataset/batch/world sizes required')
        if global_batch % (micro_batch * world) or not 0 <= rank < world:
            raise ValueError('Global batch must divide evenly over rank minibatches')
        self.size, self.global_batch, self.micro_batch = size, global_batch, micro_batch
        self.world, self.rank, self.seed = world, rank, seed
        self.updates = math.ceil(size / global_batch)
        self.accumulation = global_batch // (micro_batch * world)
        self.epoch, self.start_update = 0, 0

    def seek(self, epoch, start_update):
        if not 0 <= start_update <= self.updates:
            raise ValueError('Invalid saved epoch update position')
        self.epoch, self.start_update = int(epoch), int(start_update)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        padding = self.updates * self.global_batch - self.size
        if padding:
            indices += (indices * math.ceil(padding / self.size))[:padding]
        for update in range(self.start_update, self.updates):
            for micro in range(self.accumulation):
                logical = micro * self.world + self.rank
                offset = update * self.global_batch + logical * self.micro_batch
                yield indices[offset:offset + self.micro_batch]

    def __len__(self):
        return (self.updates - self.start_update) * self.accumulation


def atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def atomic_json(obj, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_hash():
    names = ['scripts/train_subcellular_v3.py',
             'src/spatial_pretrain/data/subcellular_v3.py',
             'src/spatial_pretrain/models/subcellular_v3.py',
             'src/spatial_pretrain/training/subcellular_v3.py',
             'src/spatial_pretrain/models/scgpt_backbone.py',
             'src/spatial_pretrain/models/subcellular_v2.py',
             'src/spatial_pretrain/data/subcellular_v2.py']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-update-limit', type=int, default=0)
    parser.add_argument('--initialize-only', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Physical minibatch; global batch remains fixed')
    args = parser.parse_args()
    os.chdir(ROOT)
    cfg = yaml.safe_load(args.config.read_text())
    train = cfg['training']
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                         [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if world not in (1, 4):
        raise ValueError('This experiment supports one or four physical GPUs')
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    seed = int(cfg['data']['seed'])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    run = Path(cfg['output_dir'])
    run.mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints').mkdir(exist_ok=True)
    if any((run / marker).exists() for marker in ('STOP_REQUESTED', 'PAUSE_REQUESTED')):
        raise RuntimeError('Run paused for review; inspect the marker before resuming')
    micro_batch = args.batch_size or train['batch_size']
    dataset = V3Dataset(cfg, split='train')
    sampler = GlobalBatchSampler(len(dataset), train['global_batch_size'], micro_batch,
                                 world, rank, seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=train['num_workers'],
                        collate_fn=collate_v3, pin_memory=True, persistent_workers=False)
    total_updates = sampler.updates * train['epochs']
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    sources = source_hash()
    model = SubcellularV3(cfg['assets_dir'], **cfg['model']).cuda()
    layers = int(train['backbone_trainable_layers'])
    model.set_backbone_trainable_layers(layers)
    if layers != 0:
        raise ValueError('Initial V3 experiment is frozen-backbone; validated phase two needs a separate config/run')
    backbone, new = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone if name.startswith('backbone.') else new).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': backbone, 'lr': train['lr_backbone'], 'base_lr': train['lr_backbone']},
        {'params': new, 'lr': train['lr_new'], 'base_lr': train['lr_new']}],
        weight_decay=train['weight_decay'])
    step = start_epoch = start_update = 0
    if (run / 'resume.pt').exists():
        state = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
        if state['config_sha256'] != fingerprint or state['source_sha256'] != sources:
            raise ValueError('Resume code/config mismatch; record and fork an explicit revision')
        if state['micro_batch_size'] != micro_batch:
            raise ValueError('Keep the validated minibatch size on resume; changing loss reduction needs an explicit revision')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, start_epoch, start_update = state['step'], state['epoch'], state['next_update']
        if state.get('complete'):
            if rank == 0:
                (run / 'TRAIN_COMPLETE').write_text(f'step={step}\n')
            if world > 1: dist.destroy_process_group()
            return
        del state
    wrapped = DDP(model, device_ids=[local], find_unused_parameters=True) if world > 1 else model
    began = time.monotonic(); initial_step = step

    def barrier():
        if world > 1: dist.barrier()

    def save(epoch, next_update, *, complete=False, archive=False):
        barrier()
        if rank == 0:
            payload = dict(model=model.state_dict(), config=cfg, config_sha256=fingerprint,
                           source_sha256=sources, step=step, backbone_provenance=model.backbone.provenance,
                           architecture='dual_cls_binned_v3', phase='frozen_backbone')
            if archive:
                atomic_save(payload, run / f'checkpoints/step_{step:08d}.pt')
            atomic_save(dict(**payload, optimizer=optimizer.state_dict(), epoch=epoch,
                            next_update=next_update, complete=complete, physical_world_size=world,
                            global_batch_size=train['global_batch_size'], micro_batch_size=micro_batch),
                        run / 'resume.pt')
            atomic_json(dict(step=step, epoch=epoch, next_update=next_update, complete=complete,
                             physical_world_size=world, saved_at=time.time(), config_sha256=fingerprint),
                        run / 'progress.json')
            if complete: (run / 'TRAIN_COMPLETE').write_text(f'step={step}\n')
            print(f'[checkpoint] step={step} epoch={epoch} next_update={next_update} complete={complete}', flush=True)
        barrier()

    if rank == 0:
        schedule = dict(selected_centers=len(dataset), updates_per_epoch=sampler.updates,
                        total_updates=total_updates, physical_world_size=world,
                        global_batch_size=train['global_batch_size'], batch_per_card=micro_batch,
                        accumulation=sampler.accumulation,
                        repeated_centers_at_epoch_end=sampler.updates * train['global_batch_size'] - len(dataset),
                        backbone_trainable_layers=layers, trainable_parameters=sum(p.numel() for p in new + backbone),
                        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                        backbone_provenance=model.backbone.provenance, source_sha256=sources,
                        stochastic_protocol='Seeded by epoch/update/logical minibatch; same minibatch size required for matching dropout across GPU counts')
        atomic_json(schedule, run / 'schedule.json')
        (run / 'run_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        print('[schedule]', json.dumps(schedule), flush=True)
    if step == 0:
        save(0, 0, archive=True)
    if args.initialize_only:
        if world > 1: dist.destroy_process_group()
        return
    deadline = float(os.environ.get('V3_TRAIN_DEADLINE', 'inf'))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, train['epochs']):
        dataset.set_epoch(epoch)
        epoch_start = start_update if epoch == start_epoch else 0
        sampler.seek(epoch, epoch_start)
        wrapped.train()
        # A frozen reference backbone should not inject dropout noise into CLS.
        # Its forward remains differentiable w.r.t. the new compartment vectors.
        model.backbone.eval()
        entries = {}
        for offset, batch in enumerate(loader):
            update = epoch_start + offset // sampler.accumulation
            micro = offset % sampler.accumulation
            logical_micro = micro * world + rank
            forward_seed = (seed + epoch * 100000007 + update * 1009 + logical_micro) % (2**63 - 1)
            torch.manual_seed(forward_seed); torch.cuda.manual_seed_all(forward_seed)
            batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            boundary = micro + 1 == sampler.accumulation
            sync = wrapped.no_sync() if world > 1 and not boundary else contextlib.nullcontext()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                output = wrapped(batch)
                loss, parts = losses(output, batch, cfg, step + 1)
                finite = torch.tensor(int(torch.isfinite(loss)), device='cuda')
                if world > 1: dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item(): raise FloatingPointError(f'Nonfinite V3 loss at step {step}')
                (loss / sampler.accumulation).backward()
            for key, value in parts.items():
                entries[key] = entries.get(key, 0.) + float(value) / sampler.accumulation
            if not boundary: continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train['grad_clip_norm'], error_if_nonfinite=True)
            warmup = max(1, round(total_updates * train['warmup_ratio']))
            if step + 1 <= warmup:
                multiplier = (step + 1) / warmup
            else:
                progress = min(1., (step + 1 - warmup) / max(1, total_updates - warmup))
                multiplier = train['min_lr_ratio'] + (1 - train['min_lr_ratio']) * .5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups: group['lr'] = group['base_lr'] * multiplier
            optimizer.step(); optimizer.zero_grad(set_to_none=True); step += 1
            keys = ['total', 'compartment', 'joint']
            summary = torch.tensor([entries[k] for k in keys], device='cuda')
            if world > 1: dist.all_reduce(summary); summary /= world
            if rank == 0 and (step <= 3 or step % train['log_every'] == 0):
                elapsed = time.monotonic() - began
                seconds = elapsed / max(1, step - initial_step)
                record = dict(step=step, total_updates=total_updates, epoch=epoch,
                              loss=dict(zip(keys, summary.tolist())), gradient_norm=float(norm),
                              lr_new=optimizer.param_groups[1]['lr'], lr_backbone_effective=0.,
                              seconds_per_update=seconds, eta_train_hours=max(0, total_updates-step)*seconds/3600,
                              allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                              physical_world_size=world, batch_per_card=micro_batch,
                              accumulation=sampler.accumulation, global_batch_size=train['global_batch_size'],
                              phase='frozen_backbone', jobid=os.environ.get('SLURM_JOB_ID'))
                print('[train]', json.dumps(record), flush=True)
                with (run / 'train_metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
            entries = {}
            should_stop = time.time() >= deadline or any((run / m).exists() for m in ('STOP_REQUESTED', 'PAUSE_REQUESTED'))
            should_stop |= world == 1 and (run / 'four_gpu_intent.json').exists()
            should_stop |= bool(args.run_update_limit and step - initial_step >= args.run_update_limit)
            stop = torch.tensor(int(should_stop), device='cuda')
            if world > 1: dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            epoch_end = update + 1 == sampler.updates
            complete = epoch_end and epoch + 1 == train['epochs']
            archive = step % train['checkpoint_every'] == 0 or complete
            if archive or step % train['resume_every'] == 0 or stop.item():
                save(epoch + 1 if epoch_end else epoch, 0 if epoch_end else update + 1,
                     complete=complete, archive=archive)
            if stop.item():
                if rank == 0: print(f'[paused] Saved optimizer boundary at step={step}', flush=True)
                if world > 1: dist.destroy_process_group()
                return
        start_update = 0
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__':
    main()
