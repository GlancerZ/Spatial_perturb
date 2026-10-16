"""Resume an existing fully-unfrozen logical four-rank run on one physical GPU.

Each optimizer update serially processes the original four rank-local batches,
with independent restored RNG streams and loss / 4. Sample grouping, rank-local
loss reductions and the LR/optimizer schedule are preserved. Floating-point
summation order differs from NCCL DDP; bitwise-identical weights are not claimed.
The caller owns the exclusive training/handoff lock.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, Sampler
import yaml
from spatial_pretrain.data.subcellular_v2 import V2Dataset, collate_v2
from spatial_pretrain.models.subcellular_v2 import SubcellularV2
from train_subcellular_v2 import atomic_save, losses


class SerialRankBatchSampler(Sampler):
    """Interleave each original rank's batch r0,r1,r2,r3 for every logical update."""
    def __init__(self, dataset, batch_size, logical_world_size, seed, epoch, start_micro=0):
        self.batch_size = int(batch_size)
        self.logical_world_size = int(logical_world_size)
        self.rank_indices = []
        for rank in range(self.logical_world_size):
            sampler = DistributedSampler(dataset, num_replicas=self.logical_world_size, rank=rank,
                                         shuffle=True, seed=seed, drop_last=False)
            sampler.set_epoch(epoch)
            self.rank_indices.append(list(sampler))
        lengths = {len(indices) for indices in self.rank_indices}
        if len(lengths) != 1:
            raise ValueError('Logical ranks have different sampler lengths')
        self.updates_per_epoch = math.ceil(len(self.rank_indices[0]) / self.batch_size)
        self.start_micro = int(start_micro)
        if not 0 <= self.start_micro <= self.updates_per_epoch:
            raise ValueError('Resume micro_step is outside the epoch sampler')

    def __iter__(self):
        for micro in range(self.start_micro, self.updates_per_epoch):
            offset = micro * self.batch_size
            for rank in range(self.logical_world_size):
                yield self.rank_indices[rank][offset:offset + self.batch_size]

    def __len__(self):
        return (self.updates_per_epoch - self.start_micro) * self.logical_world_size


def capture_rng(cuda=True):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if cuda else None)


def restore_rng(state, cuda=True):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if cuda:
        torch.cuda.set_rng_state(state['cuda'])


def accumulate_virtual_ranks(model, batches, rank_rngs, loss_function, *, cuda=True,
                             autocast_factory=contextlib.nullcontext, move_batch=lambda value: value):
    """Accumulate one logical update; caller clips/steps/saves only after return."""
    summary = {}
    for rank in range(len(rank_rngs)):
        # Dataset/collation is deterministic per index/epoch. Loading is outside
        # the virtual rank stream, matching original restore-after-fetch behavior.
        batch = move_batch(next(batches))
        restore_rng(rank_rngs[rank], cuda=cuda)
        with autocast_factory():
            output = model(batch)
            loss, parts = loss_function(output, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite serial loss at virtual rank {rank}')
            (loss / len(rank_rngs)).backward()
        rank_rngs[rank] = capture_rng(cuda=cuda)
        for key, value in parts.items():
            summary[key] = summary.get(key, 0.) + float(value) / len(rank_rngs)
        del output, loss, batch
    return summary


def schedule_multiplier(step, total_updates, training):
    warmup = max(1, round(total_updates * training['warmup_ratio']))
    if step <= warmup:
        return step / warmup
    progress = min(1., (step - warmup) / max(1, total_updates - warmup))
    return training['min_lr_ratio'] + (1 - training['min_lr_ratio']) * .5 * (1 + math.cos(math.pi * progress))


def make_resume_payload(model, optimizer, cfg, fingerprint, step, epoch, micro_step, rank_rngs, *, complete=False):
    logical_world = int(cfg['training']['expected_world_size'])
    if len(rank_rngs) != logical_world:
        raise ValueError('Checkpoint must retain one RNG stream per logical rank')
    return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), config=cfg,
                config_sha256=fingerprint, step=step, epoch=epoch, micro_step=micro_step,
                complete=complete, world_size=logical_world, logical_world_size=logical_world,
                physical_world_size=1, execution_mode='serial_logical_ranks',
                rng_by_rank=rank_rngs, backbone_provenance=model.backbone.provenance)


def validate_resume(cfg, state):
    train = cfg['training']
    if train['expected_world_size'] != 4 or train['batch_size'] != 16 or train['grad_accum_steps'] != 1:
        raise ValueError('Bridge requires the original logical world=4, rank batch=16, accumulation=1')
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if state['config_sha256'] != fingerprint or state['world_size'] != 4:
        raise ValueError('Original configuration fingerprint/logical world size differs from resume')
    if state['step'] < train['unfreeze_step']:
        raise ValueError('Bridge only supports a fully-unfrozen backbone checkpoint')
    if len(state['rng_by_rank']) != 4:
        raise ValueError('Resume lacks all four logical rank RNG states')
    for rank_state in state['rng_by_rank']:
        if set(('python', 'numpy', 'torch', 'cuda')) - set(rank_state) or rank_state['cuda'] is None:
            raise ValueError('Resume lacks a complete CPU/CUDA rank RNG state')
    if train['checkpoint_every'] != 100 or train['resume_every'] != 25:
        raise ValueError('Bridge preserves the requested original 100-step archives / 25-step resume cadence')
    return fingerprint


def stop_reason(run, step, initial_step, limit, deadline, training):
    for marker in ('SERIAL_STOP_REQUESTED', 'STOP_REQUESTED', 'PAUSE_REQUESTED'):
        if (run / marker).exists():
            return marker
    if time.time() >= deadline:
        return 'allocation_deadline'
    if limit and step - initial_step >= limit:
        return 'segment_update_limit'
    if training.get('max_steps', 0) and step >= training['max_steps']:
        return 'configured_step_limit'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-update-limit', type=int, default=0)
    args = parser.parse_args()
    if args.run_update_limit < 0:
        parser.error('--run-update-limit cannot be negative')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Run this bridge as one process, never with four-rank torchrun')
    os.chdir(ROOT)
    cfg = yaml.safe_load(args.config.read_text())
    train = cfg['training']
    run = Path(cfg['output_dir'])
    resume_path = run / 'resume.pt'
    if not resume_path.is_file():
        raise FileNotFoundError('Serial bridge cannot initialize a new run; original resume.pt is required')
    for marker in ('STOP_REQUESTED', 'PAUSE_REQUESTED', 'SERIAL_STOP_REQUESTED'):
        if (run / marker).exists():
            raise RuntimeError(f'Run is stopped: {marker}; caller must inspect before resuming')
    state = torch.load(resume_path, map_location='cpu', weights_only=False)
    fingerprint = validate_resume(cfg, state)
    if state.get('complete'):
        (run / 'TRAIN_COMPLETE').write_text(f'step={state["step"]}\n')
        print('[serial-complete] Resume already complete', flush=True)
        return
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    model = SubcellularV2(cfg['assets_dir'], **cfg['model']).cuda()
    model.set_backbone_trainable(full=True)
    pretrained, fresh = [], []
    for name, parameter in model.named_parameters():
        (pretrained if name.startswith('backbone.') else fresh).append(parameter)
    optimizer = torch.optim.AdamW([
        {'params': pretrained, 'lr': train['lr_backbone'], 'base_lr': train['lr_backbone']},
        {'params': fresh, 'lr': train['lr_new'], 'base_lr': train['lr_new']},
    ], weight_decay=train['weight_decay'])
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    if not all(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError('Backbone is not fully trainable')
    recorded = state.get('backbone_provenance', {})
    for key in ('backbone_kind', 'weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'):
        if recorded.get(key) != model.backbone.provenance.get(key):
            raise ValueError(f'Pretrained-source provenance mismatch on resume: {key}')
    step, start_epoch, start_micro = int(state['step']), int(state['epoch']), int(state['micro_step'])
    rank_rngs = state['rng_by_rank']
    del state
    dataset = V2Dataset(cfg, split='train')
    samples_per_rank = math.ceil(len(dataset) / 4)
    updates_per_epoch = math.ceil(samples_per_rank / 16)
    total_updates = updates_per_epoch * train['epochs']
    if step != start_epoch * updates_per_epoch + start_micro:
        raise ValueError('Resume step/micro_step does not match the original four-rank epoch schedule')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    deadline = float(os.environ.get('V2_TRAIN_DEADLINE', 'inf'))
    began = time.monotonic()
    initial_step = step
    print('[serial-schedule]', json.dumps(dict(step=step, micro_step=start_micro, physical_world_size=1,
        logical_world_size=4, rank_batch_size=16, global_batch_size=64, selected_centers=len(dataset),
        updates_per_epoch=updates_per_epoch, total_updates=total_updates,
        final_rank_batch_size=samples_per_rank % 16 or 16,
        execution_mode='serial_logical_ranks', config_sha256=fingerprint)), flush=True)

    def save(epoch, micro, complete=False, archive=False):
        payload = make_resume_payload(model, optimizer, cfg, fingerprint, step, epoch, micro, rank_rngs, complete=complete)
        if archive:
            archived = {key: value for key, value in payload.items() if key not in ('optimizer', 'rng_by_rank')}
            atomic_save(archived, run / 'checkpoints' / f'step_{step:08d}.pt')
        atomic_save(payload, resume_path)
        if complete:
            (run / 'TRAIN_COMPLETE').write_text(f'step={step}\n')
        print(f'[serial-checkpoint] step={step} epoch={epoch} micro={micro} complete={complete}', flush=True)

    for epoch in range(start_epoch, train['epochs']):
        dataset.set_epoch(epoch)
        if epoch > start_epoch:
            # Original rank-local DataLoader iterator creation consumes one CPU
            # base-seed draw at a new epoch. On the initial resumed epoch this
            # draw is cancelled by restore-after-fetch; subsequent epochs retain
            # it. Reproduce that behavior while using a private actual loader.
            for rank in range(4):
                restore_rng(rank_rngs[rank])
                torch.empty((), dtype=torch.int64).random_()
                rank_rngs[rank] = capture_rng()
        first_micro = start_micro if epoch == start_epoch else 0
        batch_sampler = SerialRankBatchSampler(dataset, 16, 4, cfg['data']['seed'], epoch, first_micro)
        # Iterator worker seeding uses a private generator. It never consumes any
        # saved virtual-rank CPU RNG stream; dataset augmentation is index-based.
        generator = torch.Generator().manual_seed(cfg['data']['seed'] + epoch * 1000003)
        loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_v2,
                            num_workers=train['num_workers'], pin_memory=True,
                            persistent_workers=False, generator=generator)
        batches = iter(loader)
        for micro in range(first_micro, updates_per_epoch):
            reason = stop_reason(run, step, initial_step, args.run_update_limit, deadline, train)
            if reason:
                save(epoch, micro)
                print(f'[serial-paused] step={step} reason={reason}', flush=True)
                return
            summary = accumulate_virtual_ranks(model, batches, rank_rngs,
                lambda output, batch: losses(output, batch, cfg, step + 1),
                autocast_factory=lambda: torch.autocast('cuda', dtype=torch.bfloat16),
                move_batch=lambda batch: {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                                          for key, value in batch.items()})
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train['grad_clip_norm'], error_if_nonfinite=True)
            multiplier = schedule_multiplier(step + 1, total_updates, train)
            for group in optimizer.param_groups:
                group['lr'] = group['base_lr'] * multiplier
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step - initial_step <= 3 or step % train['log_every'] == 0:
                seconds = (time.monotonic() - began) / (step - initial_step)
                record = dict(step=step, total_updates=total_updates, epoch=epoch, loss=summary,
                    gradient_norm=float(norm), lr_backbone=optimizer.param_groups[0]['lr'],
                    lr_new=optimizer.param_groups[1]['lr'], seconds_per_update=seconds,
                    eta_hours=max(0, total_updates-step)*seconds/3600,
                    allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    reserved_gib=torch.cuda.max_memory_reserved()/2**30, frozen=False,
                    physical_world_size=1, logical_world_size=4, execution_mode='serial_logical_ranks')
                with (run / 'train_metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                print('[serial-train]', json.dumps(record), flush=True)
            epoch_end = micro + 1 == updates_per_epoch
            complete = epoch_end and epoch + 1 == train['epochs']
            reason = stop_reason(run, step, initial_step, args.run_update_limit, deadline, train)
            archive = step % 100 == 0 or complete
            if archive or step % 25 == 0 or reason:
                save(epoch + 1 if epoch_end else epoch, 0 if epoch_end else micro + 1,
                     complete=complete, archive=archive)
            if reason or complete:
                print(f'[serial-{"complete" if complete else "paused"}] step={step} reason={reason}', flush=True)
                return
        del batches, loader


if __name__ == '__main__':
    main()
