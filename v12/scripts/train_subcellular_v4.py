"""Gene-set slots with cytoplasm exchange, nuclear feedback and one masked CE.

Run on an explicitly selected compute allocation. This entry point never requests
resources and never resumes or overwrites an existing run implicitly. Fixed global
windows and logical-minibatch RNG seeds preserve the V3 stochastic protocol.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
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
from spatial_pretrain.models.subcellular_v4 import SubcellularV4
from spatial_pretrain.training.subcellular_v4 import losses

MODEL_VERSION = 4
ARCHITECTURE = 'geneset_feedback_v4'
SOURCE_FILES = (
    'scripts/train_subcellular_v4.py',
    'src/spatial_pretrain/data/subcellular_v3.py',
    'src/spatial_pretrain/models/subcellular_v4.py',
    'src/spatial_pretrain/training/subcellular_v4.py',
    'src/spatial_pretrain/models/scgpt_backbone.py',
    'src/spatial_pretrain/models/subcellular_v3.py',
    'src/spatial_pretrain/models/subcellular_v2.py',
    'src/spatial_pretrain/data/subcellular_v2.py',
)


class GlobalBatchSampler(Sampler):
    """V3's allocation-independent shuffled windows; resume at optimizer boundary."""

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
        if int(epoch) != epoch or epoch < 0 or not 0 <= start_update <= self.updates:
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
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_config(cfg):
    """Fail before loading assets or writing output on a different experiment."""
    if cfg.get('loss') != {'type': 'masked_bin_cross_entropy'}:
        raise ValueError('V4 requires exactly one loss: {type: masked_bin_cross_entropy}')
    if cfg['model'].get('backbone_kind', 'scgpt') != 'scgpt':
        raise ValueError('V4 requires the ordinary scGPT backbone')
    training = cfg['training']
    if training['backbone_trainable_layers'] != 0:
        raise ValueError('Initial V4 is frozen-backbone; unfreezing requires an explicit separate revision')
    for key in ('epochs', 'global_batch_size', 'batch_size', 'checkpoint_every', 'resume_every', 'log_every'):
        value = training[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'training.{key} must be a positive integer')
    if training['num_workers'] < 0:
        raise ValueError('num_workers must be nonnegative')
    if not 0 <= training['warmup_ratio'] <= 1 or not 0 <= training['min_lr_ratio'] <= 1:
        raise ValueError('warmup_ratio and min_lr_ratio must lie in [0, 1]')
    if training['lr_new'] <= 0 or training['lr_backbone'] < 0 or training['grad_clip_norm'] <= 0:
        raise ValueError('Invalid learning rate or gradient clipping threshold')
    if int(cfg['model'].get('neighbors', 16)) != int(cfg['data']['neighbors']):
        raise ValueError('Model and data neighbor budgets must agree')


def prepare_run_directory(run, *, resume):
    """No destructive overwrite and no implicit adoption of another run."""
    run = Path(run)
    if resume:
        if not (run / 'resume.pt').is_file():
            raise FileNotFoundError('--resume requires an existing V4 resume.pt')
    elif run.exists() and any(run.iterdir()):
        raise FileExistsError('Output directory is not empty; use an explicit new directory or --resume')
    if any((run / marker).exists() for marker in ('STOP_REQUESTED', 'PAUSE_REQUESTED')):
        raise RuntimeError('Run paused for review; inspect the marker before resuming')
    run.mkdir(parents=True, exist_ok=True)
    (run / 'checkpoints').mkdir(exist_ok=True)


def validate_resume(state, cfg, sources, arch_config, *, micro_batch, updates_per_epoch,
                    selected_centers=None, backbone_provenance=None):
    """A V3 checkpoint, modified bins/loss/layers, or stale cursor cannot resume."""
    if state.get('model_version') != MODEL_VERSION or state.get('architecture') != ARCHITECTURE:
        raise ValueError('Only an explicit V4 gene-set feedback checkpoint can resume; V3 is incompatible')
    if state.get('arch_config') != arch_config:
        raise ValueError('Resume architecture mismatch (slots, widths, heads, layers or spatial settings)')
    if state.get('config_sha256') != config_hash(cfg) or state.get('config') != cfg:
        raise ValueError('Resume config mismatch; record and fork an explicit revision')
    if state.get('source_sha256') != sources:
        raise ValueError('Resume source mismatch; record and fork an explicit revision')
    if state.get('micro_batch_size') != micro_batch:
        raise ValueError('Keep the validated minibatch size on resume to preserve loss reduction and dropout')
    if state.get('global_batch_size') != cfg['training']['global_batch_size']:
        raise ValueError('Resume global batch mismatch')
    if selected_centers is not None and state.get('selected_centers') != selected_centers:
        raise ValueError('Resume dataset size mismatch')
    if backbone_provenance is not None and state.get('backbone_provenance') != backbone_provenance:
        raise ValueError('Resume ordinary-scGPT asset provenance mismatch')
    if state.get('phase') != 'frozen_backbone':
        raise ValueError('Resume phase mismatch')
    step, epoch, next_update = (state.get(key) for key in ('step', 'epoch', 'next_update'))
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (step, epoch, next_update)):
        raise ValueError('Resume must contain an integer optimizer boundary')
    epochs = cfg['training']['epochs']
    if not 0 <= epoch <= epochs or not 0 <= next_update < updates_per_epoch:
        raise ValueError('Resume epoch/update cursor is out of bounds')
    if step != epoch * updates_per_epoch + next_update:
        raise ValueError('Resume optimizer step and epoch/update cursor disagree')
    complete = epoch == epochs and next_update == 0
    if bool(state.get('complete')) != complete or (epoch == epochs and next_update):
        raise ValueError('Resume completion flag disagrees with the optimizer boundary')
    if 'model' not in state or 'optimizer' not in state:
        raise ValueError('Resume requires model and optimizer state, not an evaluation-only checkpoint')
    return step, epoch, next_update


def logical_forward_seed(seed, epoch, update, logical_micro):
    return (seed + epoch * 100000007 + update * 1009 + logical_micro) % (2**63 - 1)


def lr_multiplier(next_step, total_updates, training):
    """Keep the full-epoch schedule unchanged for bounded smoke/pilot execution."""
    warmup = max(1, round(total_updates * training['warmup_ratio']))
    if next_step <= warmup:
        return next_step / warmup
    progress = min(1., (next_step - warmup) / max(1, total_updates - warmup))
    return training['min_lr_ratio'] + (1 - training['min_lr_ratio']) * .5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Explicitly resume this output directory')
    parser.add_argument('--run-update-limit', type=int, default=0,
                        help='Save and return after N additional optimizer updates; 0 means no limit')
    parser.add_argument('--initialize-only', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Physical minibatch for an explicit capacity test; global batch stays fixed')
    args = parser.parse_args()
    if args.run_update_limit < 0 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('Update limit must be nonnegative and minibatch positive')
    cfg = yaml.safe_load(args.config.resolve().read_text())
    validate_config(cfg)
    os.chdir(ROOT)
    training = cfg['training']
    rank, world, local = (int(os.environ.get(k, default)) for k, default in
                         [('RANK', '0'), ('WORLD_SIZE', '1'), ('LOCAL_RANK', '0')])
    if not torch.cuda.is_available():
        raise RuntimeError('V4 training requires CUDA on an explicitly selected compute allocation')
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group('nccl')
    run_lock = None
    try:
        run = Path(cfg['output_dir'])
        directory_error = [None]
        if rank == 0:
            try:
                prepare_run_directory(run, resume=args.resume)
                run_lock = (run / '.trainer.lock').open('a')
                fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (OSError, RuntimeError) as error:
                directory_error[0] = f'{type(error).__name__}: {error}'
        if world > 1:
            dist.broadcast_object_list(directory_error, src=0)
        if directory_error[0]:
            raise RuntimeError(directory_error[0])
        _train(cfg, args, run, rank, world)
    finally:
        if run_lock is not None:
            run_lock.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(cfg, args, run, rank, world):
    training = cfg['training']
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    seed = int(cfg['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    micro_batch = args.batch_size if args.batch_size is not None else training['batch_size']
    dataset = V3Dataset(cfg, split='train')
    sampler = GlobalBatchSampler(len(dataset), training['global_batch_size'], micro_batch, world, rank, seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=training['num_workers'],
                        collate_fn=collate_v3, pin_memory=True, persistent_workers=False)
    total_updates = sampler.updates * training['epochs']
    fingerprint, sources = config_hash(cfg), source_hash()
    model = SubcellularV4(cfg['assets_dir'], **cfg['model']).cuda()
    model.set_backbone_trainable_layers(0)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=training['lr_new'], weight_decay=training['weight_decay'])
    step = start_epoch = start_update = 0
    if args.resume:
        state = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
        step, start_epoch, start_update = validate_resume(
            state, cfg, sources, model.arch_config, micro_batch=micro_batch,
            updates_per_epoch=sampler.updates, selected_centers=len(dataset),
            backbone_provenance=model.backbone.provenance)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        if state['complete']:
            if rank == 0:
                print(f'[complete] Previously completed and validated V4 run at step={step}', flush=True)
            return
        del state
    local = int(os.environ.get('LOCAL_RANK', '0'))
    wrapped = DDP(model, device_ids=[local], find_unused_parameters=True) if world > 1 else model
    began, initial_step = time.monotonic(), step
    torch.cuda.reset_peak_memory_stats()

    def barrier():
        if world > 1:
            dist.barrier()

    def save(epoch, next_update, *, complete=False, archive=False):
        barrier()
        if rank == 0:
            payload = dict(
                model=model.state_dict(), config=cfg, config_sha256=fingerprint, source_sha256=sources,
                model_version=MODEL_VERSION, architecture=ARCHITECTURE, arch_config=dict(model.arch_config),
                step=step, backbone_provenance=model.backbone.provenance, phase='frozen_backbone',
            )
            if archive:
                archive_path = run / f'checkpoints/step_{step:08d}.pt'
                if archive_path.exists():
                    raise FileExistsError(f'Refusing to replace existing archived checkpoint: {archive_path}')
                atomic_save(payload, archive_path)
            atomic_save(dict(
                **payload, optimizer=optimizer.state_dict(), epoch=epoch, next_update=next_update,
                complete=complete, physical_world_size=world, global_batch_size=training['global_batch_size'],
                micro_batch_size=micro_batch, selected_centers=len(dataset),
            ), run / 'resume.pt')
            atomic_json(dict(
                step=step, epoch=epoch, next_update=next_update, complete=complete,
                physical_world_size=world, saved_at=time.time(), config_sha256=fingerprint,
                model_version=MODEL_VERSION, architecture=ARCHITECTURE,
            ), run / 'progress.json')
            if complete:
                (run / 'TRAIN_COMPLETE').write_text(f'step={step}\n')
            print(f'[checkpoint] step={step} epoch={epoch} next_update={next_update} complete={complete}', flush=True)
        barrier()

    if rank == 0:
        schedule = dict(
            selected_centers=len(dataset), updates_per_epoch=sampler.updates, total_updates=total_updates,
            physical_world_size=world, global_batch_size=training['global_batch_size'],
            batch_per_card=micro_batch, accumulation=sampler.accumulation,
            repeated_centers_at_epoch_end=sampler.updates * training['global_batch_size'] - len(dataset),
            backbone_trainable_layers=0, trainable_parameters=sum(p.numel() for p in parameters),
            frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
            backbone_provenance=model.backbone.provenance, source_sha256=sources,
            model_version=MODEL_VERSION, architecture=ARCHITECTURE, arch_config=dict(model.arch_config),
            loss=dict(cfg['loss']), resumed=bool(args.resume), initial_step=step,
            stochastic_protocol='V3 fixed global windows; RNG per epoch/update/logical minibatch; fixed minibatch on resume',
        )
        atomic_json(schedule, run / 'schedule.json')
        (run / 'run_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        print('[schedule]', json.dumps(schedule), flush=True)
    if step == 0 and not args.resume:
        save(0, 0, archive=True)
    if args.initialize_only:
        return
    deadline = float(os.environ.get('V4_TRAIN_DEADLINE', 'inf'))
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, training['epochs']):
        dataset.set_epoch(epoch)
        epoch_start = start_update if epoch == start_epoch else 0
        sampler.seek(epoch, epoch_start)
        wrapped.train()
        # Freeze backbone dropout as well as weights; compartment inputs retain gradients.
        model.backbone.eval()
        entries = {}
        for offset, batch in enumerate(loader):
            update = epoch_start + offset // sampler.accumulation
            micro = offset % sampler.accumulation
            forward_seed = logical_forward_seed(seed, epoch, update, micro * world + rank)
            torch.manual_seed(forward_seed)
            torch.cuda.manual_seed_all(forward_seed)
            batch = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            boundary = micro + 1 == sampler.accumulation
            sync = wrapped.no_sync() if world > 1 and not boundary else contextlib.nullcontext()
            with sync, torch.autocast('cuda', dtype=torch.bfloat16):
                output = wrapped(batch)
                loss, parts = losses(output, batch, cfg, step + 1)
                finite = torch.tensor(int(torch.isfinite(loss)), device='cuda')
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(f'Nonfinite V4 masked CE at optimizer step {step + 1}')
                (loss / sampler.accumulation).backward()
            for key, value in parts.items():
                entries[key] = entries.get(key, 0.) + float(value) / sampler.accumulation
            if not boundary:
                continue
            norm = torch.nn.utils.clip_grad_norm_(parameters, training['grad_clip_norm'], error_if_nonfinite=True)
            multiplier = lr_multiplier(step + 1, total_updates, training)
            for group in optimizer.param_groups:
                group['lr'] = training['lr_new'] * multiplier
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            keys = ('total', 'masked_ce')
            summary = torch.tensor([entries[k] for k in keys], device='cuda')
            peaks = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                                 dtype=torch.float64, device='cuda')
            gradient_norm = norm.detach().float()
            if world > 1:
                dist.all_reduce(summary)
                summary /= world
                dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
                dist.all_reduce(gradient_norm, op=dist.ReduceOp.MAX)
            if rank == 0 and (step <= 3 or step % training['log_every'] == 0):
                # The host read synchronizes the update before wall-time throughput is recorded.
                loss_record, norm_record, peak_values = dict(zip(keys, summary.tolist())), float(gradient_norm), peaks.tolist()
                seconds = (time.monotonic() - began) / max(1, step - initial_step)
                record = dict(
                    step=step, total_updates=total_updates, epoch=epoch, loss=loss_record,
                    gradient_norm=norm_record, loss_finite=True, gradients_finite=True,
                    lr_new=optimizer.param_groups[0]['lr'], lr_backbone_effective=0.,
                    seconds_per_update=seconds, centers_per_second=training['global_batch_size'] / seconds,
                    eta_train_hours=max(0, total_updates - step) * seconds / 3600,
                    allocated_gib=peak_values[0] / 2**30, reserved_gib=peak_values[1] / 2**30,
                    memory_scope='maximum_over_physical_ranks_since_launch',
                    physical_world_size=world, batch_per_card=micro_batch, accumulation=sampler.accumulation,
                    global_batch_size=training['global_batch_size'], phase='frozen_backbone',
                    model_version=MODEL_VERSION, jobid=os.environ.get('SLURM_JOB_ID'),
                )
                print('[train]', json.dumps(record, allow_nan=False), flush=True)
                with (run / 'train_metrics.jsonl').open('a') as handle:
                    handle.write(json.dumps(record, allow_nan=False) + '\n')
            entries = {}
            should_stop = time.time() >= deadline or any(
                (run / marker).exists() for marker in ('STOP_REQUESTED', 'PAUSE_REQUESTED'))
            should_stop |= bool(args.run_update_limit and step - initial_step >= args.run_update_limit)
            stop = torch.tensor(int(should_stop), device='cuda')
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            epoch_end = update + 1 == sampler.updates
            complete = epoch_end and epoch + 1 == training['epochs']
            archive = step % training['checkpoint_every'] == 0 or complete or bool(stop.item())
            if archive or step % training['resume_every'] == 0 or stop.item():
                save(epoch + 1 if epoch_end else epoch, 0 if epoch_end else update + 1,
                     complete=complete, archive=archive)
            if stop.item():
                if rank == 0:
                    print(f'[paused] Saved V4 optimizer boundary at step={step}', flush=True)
                return
        start_update = 0


if __name__ == '__main__':
    main()
