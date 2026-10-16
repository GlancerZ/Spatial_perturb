"""Four-GPU BF16 smoke test of V2 DDP freeze, unfreeze and optimizer continuity.

Run on an allocation with torchrun --standalone --nproc_per_node=4. Uses tiny
synthetic cell patches and the actual pretrained weights/loss implementation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

from qa_subcellular_v2 import synthetic_batch, gradient_norm
from train_subcellular_v2 import losses
from spatial_pretrain.models.subcellular_v2 import SubcellularV2


def assert_rank_agreement(model, world):
    """Check full-state checksums and exact representative parameter slices."""
    parameters = list(model.parameters())
    checksum = torch.stack([parameter.detach().double().sum() for parameter in parameters]).sum().reshape(1)
    gathered_checksum = [torch.empty_like(checksum) for _ in range(world)]
    dist.all_gather(gathered_checksum, checksum)
    for value in gathered_checksum[1:]:
        torch.testing.assert_close(value, gathered_checksum[0], atol=0, rtol=0)
    first = next(model.backbone.transformer_encoder.layers[0].parameters())
    head = next(model.repr_decoder.parameters())
    sample = torch.cat([first.detach().flatten()[:128], head.detach().flatten()[:128],
                        model.compartment_embedding.weight.detach().flatten()[:128]])
    gathered_sample = [torch.empty_like(sample) for _ in range(world)]
    dist.all_gather(gathered_sample, sample)
    for value in gathered_sample[1:]:
        torch.testing.assert_close(value, gathered_sample[0], atol=0, rtol=0)
    return dict(parameter_checksum=float(checksum.item()),
                identical_rank_checksums=True, identical_parameter_slices=True,
                parameter_slice_elements=sample.numel())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/v2_ddp_qa.json')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    rank, local, world = (int(os.environ[name]) for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 4:
        raise ValueError(f'This QA must exercise all four ranks, got {world}')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=10))
    try:
        torch.set_num_threads(4)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(17931)
        torch.cuda.manual_seed_all(17931)
        model = SubcellularV2(cfg['assets_dir'], **cfg['model']).cuda()
        if model.freeze_layers < 1:
            raise ValueError('Need at least one frozen layer to test the transition')
        # Match production: optimizer owns all parameters before partial freeze.
        model.set_backbone_trainable(full=True)
        pretrained, fresh = [], []
        for name, parameter in model.named_parameters():
            (pretrained if name.startswith('backbone.') else fresh).append(parameter)
        optimizer = torch.optim.AdamW([
            {'params': pretrained, 'lr': cfg['training']['lr_backbone']},
            {'params': fresh, 'lr': cfg['training']['lr_new']},
        ], weight_decay=cfg['training']['weight_decay'])
        model.set_backbone_trainable(full=False)
        wrapped = DDP(model, device_ids=[local], find_unused_parameters=True)
        first_layer = model.backbone.transformer_encoder.layers[0]
        checks = []
        for full in (False, True):
            if full:
                dist.barrier()
                del wrapped
                model.set_backbone_trainable(full=True)
                wrapped = DDP(model, device_ids=[local], find_unused_parameters=True)
            wrapped.train()
            before = [parameter.detach().clone() for parameter in first_layer.parameters()]
            torch.manual_seed(17931 + rank + int(full) * 100)
            torch.cuda.manual_seed_all(17931 + rank + int(full) * 100)
            batch = synthetic_batch(model.vocab, torch.device('cuda', local))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = wrapped(batch)
                total, parts = losses(result, batch, cfg, max(1, cfg['training']['spatial_ramp_steps']))
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite BF16 distributed loss')
            total.backward()
            first_gradient = gradient_norm(first_layer.parameters())
            compartment_gradient = gradient_norm([model.compartment_embedding.weight])
            head_gradient = gradient_norm(model.repr_decoder.parameters())
            if compartment_gradient <= 0 or head_gradient <= 0:
                raise AssertionError('Missing distributed gradients for new modules')
            if (full and first_gradient <= 0) or (not full and first_gradient != 0):
                raise AssertionError(f'Wrong freeze-stage first-layer gradient: {first_gradient}')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip_norm'],
                                                  error_if_nonfinite=True)
            optimizer.step()
            del result
            changes = [float((parameter.detach() - previous).abs().max())
                       for parameter, previous in zip(first_layer.parameters(), before)]
            changed = max(changes)
            if (full and changed <= 0) or (not full and changed != 0):
                raise AssertionError(f'Wrong freeze-stage first-layer update: {changed}')
            rank_check = assert_rank_agreement(model, world)
            # Adam moments of the new head must continue across the rewrap.
            head_parameter = next(model.repr_decoder.parameters())
            optimizer_steps = int(optimizer.state[head_parameter]['step'].item())
            if optimizer_steps != (2 if full else 1):
                raise AssertionError('Rewrapping DDP reset the existing optimizer state')
            record = dict(full_backbone=full, rank=rank, loss=parts, gradient_norm=float(norm),
                          first_layer_gradient=first_gradient, first_layer_max_update=changed,
                          compartment_gradient=compartment_gradient, head_gradient=head_gradient,
                          new_head_optimizer_steps=optimizer_steps, rank_agreement=rank_check)
            records = [None] * world if rank == 0 else None
            dist.gather_object(record, records, dst=0)
            if rank == 0:
                checks.append(records)
                print('[ddp-qa-stage]', json.dumps(dict(full_backbone=full, ranks=records)), flush=True)
            dist.barrier()
        if rank == 0:
            report = dict(passed=True, world_size=world, mixed_precision='bfloat16',
                          stages=checks, backbone_provenance=model.backbone.provenance)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
            print(f'[ddp-qa-passed] {args.output}', flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
