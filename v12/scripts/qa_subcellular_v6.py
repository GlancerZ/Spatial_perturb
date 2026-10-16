"""Compute-only V6 acceptance: real-cache isolation, CLS identity, two updates.

Run once for each A/B configuration. This script never requests resources,
changes the experiment config, or writes model checkpoints.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from spatial_pretrain.data.subcellular_v5 import encode_whole_cell_v5
from spatial_pretrain.data.subcellular_v6 import (
    V6Dataset, collate_v6, compute_lr_features_v6, encode_context_v6,
)
from spatial_pretrain.models.subcellular_v6 import SubcellularV6
from spatial_pretrain.training.subcellular_v4 import losses
from spatial_pretrain.training.subcellular_v6_optimizer import build_optimizer
from train_subcellular_v6 import validate_config, atomic_json, lr_multiplier
from validate_subcellular_v6 import inspect_optimizer_state, optimizer_state_fingerprint
from qa_subcellular_v5 import state_digest
from evaluate_plain_scgpt_baseline import load_ordinary_collator


def critical_parameters(model):
    parameters = {
        'compartment_embedding.weight': model.compartment_embedding.weight,
        'slot_pool.queries': model.slot_pool.queries,
        'slot_pool.gene_projection.weight': model.slot_pool.gene_projection.weight,
        'slot_pool.attention.out_proj.weight': model.slot_pool.attention.out_proj.weight,
        'final_intra.attention.out_proj.weight': model.final_intra.attention.out_proj.weight,
    }
    for index, block in enumerate(model.blocks):
        parameters[f'blocks.{index}.intra.attention.out_proj.weight'] = block.intra.attention.out_proj.weight
        parameters[f'blocks.{index}.ffn.layers.2.weight'] = block.ffn.layers[-1].weight
        if model.spatial_enabled:
            parameters[f'blocks.{index}.spatial.out.weight'] = block.spatial.out.weight
            parameters[f'blocks.{index}.spatial.query.weight'] = block.spatial.query.weight
    if model.route == 'A':
        parameters['residual_projection.weight'] = model.residual_projection.weight
    else:
        parameters['self_projection.weight'] = model.self_projection.weight
        parameters['context_projection.weight'] = model.context_projection.weight
    return parameters


def raw_target_isolation(dataset, index):
    """Perturb hidden original cache values, including duplicate center rows."""
    slice_index = int(np.searchsorted(dataset.ends, index, side='right'))
    entry = dataset.slices[slice_index]
    local = index - (dataset.ends[slice_index - 1] if slice_index else 0)
    nodes = np.asarray(entry['nodes'][local], dtype=np.int64)
    rows = np.maximum(nodes, 0)
    nuclear = np.array(entry['nuclear'][rows], dtype=np.float32)
    cyto = np.array(entry['cyto'][rows], dtype=np.float32)
    # Append an explicit duplicate to exercise the physical-cell identity rule
    # even if the real cache contains no duplicate centers.
    nodes = np.r_[nodes, nodes[0]]
    nuclear = np.concatenate((nuclear, nuclear[:1]))
    cyto = np.concatenate((cyto, cyto[:1]))
    center = nodes == nodes[0]
    masks = [dict(masked_genes=np.arange(0, len(entry['genes']), 2)),
             dict(masked_compartment=0), dict(masked_compartment=1),
             dict(mask_whole_cell=True)]
    options = dict(node_indices=nodes, seed=int(dataset.data.get('seed', 42)),
                   epoch=dataset.epoch, slice_seed=entry['slice_seed'],
                   context_tokens_per_compartment=dataset.data['context_tokens_per_compartment'],
                   pad_token_id=dataset.pad_id, cls_token_id=dataset.cls_id,
                   whole_cell_max_tokens=dataset.whole_cell_max_tokens,
                   lr_pairs=dataset.lr_pairs, lr_weights=dataset.lr_weights)
    for mask in masks:
        before = encode_context_v6(nuclear, cyto, entry['genes'], **options, **mask)
        altered_n, altered_c = nuclear.copy(), cyto.copy()
        changed_value = np.log1p(np.float32(100_000))
        if 'masked_genes' in mask:
            changed = np.ix_(np.flatnonzero(center), mask['masked_genes'])
            altered_n[changed] = changed_value
            altered_c[changed] = changed_value
        elif 'masked_compartment' in mask:
            (altered_n, altered_c)[mask['masked_compartment']][center] = changed_value
        else:
            altered_n[center] = changed_value
            altered_c[center] = changed_value
        after = encode_context_v6(altered_n, altered_c, entry['genes'], **options, **mask)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=f'Hidden raw counts changed {key}')
        for key in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask'):
            np.testing.assert_array_equal(before[key][0], before[key][-1])
    return dict(mask_modes_checked=['paired_genes', 'nucleus', 'cytoplasm', 'whole_cell'],
                duplicate_center_verified=True, input_fields_verified=sorted(before))


def directed_lr_check():
    raw = np.array([[0, 10, 5], [10, 0, 5], [0, 0, 0], [8, 8, 8]])
    options = dict(node_indices=[1, 2, 3, -1], lr_pairs=np.array([[10, 11]]))
    evidence = compute_lr_features_v6(raw, [10, 11, 12], **options)
    if evidence['lr_bias'][0, 1] <= 0 or evidence['lr_bias'][1, 0] != 0:
        raise AssertionError('LR orientation must be receptor receiver row, ligand sender column')
    if evidence['lr_bias'][2, 1] != 0 or evidence['lr_coverage'][2, 1] != 1:
        raise AssertionError('An observed zero endpoint must remain known')
    visible = np.ones(raw.shape, dtype=bool)
    visible[2, 1] = False
    masked = compute_lr_features_v6(raw, [10, 11, 12], visible_mask=visible, **options)
    if masked['lr_coverage'][2, 1] != 0 or evidence['lr_coverage'][-1].any():
        raise AssertionError('Masked endpoints and PAD must carry no evidence')
    return dict(direction_verified=True, zero_unknown_distinction_verified=True, pad_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID')
            or not torch.cuda.is_available()):
        raise RuntimeError('V6 QA requires a verified Slurm GPU compute step')
    if args.output.exists() or args.batch_size < 1:
        raise ValueError('Use a new report path and a positive batch size')
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(42)
    cfg = yaml.safe_load(args.config.read_text())
    validate_config(cfg)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='subcellular_v6_real_qa_v1', status='running',
                  jobid=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
                  python=sys.executable, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(), started_at=time.time(),
                  route=cfg['model']['route'])
    atomic_json(report, args.output)
    try:
        dataset = V6Dataset(cfg, split='train')
        indices = np.unique(np.linspace(0, len(dataset) - 1, min(64, len(dataset)), dtype=np.int64))
        candidates = [(int(index), dataset[int(index)]) for index in indices]
        modes = {int(item['task_mode']) for _, item in candidates}
        if modes != {0, 1, 2}:
            raise AssertionError(f'Real-cache QA did not cover all task modes: {modes}')
        isolation = raw_target_isolation(dataset, int(indices[0]))
        lr_synthetic = directed_lr_check()
        all_coverage = np.concatenate([item['lr_coverage'].reshape(-1) for _, item in candidates])
        all_bias = np.concatenate([item['lr_bias'].reshape(-1) for _, item in candidates])
        if not np.isfinite(all_coverage).all() or not np.isfinite(all_bias).all():
            raise AssertionError('Nonfinite real-cache LR values')
        if cfg['model'].get('lr_enabled', True) and not (all_bias > 0).any():
            raise AssertionError('Frozen LR table has no positive evidence in the real-cache QA sample')
        candidates.sort(key=lambda pair: int(pair[1]['token_valid_mask'].sum()), reverse=True)
        items = [item for _, item in candidates[:args.batch_size]]
        if len(items) != args.batch_size:
            raise ValueError('Insufficient real-cache examples for requested batch size')
        batch = {key: value.cuda() for key, value in collate_v6(items, pad_token_id=dataset.pad_id).items()}
        model = SubcellularV6(cfg['assets_dir'], **cfg['model']).cuda().train()
        collator, collator_source = load_ordinary_collator(Path(cfg['assets_dir']), model.vocab, model.backbone.config)
        raw_n = np.array([0, 1, 2, 2, 7, 5, 0, 3], dtype=np.int64)
        raw_c = np.array([2, 1, 0, 2, 1, 0, 0, 3], dtype=np.int64)
        panel_ids = np.array([identifier for token, identifier in model.vocab.items()
                              if token not in ('<pad>', '<cls>', '<eoc>')][:8], dtype=np.int64)
        ours = encode_whole_cell_v5(raw_n, raw_c, panel_ids, pad_token_id=model.pad_id,
                                    cls_token_id=model.cls_id, rng=np.random.RandomState(73))
        total = raw_n + raw_c
        nonzero = total > 0
        numpy_state = np.random.get_state()
        try:
            np.random.seed(73)
            official = collator([dict(id=0, genes=torch.tensor(np.r_[model.cls_id, panel_ids[nonzero]]),
                expressions=torch.tensor(np.r_[model.pad_value, total[nonzero]], dtype=torch.float32))])
        finally:
            np.random.set_state(numpy_state)
        np.testing.assert_array_equal(ours['whole_gene_ids'], official['gene'][0].numpy())
        np.testing.assert_array_equal(ours['whole_input_bins'], official['expr'][0].numpy())
        before_hash = state_digest(model.backbone)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            initial = model(batch, return_embeddings=True)
            if model.spatial_enabled:
                incoming_counts = initial['spatial_neighbor_mask'][:, 0].sum(-1).cpu().tolist()
                report['center_incoming_neighbor_counts'] = incoming_counts
                if max(incoming_counts) < 2:
                    raise AssertionError('QA batch needs at least one center with two incoming neighbors '
                                         f'to test attention query gradients; counts={incoming_counts}')
            genes, bins, valid = (batch[key] for key in
                                  ('whole_gene_ids', 'whole_input_bins', 'whole_token_valid_mask'))
            reference = model.backbone(genes, bins, ~valid,
                torch.zeros((*genes.shape, model.backbone.dim), device='cuda'))
            reference = F.normalize(reference[:, 0].float(), dim=-1, eps=1e-8)
            for key in ('base', 'joint', 'intrinsic'):
                torch.testing.assert_close(initial['embeddings'][key], reference, atol=0, rtol=0)
            changed = dict(batch, target_bins=(batch['target_bins'] + 7).remainder(51))
            altered = model(changed, return_embeddings=True)
            torch.testing.assert_close(altered['embeddings']['joint'], reference, atol=0, rtol=0)
            if model.route == 'B':
                torch.testing.assert_close(initial['context_delta'], torch.zeros_like(initial['context_delta']), atol=0, rtol=0)
        del initial, altered, reference
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = build_optimizer(model, cfg['training'])
        total_updates = math.ceil(len(dataset) / cfg['training']['global_batch_size']) * cfg['training']['epochs']
        report['optimizer_manifest'] = optimizer.manifest
        report['optimizer_total_updates'] = total_updates
        critical = critical_parameters(model)
        projection_names = [name for name in critical if name.endswith('projection.weight')
                            and not name.startswith('slot_pool.')]
        stages = []
        torch.cuda.reset_peak_memory_stats()
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch)
                loss, parts = losses(output, batch, cfg, step)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite V6 real-cache loss')
            loss.backward()
            gradients = {name: 0. if p.grad is None else float(p.grad.detach().float().norm())
                         for name, p in critical.items()}
            if any(gradients[name] <= 0 for name in projection_names):
                raise AssertionError(f'Zero-initialized residual projection cannot learn: {gradients}')
            if step == 2 and any(not np.isfinite(value) or value <= 0 for value in gradients.values()):
                raise AssertionError(f'V6 required path has no finite gradient after projections open: {gradients}')
            if any(p.grad is not None for p in model.backbone.parameters()):
                raise AssertionError('Frozen scGPT received gradients')
            lr_gradients = {f'blocks.{i}.spatial.lr_scale_logit':
                            0. if block.spatial.lr_scale_logit.grad is None else
                            float(block.spatial.lr_scale_logit.grad.detach().float().norm())
                            for i, block in enumerate(model.blocks)}
            norm = torch.nn.utils.clip_grad_norm_(parameters, cfg['training']['grad_clip_norm'], error_if_nonfinite=True)
            optimizer.set_lr_multiplier(lr_multiplier(step, total_updates, cfg['training']))
            optimizer.step()
            stages.append(dict(step=step, loss=parts, gradient_norm=float(norm),
                               required_gradients=gradients, lr_scale_gradients=lr_gradients,
                               optimizer_lrs=optimizer.learning_rates()))
            if step == 1:
                # Round-trip real Muon momentum and AdamW moments before the
                # second QA update, using the same builder as formal training.
                snapshot = io.BytesIO()
                saved_optimizer = optimizer.state_dict()
                before_restore = inspect_optimizer_state(
                    saved_optimizer, cfg, step, total_updates, optimizer.manifest,
                    require_nonzero=False)
                digest = optimizer_state_fingerprint(saved_optimizer)
                torch.save(saved_optimizer, snapshot)
                snapshot.seek(0)
                restored = build_optimizer(model, cfg['training'])
                restored.load_state_dict(torch.load(snapshot, map_location='cpu', weights_only=False))
                restored_optimizer = restored.state_dict()
                if optimizer_state_fingerprint(restored_optimizer) != digest:
                    raise AssertionError('Muon/auxiliary AdamW state changed while restoring the QA optimizer')
                after_restore = inspect_optimizer_state(
                    restored_optimizer, cfg, step, total_updates, restored.manifest,
                    require_nonzero=False)
                if before_restore != after_restore:
                    raise AssertionError('Restored optimizer group rates, momentum or moments differ')
                report['optimizer_roundtrip'] = after_restore
                report['optimizer_roundtrip_verified'] = True
                optimizer = restored
                del saved_optimizer, restored_optimizer, snapshot, restored
        if before_hash != state_digest(model.backbone):
            raise AssertionError('Frozen plain-scGPT changed during adaptation')
        if any(not torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('V6 optimizer produced nonfinite parameters')
        report['optimizer_after_two_updates'] = inspect_optimizer_state(
            optimizer.state_dict(), cfg, 2, total_updates, optimizer.manifest)
        # Recheck target-field isolation after the residual paths are nonzero.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            final = model(batch, return_embeddings=True)
            altered = model(changed, return_embeddings=True)
            torch.testing.assert_close(final['embeddings']['joint'], altered['embeddings']['joint'], atol=0, rtol=0)
            if model.route == 'B':
                self_only = model(batch, return_embeddings=True, spatial=False)
                torch.testing.assert_close(self_only['context_delta'], torch.zeros_like(self_only['context_delta']), atol=0, rtol=0)
        report.update(status='passed', batch_size=len(items), cache_items_checked=len(candidates),
                      mask_modes=sorted(modes), step_zero_identity_verified=True,
                      unchanged_backbone_verified=True, target_field_isolation_verified=True,
                      hidden_raw_target_isolation_verified=True, raw_isolation=isolation,
                      official_binning_same_rng_verified=True, collator_provenance=collator_source,
                      backbone_provenance=model.backbone.provenance, arch_config=model.arch_config,
                      lr_prior=dataset.lr_metadata, lr_synthetic=lr_synthetic,
                      lr_real=dict(coverage_mean_including_pad_self=float(all_coverage.mean()),
                                   positive_coverage_fraction=float((all_coverage > 0).mean()),
                                   bias_mean_including_pad_self=float(all_bias.mean()),
                                   positive_bias_fraction=float((all_bias > 0).mean()),
                                   max_bias=float(all_bias.max())),
                      stages=stages, max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      limitation='Two QA updates only; no checkpoint written; not evidence of efficacy')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['finished_at'] = time.time()
        atomic_json(report, args.output)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
