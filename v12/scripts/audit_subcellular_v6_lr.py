"""Inspect LR support and actual KNN distances on training patches only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from spatial_pretrain.data.subcellular_v6 import V6Dataset
from spatial_pretrain.models.subcellular_v2 import patch_knn


def distribution(values):
    values = np.concatenate(values) if values else np.empty(0)
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite LR/distance audit')
    return dict(count=len(values), nonzero_fraction=float((values > 0).mean()) if len(values) else None,
                quantiles=dict(zip(('min', 'p25', 'median', 'p75', 'p95', 'max'),
                                   np.quantile(values, [0, .25, .5, .75, .95, 1]).tolist())) if len(values) else {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--patches-per-slice', type=int, default=8)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Audit data only inside a verified Slurm compute step')
    if args.patches_per_slice < 1:
        raise ValueError('At least one training patch per slice is required')
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = yaml.safe_load(args.config.read_text())
    dataset = V6Dataset(cfg, split='train')
    reports, start = [], 0
    for slice_index, (entry, end) in enumerate(zip(dataset.slices, dataset.ends)):
        genes = entry['genes']
        measured = np.isin(dataset.lr_pairs, genes).all(axis=1)
        indices = np.linspace(start, end-1, min(end-start, args.patches_per_slice), dtype=int)
        scores, coverage, distances, modes = [], [], [], []
        for index in indices:
            item = dataset[int(index)]
            visible = (item['token_valid_mask'] & (item['compartment_ids'] == 1)).any(axis=-1)
            neighbors, valid, distance = patch_knn(
                torch.from_numpy(item['coords'])[None], torch.from_numpy(item['node_indices'])[None],
                torch.from_numpy(visible)[None], cfg['model']['neighbors'])
            maximum = cfg['model']['max_neighbor_distance']
            if maximum is not None:
                valid &= distance <= maximum
            neighbor, mask = neighbors[0].numpy(), valid[0].numpy()
            scores.append(np.take_along_axis(item['lr_bias'], neighbor, axis=1)[mask])
            coverage.append(np.take_along_axis(item['lr_coverage'], neighbor, axis=1)[mask])
            distances.append(distance[0].numpy()[mask])
            modes.append(int(item['task_mode']))
        reports.append(dict(slice_index=slice_index, slice_seed=entry['slice_seed'],
            training_centers=end-start, panel_genes=len(genes), measured_lr_pairs=int(measured.sum()),
            measured_fraction=float(measured.mean()), sampled_patches=len(indices), mask_modes=modes,
            directed_bias=distribution(scores), prior_coverage=distribution(coverage),
            eligible_edge_distance_coordinate_units=distribution(distances)))
        start = end
    if not any(row['measured_lr_pairs'] > 0 for row in reports):
        raise ValueError('No training panel measures any pair in the frozen LR prior')
    report = dict(status='passed', split='train', checkpoint_or_probe_used=False,
        selected_training_centers=len(dataset), center_stride=cfg['data']['center_stride'],
        lr_prior=dataset.lr_metadata, slices=reports,
        graph='up to 16 coordinate KNN within each 64-cell patch; two blocks permit multi-hop context',
        distance_unit='original prepared coordinates; no unverified micrometer conversion',
        initial_lr_multiplier=cfg['model']['lr_initial_strength'],
        note='Coverage is a diagnostic, not a second bias multiplier. No score-based tuning performed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps(dict(status=report['status'], slices=len(reports),
                          selected_training_centers=len(dataset), output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
