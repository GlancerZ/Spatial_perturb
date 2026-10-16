"""Prepare cell-interaction spatial bias with scGPT per-cell bins by default.

Bin k means exactly k molecules, so nothing is fitted and every slide shares one
mapping. Run on a compute node anyway: each slide contributes up to 50k positive
nuclear/cytoplasmic values from up to 2048 TRAIN cells, used only to report how
much of the positive mass saturates in the top bin. Cache arrays stay unchanged.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spatial_pretrain.config import load_config
from spatial_pretrain.data.dataset import make_center_split
from spatial_pretrain.data.compartment_binning import count_bin_edges, positive_bins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--output-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--coordinate-scale', type=float, default=1., help='Input-coordinate units to micrometres')
    parser.add_argument('--binning', choices=['scgpt_cell', 'count_identity'], default='scgpt_cell')
    args = parser.parse_args()
    if args.output_config.exists() or args.output_dir.exists():
        raise ValueError('Use a new output config and run directory; do not overwrite an existing run')
    cfg = load_config(args.base_config)
    if cfg.data.gene_sampling_strategy != 'compartment_balanced':
        raise ValueError('This preparation requires compartment caches')
    if args.binning == 'scgpt_cell':
        if not 0 < args.coordinate_scale < float('inf'):
            raise ValueError('coordinate-scale must be positive and finite')
        cfg.data.compartment_binning = 'scgpt_cell'
        cfg.data.compartment_bin_edges = []
        cfg.model.cell_attention_mode = 'cytoplasm_mean'
        cfg.model.cell_spatial_bias = True
        cfg.model.cell_angle_bins = 16
        cfg.model.cell_coordinate_scale = args.coordinate_scale
        cfg.experiment.name = args.output_dir.name
        cfg.experiment.output_dir = str(args.output_dir.resolve())
        cfg.training.save_initial_checkpoint = True
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        args.output_config.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
        args.output_config.with_suffix('.binning.json').write_text(json.dumps(dict(
            method='scgpt_cell', compartments='joint_nuclear_cytoplasmic',
            num_bins=cfg.data.num_bins, before_token_sampling=True), indent=2))
        print(f'Wrote {args.output_config}; spatial bias enabled; per-cell scGPT bins, no global fitting.')
        return
    cfg.data.compartment_binning = 'legacy'
    rng = np.random.default_rng(cfg.data.split_seed)
    edges = count_bin_edges(cfg.data.num_bins)
    pieces, provenance = [], []
    for path in cfg.data.cache_dirs or [cfg.data.cache_dir]:
        path = Path(path)
        meta = json.loads((path / 'metadata.json').read_text())
        for key in ('split_seed', 'train_fraction', 'split_strategy', 'split_blocks_x', 'split_blocks_y'):
            if meta[key] != getattr(cfg.data, key):
                raise ValueError(f'Cache/config training split mismatch: {path}: {key}')
        coords = np.load(path / 'coords.npy', mmap_mode='r')
        split = make_center_split(coords, cfg.data.train_fraction, cfg.data.split_seed,
            strategy=cfg.data.split_strategy, blocks_x=cfg.data.split_blocks_x, blocks_y=cfg.data.split_blocks_y)
        rows = np.sort(rng.choice(split.train_indices, min(2048, len(split.train_indices)), replace=False))
        values = []
        for name in ('nuclear_log1p_float16.npy', 'cytoplasmic_log1p_float16.npy'):
            array = np.load(path / name, mmap_mode='r')
            selected = np.asarray(array[rows], dtype=np.float32).ravel()
            values.append(selected[selected > 0])
        values = np.concatenate(values)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite training expression in {path}')
        sampled = rng.choice(values, min(50000, len(values)), replace=False)
        pieces.append(sampled)
        provenance.append(dict(cache=str(path.resolve()), train_cells=len(split.train_indices),
                               sampled_train_cells=len(rows), positive_values=len(sampled)))
    values = np.concatenate(pieces)
    if not len(values):
        raise ValueError('No positive training expression sampled')
    bins = positive_bins(values, edges)
    top = cfg.data.num_bins - 1
    coverage = {int(b): int(n) for b, n in zip(*np.unique(bins, return_counts=True))}
    saturated = coverage.get(top, 0) / len(bins)
    print(f'{len(coverage)}/{top} positive classes observed; {saturated:.4%} of values saturate bin {top}')
    cfg.data.compartment_bin_edges = edges.tolist()
    cfg.model.cell_attention_mode = 'cytoplasm_mean'
    cfg.model.cell_spatial_bias = True
    cfg.model.cell_angle_bins = 16
    cfg.model.cell_coordinate_scale = args.coordinate_scale
    if not 0 < args.coordinate_scale < float('inf'):
        raise ValueError('coordinate-scale must be positive and finite')
    cfg.experiment.name = args.output_dir.name
    cfg.experiment.output_dir = str(args.output_dir.resolve())
    cfg.training.save_initial_checkpoint = True
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    args.output_config.with_suffix('.binning.json').write_text(json.dumps(dict(
        method='identity_counts_bin_k_means_k_molecules_top_bin_saturates', seed=cfg.data.split_seed,
        bin_edges=edges.tolist(), observed_class_counts=coverage,
        top_bin_saturated_fraction=saturated, sources=provenance), indent=2))
    print(f'Wrote {args.output_config}; spatial bias enabled; count-identity bins saved in config.')


if __name__ == '__main__':
    main()
