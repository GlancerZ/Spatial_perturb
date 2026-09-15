"""Fit frozen shared expression bins from cached TRAIN cells; emit a new run config.

Run on a compute node. Cache arrays stay unchanged. Each slide contributes at
most 50k positive nuclear/cytoplasmic values from up to 2048 training cells.
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
from spatial_pretrain.data.compartment_binning import validate_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, required=True)
    parser.add_argument('--output-config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--coordinate-scale', type=float, default=1., help='Input-coordinate units to micrometres')
    args = parser.parse_args()
    if args.output_config.exists() or args.output_dir.exists():
        raise ValueError('Use a new output config and run directory; do not overwrite an existing run')
    cfg = load_config(args.base_config)
    if cfg.data.gene_sampling_strategy != 'compartment_balanced':
        raise ValueError('This preparation requires compartment caches')
    rng = np.random.default_rng(cfg.data.split_seed)
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
        raise ValueError('No positive training expression to fit')
    edges = validate_edges(np.quantile(values, np.linspace(0, 1, cfg.data.num_bins)[1:-1]), cfg.data.num_bins)
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
        method='quantiles_of_sampled_positive_train_values_float16', seed=cfg.data.split_seed,
        bin_edges=edges.tolist(), sources=provenance), indent=2))
    print(f'Wrote {args.output_config}; spatial bias enabled; shared boundaries saved in config.')


if __name__ == '__main__':
    main()
