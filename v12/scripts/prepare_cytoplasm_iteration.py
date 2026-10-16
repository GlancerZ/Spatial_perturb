"""Prepare bounded cross-token pilots; run on an allocated compute node."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml
from spatial_pretrain.config import load_config
from spatial_pretrain.data.compartment_cache import build_compartment_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='cytoplasm_mean_pilot_v1')
    parser.add_argument('--attention-mode', choices=['cytoplasm_mean', 'cytoplasm_cross'], default='cytoplasm_mean')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--cells-per-slice', type=int, default=8192)
    parser.add_argument('--prepare-cache', action='store_true')
    args = parser.parse_args()
    cfg = load_config(ROOT / 'configs/xenium_compartment_ready48.yaml')
    cfg.experiment.name = args.name
    cfg.experiment.output_dir = str(ROOT / 'outputs' / args.name)
    cfg.model.cell_attention_mode = args.attention_mode
    cfg.model.cell_heads = 16
    cfg.model.require_flash_attention = True
    cfg.data.subgraph_size = 16
    cfg.data.knn_k = cfg.data.nearest_k = cfg.data.candidate_k = 15
    cfg.data.max_cells = args.cells_per_slice
    cfg.data.cache_dirs = [str(ROOT / 'cache' / f'cytoplasm_pilot_n{args.cells_per_slice}_k15' / Path(p).stem)
                           for p in cfg.data.h5ad_paths]
    cfg.data.cache_dir = cfg.data.cache_dirs[0]
    cfg.data.batch_size = 2
    cfg.data.num_workers = 2
    cfg.data.view_prep_workers = 2
    cfg.optimizer.name = 'adamw'
    cfg.optimizer.lr = args.lr
    cfg.training.epochs = 1
    cfg.training.grad_accum_steps = 4
    cfg.training.max_train_steps = args.updates * cfg.training.grad_accum_steps
    cfg.training.warmup_ratio = 0.05
    cfg.training.min_lr_ratio = 0.1
    cfg.training.val_every = 100
    cfg.training.max_val_batches = 16
    cfg.training.log_every = 10
    cfg.training.checkpoint_every = 25
    cfg.training.save_initial_checkpoint = True
    cfg.training.auto_resume = True
    cfg.wandb.enabled = False
    destination = ROOT / 'configs' / f'{args.name}.yaml'
    content = yaml.safe_dump(asdict(cfg), sort_keys=False)
    if destination.exists() and destination.read_text() != content:
        raise RuntimeError(f'Refusing to change an existing experiment: {destination}')
    destination.write_text(content)
    smoke = deepcopy(cfg)
    smoke.experiment.name = args.name + '_smoke'
    smoke.experiment.output_dir += '_smoke'
    smoke.data.h5ad_paths = [cfg.data.h5ad_paths[0]]
    smoke.data.cache_dirs = [cfg.data.cache_dirs[0]]
    smoke.training.max_train_steps = 8
    smoke.training.val_every = 1
    smoke.training.max_val_batches = 2
    (ROOT / 'configs' / f'{args.name}_smoke.yaml').write_text(yaml.safe_dump(asdict(smoke), sort_keys=False))
    out = Path(cfg.experiment.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pilot_protocol.json').write_text(json.dumps(dict(
        samples=len(cfg.data.h5ad_paths), cells_per_slice_cap=cfg.data.max_cells,
        sampling='First N reference rows per slice; exploratory pilot, not full-cohort training',
        neighbors=15, global_subgraph_batch=16, planned_updates=args.updates,
        optimizer='AdamW', lr=args.lr, selection='Fixed train/validation donor subsets; test donors excluded',
        attention_mode=args.attention_mode,
        direct_cross_cell_routing='Only valid cytoplasm; mean mode pools with valid-token denominator, attends over other cells and broadcasts updates only to cytoplasm',
        nuclear_information='Within-cell attention still mixes compartments; no claim of total nuclear isolation'
    ), indent=2))
    print(destination, flush=True)
    if args.prepare_cache:
        for index, (source, cache) in enumerate(zip(cfg.data.h5ad_paths, cfg.data.cache_dirs), 1):
            if not Path(source).exists():
                raise FileNotFoundError(source)
            print(f'[prepare] {index}/{len(cfg.data.h5ad_paths)} {Path(source).stem}', flush=True)
            build_compartment_cache(source, cache, cfg)


if __name__ == '__main__':
    main()
