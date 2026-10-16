#!/usr/bin/env python3
"""Generate local V12 paths without changing the historical scientific config."""
from pathlib import Path
import argparse
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h5ad-dir', type=Path, required=True)
    p.add_argument('--geometry-root', type=Path, required=True,
                   help='One <sample>_k255 directory containing coords.npy and knn_indices.npy per H5AD')
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--assets-dir', type=Path, default=root / 'resources/scgpt_official')
    p.add_argument('--template', type=Path, default=root / 'configs/historical/subcellular_v12_compartment_20260930.yaml')
    a = p.parse_args()
    sources = sorted(a.h5ad_dir.resolve().glob('*.h5ad'))
    if not sources:
        p.error('No *.h5ad files in --h5ad-dir')
    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    base_path, config_path = work / 'sources.yaml', work / 'v12.yaml'
    if base_path.exists() or config_path.exists():
        p.error('Refusing to overwrite sources.yaml or v12.yaml; use a new work directory')
    base = {'data': {'max_cells': 0, 'h5ad_paths': [str(f) for f in sources],
                     'cache_dirs': [str(a.geometry_root.resolve() / (f.stem + '_k255')) for f in sources]}}
    cfg = yaml.safe_load(a.template.read_text())
    cfg['base_config'] = str(base_path)
    cfg['assets_dir'] = str(a.assets_dir.resolve())
    cfg['cache_root'] = str(work / 'cache')
    cfg['output_dir'] = str(work / 'training')
    cfg['lr']['path'] = str(root / 'resources/lr_v6/curated_single_gene_pairs.csv')
    base_path.write_text(yaml.safe_dump(base, sort_keys=False))
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(config_path)
    print(f'{len(sources)} input slices; original 48-slice result requires exactly the documented 48 inputs.')


if __name__ == '__main__':
    main()
