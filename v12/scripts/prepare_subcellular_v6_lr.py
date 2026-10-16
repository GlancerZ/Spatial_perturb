"""Prepare the shared curated single-gene LR asset and immutable A/B configs.

Run once inside an authorized compute step. Existing files must match exactly;
this command never replaces an existing configuration or prior with new content.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REL = 'resources/spatialformer/upstream/downstream/cell_cell_communication/data/ligand_receptor_database.csv'
SOURCE_SHA = '791002fdbc810676306c3e89b883895da01620651c015931423d627d525d1b43'
SOURCE_REVISION = 'f02e7776e8a99c9abdc5fe5572a1a20369faf159'
SOURCE_URL = f'https://github.com/TerminatorJ/Spatialformer/blob/{SOURCE_REVISION}/downstream/cell_cell_communication/data/ligand_receptor_database.csv'
SYMBOL = re.compile(r'[A-Z][A-Z0-9.-]*\Z')


def immutable_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f'Refusing to replace different existing content: {path}')
        return
    with path.open('xb') as handle:
        handle.write(payload)


def normalize_prior(payload):
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError('Original curated LR database SHA256 differs from the reviewed source')
    rows = list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))
    pairs, excluded, annotations = set(), {}, {}
    for row in rows:
        ligand, receptor = [row[f'interaction.{name}'].strip() for name in ('ligand', 'receptor')]
        display = row['interaction.interaction_name_2']
        annotation = row['interaction.annotation']
        # Never turn a required receptor/ligand complex into independent subunits.
        reason = None
        if any(char in display for char in '()+_') or not all(SYMBOL.fullmatch(g) for g in (ligand, receptor)):
            reason = 'complex_or_noncanonical_single_gene'
        elif annotation not in ('Secreted Signaling', 'Cell-Cell Contact', 'ECM-Receptor'):
            reason = 'unsupported_annotation'
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        pairs.add((ligand, receptor))
        annotations[annotation] = annotations.get(annotation, 0) + 1
    if not pairs:
        raise ValueError('No curated single-gene LR pairs survived filtering')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['ligand', 'receptor', 'weight'])
    writer.writerows((ligand, receptor, 1) for ligand, receptor in sorted(pairs))
    normalized = stream.getvalue().encode()
    manifest = dict(source_url=SOURCE_URL, source_revision=SOURCE_REVISION,
        source_sha256=SOURCE_SHA, sha256=hashlib.sha256(normalized).hexdigest(),
        source_rows=len(rows), unique_directed_pairs=len(pairs), excluded_rows=excluded,
        retained_annotation_rows=annotations, semantics='curated_single_gene_subset',
        weight='uniform_one_per_unique_directed_pair; evidence duplicates do not increase weight',
        excluded_complexes=True, prediction_only_interactions_added=False,
        missing_panel_genes='unknown, not measured zero',
        lr_bias='weighted visible cytoplasmic activity sum / (known evidence weight + 4)',
        lr_coverage='known evidence weight / full prior weight; diagnostic only')
    return normalized, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / SOURCE_REL)
    parser.add_argument('--asset-dir', type=Path, default=ROOT / 'resources/lr_v6')
    parser.add_argument('--template', type=Path, default=ROOT / 'configs/subcellular_lr_v6_20260920.template.yaml')
    parser.add_argument('--config-dir', type=Path, default=ROOT / 'configs')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Prepare data only inside a verified Slurm compute step')
    normalized, manifest = normalize_prior(args.source.read_bytes())
    asset = args.asset_dir.resolve() / 'curated_single_gene_pairs.csv'
    immutable_write(asset, normalized)
    immutable_write(asset.with_suffix('.manifest.json'),
                    (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    template = yaml.safe_load(args.template.read_text())
    for route in ('A', 'B'):
        cfg = json.loads(json.dumps(template))
        name = f'subcellular_lr_v6_{route.lower()}_20260920'
        cfg['experiment'] = name
        cfg['output_dir'] = str(ROOT / 'outputs' / name)
        cfg['model']['route'] = route
        cfg['lr'] = dict(enabled=True, path=str(asset), sha256=manifest['sha256'], source=SOURCE_URL)
        path = args.config_dir / f'{name}.yaml'
        immutable_write(path, yaml.safe_dump(cfg, sort_keys=False).encode())
        print(json.dumps(dict(route=route, config=str(path.resolve()), output=cfg['output_dir'],
                              lr_sha256=manifest['sha256'])), flush=True)


if __name__ == '__main__':
    main()
