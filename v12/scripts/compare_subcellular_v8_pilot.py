#!/usr/bin/env python3
"""Tabulate V8 pilot arms against V6-A and V7-A at equal optimizer steps.

Reads each run's downstream_validation/step_*/metrics.json (fixed-C
validation macro-F1 on the fixed 5,200-cell cohort) and, when the saved
embeddings exist, the per-cell residual/base norm ratio. Writes a CSV and
prints a compact table. Run on Vulcan where the outputs live.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = Path('/scratch/glancerz/spatial-pretrain-v3-subcellular-scgpt-20260916/outputs')
RUNS = {
    'v6A (random, frozen)': 'subcellular_lr_v6_a_20260920',
    'v7A (left)': 'subcellular_lr_v7_a_20260921',
    'r0 (random)': 'subcellular_v8_pilot_r0_20260923',
    'l5 (left+cap)': 'subcellular_v8_pilot_l5_20260923',
    'r5 (random+cap)': 'subcellular_v8_pilot_r5_20260923',
    'r5w4 (random+cap+w4)': 'subcellular_v8_pilot_r5w4_20260923',
}
READOUTS = ('base', 'joint', 'intrinsic', 'cytoplasm', 'joint_neighbor_shuffle')


def fixed_scores(report):
    scores = {}
    for row in report.get('results', []):
        value = row.get('fixed_C_result', {}).get('val_macro_f1')
        if value is not None:
            scores[(row['embedding'], row['task'])] = float(value)
    return scores


def residual_ratio(directory):
    base, joint = directory / 'base_embeddings.npy', directory / 'joint_embeddings.npy'
    if not (base.exists() and joint.exists()):
        return None
    b, j = np.load(base).astype(np.float64), np.load(joint).astype(np.float64)
    return float(np.mean(np.linalg.norm(j - b, axis=1) / np.linalg.norm(b, axis=1)))


def masked_ce(report):
    loss = report.get('masked_validation', {}).get('loss_mean', {})
    return loss.get('masked_ce')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', default='0,500,1000,1500,2000,2500,3000')
    parser.add_argument('--csv', type=Path, default=ROOT / 'outputs/subcellular_v8_pilot_comparison.csv')
    args = parser.parse_args()
    steps = [int(value) for value in args.steps.split(',')]
    rows = []
    for label, name in RUNS.items():
        for step in steps:
            directory = OUTPUTS / name / 'downstream_validation' / f'step_{step:08d}'
            path = directory / 'metrics.json'
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            scores = fixed_scores(report)
            row = dict(run=label, step=step, masked_ce=masked_ce(report),
                       residual_over_base=residual_ratio(directory))
            for readout in READOUTS:
                for task, short in (('cell_type', 'cell'), ('region', 'region')):
                    row[f'{readout}_{short}'] = scores.get((readout, task))
            rows.append(row)
    if not rows:
        raise SystemExit('No metrics found')
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(value, digits=4):
        return '   -  ' if value is None else f'{value:.{digits}f}'

    print(f'{"run":24s} {"step":>5s} | {"base c/r":>15s} | {"joint c/r":>15s} | '
          f'{"shuffle c/r":>15s} | {"cyto region":>11s} | {"|r|/|b|":>7s} | {"maskCE":>6s}')
    for row in rows:
        print(f'{row["run"]:24s} {row["step"]:5d} | {fmt(row["base_cell"])}/{fmt(row["base_region"])} | '
              f'{fmt(row["joint_cell"])}/{fmt(row["joint_region"])} | '
              f'{fmt(row["joint_neighbor_shuffle_cell"])}/{fmt(row["joint_neighbor_shuffle_region"])} | '
              f'{fmt(row["cytoplasm_region"]):>11s} | {fmt(row["residual_over_base"], 3):>7s} | '
              f'{fmt(row["masked_ce"], 3):>6s}')
    print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
