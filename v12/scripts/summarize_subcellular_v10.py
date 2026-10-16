"""Summarize verified V10 validation evaluations; never count partial runs as done.

Produces flat metrics.csv, diagnostics.csv, scientific PNG/PDF curves, and a
Chinese Markdown report. Comparison requires the identical canonical cohort,
fixed probe C, no test donors, and equal-dimensional primary readouts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CANONICAL_COHORT = '0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc'
FIXED_C = {'cell_type': .1, 'region': .01}
VARIANTS = ('cosine', 'contrastive', 'relational')
READOUTS = ('base', 'nucleus', 'context', 'joint', 'base_augmented',
            'context_neighbor_shuffle', 'joint_neighbor_shuffle', 'base_augmented_neighbor_shuffle')
COLORS = dict(cosine='#0072B2', contrastive='#D55E00', relational='#009E73')


def atomic_json(path, payload):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate(report, path):
    if report.get('status') != 'complete' or report.get('model_version') != 10:
        raise ValueError('No valid V10 completion marker')
    if report.get('architecture') != 'nucleus_neighbor_teacher_v10':
        raise ValueError('Wrong architecture')
    if not report.get('canonical_cohort') or report.get('cohort_sha256') != CANONICAL_COHORT or report.get('cells') != 5200:
        raise ValueError('Wrong evaluation cohort')
    if not report.get('no_test') or report.get('test_donors_used') is not False:
        raise ValueError('Missing no-test provenance')
    if report.get('fixed_C') != FIXED_C or not report.get('sources_verified'):
        raise ValueError('Fixed-C/source provenance mismatch')
    if len(report.get('train_donors', [])) != 10 or len(report.get('val_donors', [])) != 3:
        raise ValueError('Wrong train/validation donor split')
    if report.get('variant') not in VARIANTS or int(report['step']) < 0:
        raise ValueError('Invalid variant/step')
    dimensions = report['dimensions']
    if any(dimensions[key] != dimensions['base'] for key in ('nucleus', 'context', 'joint')):
        raise ValueError('Primary embedding dimensions differ')
    expected = {(name, task) for name in READOUTS for task in FIXED_C}
    observed = {(row['embedding'], row['task']) for row in report.get('results', [])}
    if expected != observed or len(report['results']) != len(expected):
        raise ValueError('Missing or duplicate probe results')
    for row in report['results']:
        score = row['fixed_C_result']
        if score['C'] != FIXED_C[row['task']] or not score['solver'].get('converged_gradient'):
            raise ValueError('Probe C/solver convergence invalid')
        if not all(np.isfinite(score[key]) and 0 <= score[key] <= 1 for key in ('val_macro_f1', 'val_balanced_accuracy')):
            raise ValueError('Invalid probe score')
        predictions = path.parent / f'{row["embedding"]}_{row["task"]}_val_predictions.parquet'
        if not predictions.is_file():
            raise ValueError(f'Missing predictions: {predictions.name}')


def make_curves(rows, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 220})
    for task in FIXED_C:
        subset = [row for row in rows if row['task'] == task]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True, constrained_layout=True)
        baseline = [row['val_macro_f1'] for row in subset if row['embedding'] == 'base']
        for ax in axes:
            if baseline:
                mean = float(np.mean(baseline))
                ax.axhline(mean, color='#222222', linestyle='--', linewidth=1.5,
                           label=f'Frozen ordinary scGPT ({mean:.4f})')
                if max(baseline) - min(baseline) > 1e-6:
                    ax.axhspan(min(baseline), max(baseline), color='#222222', alpha=.1)
            ax.set_xlabel('Training optimizer step')
            ax.grid(alpha=.2)
            ax.set_ylim(0, 1)
        for variant in VARIANTS:
            for readout, style, marker in (('joint', '-', 'o'), ('nucleus', ':', '^'), ('context', '--', 's')):
                selected = sorted((row for row in subset if row['variant'] == variant and row['embedding'] == readout),
                                  key=lambda row: row['step'])
                if selected:
                    axes[0].plot([row['step'] for row in selected], [row['val_macro_f1'] for row in selected],
                                 color=COLORS[variant], linestyle=style, marker=marker, markersize=4,
                                 label=f'{variant}: {readout}')
            for readout, style, marker in (('base_augmented', '-', 'o'), ('joint_neighbor_shuffle', '--', 'x')):
                selected = sorted((row for row in subset if row['variant'] == variant and row['embedding'] == readout),
                                  key=lambda row: row['step'])
                if selected:
                    axes[1].plot([row['step'] for row in selected], [row['val_macro_f1'] for row in selected],
                                 color=COLORS[variant], linestyle=style, marker=marker, markersize=4,
                                 label=f'{variant}: {readout}')
        axes[0].set_ylabel('Validation macro-F1 (fixed C)')
        axes[0].set_title(f'{task}: equal-dimension readouts')
        axes[1].set_title('Augmented dimension and shuffled-neighbor controls')
        for ax in axes:
            ax.legend(fontsize=7, loc='best')
        fig.savefig(output / f'{task}_curves.png')
        fig.savefig(output / f'{task}_curves.pdf')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', '--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', '--output', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, nargs='*', default=[])
    parser.add_argument('--expected-variants', choices=VARIANTS, nargs='+', default=list(VARIANTS))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, incomplete = set(), []
    for entry in args.inputs:
        if entry.is_file():
            candidates.add(entry.resolve())
        elif entry.is_dir():
            candidates.update(path.resolve() for path in entry.rglob('metrics.json'))
            for contract in entry.rglob('cache_contract.json'):
                if not (contract.parent / 'metrics.json').exists():
                    progress = contract.parent / 'progress.json'
                    incomplete.append(dict(path=str(contract.parent), reason='No final metrics.json',
                        progress=json.loads(progress.read_text()) if progress.exists() else None))
        else:
            incomplete.append(dict(path=str(entry), reason='Input does not exist'))
    reports, rejected, seen = [], [], {}
    for path in sorted(candidates):
        try:
            report = json.loads(path.read_text())
            validate(report, path)
            key = (report['variant'], int(report['step']))
            if key in seen:
                raise ValueError(f'Duplicate variant/step also exists at {seen[key]}')
            seen[key] = str(path)
            reports.append((path, report))
        except (ValueError, KeyError, TypeError, OSError) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    rows, diagnostics = [], []
    for path, report in reports:
        common = dict(variant=report['variant'], step=int(report['step']),
                      checkpoint_sha256=report['checkpoint_sha256'], config_sha256=report['config_sha256'],
                      cohort_sha256=report['cohort_sha256'], metrics_path=str(path))
        for result in report['results']:
            score = result['fixed_C_result']
            row = dict(**common, embedding=result['embedding'], task=result['task'], C=score['C'],
                       dimension=result['embedding_dimension'], train_cells=result['train_cells'], val_cells=result['val_cells'],
                       val_macro_f1=score['val_macro_f1'], val_balanced_accuracy=score['val_balanced_accuracy'],
                       evaluation_seconds=report['seconds'])
            row.update({f'donor_macro_f1_{donor}': value for donor, value in score['per_donor_macro_f1'].items()})
            rows.append(row)
        for name, data in report['diagnostics']['collapse'].items():
            row = dict(**common, embedding=name, **data)
            retrieval = report['diagnostics']['context_retrieval'].get(name, {})
            row.update({f'retrieval_{key}': value for key, value in retrieval.items() if isinstance(value, (int, float))})
            diagnostics.append(row)
    rows.sort(key=lambda row: (row['task'], row['variant'], row['embedding'], row['step']))
    missing = [dict(variant=variant, step=step) for variant in args.expected_variants
               for step in args.expected_steps if (variant, step) not in seen]
    baseline_ranges = {task: [row['val_macro_f1'] for row in rows if row['task'] == task and row['embedding'] == 'base']
                       for task in FIXED_C}
    warnings = []
    for task, values in baseline_ranges.items():
        if values and max(values) - min(values) > .001:
            warnings.append(f'{task}: frozen baseline varies by {max(values) - min(values):.6f}; investigate before comparing models')
    write_csv(args.output_dir / 'metrics.csv', rows)
    write_csv(args.output_dir / 'diagnostics.csv', diagnostics)
    if rows:
        make_curves(rows, args.output_dir)
    completion = bool(args.expected_steps) and not missing and not rejected and not incomplete
    atomic_json(args.output_dir / 'summary.json', dict(valid_evaluations=len(reports),
        expected_evaluations_complete=completion,
        completion_scope='Only explicitly requested variant/step evaluations; does not certify training completion',
        canonical_cohort=CANONICAL_COHORT, fixed_C=FIXED_C, no_test=True,
        expected_steps=args.expected_steps, expected_variants=args.expected_variants,
        missing=missing, incomplete=incomplete, rejected=rejected, warnings=warnings,
        completed=[dict(variant=report['variant'], step=report['step'], path=str(path)) for path, report in reports]))
    lines = ['# V10 核／邻域预测整细胞：固定供体验证', '',
             f'已核验 {len(reports)} 个完整评测；未完成 {len(incomplete)} 项；无效结果 {len(rejected)} 项。', '',
             '所有分数来自相同 5,200 个细胞（10 个训练供体、3 个验证供体、每供体 400 个），'
             '未读取测试供体。分类器仅用训练供体拟合；cell type C=0.1，region C=0.01。', '',
             '普通 scGPT 对照是本批次冻结整细胞 teacher 的 L2 CLS，采用相同确定性分箱和相同中心细胞；'
             '它不是历史随机分箱或不同 cohort 上的分数。base、nucleus、context、joint 等维；'
             'base_augmented 单独标注维数，提升不能全部归因于邻域信息。', '',
             '| 方案 | step | 表征 | 维数 | cell-type macro-F1 | region macro-F1 |',
             '|---|---:|---|---:|---:|---:|']
    lookup = {(row['variant'], row['step'], row['embedding'], row['task']): row for row in rows}
    for variant, step in sorted(seen):
        for readout in READOUTS:
            cell = lookup.get((variant, step, readout, 'cell_type'))
            region = lookup.get((variant, step, readout, 'region'))
            if cell and region:
                lines.append(f'| {variant} | {step} | {readout} | {cell["dimension"]} | '
                             f'{cell["val_macro_f1"]:.6f} | {region["val_macro_f1"]:.6f} |')
    lines += ['', '![Cell type curves](cell_type_curves.png)', '', '![Region curves](region_curves.png)', '',
              '## 如何判断这个想法有效', '',
              '先看等维 joint 是否优于冻结 base、nucleus；再看真实邻域是否优于表达打乱邻域。'
              '同时检查各供体分数、context→whole-cell 的同供体检索排名和有效秩。'
              '核与邻域 cosine 上升本身不能证明表示变好，常量输出也可以一致。', '',
              '邻域打乱保持中心、邻域坐标与表达可见性分层不变，排除中心作为邻居；'
              '该诊断检验局部表达的贡献，不能单独证明因果细胞互作。'
              '曲线是验证诊断，反复查看后的最佳检查点不能当作独立测试结论。', '',
              'metrics.csv 保存逐检查点／表征／任务／供体分数；diagnostics.csv 保存塌缩与检索指标；'
              'summary.json 记录未完成和拒绝的评测。未提供 --expected-steps 时不宣布整轮实验完成。']
    if warnings or missing or incomplete or rejected:
        lines += ['', '## 尚未完成或需要核实', '']
        lines.extend(f'- {item}' for item in warnings)
        lines.extend(f'- 缺少 {item["variant"]} step {item["step"]}' for item in missing)
        lines.extend(f'- {item["path"]}: {item["reason"]}' for item in incomplete + rejected)
    temporary = args.output_dir / 'REPORT.zh.md.partial'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(args.output_dir / 'REPORT.zh.md')
    print(json.dumps(dict(valid_evaluations=len(reports), missing=len(missing), rejected=len(rejected),
                          incomplete=len(incomplete), output=str(args.output_dir)), indent=2))
    if rejected:
        sys.exit(2)


if __name__ == '__main__':
    main()
