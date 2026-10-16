"""Verify and plot every V11 readout on cell identity and environment tasks.

A summary is descriptive validation monitoring, never an independent test or a
claim that training completed. Explicit --expected-steps is required to certify
that a requested evaluation set is complete.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_subcellular_v11 import READOUTS, FIXED_C, CANONICAL_COHORT, validate_completion

COLORS = dict(nucleus='#0072B2', spatial='#009E73', fused='#D55E00', concat='#CC79A7',
              concat_pca512='#E69F00', scgpt_whole='#222222', scgpt_nucleus='#888888')


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def draw_curves(rows, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200})
    variants = sorted({row['variant'] for row in rows})
    tasks = [('cell_type', 'val_macro_f1', 'Cell type macro-F1 (higher better)'),
             ('region', 'val_macro_f1', 'Region macro-F1 (higher better)'),
             ('neighbor_composition', 'val_mae', 'Neighbor composition MAE (lower better)'),
             ('neighbor_composition', 'val_js_distance', 'Neighbor composition JS distance (lower better)')]
    for task, metric, title in tasks:
        fig, axes = plt.subplots(len(variants), 1, figsize=(10, 4.3*len(variants)), squeeze=False, constrained_layout=True)
        for ax, variant in zip(axes[:, 0], variants):
            for name in READOUTS:
                selected = sorted((row for row in rows if row['variant'] == variant and row['task'] == task and row['embedding'] == name),
                                  key=lambda row: row['step'])
                if not selected:
                    continue
                style = '--' if name.startswith('scgpt_') else ':' if name == 'concat' else '-'
                label = f'{name} ({selected[0]["dimension"]}D)'
                ax.plot([row['step'] for row in selected], [row[metric] for row in selected], color=COLORS[name],
                        linestyle=style, marker='o' if not name.startswith('scgpt_') else None, markersize=3, label=label)
            ax.set_title(f'{variant}: {title}'); ax.set_xlabel('Optimizer step'); ax.set_ylabel(title)
            ax.grid(alpha=.2); ax.legend(ncol=2, fontsize=8)
            if task != 'neighbor_composition':
                ax.set_ylim(0, 1)
        filename = f'{task}_{metric}_curves'
        fig.savefig(output / f'{filename}.png'); fig.savefig(output / f'{filename}.pdf'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', '--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output-dir', '--output', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, nargs='*', default=[])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates, incomplete = set(), []
    for entry in args.inputs:
        if entry.is_file():
            candidates.add(entry.resolve())
        elif entry.is_dir():
            candidates.update(path.resolve() for path in entry.rglob('metrics.json'))
            for cache in entry.rglob('cache_contract.json'):
                if not (cache.parent / 'metrics.json').exists():
                    incomplete.append(dict(path=str(cache.parent), reason='No final all-readout/all-task metrics.json'))
        else:
            incomplete.append(dict(path=str(entry), reason='Input missing'))
    reports, rejected, seen = [], [], {}
    reference = None
    for path in sorted(candidates):
        try:
            report = json.loads(path.read_text())
            validate_completion(report, path.parent)
            comparison = dict(cohort=report['cohort_sha256'], fixed_C=report['fixed_C'],
                radius=report['composition_protocol']['radius_um'], classes=report['composition_protocol']['classes'],
                evaluator_sha256=report['evaluator_sha256'], solver_sha256=report['solver_sha256'],
                input_signatures_sha256=report['input_signatures_sha256'],
                probe_row_normalization=report['probe_row_normalization'],
                protocol_sha256=report['protocol_sha256'], original_backbone_sha256=report['backbone_provenance']['weights_sha256'])
            if reference is not None and comparison != reference:
                raise ValueError('Cannot combine different cohort/baseline/probe/composition target protocols')
            reference = comparison
            key = (report['variant'], int(report['step']))
            if key in seen:
                raise ValueError(f'Duplicate experiment/step also at {seen[key]}')
            seen[key] = str(path)
            reports.append((path, report))
        except (OSError, ValueError, KeyError, TypeError) as error:
            rejected.append(dict(path=str(path), reason=str(error)))
    rows, diagnostics = [], []
    for path, report in reports:
        common = dict(variant=report['variant'], step=report['step'], checkpoint_sha256=report['checkpoint_sha256'],
            config_sha256=report['config_sha256'], cohort_sha256=report['cohort_sha256'], metrics_path=str(path))
        for result in report['results']:
            row = dict(**common, embedding=result['embedding'], task=result['task'], dimension=result['embedding_dimension'],
                       train_cells=result['train_cells'], val_cells=result['val_cells'])
            if result['task'] in FIXED_C:
                score = result['fixed_C_result']
                row.update(C=score['C'], val_macro_f1=score['val_macro_f1'], val_balanced_accuracy=score['val_balanced_accuracy'])
                row.update({f'donor_macro_f1_{donor}': value for donor, value in score['per_donor_macro_f1'].items()})
            else:
                row.update({f'val_{key}': value for key, value in result['val_result'].items()})
                row.update(ridge_alpha=result['ridge_alpha'], mean_val_label_coverage=result['mean_val_label_coverage'])
            rows.append(row)
        diagnostics.extend(dict(**common, embedding=name, **data) for name, data in report['diagnostics']['collapse'].items())
    rows.sort(key=lambda row: (row['variant'], row['task'], row['embedding'], row['step']))
    variants = sorted({variant for variant, _ in seen})
    missing = [dict(variant=variant, step=step) for variant in variants for step in args.expected_steps if (variant, step) not in seen]
    warnings = []
    for baseline in ('scgpt_whole', 'scgpt_nucleus'):
        for task in FIXED_C:
            values = [row['val_macro_f1'] for row in rows if row['embedding'] == baseline and row['task'] == task]
            if values and max(values)-min(values) > 1e-6:
                warnings.append(f'{baseline} {task} baseline varies across checkpoints by {max(values)-min(values):.6g}; verify before attribution')
    write_csv(args.output_dir / 'metrics.csv', rows); write_csv(args.output_dir / 'diagnostics.csv', diagnostics)
    if rows:
        draw_curves(rows, args.output_dir)
    complete = bool(args.expected_steps) and bool(reports) and not missing and not incomplete and not rejected
    atomic_json(args.output_dir / 'summary.json', dict(valid_evaluations=len(reports), expected_evaluations_complete=complete,
        completion_scope='Only explicitly requested evaluation steps; not a training completion certificate',
        expected_steps=args.expected_steps, missing=missing, incomplete=incomplete, rejected=rejected, warnings=warnings,
        canonical_cohort=CANONICAL_COHORT, fixed_C=FIXED_C, no_test=True,
        completed=[dict(variant=report['variant'], step=report['step'], path=str(path)) for path, report in reports]))
    lines = ['# V11 互补融合：全部表征的多任务验证', '',
        f'完整评测 {len(reports)} 个；未完成 {len(incomplete)} 项；拒绝 {len(rejected)} 项。', '',
        '固定 5,200 个中心细胞，10 个训练供体、3 个验证供体，每供体 400；不读取 test。'
        '区域任务只使用其中有区域注释的细胞。分类器沿用 train-only StandardScaler 和固定 C。', '',
        'nucleus、spatial、fused 均为 512 维；concat 为 1,024 维；concat_pca512 只在训练细胞拟合标准化与 PCA。'
        '两条 scGPT 基线独立加载原始普通 scGPT，分别编码核和整细胞；它们不参与训练或作为老师。', '',
        '全部 7 种表征统一在 probe 前做逐细胞 L2 归一化，再拟合训练集 StandardScaler。'
        '保存的 embedding 和塌缩诊断使用原生输出（concat 按模型定义拼接两路单位向量，PCA 保存归一化前输出）；'
        '基线缓存保存 raw CLS，基线分数按本轮协议重新计算，不沿用历史数字。'
        'step 0 对非空核中心核对 V11 核 CLS 与原始核基线，记录 BF16 最大差异；'
        '空核时 V11 输出零，原始基线仍有空输入 CLS，这部分不用于等值断言。', '',
        '邻域组成来自 100 µm 内排除中心的现有 final_CT（实际半径见 metrics.json）。'
        '类别词表只由训练供体定义；缺失和未见类别不伪造为零，报告覆盖率。'
        '组成为已知标签条件下的比例；Ridge 预测投影到概率单纯形，MAE、JS 距离越低越好。'
        '该任务衡量观测环境信息是否可读，不能宣称预测了隐藏邻居或证明因果通讯。', '',
        '| 实验 | step | 表征 | 维数 | 细胞类型 F1 ↑ | 区域 F1 ↑ | 组成 MAE ↓ | 组成 JS 距离 ↓ |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    lookup = {(row['variant'], row['step'], row['embedding'], row['task']): row for row in rows}
    for variant, step in sorted(seen):
        for name in READOUTS:
            cell = lookup[(variant, step, name, 'cell_type')]
            region = lookup[(variant, step, name, 'region')]
            composition = lookup[(variant, step, name, 'neighbor_composition')]
            lines.append(f'| {variant} | {step} | {name} | {cell["dimension"]} | {cell["val_macro_f1"]:.6f} | '
                         f'{region["val_macro_f1"]:.6f} | {composition["val_mae"]:.6f} | {composition["val_js_distance"]:.6f} |')
    lines += ['', '所有 readout 均保留，不能只选一个最高分表征展示。'
              '先比较同一步的核、空间、融合与 PCA 等维对照，再看从 step 0 起的变化、逐供体结果和有效秩。'
              '验证曲线用于监控与选型，反复查看后的最高点不是独立测试成绩。', '']
    for task, metric in (('cell_type', 'val_macro_f1'), ('region', 'val_macro_f1'),
                         ('neighbor_composition', 'val_mae'), ('neighbor_composition', 'val_js_distance')):
        lines += [f'![{task} {metric}]({task}_{metric}_curves.png)', '']
    lines.extend(f'- {warning}' for warning in warnings)
    lines.extend(f'- 缺少 {item["variant"]} step {item["step"]}' for item in missing)
    lines.extend(f'- {item["path"]}: {item["reason"]}' for item in incomplete + rejected)
    temporary = args.output_dir / 'REPORT.zh.md.partial'; temporary.write_text('\n'.join(lines)+'\n'); temporary.replace(args.output_dir / 'REPORT.zh.md')
    print(json.dumps(dict(valid_evaluations=len(reports), complete=complete, rejected=len(rejected), output=str(args.output_dir)), indent=2))
    return 2 if rejected else 0


if __name__ == '__main__':
    raise SystemExit(main())
