"""Frozen checkpoint evaluation on the complete historical LungPF test protocol.

Separate entry point: never changes training or the every-100-step watcher.
Uses exactly the existing per-center extractor and historical probe solver.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from evaluate_subcellular_v2 import (
    ROOT, EMBEDDINGS, atomic_array, atomic_json, extract, file_sha256, load_model,
)
import probe_lungpf_embeddings as probe

COLUMNS = ['full_cell_id', 'sample', 'patient', 'split', 'final_CT', 'region_label']
EXPECTED_CELLS = 455379


def rows_digest(obs):
    return hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest()


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.monotonic()
    prepared = ROOT / 'data/benchmarking_data/lungpf/prepared'
    protocol = json.loads((prepared / 'protocol.json').read_text())
    frames = {}
    for sample in sorted(protocol['samples']):
        obs = pd.read_parquet(prepared / sample / 'obs.parquet')
        historical = pd.read_parquet(args.historical / 'embeddings' / sample / 'cells.parquet')
        pd.testing.assert_frame_equal(obs[COLUMNS].reset_index(drop=True),
            historical[COLUMNS].reset_index(drop=True), check_dtype=False)
        if obs.patient.nunique() != 1:
            raise ValueError(f'Mixed donors: {sample}')
        if not obs['split'].eq(protocol['donor_split'][obs.patient.iloc[0]]).all():
            raise ValueError(f'Donor split mismatch: {sample}')
        frames[sample] = obs
    cohort = pd.concat(frames.values(), ignore_index=True)
    if len(cohort) != EXPECTED_CELLS or not cohort.full_cell_id.is_unique:
        raise ValueError('Full cohort cell count or uniqueness mismatch')
    historical_rows = [row for row in json.loads((args.historical / 'metrics.json').read_text())
                       if row['model'] == 'spatialformer']
    official_rows = json.loads(args.official_metrics.read_text())
    if {r['task'] for r in historical_rows} != {'cell_type', 'region'}:
        raise ValueError('Missing historical SpatialFormer tasks')
    if {r['task'] for r in official_rows if r['model'] == 'scgpt_spatial'} != {'cell_type', 'region'}:
        raise ValueError('Missing official scGPT-spatial tasks')
    model, cfg, vocab, step, provenance = load_model(args.checkpoint, args.expected_step)
    if provenance['backbone_kind'] != 'scgpt':
        raise ValueError('This evaluation is explicitly for the ordinary-scGPT main model')
    signature = dict(step=step, checkpoint_sha256=file_sha256(args.checkpoint),
        config_sha256=provenance['config_sha256'], cohort_sha256=rows_digest(cohort),
        protocol_sha256=file_sha256(prepared / 'protocol.json'), batch_size=args.batch_size,
        chunk_cells=args.chunk_cells, seed=42, smoke=args.smoke,
        source_sha256={name: file_sha256(ROOT / name) for name in (
            'scripts/evaluate_subcellular_full.py', 'scripts/evaluate_subcellular_v2.py',
            'scripts/probe_lungpf_embeddings.py', 'src/spatial_pretrain/models/subcellular_v2.py',
            'src/spatial_pretrain/data/subcellular_v2.py')})
    manifest = args.output / 'evaluation_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Output belongs to a different checkpoint, cohort or extraction contract')
    atomic_json(manifest, dict(signature=signature, checkpoint=str(args.checkpoint.resolve()),
        backbone_provenance=provenance, cells=len(cohort),
        test_donors_used=not args.smoke, primary_embedding='joint',
        protocol='Frozen encoder; train-only scaling; validation-selected C; historical full test',
        context='64-cell per-center patch as configured; patch-truncated KNN; no masking',
        historical_test_previously_viewed=True))
    print(f'[preflight] all {len(cohort)} IDs, labels and donor splits match historical baseline', flush=True)
    completed_cells = 0
    for sample, obs in frames.items():
        selected = np.arange(min(8, len(obs)) if args.smoke else len(obs))
        dest = args.output / 'embeddings' / sample
        dest.mkdir(parents=True, exist_ok=True)
        metadata = dest / 'complete.json'
        chosen = obs.iloc[selected][COLUMNS].reset_index(drop=True)
        sample_signature = dict(signature=signature, sample=sample,
            ids_sha256=rows_digest(chosen), inputs={name: dict(
                size=(prepared / sample / name).stat().st_size,
                mtime_ns=(prepared / sample / name).stat().st_mtime_ns)
                for name in ('obs.parquet', 'genes.npy', 'nuclear.npy', 'cytoplasmic.npy')})
        cached = metadata.exists() and json.loads(metadata.read_text())['signature'] == sample_signature
        if cached:
            for key in EMBEDDINGS:
                array = np.load(dest / f'{key}.npy', mmap_mode='r')
                if array.shape != (len(selected), 512) or not np.isfinite(array).all():
                    raise ValueError(f'Invalid cached array {sample}/{key}')
            pd.testing.assert_frame_equal(pd.read_parquet(dest / 'cells.parquet'), chosen)
        else:
            blocks = {key: [] for key in EMBEDDINGS}
            for start in range(0, len(selected), args.chunk_cells):
                if time.time() + 300 > args.deadline:
                    atomic_json(args.output / 'INCOMPLETE.json', dict(reason='allocation_deadline',
                        sample=sample, next_center=start, timestamp=time.time()))
                    return 75
                stop = min(start + args.chunk_cells, len(selected))
                block = dest / f'chunk_{start:08d}_{stop:08d}.npz'
                meta = block.with_suffix('.json')
                valid = meta.exists() and block.exists() and json.loads(meta.read_text())['signature'] == sample_signature
                if valid:
                    with np.load(block) as archive:
                        features = {key: archive[key] for key in EMBEDDINGS}
                else:
                    features, qa = extract(model, cfg, vocab, prepared / sample, obs,
                                           selected[start:stop], args.batch_size, False)
                    temp = block.with_name(block.stem + '.partial.npz')
                    np.savez(temp, **features)
                    temp.replace(block)
                    atomic_json(meta, dict(signature=sample_signature, qa=qa))
                for key, value in features.items():
                    if value.shape != (stop - start, 512) or not np.isfinite(value).all():
                        raise ValueError(f'Invalid extracted features: {sample}/{key}')
                    blocks[key].append(value)
                atomic_json(args.output / 'progress.json', dict(stage='embedding', sample=sample,
                    sample_cells=len(selected), sample_done=stop,
                    completed_cells=completed_cells + stop, total_cells=len(cohort),
                    elapsed_seconds=time.monotonic() - started, timestamp=time.time()))
            for key, values in blocks.items():
                atomic_array(dest / f'{key}.npy', np.concatenate(values))
            chosen.to_parquet(dest / 'cells.parquet', index=False)
            atomic_json(metadata, dict(signature=sample_signature))
        completed_cells += len(selected)
        print(f'[full-sample] {sample} completed_cells={completed_cells}', flush=True)
        if args.smoke:
            atomic_json(args.output / 'SMOKE_COMPLETE.json', dict(cells=len(selected),
                embeddings=list(EMBEDDINGS), finite=True, full_cohort_checked=True))
            return 0
    del model
    torch.cuda.empty_cache()
    atomic_json(args.output / 'progress.json', dict(stage='linear_probes',
        completed_cells=completed_cells, total_cells=len(cohort), timestamp=time.time()))
    probe.OUT = args.output
    probe.main(models=['joint', 'intrinsic', 'niche'])
    results = json.loads((args.output / 'metrics.json').read_text())
    references = historical_rows + [row for row in official_rows if row['model'] == 'scgpt_spatial']
    for row in results:
        for baseline in [r for r in references if r['task'] == row['task']]:
            for key in ('train_cells', 'val_cells', 'test_cells', 'train_classes', 'test_classes',
                        'unseen_test_classes', 'train_donors', 'val_donors', 'test_donors'):
                if row[key] != baseline[key]:
                    raise ValueError(f'Baseline protocol mismatch: {row["task"]}/{key}')
    lines = [f'# 普通scGPT亚细胞模型 step{step}：完整LungPF测试', '',
        '455379细胞、16供体；同历史基线的10/3/3供体划分，逐行核对ID与标签。',
        '冻结检查点，分类器只在训练集拟合、验证集选C。以下是测试集macro-F1。', '',
        '| 模型／表征 | 任务 | Test macro-F1 | Test balanced accuracy |',
        '|---|---|---:|---:|']
    for row in references + results:
        lines.append(f'| {row["model"]} | {row["task"]} | {row["macro_f1"]:.6f} | {row["balanced_accuracy"]:.6f} |')
    lines += ['', '主输出预先指定joint；intrinsic/niche作为诊断一并报告，不按测试成绩挑选主输出。',
        '本模型使用真实空间邻域；输入、初始化和预训练预算不同，不是单因素架构消融。',
        '测试集历史上已被查看；本轮结果不能再称为从未用于研究选择的最终独立测试。',
        '保留测试中训练未见的类别；预训练语料重叠尚未完全排除。']
    (args.output / 'REPORT.zh.md').write_text('\n'.join(lines) + '\n')
    (args.output / 'INCOMPLETE.json').unlink(missing_ok=True)
    atomic_json(args.output / 'FULL_COMPLETE.json', dict(step=step, cells=completed_cells,
        tasks=2, representations=3, seconds=time.monotonic() - started, signature=signature))
    print('[full-done] ' + str(args.output / 'REPORT.zh.md'), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--official-metrics', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--chunk-cells', type=int, default=4096)
    parser.add_argument('--deadline', type=float, default=float('inf'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk_cells < 1:
        parser.error('Batch and chunk sizes must be positive')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
