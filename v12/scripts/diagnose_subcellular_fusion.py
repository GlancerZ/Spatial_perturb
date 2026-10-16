"""Frozen, train/validation-only fusion diagnostics from existing V2 embeddings.

No model loading, expression extraction, checkpoint modification, or test reads.
Run on a compute allocation: the reused linear probe uses CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from evaluate_subcellular_v2 import (
    CANONICAL_COHORT, EMBEDDINGS, atomic_json, file_sha256, probe_embedding,
)
from probe_lungpf_embeddings import C_GRID


def ids_digest(ids):
    return hashlib.sha256('\n'.join(ids.astype(str)).encode()).hexdigest()


def load_verified_inputs(source):
    metrics = json.loads((source / 'metrics.json').read_text())
    if metrics.get('evaluation_version') != 2 or metrics.get('backbone_kind') != 'scgpt':
        raise ValueError('Requires completed V2 evaluation with ordinary scGPT provenance')
    if metrics.get('test_donors_used') is not False or metrics.get('neighbor_shuffle'):
        raise ValueError('Requires ordinary, unshuffled train/validation evaluation')
    provenance = metrics['backbone_provenance']
    if not provenance.get('checkpoint_provenance_verified'):
        raise ValueError('Source checkpoint provenance was not verified')
    if metrics['config_sha256'] != provenance['config_sha256']:
        raise ValueError('Source configuration hashes disagree')
    recorded = provenance['checkpoint_backbone_provenance']
    loaded = provenance['loaded_pretrained_backbone']
    for entry in (recorded, loaded):
        if entry.get('backbone_kind') != 'scgpt':
            raise ValueError('Ordinary scGPT provenance is required')
        if entry.get('source_repository', '').rstrip('/') != 'https://github.com/bowang-lab/scGPT':
            raise ValueError('Unexpected pretrained source repository')
    for key in ('weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'):
        if not recorded.get(key) or recorded[key] != loaded.get(key):
            raise ValueError(f'Pretrained provenance mismatch: {key}')

    cells = pd.read_parquet(source / 'cells.parquet')
    required = {'full_cell_id', 'patient', 'split', 'final_CT', 'region_label'}
    if not required.issubset(cells.columns):
        raise ValueError(f'Missing cohort columns: {required - set(cells.columns)}')
    if set(cells['split']) != {'train', 'val'} or not cells.full_cell_id.is_unique:
        raise ValueError('Cohort must contain unique train/validation cells only')
    if cells.groupby('patient')['split'].nunique().max() != 1:
        raise ValueError('Donor occurs in both train and validation')
    digest = ids_digest(cells.full_cell_id)
    if digest != CANONICAL_COHORT or digest != metrics['cohort_sha256']:
        raise ValueError('Cohort IDs/order differ from the fixed validation protocol')
    if len(cells) != metrics['cells']:
        raise ValueError('Cohort size disagrees with source metrics')
    for split in ('train', 'val'):
        if sorted(cells.loc[cells['split'].eq(split), 'patient'].unique()) != metrics[f'{split}_donors']:
            raise ValueError(f'{split} donor list disagrees with source metrics')

    features, fingerprints = {}, {}
    for key in EMBEDDINGS:
        path = source / f'{key}_embeddings.npy'
        array = np.load(path, allow_pickle=False)
        if array.shape != (len(cells), 512) or not np.isfinite(array).all():
            raise ValueError(f'Invalid {key} embedding shape or values')
        features[key] = array
        fingerprints[path.name] = file_sha256(path)

    # The aggregate NPY has no embedded provenance. Cross-check every row
    # against the per-slide cache that is bound to checkpoint/config/source.
    offset = 0
    verified_samples = []
    for coverage in metrics['coverage']:
        sample, count = coverage['sample'], int(coverage['centers'])
        cache = source / 'sample_embeddings' / f'{sample}.npz'
        metadata_path = cache.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        signature = metadata['signature']
        expected = dict(checkpoint_sha256=metrics['checkpoint_sha256'],
                        config_sha256=metrics['config_sha256'], backbone_kind='scgpt',
                        pretrained_weights_sha256=loaded['weights_sha256'],
                        vocab_mapping_sha256=metrics['vocab_mapping_sha256'],
                        step=metrics['step'], extraction_version=2, neighbor_shuffle=False,
                        context_cells=metrics['context_cells'],
                        selected_ids_sha256=ids_digest(cells.iloc[offset:offset + count].full_cell_id))
        for key, value in expected.items():
            if signature.get(key) != value:
                raise ValueError(f'Cache signature mismatch: {sample}/{key}')
        with np.load(cache, allow_pickle=False) as archive:
            for key in EMBEDDINGS:
                if not np.array_equal(features[key][offset:offset + count], archive[key]):
                    raise ValueError(f'Aggregate/cache embedding mismatch: {sample}/{key}')
        verified_samples.append(sample)
        offset += count
    if offset != len(cells):
        raise ValueError('Per-sample caches do not cover the complete cohort')
    fingerprints['metrics.json'] = file_sha256(source / 'metrics.json')
    fingerprints['cells.parquet'] = file_sha256(source / 'cells.parquet')
    return metrics, cells, features, dict(file_sha256=fingerprints,
        verified_samples=verified_samples, aggregate_cache_arrays_identical=True)


def verify_reproduction(actual, original, tolerance):
    references = {(row['embedding'], row['task']): row for row in original}
    checks = []
    for row in actual:
        key = (row['embedding'], row['task'])
        reference = references[key]
        for field in ('train_cells', 'val_cells', 'unseen_val_classes'):
            if row[field] != reference[field]:
                raise ValueError(f'Reproduction cohort mismatch: {key}/{field}')
        if [item['C'] for item in row['grid']] != C_GRID:
            raise ValueError('Probe C grid changed')
        if [item['C'] for item in reference['grid']] != C_GRID:
            raise ValueError('Source C grid differs')
        differences = []
        for current, previous in zip(row['grid'], reference['grid']):
            if not current['solver']['converged_gradient'] or not previous['solver']['converged_gradient']:
                raise ValueError(f'Unconverged probe: {key}, C={current["C"]}')
            differences.extend(abs(current[field] - previous[field])
                               for field in ('val_macro_f1', 'val_balanced_accuracy'))
            if set(current['per_donor_macro_f1']) != set(previous['per_donor_macro_f1']):
                raise ValueError('Probe donor set changed')
            differences.extend(abs(score - previous['per_donor_macro_f1'][donor])
                               for donor, score in current['per_donor_macro_f1'].items())
        maximum = max(differences)
        if maximum > tolerance or row['best']['C'] != reference['best']['C']:
            raise ValueError(f'Original probe not reproduced: {key}, max score delta={maximum}')
        checks.append(dict(embedding=key[0], task=key[1], max_score_difference=maximum,
                           same_best_C=True, passed=True))
    return checks


def write_report(output, report):
    lines = [f'# 检查点 {report["step"]}：冻结表征融合诊断', '',
             '使用已有 embedding，不更新模型。原始三种表征的所有 C 和供体分数已复现。', '',
             '| 表征 | 任务 | 维度 | 验证 macro-F1 | 最优 C |',
             '|---|---|---:|---:|---:|']
    for row in report['results']:
        lines.append(f'| {row["embedding"]} | {row["task"]} | '
                     f'{report["dimensions"][row["embedding"]]} | '
                     f'{row["best"]["val_macro_f1"]:.6f} | {row["best"]["C"]:g} |')
    lines += ['', '同一固定 5,200 中心细胞、10/3 训练/验证供体；未读取测试集。'
              '每任务的标签筛选、训练集 StandardScaler、class-balanced 线性分类器、'
              'C=[0.01,0.1,1,10] 和梯度收敛检查直接复用原评测函数。', '',
              'PCA512 先在全部训练供体的拼接表征上拟合 StandardScaler 和 PCA，'
              '然后固定变换验证表征；不使用验证数据拟合或选择 PCA。'
              '每任务线性 probe 仍按原协议仅用该任务有标签的训练细胞拟合标准化。', '',
              'intrinsic+niche 拼接用于诊断两种表征是否互补；intrinsic+joint 是同维度的组合对照。'
              'PCA512 控制输出维度，但也改变几何和正则化，不能单独证明某个训练模块是原因。'
              '这些是验证集诊断，不是新模型训练结果或最终测试成绩。']
    target = output / 'REPORT.zh.md'
    temporary = target.with_suffix('.md.partial')
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', '--evaluation-dir', dest='source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--reproduction-tolerance', type=float, default=1e-6)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or source in output.parents:
        parser.error('Output must be outside the source evaluation directory')
    if args.reproduction_tolerance < 0 or not np.isfinite(args.reproduction_tolerance):
        parser.error('Reproduction tolerance must be finite and nonnegative')
    if output.exists() and any(output.iterdir()):
        parser.error('Output must be a new or empty directory; source results are never overwritten')
    original, cells, features, verification = load_verified_inputs(source)
    if args.expected_step is not None and original['step'] != args.expected_step:
        parser.error('Source checkpoint step differs from --expected-step')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    started = time.monotonic()
    report = dict(kind='frozen_subcellular_fusion_diagnostic', version=1, step=original['step'],
                  source=str(source), checkpoint_sha256=original['checkpoint_sha256'],
                  config_sha256=original['config_sha256'], backbone_kind=original['backbone_kind'],
                  backbone_provenance=original['backbone_provenance'],
                  cohort_sha256=original['cohort_sha256'], train_donors=original['train_donors'],
                  val_donors=original['val_donors'], test_donors_used=False, seed=42, C_grid=C_GRID,
                  input_verification=verification, reproduction_tolerance=args.reproduction_tolerance,
                  probe_source_sha256=file_sha256(Path(__file__).with_name('evaluate_subcellular_v2.py')),
                  solver_source_sha256=file_sha256(Path(__file__).with_name('probe_lungpf_embeddings.py')),
                  diagnostic_source_sha256=file_sha256(Path(__file__)), results=[], dimensions={})
    try:
        for name in EMBEDDINGS:
            report['results'].extend(probe_embedding(cells, features[name], name, output))
            report['dimensions'][name] = 512
            atomic_json(output / 'progress.json', report)
        report['reproduction'] = verify_reproduction(report['results'], original['results'],
                                                     args.reproduction_tolerance)
        atomic_json(output / 'reproduction.json', report['reproduction'])

        combined = np.concatenate([features['intrinsic'], features['niche']], axis=1)
        train = cells['split'].eq('train').to_numpy()
        pca_scaler = StandardScaler().fit(combined[train])
        standardized = pca_scaler.transform(combined).astype(np.float64)
        pca = PCA(n_components=512, svd_solver='full', random_state=42).fit(standardized[train])
        projected = pca.transform(standardized).astype(np.float32)
        np.savez(output / 'intrinsic_niche_pca512_transform.npz',
                 scaler_mean=pca_scaler.mean_, scaler_scale=pca_scaler.scale_,
                 pca_mean=pca.mean_, components=pca.components_,
                 explained_variance_ratio=pca.explained_variance_ratio_)
        report['pca'] = dict(fit_split='train', fit_cells=int(train.sum()), components=512,
                            solver='full', whiten=False, prestandardize=True,
                            explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()))
        variants = {
            'intrinsic_niche_concat1024': combined,
            'intrinsic_niche_pca512': projected,
            'intrinsic_joint_concat1024': np.concatenate([features['intrinsic'], features['joint']], axis=1),
        }
        for name, value in variants.items():
            report['dimensions'][name] = value.shape[1]
            report['results'].extend(probe_embedding(cells, value, name, output))
            atomic_json(output / 'progress.json', report)
        report['seconds'] = time.monotonic() - started
        write_report(output, report)
        atomic_json(output / 'metrics.json', report)
        print(f'[done] step={report["step"]} seconds={report["seconds"]:.1f} output={output}', flush=True)
    except Exception as error:
        atomic_json(output / 'diagnostic_failure.json', dict(error_type=type(error).__name__,
                    detail=str(error), seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
