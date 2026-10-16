"""Frozen ordinary scGPT whole-human baseline on the canonical 5,200-cell cohort.

Raw nuclear + cytoplasmic counts enter the ordinary upstream DataCollator.
There is no spatial-model gene/slide normalization, compartment addition, mask,
neighbor input, or encoder training. CLS is L2-normalized as upstream specifies;
a separate raw gene-token mean readout excludes CLS and PAD for diagnosis.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
import types
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd
import torch
from spatial_pretrain.models.scgpt_backbone import ScGPTBackbone
from evaluate_subcellular_v2 import (CANONICAL_COHORT, atomic_array, atomic_json,
                                   file_sha256, probe_embedding)
import hashlib


def load_ordinary_collator(assets, vocab, cfg, tie_policy='random'):
    """Execute the ordinary upstream collator without package __init__ imports."""
    code = assets / 'code/scgpt'
    package_name = '_plain_scgpt_baseline_upstream'
    package = types.ModuleType(package_name)
    package.__path__ = [str(code)]
    sys.modules[package_name] = package
    preprocess_path = code / 'preprocess.py'
    preprocess = types.ModuleType(package_name + '.preprocess')
    preprocess.__dict__.update(np=np, torch=torch, Union=Union,
                               logger=logging.getLogger(package_name))
    tree = ast.parse(preprocess_path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('_digitize', 'binning')]
    if len(definitions) != 2:
        raise ValueError('Ordinary upstream preprocess lacks the expected binning functions')
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(preprocess_path), 'exec'),
         preprocess.__dict__)
    if tie_policy == 'left':
        # Keep equal expression values in the same deterministic bin.  The
        # upstream implementation instead samples uniformly between the left
        # and right insertion indices when quantile boundaries are repeated.
        def deterministic_digitize(values, bins, side='both'):
            if values.ndim != 1 or bins.ndim != 1:
                raise ValueError('deterministic digitize expects one-dimensional arrays')
            return np.digitize(values, bins)
        preprocess._digitize = deterministic_digitize
    elif tie_policy != 'random':
        raise ValueError(f'Unsupported tie policy: {tie_policy}')
    sys.modules[preprocess.__name__] = preprocess
    collator_path = code / 'data_collator.py'
    specification = importlib.util.spec_from_file_location(package_name + '.data_collator', collator_path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    collator = module.DataCollator(do_padding=True, pad_token_id=vocab[cfg['pad_token']],
        pad_value=cfg['pad_value'], do_mlm=False, do_binning=True, max_length=1200,
        sampling=True, keep_first_n_tokens=1)
    return collator, dict(collator_source=str(collator_path), collator_sha256=file_sha256(collator_path),
        preprocessing_source=str(preprocess_path), preprocessing_sha256=file_sha256(preprocess_path),
        embedding_helper_source=str(code / 'tasks/cell_emb.py'),
        embedding_helper_sha256=file_sha256(code / 'tasks/cell_emb.py'),
        do_mlm=False, do_binning=True, n_bins=51, max_length=1200, keep_first_n_tokens=1,
        sampling=True, tie_policy=tie_policy,
        normalization='None before quantile binning; raw total counts')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--tie-policy', choices=('random', 'left'), default='random',
                        help='random reproduces upstream tie spreading; left keeps equal values in one bin')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.monotonic()
    model = ScGPTBackbone(args.assets_dir, checkpointing=False, backbone_kind='scgpt')
    provenance = dict(model.provenance)
    vocab, cfg = model.vocab, model.config
    if cfg.get('n_bins', 51) != 51 or cfg['input_emb_style'] != 'continuous':
        raise ValueError('This baseline requires the verified ordinary 51-bin continuous-input checkpoint')
    collator, collator_provenance = load_ordinary_collator(
        args.assets_dir, vocab, cfg, tie_policy=args.tie_policy)
    model = model.eval().requires_grad_(False).cuda()
    prepared = ROOT / 'data/benchmarking_data/lungpf/prepared'
    protocol_path = prepared / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    frames, matrices, coverage = [], {'cls': [], 'gene_mean': []}, []
    for sample in protocol['samples']:
        # Resolve and reject test donors before opening any of their slide files.
        matches = [donor for donor in protocol['donor_split'] if sample.startswith(donor)]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely resolve donor before reading {sample}')
        donor = matches[0]
        split = protocol['donor_split'][donor]
        if split == 'test':
            continue
        if split not in ('train', 'val'):
            raise ValueError(f'Unexpected donor split {split}')
        directory = prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        if not obs.patient.eq(donor).all() or ('split' in obs and not obs['split'].eq(split).all()):
            raise ValueError(f'Observation/protocol mismatch for {sample}')
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy()
        chosen['split'] = split
        genes = np.load(directory / 'genes.npy').astype(str)
        keep = np.array([index for index, gene in enumerate(genes) if gene in vocab], dtype=np.int64)
        if not len(keep):
            raise ValueError(f'No ordinary scGPT vocabulary matches in {sample}')
        ids = np.array([vocab[gene] for gene in genes[keep]], dtype=np.int64)
        nuclear = np.load(directory / 'nuclear.npy', mmap_mode='r')
        cytoplasm = np.load(directory / 'cytoplasmic.npy', mmap_mode='r')
        examples = []
        lengths = []
        for row in selected:
            total = nuclear[row, keep].astype(np.float32) + cytoplasm[row, keep].astype(np.float32)
            if not np.isfinite(total).all() or (total < 0).any():
                raise ValueError(f'Invalid raw count input: {sample} row {row}')
            nonzero = np.flatnonzero(total > 0)
            if not len(nonzero):
                raise ValueError(f'{sample} row {row} has no matched expressed genes; ordinary collator cannot bin an empty row')
            examples.append(dict(id=int(row),
                genes=torch.from_numpy(np.r_[vocab['<cls>'], ids[nonzero]]).long(),
                expressions=torch.from_numpy(np.r_[cfg['pad_value'], total[nonzero]]).float()))
            lengths.append(len(nonzero))
        extracted = {'cls': [], 'gene_mean': []}
        for offset in range(0, len(examples), args.batch_size):
            batch = collator(examples[offset:offset + args.batch_size])
            gene = batch['gene'].cuda()
            padding = gene.eq(model.pad_id)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                hidden = model(gene, batch['expr'].cuda(), padding,
                               torch.zeros((*gene.shape, model.dim), device='cuda'))
            hidden = hidden.float()
            cls = hidden[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
            valid = ~padding
            valid[:, 0] = False
            mean = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            if not torch.isfinite(cls).all() or not torch.isfinite(mean).all():
                raise FloatingPointError(f'Nonfinite ordinary scGPT embedding in {sample}')
            extracted['cls'].append(cls.cpu().numpy())
            extracted['gene_mean'].append(mean.cpu().numpy())
        for name in matrices:
            matrices[name].append(np.concatenate(extracted[name]))
        frames.append(chosen)
        coverage.append(dict(sample=sample, donor=donor, split=split, selected_cells=len(selected),
            matched_genes=len(keep), total_genes=len(genes), expressed_gene_count_min=int(min(lengths)),
            expressed_gene_count_max=int(max(lengths)), truncated_cells=int(sum(length > 1199 for length in lengths))))
        print(f'[plain-scgpt-embed] {sample} {split} n={len(selected)} genes={len(keep)}/{len(genes)}', flush=True)
    obs = pd.concat(frames, ignore_index=True)
    cohort = hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest()
    if cohort != CANONICAL_COHORT or len(obs) != 5200 or not obs.full_cell_id.is_unique:
        raise ValueError(f'Wrong fixed validation cohort: cells={len(obs)}, sha256={cohort}')
    features = {name: np.concatenate(values) for name, values in matrices.items()}
    for name, values in features.items():
        atomic_array(args.output / f'embeddings_{name}.npy', values)
    temporary = args.output / 'cells.parquet.partial'
    obs.to_parquet(temporary, index=False)
    temporary.replace(args.output / 'cells.parquet')
    del model
    torch.cuda.empty_cache()
    results = []
    for name, values in features.items():
        results.extend(probe_embedding(obs, values, name, args.output))
    report = dict(model='Official ordinary scGPT whole-human; frozen whole-cell encoder',
        backbone_kind='scgpt', baseline_version=1, cells=len(obs), cohort_sha256=cohort,
        seed=42, selected_cells_per_donor=400, batch_size=args.batch_size,
        tie_policy=args.tie_policy,
        backbone_provenance=provenance, collator_provenance=collator_provenance,
        protocol_sha256=file_sha256(protocol_path), test_donors_read=False,
        train_donors=sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique().tolist()),
        val_donors=sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique().tolist()),
        input='Raw nuclear + cytoplasmic counts; nonzero matched genes only; no log or slide/gene normalization',
        pooling={'cls': 'CLS token, L2 normalized (ordinary official inference)',
                 'gene_mean': 'Mean of non-CLS, non-PAD gene tokens; no L2 normalization (readout diagnostic)'},
        context='Individual whole-cell expression only; zero compartment addition; no spatial adapter or neighbor input',
        inference_dtype='float16 autocast; pooling in float32', gene_coverage=coverage,
        results=results, elapsed_seconds=time.monotonic() - started,
        interpretation='Frozen encoder with supervised donor-held-out linear probes. Pretraining data overlap is not excluded.')
    atomic_json(args.output / 'metrics.json', report)
    lines = ['# 普通 scGPT whole-human 固定验证基线', '',
        '5200 个相同中心细胞；10 个训练供体、3 个验证供体，未读取测试供体切片。',
        '冻结普通 scGPT 官方权重；输入为核＋胞质原始计数，经普通官方 collator 分箱。',
        f'并列表达量策略：{args.tie_policy}；left 表示相同表达量固定进入同一个箱，不做随机拆分。',
        'CLS 按官方方法做 L2 归一化；gene_mean 仅作为额外读出诊断，不做 L2。', '',
        '| 读出 | 任务 | 验证 macro-F1 | 验证 balanced accuracy |', '|---|---|---:|---:|']
    for row in results:
        lines.append(f'| {row["embedding"]} | {row["task"]} | {row["best"]["val_macro_f1"]:.6f} | '
                     f'{row["best"]["val_balanced_accuracy"]:.6f} |')
    lines += ['', '编码器、权重、词表、collator 与预处理源码哈希，以及各供体结果和分类器收敛证据见 metrics.json。',
              '这是冻结表征加监督线性分类器的验证，不是独立测试；预训练数据重叠尚未完全排除。']
    (args.output / 'REPORT.zh.md').write_text('\n'.join(lines) + '\n')
    print(f'[done] {args.output / "metrics.json"}', flush=True)


if __name__ == '__main__':
    main()
