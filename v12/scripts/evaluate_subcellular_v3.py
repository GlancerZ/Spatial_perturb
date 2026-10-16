"""Frozen dual-CLS V3 evaluation on the existing fixed donor-held-out cohort.

Five readouts share the original linear-probe protocol. Masked reconstruction
uses the new pre-binned target space, uniform queries and the actual V3 losses.
No test-donor expression/labels are opened. This evaluator never trains a model.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from evaluate_subcellular_v2 import (
    CANONICAL_COHORT, as_dict, atomic_array, atomic_json, file_sha256,
    pack_tokens, probe_embedding,
)
from spatial_pretrain.data.subcellular_v3 import V3Dataset, collate_v3, encode_context_v3
from spatial_pretrain.models.subcellular_v3 import SubcellularV3
from spatial_pretrain.training.subcellular_v3 import losses as training_losses

EVALUATION_VERSION = 3
EMBEDDING_DIMS = {'base': 1024, 'intrinsic': 1024, 'joint': 1024,
                  'nucleus': 512, 'cytoplasm': 512}
BINNING_PROTOCOL = 'per-cell full nuclear+cytoplasmic panel quantiles before masking; stable 51-bin targets'
ENCODER_FIELDS = ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask', 'node_indices', 'coords')
SOURCE_FILES = ('scripts/train_subcellular_v3.py', 'src/spatial_pretrain/data/subcellular_v3.py',
                'src/spatial_pretrain/models/subcellular_v3.py', 'src/spatial_pretrain/training/subcellular_v3.py',
                'src/spatial_pretrain/models/scgpt_backbone.py', 'src/spatial_pretrain/models/subcellular_v2.py',
                'src/spatial_pretrain/data/subcellular_v2.py')


def validate_project_sources(recorded):
    if not isinstance(recorded, dict) or set(recorded) != set(SOURCE_FILES):
        raise ValueError('V3 checkpoint must record the exact seven training/model/data dependency hashes')
    current = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    differing = [name for name in SOURCE_FILES if recorded[name] != current[name]]
    if differing:
        raise ValueError(f'V3 evaluation source differs from checkpoint: {differing}')
    return current


def load_model(path, expected_step=None, config_path=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    step = int(saved['step'])
    if expected_step is not None and step != expected_step:
        raise ValueError(f'Checkpoint step mismatch: {step} != {expected_step}')
    cfg = as_dict(saved['config'])
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if saved.get('config_sha256') != fingerprint:
        raise ValueError('V3 checkpoint configuration fingerprint is missing or invalid')
    sources = validate_project_sources(saved.get('source_sha256'))
    if config_path is not None:
        requested = yaml.safe_load(config_path.read_text())
        if hashlib.sha256(json.dumps(requested, sort_keys=True).encode()).hexdigest() != fingerprint:
            raise ValueError('--config does not match the checkpoint configuration')
    options = dict(cfg['model'])
    if options.get('backbone_kind') != 'scgpt':
        raise ValueError('V3 evaluation requires explicit ordinary scGPT configuration')
    supported = set(inspect.signature(SubcellularV3).parameters) - {'asset_dir', 'backbone'}
    if set(options) - supported:
        raise ValueError(f'Unsupported V3 model options: {sorted(set(options) - supported)}')
    options['checkpointing'] = False
    model = SubcellularV3(asset_dir=cfg['assets_dir'], **options)
    recorded, loaded = as_dict(saved.get('backbone_provenance', {})), as_dict(model.backbone.provenance)
    for key in ('backbone_kind', 'weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'):
        if not recorded.get(key) or recorded[key] != loaded.get(key):
            raise ValueError(f'V3 checkpoint/pretrained-source provenance mismatch: {key}')
    if recorded['backbone_kind'] != 'scgpt':
        raise ValueError('scGPT-spatial checkpoints cannot be relabeled as V3')
    if step == 0:
        for key, value in model.backbone.state_dict().items():
            archived = saved['model'].get(f'backbone.{key}')
            if archived is None or not torch.equal(value.cpu(), archived.cpu()):
                raise ValueError(f'Step-0 backbone differs from ordinary scGPT: {key}')
    # A V2 state cannot silently populate the dual-CLS/local-exchange model.
    model.load_state_dict(saved['model'], strict=True)
    if not isinstance(model.vocab, dict):
        raise TypeError('V3 requires the verified ordinary scGPT vocabulary mapping')
    provenance = dict(backbone_kind='scgpt', config_sha256=fingerprint,
        checkpoint_source_sha256=sources, checkpoint_sources_verified=True,
        checkpoint_backbone_provenance=recorded, loaded_pretrained_backbone=loaded,
        checkpoint_provenance_verified=True, ordinary_scgpt_initial_weights_verified=step == 0,
        evaluation_model_source_sha256=file_sha256(ROOT / 'src/spatial_pretrain/models/subcellular_v3.py'),
        evaluation_data_source_sha256=file_sha256(ROOT / 'src/spatial_pretrain/data/subcellular_v3.py'),
        evaluation_loss_source_sha256=file_sha256(ROOT / 'src/spatial_pretrain/training/subcellular_v3.py'))
    del saved
    return model.eval().requires_grad_(False).cuda(), cfg, model.vocab, step, provenance


def expected_dimensions(neighbor_shuffle=False):
    dimensions = dict(EMBEDDING_DIMS)
    if neighbor_shuffle:
        dimensions['joint_neighbor_shuffle'] = 1024
    return dimensions


def extract(model, cfg, vocab, directory, obs, selected, batch_size, neighbor_shuffle):
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not obs.full_cell_id.is_unique:
        raise ValueError(f'Invalid coordinates or duplicate cell IDs: {directory.name}')
    context = min(int(cfg['data'].get('context_cells', 64)), len(obs))
    if context < 2:
        raise ValueError('V3 neighborhood evaluation requires at least two cells')
    nearest = NearestNeighbors(n_neighbors=context).fit(coords).kneighbors(coords[selected], return_distance=False)
    nodes = np.stack([np.r_[center, row[row != center][:context - 1]] for center, row in zip(selected, nearest)])
    if any(len(np.unique(row)) != context for row in nodes):
        raise ValueError('KNN patch contains duplicate row IDs')
    expressions = {'real': nodes}
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    if neighbor_shuffle:
        permutation = np.random.default_rng(np.random.SeedSequence([42, slice_seed, 815])).permutation(len(obs))
        shuffled = permutation[nodes].copy()
        for row, center in enumerate(selected):
            shuffled[row, shuffled[row] == center] = permutation[center]
            shuffled[row, 0] = center
        if any(len(np.unique(row)) != context for row in shuffled):
            raise ValueError('Neighbor shuffle duplicated a cell inside a patch')
        expressions['neighbor_shuffle'] = shuffled
    unique = np.unique(np.concatenate(list(expressions.values())))
    genes = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([index for index, gene in enumerate(genes) if gene in vocab], dtype=np.int64)
    if not len(keep):
        raise ValueError(f'No panel genes match the ordinary scGPT vocabulary: {directory.name}')
    global_ids = np.array([vocab[gene] for gene in genes[keep]], dtype=np.int64)
    nuclear = np.load(directory / 'nuclear.npy', mmap_mode='r', allow_pickle=False)
    cytoplasm = np.load(directory / 'cytoplasmic.npy', mmap_mode='r', allow_pickle=False)
    encoded = encode_context_v3(
        np.log1p(nuclear[np.ix_(unique, keep)].astype(np.float32)),
        np.log1p(cytoplasm[np.ix_(unique, keep)].astype(np.float32)), global_ids,
        node_indices=unique, seed=42, epoch=0, slice_seed=slice_seed,
        context_tokens_per_compartment=int(cfg['data'].get('context_tokens_per_compartment', 256)),
        pad_token_id=vocab['<pad>'], cls_token_id=vocab['<cls>'])
    packed = pack_tokens(encoded)
    del encoded
    result = {name: [] for name in expected_dimensions(neighbor_shuffle)}
    started = time.monotonic()
    for condition, expression in expressions.items():
        mapped = np.searchsorted(unique, expression)
        for start in range(0, len(nodes), batch_size):
            end = min(start + batch_size, len(nodes))
            index = mapped[start:end]
            width = int(packed['token_valid_mask'][index].sum(axis=-1).max())
            batch = {key: torch.as_tensor(values[index, :width], device='cuda') for key, values in packed.items()}
            batch['node_indices'] = torch.as_tensor(nodes[start:end], device='cuda')
            batch['coords'] = torch.as_tensor(coords[nodes[start:end]], device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch, return_embeddings=True)['embeddings']
            for key in (EMBEDDING_DIMS if condition == 'real' else ('joint',)):
                value = output[key][:, 0].float().cpu().numpy()
                if value.shape != (end - start, EMBEDDING_DIMS[key]) or not np.isfinite(value).all():
                    raise FloatingPointError(f'Invalid {key} output in {directory.name}: {value.shape}')
                result[key if condition == 'real' else 'joint_neighbor_shuffle'].append(value)
            if start == 0 or end == len(nodes) or end % 50 < batch_size:
                print(f'[embed-progress-v3] {directory.name} {condition} {end}/{len(nodes)} '
                      f'tokens={width} elapsed={time.monotonic()-started:.1f}s', flush=True)
    qa = dict(sample=directory.name, centers=len(selected), graph_cells=context,
              unique_encoded_cells=len(unique), matched_genes=len(keep), panel_genes=len(genes),
              packed_tokens=int(packed['gene_ids'].shape[1]), seconds=time.monotonic()-started)
    return {key: np.concatenate(values) for key, values in result.items()}, qa


def compartment_query_mask(batch):
    valid, compartment = batch['token_valid_mask'][:, 0].bool(), batch['compartment_ids'][:, 0]
    visible = torch.stack([(valid & compartment.eq(index)).any(-1) for index in (0, 1)], -1)
    return visible.gather(1, batch['query_compartment_ids'])


def decoder_metrics(prediction, batch, valid):
    """Query-weighted diagnostics, separate from compartment/cell-weighted loss."""
    valid = valid.bool()
    target = batch['target_bins']
    if bool((((target < 0) | (target > 50) | ~torch.isfinite(target) | (target != target.round())) & valid).any()):
        raise ValueError('Masked-validation targets must be integer bins 0..50 on valid queries')
    logits = prediction[..., 0].float()[valid].cpu().numpy().astype(np.float64)
    bins = target[valid].cpu().numpy().astype(np.int64)
    labels = bins > 0
    predicted = logits >= 0
    count, positive = len(bins), int(labels.sum())
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    both = 0 < positive < count
    probabilities = 1. / (1. + np.exp(-np.clip(logits, -709., 709.)))
    rank = prediction[..., 1].float().sigmoid()[valid].cpu().numpy()[labels].astype(np.float64)
    relative_target = (bins[labels].astype(np.float64) - 1.) / 49.
    error = rank - relative_target
    result = dict(queries=count, positive_queries=positive, true_positive=tp, false_positive=fp,
        false_negative=fn, true_negative=tn,
        nonzero_bce=float(np.mean(np.logaddexp(0., logits) - labels * logits)) if count else None,
        nonzero_accuracy=(tp + tn) / count if count else None,
        nonzero_precision=tp / (tp + fp) if tp + fp else None,
        nonzero_recall=tp / positive if positive else None,
        nonzero_specificity=tn / (tn + fp) if tn + fp else None,
        nonzero_balanced_accuracy=.5 * (tp / positive + tn / (count - positive)) if both else None,
        nonzero_average_precision=float(average_precision_score(labels, logits)) if both else None,
        nonzero_auroc=float(roc_auc_score(labels, logits)) if both else None,
        positive_prevalence=positive / count if count else None,
        predicted_positive_fraction=(tp + fp) / count if count else None,
        positive_probability_mean=float(probabilities[labels].mean()) if positive else None,
        negative_probability_mean=float(probabilities[~labels].mean()) if positive < count else None,
        always_zero_accuracy=(count - positive) / count if count else None,
        always_zero_balanced_accuracy=.5 if both else None,
        positive_rank_mse=float(np.mean(error ** 2)) if positive else None,
        positive_rank_mae=float(np.mean(np.abs(error))) if positive else None,
        positive_bin_mae=float(49. * np.mean(np.abs(error))) if positive else None,
        midpoint_rank_baseline_mse=float(np.mean((.5 - relative_target) ** 2)) if positive else None)
    return result


def aggregate_masked_predictions(outputs, batch, cfg, step):
    total, parts = training_losses(outputs, batch, cfg, step)
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite V3 masked-validation loss')
    result = dict(loss_mean=parts, decoders={})
    comp_valid = batch['query_valid_mask'].bool() & compartment_query_mask(batch)
    result['decoders']['compartment'] = decoder_metrics(outputs['compartment_pred'], batch, comp_valid)
    result['decoders']['joint'] = decoder_metrics(outputs['joint_pred'], batch, batch['query_valid_mask'])
    return result


def masked_validation(model, cfg, step, requested_batches, batch_size):
    if requested_batches <= 0:
        return dict(enabled=False)
    dataset = V3Dataset(cfg, split='val')
    dataset.set_epoch(0)
    ends = np.asarray(dataset.ends, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    chosen = list(dict.fromkeys(int(start + (end - start) // 2) for start, end in zip(starts, ends) if end > start))
    budget = min(len(dataset), max(len(chosen), requested_batches * batch_size))
    if len(chosen) < budget:
        available = np.setdiff1d(np.arange(len(dataset)), chosen, assume_unique=False)
        chosen += np.random.default_rng(173).choice(available, budget - len(chosen), replace=False).tolist()
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    fields = ('query_gene_ids', 'query_valid_mask', 'query_compartment_ids', 'target_bins', 'task_mode')
    gathered = {key: [] for key in fields + ('token_valid_mask', 'compartment_ids')}
    predictions = {'compartment_pred': [], 'joint_pred': []}
    target_hash = hashlib.sha256()
    started = time.monotonic()
    for start in range(0, len(chosen), batch_size):
        batch = collate_v3([dataset[int(index)] for index in chosen[start:start + batch_size]])
        device_batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(device_batch)
        for key in predictions:
            if not torch.isfinite(output[key]).all():
                raise FloatingPointError(f'Nonfinite V3 masked prediction: {key}')
            predictions[key].append(output[key].detach().float().cpu())
        for key in fields:
            gathered[key].append(batch[key])
        # Loss only inspects the center's original visible-token masks. Keep
        # actual center fields and pad them below, never fabricate visibility.
        for key in ('token_valid_mask', 'compartment_ids'):
            gathered[key].append(batch[key][:, :1])
        del device_batch, output
        print(f'[masked-v3] {min(start+batch_size,len(chosen))}/{len(chosen)}', flush=True)
    width = max(value.shape[-1] for value in gathered['token_valid_mask'])
    for key, pad in (('token_valid_mask', False), ('compartment_ids', 2)):
        gathered[key] = [F.pad(value, (0, width - value.shape[-1]), value=pad) for value in gathered[key]]
    batch = {key: torch.cat(values) for key, values in gathered.items()}
    for key in ('query_gene_ids', 'query_compartment_ids', 'query_valid_mask', 'target_bins'):
        target_hash.update(batch[key].numpy().tobytes())
    outputs = {key: torch.cat(values) for key, values in predictions.items()}
    result = aggregate_masked_predictions(outputs, batch, cfg, step)
    by_mode = {}
    for mode, name in ((0, 'gene_mask'), (1, 'compartment_mask'), (2, 'whole_cell_mask')):
        selection = batch['task_mode'].eq(mode)
        if selection.any():
            subset = {key: value[selection] for key, value in batch.items()}
            subset_output = {key: value[selection] for key, value in outputs.items()}
            by_mode[name] = dict(cells=int(selection.sum()), **aggregate_masked_predictions(subset_output, subset, cfg, step))
    result.update(enabled=True, centers=len(chosen), total_validation_centers=len(dataset),
        slices_covered=int((ends > starts).sum()), requested_minimum_batches=requested_batches,
        effective_batches=(len(chosen) + batch_size - 1) // batch_size,
        sample_indices_sha256=hashlib.sha256(chosen.tobytes()).hexdigest(),
        query_targets_sha256=target_hash.hexdigest(), by_task_mode=by_mode,
        protocol='Fixed epoch-0 masks and uniform masked-panel queries; at least one spatial-held-out center per pretraining slide. '
                 'Actual training loss reduced once over the full fixed cohort; decoder diagnostics weighted by valid queries.',
        binning_protocol=BINNING_PROTOCOL,
        limitation='Pre-mask quantiles contain statistics of subsequently hidden counts; this is denoising in a pre-binned space, '
                   'not raw-count visible-only isolation. No absolute-count target is used.',
        positive_rank_target='(positive_bin - 1) / 49; sigmoid prediction; constant midpoint baseline 0.5 is not fitted to validation targets',
        seconds=time.monotonic() - started)
    return result


def write_report(output, report):
    lines = [f'# V3 checkpoint {report["step"]}：双区室 CLS 固定验证', '',
             '仅使用普通 scGPT 骨干；核/胞质独立编码，细胞间直接连接仅为胞质。',
             'base、intrinsic、joint 为 1024 维，nucleus、cytoplasm 为 512 维；'
             '不能把相对历史 512 维模型的差异全部归因于结构。', '',
             '| 表征 | 任务 | 维度 | Best macro-F1 | Best C | 固定C macro-F1 |',
             '|---|---|---:|---:|---:|---:|']
    for row in report['results']:
        fixed = row['fixed_C_result']
        lines.append(f'| {row["embedding"]} | {row["task"]} | {report["embedding_dimensions"][row["embedding"]]} | '
                     f'{row["best"]["val_macro_f1"]:.6f} | {row["best"]["C"]:g} | {fixed["val_macro_f1"]:.6f} |')
    lines += ['', '固定 C：细胞类型 0.1、区域 0.01。训练供体拟合 StandardScaler，'
              'class-balanced 线性 probe，C=[0.01,0.1,1,10] 仅在验证集选择；全部求解器必须收敛。',
              f'中心细胞数 {report["cells"]}，cohort SHA256：`{report["cohort_sha256"]}`。未读取 test 切片。',
              'base 是交互前两个区室 CLS；intrinsic 是关闭空间消息后执行相同 local 模块；'
              'joint 完成 Local0→Spatial1→Local1→Spatial2→Local2。nucleus/cytoplasm 为最终状态，'
              '可能已间接包含邻域信息。空间增益应比较相同维度的 joint 与 intrinsic。']
    if report['neighbor_shuffle']:
        lines += ['邻域打乱保持中心表达、坐标与图不变，打乱同切片邻居表达并排除中心表达重复；'
                  '检验邻域组成贡献，不单独证明因果互作。']
    masked = report['masked_validation']
    if masked.get('enabled'):
        lines += ['', '固定遮盖验证：', '', '| Loss | Mean |', '|---|---:|']
        lines += [f'| {key} | {value:.6f} |' for key, value in masked['loss_mean'].items()]
        lines += ['', '目标仅为非零分类与正 bin 的相对等级，不使用绝对 count。'
                  '完整核/胞质共用每细胞分箱表，然后遮盖；可见 bin 含隐藏原值的统计依赖，'
                  '本协议不声称原始计数级 visible-only 隔离。',
                  'query 从被遮盖的已测 panel 均匀采样。PAD 无监督；空区室不计算对应 compartment loss。'
                  '逐mask模式的precision/recall/AP/AUROC、全零基线及positive-rank MSE/MAE见metrics.json。']
    temporary = output / 'REPORT.zh.md.partial'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'REPORT.zh.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', '--output', dest='output', type=Path, required=True)
    parser.add_argument('--cohort', default='fixed5200', help='fixed5200 or a reference cells.parquet for exact ID/split validation')
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cells-per-donor', type=int, default=400)
    parser.add_argument('--neighbor-shuffle', action='store_true')
    parser.add_argument('--mask-validation-batches', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1 or args.cells_per_donor < 1 or args.mask_validation_batches < 0:
        parser.error('batch size and cells per donor must be positive; mask validation batches nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.monotonic()
    model, cfg, vocab, step, provenance = load_model(args.checkpoint, args.expected_step, args.config)
    existing = args.output / 'metrics.json'
    if existing.exists() and json.loads(existing.read_text()).get('evaluation_version') != EVALUATION_VERSION:
        raise ValueError('Refusing to mix V2 or baseline metrics into a V3 evaluation directory')
    checkpoint_digest = file_sha256(args.checkpoint)
    vocab_digest = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()
    prepared = ROOT / 'data/benchmarking_data/lungpf/prepared'
    protocol = json.loads((prepared / 'protocol.json').read_text())
    frames, matrices, coverage = [], {}, []
    cache_dir = args.output / 'sample_embeddings'
    cache_dir.mkdir(exist_ok=True)
    for sample in protocol['samples']:
        donors = [donor for donor in protocol['donor_split'] if sample.startswith(donor)]
        if len(donors) != 1:
            raise ValueError(f'Ambiguous donor: {sample}')
        donor = donors[0]
        split = protocol['donor_split'][donor]
        if split == 'test':
            continue  # Resolve split before opening any sample observation/expression file.
        if split not in ('train', 'val'):
            raise ValueError(f'Unexpected split: {split}')
        directory = prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        if not obs.patient.eq(donor).all() or ('split' in obs and not obs['split'].eq(split).all()):
            raise ValueError(f'Observation/protocol donor split mismatch: {sample}')
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(args.cells_per_donor, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy()
        chosen['split'] = split
        signature = dict(evaluation_version=EVALUATION_VERSION, step=step,
            checkpoint_sha256=checkpoint_digest, config_sha256=provenance['config_sha256'],
            backbone_kind='scgpt', pretrained_weights_sha256=provenance['loaded_pretrained_backbone']['weights_sha256'],
            model_source_sha256=provenance['evaluation_model_source_sha256'],
            data_source_sha256=provenance['evaluation_data_source_sha256'],
            evaluator_source_sha256=file_sha256(Path(__file__)), vocab_mapping_sha256=vocab_digest,
            binning_protocol=BINNING_PROTOCOL, context_cells=int(cfg['data'].get('context_cells', 64)),
            context_tokens_per_compartment=int(cfg['data'].get('context_tokens_per_compartment', 256)),
            neighbor_shuffle=args.neighbor_shuffle,
            selected_ids_sha256=hashlib.sha256('\n'.join(chosen.full_cell_id.astype(str)).encode()).hexdigest())
        cache, metadata = cache_dir / f'{sample}.npz', cache_dir / f'{sample}.json'
        prior = json.loads(metadata.read_text()) if metadata.exists() else {}
        if cache.exists() and prior.get('signature') == signature:
            with np.load(cache, allow_pickle=False) as archive:
                features = {key: archive[key] for key in archive.files}
            qa = prior['qa']
        else:
            features, qa = extract(model, cfg, vocab, directory, obs, selected, args.batch_size, args.neighbor_shuffle)
            temporary = cache.with_name(cache.stem + '.partial.npz')
            np.savez(temporary, **features)
            temporary.replace(cache)
            atomic_json(metadata, dict(signature=signature, qa=qa))
        if set(features) != set(expected_dimensions(args.neighbor_shuffle)):
            raise ValueError(f'Cached readout set mismatch: {sample}')
        for key, dim in expected_dimensions(args.neighbor_shuffle).items():
            value = features[key]
            if value.shape != (len(selected), dim) or not np.isfinite(value).all():
                raise ValueError(f'Invalid cached {key} embedding: {sample}')
            matrices.setdefault(key, []).append(value)
        frames.append(chosen)
        coverage.append(qa)
    obs = pd.concat(frames, ignore_index=True)
    if not obs.full_cell_id.is_unique or set(obs['split']) != {'train', 'val'}:
        raise ValueError('Evaluation cohort must contain unique train/validation cells only')
    cohort = hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest()
    if args.cells_per_donor == 400 and cohort != CANONICAL_COHORT:
        raise ValueError('V3 evaluation does not match the fixed 5,200-cell cohort')
    if args.cohort not in ('fixed5200', CANONICAL_COHORT):
        reference = pd.read_parquet(Path(args.cohort), columns=['full_cell_id', 'patient', 'split'])
        if not reference.reset_index(drop=True).equals(obs[['full_cell_id', 'patient', 'split']].reset_index(drop=True)):
            raise ValueError('Reference cohort IDs, order, donors or splits differ')
    temporary = args.output / 'cells.parquet.partial'
    obs.to_parquet(temporary, index=False)
    temporary.replace(args.output / 'cells.parquet')
    reconstruction = masked_validation(model, cfg, step, args.mask_validation_batches, args.batch_size)
    del model
    torch.cuda.empty_cache()
    results, diagnostics = [], {}
    for key, blocks in matrices.items():
        features = np.concatenate(blocks)
        atomic_array(args.output / f'{key}_embeddings.npy', features)
        precise = features.astype(np.float64)
        diagnostics[key] = dict(mean_dimension_std=float(precise.std(axis=0).mean()),
            mean_embedding_norm=float(np.linalg.norm(precise, axis=1).mean()),
            near_constant_dimensions=int((precise.std(axis=0) < 1e-6).sum()))
        del precise
        for row in probe_embedding(obs, features, key, args.output):
            fixed = .1 if row['task'] == 'cell_type' else .01
            row['fixed_C_result'] = next(item for item in row['grid'] if item['C'] == fixed)
            results.append(row)
    report = dict(model='SubcellularV3', evaluation_version=EVALUATION_VERSION, diagnostic_schema_version=1,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_digest, step=step,
        backbone_kind='scgpt', backbone_provenance=provenance,
        source_sha256=provenance['checkpoint_source_sha256'],
        config_sha256=provenance['config_sha256'], vocab_mapping_sha256=vocab_digest,
        seed=42, evaluation_batch_size=args.batch_size, cells=len(obs),
        cells_per_donor=args.cells_per_donor, cohort_sha256=cohort,
        canonical_cohort=cohort == CANONICAL_COHORT, test_donors_used=False,
        train_donors=sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique()),
        val_donors=sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique()),
        context_cells=int(cfg['data'].get('context_cells', 64)), embedding_dimensions=expected_dimensions(args.neighbor_shuffle),
        input='Independent nuclear/cytoplasmic scGPT CLS sequences; full visible expression; no downstream mask',
        binning_protocol=BINNING_PROTOCOL, graph='Per-center KNN patch; all spatial neighborhoods truncated to this patch',
        neighbor_shuffle=args.neighbor_shuffle, masked_validation=reconstruction,
        diagnostics=diagnostics, coverage=coverage, results=results, seconds=time.monotonic() - started)
    write_report(args.output, report)
    atomic_json(args.output / 'metrics.json', report)
    (args.output / 'diagnostic_failure.json').unlink(missing_ok=True)
    print(f'[done-v3] step={step} seconds={report["seconds"]:.1f} output={args.output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        for option in ('--output-dir', '--output'):
            if option in sys.argv:
                directory = Path(sys.argv[sys.argv.index(option) + 1])
                directory.mkdir(parents=True, exist_ok=True)
                kind = ('nonfinite' if isinstance(error, FloatingPointError) else
                        'out_of_memory' if isinstance(error, torch.cuda.OutOfMemoryError) else 'runtime')
                atomic_json(directory / 'diagnostic_failure.json', dict(kind=kind, error_type=type(error).__name__,
                            detail=str(error)[:3000], timestamp=time.time()))
                break
        raise
