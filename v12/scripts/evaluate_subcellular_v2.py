"""Frozen V2 probes on the existing, fixed 5,200-cell validation cohort.

Each cell is the center of its own fixed KNN patch. Labels never enter encoding;
held-out test slides are not opened. The graph is explicitly patch-truncated.
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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from spatial_pretrain.data.subcellular_v2 import V2Dataset, collate_v2, encode_context
from spatial_pretrain.models.subcellular_v2 import SubcellularV2
from probe_lungpf_embeddings import C_GRID, fit_linear, predict
from train_subcellular_v2 import losses as training_losses

CANONICAL_COHORT = '0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc'
EMBEDDINGS = ('intrinsic', 'joint', 'niche')
DIAGNOSTIC_SCHEMA_VERSION = 1


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def atomic_array(path, value):
    temporary = path.with_name(path.stem + '.partial.npy')
    np.save(temporary, value)
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def as_dict(value):
    if isinstance(value, dict):
        return {key: as_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_dict(item) for item in value]
    if hasattr(value, '__dict__'):
        return as_dict(vars(value))
    return value


def load_model(path, expected_step):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    step = int(checkpoint['step'])
    if expected_step is not None and step != expected_step:
        raise ValueError(f'Checkpoint step mismatch: {step} != {expected_step}')
    cfg = as_dict(checkpoint['config'])
    model_cfg = dict(cfg.get('model', {}))
    backbone_kind = model_cfg.get('backbone_kind', 'scgpt_spatial')
    if backbone_kind not in ('scgpt', 'scgpt_spatial'):
        raise ValueError(f'Unknown pretrained backbone family: {backbone_kind}')
    cfg_digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    recorded_cfg_digest = checkpoint.get('config_sha256')
    if recorded_cfg_digest is not None and recorded_cfg_digest != cfg_digest:
        raise ValueError('Checkpoint configuration fingerprint does not match its contents')
    recorded = as_dict(checkpoint.get('backbone_provenance', {}))
    if backbone_kind == 'scgpt':
        if recorded_cfg_digest is None or recorded.get('backbone_kind') != 'scgpt':
            raise ValueError('Ordinary scGPT evaluation requires explicit matching checkpoint provenance; '
                             'a legacy/spatial checkpoint cannot be relabeled as scGPT')
    assets = cfg.get('assets_dir', model_cfg.pop('asset_dir', None))
    if assets is None:
        raise ValueError('Checkpoint must identify assets_dir')
    supported = set(inspect.signature(SubcellularV2).parameters) - {'asset_dir'}
    unknown = set(model_cfg) - supported
    if unknown:
        raise ValueError(f'Unrecognized V2 model options: {sorted(unknown)}')
    model_cfg['checkpointing'] = False
    model = SubcellularV2(asset_dir=assets, **model_cfg)
    loaded = as_dict(model.backbone.provenance)
    if loaded.get('backbone_kind', 'scgpt_spatial') != backbone_kind:
        raise ValueError('Configured and actually loaded backbone families differ')
    provenance_keys = ('weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256')
    for key in provenance_keys:
        if backbone_kind == 'scgpt' and (key not in recorded or key not in loaded):
            raise ValueError(f'Ordinary scGPT checkpoint is missing required provenance: {key}')
        if key in recorded and recorded[key] != loaded.get(key):
            raise ValueError(f'Pretrained asset differs from checkpoint provenance: {key}')
    if backbone_kind == 'scgpt' and step == 0:
        # Verify the new experiment's initial encoder against the actual plain
        # scGPT asset. Shape compatibility alone cannot distinguish the two
        # foundation models and must never justify loading a spatial checkpoint.
        for key, value in model.backbone.state_dict().items():
            archived = checkpoint['model'].get(f'backbone.{key}')
            if archived is None or not torch.equal(value.cpu(), archived.cpu()):
                raise ValueError(f'Step-0 backbone differs from ordinary scGPT pretrained weights: {key}')
    model.load_state_dict(checkpoint['model'], strict=True)
    vocab = model.vocab
    if not isinstance(vocab, dict):
        raise TypeError('SubcellularV2.vocab must be the official gene-to-ID dictionary')
    provenance = dict(backbone_kind=backbone_kind, config_sha256=cfg_digest,
        checkpoint_backbone_provenance=recorded, loaded_pretrained_backbone=loaded,
        checkpoint_provenance_verified=bool(recorded),
        ordinary_scgpt_initial_weights_verified=backbone_kind == 'scgpt' and step == 0)
    del checkpoint
    return model.eval().requires_grad_(False).cuda(), cfg, vocab, step, provenance


def pack_tokens(encoded):
    """No sequence positions are used; retain every valid token and omit PAD."""
    valid = encoded['token_valid_mask']
    width = int(valid.sum(axis=1).max())
    if width < 1:
        raise ValueError('Every cell must retain its non-gene sentinel')
    packed = {}
    for key in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask'):
        values = encoded[key]
        # Initial padding values come from the helper, not inferred gene IDs.
        if key == 'token_valid_mask':
            destination = np.zeros((len(values), width), dtype=bool)
        else:
            padded = values[~valid]
            fill = padded[0] if len(padded) else (2 if key == 'compartment_ids' else 0)
            destination = np.full((len(values), width), fill, dtype=values.dtype)
        for row in range(len(values)):
            destination[row, :int(valid[row].sum())] = values[row, valid[row]]
        packed[key] = destination
    return packed


def extract(model, cfg, vocab, directory, obs, selected, batch_size, neighbor_shuffle):
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all():
        raise ValueError(f'Non-finite spatial coordinates in {directory.name}')
    if not obs.full_cell_id.is_unique:
        raise ValueError(f'Duplicate cell IDs in {directory.name}')
    context = min(int(cfg['data'].get('context_cells', 64)), len(obs))
    if context < 2:
        raise ValueError('At least two cells are required for neighborhood evaluation')
    search = NearestNeighbors(n_neighbors=context).fit(coords)
    nearest = search.kneighbors(coords[selected], return_distance=False)
    nodes = np.stack([np.r_[center, row[row != center][:context - 1]]
                      for center, row in zip(selected, nearest)])
    if any(len(np.unique(row)) != context for row in nodes):
        raise ValueError('KNN context contains duplicate row IDs')

    expression_nodes = {'real': nodes}
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    if neighbor_shuffle:
        # One deterministic slide-wide expression permutation, corrected per
        # patch to keep the target expression fixed and absent from neighbors.
        permutation = np.random.default_rng(np.random.SeedSequence([42, slice_seed, 815])).permutation(len(obs))
        shuffled = permutation[nodes].copy()
        for row, center in enumerate(selected):
            shuffled[row, shuffled[row] == center] = permutation[center]
            shuffled[row, 0] = center
        if any(len(np.unique(row)) != context for row in shuffled):
            raise ValueError('Expression permutation duplicated a cell within a patch')
        expression_nodes['neighbor_shuffle'] = shuffled

    unique = np.unique(np.concatenate(list(expression_nodes.values())))
    genes = np.load(directory / 'genes.npy').astype(str)
    keep = np.array([i for i, gene in enumerate(genes) if gene in vocab], dtype=np.int64)
    if not len(keep):
        raise ValueError(f'No genes matched the official vocabulary in {directory.name}')
    global_ids = np.array([vocab[gene] for gene in genes[keep]], dtype=np.int64)
    nuclear = np.load(directory / 'nuclear.npy', mmap_mode='r')
    cytoplasm = np.load(directory / 'cytoplasmic.npy', mmap_mode='r')
    nuclear_log = np.log1p(nuclear[np.ix_(unique, keep)].astype(np.float32))
    cyto_log = np.log1p(cytoplasm[np.ix_(unique, keep)].astype(np.float32))
    encoded = encode_context(nuclear_log, cyto_log, global_ids, node_indices=unique,
        seed=42, epoch=0, slice_seed=slice_seed,
        context_tokens_per_compartment=int(cfg['data'].get('context_tokens_per_compartment', 256)),
        pad_token_id=vocab['<pad>'], cls_token_id=vocab['<cls>'])
    packed = pack_tokens(encoded)
    del nuclear_log, cyto_log, encoded
    result = {key: [] for key in EMBEDDINGS}
    if neighbor_shuffle:
        result['joint_neighbor_shuffle'] = []
    started = time.monotonic()
    for condition, expression in expression_nodes.items():
        mapped = np.searchsorted(unique, expression)
        for start in range(0, len(nodes), batch_size):
            end = min(start + batch_size, len(nodes))
            index = mapped[start:end]
            valid = packed['token_valid_mask'][index]
            width = int(valid.sum(axis=-1).max())
            batch = {key: torch.as_tensor(values[index, :width], device='cuda')
                     for key, values in packed.items()}
            # These IDs describe positions in the graph, not permuted expression
            # donors. No coordinate or cell-ID permutation changes the center.
            batch['node_indices'] = torch.as_tensor(nodes[start:end], device='cuda')
            batch['coords'] = torch.as_tensor(coords[nodes[start:end]], device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch, return_embeddings=True)['embeddings']
            for key in (EMBEDDINGS if condition == 'real' else ('joint',)):
                value = output[key][:, 0].float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise FloatingPointError(f'Nonfinite {key} embedding in {directory.name}')
                if value.shape != (end - start, 512):
                    raise ValueError(f'Invalid {key} embedding: {value.shape}')
                result[key if condition == 'real' else 'joint_neighbor_shuffle'].append(value)
            if start == 0 or end % 50 < batch_size or end == len(nodes):
                print(f'[embed-progress] {directory.name} {condition} {end}/{len(nodes)} '
                      f'tokens={width} elapsed={time.monotonic()-started:.1f}s', flush=True)
    qa = dict(sample=directory.name, centers=len(selected), graph_cells=context,
              unique_encoded_cells=len(unique), matched_genes=len(keep), panel_genes=len(genes),
              packed_tokens=int(packed['gene_ids'].shape[1]), seconds=time.monotonic()-started)
    return {key: np.concatenate(values) for key, values in result.items()}, qa


def visible_neighbor_positive_baseline(dataset, index, item):
    """Per-compartment median of observed neighbor-token raw log counts.

    Only gene tokens actually present in the encoded neighbors are eligible.
    Center counts, query targets, omitted genes, CLS and PAD are not consulted.
    The baseline is unavailable when no positive neighbor token is visible.
    """
    slice_index = int(np.searchsorted(dataset.ends, int(index), side='right'))
    entry = dataset.slices[slice_index]
    lookup = {int(gene): column for column, gene in enumerate(entry['genes'])}
    nodes = item['node_indices']
    token_rows = np.broadcast_to(np.arange(len(nodes))[:, None], item['gene_ids'].shape)
    center = int(nodes[0])
    is_neighbor = (nodes >= 0) & (nodes != center)
    values = np.zeros_like(item['query_gene_ids'], dtype=np.float32)
    available = np.zeros_like(item['query_valid_mask'], dtype=bool)
    for compartment, raw_key in ((0, 'nuclear'), (1, 'cyto')):
        selected = (item['token_valid_mask'] & (item['compartment_ids'] == compartment)
                    & is_neighbor[:, None])
        if not selected.any():
            continue
        rows = nodes[token_rows[selected]]
        columns = np.fromiter((lookup[int(gene)] for gene in item['gene_ids'][selected]), dtype=np.int64)
        raw = np.asarray(entry[raw_key][rows, columns], dtype=np.float32)
        if not np.isfinite(raw).all() or (raw <= 0).any():
            raise ValueError('Baseline source must consist only of visible positive neighbor counts')
        median = float(np.median(raw))
        queries = item['query_compartment_ids'] == compartment
        values[queries] = median
        available[queries] = True
    return values, available


def masked_validation(model, cfg, step, requested_batches, batch_size):
    """Fixed masked validation contexts, stratified across all pretrain slides.

    At least one center per slide is retained; this can exceed the requested
    minimum batch budget. These are spatial-block held-out pretraining cells,
    separate from the LungPF donor-held-out downstream evaluation.
    """
    if requested_batches <= 0:
        return dict(enabled=False)
    dataset = V2Dataset(cfg, split='val')
    ends = np.asarray(dataset.ends, dtype=np.int64)
    starts = np.r_[0, ends[:-1]]
    chosen = [int(start + (end - start) // 2) for start, end in zip(starts, ends) if end > start]
    budget = min(len(dataset), max(len(chosen), requested_batches * batch_size))
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < budget:
        # A fixed, reproducible sample supplements per-slide representatives.
        available = np.setdiff1d(np.arange(len(dataset)), chosen, assume_unique=False)
        chosen += np.random.default_rng(173).choice(available, budget - len(chosen), replace=False).tolist()
    chosen = np.asarray(sorted(chosen), dtype=np.int64)
    modes = {0: 'gene_mask', 1: 'compartment_mask', 2: 'whole_cell_mask'}
    predictions = {'gene': 'gene_pred', 'intrinsic': 'intrinsic_pred',
                   'joint': 'joint_pred', 'neighbor': 'neighbor_pred'}
    total_losses, by_mode, cells = {}, {}, 0
    started = time.monotonic()
    for start in range(0, len(chosen), batch_size):
        items = []
        for index in chosen[start:start + batch_size]:
            item = dataset[int(index)]
            constant, available = visible_neighbor_positive_baseline(dataset, int(index), item)
            item['diagnostic_positive_baseline'] = constant
            item['diagnostic_baseline_available'] = available
            items.append(item)
        batch = collate_v2(items)
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        size = len(batch['task_mode'])
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch)
            if any(not torch.isfinite(output[key]).all() for key in predictions.values()):
                raise FloatingPointError('Nonfinite held-out masked reconstruction prediction')
            total, parts = training_losses(output, batch, cfg, max(1, step))
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite held-out masked reconstruction loss')
            for key, value in parts.items():
                total_losses[key] = total_losses.get(key, 0.) + size * value
            cells += size
            for mode, name in modes.items():
                selected_cells = batch['task_mode'].eq(mode)
                count = int(selected_cells.sum())
                if not count:
                    continue
                state = by_mode.setdefault(name, dict(cells=0, loss_sums={}, decoders={}))
                state['cells'] += count
                subset = {key: value[selected_cells] for key, value in batch.items()}
                subset_output = {key: output[key][selected_cells] for key in predictions.values()}
                _, subparts = training_losses(subset_output, subset, cfg, max(1, step))
                for key, value in subparts.items():
                    state['loss_sums'][key] = state['loss_sums'].get(key, 0.) + count * value
                for decoder, key in predictions.items():
                    pred = subset_output[key].float()
                    valid = subset['query_valid_mask'].bool()
                    if decoder == 'neighbor':
                        valid = valid & subset['neighbor_query_mask'].bool()
                    positive = valid & subset['target_nonzero'].bool()
                    nonzero = subset['target_nonzero'].float()
                    entry = state['decoders'].setdefault(decoder,
                        dict(queries=0, positive_queries=0, bce_sum=0., correct=0, positive_abs_error_sum=0.,
                             true_positive=0, false_positive=0, false_negative=0, true_negative=0,
                             positive_huber_sum=0., baseline_covered_positive_queries=0,
                             baseline_huber_sum=0., baseline_abs_error_sum=0.,
                             matched_positive_huber_sum=0., matched_positive_abs_error_sum=0.,
                             _nonzero_scores=[], _nonzero_labels=[]))
                    entry['queries'] += int(valid.sum())
                    entry['positive_queries'] += int(positive.sum())
                    entry['bce_sum'] += float(torch.nn.functional.binary_cross_entropy_with_logits(
                        pred[..., 0], nonzero, reduction='none')[valid].sum())
                    predicted_positive = pred[..., 0].ge(0)
                    entry['correct'] += int((predicted_positive.eq(nonzero.bool()) & valid).sum())
                    entry['true_positive'] += int((predicted_positive & positive).sum())
                    entry['false_positive'] += int((predicted_positive & ~nonzero.bool() & valid).sum())
                    entry['false_negative'] += int((~predicted_positive & positive).sum())
                    entry['true_negative'] += int((~predicted_positive & ~nonzero.bool() & valid).sum())
                    entry['_nonzero_scores'].append(pred[..., 0][valid].cpu().numpy())
                    entry['_nonzero_labels'].append(nonzero[valid].bool().cpu().numpy())
                    entry['positive_abs_error_sum'] += float((pred[..., 1] -
                        subset['target_log_counts']).abs()[positive].sum())
                    huber = torch.nn.functional.huber_loss(pred[..., 1], subset['target_log_counts'], reduction='none')
                    entry['positive_huber_sum'] += float(huber[positive].sum())
                    covered = positive & subset['diagnostic_baseline_available'].bool()
                    entry['baseline_covered_positive_queries'] += int(covered.sum())
                    constant = subset['diagnostic_positive_baseline']
                    entry['baseline_huber_sum'] += float(torch.nn.functional.huber_loss(
                        constant, subset['target_log_counts'], reduction='none')[covered].sum())
                    entry['baseline_abs_error_sum'] += float((constant -
                        subset['target_log_counts']).abs()[covered].sum())
                    entry['matched_positive_huber_sum'] += float(huber[covered].sum())
                    entry['matched_positive_abs_error_sum'] += float((pred[..., 1] -
                        subset['target_log_counts']).abs()[covered].sum())
        if start == 0 or start + batch_size >= len(chosen):
            print(f'[masked-validation] {min(start+batch_size,len(chosen))}/{len(chosen)} '
                  f'elapsed={time.monotonic()-started:.1f}s', flush=True)
    for state in by_mode.values():
        state['loss_mean'] = {key: value/state['cells'] for key, value in state.pop('loss_sums').items()}
        for entry in state['decoders'].values():
            queries, positive = entry['queries'], entry['positive_queries']
            entry['nonzero_bce'] = entry.pop('bce_sum')/queries if queries else None
            entry['nonzero_accuracy'] = entry.pop('correct')/queries if queries else None
            entry['positive_log_count_mae'] = entry.pop('positive_abs_error_sum')/positive if positive else None
            tp, fp, fn, tn = [entry[key] for key in ('true_positive', 'false_positive', 'false_negative', 'true_negative')]
            if tp + fp + fn + tn != queries or tp + fn != positive:
                raise AssertionError('Nonzero diagnostic confusion counts are inconsistent')
            entry['nonzero_precision'] = tp/(tp+fp) if tp+fp else None
            entry['nonzero_recall'] = tp/(tp+fn) if tp+fn else None
            entry['nonzero_specificity'] = tn/(tn+fp) if tn+fp else None
            entry['nonzero_balanced_accuracy'] = (.5*(tp/(tp+fn)+tn/(tn+fp))
                                                  if tp+fn and tn+fp else None)
            entry['predicted_positive_fraction'] = (tp+fp)/queries if queries else None
            entry['always_zero_accuracy'] = (queries-positive)/queries if queries else None
            entry['always_zero_balanced_accuracy'] = .5 if 0 < positive < queries else None
            entry['always_zero_positive_recall'] = 0. if positive else None
            scores = np.concatenate(entry.pop('_nonzero_scores'))
            labels = np.concatenate(entry.pop('_nonzero_labels'))
            if len(scores) != queries or int(labels.sum()) != positive:
                raise AssertionError('Nonzero ranking diagnostics have inconsistent query counts')
            # Rank by unrounded logits; compute probability means in float64.
            probabilities = 1./(1.+np.exp(-np.clip(scores.astype(np.float64), -709., 709.)))
            has_both_classes = 0 < positive < queries
            entry['nonzero_average_precision'] = float(average_precision_score(labels, scores)) if has_both_classes else None
            entry['nonzero_auroc'] = float(roc_auc_score(labels, scores)) if has_both_classes else None
            entry['positive_probability_mean'] = float(probabilities[labels].mean()) if positive else None
            entry['negative_probability_mean'] = float(probabilities[~labels].mean()) if positive < queries else None
            entry['positive_prevalence'] = positive/queries if queries else None
            entry['positive_huber'] = entry.pop('positive_huber_sum')/positive if positive else None
            covered = entry['baseline_covered_positive_queries']
            entry['context_median_baseline_coverage'] = covered/positive if positive else None
            entry['context_median_positive_huber'] = entry.pop('baseline_huber_sum')/covered if covered else None
            entry['context_median_positive_log_count_mae'] = entry.pop('baseline_abs_error_sum')/covered if covered else None
            entry['positive_huber_on_baseline_covered_queries'] = entry.pop('matched_positive_huber_sum')/covered if covered else None
            entry['positive_log_count_mae_on_baseline_covered_queries'] = entry.pop('matched_positive_abs_error_sum')/covered if covered else None
    return dict(enabled=True, centers=cells, total_validation_centers=len(dataset),
        slices_covered=len([1 for start, end in zip(starts, ends) if end > start]),
        requested_minimum_batches=requested_batches, effective_batches=(cells+batch_size-1)//batch_size,
        sample_indices_sha256=hashlib.sha256(chosen.tobytes()).hexdigest(),
        loss_mean={key: value/cells for key, value in total_losses.items()}, by_task_mode=by_mode,
        protocol='Fixed epoch-0 masks; at least one spatial-block held-out center per pretraining slide; '
                 'loss batch means match training reduction; decoder metrics are query-weighted',
        diagnostic_schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        nonzero_diagnostic_protocol='Positive threshold sigmoid(logit)>=0.5. Undefined precision without positive predictions '
                 'and balanced accuracy/AP/AUROC without both true classes are null; always-zero baseline uses the same queries. '
                 'AP/AUROC rank raw logits and do not require a classification threshold. Positive prevalence is the '
                 'random-ranking AP reference; positive/negative probability means diagnose score calibration.',
        positive_constant_baseline='Per-patch, per-compartment median raw log1p count from actually visible neighbor '
                 'gene tokens only, excluding center/CLS/PAD. No validation target is used to fit the constant. '
                 'An empty compartment has no baseline; report coverage and compare model on the same covered queries. '
                 'The baseline uses visible raw counts, while the encoder uses their quantile bins. Huber delta=1.',
        seconds=time.monotonic()-started)


def probe_embedding(obs, features, embedding, output):
    results = []
    for task, column in [('cell_type', 'final_CT'), ('region', 'region_label')]:
        labels = obs[column]
        usable = labels.notna() & ~labels.astype(str).isin(['', 'nan', 'None'])
        train = usable & obs['split'].eq('train')
        val = usable & obs['split'].eq('val')
        classes = np.unique(labels[train].astype(str))
        if len(classes) < 2 or not val.any():
            raise ValueError(f'Insufficient labels for {task}')
        lookup = {name: index for index, name in enumerate(classes)}
        scaler = StandardScaler().fit(features[train])
        tx = torch.as_tensor(scaler.transform(features[train]), device='cuda', dtype=torch.float32)
        vx = torch.as_tensor(scaler.transform(features[val]), device='cuda', dtype=torch.float32)
        y = torch.tensor([lookup[name] for name in labels[train].astype(str)], device='cuda')
        true = labels[val].astype(str).to_numpy()
        donors = obs.loc[val, 'patient'].to_numpy()
        grid, best_prediction, best_score = [], None, -1.
        for regularization in C_GRID:
            weights, bias, info = fit_linear(tx, y, regularization)
            prediction = predict(vx, weights, bias, classes)
            f1 = float(f1_score(true, prediction, labels=np.unique(true), average='macro', zero_division=0))
            donor_scores = {str(donor): float(f1_score(true[donors == donor], prediction[donors == donor],
                labels=np.unique(true[donors == donor]), average='macro', zero_division=0))
                for donor in np.unique(donors)}
            grid.append(dict(C=regularization, val_macro_f1=f1,
                val_balanced_accuracy=float(balanced_accuracy_score(true, prediction)),
                per_donor_macro_f1=donor_scores, solver=info))
            if f1 > best_score:
                best_score, best_prediction = f1, prediction.copy()
        best = max(grid, key=lambda item: item['val_macro_f1'])
        prediction_frame = obs.loc[val, ['full_cell_id', 'patient']].copy()
        prediction_frame['label'] = true
        prediction_frame['prediction'] = best_prediction
        target = output / f'{embedding}_{task}_val_predictions.parquet'
        temporary = target.with_name(target.name + '.partial')
        prediction_frame.to_parquet(temporary, index=False)
        temporary.replace(target)
        row = dict(embedding=embedding, task=task, train_cells=int(train.sum()), val_cells=int(val.sum()),
            unseen_val_classes=sorted(set(true) - set(classes)), best=best, grid=grid)
        results.append(row)
        print(f'[probe] {embedding} {task} best={json.dumps(best)}', flush=True)
        del tx, vx, y, weights, bias
        torch.cuda.empty_cache()
    return results


def write_report(output, report):
    lines = [f'# V2 检查点 {report["step"]}：固定供体验证', '',
        f'预训练骨干：**{report["backbone_name"]}**；来源与权重哈希已写入 metrics.json。'
        '这里记录的是本检查点实际声明并核验的来源，不由实验目录名推断。', '',
        f'固定 {report["cells"]:,} 个中心细胞；训练 {len(report["train_donors"])} 个供体，'
        f'验证 {len(report["val_donors"])} 个供体；未读取测试切片。', '',
        f'每个中心使用自己的 {report["context_cells"]} 细胞 KNN patch；'
        '每层通信邻居在该 patch 内选取，属于截断邻域图。基因输入不做 mask。', '',
        '| 表征 | 任务 | 验证 macro-F1 | 验证 balanced accuracy | 最优 C |',
        '|---|---|---:|---:|---:|']
    for result in report['results']:
        best = result['best']
        lines.append(f'| {result["embedding"]} | {result["task"]} | {best["val_macro_f1"]:.6f} | '
                     f'{best["val_balanced_accuracy"]:.6f} | {best["C"]:g} |')
    lines += ['', '所有表征使用相同的训练供体标准化、class-balanced 线性 probe 和 C 网格；'
              'C 仅在验证集选择。macro-F1 包含验证中出现、训练中未见的类别。', '',
              '这些结果用于训练诊断和方案选择，不是最终独立测试；不要求每个检查点单调改善。'
              '供体级分数及求解器梯度收敛证据保存在 metrics.json。']
    if report['neighbor_shuffle']:
        lines += ['', '邻域打乱诊断保持中心表达、位置和图不变，将同切片细胞表达随机分配给邻居；'
                  '排除中心表达重复作为邻居。它检验真实局部表达组成的贡献，不能单独证明因果互作。']
    masked = report.get('masked_validation', {})
    if masked.get('enabled'):
        lines += ['', '## 固定遮盖重建验证', '',
            f'覆盖 {masked["slices_covered"]} 张预训练切片、{masked["centers"]} 个空间留出中心；'
            '使用固定 mask，不读取 LungPF 测试数据。', '',
            '| Loss | Mean |', '|---|---:|']
        lines.extend(f'| {key} | {value:.6f} |' for key, value in masked['loss_mean'].items())
        lines += ['', '非零预测 precision/recall、balanced accuracy、AP/AUROC、正负类预测概率均值、'
                  '预测正类比例和“全部预测零”基线，'
                  '以及正值 Huber/MAE 均按遮盖模式和解码器保存在 metrics.json。'
                  'query 经过分层采样，准确率不代表原始 panel 的真实非零比例；总 loss 的邻域权重随训练步数变化。', '',
                  '正值常数基线仅使用实际可见邻居 token 的同区室原始 log1p(count) 中位数，排除中心、CLS 与 PAD；'
                  '不以中心验证目标拟合常数。无可见正值的区室不计算基线，报告覆盖率，并在相同覆盖子集比较模型。']
    temporary = output / 'REPORT.zh.md.partial'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'REPORT.zh.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cells-per-donor', type=int, default=400)
    parser.add_argument('--neighbor-shuffle', action='store_true')
    parser.add_argument('--mask-validation-batches', type=int, default=16,
                        help='Minimum masked-validation batches; raised to cover every pretraining slide. 0 disables.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.cells_per_donor < 1:
        parser.error('batch size and cells per donor must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.monotonic()
    model, cfg, vocab, step, provenance = load_model(args.checkpoint, args.expected_step)
    checkpoint_stat = args.checkpoint.stat()
    checkpoint_digest = file_sha256(args.checkpoint)
    vocab_digest = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()
    existing_metrics = args.output / 'metrics.json'
    if existing_metrics.exists() and provenance['backbone_kind'] == 'scgpt':
        existing = json.loads(existing_metrics.read_text())
        if existing.get('backbone_kind') != 'scgpt':
            raise ValueError('Refusing to mix ordinary scGPT evaluation with legacy/spatial results; '
                             'choose the new experiment output directory')
    prepared = ROOT / 'data/benchmarking_data/lungpf/prepared'
    protocol = json.loads((prepared / 'protocol.json').read_text())
    frames, matrices, coverage = [], {}, []
    for sample in protocol['samples']:
        # The prepared LungPF protocol names samples by donor plus LA/MA.
        # Resolve the split before opening the observation table or expression.
        matches = [donor for donor in protocol['donor_split'] if sample.startswith(donor)]
        if len(matches) != 1:
            raise ValueError(f'Cannot uniquely resolve sample donor from protocol: {sample}')
        donor = matches[0]
        split = protocol['donor_split'][donor]
        if split == 'test':
            continue
        if split not in ('train', 'val'):
            raise ValueError(f'Unexpected protocol split: {split}')
        directory = prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        if not obs.patient.eq(donor).all() or ('split' in obs and not obs['split'].eq(split).all()):
            raise ValueError(f'Observation/protocol donor split mismatch: {sample}')
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(args.cells_per_donor, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy()
        chosen['split'] = split
        cache_dir = args.output / 'sample_embeddings'
        cache_dir.mkdir(exist_ok=True)
        cache = cache_dir / f'{sample}.npz'
        metadata = cache_dir / f'{sample}.json'
        signature = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_size=checkpoint_stat.st_size,
            checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns, step=step, extraction_version=2,
            checkpoint_sha256=checkpoint_digest, config_sha256=provenance['config_sha256'],
            backbone_kind=provenance['backbone_kind'], vocab_mapping_sha256=vocab_digest,
            pretrained_weights_sha256=provenance['loaded_pretrained_backbone']['weights_sha256'],
            context_cells=int(cfg['data'].get('context_cells', 64)),
            context_tokens_per_compartment=int(cfg['data'].get('context_tokens_per_compartment', 256)),
            neighbor_shuffle=args.neighbor_shuffle,
            selected_ids_sha256=hashlib.sha256('\n'.join(chosen.full_cell_id.astype(str)).encode()).hexdigest())
        cached = json.loads(metadata.read_text()) if metadata.exists() else {}
        if cache.exists() and cached.get('signature') == signature:
            with np.load(cache) as archive:
                features = {key: archive[key] for key in archive.files}
            qa = cached['qa']
        else:
            features, qa = extract(model, cfg, vocab, directory, obs, selected, args.batch_size, args.neighbor_shuffle)
            temporary = cache.with_name(cache.stem + '.partial.npz')
            np.savez(temporary, **features)
            temporary.replace(cache)
            atomic_json(metadata, dict(signature=signature, qa=qa))
        required = set(EMBEDDINGS) | ({'joint_neighbor_shuffle'} if args.neighbor_shuffle else set())
        if set(features) != required:
            raise ValueError(f'Cached representation mismatch: {sample}')
        for key, value in features.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f'Nonfinite cached {key} embedding in {sample}')
            if value.shape != (len(selected), 512):
                raise ValueError(f'Invalid cached embedding in {sample}: {key}')
            matrices.setdefault(key, []).append(value)
        frames.append(chosen)
        coverage.append(qa)
        print(f'[embed] step={step} {sample} {split} n={len(selected)} {qa}', flush=True)
    obs = pd.concat(frames, ignore_index=True)
    if not obs.full_cell_id.is_unique or obs['split'].eq('test').any():
        raise ValueError('Duplicate IDs or held-out test cells in evaluation cohort')
    cohort = hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest()
    if args.cells_per_donor == 400 and cohort != CANONICAL_COHORT:
        raise ValueError(f'Evaluation cohort differs from existing 5,200-cell cohort: {cohort}')
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
        results.extend(probe_embedding(obs, features, key, args.output))
    report = dict(model='SubcellularV2', evaluation_version=2, diagnostic_schema_version=DIAGNOSTIC_SCHEMA_VERSION,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_digest, step=step,
        backbone_kind=provenance['backbone_kind'],
        backbone_name='scGPT (ordinary single-cell pretrained model)' if provenance['backbone_kind'] == 'scgpt'
                      else 'scGPT-spatial V1 (historical comparison)',
        backbone_provenance=provenance, vocab_mapping_sha256=vocab_digest,
        seed=42, cells_per_donor=args.cells_per_donor, cells=len(obs), cohort_sha256=cohort,
        train_donors=sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique()),
        val_donors=sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique()),
        context_cells=int(cfg['data'].get('context_cells', 64)),
        graph='Per-center KNN patch; all adapter neighborhoods are truncated to this patch',
        input='Joint nuclear/cytoplasmic 51-bin visible context; no expression masking; PAD omitted',
        neighbor_shuffle=args.neighbor_shuffle, test_donors_used=False,
        config_sha256=provenance['config_sha256'],
        masked_validation=reconstruction, diagnostics=diagnostics, coverage=coverage,
        results=results, seconds=time.monotonic()-started)
    write_report(args.output, report)
    # Completion marker is last: interrupted/failed probes must be retried.
    atomic_json(args.output / 'metrics.json', report)
    print(f'[done] step={step} seconds={report["seconds"]:.1f} output={args.output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--output' in sys.argv:
            folder = Path(sys.argv[sys.argv.index('--output') + 1])
            folder.mkdir(parents=True, exist_ok=True)
            kind = ('nonfinite' if isinstance(error, FloatingPointError) else
                    'out_of_memory' if isinstance(error, torch.cuda.OutOfMemoryError) else 'runtime')
            atomic_json(folder / 'diagnostic_failure.json', dict(kind=kind, error_type=type(error).__name__,
                detail=str(error)[:3000], timestamp=time.time()))
        raise
