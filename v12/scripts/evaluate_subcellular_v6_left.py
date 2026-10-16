"""Paired frozen scGPT CLS and V6-left A/B on the fixed donor cohort.

Seven readouts include nuclear states before/after spatial feedback. Masked
reconstruction uses the sole 51-bin cross entropy, with probability diagnostics.
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
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from evaluate_subcellular_v2 import (
    CANONICAL_COHORT, as_dict, atomic_array, atomic_json, file_sha256,
    pack_tokens, probe_embedding,
)
from spatial_pretrain.data.subcellular_v6 import (
    load_lr_prior, compute_lr_features_v6,
)
from spatial_pretrain.data.subcellular_v5 import collate_whole_cell_v5
from spatial_pretrain.data.subcellular_v7 import (
    encode_compartment_context_v7,
)
from spatial_pretrain.data.subcellular_v6_left import (
    V6LeftDataset, collate_v6_left, encode_whole_cell_v6_left,
)
from spatial_pretrain.models.subcellular_v6 import SubcellularV6
from spatial_pretrain.training.subcellular_v4 import losses as training_losses
from spatial_pretrain.training.subcellular_v6_optimizer import optimizer_manifest
from train_subcellular_v6_left import SOURCE_FILES, validate_config

EVALUATION_VERSION = 61
MODEL_VERSION = 61
ARCHITECTURE = 'subcellular_lr_routes_v6_deterministic_left_51'
ACTIVE_OUTPUT = None
READOUTS = ('base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
            'nucleus_local', 'cytoplasm_local')
SHUFFLE_READOUTS = ('joint', 'nucleus', 'cytoplasm')
BINNING_PROTOCOL = ('deterministic-left per-cell 51-bin quantiles for visible N/C and whole-cell inputs; '
                    'deterministic-left full-panel targets')


def validate_project_sources(recorded):
    if not isinstance(recorded, dict) or set(recorded) != set(SOURCE_FILES):
        raise ValueError('V6 checkpoint must record all training/model/data dependency hashes')
    current = {name: file_sha256(ROOT / name) for name in SOURCE_FILES}
    differing = [name for name in SOURCE_FILES if recorded[name] != current[name]]
    if differing:
        raise ValueError(f'V6 evaluation source differs from checkpoint: {differing}')
    return current


def validate_checkpoint_contract(saved):
    """Reject older architectures before constructing or loading any model."""
    if saved.get('model_version') != MODEL_VERSION or saved.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected V6-left frozen-backbone deterministic checkpoint')
    if not isinstance(saved.get('arch_config'), dict) or not saved['arch_config']:
        raise ValueError('V6 checkpoint must record a nonempty arch_config')


def load_model(path, expected_step=None, config_path=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint_contract(saved)
    step = int(saved['step'])
    if expected_step is not None and step != expected_step:
        raise ValueError(f'Checkpoint step mismatch: {step} != {expected_step}')
    cfg = as_dict(saved['config'])
    validate_config(cfg)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if saved.get('config_sha256') != fingerprint:
        raise ValueError('V6 checkpoint configuration fingerprint is missing or invalid')
    sources = validate_project_sources(saved.get('source_sha256'))
    if config_path is not None:
        requested = yaml.safe_load(config_path.read_text())
        if hashlib.sha256(json.dumps(requested, sort_keys=True).encode()).hexdigest() != fingerprint:
            raise ValueError('--config does not match the checkpoint configuration')
    options = dict(cfg['model'])
    if options.get('backbone_kind') != 'scgpt':
        raise ValueError('V6 evaluation requires explicit ordinary scGPT configuration')
    supported = set(inspect.signature(SubcellularV6).parameters) - {'asset_dir', 'backbone'}
    if set(options) - supported:
        raise ValueError(f'Unsupported V6 model options: {sorted(set(options) - supported)}')
    options['checkpointing'] = False
    model = SubcellularV6(asset_dir=cfg['assets_dir'], **options)
    if as_dict(saved['arch_config']) != as_dict(model.arch_config):
        raise ValueError('V6 checkpoint arch_config does not match the constructed model')
    recorded, loaded = as_dict(saved.get('backbone_provenance', {})), as_dict(model.backbone.provenance)
    for key in ('backbone_kind', 'weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'):
        if not recorded.get(key) or recorded[key] != loaded.get(key):
            raise ValueError(f'V6 checkpoint/pretrained-source provenance mismatch: {key}')
    if recorded['backbone_kind'] != 'scgpt':
        raise ValueError('scGPT-spatial checkpoints cannot be relabeled as V6')
    # The paired reference remains ordinary scGPT at EVERY checkpoint. Merely
    # retaining its initialization provenance would not establish this.
    for key, value in model.backbone.state_dict().items():
        archived = saved['model'].get(f'backbone.{key}')
        if archived is None or not torch.equal(value.cpu(), archived.cpu()):
            raise ValueError(f'Frozen V6 backbone differs from ordinary scGPT: {key}')
    # A V2/V3 state cannot silently populate the slot-based feedback model.
    model.load_state_dict(saved['model'], strict=True)
    optimizer_description = optimizer_manifest(model, cfg['training'])
    if saved.get('optimizer_manifest') != optimizer_description:
        raise ValueError('Checkpoint Muon/AdamW parameter grouping differs from the current model/config')
    if not isinstance(model.vocab, dict):
        raise TypeError('V6 requires the verified ordinary scGPT vocabulary mapping')
    prior = load_lr_prior(cfg, model.vocab)
    if saved.get('lr_prior') != prior['metadata']:
        raise ValueError('Checkpoint LR prior differs from the verified current asset')
    model.evaluation_lr_prior = prior
    provenance = dict(backbone_kind='scgpt', config_sha256=fingerprint,
        model_version=MODEL_VERSION, architecture=ARCHITECTURE, arch_config=dict(model.arch_config),
        checkpoint_source_sha256=sources, checkpoint_sources_verified=True,
        checkpoint_backbone_provenance=recorded, loaded_pretrained_backbone=loaded,
        checkpoint_provenance_verified=True, ordinary_scgpt_initial_weights_verified=step == 0,
        frozen_backbone_weights_verified=True,
        lr_prior=prior['metadata'],
        optimizer_manifest=optimizer_description,
        evaluation_model_source_sha256=file_sha256(ROOT / 'src/spatial_pretrain/models/subcellular_v6.py'),
        evaluation_data_source_sha256=file_sha256(ROOT / 'src/spatial_pretrain/data/subcellular_v6_left.py'),
        evaluation_loss_source_sha256=file_sha256(ROOT / 'src/spatial_pretrain/training/subcellular_v4.py'))
    del saved
    return model.eval().requires_grad_(False).cuda(), cfg, model.vocab, step, provenance


def expected_dimensions(neighbor_shuffle=False, compartment_dim=512, backbone_dim=512):
    if int(compartment_dim) != compartment_dim or compartment_dim <= 0:
        raise ValueError('compartment_dim must be a positive integer')
    if int(backbone_dim) != backbone_dim or backbone_dim <= 0:
        raise ValueError('backbone_dim must be a positive integer')
    dimensions = {name: int(backbone_dim if name in ('base', 'intrinsic', 'joint') else compartment_dim)
                  for name in READOUTS}
    if neighbor_shuffle:
        dimensions.update({f'{name}_neighbor_shuffle': dimensions[name] for name in SHUFFLE_READOUTS})
    return dimensions


def shuffle_expression_nodes(nodes, visibility, seed):
    """Shuffle expression-to-position within one slide and visibility stratum.

    The caller supplies a single measured panel. A shared stratified cyclic
    permutation moves whole nuclear/cytoplasmic expression pairs; patch-specific
    repair reserves its own center and keeps all expression source IDs unique.
    This is only for unmasked downstream extraction, never masked validation.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    visibility = np.asarray(visibility, dtype=np.bool_)
    if nodes.ndim != 2 or nodes.shape[1] < 2 or visibility.ndim != 2 or visibility.shape[1] != 2:
        raise ValueError('Expected patches [centers, nodes] and visibility [cells, 2]')
    if (nodes < 0).any() or (nodes >= len(visibility)).any():
        raise ValueError('Shuffle cell IDs must lie in the same slide/panel')
    if any(len(np.unique(row)) != len(row) for row in nodes):
        raise ValueError('Shuffle input patches must have unique cell IDs')
    strata = visibility[:, 0].astype(np.int8) + 2 * visibility[:, 1].astype(np.int8)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(visibility), dtype=np.int64)
    for group in np.unique(strata):
        members = rng.permutation(np.flatnonzero(strata == group))
        if len(members) > 1:
            permutation[members] = np.roll(members, 1)
    shuffled = permutation[nodes].copy()
    for row, center in enumerate(nodes[:, 0]):
        # Reserve center expression for position zero. If another position was
        # assigned it, transfer the now-displaced permutation[center] there.
        shuffled[row, shuffled[row] == center] = permutation[center]
        shuffled[row, 0] = center
    if (not np.array_equal(visibility[nodes], visibility[shuffled])
            or any(len(np.unique(row)) != len(row) for row in shuffled)
            or not np.array_equal(shuffled[:, 0], nodes[:, 0])):
        raise AssertionError('Neighbor shuffle violated center, visibility or unique-expression invariants')
    return shuffled


def measured_visibility(nuclear, cytoplasm, keep):
    """Bounded-memory visibility after vocabulary filtering, before masking."""
    result = np.empty((len(nuclear), 2), dtype=np.bool_)
    for start in range(0, len(nuclear), 2048):
        rows = np.arange(start, min(start + 2048, len(nuclear)))
        result[start:start + len(rows), 0] = (nuclear[np.ix_(rows, keep)] > 0).any(axis=-1)
        result[start:start + len(rows), 1] = (cytoplasm[np.ix_(rows, keep)] > 0).any(axis=-1)
    return result


def extract(model, cfg, vocab, directory, obs, selected, batch_size, neighbor_shuffle, shuffle_seed=815):
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not obs.full_cell_id.is_unique:
        raise ValueError(f'Invalid coordinates or duplicate cell IDs: {directory.name}')
    context = min(int(cfg['data'].get('context_cells', 64)), len(obs))
    if context < 2:
        raise ValueError('V6 neighborhood evaluation requires at least two cells')
    nearest = NearestNeighbors(n_neighbors=context).fit(coords).kneighbors(coords[selected], return_distance=False)
    nodes = np.stack([np.r_[center, row[row != center][:context - 1]] for center, row in zip(selected, nearest)])
    if any(len(np.unique(row)) != context for row in nodes):
        raise ValueError('KNN patch contains duplicate row IDs')
    genes = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([index for index, gene in enumerate(genes) if gene in vocab], dtype=np.int64)
    if not len(keep):
        raise ValueError(f'No panel genes match the ordinary scGPT vocabulary: {directory.name}')
    global_ids = np.array([vocab[gene] for gene in genes[keep]], dtype=np.int64)
    nuclear = np.load(directory / 'nuclear.npy', mmap_mode='r', allow_pickle=False)
    cytoplasm = np.load(directory / 'cytoplasmic.npy', mmap_mode='r', allow_pickle=False)
    if nuclear.shape != cytoplasm.shape or nuclear.shape != (len(obs), len(genes)):
        raise ValueError(f'Expression/observation/panel shape mismatch: {directory.name}')
    expressions = {'real': nodes}
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    shuffle_fraction = None
    if neighbor_shuffle:
        visibility = measured_visibility(nuclear, cytoplasm, keep)
        shuffled = shuffle_expression_nodes(nodes, visibility,
            np.random.SeedSequence([42, slice_seed, shuffle_seed]))
        shuffle_fraction = float(np.mean(shuffled[:, 1:] != nodes[:, 1:]))
        if shuffle_fraction == 0.:
            raise ValueError(f'No neighbors can be shuffled within visibility strata: {directory.name}')
        expressions['neighbor_shuffle'] = shuffled
    unique = np.unique(np.concatenate(list(expressions.values())))
    encoded = encode_compartment_context_v7(
        np.log1p(nuclear[np.ix_(unique, keep)].astype(np.float32)),
        np.log1p(cytoplasm[np.ix_(unique, keep)].astype(np.float32)), global_ids,
        node_indices=unique, seed=42, epoch=0, slice_seed=slice_seed,
        context_tokens_per_compartment=int(cfg['data'].get('context_tokens_per_compartment', 256)),
        pad_token_id=vocab['<pad>'], cls_token_id=vocab['<cls>'])
    packed = pack_tokens(encoded)
    del encoded
    dimensions = expected_dimensions(neighbor_shuffle, model.dim, model.backbone.dim)
    whole = [encode_whole_cell_v6_left(
        nuclear[int(center), keep], cytoplasm[int(center), keep], global_ids,
        pad_token_id=vocab['<pad>'], cls_token_id=vocab['<cls>'],
        max_tokens=int(cfg['data']['whole_cell_max_tokens']),
        seed=np.random.SeedSequence([42, slice_seed, int(center), 355])) for center in selected]
    result = {name: [] for name in dimensions}
    prior = model.evaluation_lr_prior
    lr_summaries = []
    started = time.monotonic()
    for condition, expression in expressions.items():
        mapped = np.searchsorted(unique, expression)
        references = ({key: np.concatenate(result[key], axis=0)
                       for key in ('base', 'nucleus_local', 'cytoplasm_local')}
                      if condition != 'real' else {})
        for start in range(0, len(nodes), batch_size):
            end = min(start + batch_size, len(nodes))
            index = mapped[start:end]
            width = int(packed['token_valid_mask'][index].sum(axis=-1).max())
            batch = {key: torch.as_tensor(values[index, :width], device='cuda') for key, values in packed.items()}
            batch['node_indices'] = torch.as_tensor(nodes[start:end], device='cuda')
            batch['coords'] = torch.as_tensor(coords[nodes[start:end]], device='cuda')
            batch.update({key: value.cuda() for key, value in
                          collate_whole_cell_v5(whole[start:end], pad_token_id=model.pad_id).items()})
            # Rebuild expression-derived evidence after neighbor shuffling;
            # physical node IDs and coordinates still define the same graph.
            patch_lr = [compute_lr_features_v6(
                cytoplasm[np.ix_(row, keep)], global_ids, node_indices=physical,
                lr_pairs=prior['pairs'], lr_weights=prior['weights'])
                for row, physical in zip(expression[start:end], nodes[start:end])]
            for key in ('lr_bias', 'lr_coverage'):
                values = np.stack([item[key] for item in patch_lr])
                batch[key] = torch.as_tensor(values, device='cuda')
            lr_summaries.append(dict(condition=condition, centers=end-start,
                bias_mean=float(batch['lr_bias'].mean()),
                coverage_mean=float(batch['lr_coverage'].mean())))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch, return_embeddings=True)['embeddings']
            for key in (READOUTS if condition == 'real' else SHUFFLE_READOUTS):
                value = output[key].float().cpu().numpy()
                if value.shape != (end - start, dimensions[key]) or not np.isfinite(value).all():
                    raise FloatingPointError(f'Invalid {key} output in {directory.name}: {value.shape}')
                result[key if condition == 'real' else f'{key}_neighbor_shuffle'].append(value)
            if condition != 'real':
                # The center and pre-spatial computation must be unaffected.
                # Check this directly, not only graph/permutation bookkeeping.
                for key in ('base', 'nucleus_local', 'cytoplasm_local'):
                    reference = references[key][start:end]
                    value = output[key].float().cpu().numpy()
                    if not np.allclose(value, reference, rtol=5e-3, atol=1e-3):
                        raise ValueError(f'Shuffle unexpectedly changed pre-spatial center readout: {key}')
            if start == 0 or end == len(nodes) or end % 50 < batch_size:
                print(f'[embed-progress-v6] {directory.name} {condition} {end}/{len(nodes)} '
                      f'tokens={width} elapsed={time.monotonic()-started:.1f}s', flush=True)
    qa = dict(sample=directory.name, centers=len(selected), graph_cells=context,
              unique_encoded_cells=len(unique), matched_genes=len(keep), panel_genes=len(genes),
              packed_tokens=int(packed['gene_ids'].shape[1]), seconds=time.monotonic()-started,
              neighbor_expression_changed_fraction=shuffle_fraction,
              shuffle_visibility_matched=bool(neighbor_shuffle), shuffle_seed=shuffle_seed if neighbor_shuffle else None)
    qa['lr_batches'] = lr_summaries
    return {key: np.concatenate(values) for key, values in result.items()}, qa


def decoder_metrics(prediction, batch, valid):
    """Probability diagnostics from the single 51-bin head; not extra losses."""
    valid = valid.bool()
    target = batch['target_bins']
    if prediction.shape != (*target.shape, 51) or valid.shape != target.shape:
        raise ValueError('V6 decoder must produce [batch, queries, 51] bin logits')
    if bool((((target < 0) | (target > 50) | ~torch.isfinite(target) | (target != target.round())) & valid).any()):
        raise ValueError('Masked-validation targets must be integer bins 0..50 on valid queries')
    selected = prediction.detach().float()[valid].cpu().double()
    if not torch.isfinite(selected).all():
        raise FloatingPointError('Nonfinite V6 decoder logits on valid queries')
    bins = target[valid].detach().cpu().numpy().astype(np.int64)
    log_prob = selected.log_softmax(-1)
    # -expm1(log p0) evaluates 1-p0 without cancellation near p0 == 1.
    probabilities = (-torch.expm1(log_prob[:, 0])).numpy()
    # Rank with nonzero log-odds, not rounded probabilities. Even float64
    # probabilities saturate at zero/one for distinct confident predictions.
    # This score is mathematically monotone in 1-p0 without those false ties.
    ranking_scores = (selected[:, 1:].logsumexp(-1) - selected[:, 0]).numpy()
    labels = bins > 0
    predicted = probabilities >= .5
    count, positive = len(bins), int(labels.sum())
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    fn = int((~predicted & labels).sum())
    tn = int((~predicted & ~labels).sum())
    both = 0 < positive < count
    # Conditioning is a softmax of positive-bin logits, stable even when total
    # positive probability is tiny. Do not use argmax or multiply by p(nonzero).
    positive_distribution = selected[:, 1:].softmax(-1)
    ranks = torch.arange(50, dtype=torch.float64) / 49.
    rank = (positive_distribution * ranks).sum(-1).numpy()[labels]
    relative_target = (bins[labels].astype(np.float64) - 1.) / 49.
    error = rank - relative_target
    result = dict(queries=count, positive_queries=positive, true_positive=tp, false_positive=fp,
        false_negative=fn, true_negative=tn,
        masked_bin_ce_query_mean=float(-log_prob[torch.arange(count), torch.as_tensor(bins)].mean()) if count else None,
        bin_accuracy=float((selected.argmax(-1).numpy() == bins).mean()) if count else None,
        nonzero_accuracy=(tp + tn) / count if count else None,
        nonzero_precision=tp / (tp + fp) if tp + fp else None,
        nonzero_recall=tp / positive if positive else None,
        nonzero_specificity=tn / (tn + fp) if tn + fp else None,
        nonzero_balanced_accuracy=.5 * (tp / positive + tn / (count - positive)) if both else None,
        nonzero_average_precision=float(average_precision_score(labels, ranking_scores)) if both else None,
        nonzero_auroc=float(roc_auc_score(labels, ranking_scores)) if both else None,
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
        raise FloatingPointError('Nonfinite V6 masked-validation loss')
    result = dict(loss_mean=parts, decoders={})
    result['decoders']['joint'] = decoder_metrics(outputs['joint_logits'], batch, batch['query_valid_mask'])
    result['by_query_compartment'] = {
        name: decoder_metrics(outputs['joint_logits'], batch,
            batch['query_valid_mask'].bool() & batch['query_compartment_ids'].eq(index))
        for index, name in ((0, 'nucleus'), (1, 'cytoplasm'))}
    return result


def masked_validation(model, cfg, step, requested_batches, batch_size):
    if requested_batches <= 0:
        return dict(enabled=False)
    dataset = V6LeftDataset(cfg, split='val')
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
    gathered = {key: [] for key in fields}
    predictions = {'joint_logits': []}
    target_hash = hashlib.sha256()
    started = time.monotonic()
    for start in range(0, len(chosen), batch_size):
        batch = collate_v6_left([dataset[int(index)] for index in chosen[start:start + batch_size]],
                           pad_token_id=dataset.pad_id)
        device_batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(device_batch)
        for key in predictions:
            if not torch.isfinite(output[key]).all():
                raise FloatingPointError(f'Nonfinite V6 masked prediction: {key}')
            predictions[key].append(output[key].detach().float().cpu())
        for key in fields:
            gathered[key].append(batch[key])
        del device_batch, output
        print(f'[masked-v6] {min(start+batch_size,len(chosen))}/{len(chosen)}', flush=True)
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
        limitation='Full-panel ranks are reconstruction targets only; all input bins and LR evidence are recomputed '
                   'from masked-visible counts. LR uses the full visible panel before gene-token truncation. '
                   'No absolute-count target or auxiliary LR loss is used.',
        positive_rank_target='(positive_bin - 1) / 49; predicted conditional expectation over bins 1..50; '
                             'constant midpoint baseline 0.5 is not fitted to validation targets',
        loss_definition='Single masked-bin cross entropy, reduced by query, valid compartment, then valid center',
        neighbor_shuffle_applied=False,
        seconds=time.monotonic() - started)
    return result


def write_report(output, report):
    dimensions = report['embedding_dimensions']
    lines = [f'# V6-left checkpoint {report["step"]}：原始scGPT CLS与亚细胞残差配对验证', '',
             'base是本次同输入、同精度的原始scGPT整细胞CLS+L2基线；joint=base+亚细胞残差。'
             '骨干每个checkpoint逐张量核对保持官方权重；只允许胞质直接跨细胞。',
             f'联合读出 {dimensions["joint"]} 维；区室侧读出 {dimensions["nucleus"]} 维。'
             '不同维度、参数量或训练预算的比较不能全部归因于结构。', '',
             '| 表征 | 任务 | 维度 | Best macro-F1 | Best C | 固定C macro-F1 |',
             '|---|---|---:|---:|---:|---:|']
    lines[2:2] = [f'路线 {report["arch_config"]["route"]}；两个核质/空间交替block及末端核质融合。'
                   'A投影最终gene-set；B分别投影自身及有/无邻域gene-set差，空邻域context严格为零。', '']
    for row in report['results']:
        fixed = row['fixed_C_result']
        lines.append(f'| {row["embedding"]} | {row["task"]} | {report["embedding_dimensions"][row["embedding"]]} | '
                     f'{row["best"]["val_macro_f1"]:.6f} | {row["best"]["C"]:g} | {fixed["val_macro_f1"]:.6f} |')
    lines += ['', '固定 C：细胞类型 0.1、区域 0.01。训练供体拟合 StandardScaler，'
              'class-balanced 线性 probe，C=[0.01,0.1,1,10] 仅在验证集选择；全部求解器必须收敛。',
              f'中心细胞数 {report["cells"]}，cohort SHA256：`{report["cohort_sha256"]}`。未读取 test 切片。',
              'base 是原始整细胞CLS，不经过区室embedding或slot pooling；intrinsic 是base加关闭空间消息的残差；'
              'joint 采用核质内部→胞质邻域交替更新，最后再做核质融合。nucleus_local/cytoplasm_local 是首次空间交互前读出；'
              'nucleus/cytoplasm 是反馈后的最终读出。核侧是否改善须单独比较 nucleus 与 nucleus_local，'
              '并结合真实/打乱邻居检验；向量变化或联合重建下降本身不等于核表征改善。']
    lines += ['', f'配对起点核验与残差幅度：`{json.dumps(report["anchor_comparison"], ensure_ascii=False)}`。',
              '增量投影零初始化，step0的joint必须逐值等于base；之后没有对和向量再次L2。'
              '核与胞质读出分别对flatten后向量L2。单一重建监督不保证这些读出均改善。',
              '本次base是相同token和确定性left分箱下的新鲜scGPT基线。'
              '跨实验的历史scGPT分数可能使用不同分箱、截断token和推理精度，'
              '比较V6-left增益时应以本报告配对base及相同固定C为准。']
    if report['neighbor_shuffle']:
        lines += ['邻域打乱保持中心表达、坐标、图和核/胞质可见性不变；同切片、同panel、'
                  '同可见性组内打乱表达与位置的对应关系，排除中心表达重复及重复细胞。'
                  '记录每切片实际替换比例，提供 joint/nucleus/cytoplasm 的打乱对照。'
                  '本打乱仅用于无mask下游提取，未对masked reconstruction执行shuffle；'
                  '一个shuffle种子不足以证明稳健提升或因果互作。']
    masked = report['masked_validation']
    if masked.get('enabled'):
        lines += ['', '固定遮盖验证：', '', '| Loss | Mean |', '|---|---:|']
        lines += [f'| {key} | {value:.6f} |' for key, value in masked['loss_mean'].items()]
        lines += ['', '唯一监督目标是最终共享预测头的51类masked-bin交叉熵；total和masked_ce是同一损失的两个字段。'
                  '核/胞质输入与整细胞CLS均先遮盖原计数，再分箱；完整panel分箱只用于监督目标。'
                  'LR及其归一化分母仅使用可见胞质计数，未测基因不等于真实零。',
                  'query 从被遮盖的已测 panel 均匀采样；PAD无监督。真实测量零bin仍有监督。'
                  '非零概率1−p(bin0)的AP/AUROC、正bin条件期望等级误差及全零基线仅作诊断，不增加loss。'
                  '逐mask模式和query区室结果见metrics.json。']
    temporary = output / 'REPORT.zh.md.partial'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'REPORT.zh.md')


def fixed_probe_comparisons(results):
    """Same-C readout differences, including independent nuclear feedback QA."""
    indexed = {(row['embedding'], row['task']): row['fixed_C_result'] for row in results}
    pairs = [('joint', 'base'), ('intrinsic', 'base'), ('joint', 'intrinsic'),
             ('nucleus', 'nucleus_local'), ('cytoplasm', 'cytoplasm_local')]
    pairs += [(name, f'{name}_neighbor_shuffle') for name in SHUFFLE_READOUTS]
    comparisons = []
    for left, right in pairs:
        for task in ('cell_type', 'region'):
            if (left, task) not in indexed or (right, task) not in indexed:
                continue
            first, second = indexed[left, task], indexed[right, task]
            if first['C'] != second['C']:
                raise ValueError('Cannot compare probes with different regularization')
            donor_a, donor_b = first.get('per_donor_macro_f1', {}), second.get('per_donor_macro_f1', {})
            if set(donor_a) != set(donor_b):
                raise ValueError('Cannot compare probes with different validation donors')
            comparisons.append(dict(left=left, right=right, task=task, C=first['C'],
                val_macro_f1_delta=first['val_macro_f1'] - second['val_macro_f1'],
                per_donor_macro_f1_delta={key: donor_a[key] - donor_b[key] for key in donor_a}))
    return comparisons


def anchor_comparison(features, step, *, spatial_enabled):
    """Compare the current plain CLS with fused embeddings, before any probe."""
    base, joint, intrinsic = (np.asarray(features[key]) for key in ('base', 'joint', 'intrinsic'))
    if base.ndim != 2 or base.shape != joint.shape or base.shape != intrinsic.shape:
        raise ValueError('Plain CLS and fused readouts must have identical [cells,dim] shapes')
    if not all(np.isfinite(value).all() for value in (base, joint, intrinsic)):
        raise FloatingPointError('Nonfinite anchor comparison')
    equal = np.array_equal(base, joint)
    if step == 0 and (not equal or not np.array_equal(base, intrinsic)):
        raise ValueError('Step-zero V6 must equal the paired plain scGPT CLS exactly')
    if not spatial_enabled and not np.array_equal(joint, intrinsic):
        raise ValueError('No-spatial arm must equal its intrinsic readout')
    delta = joint.astype(np.float64) - base.astype(np.float64)
    norms = np.linalg.norm(delta, axis=-1)
    base_norms = np.linalg.norm(base.astype(np.float64), axis=-1)
    return dict(step_zero_identity_verified=bool(step == 0 and equal),
                current_joint_equals_base=bool(equal),
                base_is_current_plain_scgpt=True, dimension=int(base.shape[-1]),
                spatial_enabled=bool(spatial_enabled),
                residual_mean_norm=float(norms.mean()), residual_max_norm=float(norms.max()),
                residual_to_base_mean_norm_ratio=float((norms / np.maximum(base_norms, 1e-12)).mean()),
                joint_minus_base_max_abs=float(np.abs(delta).max()),
                reference='Same whole-cell tokens, deterministic-left bins, frozen encoder, autocast dtype and probe protocol; '
                          'historical official-collator baseline is a separate reference')


def reserve_output(output, identity):
    """Allow resuming this evaluation, refuse to overwrite a different run."""
    manifest = output / 'evaluation_identity.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Output directory belongs to a different evaluation; choose a new output directory')
    elif any(path.name != 'diagnostic_failure.json' for path in output.iterdir()):
        raise ValueError('Nonempty output directory lacks V6 evaluation identity; refusing overwrite')
    else:
        atomic_json(manifest, identity)


def main():
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', '--output', dest='output', type=Path, required=True)
    parser.add_argument('--cohort', default='fixed5200', help='fixed5200 or a reference cells.parquet for exact ID/split validation')
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--cells-per-donor', type=int, default=400)
    parser.add_argument('--neighbor-shuffle', action='store_true')
    parser.add_argument('--neighbor-shuffle-seed', type=int, default=815,
                        help='Repeat in a distinct output directory for each shuffle seed')
    parser.add_argument('--mask-validation-batches', type=int, default=16)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.cells_per_donor < 1 or args.mask_validation_batches < 0
            or args.neighbor_shuffle_seed < 0):
        parser.error('batch size and cells per donor must be positive; mask batches and shuffle seed nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.monotonic()
    model, cfg, vocab, step, provenance = load_model(args.checkpoint, args.expected_step, args.config)
    dimensions = expected_dimensions(args.neighbor_shuffle, model.dim, model.backbone.dim)
    checkpoint_digest = file_sha256(args.checkpoint)
    vocab_digest = hashlib.sha256(json.dumps(vocab, sort_keys=True).encode()).hexdigest()
    prepared = ROOT / 'data/benchmarking_data/lungpf/prepared'
    protocol = json.loads((prepared / 'protocol.json').read_text())
    reserve_output(args.output, dict(evaluation_version=EVALUATION_VERSION,
        checkpoint_sha256=checkpoint_digest, config_sha256=provenance['config_sha256'],
        evaluator_source_sha256=file_sha256(Path(__file__)), protocol_sha256=file_sha256(prepared / 'protocol.json'),
        cohort=str(args.cohort), cells_per_donor=args.cells_per_donor, batch_size=args.batch_size,
        neighbor_shuffle=args.neighbor_shuffle, neighbor_shuffle_seed=args.neighbor_shuffle_seed,
        mask_validation_batches=args.mask_validation_batches))
    ACTIVE_OUTPUT = args.output
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
            neighbor_shuffle=args.neighbor_shuffle, neighbor_shuffle_seed=args.neighbor_shuffle_seed,
            embedding_dimensions=dimensions, arch_config=provenance['arch_config'],
            selected_ids_sha256=hashlib.sha256('\n'.join(chosen.full_cell_id.astype(str)).encode()).hexdigest())
        cache, metadata = cache_dir / f'{sample}.npz', cache_dir / f'{sample}.json'
        prior = json.loads(metadata.read_text()) if metadata.exists() else {}
        if cache.exists() and prior.get('signature') == signature:
            with np.load(cache, allow_pickle=False) as archive:
                features = {key: archive[key] for key in archive.files}
            qa = prior['qa']
        else:
            features, qa = extract(model, cfg, vocab, directory, obs, selected, args.batch_size,
                                   args.neighbor_shuffle, args.neighbor_shuffle_seed)
            temporary = cache.with_name(cache.stem + '.partial.npz')
            np.savez(temporary, **features)
            temporary.replace(cache)
            atomic_json(metadata, dict(signature=signature, qa=qa))
        if set(features) != set(dimensions):
            raise ValueError(f'Cached readout set mismatch: {sample}')
        for key, dim in dimensions.items():
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
        raise ValueError('V6 evaluation does not match the fixed 5,200-cell cohort')
    if args.cohort not in ('fixed5200', CANONICAL_COHORT):
        reference = pd.read_parquet(Path(args.cohort), columns=['full_cell_id', 'patient', 'split'])
        if not reference.reset_index(drop=True).equals(obs[['full_cell_id', 'patient', 'split']].reset_index(drop=True)):
            raise ValueError('Reference cohort IDs, order, donors or splits differ')
    temporary = args.output / 'cells.parquet.partial'
    obs.to_parquet(temporary, index=False)
    temporary.replace(args.output / 'cells.parquet')
    matrices = {key: np.concatenate(blocks) for key, blocks in matrices.items()}
    paired_anchor = anchor_comparison(matrices, step, spatial_enabled=cfg['model']['spatial_enabled'])
    reconstruction = masked_validation(model, cfg, step, args.mask_validation_batches, args.batch_size)
    del model
    torch.cuda.empty_cache()
    results, diagnostics = [], {}
    for key, features in matrices.items():
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
    report = dict(model='SubcellularV6DeterministicLeft', evaluation_version=EVALUATION_VERSION, diagnostic_schema_version=1,
        model_version=MODEL_VERSION, architecture=ARCHITECTURE, arch_config=provenance['arch_config'],
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_digest, step=step,
        backbone_kind='scgpt', backbone_provenance=provenance,
        lr_prior=provenance['lr_prior'],
        optimizer_manifest=provenance['optimizer_manifest'],
        source_sha256=provenance['checkpoint_source_sha256'],
        config_sha256=provenance['config_sha256'], vocab_mapping_sha256=vocab_digest,
        seed=42, evaluation_batch_size=args.batch_size, cells=len(obs),
        cells_per_donor=args.cells_per_donor, cohort_sha256=cohort,
        canonical_cohort=cohort == CANONICAL_COHORT, test_donors_used=False,
        train_donors=sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique()),
        val_donors=sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique()),
        context_cells=int(cfg['data'].get('context_cells', 64)), embedding_dimensions=dimensions,
        input='Frozen whole-cell CLS plus independent nuclear/cytoplasmic slot residual; deterministic-left bins; no downstream mask',
        whole_cell_protocol='Raw nuclear+cytoplasmic sums; deterministic-left nonzero per-cell 51 bins, CLS+L2; '
                            'V6-identical random stream and gene order when truncated; bfloat16 encoder/float32 pooling',
        protocol_sha256=file_sha256(prepared / 'protocol.json'),
        cohort_labels_split_sha256=hashlib.sha256(obs[
            ['full_cell_id', 'patient', 'split', 'final_CT', 'region_label']
        ].fillna('').astype(str).to_csv(index=False).encode()).hexdigest(),
        anchor_comparison=paired_anchor,
        binning_protocol=BINNING_PROTOCOL, graph='Per-center KNN patch; all spatial neighborhoods truncated to this patch',
        neighbor_shuffle=args.neighbor_shuffle, neighbor_shuffle_seed=args.neighbor_shuffle_seed,
        masked_validation=reconstruction, fixed_probe_comparisons=fixed_probe_comparisons(results),
        diagnostics=diagnostics, coverage=coverage, results=results, seconds=time.monotonic() - started)
    write_report(args.output, report)
    atomic_json(args.output / 'metrics.json', report)
    (args.output / 'diagnostic_failure.json').unlink(missing_ok=True)
    print(f'[done-v6] step={step} seconds={report["seconds"]:.1f} output={args.output}', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Do not write even a failure marker into an unvalidated foreign output.
        if ACTIVE_OUTPUT is not None:
            kind = ('nonfinite' if isinstance(error, FloatingPointError) else
                    'out_of_memory' if isinstance(error, torch.cuda.OutOfMemoryError) else 'runtime')
            atomic_json(ACTIVE_OUTPUT / 'diagnostic_failure.json', dict(kind=kind, error_type=type(error).__name__,
                        detail=str(error)[:3000], timestamp=time.time()))
        raise
