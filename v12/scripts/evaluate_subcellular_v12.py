"""All V12 readouts, three tasks, canonical train10/val3 5,200-cell cohort.

Run on a compute allocation. No labels enter the network. Checkpoint-dependent
encodings are never shared between checkpoints. Only original frozen scGPT
baseline encodings may be shared. Atomic metrics.json is the completion marker.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from evaluate_subcellular_v2 import CANONICAL_COHORT, atomic_array, atomic_json, file_sha256
from probe_lungpf_embeddings import fit_linear, predict

MODEL_VERSION = 12
ARCHITECTURE = 'subcellular_compartment_counts_v12'
MODEL_READOUTS = ('nucleus', 'cytoplasm', 'local', 'spatial', 'fused', 'concat')
COUNT_MODES = ('nucleus', 'cytoplasm', 'local', 'spatial', 'fused')
BASELINES = ('scgpt_whole', 'scgpt_nucleus', 'scgpt_cytoplasm')
READOUTS = MODEL_READOUTS + ('concat_pca512',) + BASELINES
TASKS = ('cell_type', 'region', 'neighbor_composition')
FIXED_C = {'cell_type': .1, 'region': .01}
PROBE_ROW_NORMALIZATION = 'L2 for all ten readouts before train-only StandardScaler'
INVALID_LABELS = {'', 'nan', 'None', '<NA>'}
UNKNOWN_COMPOSITION_LABELS = INVALID_LABELS | {'Unknown', 'unknown', 'Unassigned', 'unassigned', 'Unlabeled', 'unlabeled', 'NA'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ids_hash(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()


def write_frame(path, frame):
    temporary = path.with_name(path.name + '.partial')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def valid_labels(labels):
    return labels.notna() & ~labels.astype(str).isin(INVALID_LABELS)


def sample_inputs(directory):
    return dict(obs_sha256=file_sha256(directory / 'obs.parquet'),
                genes_sha256=file_sha256(directory / 'genes.npy'),
                counts={name: dict(size=(directory / name).stat().st_size,
                                  mtime_ns=(directory / name).stat().st_mtime_ns)
                        for name in ('nuclear.npy', 'cytoplasmic.npy')})


def load_model(path, expected_step):
    from spatial_pretrain.models.subcellular_v12 import build_model
    from spatial_pretrain.models.scgpt_backbone import ScGPTBackbone
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('model_version') != MODEL_VERSION or saved.get('architecture') != ARCHITECTURE:
        raise ValueError(f'Expected V12 {ARCHITECTURE}')
    step = int(saved['step'])
    if step < 0 or (expected_step is not None and step != expected_step):
        raise ValueError('Checkpoint step mismatch')
    cfg = saved['config']
    if saved.get('config_sha256') != fingerprint(cfg):
        raise ValueError('Missing or incorrect config_sha256')
    sources = saved.get('source_sha256')
    required = {'src/spatial_pretrain/models/subcellular_v12.py',
                'src/spatial_pretrain/data/subcellular_v12.py'}
    if not isinstance(sources, dict) or not required.issubset(sources):
        raise ValueError('Checkpoint lacks model/data source hashes')
    for name, digest in sources.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or file_sha256(source) != digest:
            raise ValueError(f'Checkpoint source mismatch: {name}')
    model = build_model(cfg)
    original = dict(model.backbone.provenance)
    if original.get('backbone_kind') != 'scgpt':
        raise ValueError('V12 requires ordinary scGPT initialization')
    if saved.get('backbone_provenance') != original:
        raise ValueError('Checkpoint initial-backbone provenance mismatch')
    if step == 0:
        for name, tensor in model.backbone.state_dict().items():
            archived = saved['model'].get(f'backbone.{name}')
            if archived is None or not torch.equal(tensor.detach().cpu(), archived.detach().cpu()):
                raise ValueError(f'Step-zero encoder differs from original scGPT: {name}')
    model.load_state_dict(saved['model'], strict=True)
    # A new original asset loader, NEVER the fine-tuned model backbone.
    baseline = ScGPTBackbone(cfg['assets_dir'], checkpointing=False, backbone_kind='scgpt')
    if baseline.provenance != original:
        raise ValueError('Original baseline and initialization provenance differ')
    provenance = dict(config_sha256=fingerprint(cfg), source_sha256=sources,
        sources_verified=True, backbone_provenance=original,
        baseline_separate_pretrained_encoder=True)
    del saved
    return (model.eval().requires_grad_(False).cuda(),
            baseline.eval().requires_grad_(False).cuda(), cfg, step, provenance)


def real_neighbors(coords, centers, k):
    count = min(int(k) + 1, len(coords))
    if count < 2:
        raise ValueError('At least two physical cells required')
    nearest = NearestNeighbors(n_neighbors=count).fit(coords).kneighbors(coords[centers], return_distance=False)
    result = np.full((len(centers), k), -1, dtype=np.int64)
    for index, (center, row) in enumerate(zip(centers, nearest)):
        row = row[row != center][:k]
        if len(np.unique(row)) != len(row):
            raise ValueError('Duplicate physical neighbors')
        result[index, :len(row)] = row
    if (result == np.asarray(centers)[:, None]).any():
        raise ValueError('Center leaked into neighbors')
    return result


def neighbor_composition(obs, centers, classes, radius):
    """External labels in a fixed physical radius; no model-predicted labels.

    Classes come from TRAIN donors only. Unknown/missing labels are excluded
    from the normalized known-label target and explicitly retained in coverage.
    """
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float64)
    if radius <= 0 or not np.isfinite(coords).all():
        raise ValueError('Invalid composition radius/coordinates')
    rows = NearestNeighbors(radius=radius).fit(coords).radius_neighbors(coords[centers], return_distance=False)
    lookup = {str(name): index for index, name in enumerate(classes)}
    labels = obs['final_CT'].astype('string').fillna('').to_numpy(dtype=str)
    counts = np.zeros((len(centers), len(classes)), dtype=np.float64)
    total = np.zeros(len(centers), dtype=np.int64)
    missing = total.copy()
    unseen = total.copy()
    for index, (center, row) in enumerate(zip(centers, rows)):
        row = np.unique(row[row != center])
        total[index] = len(row)
        for label in labels[row]:
            if label in lookup:
                counts[index, lookup[label]] += 1
            elif label in UNKNOWN_COMPOSITION_LABELS:
                missing[index] += 1
            else:
                unseen[index] += 1
    known = counts.sum(axis=1).astype(np.int64)
    target = counts / np.maximum(known[:, None], 1)
    return target, dict(total=total, known=known, missing=missing, unseen=unseen,
                        coverage=known / np.maximum(total, 1))


def count_model_inputs(batch):
    """Query identities are allowed; observed target counts never enter forward."""
    allowed_query = {'query_gene_ids', 'query_valid_mask'}
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()
            if not key.startswith('query_') or key in allowed_query}


def joint_count_nll(nuclear, cytoplasm, parameters):
    """Proper NB(total)*beta-binomial(nucleus|total) score in float64.

    Parameterization: NB variance=mu+mu^2/theta; BB alpha=p*kappa,
    beta=(1-p)*kappa. Delegate to the exact training probability law after
    checking support; no continuous rounding approximation or formula fork.
    """
    n, c = nuclear.double(), cytoplasm.double()
    if (n < 0).any() or (c < 0).any() or not torch.equal(n, n.round()) or not torch.equal(c, c.round()):
        raise ValueError('Joint count evaluation requires nonnegative integers')
    mu, theta, p, kappa = [parameters[key].double() for key in
                          ('mu_total', 'theta_total', 'p_nucleus', 'kappa_allocation')]
    if n.shape != c.shape or any(value.shape != n.shape for value in (mu, theta, p, kappa)):
        raise ValueError('Joint count parameters and targets must share [B,Q] shape')
    if not all(torch.isfinite(value).all() for value in (mu, theta, p, kappa)):
        raise ValueError('Nonfinite count decoder parameters')
    if (mu <= 0).any() or (theta <= 0).any() or (kappa <= 0).any() or (p <= 0).any() or (p >= 1).any():
        raise ValueError('Invalid NB/beta-binomial parameters')
    from spatial_pretrain.training.subcellular_v12 import joint_count_nll as training_nll
    result = training_nll(parameters, n, n+c)
    if not torch.isfinite(result).all() or (result < -1e-6).any():
        raise ValueError('Invalid discrete joint NLL')
    return result.clamp_min(0)


def score_masked_counts(result, targets, centers):
    n = targets['query_nucleus_counts'].to('cuda', dtype=torch.float64)
    c = targets['query_cytoplasm_counts'].to('cuda', dtype=torch.float64)
    valid = targets['query_valid_mask'].to('cuda').bool()
    if not torch.equal(n+c, targets['query_total_counts'].to('cuda', dtype=torch.float64)):
        raise ValueError('Held-out nucleus/cytoplasm totals do not reconcile')
    rows = []
    for mode in COUNT_MODES:
        parameters = result['count_parameters'][mode]
        score = joint_count_nll(n, c, parameters)
        active = valid & result['mode_available'][mode][:, None]
        positive = active & ((n+c) > 0)
        mu, p = parameters['mu_total'].double(), parameters['p_nucleus'].double()
        sums = dict(query_count=active.sum(1), positive_query_count=positive.sum(1),
            nll_sum=(score*active).sum(1), total_abs_error_sum=((mu-n-c).abs()*active).sum(1),
            nucleus_abs_error_sum=((mu*p-n).abs()*active).sum(1),
            cytoplasm_abs_error_sum=((mu*(1-p)-c).abs()*active).sum(1),
            allocation_abs_error_sum=((p-n/(n+c).clamp_min(1)).abs()*positive).sum(1))
        sums = {key: value.detach().cpu().tolist() for key, value in sums.items()}
        for index, center in enumerate(centers):
            rows.append(dict(center_row=int(center), mode=mode, **{key: value[index] for key, value in sums.items()}))
    return rows


def aggregate_masked_counts(frame):
    """Report available-query means; missing views have no fabricated score."""
    rows = []
    for (split, mode), part in frame.groupby(['split', 'mode'], sort=True):
        for donor, subset in [('pooled', part), *list(part.groupby('patient', sort=True))]:
            count, positive = int(subset.query_count.sum()), int(subset.positive_query_count.sum())
            means = {name.removesuffix('_sum'): float(subset[name].sum()/count) if count else None
                     for name in ('nll_sum', 'total_abs_error_sum', 'nucleus_abs_error_sum', 'cytoplasm_abs_error_sum')}
            means['allocation_abs_error'] = float(subset.allocation_abs_error_sum.sum()/positive) if positive else None
            rows.append(dict(split=str(split), mode=str(mode), donor=str(donor), query_count=count,
                positive_query_count=positive, available_cells=int((subset.query_count > 0).sum()), **means))
    return rows


def frozen_baselines(backbone, cfg, directory, selected, nuclear, cyto, genes, slice_seed,
                     batch_size, cache_root):
    from spatial_pretrain.data.subcellular_v6_left import encode_whole_cell_v6_left
    max_tokens = int(cfg.get('v12', {}).get('max_tokens', 512))
    contract = dict(schema_version=2, sample=str(directory.resolve()), inputs=sample_inputs(directory),
        selected_sha256=hashlib.sha256(np.asarray(selected, dtype=np.int64).tobytes()).hexdigest(),
        backbone=backbone.provenance, max_tokens=max_tokens, seed=42, slice_seed=slice_seed,
        tokenization='deterministic-left V6 packing; unmasked nuclear, cytoplasmic and summed raw whole counts',
        readout='raw CLS; common row L2 normalization occurs only before downstream probes',
        sources={name: file_sha256(ROOT / name) for name in
                 ('src/spatial_pretrain/models/scgpt_backbone.py',
                  'src/spatial_pretrain/data/subcellular_v6_left.py',
                  'src/spatial_pretrain/data/subcellular_v5.py', 'scripts/evaluate_subcellular_v12.py')},
        torch_version=torch.__version__, dtype='bf16 autocast float32 stored', batch_size=batch_size)
    contract = json.loads(json.dumps(contract))
    output = cache_root / fingerprint(contract)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = output / 'complete.json'
        if complete.is_file():
            if json.loads(complete.read_text())['contract'] != contract:
                raise ValueError('Original baseline cache contract mismatch')
            values = {name: np.load(output / f'{name}.npy', allow_pickle=False) for name in BASELINES}
            reused = True
        else:
            blocks = {name: [] for name in BASELINES}
            for start in range(0, len(selected), batch_size):
                rows = selected[start:start + batch_size]
                for name in BASELINES:
                    packed = [encode_whole_cell_v6_left(
                              nuclear[row] if name != 'scgpt_cytoplasm' else np.zeros(len(genes), dtype=np.int64),
                              cyto[row] if name != 'scgpt_nucleus' else np.zeros(len(genes), dtype=np.int64), genes,
                              pad_token_id=backbone.vocab['<pad>'], cls_token_id=backbone.vocab['<cls>'],
                              max_tokens=max_tokens, seed=np.random.SeedSequence([42, slice_seed, int(row), 355]))
                              for row in rows]
                    length = max(len(item['whole_gene_ids']) for item in packed)
                    token = np.full((len(rows), length), backbone.vocab['<pad>'], dtype=np.int64)
                    bins = np.full((len(rows), length), -2., dtype=np.float32)
                    valid = np.zeros((len(rows), length), dtype=bool)
                    for index, item in enumerate(packed):
                        size = len(item['whole_gene_ids'])
                        token[index, :size] = item['whole_gene_ids']
                        bins[index, :size] = item['whole_input_bins']
                        valid[index, :size] = item['whole_token_valid_mask']
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        encoded = backbone(torch.as_tensor(token, device='cuda'), torch.as_tensor(bins, device='cuda'),
                                           ~torch.as_tensor(valid, device='cuda'), 0.)[:, 0]
                    encoded = encoded.float().cpu().numpy()
                    blocks[name].append(encoded)
            values = {name: np.concatenate(blocks[name]) for name in BASELINES}
            for name, value in values.items():
                atomic_array(output / f'{name}.npy', value)
            atomic_json(complete, dict(contract=contract))
            reused = False
    for name, value in values.items():
        if value.shape != (len(selected), backbone.dim) or not np.isfinite(value).all():
            raise ValueError(f'Invalid original baseline {name}')
    return values, dict(path=str(output), reused=reused)


def extract(model, baseline, cfg, directory, obs, selected, batch_size, slice_index, baseline_cache, step=None):
    from spatial_pretrain.data.subcellular_v12 import prepare_view_batch_v12
    from spatial_pretrain.data.subcellular_v6 import load_lr_prior
    started = time.monotonic()
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not obs.full_cell_id.is_unique:
        raise ValueError('Invalid coordinates or duplicate cells')
    settings = cfg.get('v12', {})
    k = int(settings.get('neighbors', 8))
    if k != model.neighbors:
        raise ValueError('V12 data/model neighbor count mismatch')
    neighbors = real_neighbors(coords, selected, k)
    panel = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([i for i, gene in enumerate(panel) if gene in model.vocab], dtype=np.int64)
    if not len(keep):
        raise ValueError('No genes map to scGPT vocabulary')
    # V12 canonicalizes panel columns before token truncation. Use the identical
    # ordering for the original baseline, so a token-budget choice is not
    # mistaken for a fine-tuning effect.
    keep = keep[np.argsort([model.vocab[gene] for gene in panel[keep]], kind='stable')]
    genes = np.array([model.vocab[gene] for gene in panel[keep]], dtype=np.int64)
    # Advanced indexing materializes only mapped panel columns on compute node.
    nuclear_all = np.load(directory / 'nuclear.npy', mmap_mode='r', allow_pickle=False)
    cyto_all = np.load(directory / 'cytoplasmic.npy', mmap_mode='r', allow_pickle=False)
    if nuclear_all.shape != cyto_all.shape or nuclear_all.shape != (len(obs), len(panel)):
        raise ValueError('Panel/counts/observations shape mismatch')
    nuclear = np.asarray(nuclear_all[:, keep])
    cyto = np.asarray(cyto_all[:, keep])
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    lr = load_lr_prior(cfg, vocab=model.vocab)
    blocks = {name: [] for name in MODEL_READOUTS}
    availability = {name: [] for name in COUNT_MODES}
    masked_count_blocks = []
    for start in range(0, len(selected), batch_size):
        end = min(start + batch_size, len(selected))
        batch = prepare_view_batch_v12(nuclear, cyto, genes, selected[start:end], neighbors[start:end], coords,
            pad_token_id=model.vocab['<pad>'], cls_token_id=model.vocab['<cls>'],
            max_tokens=int(settings.get('max_tokens', 512)), query_genes=int(settings.get('query_genes', 64)),
            seed=42, epoch=0, slice_seed=slice_seed, slice_index=slice_index, mask=False,
            lr_pairs=lr['pairs'], lr_weights=lr['weights'],
            max_neighbor_distance=settings.get('max_neighbor_distance', 100.0))
        # Even evaluation-only expression targets are never passed to the model.
        batch = {key: value.cuda(non_blocking=True) for key, value in batch.items() if not key.startswith('query_')}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            result = model(batch, return_embeddings=True)
        for name in MODEL_READOUTS:
            value = result['embeddings'][name].float().cpu().numpy()
            if value.ndim != 2 or len(value) != end - start or not np.isfinite(value).all():
                raise ValueError(f'Invalid V12 {name} encoding')
            blocks[name].append(value)
        for name in availability:
            availability[name].append(result['mode_available'][name].cpu().numpy().astype(bool))
        # A second, genuinely held-out forward pass. Each query is removed from
        # BOTH center compartments before binning/count adapters/LR features.
        # Fixed query RNG, no extra gene masks or neighbor dropout: checkpoint
        # comparisons measure the same prediction problem at every step.
        heldout = prepare_view_batch_v12(nuclear, cyto, genes, selected[start:end], neighbors[start:end], coords,
            pad_token_id=model.vocab['<pad>'], cls_token_id=model.vocab['<cls>'],
            max_tokens=int(settings.get('max_tokens', 512)), query_genes=int(settings.get('query_genes', 64)),
            seed=42, epoch=0, slice_seed=slice_seed, slice_index=slice_index, mask=True,
            gene_mask_ratio=0., neighbor_dropout=0., lr_pairs=lr['pairs'], lr_weights=lr['weights'],
            max_neighbor_distance=settings.get('max_neighbor_distance', 100.0))
        for count_name in ('nucleus', 'cytoplasm', 'total'):
            if f'query_{count_name}_counts' not in heldout:
                raise ValueError('Missing raw count targets in V12 evaluation batch')
        inputs = count_model_inputs(heldout)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(inputs)
        masked_count_blocks.extend(score_masked_counts(prediction, heldout, selected[start:end]))
        print(f'[embed-v12] {directory.name} {end}/{len(selected)} {time.monotonic()-started:.1f}s', flush=True)
    values = {name: np.concatenate(blocks[name]) for name in MODEL_READOUTS}
    frozen, cache_info = frozen_baselines(baseline, cfg, directory, selected, nuclear, cyto, genes,
                                         slice_seed, batch_size, baseline_cache)
    values.update(frozen)
    if any(values[name].shape[1] != 512 for name in (*COUNT_MODES, *BASELINES)) or values['concat'].shape[1] != 1536:
        raise ValueError('V12 primary readout dimension contract violated')
    available = {name: np.concatenate(value) for name, value in availability.items()}
    initialization = None
    if step == 0:
        initialization = {}
        for name in ('nucleus', 'cytoplasm'):
            present = available[name]
            maximum_difference = (float(np.max(np.abs(values[name][present] - values[f'scgpt_{name}'][present])))
                                  if present.any() else None)
            initialization[name] = dict(checked_available_cells=int(present.sum()), raw_cls_max_absolute_difference=maximum_difference,
                atol=.03, rtol=.02, tolerance_note='BF16 packing differences; encoder weights checked exactly',
                absent_cells_excluded=int((~present).sum()))
            if present.any() and not np.allclose(values[name][present], values[f'scgpt_{name}'][present], atol=.03, rtol=.02):
                raise ValueError(f'Step-zero {name} CLS differs from original scGPT baseline: {initialization}')
    return values, dict(sample=directory.name, selected_cells=len(selected), matched_genes=len(keep),
        panel_genes=len(panel), center_excluded=True, mask=False, neighbors=k, lr=lr['metadata'],
        available_counts={name: int(value.sum()) for name, value in available.items()},
        absent_nucleus_policy='V12 uses zero embedding; original baseline uses empty-input CLS',
        step_zero_initialization_comparison=initialization,
        masked_count_rows=masked_count_blocks,
        frozen_baseline_cache=cache_info, fresh_checkpoint_encoder=True, seconds=time.monotonic()-started)


def concat_pca(features, train, components=512):
    """Every preprocessing statistic fits TRAIN cells; never validation."""
    if int(np.asarray(train).sum()) < components or features.shape[1] < components:
        raise ValueError('Insufficient training samples/features for fixed-dimension PCA')
    scaler = StandardScaler().fit(features[train])
    scaled = scaler.transform(features)
    pca = PCA(n_components=components, svd_solver='randomized', random_state=42).fit(scaled[train])
    result = pca.transform(scaled).astype(np.float32)
    return result, dict(components=components, fit_split='train', seed=42,
        explained_variance_ratio_sum=float(pca.explained_variance_ratio_.sum()),
        scaler_mean_sha256=hashlib.sha256(scaler.mean_.tobytes()).hexdigest(),
        components_sha256=hashlib.sha256(pca.components_.tobytes()).hexdigest())


def normalize_for_probe(features):
    """Common row L2 for every readout; leave saved raw arrays untouched."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Probe normalization requires a finite embedding matrix')
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def project_simplex(values):
    """Euclidean row projection onto probabilities, including negative ridge outputs."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1] or not np.isfinite(values).all():
        raise ValueError('Simplex projection requires finite nonempty vectors')
    ordered = np.sort(values, axis=1)[:, ::-1]
    partial = np.cumsum(ordered, axis=1) - 1
    index = np.arange(1, values.shape[1] + 1)
    rho = (ordered - partial / index > 0).sum(axis=1) - 1
    theta = partial[np.arange(len(values)), rho] / (rho + 1)
    return np.maximum(values - theta[:, None], 0)


def composition_scores(target, probability):
    for values in (target, probability):
        if not np.isfinite(values).all() or (values < -1e-10).any() or not np.allclose(values.sum(axis=1), 1):
            raise ValueError('Composition vectors must be probabilities')
    midpoint = (target + probability) / 2
    def kl(first):
        return np.sum(np.where(first > 0, first * np.log(np.maximum(first, 1e-30) / np.maximum(midpoint, 1e-30)), 0), axis=1)
    divergence = .5 * (kl(target) + kl(probability))
    return dict(mae=float(np.abs(target-probability).mean()),
                js_divergence=float(divergence.mean()), js_distance=float(np.sqrt(np.maximum(divergence, 0)).mean()))


def composition_probe(obs, features, target, coverage, classes, name, output, alpha=1.):
    train = obs['split'].eq('train').to_numpy() & (coverage['known'] > 0)
    val = obs['split'].eq('val').to_numpy() & (coverage['known'] > 0)
    if not train.any() or not val.any():
        raise ValueError('No labeled physical-neighbor targets in train or validation')
    scaler = StandardScaler().fit(features[train])
    ridge = Ridge(alpha=alpha, solver='cholesky').fit(scaler.transform(features[train]), target[train])
    probability = project_simplex(ridge.predict(scaler.transform(features[val])))
    score = composition_scores(target[val], probability)
    donors = obs.loc[val, 'patient'].astype(str).to_numpy()
    per_donor = {donor: composition_scores(target[val][donors == donor], probability[donors == donor])
                 for donor in np.unique(donors)}
    frame = obs.loc[val, ['full_cell_id', 'patient']].copy()
    for index, label in enumerate(classes):
        frame[f'target_{index}'] = target[val, index]
        frame[f'prediction_{index}'] = probability[:, index]
    frame['known_neighbors'] = coverage['known'][val]
    frame['total_neighbors'] = coverage['total'][val]
    frame['label_coverage'] = coverage['coverage'][val]
    write_frame(output / f'{name}_neighbor_composition_val_predictions.parquet', frame)
    result = dict(embedding=name, task='neighbor_composition', embedding_dimension=features.shape[1],
        train_cells=int(train.sum()), val_cells=int(val.sum()), ridge_alpha=alpha,
        probability_transform='Euclidean simplex projection; probabilities nonnegative and sum to one',
        class_names=list(map(str, classes)), val_result=score, per_donor=per_donor,
        mean_val_label_coverage=float(coverage['coverage'][val].mean()),
        zero_known_val_cells=int((obs['split'].eq('val').to_numpy() & (coverage['known'] == 0)).sum()),
        js_units='natural log; divergence [0,ln2], distance [0,sqrt(ln2)]; lower is better')
    print(f'[probe-v12] {name} composition MAE={score["mae"]:.6f} JSD={score["js_distance"]:.6f}', flush=True)
    return result


def classification_probe(obs, features, name, task, output):
    column = {'cell_type': 'final_CT', 'region': 'region_label'}[task]
    labels = obs[column]
    usable = valid_labels(labels)
    train = usable & obs['split'].eq('train')
    val = usable & obs['split'].eq('val')
    classes = np.unique(labels[train].astype(str))
    if len(classes) < 2 or not val.any():
        raise ValueError(f'Insufficient labels: {task}')
    lookup = {label: index for index, label in enumerate(classes)}
    scaler = StandardScaler().fit(features[train])
    tx = torch.as_tensor(scaler.transform(features[train]), device='cuda', dtype=torch.float32)
    vx = torch.as_tensor(scaler.transform(features[val]), device='cuda', dtype=torch.float32)
    y = torch.tensor([lookup[label] for label in labels[train].astype(str)], device='cuda')
    weights, bias, info = fit_linear(tx, y, FIXED_C[task])
    prediction = predict(vx, weights, bias, classes)
    true = labels[val].astype(str).to_numpy()
    donors = obs.loc[val, 'patient'].astype(str).to_numpy()
    def scores(mask):
        return dict(macro_f1=float(f1_score(true[mask], prediction[mask], labels=np.unique(true[mask]), average='macro', zero_division=0)),
                    balanced_accuracy=float(balanced_accuracy_score(true[mask], prediction[mask])))
    all_scores = scores(np.ones(len(true), dtype=bool))
    per_donor = {donor: dict(cells=int((donors == donor).sum()), **scores(donors == donor)) for donor in np.unique(donors)}
    score = dict(C=FIXED_C[task], val_macro_f1=all_scores['macro_f1'], val_balanced_accuracy=all_scores['balanced_accuracy'],
        per_donor=per_donor, per_donor_macro_f1={donor: row['macro_f1'] for donor, row in per_donor.items()}, solver=info,
        per_class=classification_report(true, prediction, output_dict=True, zero_division=0))
    frame = obs.loc[val, ['full_cell_id', 'patient']].copy()
    frame['label'], frame['prediction'] = true, prediction
    write_frame(output / f'{name}_{task}_val_predictions.parquet', frame)
    print(f'[probe-v12] {name} {task} macroF1={score["val_macro_f1"]:.6f}', flush=True)
    return dict(embedding=name, task=task, embedding_dimension=features.shape[1], train_cells=int(train.sum()),
                val_cells=int(val.sum()), unseen_val_classes=sorted(set(true)-set(classes)), fixed_C_result=score)


def representation_diagnostics(obs, features):
    val = obs['split'].eq('val').to_numpy()
    result = {}
    for name, value in features.items():
        value = np.asarray(value[val], dtype=np.float64)
        centered = value - value.mean(axis=0)
        eigen = np.linalg.eigvalsh(centered.T @ centered / max(len(value)-1, 1)).clip(0)
        probability = eigen[eigen > 0] / eigen.sum() if eigen.sum() else np.array([])
        result[name] = dict(mean_feature_variance=float(value.var(axis=0, ddof=1).mean()),
            mean_feature_std=float(value.std(axis=0, ddof=1).mean()),
            effective_rank=float(np.exp(-(probability*np.log(probability)).sum())) if len(probability) else 0.,
            zero_norm_cells=int((np.linalg.norm(value, axis=1) < 1e-12).sum()), embedding_dimension=value.shape[1])
    return dict(split='val', collapse=result, effective_rank_definition='exp(entropy(normalized covariance eigenvalues))')


def validate_completion(report, directory=None):
    """Shared by evaluator, watcher, summarizer and unit tests."""
    if report.get('status') != 'complete' or report.get('model_version') != 12 or report.get('architecture') != ARCHITECTURE:
        raise ValueError('Invalid V12 completion marker')
    if report.get('cohort_sha256') != CANONICAL_COHORT or report.get('cells') != 5200:
        raise ValueError('Not the canonical 5200-cell cohort')
    if not report.get('no_test') or report.get('test_donors_used') is not False or not report.get('sources_verified'):
        raise ValueError('Missing split/source verification')
    if not report.get('baseline_separate_pretrained_encoder'):
        raise ValueError('Missing independent original-scGPT baseline provenance')
    if report.get('probe_row_normalization') != PROBE_ROW_NORMALIZATION:
        raise ValueError('Missing uniform L2 probe normalization protocol')
    masked = report.get('masked_count_prediction', {})
    if not masked.get('both_center_compartments_query_hidden') or masked.get('target_counts_passed_to_forward') is not False:
        raise ValueError('Missing held-out count prediction safeguards')
    count_rows = masked.get('results', [])
    if {(row['split'], row['mode']) for row in count_rows if row['donor'] == 'pooled'} != {(split, mode) for split in ('train', 'val') for mode in COUNT_MODES}:
        raise ValueError('Incomplete masked count mode evaluation')
    for row in count_rows:
        if row['query_count'] and (row['nll'] is None or not np.isfinite(row['nll']) or row['nll'] < 0):
            raise ValueError('Invalid masked count NLL')
    if report.get('fixed_C') != FIXED_C or len(report.get('train_donors', [])) != 10 or len(report.get('val_donors', [])) != 3:
        raise ValueError('Wrong fixed probe/donor protocol')
    expected = {(name, task) for name in READOUTS for task in TASKS}
    rows = report.get('results', [])
    if len(rows) != len(expected) or {(row['embedding'], row['task']) for row in rows} != expected:
        raise ValueError('All readouts and all three tasks must complete without duplicates')
    if set(report.get('diagnostics', {}).get('collapse', {})) != set(READOUTS):
        raise ValueError('Missing readout diagnostics')
    dimensions = report.get('dimensions', {})
    if any(dimensions.get(name) != (1536 if name == 'concat' else 512) for name in READOUTS):
        raise ValueError('Unexpected readout dimensions')
    for row in rows:
        if row.get('embedding_dimension') != dimensions[row['embedding']] or row.get('train_cells', 0) < 1 or row.get('val_cells', 0) < 1:
            raise ValueError('Invalid probe dimension or sample counts')
        if row['task'] in FIXED_C:
            score = row['fixed_C_result']
            if score['C'] != FIXED_C[row['task']] or not score['solver'].get('converged_gradient'):
                raise ValueError('Unconverged/wrong-C classifier')
            if not all(np.isfinite(score[key]) and 0 <= score[key] <= 1 for key in ('val_macro_f1', 'val_balanced_accuracy')):
                raise ValueError('Invalid classification score')
        else:
            score = row['val_result']
            if not all(np.isfinite(score[key]) and score[key] >= 0 for key in ('mae', 'js_divergence', 'js_distance')):
                raise ValueError('Invalid neighbor-composition score')
            if row.get('ridge_alpha') != 1. or score['mae'] > 1 or score['js_divergence'] > np.log(2) + 1e-10 or score['js_distance'] > np.sqrt(np.log(2)) + 1e-10:
                raise ValueError('Wrong composition probe or out-of-range score')
        if directory is not None and not (directory / f'{row["embedding"]}_{row["task"]}_val_predictions.parquet').is_file():
            raise ValueError('Missing per-cell predictions')
    if directory is not None and not (directory / 'masked_count_per_cell.parquet').is_file():
        raise ValueError('Missing per-cell held-out count evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', '--output', required=True, type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/benchmarking_data/lungpf/prepared')
    parser.add_argument('--baseline-cache-dir', type=Path, default=ROOT / 'outputs/subcellular_v12_original_baseline_cache')
    parser.add_argument('--composition-radius-um', default=100., type=float)
    args = parser.parse_args()
    if args.batch_size < 1 or args.composition_radius_um <= 0:
        parser.error('Batch size and radius must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'metrics.json').exists():
        raise FileExistsError('Refusing to overwrite completed metrics')
    started = time.monotonic()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint_sha = file_sha256(args.checkpoint)
    model, baseline, cfg, step, provenance = load_model(args.checkpoint, args.expected_step)
    protocol_path = args.prepared / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    metadata, classes = [], set()
    for index, sample in enumerate(protocol['samples']):
        matches = [donor for donor in protocol['donor_split'] if sample.startswith(donor)]
        if len(matches) != 1:
            raise ValueError('Ambiguous donor before opening sample')
        donor = matches[0]
        split = protocol['donor_split'][donor]
        if split == 'test':
            continue
        if split not in ('train', 'val'):
            raise ValueError('Unexpected split')
        directory = args.prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        if not obs.patient.eq(donor).all() or ('split' in obs and not obs['split'].eq(split).all()):
            raise ValueError('Observation/protocol split mismatch')
        if split == 'train':
            classes.update(label for label in obs.loc[valid_labels(obs.final_CT), 'final_CT'].astype(str)
                           if label not in UNKNOWN_COMPOSITION_LABELS)
        metadata.append((index, directory, obs, split))
    classes = sorted(classes)
    if len(classes) < 2:
        raise ValueError('Insufficient training-donor neighbor classes')
    contract = dict(schema_version=12, checkpoint_sha256=checkpoint_sha, step=step,
        config_sha256=provenance['config_sha256'], source_sha256=provenance['source_sha256'],
        evaluator_sha256=file_sha256(Path(__file__)), solver_sha256=file_sha256(ROOT / 'scripts/probe_lungpf_embeddings.py'),
        protocol_sha256=file_sha256(protocol_path), prepared=str(args.prepared.resolve()),
        canonical_cohort=CANONICAL_COHORT, fixed_C=FIXED_C, readouts=READOUTS, tasks=TASKS, mask=False, seed=42,
        composition_radius_um=args.composition_radius_um, composition_classes=classes, ridge_alpha=1.,
        batch_size=args.batch_size, input_signatures={directory.name: sample_inputs(directory) for _, directory, _, _ in metadata})
    contract['embedding_storage'] = 'raw native outputs; concat retains architecture-defined normalized nucleus/cytoplasm/spatial parts; PCA stored before row normalization'
    contract['probe_row_normalization'] = PROBE_ROW_NORMALIZATION
    contract['baseline_cache_schema_version'] = 2
    contract['masked_count_prediction'] = dict(seed=42, epoch=0, query_genes=int(cfg.get('v12', {}).get('query_genes', 64)),
        both_center_compartments_query_hidden=True, gene_mask_ratio=0., neighbor_dropout=0.,
        target_counts_passed_to_forward=False, decoder='NB(total)*beta-binomial(nuclear|total)',
        interpretation='Held-out reconstruction, not temporal direction validation')
    contract = json.loads(json.dumps(contract))
    contract_path = args.output_dir / 'cache_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Output belongs to a different checkpoint/source/cohort/protocol')
    else:
        # Watcher-created failure artifacts are not a license to reuse ambiguous caches.
        if any(args.output_dir.iterdir()):
            raise ValueError('Nonempty output lacks verified cache contract')
        atomic_json(contract_path, contract)
    frames, targets, coverage_blocks, qa, masked_frames = [], [], [], [], []
    blocks = {name: [] for name in MODEL_READOUTS + BASELINES}
    for index, directory, obs, split in metadata:
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy(); chosen['split'] = split
        cache = args.output_dir / 'slices' / directory.name
        cache.mkdir(parents=True, exist_ok=True)
        sample_contract = dict(cell_ids_sha256=ids_hash(chosen.full_cell_id), **sample_inputs(directory))
        if (cache / 'complete.json').exists():
            entry = json.loads((cache / 'complete.json').read_text())
            if entry['contract'] != sample_contract:
                raise ValueError('Slice cache input mismatch')
            values = {name: np.load(cache / f'{name}.npy', allow_pickle=False) for name in blocks}
            item_qa = entry['qa']
        else:
            values, item_qa = extract(model, baseline, cfg, directory, obs, selected, args.batch_size, index, args.baseline_cache_dir, step=step)
            for name, value in values.items():
                atomic_array(cache / f'{name}.npy', value)
            atomic_json(cache / 'complete.json', dict(contract=sample_contract, qa=item_qa))
        for name, value in values.items():
            if value.ndim != 2 or len(value) != len(selected) or not np.isfinite(value).all():
                raise ValueError('Invalid checkpoint-dependent embedding cache')
            blocks[name].append(value)
        count_frame = pd.DataFrame(item_qa['masked_count_rows'])
        count_frame['full_cell_id'] = obs.iloc[count_frame.center_row.to_numpy()].full_cell_id.to_numpy()
        count_frame['patient'] = obs.iloc[count_frame.center_row.to_numpy()].patient.to_numpy()
        count_frame['split'] = split
        if count_frame.full_cell_id.isna().any() or count_frame.duplicated(['full_cell_id', 'mode']).any():
            raise ValueError('Invalid held-out count cell identities')
        masked_frames.append(count_frame)
        target, coverage = neighbor_composition(obs, selected, classes, args.composition_radius_um)
        targets.append(target); coverage_blocks.append(coverage); frames.append(chosen); qa.append(item_qa)
        atomic_json(args.output_dir / 'progress.json', dict(stage='embedding', step=step,
            completed_slices=len(frames), total_slices=len(metadata), last_sample=directory.name, seconds=time.monotonic()-started))
    obs = pd.concat(frames, ignore_index=True)
    cohort = ids_hash(obs.full_cell_id)
    train_donors = sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique().tolist())
    val_donors = sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique().tolist())
    if cohort != CANONICAL_COHORT or len(obs) != 5200 or not obs.full_cell_id.is_unique or len(train_donors) != 10 or len(val_donors) != 3:
        raise ValueError(f'Not canonical train10/val3 cohort: {cohort}')
    features = {name: np.concatenate(value) for name, value in blocks.items()}
    features['concat_pca512'], pca = concat_pca(features['concat'], obs['split'].eq('train').to_numpy())
    target = np.concatenate(targets)
    coverage = {key: np.concatenate([block[key] for block in coverage_blocks]) for key in coverage_blocks[0]}
    write_frame(args.output_dir / 'cells.parquet', obs)
    masked_frame = pd.concat(masked_frames, ignore_index=True)
    write_frame(args.output_dir / 'masked_count_per_cell.parquet', masked_frame)
    masked_summary = aggregate_masked_counts(masked_frame)
    atomic_array(args.output_dir / 'neighbor_composition_targets.npy', target)
    atomic_json(args.output_dir / 'neighbor_composition_classes.json', classes)
    for name, values in features.items():
        atomic_array(args.output_dir / f'{name}_embeddings.npy', values)
    del model, baseline
    torch.cuda.empty_cache()
    diagnostics = representation_diagnostics(obs, features)
    atomic_json(args.output_dir / 'diagnostics.json', diagnostics)
    probe_features = {name: normalize_for_probe(value) for name, value in features.items()}
    probe_dir = args.output_dir / 'probes'; probe_dir.mkdir(exist_ok=True)
    results = []
    for name in READOUTS:
        for task in TASKS:
            cache = probe_dir / f'{name}_{task}.json'
            if cache.is_file():
                row = json.loads(cache.read_text())
                if row['embedding'] != name or row['task'] != task or not (args.output_dir / f'{name}_{task}_val_predictions.parquet').is_file():
                    raise ValueError('Invalid cached probe')
            else:
                row = (classification_probe(obs, probe_features[name], name, task, args.output_dir) if task in FIXED_C
                       else composition_probe(obs, probe_features[name], target, coverage, classes, name, args.output_dir))
                atomic_json(cache, row)
            results.append(row)
            atomic_json(args.output_dir / 'progress.json', dict(stage='probing', step=step,
                completed_probes=len(results), total_probes=len(READOUTS)*len(TASKS), seconds=time.monotonic()-started))
    composition_qa = {split: {key: float(value[obs['split'].eq(split).to_numpy()].mean()) for key, value in coverage.items()}
                      for split in ('train', 'val')}
    report = dict(status='complete', model_version=12, architecture=ARCHITECTURE, variant=cfg.get('experiment', 'v12'), step=step,
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha, **provenance, config=cfg,
        cells=len(obs), cohort_sha256=cohort, canonical_cohort=True, no_test=True, test_donors_used=False,
        train_donors=train_donors, val_donors=val_donors, fixed_C=FIXED_C, seed=42,
        protocol_sha256=contract['protocol_sha256'], contract_sha256=fingerprint(contract),
        evaluator_sha256=contract['evaluator_sha256'], solver_sha256=contract['solver_sha256'],
        input_signatures_sha256=fingerprint(contract['input_signatures']),
        dimensions={name: value.shape[1] for name, value in features.items()}, pca=pca,
        baseline_protocol='Separately loaded original ordinary scGPT; unmasked deterministic-left whole/nuclear/cytoplasmic raw CLS; never teacher; scores recomputed in current protocol, not copied from prior experiments',
        masked_count_prediction=dict(**contract['masked_count_prediction'], results=masked_summary),
        embedding_storage=contract['embedding_storage'], diagnostics_input='raw stored readouts before probe row normalization',
        probe_row_normalization=contract['probe_row_normalization'],
        probe_protocol='Common row L2 for all ten readouts, then train-only StandardScaler; balanced logistic fixed C; Ridge alpha=1 then simplex for neighborhood compositions; no hyperparameter search',
        composition_protocol=dict(radius_um=args.composition_radius_um, center_excluded=True, labels='external existing final_CT',
            class_vocabulary='all nonmissing labels in train donors only', classes=classes, coverage=composition_qa,
            unknown_label_sentinels=sorted(UNKNOWN_COMPOSITION_LABELS),
            interpretation='Environmental information readability; input and target neighbors can overlap, not held-out neighbor prediction',
            missing_unseen='Excluded from normalized known-label composition, explicitly counted in coverage'),
        inference=dict(mask=False, dtype='bfloat16', batch_size=args.batch_size, no_checkpoint_shared_trainable_encoder_cache=True),
        gene_coverage=[{key: value for key, value in item.items() if key != 'masked_count_rows'} for item in qa],
        diagnostics=diagnostics, results=results, seconds=time.monotonic()-started)
    validate_completion(report, args.output_dir)
    atomic_json(args.output_dir / 'metrics.json', report)
    atomic_json(args.output_dir / 'progress.json', dict(stage='complete', step=step, metrics='metrics.json'))
    print(f'[done-v12] step={step} all {len(READOUTS)} readouts × {len(TASKS)} tasks complete', flush=True)


if __name__ == '__main__':
    main()
