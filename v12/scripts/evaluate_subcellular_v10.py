"""V10 frozen probes and view diagnostics on the canonical 5,200-cell cohort.

Train 10 / validation 3 donors, 400 centers per donor, no test files opened.
Fixed balanced-logistic C: cell type 0.1; region 0.01. No probe hyperparameter
search. Embeddings are unmasked; shuffled neighbors retain the real coordinates.
Only the final atomic metrics.json marks successful completion. Per-slide and
per-probe caches permit safe restart, with checkpoint/source/cohort fingerprints.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from evaluate_subcellular_v2 import CANONICAL_COHORT, atomic_array, atomic_json, file_sha256
from evaluate_subcellular_v7 import measured_visibility, shuffle_expression_nodes
from probe_lungpf_embeddings import fit_linear, predict
from spatial_pretrain.data.subcellular_v10 import collate_v10, encode_v10_views
from spatial_pretrain.models.subcellular_v10 import build_model

MODEL_VERSION = 10
ARCHITECTURE = 'nucleus_neighbor_teacher_v10'
READOUTS = ('base', 'nucleus', 'context', 'joint', 'base_augmented')
SHUFFLED = ('context', 'joint', 'base_augmented')
ALL_READOUTS = READOUTS + tuple(f'{name}_neighbor_shuffle' for name in SHUFFLED)
FIXED_C = {'cell_type': .1, 'region': .01}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def ids_hash(values):
    return hashlib.sha256('\n'.join(map(str, values)).encode()).hexdigest()


def write_frame(path, frame):
    temporary = path.with_name(path.name + '.partial')
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def load_model(path, expected_step):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('model_version') != MODEL_VERSION or saved.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected model_version=10, architecture=nucleus_neighbor_teacher_v10')
    step = int(saved['step'])
    if step < 0 or (expected_step is not None and step != expected_step):
        raise ValueError(f'Unexpected checkpoint step: {step} (requested {expected_step})')
    cfg = saved['config']
    if saved.get('config_sha256') != fingerprint(cfg):
        raise ValueError('Missing or incorrect checkpoint config_sha256')
    sources = saved.get('source_sha256')
    required = {'src/spatial_pretrain/models/subcellular_v10.py',
                'src/spatial_pretrain/data/subcellular_v10.py'}
    if not isinstance(sources, dict) or not required.issubset(sources):
        raise ValueError('Checkpoint must record the V10 model/data dependency source hashes')
    for name, digest in sources.items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or file_sha256(source) != digest:
            raise ValueError(f'Checkpoint source does not match evaluator checkout: {name}')
    model = build_model(cfg)
    loaded_provenance = dict(model.backbone.provenance)
    if loaded_provenance.get('backbone_kind') != 'scgpt':
        raise ValueError('V10 base must be the ordinary scGPT pretrained encoder')
    recorded_provenance = saved.get('backbone_provenance')
    if recorded_provenance is not None and recorded_provenance != loaded_provenance:
        raise ValueError('Checkpoint backbone provenance differs from the loaded ordinary scGPT assets')
    for name, tensor in model.backbone.state_dict().items():
        archived = saved['model'].get(f'backbone.{name}')
        if archived is None or not torch.equal(tensor.detach().cpu(), archived.detach().cpu()):
            raise ValueError(f'Frozen baseline backbone differs from its pretrained asset: {name}')
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise ValueError('V10 teacher/backbone must be frozen')
    model.load_state_dict(saved['model'], strict=True)
    provenance = dict(config_sha256=fingerprint(cfg), source_sha256=sources,
                      sources_verified=True, backbone_provenance=loaded_provenance,
                      frozen_backbone_matches_pretrained=True)
    del saved
    return model.eval().requires_grad_(False).cuda(), cfg, step, provenance


def frozen_inputs(model, cfg, directory, obs, selected, nodes, expression, coords, nuclear, cytoplasm,
                  keep, global_ids, batch_size, slice_index, slice_seed, cache_root, condition):
    """Checkpoint-independent frozen raw CLS cache, protected by a file lock.

    No learned predictor/normalization is cached here. Conditions have separate
    entries because expression sources differ while physical coordinates agree.
    """
    settings = cfg.get('v10', {})
    neighbors, max_tokens = int(settings.get('neighbors', 8)), int(settings.get('max_tokens', 512))
    dependencies = ('src/spatial_pretrain/models/subcellular_v10.py',
                    'src/spatial_pretrain/models/scgpt_backbone.py',
                    'src/spatial_pretrain/data/subcellular_v10.py',
                    'src/spatial_pretrain/data/subcellular_v6_left.py',
                    'src/spatial_pretrain/data/subcellular_v5.py',
                    'scripts/evaluate_subcellular_v7.py')
    contract = dict(schema_version=2, condition=condition, sample=directory.name,
        prepared=str(directory.parent.resolve()), backbone=dict(model.backbone.provenance),
        sources={name: file_sha256(ROOT / name) for name in dependencies},
        cell_ids_sha256=ids_hash(obs.iloc[selected].full_cell_id),
        physical_nodes_sha256=hashlib.sha256(nodes.tobytes()).hexdigest(),
        expression_nodes_sha256=hashlib.sha256(expression.tobytes()).hexdigest(),
        coords_sha256=hashlib.sha256(coords[nodes].tobytes()).hexdigest(),
        genes_sha256=file_sha256(directory / 'genes.npy'),
        counts={name: dict(size=(directory / name).stat().st_size, mtime_ns=(directory / name).stat().st_mtime_ns)
                for name in ('nuclear.npy', 'cytoplasmic.npy')},
        seed=42, slice_seed=slice_seed, neighbors=neighbors, max_tokens=max_tokens, mask=False,
        encoding='raw frozen CLS before heads or L2 normalization', torch_version=torch.__version__,
        dtype='bfloat16 autocast, float32 stored', batch_size=batch_size,
        encoder_chunk_size=model.encoder_chunk_size)
    contract = json.loads(json.dumps(contract))
    directory_cache = cache_root / fingerprint(contract)
    directory_cache.mkdir(parents=True, exist_ok=True)
    fields = ('teacher_features', 'nucleus_features', 'neighbor_features',
              'nucleus_present', 'neighbor_present', 'relative_coords')
    cache_started = time.monotonic()
    with (directory_cache / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        complete = directory_cache / 'complete.json'
        if complete.exists():
            saved = json.loads(complete.read_text())
            if saved['contract'] != contract:
                raise ValueError('Frozen feature cache contract mismatch')
            arrays = {key: np.load(directory_cache / f'{key}.npy', allow_pickle=False) for key in fields}
            reused = True
        else:
            blocks = {key: [] for key in fields}
            for start in range(0, len(nodes), batch_size):
                end = min(start + batch_size, len(nodes))
                items = []
                for index in range(start, end):
                    row = expression[index]
                    raw_n = np.asarray(nuclear[np.ix_(row, keep)], dtype=np.float32)
                    raw_c = np.asarray(cytoplasm[np.ix_(row, keep)], dtype=np.float32)
                    items.append(encode_v10_views(
                        raw_n, raw_c, global_ids, row, coords[nodes[index]],
                        pad_token_id=model.vocab['<pad>'], cls_token_id=model.vocab['<cls>'],
                        neighbors=neighbors, max_tokens=max_tokens, seed=42, epoch=0,
                        slice_seed=slice_seed, slice_index=slice_index, mask=False))
                batch = {key: value.cuda(non_blocking=True) for key, value in collate_v10(items).items()}
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    values = model.encode_views(batch)
                values.update(nucleus_present=batch['nucleus_present'].bool(),
                              neighbor_present=batch['neighbor_present'].bool(), relative_coords=batch['relative_coords'])
                for key, value in values.items():
                    blocks[key].append(value.detach().cpu().numpy())
                if start == 0 or end == len(nodes):
                    print(f'[frozen-v10] {directory.name} {condition} {end}/{len(nodes)} '
                          f'{time.monotonic() - cache_started:.1f}s', flush=True)
            arrays = {key: np.concatenate(value) for key, value in blocks.items()}
            for key, value in arrays.items():
                if not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite frozen features in {key}')
                atomic_array(directory_cache / f'{key}.npy', value)
            atomic_json(complete, dict(contract=contract, seconds=time.monotonic() - cache_started))
            reused = False
    shapes = dict(teacher_features=(len(nodes), model.dim), nucleus_features=(len(nodes), model.dim),
                  neighbor_features=(len(nodes), neighbors, model.dim), nucleus_present=(len(nodes),),
                  neighbor_present=(len(nodes), neighbors),
                  relative_coords=(len(nodes), neighbors, 2))
    if any(arrays[key].shape != shapes[key] or not np.isfinite(arrays[key]).all() for key in fields):
        raise ValueError('Invalid shape or values in frozen feature cache')
    return arrays, dict(path=str(directory_cache), reused=reused, seconds=time.monotonic() - cache_started)


def extract(model, cfg, directory, obs, selected, batch_size, slice_index, cache_root):
    started = time.monotonic()
    vocab = model.vocab
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not obs.full_cell_id.is_unique:
        raise ValueError(f'Invalid coordinates or duplicate cell IDs: {directory.name}')
    settings = cfg.get('v10', {})
    requested_neighbors = int(settings.get('neighbors', 8))
    if requested_neighbors != model.neighbors:
        raise ValueError('V10 data and model neighbor counts differ')
    if requested_neighbors < 1 or len(obs) < 2:
        raise ValueError('Spatial evaluation requires at least one noncentral neighbor')
    count = min(requested_neighbors + 1, len(obs))
    nearest = NearestNeighbors(n_neighbors=count).fit(coords).kneighbors(
        coords[selected], return_distance=False)
    nodes = np.stack([np.r_[center, row[row != center][:count - 1]]
                      for center, row in zip(selected, nearest)])
    if nodes.shape != (len(selected), count) or any(
            len(np.unique(row)) != count or row[0] in row[1:] for row in nodes):
        raise ValueError('Center exclusion/unique neighbor invariant failed')
    genes = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([i for i, gene in enumerate(genes) if gene in vocab], dtype=np.int64)
    if not len(keep):
        raise ValueError(f'No genes match model vocabulary: {directory.name}')
    global_ids = np.array([vocab[gene] for gene in genes[keep]], dtype=np.int64)
    nuclear = np.load(directory / 'nuclear.npy', mmap_mode='r', allow_pickle=False)
    cytoplasm = np.load(directory / 'cytoplasmic.npy', mmap_mode='r', allow_pickle=False)
    if nuclear.shape != cytoplasm.shape or nuclear.shape != (len(obs), len(genes)):
        raise ValueError('Expression/observations/panel shape mismatch')
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    visibility = measured_visibility(nuclear, cytoplasm, keep)
    shuffled = shuffle_expression_nodes(nodes, visibility, np.random.SeedSequence([42, slice_seed, 815]))
    if not np.array_equal(shuffled[:, 0], selected) or (shuffled[:, 1:] == selected[:, None]).any():
        raise ValueError('Neighbor shuffle leaks central expression')
    features = {key: [] for key in ALL_READOUTS}
    frozen_cache_qa = {}
    for condition, expression in (('real', nodes), ('neighbor_shuffle', shuffled)):
        frozen, cache_qa = frozen_inputs(model, cfg, directory, obs, selected, nodes, expression, coords,
            nuclear, cytoplasm, keep, global_ids, batch_size, slice_index, slice_seed, cache_root, condition)
        frozen_cache_qa[condition] = cache_qa
        for start in range(0, len(nodes), batch_size):
            end = min(start + batch_size, len(nodes))
            batch = {key: torch.as_tensor(value[start:end], device='cuda') for key, value in frozen.items()}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                encoded = model(batch, return_embeddings=True)['embeddings']
            converted = {}
            for key in READOUTS:
                value = encoded[key].float().cpu().numpy()
                if value.ndim != 2 or len(value) != end - start or not np.isfinite(value).all():
                    raise ValueError(f'Invalid {key} embeddings {value.shape}')
                if key == 'base':
                    value = value / np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
                converted[key] = value
            if condition == 'real':
                for key in READOUTS:
                    features[key].append(converted[key])
            else:
                # These views cannot depend on neighbor expression. Verify this
                # rather than silently assuming the model obeys its contract.
                block = start // batch_size
                for key in ('base', 'nucleus'):
                    if not np.allclose(converted[key], features[key][block], atol=1e-5, rtol=1e-5):
                        raise ValueError(f'{key} unexpectedly changed under neighbor-expression shuffle')
                for key in SHUFFLED:
                    features[f'{key}_neighbor_shuffle'].append(converted[key])
            if start == 0 or end == len(nodes):
                print(f'[embed-v10] {directory.name} {condition} {end}/{len(nodes)} '
                      f'{time.monotonic() - started:.1f}s', flush=True)
    result = {key: np.concatenate(value) for key, value in features.items()}
    base_dim = result['base'].shape[1]
    if any(result[key].shape[1] != base_dim for key in ('nucleus', 'context', 'joint')):
        raise ValueError('Base and new primary branches must have the same embedding dimension')
    qa = dict(sample=directory.name, selected_cells=len(selected), panel_genes=len(genes),
              matched_genes=len(keep), neighbors_requested=requested_neighbors,
              neighbors_actual=count - 1, center_excluded=True, neighbor_shuffle_center_excluded=True,
              neighbor_shuffle_coordinates_unchanged=True, mask=False,
              neighbor_expression_changed_fraction=float(np.mean(nodes[:, 1:] != shuffled[:, 1:])),
              frozen_cache=frozen_cache_qa,
              expression_sources={name: dict(size=(directory / name).stat().st_size,
                                            mtime_ns=(directory / name).stat().st_mtime_ns)
                                  for name in ('nuclear.npy', 'cytoplasmic.npy')},
              seconds=time.monotonic() - started)
    return result, qa


def fixed_probe(obs, features, embedding, task, output):
    started = time.monotonic()
    column = {'cell_type': 'final_CT', 'region': 'region_label'}[task]
    labels = obs[column]
    usable = labels.notna() & ~labels.astype(str).isin(['', 'nan', 'None'])
    train, val = usable & obs['split'].eq('train'), usable & obs['split'].eq('val')
    classes = np.unique(labels[train].astype(str))
    if len(classes) < 2 or not val.any():
        raise ValueError(f'Insufficient labels for {task}')
    lookup = {name: index for index, name in enumerate(classes)}
    scaler = StandardScaler().fit(features[train])
    tx = torch.as_tensor(scaler.transform(features[train]), device='cuda', dtype=torch.float32)
    vx = torch.as_tensor(scaler.transform(features[val]), device='cuda', dtype=torch.float32)
    y = torch.tensor([lookup[name] for name in labels[train].astype(str)], device='cuda')
    weights, bias, info = fit_linear(tx, y, FIXED_C[task])
    prediction = predict(vx, weights, bias, classes)
    true = labels[val].astype(str).to_numpy()
    donors = obs.loc[val, 'patient'].astype(str).to_numpy()
    donor_scores = {}
    for donor in np.unique(donors):
        which = donors == donor
        donor_scores[str(donor)] = dict(
            cells=int(which.sum()), macro_f1=float(f1_score(true[which], prediction[which],
                labels=np.unique(true[which]), average='macro', zero_division=0)),
            balanced_accuracy=float(balanced_accuracy_score(true[which], prediction[which])))
    score = dict(C=FIXED_C[task], val_macro_f1=float(f1_score(true, prediction,
        labels=np.unique(true), average='macro', zero_division=0)),
        val_balanced_accuracy=float(balanced_accuracy_score(true, prediction)),
        per_donor_macro_f1={key: value['macro_f1'] for key, value in donor_scores.items()},
        per_donor=donor_scores, solver=info)
    frame = obs.loc[val, ['full_cell_id', 'patient']].copy()
    frame['label'], frame['prediction'] = true, prediction
    write_frame(output / f'{embedding}_{task}_val_predictions.parquet', frame)
    row = dict(embedding=embedding, task=task, embedding_dimension=int(features.shape[1]),
        train_cells=int(train.sum()), val_cells=int(val.sum()), unseen_val_classes=sorted(set(true) - set(classes)),
        fixed_C_result=score, seconds=time.monotonic() - started)
    del tx, vx, y, weights, bias
    torch.cuda.empty_cache()
    print(f'[probe-v10] {embedding} {task} C={FIXED_C[task]} macroF1={score["val_macro_f1"]:.6f}', flush=True)
    return row


def shared_baseline_probe(obs, features, task, output, cache_root):
    """Fit the exactly identical frozen baseline once across all checkpoints."""
    contract = dict(schema_version=1, cohort_sha256=ids_hash(obs.full_cell_id),
        labels_sha256=fingerprint(obs[['patient', 'split', 'final_CT', 'region_label']].astype('string').fillna('').to_dict('list')),
        features_sha256=hashlib.sha256(np.ascontiguousarray(features).tobytes()).hexdigest(),
        C=FIXED_C[task], task=task, torch_version=torch.__version__,
        solver_sha256=file_sha256(ROOT / 'scripts/probe_lungpf_embeddings.py'),
        evaluator_sha256=file_sha256(Path(__file__)))
    shared = cache_root / 'baseline_probes' / fingerprint(contract)
    shared.mkdir(parents=True, exist_ok=True)
    prediction_name = f'base_{task}_val_predictions.parquet'
    with (shared / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (shared / 'complete.json').exists():
            cached = json.loads((shared / 'complete.json').read_text())
            if cached['contract'] != contract or not (shared / prediction_name).exists():
                raise ValueError('Invalid shared frozen-baseline probe cache')
            row = cached['result']
            reused = True
        else:
            row = fixed_probe(obs, features, 'base', task, shared)
            atomic_json(shared / 'complete.json', dict(contract=contract, result=row))
            reused = False
        temporary = output / (prediction_name + '.partial')
        shutil.copyfile(shared / prediction_name, temporary)
        temporary.replace(output / prediction_name)
    return dict(row, shared_frozen_baseline_cache=dict(path=str(shared), reused=reused))


def unit_rows(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def representation_diagnostics(obs, features):
    """Use held-out validation donors only; rank galleries are within donor."""
    val = obs['split'].eq('val').to_numpy()
    selected = {key: np.asarray(value[val], dtype=np.float64) for key, value in features.items()}
    diagnostics = dict(split='val', cells=int(val.sum()), collapse={}, cosine={}, context_retrieval={})
    for name, values in selected.items():
        centered = values - values.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / max(1, len(values) - 1)).clip(0)
        total = float(eigenvalues.sum())
        probabilities = eigenvalues[eigenvalues > 0] / total if total > 0 else np.array([])
        diagnostics['collapse'][name] = dict(
            mean_feature_variance=float(np.var(values, axis=0, ddof=1).mean()),
            mean_feature_std=float(np.std(values, axis=0, ddof=1).mean()),
            effective_rank=float(np.exp(-(probabilities * np.log(probabilities)).sum())) if len(probabilities) else 0.,
            effective_rank_definition='exp(entropy(normalized covariance eigenvalues))',
            embedding_dimension=int(values.shape[1]),
            zero_norm_cells=int((np.linalg.norm(values, axis=1) < 1e-12).sum()))
    for first, second in (('nucleus', 'base'), ('context', 'base'), ('nucleus', 'context'),
                          ('joint', 'base'), ('context_neighbor_shuffle', 'base')):
        dots = (unit_rows(selected[first]) * unit_rows(selected[second])).sum(axis=1)
        diagnostics['cosine'][f'{first}_to_{second}'] = dict(
            mean=float(dots.mean()), std=float(dots.std()), median=float(np.median(dots)))
    donors = obs.loc[val, 'patient'].astype(str).to_numpy()
    for name in ('nucleus', 'context', 'context_neighbor_shuffle'):
        by_donor = {}
        ranks_all, rr_all, top1_all, top10_all = [], [], [], []
        for donor in np.unique(donors):
            mask = donors == donor
            scores = unit_rows(selected[name][mask]) @ unit_rows(selected['base'][mask]).T
            matched = np.diag(scores)
            greater = (scores > matched[:, None] + 1e-6).sum(axis=1)
            tied = (np.abs(scores - matched[:, None]) <= 1e-6).sum(axis=1)
            # Expected rank and reciprocal rank under uniform tie breaking;
            # complete collapse therefore yields chance rather than top1=1.
            lower, upper = greater + 1, greater + tied
            rank = (lower + upper) / 2.
            harmonic = np.r_[0., np.cumsum(1. / np.arange(1, len(scores) + 1))]
            reciprocal = (harmonic[upper] - harmonic[lower - 1]) / tied
            top1 = np.clip(1 - greater, 0, tied) / tied
            top10 = np.clip(10 - greater, 0, tied) / tied
            by_donor[str(donor)] = dict(gallery_cells=len(scores), matched_cosine=float(matched.mean()),
                mean_rank=float(rank.mean()), median_rank=float(np.median(rank)),
                mean_reciprocal_rank=float(reciprocal.mean()), top1=float(top1.mean()), top10=float(top10.mean()),
                chance_top1=1. / len(scores), chance_top10=min(10, len(scores)) / len(scores))
            ranks_all.extend(rank.tolist()); rr_all.extend(reciprocal.tolist())
            top1_all.extend(top1.tolist()); top10_all.extend(top10.tolist())
        diagnostics['context_retrieval'][name] = dict(per_donor=by_donor,
            mean_rank=float(np.mean(ranks_all)), mean_reciprocal_rank=float(np.mean(rr_all)),
            top1=float(np.mean(top1_all)), top10=float(np.mean(top10_all)),
            protocol='Within-donor center matching by cosine to frozen whole-cell base; uniform tie breaking at 1e-6')
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', '--output', required=True, type=Path)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/benchmarking_data/lungpf/prepared')
    parser.add_argument('--frozen-cache-dir', type=Path, default=ROOT / 'outputs/subcellular_v10_frozen_cache')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'metrics.json').exists():
        raise FileExistsError('metrics.json already exists; refusing to overwrite completed or ambiguous results')
    started = time.monotonic()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint_sha = file_sha256(args.checkpoint)
    model, cfg, step, provenance = load_model(args.checkpoint, args.expected_step)
    protocol_path = args.prepared / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    variant = cfg['loss']['type']
    if variant not in ('cosine', 'contrastive', 'relational'):
        raise ValueError(f'Unknown V10 loss variant {variant}')
    contract = dict(schema_version=10, checkpoint_sha256=checkpoint_sha, step=step,
                    config_sha256=provenance['config_sha256'], source_sha256=provenance['source_sha256'],
                    evaluator_sha256=file_sha256(Path(__file__)), protocol_sha256=file_sha256(protocol_path),
                    prepared=str(args.prepared.resolve()), canonical_cohort=CANONICAL_COHORT,
                    fixed_C=FIXED_C, readouts=ALL_READOUTS, seed=42, mask=False)
    # Round-trip converts tuples to lists, making resume contract equality exact.
    contract = json.loads(json.dumps(contract))
    contract_path = args.output_dir / 'cache_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Output cache belongs to a different checkpoint/source/protocol')
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError('Nonempty output has no cache contract; use a fresh directory')
        atomic_json(contract_path, contract)
    frames, blocks, slice_qa = [], {name: [] for name in ALL_READOUTS}, []
    for slice_index, sample in enumerate(protocol['samples']):
        matches = [donor for donor in protocol['donor_split'] if sample.startswith(donor)]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous donor before opening sample {sample}')
        donor = matches[0]
        split = protocol['donor_split'][donor]
        if split == 'test':
            continue
        if split not in ('train', 'val'):
            raise ValueError(f'Unexpected split {split}')
        directory = args.prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        if not obs.patient.eq(donor).all() or ('split' in obs and not obs['split'].eq(split).all()):
            raise ValueError(f'Observation/protocol mismatch in {sample}')
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy()
        chosen['split'] = split
        cache = args.output_dir / 'slices' / sample
        cache.mkdir(parents=True, exist_ok=True)
        sample_contract = dict(cell_ids_sha256=ids_hash(chosen.full_cell_id),
            obs_sha256=file_sha256(directory / 'obs.parquet'), genes_sha256=file_sha256(directory / 'genes.npy'),
            counts={name: dict(size=(directory / name).stat().st_size, mtime_ns=(directory / name).stat().st_mtime_ns)
                    for name in ('nuclear.npy', 'cytoplasmic.npy')})
        if (cache / 'complete.json').exists():
            entry = json.loads((cache / 'complete.json').read_text())
            if entry['contract'] != sample_contract:
                raise ValueError(f'Cached sample input changed: {sample}')
            values = {name: np.load(cache / f'{name}.npy', allow_pickle=False) for name in ALL_READOUTS}
            qa = entry['qa']
        else:
            values, qa = extract(model, cfg, directory, obs, selected, args.batch_size, slice_index,
                                 args.frozen_cache_dir)
            for name, value in values.items():
                atomic_array(cache / f'{name}.npy', value)
            atomic_json(cache / 'complete.json', dict(contract=sample_contract, qa=qa))
        for name, value in values.items():
            if value.ndim != 2 or len(value) != len(chosen) or not np.isfinite(value).all():
                raise ValueError(f'Invalid cached features for {sample}/{name}')
            blocks[name].append(value)
        frames.append(chosen)
        slice_qa.append(qa)
        atomic_json(args.output_dir / 'progress.json', dict(stage='embedding', completed_slices=len(frames),
            last_sample=sample, step=step, elapsed_seconds=time.monotonic() - started))
    obs = pd.concat(frames, ignore_index=True)
    cohort = ids_hash(obs.full_cell_id)
    train_donors = sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique().tolist())
    val_donors = sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique().tolist())
    if cohort != CANONICAL_COHORT or len(obs) != 5200 or not obs.full_cell_id.is_unique or len(train_donors) != 10 or len(val_donors) != 3:
        raise ValueError(f'Not the canonical 5200-cell train10/val3 cohort: {cohort}')
    features = {name: np.concatenate(values) for name, values in blocks.items()}
    write_frame(args.output_dir / 'cells.parquet', obs)
    for name, values in features.items():
        atomic_array(args.output_dir / f'{name}_embeddings.npy', values)
    del model
    torch.cuda.empty_cache()
    diagnostics = representation_diagnostics(obs, features)
    atomic_json(args.output_dir / 'diagnostics.json', diagnostics)
    results = []
    probe_dir = args.output_dir / 'probes'
    probe_dir.mkdir(exist_ok=True)
    for name in ALL_READOUTS:
        for task in FIXED_C:
            cache = probe_dir / f'{name}_{task}.json'
            if cache.exists():
                row = json.loads(cache.read_text())
                if row['embedding'] != name or row['task'] != task or row['fixed_C_result']['C'] != FIXED_C[task]:
                    raise ValueError(f'Invalid cached probe {cache}')
                if not (args.output_dir / f'{name}_{task}_val_predictions.parquet').exists():
                    raise ValueError(f'Cached probe lacks predictions: {cache}')
            else:
                row = (shared_baseline_probe(obs, features[name], task, args.output_dir, args.frozen_cache_dir)
                       if name == 'base' else fixed_probe(obs, features[name], name, task, args.output_dir))
                atomic_json(cache, row)
            results.append(row)
            atomic_json(args.output_dir / 'progress.json', dict(stage='probing', completed_probes=len(results),
                total_probes=len(ALL_READOUTS) * len(FIXED_C), step=step, elapsed_seconds=time.monotonic() - started))
    report = dict(status='complete', model='SubcellularV10', model_version=MODEL_VERSION,
        architecture=ARCHITECTURE, variant=variant, step=step, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=checkpoint_sha, **provenance, config=cfg,
        cells=len(obs), cohort_sha256=cohort, canonical_cohort=True, no_test=True, test_donors_used=False,
        train_donors=train_donors, val_donors=val_donors, fixed_C=FIXED_C,
        protocol_sha256=contract['protocol_sha256'], seed=42, selected_cells_per_donor=400,
        dimensions={name: int(value.shape[1]) for name, value in features.items()},
        base_protocol='Frozen whole-cell ordinary-scGPT teacher CLS, L2-normalized; identical teacher input fields',
        primary_comparison='base, nucleus, context and joint have equal dimension; base_augmented is a separate capacity comparison',
        probe_protocol='Train-only StandardScaler; class-balanced logistic fixed C; unseen validation classes retained; no hyperparameter search',
        inference=dict(mask=False, dtype='bfloat16', batch_size=args.batch_size, center_excluded_from_neighbors=True),
        gene_coverage=slice_qa, diagnostics=diagnostics, results=results,
        seconds=time.monotonic() - started,
        timing_note='seconds is this invocation wall time; gene_coverage/probe seconds retain original computation times on resume')
    atomic_json(args.output_dir / 'metrics.json', report)
    atomic_json(args.output_dir / 'progress.json', dict(stage='complete', step=step, metrics='metrics.json'))
    print(f'[done-v10] {variant} step={step} metrics={args.output_dir / "metrics.json"}', flush=True)


if __name__ == '__main__':
    main()
