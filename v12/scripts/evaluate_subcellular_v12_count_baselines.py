"""Trivial train-donor baselines for V12 held-out joint count prediction.

Same canonical 5,200-cell cohort, same fixed query draw (seed 42, epoch 0, 64
genes hidden in both centre compartments, no extra gene masks, no neighbour
dropout) and the same metric/NLL code as evaluate_subcellular_v12.py. Every
baseline parameter is fitted on TRAIN donors only and predictions use only the
visible (non-query) counts a model would also receive. No checkpoint or GPU is
needed; --checkpoint optionally re-scores V12 through this very code path and
splits its NLL into NB-total and allocation parts. Atomic metrics.json is the
completion marker. Run on a compute allocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import torch

import evaluate_subcellular_v12 as ev12

EXPERIMENT = 'subcellular_v12_count_baselines'
PARAMETER_KEYS = ('mu_total', 'theta_total', 'p_nucleus', 'kappa_allocation')
# JointCountDecoder output ranges (models/subcellular_v12.py:105-108): baselines stay inside the model's family.
BOUNDS = dict(mu_total=(1e-4, 1e7), theta_total=(1e-3, 1e5), p_nucleus=(1e-4, 1 - 1e-4), kappa_allocation=(2e-3, 1e5))
METRICS = ('nll', 'total_abs_error', 'nucleus_abs_error', 'cytoplasm_abs_error', 'allocation_abs_error')
PART_METRICS = ('nll_total_part', 'nll_allocation_part', 'allocation_nll_per_positive')
# Not an error of its own: E|p - N/T| if the row's own (p, kappa) were the true allocation law. No skill is computed.
DIAGNOSTICS = ('allocation_abs_error_calibrated',)
COUNT_FIELDS = ('query_count', 'positive_query_count', 'available_cells')
SUM_FIELDS = ('nll_sum', 'total_abs_error_sum', 'nucleus_abs_error_sum', 'cytoplasm_abs_error_sum', 'allocation_abs_error_sum')
TOTALS = ('t_zero', 't_cell_mean', 't_gene_mean', 't_gene_depth', 't_gene_depth_nbr')
ALLOCATIONS = ('a_global', 'a_cell', 'a_gene', 'a_gene_cell', 'a_gene_median')
# Estimators whose mean carries no gene identity also get ONE global ML dispersion, so their NLL stays gene-free.
GENE_FREE = ('t_zero', 't_cell_mean', 'a_global', 'a_cell')
CENTER = frozenset(('center_nucleus', 'center_cytoplasm'))
ESTIMATOR_INPUTS = dict(t_zero=frozenset(), t_cell_mean=CENTER, t_gene_mean=frozenset(), t_gene_depth=CENTER,
    t_gene_depth_nbr=CENTER | {'neighbors'}, a_global=frozenset(), a_cell=CENTER, a_gene=frozenset(),
    a_gene_cell=CENTER, a_gene_median=frozenset())
# What each V12 input mode encodes (models/subcellular_v12.py:235-275); spatial has no centre expression.
MODE_INPUTS = dict(nucleus=frozenset(('center_nucleus',)), cytoplasm=frozenset(('center_cytoplasm',)), local=CENTER,
    spatial=frozenset(('neighbors',)), fused=CENTER | {'neighbors'})
JOINT = (('t_gene_mean', 'a_global'), ('t_gene_mean', 'a_gene'), ('t_gene_depth', 'a_global'),
         ('t_gene_depth', 'a_gene'), ('t_gene_depth', 'a_gene_cell'), ('t_gene_depth_nbr', 'a_gene'),
         ('t_gene_depth_nbr', 'a_gene_cell'), ('t_cell_mean', 'a_cell'), ('t_zero', 'a_gene_median'))
SHRINK_GRID = (1., 2., 5., 10., 20., 50., 100., 200.)
WEIGHT_GRID = tuple(round(.05 * index, 2) for index in range(21))
ANCHOR_TOLERANCE = 2e-3
# Official evaluator, V12 step 3400, validation pooled, fused input mode.
ANCHORS = {3400: dict(split='val', mode='fused', donor='pooled', nll=.6006, total_abs_error=.386, allocation_abs_error=.390)}
MASK_PROTOCOL = dict(seed=42, epoch=0, query_genes=64, both_center_compartments_query_hidden=True,
    gene_mask_ratio=0., neighbor_dropout=0., target_counts_passed_to_forward=False,
    decoder='NB(total)*beta-binomial(nuclear|total)')
SOURCES = ('scripts/evaluate_subcellular_v12_count_baselines.py', 'scripts/evaluate_subcellular_v12.py',
    'src/spatial_pretrain/data/subcellular_v12.py', 'src/spatial_pretrain/data/subcellular_v10.py',
    'src/spatial_pretrain/data/subcellular_v6.py', 'src/spatial_pretrain/data/subcellular_v6_left.py',
    'src/spatial_pretrain/data/subcellular_v5.py', 'src/spatial_pretrain/models/subcellular_v12.py',
    'src/spatial_pretrain/models/subcellular_v11.py', 'src/spatial_pretrain/training/subcellular_v12.py')


def plain(value):
    """JSON-safe copy; nonfinite floats become null instead of invalid JSON."""
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    temporary = path.with_name(path.name + '.partial')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(path)


def prepare_output(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'metrics.json').exists():
        raise FileExistsError('Refusing to overwrite completed metrics')
    return directory


def anchor_failure(directory, failed, payload):
    """Diagnostic file first, then a non-zero exit; metrics.json is never written."""
    ev12.atomic_json(directory / 'anchor_mismatch.json', plain(dict(status='anchor_mismatch', experiment=EXPERIMENT,
                                                                    failed=failed, **payload)))
    print(f'[count-baseline] ANCHOR MISMATCH {failed}; see {directory / "anchor_mismatch.json"}', flush=True)
    sys.exit(2)


# ---------------------------------------------------------------- cohort and fixed held-out batches

def select_cohort_rows(cells):
    """evaluate_subcellular_v12.py:644, verbatim."""
    return np.sort(np.random.default_rng(42).choice(cells, min(400, cells), replace=False))


def cohort_check(cells, expected):
    """The evaluator's canonical train10/val3 test (evaluate_subcellular_v12.py:676-680); never waivable."""
    cohort = ev12.ids_hash(cells.full_cell_id)
    donors = {split: sorted(cells.loc[cells['split'].eq(split), 'patient'].astype(str).unique().tolist())
              for split in ('train', 'val')}
    passed = (cohort == expected and len(cells) == 5200 and cells.full_cell_id.is_unique
              and len(donors['train']) == 10 and len(donors['val']) == 3)
    return dict(name='canonical_cohort', status='passed' if passed else 'failed', cohort_sha256=cohort,
                expected=expected, cells=len(cells), train_donors=donors['train'], val_donors=donors['val'])


def cohort_metadata(prepared):
    """evaluate_subcellular_v12.py:593-613 without labels: no test donor is opened, slice indices are original."""
    protocol = json.loads((prepared / 'protocol.json').read_text())
    metadata = []
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
        directory = prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        if not obs.patient.eq(donor).all() or ('split' in obs and not obs['split'].eq(split).all()):
            raise ValueError('Observation/protocol split mismatch')
        metadata.append((index, directory, obs, split))
    return metadata


def load_vocabulary(assets_dir, cfg):
    """Ordinary scGPT vocab.json in the ScGPTBackbone layout (models/scgpt_backbone.py:20-23,79-80,101)."""
    from spatial_pretrain.models.scgpt_backbone import BACKBONE_LAYOUTS
    roots = [assets_dir] if assets_dir else [cfg.get('assets_dir'), ROOT / 'resources/scgpt_human']
    for root in roots:
        weights = Path(root) / BACKBONE_LAYOUTS['scgpt'][0] if root else None
        if weights is not None and (weights / 'vocab.json').is_file():
            break
    else:
        raise FileNotFoundError(f'No ordinary scGPT vocab.json under {roots}; pass --assets-dir')
    vocab = json.loads((weights / 'vocab.json').read_text())
    arguments = json.loads((weights / 'args.json').read_text()) if (weights / 'args.json').is_file() else {}
    return vocab, int(vocab[arguments.get('pad_token', '<pad>')]), dict(
        path=str((weights / 'vocab.json').resolve()), sha256=ev12.file_sha256(weights / 'vocab.json'))


def load_sample(directory, obs, vocab):
    """evaluate_subcellular_v12.py:299-323: vocabulary-ordered mapped panel, raw counts and slice seed."""
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not obs.full_cell_id.is_unique:
        raise ValueError('Invalid coordinates or duplicate cells')
    panel = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([i for i, gene in enumerate(panel) if gene in vocab], dtype=np.int64)
    if not len(keep):
        raise ValueError('No genes map to scGPT vocabulary')
    keep = keep[np.argsort([vocab[gene] for gene in panel[keep]], kind='stable')]
    genes = np.array([vocab[gene] for gene in panel[keep]], dtype=np.int64)
    nuclear_all = np.load(directory / 'nuclear.npy', mmap_mode='r', allow_pickle=False)
    cyto_all = np.load(directory / 'cytoplasmic.npy', mmap_mode='r', allow_pickle=False)
    if nuclear_all.shape != cyto_all.shape or nuclear_all.shape != (len(obs), len(panel)):
        raise ValueError('Panel/counts/observations shape mismatch')
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    return coords, genes, np.asarray(nuclear_all[:, keep]), np.asarray(cyto_all[:, keep]), slice_seed, len(panel)


def heldout_batch(nuclear, cyto, genes, centers, neighbors, coords, vocab, settings, slice_seed, slice_index, lr):
    """The evaluator's held-out prepare_view_batch_v12 call, evaluate_subcellular_v12.py:351-356."""
    from spatial_pretrain.data.subcellular_v12 import prepare_view_batch_v12
    return prepare_view_batch_v12(nuclear, cyto, genes, centers, neighbors, coords,
        pad_token_id=vocab['<pad>'], cls_token_id=vocab['<cls>'],
        max_tokens=int(settings.get('max_tokens', 512)), query_genes=int(settings.get('query_genes', 64)),
        seed=42, epoch=0, slice_seed=slice_seed, slice_index=slice_index, mask=True,
        gene_mask_ratio=0., neighbor_dropout=0., lr_pairs=lr['pairs'], lr_weights=lr['weights'],
        max_neighbor_distance=settings.get('max_neighbor_distance', 100.0))


def mode_available_from_batch(batch, pad_id, cls_id, max_neighbor_distance):
    """Input-only replica of SubcellularV12.mode_availability.

    models/subcellular_v11.py:267-302 (_neighbor_presence, _input_presence) and
    models/subcellular_v12.py:207-222. Returns numpy masks and effective [B,K] neighbour presence.
    """
    present = batch['neighbor_present'].bool().clone()
    ids = batch.get('neighbor_cell_identity')
    if ids is not None:
        present &= ~ids.eq(batch['cell_identity'][:, None]).all(-1)
        same = ids[:, :, None, :].eq(ids[:, None, :, :]).all(-1)
        earlier = torch.ones_like(same).tril(diagonal=-1)
        present &= ~(same & earlier & present[:, None, :]).any(-1)
    if max_neighbor_distance is not None:
        present &= batch['relative_coords'].float().square().sum(-1).le(max_neighbor_distance ** 2)

    def tokens(prefix):
        genes, valid = batch[f'{prefix}_gene_ids'], batch[f'{prefix}_valid']
        return (valid.bool() & genes.ne(pad_id) & genes.ne(cls_id)).any(-1)

    present &= tokens('neighbor')
    nucleus = batch['nucleus_present'].bool() & tokens('nucleus')
    cytoplasm = batch['cytoplasm_present'].bool() & tokens('cytoplasm')
    local, spatial = nucleus | cytoplasm, present.any(-1)
    available = dict(nucleus=nucleus, cytoplasm=cytoplasm, local=local, spatial=spatial, fused=local | spatial)
    return {mode: available[mode].numpy().astype(bool) for mode in ev12.COUNT_MODES}, present.numpy().astype(bool)


def model_method_availability(batch, pad_id, cls_id, max_neighbor_distance):
    """The real SubcellularV12.mode_availability on a weightless instance; None if torch refuses the stub."""
    from spatial_pretrain.models.subcellular_v12 import SubcellularV12
    try:
        stub = SubcellularV12.__new__(SubcellularV12)
        for key, value in (('pad_id', pad_id), ('cls_id', cls_id), ('max_neighbor_distance', max_neighbor_distance)):
            object.__setattr__(stub, key, value)
        result = SubcellularV12.mode_availability(stub, batch)
    except (AttributeError, TypeError, RuntimeError):
        return None
    return {mode: result[mode].numpy().astype(bool) for mode in ev12.COUNT_MODES}


def visible_depth(full_depth, query_counts, valid):
    """Depth over non-query genes only; hidden targets never enter a baseline input."""
    depth = np.asarray(full_depth, dtype=np.int64) - np.where(valid, query_counts, 0).sum(1)
    if (depth < 0).any():
        raise ValueError('Query counts exceed the cell depth')
    return depth


def check_center_statistics(statistics, depth_nucleus, depth_cytoplasm):
    """V12's own visible sums (data/subcellular_v12.py:173-176) must equal the baseline depths."""
    observed = np.expm1(np.asarray(statistics, dtype=np.float64)[:, :2])
    expected = np.stack((depth_nucleus, depth_cytoplasm), 1).astype(np.float64)
    if observed.shape != expected.shape or not np.allclose(observed, expected, rtol=2e-5, atol=1e-2):
        raise ValueError('Visible depth differs from V12 center_statistics; query counts may leak')


def visible_inputs(batch, nuclear, cyto, genes, centers, present):
    """Targets plus the baseline inputs: non-query centre depth and unmasked neighbour totals."""
    valid = batch['query_valid_mask'].numpy().astype(bool)
    gene_ids = batch['query_gene_ids'].numpy().astype(np.int64)
    query = {name: batch[f'query_{name}_counts'].numpy().astype(np.int64) for name in ('nucleus', 'cytoplasm', 'total')}
    if not np.array_equal(query['nucleus'] + query['cytoplasm'], query['total']) or query['total'][~valid].any():
        raise ValueError('Held-out nucleus/cytoplasm totals do not reconcile')
    columns = np.searchsorted(genes, np.where(valid, gene_ids, genes[0]))
    columns = np.minimum(columns, len(genes) - 1)
    rows = np.asarray(centers)[:, None]
    if (not np.array_equal(genes[columns][valid], gene_ids[valid])
            or not np.array_equal(nuclear[rows, columns][valid], query['nucleus'][valid])
            or not np.array_equal(cyto[rows, columns][valid], query['cytoplasm'][valid])):
        raise ValueError('Query genes/targets do not match the raw count matrices')
    if any(len(np.unique(row[mask])) != mask.sum() for row, mask in zip(gene_ids, valid)):
        raise ValueError('Repeated query gene within a cell')
    depth_nucleus = visible_depth(nuclear[centers].sum(1, dtype=np.int64), query['nucleus'], valid)
    depth_cytoplasm = visible_depth(cyto[centers].sum(1, dtype=np.int64), query['cytoplasm'], valid)
    check_center_statistics(batch['center_statistics'].numpy(), depth_nucleus, depth_cytoplasm)
    identity = batch['neighbor_cell_identity'].numpy()[..., 1]
    neighbor_counts = np.zeros(valid.shape, dtype=np.int64)
    neighbor_depth = np.zeros(len(valid), dtype=np.int64)
    for index, center in enumerate(centers):
        nearby = identity[index][present[index]]
        if (nearby < 0).any() or (nearby == center).any():
            raise ValueError('Invalid present neighbour identity')
        if len(nearby):
            pooled = nuclear[nearby].sum(0, dtype=np.int64) + cyto[nearby].sum(0, dtype=np.int64)
            neighbor_counts[index] = np.where(valid[index], pooled[columns[index]], 0)
            neighbor_depth[index] = pooled.sum()
    return dict(gene_ids=gene_ids, valid=valid, nucleus=query['nucleus'], cytoplasm=query['cytoplasm'],
        depth_nucleus=depth_nucleus, depth_cytoplasm=depth_cytoplasm,
        visible_genes=len(genes) - valid.sum(1), neighbor_counts=neighbor_counts, neighbor_depth=neighbor_depth,
        present_neighbors=present.sum(1))


def model_parameters(model, batch, available):
    """Optional V12 forward on the identical held-out batch (evaluate_subcellular_v12.py:360-362)."""
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(ev12.count_model_inputs(batch))
    output = {}
    for mode in ev12.COUNT_MODES:
        if not np.array_equal(prediction['mode_available'][mode].cpu().numpy().astype(bool), available[mode]):
            raise ValueError(f'Input-only {mode} availability differs from the V12 forward')
        for key in PARAMETER_KEYS:
            output[f'model_{mode}_{key}'] = prediction['count_parameters'][mode][key].double().cpu().numpy()
    return output


def collect_sample(directory, obs, selected, slice_index, vocab, ids, cfg, lr, batch_size, model=None):
    settings = cfg.get('v12', {})
    k = int(settings.get('neighbors', 8))
    if k != int(cfg.get('model', {}).get('neighbors', 8)):
        raise ValueError('V12 data/model neighbor count mismatch')
    started = time.monotonic()
    coords, genes, nuclear, cyto, slice_seed, panel = load_sample(directory, obs, vocab)
    neighbors = ev12.real_neighbors(coords, selected, k)
    blocks, method_checked = [], 0
    for start in range(0, len(selected), batch_size):
        centers = selected[start:start + batch_size]
        batch = heldout_batch(nuclear, cyto, genes, centers, neighbors[start:start + batch_size], coords,
                              vocab, settings, slice_seed, slice_index, lr)
        available, present = mode_available_from_batch(batch, ids['pad'], ids['cls'], ids['distance'])
        official = model_method_availability(batch, ids['pad'], ids['cls'], ids['distance'])
        if official is not None:
            if any(not np.array_equal(official[mode], available[mode]) for mode in ev12.COUNT_MODES):
                raise ValueError('Input-only availability differs from SubcellularV12.mode_availability')
            method_checked += len(centers)
        block = visible_inputs(batch, nuclear, cyto, genes, centers, present)
        block.update({f'available_{mode}': value for mode, value in available.items()})
        block['center_row'] = np.asarray(centers, dtype=np.int64)
        if model is not None:
            block.update(model_parameters(model, batch, available))
        blocks.append(block)
        done = min(start + batch_size, len(selected))
        if done == len(selected) or (start // batch_size) % 10 == 0:
            print(f'[count-baseline] {directory.name} {done}/{len(selected)} {time.monotonic()-started:.1f}s', flush=True)
    values = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    return values, dict(sample=directory.name, selected_cells=len(selected), matched_genes=len(genes), panel_genes=panel,
        slice_index=slice_index, slice_seed=slice_seed, neighbors=k, model_method_availability_checked_cells=method_checked,
        available_counts={mode: int(values[f'available_{mode}'].sum()) for mode in ev12.COUNT_MODES},
        seconds=time.monotonic() - started)


def build_data(values, train):
    """Dense gene index over every queried vocabulary ID; all arrays are cohort-ordered."""
    valid = values['valid']
    vocabulary = np.unique(values['gene_ids'][valid])
    gene = np.minimum(np.searchsorted(vocabulary, values['gene_ids']), len(vocabulary) - 1)
    if not np.array_equal(vocabulary[gene][valid], values['gene_ids'][valid]):
        raise ValueError('Gene index construction failed')
    data = {key: values[key] for key in ('valid', 'nucleus', 'cytoplasm', 'depth_nucleus', 'depth_cytoplasm',
                                         'visible_genes', 'neighbor_counts', 'neighbor_depth')}
    data.update(gene=np.where(valid, gene, 0), genes=len(vocabulary), train=np.asarray(train, dtype=bool))
    return data, vocabulary


# ---------------------------------------------------------------- train-only estimators (pure numpy)

def logit(probability):
    probability = np.clip(probability, *BOUNDS['p_nucleus'])
    return np.log(probability) - np.log1p(-probability)


def train_observations(data):
    """Flattened valid (cell, query) pairs of TRAIN donors; no fitted quantity sees anything else."""
    mask = data['valid'] & data['train'][:, None]
    cell = np.broadcast_to(np.arange(len(mask))[:, None], mask.shape)[mask]
    nucleus = data['nucleus'][mask].astype(np.float64)
    return dict(mask=mask, cell=cell, gene=data['gene'][mask], nucleus=nucleus,
                total=nucleus + data['cytoplasm'][mask].astype(np.float64))


def gene_statistics(gene, nucleus, total, depth, genes):
    return dict(count=np.bincount(gene, minlength=genes).astype(np.float64),
        total_sum=np.bincount(gene, weights=total, minlength=genes),
        nucleus_sum=np.bincount(gene, weights=nucleus, minlength=genes),
        depth_sum=np.bincount(gene, weights=depth, minlength=genes))


def total_estimators(statistics):
    """Per-gene mean and Poisson size-factor rate (sum T / sum visible depth); global fallback for unseen genes."""
    count, total, depth = statistics['count'], statistics['total_sum'], statistics['depth_sum']
    global_mean = float(total.sum() / max(count.sum(), 1.))
    global_rate = float(total.sum() / max(depth.sum(), 1.))
    return dict(global_mean=global_mean, global_rate=global_rate,
        gene_mean=np.where(count > 0, total / np.maximum(count, 1.), global_mean),
        gene_rate=np.where(depth > 0, total / np.maximum(depth, 1.), global_rate))


def allocation_estimators(statistics, pseudocount):
    """Pooled nuclear fraction and per-gene fraction shrunk toward it by `pseudocount` pseudo-molecules."""
    if not pseudocount > 0:
        raise ValueError('Allocation pseudocount must be positive')
    nucleus, total = statistics['nucleus_sum'], statistics['total_sum']
    global_fraction = float(nucleus.sum() / total.sum()) if total.sum() > 0 else .5
    return dict(global_fraction=global_fraction,
                gene_fraction=(nucleus + pseudocount * global_fraction) / (total + pseudocount))


def gene_median_fraction(gene, nucleus, total, genes):
    """Abs-error-optimal constant per gene: train median of N/T over T>0 (global median fallback)."""
    positive = total > 0
    ratio = nucleus[positive] / total[positive]
    result = np.full(genes, float(np.median(ratio)) if len(ratio) else .5)
    if len(ratio):
        medians = pd.Series(ratio).groupby(gene[positive]).median()
        result[medians.index.to_numpy()] = medians.to_numpy()
    return result


def shrunk_fraction(nucleus_depth, depth, target, pseudocount):
    """Cell visible nuclear fraction shrunk toward `target`; a zero-depth cell returns the target."""
    return (np.asarray(nucleus_depth, dtype=np.float64) + pseudocount * target) / (np.asarray(depth, dtype=np.float64) + pseudocount)


def gene_cell_allocation(gene_fraction, cell_fraction, reference):
    """logit p = logit(gene) + logit(cell visible fraction) - logit(reference visible fraction)."""
    return 1 / (1 + np.exp(-(logit(gene_fraction) + logit(cell_fraction) - logit(reference))))


def neighbor_total(neighbor_counts, neighbor_depth, depth, scale, fallback):
    """Pooled neighbour rate of the query gene times the centre visible depth; fallback without neighbours."""
    rate = neighbor_counts / np.maximum(neighbor_depth, 1)
    return np.where(neighbor_depth > 0, scale * rate * depth, fallback)


def poisson_nll(total, mu):
    """Up to the mu-free lgamma(T+1) term."""
    return mu - total * np.log(mu)


def binomial_nll(nucleus, total, probability):
    """Up to the p-free binomial coefficient."""
    probability = np.clip(probability, *BOUNDS['p_nucleus'])
    return -(nucleus * np.log(probability) + (total - nucleus) * np.log1p(-probability))


def choose_blend_weight(total, first, second, grid=WEIGHT_GRID):
    losses = [float(poisson_nll(total, np.maximum((1 - weight) * first + weight * second, BOUNDS['mu_total'][0])).mean())
              for weight in grid]
    return float(grid[int(np.argmin(losses))]), losses


def fit_baselines(data, pseudocount=10., shrink_grid=SHRINK_GRID, weight_grid=WEIGHT_GRID):
    """All mean/allocation parameters; inputs are restricted to TRAIN donor rows before any statistic."""
    obs = train_observations(data)
    if not len(obs['cell']):
        raise ValueError('No training observations')
    depth_nucleus = data['depth_nucleus'].astype(np.float64)
    depth = depth_nucleus + data['depth_cytoplasm'].astype(np.float64)
    statistics = gene_statistics(obs['gene'], obs['nucleus'], obs['total'], depth[obs['cell']], data['genes'])
    fitted = dict(**total_estimators(statistics), **allocation_estimators(statistics, pseudocount),
        gene_train_queries=statistics['count'], gene_train_total=statistics['total_sum'],
        gene_median=gene_median_fraction(obs['gene'], obs['nucleus'], obs['total'], data['genes']),
        allocation_pseudocount=float(pseudocount), train_queries=int(len(obs['cell'])))
    train = data['train']
    visible = float(depth_nucleus[train].sum() / depth[train].sum()) if depth[train].sum() > 0 else .5
    fitted['visible_fraction'] = float(np.clip(visible, *BOUNDS['p_nucleus']))
    fitted['shrink'], fitted['shrink_losses'] = {}, {}
    for name in ('a_cell', 'a_gene_cell'):
        losses = []
        for value in shrink_grid:
            probability = shrunk_fraction(depth_nucleus, depth, fitted['visible_fraction'], value)[obs['cell']]
            if name == 'a_gene_cell':
                probability = gene_cell_allocation(fitted['gene_fraction'][obs['gene']], probability, fitted['visible_fraction'])
            losses.append(float(binomial_nll(obs['nucleus'], obs['total'], probability).sum()))
        fitted['shrink'][name] = float(shrink_grid[int(np.argmin(losses))])
        fitted['shrink_losses'][name] = losses
    by_depth = fitted['gene_rate'][obs['gene']] * depth[obs['cell']]
    neighbor_depth = data['neighbor_depth'][obs['cell']].astype(np.float64)
    raw = neighbor_total(data['neighbor_counts'][obs['mask']], neighbor_depth, depth[obs['cell']], 1., 0.)
    has = neighbor_depth > 0
    fitted['neighbor_scale'] = float(obs['total'][has].sum() / raw[has].sum()) if raw[has].sum() > 0 else 1.
    by_neighbor = np.where(has, fitted['neighbor_scale'] * raw, by_depth)
    fitted['neighbor_weight'], fitted['neighbor_weight_losses'] = choose_blend_weight(obs['total'], by_depth, by_neighbor, weight_grid)
    return fitted


def predict_components(data, fitted):
    """[cells,Q] means and nuclear probabilities from gene identity and VISIBLE inputs only; targets are never read."""
    gene, shape = data['gene'], data['gene'].shape
    depth_nucleus = data['depth_nucleus'].astype(np.float64)
    depth = depth_nucleus + data['depth_cytoplasm'].astype(np.float64)
    by_depth = fitted['gene_rate'][gene] * depth[:, None]
    by_neighbor = neighbor_total(data['neighbor_counts'], data['neighbor_depth'][:, None], depth[:, None],
                                 fitted['neighbor_scale'], by_depth)
    weight = fitted['neighbor_weight']
    totals = dict(t_zero=np.zeros(shape),
        t_cell_mean=np.broadcast_to((depth / np.maximum(data['visible_genes'], 1))[:, None], shape),
        t_gene_mean=fitted['gene_mean'][gene], t_gene_depth=by_depth,
        t_gene_depth_nbr=(1 - weight) * by_depth + weight * by_neighbor)

    def cell(name):
        return shrunk_fraction(depth_nucleus, depth, fitted['visible_fraction'], fitted['shrink'][name])[:, None]

    allocations = dict(a_global=np.full(shape, fitted['global_fraction']),
        a_cell=np.broadcast_to(cell('a_cell'), shape), a_gene=fitted['gene_fraction'][gene],
        a_gene_cell=gene_cell_allocation(fitted['gene_fraction'][gene], cell('a_gene_cell'), fitted['visible_fraction']),
        a_gene_median=fitted['gene_median'][gene])
    totals = {name: np.clip(np.asarray(value, dtype=np.float64), *BOUNDS['mu_total']) for name, value in totals.items()}
    allocations = {name: np.clip(np.asarray(value, dtype=np.float64), *BOUNDS['p_nucleus']) for name, value in allocations.items()}
    return totals, allocations


# ---------------------------------------------------------------- likelihood, dispersion fits, scoring

def joint_nll(nucleus, cytoplasm, mu, theta, p, kappa):
    """Elementwise NLL through the training law that ev12.joint_count_nll delegates to; no formula fork."""
    from spatial_pretrain.training.subcellular_v12 import joint_count_nll as training_nll
    return training_nll(dict(mu_total=mu, theta_total=theta, p_nucleus=p, kappa_allocation=kappa), nucleus, nucleus + cytoplasm)


def nb_nll(total, mu, theta):
    """NB-total term alone (training/subcellular_v12.py:63-65); used only to split a joint NLL into parts."""
    log_denominator = (theta + mu).log()
    return -(torch.lgamma(total + theta) - torch.lgamma(theta) - torch.lgamma(total + 1)
             + theta * (theta.log() - log_denominator) + total * (mu.log() - log_denominator))


def fit_grouped_log_scalar(loss, lower, upper, group, groups, points=17, rounds=3):
    """Per-group argmin of sum_i loss(value[group_i])_i over [lower, upper] by a zooming log10 grid."""
    group = torch.as_tensor(np.asarray(group), dtype=torch.int64)
    floor, ceiling = math.log10(lower), math.log10(upper)
    low = torch.full((groups,), floor, dtype=torch.float64)
    high = torch.full((groups,), ceiling, dtype=torch.float64)
    fractions = torch.linspace(0, 1, points, dtype=torch.float64)
    for _ in range(rounds):
        scores = torch.stack([torch.bincount(group, minlength=groups, weights=loss(
            torch.pow(10., low + fraction * (high - low)).clamp(lower, upper)[group])) for fraction in fractions])
        step = (high - low) / (points - 1)
        center = low + scores.argmin(0) * step
        low, high = (center - step).clamp_min(floor), (center + step).clamp_max(ceiling)
    return torch.pow(10., center).clamp(lower, upper).numpy(), scores.min(0).values.numpy()


def fit_dispersion(loss, bounds, gene, genes, informative, min_informative):
    """Maximum likelihood: one global value, per-gene values where enough informative TRAIN queries exist."""
    global_value, _ = fit_grouped_log_scalar(loss, *bounds, np.zeros(len(gene), dtype=np.int64), 1)
    per_gene, _ = fit_grouped_log_scalar(loss, *bounds, gene, genes)
    use = np.bincount(gene[informative], minlength=genes) >= min_informative
    return dict(global_value=float(global_value[0]), per_gene=np.where(use, per_gene, global_value[0]),
                fitted_genes=int(use.sum()), fallback_genes=int((~use).sum()))


def fit_dispersions(data, totals, allocations, min_informative=20, gene_free=GENE_FREE):
    """NB theta per total estimator and beta-binomial kappa per allocation estimator, TRAIN donors only.

    Both minimise the imported joint NLL: theta with allocation fixed (a theta-free term) and kappa with
    the total fixed (a kappa-free term), so each is exact ML under the evaluator's own probability law.
    Estimators in `gene_free` keep the single global ML value for every gene: a per-gene dispersion would
    smuggle gene knowledge (e.g. the gene's zero fraction) into a mean that claims to have none.
    """
    obs = train_observations(data)
    n, total = torch.as_tensor(obs['nucleus']), torch.as_tensor(obs['total'])
    c, ones, gene = total - n, torch.ones_like(n), obs['gene']
    positive = obs['total'] > 0

    def expand(name, fit, parameter, informative):
        shared = name in gene_free
        per_gene = np.full(data['genes'], fit['global_value']) if shared else fit['per_gene']
        return per_gene[data['gene']], dict(parameter=parameter, informative=informative,
            scope='global' if shared else 'per_gene_with_global_fallback', global_value=fit['global_value'],
            fitted_genes=0 if shared else fit['fitted_genes'], fallback_genes=data['genes'] if shared else fit['fallback_genes'])

    theta, kappa, summary = {}, {}, {}
    for name, value in totals.items():
        mu = torch.as_tensor(value[obs['mask']])
        fit = fit_dispersion(lambda item: joint_nll(n, c, mu, item, .5 * ones, ones), BOUNDS['theta_total'],
                             gene, data['genes'], positive, min_informative)
        theta[name], summary[name] = expand(name, fit, 'theta_total', 'train queries with T>0')
    chosen = torch.as_tensor(positive)
    for name, value in allocations.items():
        p = torch.as_tensor(value[obs['mask']])[chosen]
        fit = fit_dispersion(lambda item: joint_nll(n[chosen], c[chosen], ones[chosen], ones[chosen], p, item),
                             BOUNDS['kappa_allocation'], gene[positive], data['genes'], obs['total'][positive] >= 2, min_informative)
        kappa[name], summary[name] = expand(name, fit, 'kappa_allocation', 'train queries with T>=2')
    return theta, kappa, summary


def expected_allocation_abs_error(total, p, kappa):
    """E|p - N/T| for N ~ beta-binomial(T, p*kappa, (1-p)*kappa) by exact enumeration of N=0..T; 0 where T=0.

    The value allocation_abs_error would take on average if (p, kappa) were the true allocation law. The PMF is
    the allocation term of training/subcellular_v12.py:66-69.
    """
    total, p, kappa = [torch.as_tensor(np.asarray(value), dtype=torch.float64) for value in (total, p, kappa)]
    if total.shape != p.shape or total.shape != kappa.shape:
        raise ValueError('Totals and allocation parameters must share one shape')
    result = torch.zeros_like(p)
    for value in torch.unique(total[total > 0]).tolist():
        chosen = total == value
        q, k = p[chosen][:, None], kappa[chosen][:, None]
        alpha, beta = q * k, (1 - q) * k
        n = torch.arange(int(value) + 1, dtype=torch.float64)[None]
        c = value - n
        log_pmf = (math.lgamma(value + 1) - torch.lgamma(n + 1) - torch.lgamma(c + 1)
                   + torch.lgamma(n + alpha) + torch.lgamma(c + beta) - torch.lgamma(value + k)
                   + torch.lgamma(k) - torch.lgamma(alpha) - torch.lgamma(beta))
        result[chosen] = (log_pmf.exp() * (q - n / value).abs()).sum(1)
    return result


def clip_parameters(mu, theta, p, kappa, valid):
    """Decoder-range parameters; padded query slots get harmless valid values and are never scored."""
    values = dict(mu_total=mu, theta_total=theta, p_nucleus=p, kappa_allocation=kappa)
    fill = dict(mu_total=1., theta_total=1., p_nucleus=.5, kappa_allocation=1.)
    return {key: np.where(valid, np.clip(np.asarray(value, dtype=np.float64), *BOUNDS[key]), fill[key])
            for key, value in values.items()}


def score_parameters(nucleus, cytoplasm, valid, available, parameters):
    """CPU float64 replica of ev12.score_masked_counts (evaluate_subcellular_v12.py:196-217), per cell and mode.

    `parameters` is one {mu_total,theta_total,p_nucleus,kappa_allocation} dict of [cells,Q] arrays used for every
    availability mode (baselines) or {mode: dict} (a model). Adds the NB/allocation split of the same NLL.
    """
    n = torch.as_tensor(np.asarray(nucleus), dtype=torch.float64)
    c = torch.as_tensor(np.asarray(cytoplasm), dtype=torch.float64)
    valid = torch.as_tensor(np.asarray(valid)).bool()
    frames = []
    for mode in ev12.COUNT_MODES:
        item = parameters[mode] if mode in parameters else parameters
        item = {key: torch.as_tensor(np.asarray(item[key]), dtype=torch.float64) for key in PARAMETER_KEYS}
        score = ev12.joint_count_nll(n, c, item)
        active = valid & torch.as_tensor(np.asarray(available[mode])).bool()[:, None]
        positive = active & ((n + c) > 0)
        mu, p = item['mu_total'], item['p_nucleus']
        total_part = nb_nll(n + c, mu, item['theta_total'])
        calibrated = expected_allocation_abs_error(n + c, p, item['kappa_allocation'])
        sums = dict(query_count=active.sum(1), positive_query_count=positive.sum(1),
            nll_sum=(score * active).sum(1), total_abs_error_sum=((mu - n - c).abs() * active).sum(1),
            nucleus_abs_error_sum=((mu * p - n).abs() * active).sum(1),
            cytoplasm_abs_error_sum=((mu * (1 - p) - c).abs() * active).sum(1),
            allocation_abs_error_sum=((p - n / (n + c).clamp_min(1)).abs() * positive).sum(1),
            nll_total_sum=(total_part * active).sum(1), nll_allocation_sum=((score - total_part) * positive).sum(1),
            allocation_calibrated_abs_error_sum=(calibrated * positive).sum(1))
        frame = pd.DataFrame({key: value.numpy() for key, value in sums.items()})
        frame.insert(0, 'mode', mode)
        frame.insert(0, 'cell', np.arange(len(frame)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def attach_cells(frame, cells):
    for column in ('center_row', 'full_cell_id', 'patient', 'split'):
        frame[column] = cells[column].to_numpy()[frame.cell.to_numpy()]
    return frame


def aggregate_parts(frame):
    """NB/allocation NLL split on the grouping of ev12.aggregate_masked_counts; the two parts sum to nll."""
    rows = {}
    for (split, mode), part in frame.groupby(['split', 'mode'], sort=True):
        for donor, subset in [('pooled', part), *list(part.groupby('patient', sort=True))]:
            count, positive = int(subset.query_count.sum()), int(subset.positive_query_count.sum())
            rows[str(split), str(mode), str(donor)] = dict(
                nll_total_part=float(subset.nll_total_sum.sum() / count) if count else None,
                nll_allocation_part=float(subset.nll_allocation_sum.sum() / count) if count else None,
                allocation_nll_per_positive=float(subset.nll_allocation_sum.sum() / positive) if positive else None,
                allocation_abs_error_calibrated=(float(subset.allocation_calibrated_abs_error_sum.sum() / positive)
                                                 if positive else None))
    return rows


def summarize(frame, name, extra):
    """Evaluator rows (ev12.aggregate_masked_counts, unchanged definitions) plus identification and NLL parts."""
    parts = aggregate_parts(frame)
    return [dict(baseline=name, **extra, **row, **parts[row['split'], row['mode'], row['donor']])
            for row in ev12.aggregate_masked_counts(frame)]


def run_baselines(data, cells, available, joint=JOINT, pseudocount=10., min_informative=20):
    fitted = fit_baselines(data, pseudocount)
    totals, allocations = predict_components(data, fitted)
    theta, kappa, dispersion = fit_dispersions(data, totals, allocations, min_informative)
    rows, frames, parameters = [], [], {}
    for total_name, allocation_name in joint:
        name = f'{total_name}+{allocation_name}'
        parameters[name] = clip_parameters(totals[total_name], theta[total_name], allocations[allocation_name],
                                           kappa[allocation_name], data['valid'])
        frame = attach_cells(score_parameters(data['nucleus'], data['cytoplasm'], data['valid'], available, parameters[name]), cells)
        rows.extend(summarize(frame, name, dict(source='train_donor_trivial_baseline',
                                                total_estimator=total_name, allocation_estimator=allocation_name)))
        frame.insert(0, 'baseline', name)
        frames.append(frame)
        print(f'[count-baseline] scored {name}', flush=True)
    return dict(rows=rows, frame=pd.concat(frames, ignore_index=True), fitted=fitted, dispersion=dispersion,
                totals=totals, allocations=allocations, theta=theta, kappa=kappa, parameters=parameters)


def target_summary(data):
    result = {}
    for split, cells in (('train', data['train']), ('val', ~data['train'])):
        active = data['valid'] & cells[:, None]
        nucleus = data['nucleus'][active].astype(np.float64)
        total = nucleus + data['cytoplasm'][active]
        positive = total > 0
        fraction = float(nucleus.sum() / total.sum()) if total.sum() > 0 else None
        result[split] = dict(queries=int(active.sum()), mean_total=float(total.mean()) if len(total) else None,
            positive_fraction=float(positive.mean()) if len(total) else None,
            single_count_share_of_positive=float((total[positive] == 1).mean()) if positive.any() else None,
            pooled_nuclear_fraction=fraction,
            mean_query_nuclear_fraction=float((nucleus[positive] / total[positive]).mean()) if positive.any() else None)
    return result


# ---------------------------------------------------------------- reference comparison and anchors

def row_key(row):
    return str(row['split']), str(row['mode']), str(row['donor'])


def load_reference(path):
    path = Path(path)
    report = json.loads(path.read_text())
    masked = report['masked_count_prediction']
    step, variant = int(report['step']), str(report.get('variant', 'v12'))
    return dict(path=str(path.resolve()), sha256=ev12.file_sha256(path), step=step, variant=variant,
        label=f'{variant}@step{step:06d}', rows=masked['results'],
        protocol={key: masked.get(key) for key in MASK_PROTOCOL}, cohort_sha256=report.get('cohort_sha256'),
        evaluator_sha256=report.get('evaluator_sha256'), settings=report.get('config', {}).get('v12', {}),
        batch_size=report.get('inference', {}).get('batch_size'),
        vocab_sha256=report.get('backbone_provenance', {}).get('vocab_sha256'))


def count_mismatches(rows, reference_rows):
    """An identical masked query set implies identical counts for every (split, mode, donor)."""
    mine = {row_key(row): row for row in rows}
    theirs = {row_key(row): row for row in reference_rows}
    problems = [dict(zip(('split', 'mode', 'donor'), key), reason='missing in baselines' if key in theirs else 'missing in reference')
                for key in sorted(set(mine) ^ set(theirs))]
    for key in sorted(set(mine) & set(theirs)):
        for field in COUNT_FIELDS:
            if int(mine[key][field]) != int(theirs[key][field]):
                problems.append(dict(zip(('split', 'mode', 'donor'), key), field=field,
                                     baseline=int(mine[key][field]), reference=int(theirs[key][field])))
    return problems


def per_cell_mismatches(frame, reference):
    """Per (cell, mode) query_count and positive_query_count against the evaluator's masked_count_per_cell.parquet."""
    columns = ['full_cell_id', 'mode', 'query_count', 'positive_query_count']
    merged = reference[columns].merge(frame[columns], on=['full_cell_id', 'mode'], how='outer',
                                      suffixes=('_reference', ''), indicator=True)
    bad = merged[(merged['_merge'] != 'both') | (merged.query_count != merged.query_count_reference)
                 | (merged.positive_query_count != merged.positive_query_count_reference)]
    return int(len(bad)), bad.head(20).astype(str).to_dict('records')


def metric_differences(rows, reference_rows, metrics=METRICS):
    """Largest absolute metric gap between two row sets on their shared (split, mode, donor) keys."""
    theirs = {row_key(row): row for row in reference_rows}
    worst = {metric: 0. for metric in metrics}
    compared = 0
    for row in rows:
        other = theirs.get(row_key(row))
        if other is None:
            continue
        compared += 1
        for metric in metrics:
            if row.get(metric) is None or other.get(metric) is None:
                if row.get(metric) is not other.get(metric):
                    worst[metric] = float('inf')
            else:
                worst[metric] = max(worst[metric], abs(float(row[metric]) - float(other[metric])))
    return worst, compared


def expected_value_check(rows, step):
    """Hard-coded official numbers for a known step; None when the step has no anchor."""
    anchor = ANCHORS.get(int(step))
    if anchor is None:
        return None
    row = next((row for row in rows if row_key(row) == (anchor['split'], anchor['mode'], anchor['donor'])), None)
    if row is None:
        return dict(passed=False, reason='anchor row missing')
    differences = {metric: abs(float(row[metric]) - anchor[metric]) for metric in METRICS if metric in anchor}
    return dict(passed=all(value <= ANCHOR_TOLERANCE for value in differences.values()), expected=anchor,
                observed={metric: row[metric] for metric in differences}, absolute_differences=differences)


def scorer_is_anchored(checks):
    """True only if this script's own metric arithmetic was compared numerically with the evaluator's in this run."""
    return any(check['status'] == 'passed' and (check['name'] == 'scorer_vs_evaluator_score_masked_counts'
               or check['name'].startswith('rescored_model_vs_reference:')) for check in checks)


def skill(model, baseline):
    """1 - model_error / baseline_error; positive means the model beats the baseline."""
    if model is None or baseline is None or not baseline > 0:
        return None
    return float(1 - model / baseline)


def inputs_within_mode(total_estimator, allocation_estimator, mode):
    """True when the baseline reads no cell input that the V12 input mode does not also encode; None if unknown.

    Only such rows are a like-for-like skill: e.g. t_gene_depth uses both centre compartments' visible depth,
    which the nucleus-only, cytoplasm-only and neighbours-only (spatial) model modes never see.
    """
    if total_estimator not in ESTIMATOR_INPUTS or allocation_estimator not in ESTIMATOR_INPUTS or mode not in MODE_INPUTS:
        return None
    return (ESTIMATOR_INPUTS[total_estimator] | ESTIMATOR_INPUTS[allocation_estimator]) <= MODE_INPUTS[mode]


def side_by_side(rows, model_rows, label, metrics=METRICS, carry=DIAGNOSTICS):
    """One row per (baseline, split, mode, donor): baseline value, model value and skill for every metric.

    `carry` columns are copied from whichever side has them, without a skill.
    """
    models = {row_key(row): row for row in model_rows}
    table = []
    for row in rows:
        model = models.get(row_key(row))
        if model is None:
            continue
        item = dict(reference=label, baseline=row['baseline'], split=row['split'], mode=row['mode'], donor=row['donor'],
                    query_count=row['query_count'], positive_query_count=row['positive_query_count'],
                    baseline_inputs_within_model_mode=inputs_within_mode(
                        row.get('total_estimator'), row.get('allocation_estimator'), row['mode']))
        for metric in metrics:
            if metric in model and metric in row:
                item[f'{metric}_baseline'], item[f'{metric}_model'] = row[metric], model[metric]
                item[f'{metric}_skill'] = skill(model[metric], row[metric])
        for name in carry:
            for side, source in (('baseline', row), ('model', model)):
                if name in source:
                    item[f'{name}_{side}'] = source[name]
        table.append(item)
    return table


def cuda_scorer_check(data, available, parameters, frame):
    """Run the evaluator's own score_masked_counts (CUDA-only) on one baseline and compare every per-cell sum."""
    if not torch.cuda.is_available():
        return dict(status='skipped', reason='CUDA unavailable; ev12.score_masked_counts is hard-wired to cuda')
    result = dict(count_parameters={mode: {key: torch.as_tensor(parameters[key], device='cuda') for key in PARAMETER_KEYS}
                                    for mode in ev12.COUNT_MODES},
                  mode_available={mode: torch.as_tensor(available[mode], device='cuda') for mode in ev12.COUNT_MODES})
    targets = dict(query_nucleus_counts=torch.as_tensor(data['nucleus']), query_cytoplasm_counts=torch.as_tensor(data['cytoplasm']),
                   query_total_counts=torch.as_tensor(data['nucleus'] + data['cytoplasm']),
                   query_valid_mask=torch.as_tensor(data['valid']))
    try:
        official = pd.DataFrame(ev12.score_masked_counts(result, targets, np.arange(len(data['valid']))))
    except RuntimeError as error:                             # e.g. a busy GPU; value errors still propagate
        return dict(status='skipped', reason=f'CUDA scoring failed: {error}')
    merged = official.merge(frame.drop(columns='center_row', errors='ignore'), left_on=['center_row', 'mode'],
                            right_on=['cell', 'mode'], suffixes=('_official', ''))
    if len(merged) != len(official) or len(merged) != len(frame):
        return dict(status='failed', reason='row sets differ')
    worst = max(float((merged[f'{key}_official'] - merged[key]).abs().max())
                for key in ('query_count', 'positive_query_count', *SUM_FIELDS))
    return dict(status='passed' if worst <= 1e-6 else 'failed', max_absolute_per_cell_sum_difference=worst, tolerance=1e-6)


CAVEATS = [
    'allocation_abs_error definition (evaluate_subcellular_v12.py:207,213,228): |p_nucleus - N/max(T,1)| is summed only over '
    'queries that are valid, belong to a cell whose input mode is available AND have T=N+C>0, and is divided by '
    'positive_query_count. Queries with T=0 enter neither numerator nor denominator. Baselines are scored with the identical '
    'expression and ev12.aggregate_masked_counts.',
    'nll, total_abs_error, nucleus_abs_error (|mu*p-N|) and cytoplasm_abs_error (|mu*(1-p)-C|) are sums over all valid '
    'queries of available cells divided by query_count (evaluate_subcellular_v12.py:205-212,226-227): a pooled query mean, '
    'not the training loss reduction (gene->cell->mode means).',
    'Absolute errors are not proper scores. On sparse counts the abs-error-optimal prediction is the conditional median: '
    't_zero (mu at the decoder floor 1e-4) gives total_abs_error = mean T, and a_gene_median is the best per-gene constant '
    'for allocation_abs_error on train. For a T=1 query N/T is 0 or 1, so a perfectly calibrated p=q has expected error '
    '2q(1-q) rather than 0. Whether this explains a flat model allocation_abs_error is NOT assumed here: it depends on the '
    'share of T=1 queries and the nuclear fractions (target_summary) and is measured per row by '
    'allocation_abs_error_calibrated = mean over the same T>0 queries of E|p - N/T| with N ~ beta-binomial(T, p*kappa, '
    '(1-p)*kappa), i.e. the value the row would score if its own (p, kappa) were the true law. An observed '
    'allocation_abs_error close to that expectation is what calibration looks like and cannot shrink much further; one well '
    'above it indicates miscalibration. For V12 this column exists only on --checkpoint re-scored rows. Judge allocation '
    'learning by nll_allocation_part / allocation_nll_per_positive.',
    'NLL comparability: baseline nll is ev12.joint_count_nll, i.e. the training NB(total)*beta-binomial(nucleus|total) law, on '
    'the same targets and query set; baseline parameters are clipped to the JointCountDecoder output ranges, so they lie '
    'inside the model family. nll_total_part + nll_allocation_part = nll. The model NB/allocation split is only available '
    'with --checkpoint, because evaluator metrics.json stores the joint NLL alone.',
    'Dispersion: theta (per total estimator) and kappa (per allocation estimator) are maximum-likelihood fits on TRAIN donor '
    'queries by a three-round zooming log10 grid within decoder bounds; per-gene values where a gene has at least '
    '--min-informative informative train queries (T>0 for theta, T>=2 for kappa), otherwise the global ML value. A T<=1 '
    'query carries no kappa information (beta-binomial reduces to Bernoulli(p)). The gene-free estimators t_zero, '
    't_cell_mean, a_global and a_cell always use the single global ML dispersion (fitted.dispersion.*.scope = global), '
    'because a per-gene theta/kappa would encode gene knowledge such as the gene zero fraction.',
    'Baselines are fitted on the 10 LungPF train donors of this cohort (in-domain per-gene rates/fractions). V12 is evaluated '
    'zero-shot: its pretraining corpus (configs/xenium48_compartment_sources.yaml) lists no LungPF sample (it does list '
    'other human lung Xenium slides), so V12 never saw these donors or this panel\'s LungPF gene statistics. t_cell_mean+a_cell '
    'is the gene-free reference in every column including nll: its mean, nuclear probability, theta and kappa use no gene '
    'identity, only LungPF-train global scalars (visible nuclear fraction, shrinkage pseudo-count, one theta, one kappa).',
    'Train-split baseline rows are in-sample for the fitted gene parameters; validation-donor rows are the honest ones.',
    'The mode column of a baseline row only selects the cells on which the corresponding V12 input mode is available '
    '(same active query set as the model row); baseline predictions themselves do not depend on the mode. Skill is '
    'therefore like-for-like only where comparison.baseline_inputs_within_model_mode is true: every baseline for fused, '
    'all but t_gene_depth_nbr for local, and only the centre-free ones (t_gene_mean / t_zero with a_global / a_gene / '
    'a_gene_median) for nucleus, cytoplasm and spatial, whose model inputs are one compartment or the neighbours alone. '
    'Elsewhere the baseline reads centre depth / nuclear fraction the model mode never sees, and a negative skill there '
    'does not mean the model loses to a trivial estimator given the same inputs.',
    't_gene_depth_nbr uses raw neighbour N+C counts of the query gene, pooled over the effectively present neighbours; '
    'V12 sees these neighbours only as 51-bin tokens, so this baseline has slightly richer neighbour input.',
    'a_gene_cell centres the cell effect on the TRAIN pooled visible nuclear fraction (sum visible N / sum visible depth); '
    'cell shrinkage pseudo-counts and the neighbour blend weight are chosen on train donors only.',
    'Without --checkpoint the LR prior is not loaded (lr_pairs=None): LR features are computed after, and independently '
    'of, the query draw, visible counts and availability masks (data/subcellular_v12.py:141-212).',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/benchmarking_data/lungpf/prepared')
    parser.add_argument('--output-dir', '--output', required=True, type=Path)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/subcellular_v12_compartment_20260930.yaml')
    parser.add_argument('--assets-dir', type=Path, help='ordinary scGPT asset root; default config assets_dir, then resources/scgpt_human')
    parser.add_argument('--reference-metrics', type=Path, nargs='*', default=[], help='evaluator metrics.json files')
    parser.add_argument('--allow-anchor-mismatch', action='store_true',
                        help='finish despite failed reference anchors (a non-canonical cohort is never waived)')
    parser.add_argument('--checkpoint', type=Path, help='optional: re-score this V12 checkpoint through the same path (CUDA)')
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--allocation-pseudocount', default=10., type=float)
    parser.add_argument('--min-informative', default=20, type=int)
    parser.add_argument('--all-combinations', action='store_true', help='score every total x allocation pair')
    args = parser.parse_args()
    if args.batch_size < 1 or args.allocation_pseudocount <= 0 or args.min_informative < 1:
        parser.error('Batch size, pseudocount and min-informative must be positive')
    output = prepare_output(args.output_dir)
    started = time.monotonic()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    caveats = list(CAVEATS)
    references = [load_reference(path) for path in args.reference_metrics]
    model, step, checkpoint = None, None, None
    if args.checkpoint:
        if not torch.cuda.is_available():
            parser.error('--checkpoint requires CUDA')
        from spatial_pretrain.data.subcellular_v6 import load_lr_prior
        torch.backends.mha.set_fastpath_enabled(False)
        checkpoint = dict(path=str(args.checkpoint.resolve()), sha256=ev12.file_sha256(args.checkpoint))
        model, original, cfg, step, provenance = ev12.load_model(args.checkpoint, args.expected_step)
        del original
        torch.cuda.empty_cache()
        vocab, lr = model.vocab, load_lr_prior(cfg, vocab=model.vocab)
        ids = dict(pad=model.pad_id, cls=model.cls_id, distance=model.max_neighbor_distance)
        vocabulary = dict(path='checkpoint backbone', sha256=provenance['backbone_provenance'].get('vocab_sha256'))
        checkpoint.update(step=step, config_sha256=provenance['config_sha256'])
    else:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        vocab, pad, vocabulary = load_vocabulary(args.assets_dir, cfg)
        lr = dict(pairs=None, weights=None)
        ids = dict(pad=pad, cls=int(vocab['<cls>']), distance=cfg.get('model', {}).get('max_neighbor_distance'))
    settings = cfg.get('v12', {})
    used = dict(neighbors=int(settings.get('neighbors', 8)), max_tokens=int(settings.get('max_tokens', 512)),
                query_genes=int(settings.get('query_genes', 64)), max_neighbor_distance=settings.get('max_neighbor_distance', 100.0))
    protocol = dict(MASK_PROTOCOL, query_genes=used['query_genes'])
    metadata = cohort_metadata(args.prepared)
    selections = [select_cohort_rows(len(obs)) for _, _, obs, _ in metadata]
    frames = []
    for (_, _, obs, split), selected in zip(metadata, selections):
        chosen = obs.iloc[selected].copy(); chosen['split'] = split
        frames.append(chosen)
    cells = pd.concat(frames, ignore_index=True)
    # Like the evaluator, a wrong cohort is fatal before any encoding; --allow-anchor-mismatch cannot waive it.
    canonical = cohort_check(cells, ev12.CANONICAL_COHORT)
    cohort, train_donors, val_donors = canonical['cohort_sha256'], canonical['train_donors'], canonical['val_donors']
    if canonical['status'] != 'passed':
        anchor_failure(output, ['canonical_cohort'], dict(vocabulary=vocabulary, cohort_sha256=cohort, anchor_check=dict(
            passed=False, failed=['canonical_cohort'], waivable=False, checks=[canonical])))
    checks = [canonical]
    blocks, qa = [], []
    for (index, directory, obs, split), selected in zip(metadata, selections):
        values, item = collect_sample(directory, obs, selected, index, vocab, ids, cfg, lr, args.batch_size, model)
        if not np.array_equal(values['center_row'], selected):
            raise ValueError('Cohort row order changed during collection')
        blocks.append(values); qa.append(item)
        ev12.atomic_json(output / 'progress.json', dict(stage='collect', completed_slices=len(blocks),
            total_slices=len(metadata), last_sample=directory.name, seconds=time.monotonic() - started))
    del model
    values = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    if len(values['center_row']) != len(cells):
        raise ValueError('Collected cells do not match the cohort')
    cells['center_row'] = values['center_row']
    data, gene_vocabulary = build_data(values, cells['split'].eq('train').to_numpy())
    available = {mode: values[f'available_{mode}'] for mode in ev12.COUNT_MODES}
    checked = sum(item['model_method_availability_checked_cells'] for item in qa)
    checks.append(dict(name='mode_availability_model_method', status='passed' if checked == len(cells) else 'skipped',
                       cells_checked_against_SubcellularV12_mode_availability=checked))
    if checked != len(cells):
        caveats.append('SubcellularV12.mode_availability could not be called on a weightless instance for every cell; '
                       'availability relies on the input-only replica (count anchors below still test it).')
    joint = tuple((t, a) for t in TOTALS for a in ALLOCATIONS) if args.all_combinations else JOINT
    ev12.atomic_json(output / 'progress.json', dict(stage='fit_and_score', seconds=time.monotonic() - started))
    outcome = run_baselines(data, cells, available, joint, args.allocation_pseudocount, args.min_informative)
    rows, frame = outcome['rows'], outcome['frame']
    first = f'{joint[0][0]}+{joint[0][1]}'
    primary = 't_gene_depth+a_gene_cell' if 't_gene_depth+a_gene_cell' in outcome['parameters'] else first
    scorer = cuda_scorer_check(data, available, outcome['parameters'][primary], frame[frame.baseline.eq(primary)])
    checks.append(dict(name='scorer_vs_evaluator_score_masked_counts', baseline=primary, **scorer))
    count_rows = [row for row in rows if row['baseline'] == first]
    model_rows = []
    if step is not None:
        parameters = {mode: {key: values[f'model_{mode}_{key}'] for key in PARAMETER_KEYS} for mode in ev12.COUNT_MODES}
        name = f'v12_rescored@step{step:06d}'
        model_frame = attach_cells(score_parameters(data['nucleus'], data['cytoplasm'], data['valid'], available, parameters), cells)
        model_rows = summarize(model_frame, name, dict(source='v12_checkpoint_rescored_by_this_script',
                                                       total_estimator='v12_decoder', allocation_estimator='v12_decoder'))
        model_frame.insert(0, 'baseline', name)
        frame = pd.concat([frame, model_frame], ignore_index=True)
        expected = expected_value_check(model_rows, step)
        if expected is not None:
            checks.append(dict(name='rescored_model_expected_values', status='passed' if expected['passed'] else 'failed', **expected))
    comparison = side_by_side(rows, model_rows, f'rescored@step{step:06d}', METRICS + PART_METRICS) if model_rows else []
    for reference in references:
        label = reference['label']
        problems = []
        if reference['cohort_sha256'] != cohort:
            problems.append('cohort_sha256 differs')
        if reference['protocol'] != protocol:
            problems.append(f'masked protocol differs: {reference["protocol"]}')
        if any(reference['settings'].get(key, used[key]) != used[key] for key in used):
            problems.append(f'v12 data settings differ: {reference["settings"]}')
        if reference['vocab_sha256'] and vocabulary['sha256'] and reference['vocab_sha256'] != vocabulary['sha256']:
            problems.append('scGPT vocabulary sha256 differs')
        checks.append(dict(name=f'reference_protocol:{label}', status='failed' if problems else 'passed', problems=problems,
                           evaluator_sha256_matches=reference['evaluator_sha256'] == ev12.file_sha256(ROOT / SOURCES[1])))
        mismatches = count_mismatches(count_rows, reference['rows'])
        checks.append(dict(name=f'reference_query_counts:{label}', status='failed' if mismatches else 'passed',
                           compared_rows=len(reference['rows']), mismatches=len(mismatches), examples=mismatches[:20]))
        per_cell = Path(reference['path']).parent / 'masked_count_per_cell.parquet'
        if per_cell.is_file():
            bad, examples = per_cell_mismatches(frame[frame.baseline.eq(first)], pd.read_parquet(per_cell))
            checks.append(dict(name=f'reference_per_cell_counts:{label}', status='failed' if bad else 'passed',
                               mismatched_cell_modes=bad, examples=examples))
        else:
            checks.append(dict(name=f'reference_per_cell_counts:{label}', status='skipped', reason=f'{per_cell} not found'))
        expected = expected_value_check(reference['rows'], reference['step'])
        if expected is not None:
            checks.append(dict(name=f'reference_file_expected_values:{label}', status='passed' if expected['passed'] else 'failed',
                scope='reads the reference metrics.json only (is it the known official run?); it does NOT exercise this '
                      'script\'s scorer', **expected))
        if model_rows and reference['step'] == step:
            worst, compared = metric_differences(model_rows, reference['rows'])
            passed = compared == len(reference['rows']) and all(value <= ANCHOR_TOLERANCE for value in worst.values())
            checks.append(dict(name=f'rescored_model_vs_reference:{label}', status='passed' if passed else 'failed',
                               compared_rows=compared, max_absolute_differences=worst, tolerance=ANCHOR_TOLERANCE,
                               reference_batch_size=reference['batch_size'], batch_size=args.batch_size))
        comparison.extend(side_by_side(rows, reference['rows'], label))
    if not references:
        caveats.append('No --reference-metrics given: the query-set identity anchor against the official evaluator was not run.')
    scorer_anchored = scorer_is_anchored(checks)
    if not scorer_anchored:
        caveats.append('The metric arithmetic of this run (score_parameters, a CPU replica of ev12.score_masked_counts) was '
                       'NOT numerically cross-checked against the evaluator: that needs CUDA (scorer_vs_evaluator_score_masked_'
                       'counts) or --checkpoint (rescored_model_vs_reference). The passed query-count anchors prove an identical '
                       'query set and availability, not identical nll/abs-error arithmetic; that rests on the unit tests.')
    failed = [check['name'] for check in checks if check['status'] == 'failed']
    anchor = dict(passed=not failed, failed=failed, tolerance=ANCHOR_TOLERANCE, allow_anchor_mismatch=args.allow_anchor_mismatch,
                  scorer_numerically_anchored=scorer_anchored, checks=checks)
    if failed and not args.allow_anchor_mismatch:
        anchor_failure(output, failed, dict(anchor_check=anchor, cohort_sha256=cohort, vocabulary=vocabulary, gene_coverage=qa))
    if failed:
        caveats.append(f'Anchor checks failed but were allowed: {failed}')
    (output / 'anchor_mismatch.json').unlink(missing_ok=True)   # stale diagnostic of an earlier failed attempt

    ev12.write_frame(output / 'cells.parquet', cells)
    ev12.write_frame(output / 'masked_count_per_cell.parquet', frame)
    for name in ('gene_ids', 'valid', 'nucleus', 'cytoplasm', 'neighbor_counts'):
        ev12.atomic_array(output / f'query_{name}.npy', values[name])
    ev12.atomic_array(output / 'cell_visible_inputs.npy', np.stack([values[key] for key in
        ('depth_nucleus', 'depth_cytoplasm', 'visible_genes', 'neighbor_depth', 'present_neighbors')], 1))
    ev12.atomic_array(output / 'mode_available.npy', np.stack([available[mode] for mode in ev12.COUNT_MODES], 1))
    for kind, source in (('mu', outcome['totals']), ('theta', outcome['theta']), ('p', outcome['allocations']), ('kappa', outcome['kappa'])):
        for name, value in source.items():
            ev12.atomic_array(output / f'baseline_{kind}_{name}.npy', np.asarray(value, dtype=np.float32))
    fitted = outcome['fitted']
    write_csv(output / 'gene_parameters.csv', [dict(vocab_id=int(gene_vocabulary[index]),
        **{key: float(fitted[key][index]) for key in ('gene_train_queries', 'gene_train_total', 'gene_mean', 'gene_rate',
                                                       'gene_fraction', 'gene_median')}) for index in range(data['genes'])])
    all_rows = rows + model_rows
    write_csv(output / 'metrics.csv', [dict(variant=f'{EXPERIMENT}:{row["baseline"]}', cohort_sha256=cohort, **row) for row in all_rows])
    if comparison:
        write_csv(output / 'comparison.csv', comparison)
    headline = [row for row in comparison if row['split'] == 'val' and row['donor'] == 'pooled']
    report = dict(status='complete', experiment=EXPERIMENT, cells=len(cells), cohort_sha256=cohort,
        canonical_cohort=canonical['status'] == 'passed', no_test=not set(cells['split']) - {'train', 'val'},
        test_donors_used=False, train_donors=train_donors, val_donors=val_donors, seed=42,
        source_sha256={name: ev12.file_sha256(ROOT / name) for name in SOURCES},
        protocol_sha256=ev12.file_sha256(args.prepared / 'protocol.json'), prepared=str(args.prepared.resolve()),
        config=dict(path=None if args.checkpoint else str(args.config.resolve()), v12=used,
                    presence_max_neighbor_distance=ids['distance']),
        vocabulary=vocabulary, checkpoint=checkpoint, lr_prior_loaded=bool(args.checkpoint), batch_size=args.batch_size,
        protocol='Fixed held-out query evaluation of evaluate_subcellular_v12.py (prepare_view_batch_v12 mask=True, seed 42, '
                 'epoch 0, gene_mask_ratio 0, neighbor_dropout 0); baselines fitted on train donors only from visible '
                 'non-query counts; scored by ev12.joint_count_nll and ev12.aggregate_masked_counts definitions',
        masked_count_prediction=protocol, row_fields=dict(evaluator=['split', 'mode', 'donor', *COUNT_FIELDS, *METRICS],
            added=['baseline', 'source', 'total_estimator', 'allocation_estimator', *PART_METRICS, *DIAGNOSTICS]),
        gene_free_estimators=list(GENE_FREE), estimator_inputs={name: sorted(value) for name, value in ESTIMATOR_INPUTS.items()},
        model_mode_inputs={name: sorted(value) for name, value in MODE_INPUTS.items()},
        estimators=dict(
            t_zero='mu at the decoder floor 1e-4 (abs-error reference, not a likelihood model; one global theta)',
            t_cell_mean='cell visible depth / number of visible panel genes (no gene knowledge; one global theta)',
            t_gene_mean='per-gene mean total over train queries',
            t_gene_depth='per-gene rate (sum train T / sum train visible depth) x cell visible depth',
            t_gene_depth_nbr='(1-w) t_gene_depth + w * scale * neighbour pooled rate x cell visible depth; w by train Poisson NLL',
            a_global='pooled train nuclear fraction sum N / sum T (one global kappa)',
            a_cell='cell visible nuclear fraction shrunk toward the train visible fraction (no gene knowledge; one global kappa)',
            a_gene='per-gene train nuclear fraction with pseudo-count shrinkage toward a_global',
            a_gene_cell='logit(a_gene) + logit(shrunk cell visible fraction) - logit(train visible fraction)',
            a_gene_median='per-gene train median of N/T over T>0 (abs-error-optimal, poor likelihood)'),
        fitted=dict({key: fitted[key] for key in ('global_mean', 'global_rate', 'global_fraction', 'visible_fraction',
            'allocation_pseudocount', 'train_queries', 'shrink', 'shrink_losses', 'neighbor_scale', 'neighbor_weight',
            'neighbor_weight_losses')}, shrink_grid=SHRINK_GRID, weight_grid=WEIGHT_GRID, genes=data['genes'],
            min_informative=args.min_informative, dispersion=outcome['dispersion'],
            dispersion_method='maximum likelihood, zooming log10 grid (17 points x 3 rounds) within decoder bounds'),
        parameter_bounds=BOUNDS, target_summary=target_summary(data), gene_coverage=qa, caveats=caveats,
        references=[{key: value for key, value in reference.items() if key != 'rows'} for reference in references],
        anchor_check=anchor, results=all_rows, comparison=comparison, headline_val_pooled=headline,
        headline_val_pooled_matched_inputs=[row for row in headline if row['baseline_inputs_within_model_mode']],
        skill_definition='1 - model_error / baseline_error on the same (split, mode, donor) active query set; like-for-like '
                         'only where baseline_inputs_within_model_mode is true',
        seconds=time.monotonic() - started)
    ev12.atomic_json(output / 'metrics.json', plain(report))
    ev12.atomic_json(output / 'progress.json', dict(stage='complete', metrics='metrics.json'))
    for row in all_rows:
        if row['split'] == 'val' and row['donor'] == 'pooled' and row['mode'] == 'fused':
            print('[count-baseline] val pooled fused-cells ' + row['baseline'] + ' ' + ' '.join(
                f'{metric}={row[metric]:.4f}' for metric in (*METRICS, *PART_METRICS, *DIAGNOSTICS)
                if row.get(metric) is not None), flush=True)
    for row in headline:
        if row['mode'] == 'fused' and row['baseline'] in ('t_gene_depth+a_gene', 't_gene_depth+a_gene_cell'):
            print(f'[count-baseline] skill {row["reference"]} vs {row["baseline"]}: ' + ' '.join(
                f'{metric}={row[f"{metric}_skill"]:+.3f}' for metric in METRICS if row.get(f'{metric}_skill') is not None), flush=True)
    print(f'[done-count-baseline] {len(all_rows)} rows, anchors {"passed" if not failed else "FAILED (allowed)"}', flush=True)


if __name__ == '__main__':
    main()
