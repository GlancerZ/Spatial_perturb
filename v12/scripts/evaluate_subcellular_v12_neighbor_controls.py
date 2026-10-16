"""Untrained neighbor-aggregation controls on the canonical V12 5,200-cell cohort.

Question: is V12's learned 'spatial' readout worth more than "encode every
physical neighbor with frozen ordinary scGPT and average"? No checkpoint is
loaded and nothing is trained. Cohort, frozen baselines, probes and metrics are
the reference evaluator's (evaluate_subcellular_v12), imported, not rewritten.
Run on a compute allocation. Atomic metrics.json is the completion marker.
"""
from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import torch

import evaluate_subcellular_v12 as ev12
from evaluate_subcellular_v12 import CANONICAL_COHORT, FIXED_C, TASKS, atomic_array, atomic_json, file_sha256

EXPERIMENT = 'v12_untrained_neighbor_controls'
ANCHOR_READOUTS = ('scgpt_whole', 'scgpt_nucleus')
BASE_READOUTS = ANCHOR_READOUTS + ('nbr_mean', 'nbr_mean_dw', 'nbr_mean_shuffled', 'nbr_expr_mean')
FUSION_READOUTS = ('whole_plus_nbr', 'whole_plus_nbr_pca512', 'nucleus_plus_nbr', 'nucleus_plus_nbr_pca512',
                   'whole_add_nbr', 'whole_add_nbr_shuffled')
READOUTS = BASE_READOUTS + FUSION_READOUTS
NEIGHBOR_TABLES = ('real_rows', 'real_distance', 'shuffled_rows', 'shuffled_distance')
METRIC = {'cell_type': 'val_macro_f1', 'region': 'val_macro_f1', 'neighbor_composition': 'val_js_distance'}
ANCHOR_TOLERANCE = 2e-3
SHUFFLE_TAG = 1297
# Own-encoding check against the evaluator baseline, calibrated on wrong-row (impostor) pairs.
PATH_TIE_TOLERANCE = .02
PATH_MIN_TOP1 = .9
OFFICIAL_EVAL_ROOT_PREFIX = 'subcellular_v12_evaluations'
# Fields of evaluate_subcellular_v12.frozen_baselines' cache contract that must equal the current checkout.
REFERENCE_CACHE_SOURCES = ('src/spatial_pretrain/models/scgpt_backbone.py', 'src/spatial_pretrain/data/subcellular_v6_left.py',
                           'src/spatial_pretrain/data/subcellular_v5.py', 'scripts/evaluate_subcellular_v12.py')
REFERENCE_CACHE_LITERALS = dict(
    tokenization='deterministic-left V6 packing; unmasked nuclear, cytoplasmic and summed raw whole counts',
    readout='raw CLS; common row L2 normalization occurs only before downstream probes',
    dtype='bf16 autocast float32 stored')
BACKBONE_IDENTITY = ('backbone_kind', 'source_sha256', 'weights_sha256', 'args_sha256', 'vocab_sha256')
# Measured with the official evaluator on this cohort (fixed C, common row L2).
ANCHORS = {('scgpt_whole', 'cell_type'): .545217, ('scgpt_whole', 'region'): .280141,
           ('scgpt_whole', 'neighbor_composition'): .381156,
           ('scgpt_nucleus', 'cell_type'): .566397, ('scgpt_nucleus', 'region'): .212978,
           ('scgpt_nucleus', 'neighbor_composition'): .391247}
V12_REFERENCE = {
    0: dict(spatial=dict(cell_type=.139900, region=.399528), fused=dict(cell_type=.572920, region=.266189),
            local=dict(cell_type=.5724, region=.2671)),
    3400: dict(nucleus=dict(cell_type=.579559, region=.255758), local=dict(cell_type=.588161, region=.271975),
               spatial=dict(cell_type=.137009, region=.391389), fused=dict(cell_type=.569586, region=.373513)),
}
V12_MASKED_COUNT_REFERENCE = dict(step=3400, split='val', donor='pooled', mode='fused', nll=.6006,
                                  total_abs_error=.386, allocation_abs_error=.390,
                                  note='Untrained controls have no count decoder; listed for context only')
SOURCES = ('scripts/evaluate_subcellular_v12_neighbor_controls.py', 'scripts/evaluate_subcellular_v12.py',
           'scripts/evaluate_subcellular_v2.py', 'scripts/probe_lungpf_embeddings.py',
           'src/spatial_pretrain/models/scgpt_backbone.py', 'src/spatial_pretrain/data/subcellular_v12.py',
           'src/spatial_pretrain/data/subcellular_v10.py', 'src/spatial_pretrain/data/subcellular_v6_left.py',
           'src/spatial_pretrain/data/subcellular_v5.py')


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def check_output_dir(path):
    """Cheap launch guard, run before any model load."""
    path = Path(path)
    if any(part.startswith(OFFICIAL_EVAL_ROOT_PREFIX) for part in path.resolve().parts):
        raise ValueError('--output-dir must not live inside the official V12 evaluation root: its summarizer '
                         'collects every metrics.json below that root and would reject this report')
    if (path / 'metrics.json').exists():
        raise FileExistsError('Refusing to overwrite completed metrics')
    if path.is_dir() and not (path / 'control_contract.json').exists() and any(path.iterdir()):
        raise ValueError('Nonempty --output-dir lacks control_contract.json: use a new empty directory and keep '
                         'Slurm/stdout logs outside it')


def contract_differences(saved, current):
    return sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))


def model_input_neighbors(coords, centers, expressed, k, data_radius, model_radius=None):
    """Cells whose CLS reaches V12's spatial encoder in unmasked evaluation.

    Mirrors extract(): real_neighbors (k nearest physical cells of the whole
    sample, center identity excluded) -> encode_v12_views (_neighbor_rows
    dedup/nearest-first, float64 distance <= data radius, a neighbor without
    mapped whole-cell counts is dropped and never replaced) -> the model's
    _neighbor_presence (float32 squared offset <= model radius squared).
    Returns compacted rows [B,k] (-1 padded) and distances [B,k] (inf padded).
    """
    from spatial_pretrain.data.subcellular_v10 import _neighbor_rows
    coords = np.asarray(coords, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.int64)
    expressed = np.asarray(expressed, dtype=bool)
    if coords.ndim != 2 or coords.shape[1] != 2 or expressed.shape != (len(coords),):
        raise ValueError('Expected [N,2] coordinates and [N] expression flags')
    nearest = ev12.real_neighbors(coords, centers, k)
    rows = np.full((len(centers), k), -1, dtype=np.int64)
    distance = np.full((len(centers), k), np.inf, dtype=np.float64)
    dropped = dict(candidates=0, beyond_data_radius=0, beyond_model_radius=0, unexpressed=0)
    for index, (center, nearby) in enumerate(zip(centers, nearest)):
        nodes = np.concatenate(([center], nearby[nearby >= 0])).astype(np.int64)
        local = np.asarray(coords[nodes], dtype=np.float64)
        selected = _neighbor_rows(nodes, local, k)
        offset = local[selected] - local[0]
        length = np.linalg.norm(offset, axis=1)
        dropped['candidates'] += len(selected)
        keep = np.ones(len(selected), dtype=bool) if data_radius is None else length <= float(data_radius)
        dropped['beyond_data_radius'] += int((~keep).sum())
        present = expressed[nodes[selected]]
        dropped['unexpressed'] += int((keep & ~present).sum())
        keep &= present
        if model_radius is not None:
            inside = np.square(offset.astype(np.float32)).sum(axis=-1) <= np.float32(float(model_radius) ** 2)
            dropped['beyond_model_radius'] += int((keep & ~inside).sum())
            keep &= inside
        chosen = nodes[selected][keep]
        if (chosen == center).any() or len(np.unique(chosen)) != len(chosen):
            raise ValueError('Center or duplicate cell among model-input neighbors')
        rows[index, :len(chosen)] = chosen
        distance[index, :len(chosen)] = length[keep]
    return rows, distance, dropped


def shuffled_neighbors(rows, centers, eligible, seed, slice_seed):
    """Same neighbor count per center, uniform without replacement from the same sample.

    Candidates are every eligible (mapped-expression) cell except the center, so
    sample/donor identity is kept and physical locality is destroyed. The RNG is
    keyed by center identity, never by processing order.
    """
    rows = np.asarray(rows, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.int64)
    pool = np.flatnonzero(np.asarray(eligible, dtype=bool))
    if rows.ndim != 2 or len(rows) != len(centers):
        raise ValueError('rows must be [centers,K]')
    result = np.full_like(rows, -1)
    for index, (center, row) in enumerate(zip(centers, rows)):
        count = int((row >= 0).sum())
        if not count:
            continue
        candidates = pool[pool != center]
        if len(candidates) < count:
            raise ValueError('Too few same-sample cells for a shuffled control')
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(slice_seed), int(center), SHUFFLE_TAG]))
        result[index, :count] = rng.choice(candidates, count, replace=False)
    if (result == centers[:, None]).any():
        raise ValueError('Center leaked into shuffled neighbors')
    return result


def table_distances(coords, centers, rows):
    coords = np.asarray(coords, dtype=np.float32).astype(np.float64)
    length = np.linalg.norm(coords[np.maximum(rows, 0)] - coords[np.asarray(centers)][:, None], axis=2)
    return np.where(rows >= 0, length, np.inf)


def _positions(unique, rows):
    unique, rows = np.asarray(unique, dtype=np.int64), np.asarray(rows, dtype=np.int64)
    if rows.ndim != 2 or unique.ndim != 1 or (np.diff(unique) <= 0).any():
        raise ValueError('rows must be [B,K] and unique rows strictly increasing')
    present = rows >= 0
    position = np.zeros(rows.shape, dtype=np.int64)
    if present.any():
        found = np.searchsorted(unique, rows[present])
        if (found >= len(unique)).any() or not np.array_equal(unique[found], rows[present]):
            raise ValueError('A neighbor has no encoded row')
        position[present] = found
    return position, present


def distance_weights(distance, scale):
    if not scale > 0:
        raise ValueError('Distance scale must be positive')
    return np.exp(-np.asarray(distance, dtype=np.float64) / float(scale))


def aggregate_neighbors(encoded, unique, rows, weights=None):
    """Normalized (weighted) mean of per-cell vectors; zero vector when no neighbor."""
    encoded = np.asarray(encoded)
    position, present = _positions(unique, rows)
    if encoded.ndim != 2 or len(encoded) != len(unique):
        raise ValueError('encoded must hold one row per unique cell')
    if not present.any():
        return np.zeros((len(present), encoded.shape[1]), dtype=np.float32)
    weight = present.astype(np.float64) if weights is None else np.where(present, np.asarray(weights, dtype=np.float64), 0.)
    if weight.shape != present.shape or not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Weights must be finite nonnegative [B,K]')
    total = weight.sum(axis=1, keepdims=True)
    if (total[present.any(axis=1)] <= 0).any():
        raise ValueError('Present neighbors need positive total weight')
    weight = np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)
    return np.einsum('bk,bkd->bd', weight, encoded[position].astype(np.float64)).astype(np.float32)


def pooled_log_expression(counts, unique, rows, target=1e4):
    """Sum raw neighbor counts, library-normalize the pool to ``target``, log1p."""
    counts = np.asarray(counts, dtype=np.float64)
    position, present = _positions(unique, rows)
    if counts.ndim != 2 or len(counts) != len(unique) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('counts must be finite nonnegative [unique cells, genes]')
    if not present.any():
        return np.zeros((len(present), counts.shape[1]), dtype=np.float32)
    pooled = (counts[position] * present[..., None]).sum(axis=1)
    library = pooled.sum(axis=1, keepdims=True)
    return np.log1p(pooled / np.maximum(library, 1.) * target).astype(np.float32)


def common_panel(panels):
    """Sorted gene names measured in every sample; absent genes are never zero-filled."""
    names = []
    for panel in panels:
        panel = np.asarray(panel).astype(str)
        if len(np.unique(panel)) != len(panel):
            raise ValueError('Duplicate panel genes')
        names.append(set(panel.tolist()))
    genes = np.asarray(sorted(set.intersection(*names))) if names else np.asarray([])
    if not len(genes):
        raise ValueError('No commonly measured genes')
    return genes


def panel_columns(panel, genes):
    lookup = {str(name): index for index, name in enumerate(np.asarray(panel).astype(str))}
    return np.array([lookup[str(name)] for name in genes], dtype=np.int64)


def fuse(first, second, how):
    """Training-free fusion of row-L2-normalized parts."""
    first, second = ev12.normalize_for_probe(first), ev12.normalize_for_probe(second)
    if len(first) != len(second):
        raise ValueError('Fusion parts must describe the same cells')
    if how == 'concat':
        return np.concatenate((first, second), axis=1)
    if how == 'add':
        if first.shape != second.shape:
            raise ValueError('Additive fusion requires equal dimensions')
        return first + second
    raise ValueError(f'Unknown fusion: {how}')


def fusion_readouts(features, train, components=512):
    values, pca = {}, {}
    for prefix in ('whole', 'nucleus'):
        name = f'{prefix}_plus_nbr'
        values[name] = fuse(features[f'scgpt_{prefix}'], features['nbr_mean'], 'concat')
        values[f'{name}_pca512'], pca[f'{name}_pca512'] = ev12.concat_pca(values[name], train, components=components)
    values['whole_add_nbr'] = fuse(features['scgpt_whole'], features['nbr_mean'], 'add')
    values['whole_add_nbr_shuffled'] = fuse(features['scgpt_whole'], features['nbr_mean_shuffled'], 'add')
    return values, pca


def score_of(result):
    return float(result['fixed_C_result']['val_macro_f1'] if result['task'] in FIXED_C
                 else result['val_result']['js_distance'])


def flat_row(result, common):
    """Same columns as summarize_subcellular_v12's comparison/metrics.csv."""
    row = dict(**common, embedding=result['embedding'], task=result['task'], dimension=result['embedding_dimension'],
               train_cells=result['train_cells'], val_cells=result['val_cells'])
    if result['task'] in FIXED_C:
        score = result['fixed_C_result']
        row.update(C=score['C'], val_macro_f1=score['val_macro_f1'], val_balanced_accuracy=score['val_balanced_accuracy'])
        row.update({f'donor_macro_f1_{donor}': value for donor, value in score['per_donor_macro_f1'].items()})
    else:
        row.update({f'val_{key}': value for key, value in result['val_result'].items()})
        row.update(ridge_alpha=result['ridge_alpha'], mean_val_label_coverage=result['mean_val_label_coverage'])
    return row


def encoder_path_check(own, reference, tie_tolerance=PATH_TIE_TOLERANCE, min_top1=PATH_MIN_TOP1):
    """Do my own whole-cell encodings of the centers line up, row by row, with the evaluator baseline?

    Null hypothesis = a wrong row. After centering on the reference mean, every center's own CLS is compared
    (cosine) with the reference CLS of its own row (matched) and of every other center of the sample
    (impostors). Passing requires each row to be its own best match up to ``tie_tolerance`` (identical-input
    cells and BF16 noise) and a strict top-1 match for at least ``min_top1`` of the rows. Raw cosine and the
    evaluator's step-0 allclose tolerance are reported, not enforced: they have no null and were calibrated
    on identical batch padding.
    """
    own, reference = np.asarray(own, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if own.ndim != 2 or own.shape != reference.shape or len(own) < 2 or not (np.isfinite(own).all() and np.isfinite(reference).all()):
        raise ValueError('Encoder path check needs matching finite [cells>=2, dim] arrays')
    def unit(values):
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    raw = (unit(own) * unit(reference)).sum(axis=1)
    center = reference.mean(axis=0)
    similarity = unit(own - center) @ unit(reference - center).T
    matched = np.diag(similarity).copy()
    wrong = ~np.eye(len(own), dtype=bool)
    best_impostor = np.where(wrong, similarity, -np.inf).max(axis=1)
    margin = matched - best_impostor
    null = similarity[wrong]
    top1 = float((margin > 0).mean())
    error = np.linalg.norm(own - reference, axis=1) / np.maximum(np.linalg.norm(reference - center, axis=1), 1e-12)
    return dict(cells=int(len(own)), passed=bool(margin.min() >= -tie_tolerance and top1 >= min_top1),
        criterion=f'every row: matched centered cosine >= best impostor - {tie_tolerance}; strict top-1 fraction >= {min_top1}',
        min_margin=float(margin.min()), mean_margin=float(margin.mean()), top1_fraction=top1,
        rows_below_tolerance=int((margin < -tie_tolerance).sum()),
        matched_centered_cosine=dict(min=float(matched.min()), mean=float(matched.mean())),
        null_centered_cosine=dict(mean=float(null.mean()), p99=float(np.quantile(null, .99)), max=float(null.max()),
                                  pairs=int(null.size), definition='own row i vs reference row j != i, all pairs of the sample'),
        min_raw_cosine=float(raw.min()), mean_raw_cosine=float(raw.mean()),
        median_error_over_centered_norm=float(np.median(error)),
        max_absolute_difference=float(np.max(np.abs(own - reference))),
        allclose_atol_0p03_rtol_0p02=bool(np.allclose(own, reference, atol=.03, rtol=.02)),
        meaning='own chunked whole-cell encoding of the centers vs the evaluator scgpt_whole baseline (BF16, other batch padding)')


def anchor_check(scores, anchors=None, tolerance=ANCHOR_TOLERANCE, source=None):
    """``source`` counts samples whose anchor embeddings are official cache arrays vs recomputed here."""
    rows = []
    for (name, task), reference in (ANCHORS if anchors is None else anchors).items():
        observed = scores.get((name, task))
        difference = None if observed is None else float(observed) - reference
        rows.append(dict(embedding=name, task=task, metric=METRIC[task], reference=reference,
            observed=None if observed is None else float(observed), difference=difference,
            within_tolerance=bool(difference is not None and np.isfinite(difference) and abs(difference) <= tolerance)))
    differences = [abs(row['difference']) for row in rows if row['difference'] is not None]
    scope = None
    if source is not None:
        scope = ('Anchor embeddings of %d sample(s) are the official evaluator cache arrays, read-only: for those this check '
                 'covers cohort selection and the probe path only, not scGPT encoding; the control encoder is checked per '
                 'sample by encoder_path_check. %d sample(s) were recomputed here by the evaluator frozen_baselines() path.'
                 % (source.get('official_cache_read_only', 0), source.get('evaluator_path_private_cache', 0)))
    return dict(tolerance=tolerance, passed=all(row['within_tolerance'] for row in rows),
                max_abs_difference=max(differences) if differences else None, embedding_source=source, scope=scope, rows=rows)


PER_DONOR_PAIRS = dict(nbr_mean_minus_nbr_mean_shuffled=('nbr_mean', 'nbr_mean_shuffled'),
                       whole_add_nbr_minus_whole_add_nbr_shuffled=('whole_add_nbr', 'whole_add_nbr_shuffled'),
                       nbr_mean_minus_scgpt_whole=('nbr_mean', 'scgpt_whole'))


def comparison_hint(scores, dimensions, per_donor=None):
    """Controls next to the official V12 numbers; a reading aid, not a test.

    ``per_donor`` maps (readout, task) to the probe's per-validation-donor macro-F1.
    """
    hint = {}
    for task in ('region', 'cell_type'):
        controls = [dict(source='untrained_control', readout=name, dimension=dimensions.get(name),
                         val_macro_f1=scores[(name, task)]) for name in READOUTS if (name, task) in scores]
        reference = [dict(source='v12_official_evaluator', step=step, readout=name, dimension=512, val_macro_f1=values[task])
                     for step, block in V12_REFERENCE.items() for name, values in block.items()]
        pairs = dict(
            nbr_mean_minus_v12_spatial_step0=('nbr_mean', V12_REFERENCE[0]['spatial'][task]),
            nbr_mean_minus_v12_spatial_step3400=('nbr_mean', V12_REFERENCE[3400]['spatial'][task]),
            nbr_mean_dw_minus_v12_spatial_step3400=('nbr_mean_dw', V12_REFERENCE[3400]['spatial'][task]),
            nbr_mean_minus_nbr_mean_shuffled=('nbr_mean', scores.get(('nbr_mean_shuffled', task))),
            nbr_mean_shuffled_minus_scgpt_whole=('nbr_mean_shuffled', scores.get(('scgpt_whole', task))),
            nbr_expr_mean_minus_nbr_mean=('nbr_expr_mean', scores.get(('nbr_mean', task))),
            whole_add_nbr_minus_v12_fused_step3400=('whole_add_nbr', V12_REFERENCE[3400]['fused'][task]),
            whole_plus_nbr_pca512_minus_v12_fused_step3400=('whole_plus_nbr_pca512', V12_REFERENCE[3400]['fused'][task]),
            whole_add_nbr_minus_whole_add_nbr_shuffled=('whole_add_nbr', scores.get(('whole_add_nbr_shuffled', task))),
            whole_add_nbr_minus_scgpt_whole=('whole_add_nbr', scores.get(('scgpt_whole', task))))
        deltas = {key: float(scores[(name, task)] - other) for key, (name, other) in pairs.items()
                  if (name, task) in scores and other is not None}
        donor_deltas = {}
        for key, (first, second) in PER_DONOR_PAIRS.items():
            one, other = (per_donor or {}).get((first, task)), (per_donor or {}).get((second, task))
            if one and other:
                donor_deltas[key] = {donor: float(one[donor] - other[donor]) for donor in sorted(set(one) & set(other))}
        hint[task] = dict(metric='val_macro_f1 (higher better)', controls=controls, v12_reference=reference, deltas=deltas,
                          per_val_donor_macro_f1_deltas=donor_deltas)
    hint['neighbor_composition'] = dict(metric='val_js_distance (lower better)',
        controls=[dict(source='untrained_control', readout=name, dimension=dimensions.get(name),
                       val_js_distance=scores[(name, 'neighbor_composition')])
                  for name in READOUTS if (name, 'neighbor_composition') in scores],
        official_reference=[dict(readout=name, val_js_distance=value) for (name, task), value in ANCHORS.items()
                            if task == 'neighbor_composition'])
    hint['v12_masked_count_reference'] = V12_MASKED_COUNT_REFERENCE
    hint['reading_guide'] = [
        'nbr_mean ~ V12 spatial: no evidence on these probes that the learned spatial module adds information '
        'beyond averaging frozen ordinary-scGPT neighbor CLS.',
        'nbr_mean ~ nbr_mean_shuffled: the signal is recoverable from the average expression of random cells of the '
        'same sample (sample-level composition); no evidence that physical locality adds information. Train and '
        'validation donors are disjoint, so this is not donor-identity leakage through the probe.',
        'nbr_mean clearly above nbr_mean_shuffled: physical locality carries signal beyond sample-level composition; '
        'per_val_donor_macro_f1_deltas shows whether that holds in each of the 3 validation donors.',
        'nbr_mean_shuffled keeps every center\'s neighbor count: zero-neighbor centers stay zero vectors, so an '
        'isolated-cell/count indicator survives the shuffle, and the draw pool does not exclude true neighbors '
        '(overlap fraction in sample_qa).',
        'whole_add_nbr vs V12 fused (both 512-d): training-free fusion against the trained fusion.',
        'Single cohort, fixed-C probes, 3 validation donors: differences of a few 1e-3 are within the anchor tolerance.',
        'V12 reference rows are copied constants from the official evaluator, not recomputed here.']
    return hint


def validate_report(report):
    if report.get('status') != 'complete' or report.get('experiment') != EXPERIMENT:
        raise ValueError('Invalid neighbor-control completion marker')
    if report.get('cohort_sha256') != CANONICAL_COHORT or report.get('cells') != 5200:
        raise ValueError('Not the canonical 5200-cell cohort')
    if report.get('test_donors_used') is not False or report.get('trained_parameters') != 0:
        raise ValueError('Controls must be untrained and test-free')
    if report.get('fixed_C') != FIXED_C or len(report.get('train_donors', [])) != 10 or len(report.get('val_donors', [])) != 3:
        raise ValueError('Wrong fixed probe/donor protocol')
    rows = report.get('results', [])
    expected = {(name, task) for name in READOUTS for task in TASKS}
    if len(rows) != len(expected) or {(row['embedding'], row['task']) for row in rows} != expected:
        raise ValueError('All control readouts and all three tasks must complete without duplicates')
    for row in rows:
        if row['embedding_dimension'] != report['dimensions'][row['embedding']]:
            raise ValueError('Probe/readout dimension mismatch')
        if row['task'] in FIXED_C:
            score = row['fixed_C_result']
            if score['C'] != FIXED_C[row['task']] or not score['solver'].get('converged_gradient'):
                raise ValueError('Unconverged/wrong-C classifier')
        elif row.get('ridge_alpha') != 1.:
            raise ValueError('Wrong composition probe')
    check = report.get('anchor_check', {})
    if not check.get('rows') or not (check.get('passed') or report.get('anchor_mismatch_allowed')):
        raise ValueError('Anchor readouts do not reproduce the official evaluator')
    path = report.get('encoder_path_check', {})
    if path.get('samples') != 13 or not (path.get('failed_samples') == [] or report.get('encoder_path_mismatch_allowed')):
        raise ValueError('Control encoder path is not verified against the evaluator baseline in every sample')
    if not {'region', 'cell_type'} <= set(report.get('comparison_hint', {})):
        raise ValueError('Missing comparison hint')


def tokenize_whole(nuclear, cyto, genes, row, slice_seed, max_tokens, pad_token_id, cls_token_id):
    """The scgpt_whole baseline packing, which is also V12's neighbor-stream packing."""
    from spatial_pretrain.data.subcellular_v6_left import encode_whole_cell_v6_left
    return encode_whole_cell_v6_left(nuclear, cyto, genes, pad_token_id=pad_token_id, cls_token_id=cls_token_id,
        max_tokens=max_tokens, seed=np.random.SeedSequence([42, int(slice_seed), int(row), 355]))


def pack_tokens(packed, pad_token_id):
    length = max(len(item['whole_gene_ids']) for item in packed)
    token = np.full((len(packed), length), pad_token_id, dtype=np.int64)
    bins = np.full((len(packed), length), -2., dtype=np.float32)
    valid = np.zeros((len(packed), length), dtype=bool)
    for index, item in enumerate(packed):
        size = len(item['whole_gene_ids'])
        token[index, :size] = item['whole_gene_ids']
        bins[index, :size] = item['whole_input_bins']
        valid[index, :size] = item['whole_token_valid_mask']
    return token, bins, valid


def verify_against_v12_inputs(nuclear, cyto, genes, coords, centers, rows, *, k, max_tokens, query_genes,
                              data_radius, model_radius, pad_token_id, cls_token_id, slice_seed, slice_index, chunk=64):
    """CPU check: the reference data path yields the same neighbor cells and tokens."""
    from spatial_pretrain.data.subcellular_v12 import prepare_view_batch_v12
    centers = np.asarray(centers, dtype=np.int64)
    nearest = ev12.real_neighbors(coords, centers, k)
    tokens = 0
    for start in range(0, len(centers), chunk):
        batch = prepare_view_batch_v12(nuclear, cyto, genes, centers[start:start + chunk], nearest[start:start + chunk], coords,
            pad_token_id=pad_token_id, cls_token_id=cls_token_id, max_tokens=max_tokens, query_genes=query_genes,
            seed=42, epoch=0, slice_seed=slice_seed, slice_index=slice_index, mask=False,
            lr_pairs=None, lr_weights=None, max_neighbor_distance=data_radius)
        present = batch['neighbor_present'].bool().clone()
        if model_radius is not None:
            present &= batch['relative_coords'].float().square().sum(-1).le(float(model_radius) ** 2)
        present = present.numpy()
        identity = batch['neighbor_cell_identity'][..., 1].numpy()
        gene_ids, bins, valid = [batch[f'neighbor_{name}'].numpy() for name in ('gene_ids', 'bins', 'valid')]
        for offset in range(len(present)):
            mine = rows[start + offset]
            if not np.array_equal(np.sort(identity[offset][present[offset]]), np.sort(mine[mine >= 0])):
                raise ValueError(f'Neighbor set differs from the V12 input path at center row {int(centers[start + offset])}')
            for slot in np.flatnonzero(present[offset]):
                row = int(identity[offset, slot])
                item = tokenize_whole(nuclear[row], cyto[row], genes, row, slice_seed, max_tokens, pad_token_id, cls_token_id)
                size = len(item['whole_gene_ids'])
                if (int(valid[offset, slot].sum()) != size or not valid[offset, slot, :size].all()
                        or not np.array_equal(gene_ids[offset, slot, :size], item['whole_gene_ids'])
                        or not np.array_equal(bins[offset, slot, :size], item['whole_input_bins'])):
                    raise ValueError(f'Neighbor tokens differ from the V12 input path at cell row {row}')
                tokens += 1
    return dict(checked_centers=int(len(centers)), checked_neighbor_token_rows=int(tokens))


def reference_cache_matches(contract, expected):
    """Is this evaluator baseline-cache entry what frozen_baselines() would compute from the current checkout?

    Everything the evaluator pins is compared (packing/backbone/evaluator source hashes, tokenization, readout,
    dtype, inputs, selection, original-weight hashes) except the absolute sample/asset paths, counts mtime,
    batch size and torch version; the latter two only rank several matching entries.
    """
    try:
        return bool(contract['schema_version'] == 2 and Path(contract['sample']).name == expected['sample_name']
            and contract['inputs']['obs_sha256'] == expected['obs_sha256']
            and contract['inputs']['genes_sha256'] == expected['genes_sha256']
            and {name: item['size'] for name, item in contract['inputs']['counts'].items()} == expected['count_sizes']
            and contract['selected_sha256'] == expected['selected_sha256']
            and {key: contract['backbone'].get(key) for key in BACKBONE_IDENTITY} == expected['backbone']
            and contract['sources'] == expected['sources']
            and all(contract[key] == value for key, value in REFERENCE_CACHE_LITERALS.items())
            and contract['max_tokens'] == expected['max_tokens'] and contract['seed'] == 42
            and contract['slice_seed'] == expected['slice_seed'])
    except (KeyError, TypeError, AttributeError):
        return False


def load_reference_baselines(cache_root, expected, batch_size, dim, torch_version=None):
    """Strictly read-only reuse of the official evaluator's frozen baseline cache (no lock, no mkdir, no write)."""
    if cache_root is None or not Path(cache_root).is_dir():
        return None, None
    candidates = []
    for marker in sorted(Path(cache_root).glob('*/complete.json')):
        try:
            contract = json.loads(marker.read_text())['contract']
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if reference_cache_matches(contract, expected):
            candidates.append((contract.get('batch_size') != batch_size, contract.get('torch_version') != torch_version,
                               str(marker.parent), contract))
    candidates.sort(key=lambda item: item[:3])
    listed = [dict(path=path, batch_size=contract.get('batch_size'), torch_version=contract.get('torch_version'))
              for _, _, path, contract in candidates]
    for other_batch, other_torch, path, contract in candidates:
        try:
            values = {name: np.load(Path(path) / f'{name}.npy', allow_pickle=False) for name in ANCHOR_READOUTS}
        except (OSError, ValueError):
            continue
        if all(value.shape == (expected['cells'], dim) and np.isfinite(value).all() for value in values.values()):
            return values, dict(path=path, reused=True, read_only_official_cache=True,
                batch_size=contract.get('batch_size'), torch_version=contract.get('torch_version'),
                batch_size_matches=not other_batch, torch_version_matches=not other_torch, matching_entries=listed,
                selection='same batch size, then same torch version, then directory name')
    return None, None


def encode_whole_cells(backbone, nuclear, cyto, genes, rows, slice_seed, max_tokens, batch_size, label=''):
    """Raw whole-cell CLS of original scGPT; same tokens/autocast as frozen_baselines('scgpt_whole')."""
    pad, cls = backbone.vocab['<pad>'], backbone.vocab['<cls>']
    blocks, started = [], time.monotonic()
    for start in range(0, len(rows), batch_size):
        part = rows[start:start + batch_size]
        token, bins, valid = pack_tokens([tokenize_whole(nuclear[row], cyto[row], genes, row, slice_seed,
                                                         max_tokens, pad, cls) for row in part], pad)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            encoded = backbone(torch.as_tensor(token, device='cuda'), torch.as_tensor(bins, device='cuda'),
                               ~torch.as_tensor(valid, device='cuda'), 0.)[:, 0]
        blocks.append(encoded.float().cpu().numpy())
        if (start // batch_size) % 100 == 0:
            print(f'[nbr-ctl] {label} encoded {start + len(part)}/{len(rows)} cells {time.monotonic()-started:.1f}s', flush=True)
    values = np.concatenate(blocks) if blocks else np.zeros((0, backbone.dim), dtype=np.float32)
    if values.shape != (len(rows), backbone.dim) or not np.isfinite(values).all():
        raise ValueError('Invalid frozen scGPT neighbor encoding')
    return values


def process_sample(backbone, cfg, directory, obs, selected, slice_index, common, args):
    started = time.monotonic()
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not obs.full_cell_id.is_unique:
        raise ValueError('Invalid coordinates or duplicate cells')
    settings = cfg.get('v12', {})
    k, max_tokens = int(settings.get('neighbors', 8)), int(settings.get('max_tokens', 512))
    if int(cfg.get('model', {}).get('neighbors', k)) != k:
        raise ValueError('V12 data/model neighbor count mismatch')
    data_radius = settings.get('max_neighbor_distance', 100.0)
    model_radius = cfg.get('model', {}).get('max_neighbor_distance')
    vocab = backbone.vocab
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
    nuclear, cyto = np.asarray(nuclear_all[:, keep]), np.asarray(cyto_all[:, keep])
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    signature = ev12.sample_inputs(directory)
    expected = dict(sample_name=directory.name, obs_sha256=signature['obs_sha256'], genes_sha256=signature['genes_sha256'],
        count_sizes={name: item['size'] for name, item in signature['counts'].items()},
        selected_sha256=hashlib.sha256(np.asarray(selected, dtype=np.int64).tobytes()).hexdigest(),
        backbone={key: backbone.provenance[key] for key in BACKBONE_IDENTITY},
        sources={name: file_sha256(ROOT / name) for name in REFERENCE_CACHE_SOURCES},
        max_tokens=max_tokens, slice_seed=slice_seed, cells=len(selected))
    frozen, cache_info = load_reference_baselines(args.reference_baseline_cache, expected, args.batch_size, backbone.dim,
                                                  torch.__version__)
    if frozen is None:
        # The evaluator's own path, but into a private cache below --output-dir.
        private = (args.output_dir / 'baseline_cache', None)
        for attempt, root in enumerate(private):
            try:
                root = Path(tempfile.mkdtemp(prefix='v12ctl_baseline_')) if root is None else root
                frozen, cache_info = ev12.frozen_baselines(backbone, cfg, directory, selected, nuclear, cyto, genes,
                                                           slice_seed, args.batch_size, root)
                break
            except OSError as error:
                # A filesystem without flock support: recompute in node-local temporary storage.
                if attempt or error.errno not in (errno.ENOSYS, errno.ENOLCK, errno.EOPNOTSUPP):
                    raise
                print(f'[nbr-ctl] {directory.name} private cache lock unsupported ({error}); using TMPDIR', flush=True)
        cache_info = dict(cache_info, read_only_official_cache=False)
    expressed = (nuclear > 0).any(axis=1) | (cyto > 0).any(axis=1)
    if not expressed[selected].all():
        raise ValueError('A center has zero mapped expression; the reference evaluator rejects it')
    rows, distance, dropped = model_input_neighbors(coords, selected, expressed, k, data_radius, model_radius)
    shuffled = shuffled_neighbors(rows, selected, expressed, args.shuffle_seed, slice_seed)
    verification = dict(checked_centers=0, checked_neighbor_token_rows=0)
    if args.verify_centers:
        limit = min(args.verify_centers, len(selected))
        verification = verify_against_v12_inputs(nuclear, cyto, genes, coords, selected[:limit], rows[:limit], k=k,
            max_tokens=max_tokens, query_genes=int(settings.get('query_genes', 64)), data_radius=data_radius,
            model_radius=model_radius, pad_token_id=vocab['<pad>'], cls_token_id=vocab['<cls>'],
            slice_seed=slice_seed, slice_index=slice_index)
        print(f'[nbr-ctl] {directory.name} V12 input path verified {verification}', flush=True)
    real = np.unique(rows[rows >= 0])
    unique = np.unique(np.concatenate((real, shuffled[shuffled >= 0], selected)))
    encoded = encode_whole_cells(backbone, nuclear, cyto, genes, unique, slice_seed, max_tokens, args.batch_size, directory.name)
    path_check = encoder_path_check(encoded[np.searchsorted(unique, selected)], frozen['scgpt_whole'])
    print(f'[nbr-ctl] {directory.name} encoder path check passed={path_check["passed"]} min_margin={path_check["min_margin"]:.4f} '
          f'top1={path_check["top1_fraction"]:.4f} null_max={path_check["null_centered_cosine"]["max"]:.4f}', flush=True)
    if not path_check['passed']:
        atomic_json(args.output_dir / 'encoder_path_check_failed.json', dict(sample=directory.name,
            allowed=bool(args.allow_encoder_path_mismatch), frozen_baseline_cache=cache_info, **path_check))
        if not args.allow_encoder_path_mismatch:
            raise ValueError(f'Neighbor encoder path disagrees with scgpt_whole baseline: {path_check}')
    values = dict(scgpt_whole=frozen['scgpt_whole'], scgpt_nucleus=frozen['scgpt_nucleus'],
        nbr_mean=aggregate_neighbors(encoded, unique, rows),
        nbr_mean_dw=aggregate_neighbors(encoded, unique, rows, distance_weights(distance, args.distance_scale_um)),
        nbr_mean_shuffled=aggregate_neighbors(encoded, unique, shuffled))
    columns = panel_columns(panel, common)
    counts = (np.asarray(nuclear_all[real], dtype=np.float64) + np.asarray(cyto_all[real], dtype=np.float64))[:, columns]
    values['nbr_expr_mean'] = pooled_log_expression(counts, real, rows)
    shuffled_distance = table_distances(coords, selected, shuffled)
    values.update(real_rows=rows, real_distance=distance, shuffled_rows=shuffled, shuffled_distance=shuffled_distance)
    count = (rows >= 0).sum(axis=1)
    overlap = np.array([np.isin(a[a >= 0], b[b >= 0]).sum() for a, b in zip(shuffled, rows)])
    def mean(table, mask):
        return float(table[mask].mean()) if mask.any() else None
    qa = dict(sample=directory.name, cells_in_sample=int(len(obs)), selected_cells=int(len(selected)),
        expressed_cells=int(expressed.sum()), panel_genes=int(len(panel)), matched_genes=int(len(keep)),
        common_panel_genes=int(len(common)), neighbors=k, max_tokens=max_tokens, data_radius_um=data_radius,
        model_radius_um=model_radius, neighbor_count_histogram=np.bincount(count, minlength=k + 1).tolist(),
        zero_neighbor_centers=int((count == 0).sum()), mean_neighbors=float(count.mean()), dropped=dropped,
        unique_real_neighbor_cells=int(len(real)), unique_shuffled_cells=int(len(np.unique(shuffled[shuffled >= 0]))),
        encoded_unique_cells=int(len(unique)), mean_real_distance_um=mean(distance, rows >= 0),
        mean_shuffled_distance_um=mean(shuffled_distance, shuffled >= 0),
        shuffled_cells_that_are_real_neighbors_fraction=float(overlap.sum() / max(int(count.sum()), 1)),
        v12_input_path_verification=verification, encoder_path_check=path_check, frozen_baseline_cache=cache_info,
        seconds=time.monotonic() - started)
    return values, qa


def neighbor_frame(obs, selected, sample, values):
    ids = obs.full_cell_id.to_numpy()
    frames = []
    for kind in ('real', 'shuffled'):
        table, distance = values[f'{kind}_rows'], values[f'{kind}_distance']
        center, slot = np.nonzero(table >= 0)
        frames.append(pd.DataFrame(dict(sample=sample, kind=kind, center_full_cell_id=ids[selected[center]], slot=slot,
            neighbor_row=table[center, slot], neighbor_full_cell_id=ids[table[center, slot]],
            distance_um=distance[center, slot])))
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/benchmarking_data/lungpf/prepared')
    parser.add_argument('--output-dir', '--output', required=True, type=Path)
    parser.add_argument('--batch-size', default=8, type=int, help='8 reproduces the official V12 evaluation batches')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/subcellular_v12_compartment_20260930.yaml')
    parser.add_argument('--assets-dir', type=Path, help='Original scGPT assets; default is assets_dir of --config')
    parser.add_argument('--reference-baseline-cache', type=Path, default=ROOT / 'outputs/subcellular_v12_original_baseline_cache',
                        help='Official evaluator cache, only ever read; missing/unmatched entries are recomputed privately')
    parser.add_argument('--composition-radius-um', default=100., type=float)
    parser.add_argument('--distance-scale-um', default=50., type=float)
    parser.add_argument('--shuffle-seed', default=42, type=int)
    parser.add_argument('--verify-centers', default=400, type=int, help='Centers per sample checked against the V12 data path; 0 skips')
    parser.add_argument('--anchor-tolerance', default=ANCHOR_TOLERANCE, type=float)
    parser.add_argument('--allow-anchor-mismatch', action='store_true',
                        help='Finish although the scgpt_whole/scgpt_nucleus probe scores miss the official numbers')
    parser.add_argument('--allow-encoder-path-mismatch', action='store_true',
                        help='Finish although this script\'s own center encodings fail the row-identity check against scgpt_whole')
    args = parser.parse_args()
    if args.batch_size < 1 or args.composition_radius_um <= 0 or args.distance_scale_um <= 0 or args.verify_centers < 0:
        parser.error('Batch size, radius and scale must be positive; verify-centers nonnegative')
    check_output_dir(args.output_dir)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA on an authorized compute allocation is required')
    import yaml
    from spatial_pretrain.models.scgpt_backbone import ScGPTBackbone
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'[nbr-ctl] official baseline cache {args.reference_baseline_cache}: '
          + ('found, matching entries are reused read-only' if args.reference_baseline_cache.is_dir() else
             'MISSING, anchors are recomputed by the evaluator path into --output-dir/baseline_cache'), flush=True)
    started = time.monotonic()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    cfg = yaml.safe_load(args.config.resolve().read_text())
    if args.assets_dir is not None:
        cfg = dict(cfg, assets_dir=str(args.assets_dir))
    # Original assets only; no fine-tuned checkpoint is ever opened by this script.
    backbone = ScGPTBackbone(cfg['assets_dir'], checkpointing=False, backbone_kind='scgpt')
    backbone = backbone.eval().requires_grad_(False).cuda()
    protocol_path = args.prepared / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    metadata, classes, panels = [], set(), []
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
            classes.update(label for label in obs.loc[ev12.valid_labels(obs.final_CT), 'final_CT'].astype(str)
                           if label not in ev12.UNKNOWN_COMPOSITION_LABELS)
        metadata.append((index, directory, obs, split))
        panels.append(np.load(directory / 'genes.npy', allow_pickle=False).astype(str))
    classes = sorted(classes)
    if len(classes) < 2:
        raise ValueError('Insufficient training-donor neighbor classes')
    common = common_panel(panels)
    sources = {name: file_sha256(ROOT / name) for name in SOURCES}
    contract = dict(schema_version=1, experiment=EXPERIMENT, source_sha256=sources, config_sha256=ev12.fingerprint(cfg),
        config_file_sha256=file_sha256(args.config), backbone_provenance=backbone.provenance,
        protocol_sha256=file_sha256(protocol_path), prepared=str(args.prepared.resolve()), canonical_cohort=CANONICAL_COHORT,
        fixed_C=FIXED_C, readouts=READOUTS, tasks=TASKS, seed=42, shuffle_seed=args.shuffle_seed, shuffle_tag=SHUFFLE_TAG,
        composition_radius_um=args.composition_radius_um, distance_scale_um=args.distance_scale_um,
        composition_classes=classes, ridge_alpha=1., batch_size=args.batch_size,
        common_panel_sha256=ev12.ids_hash(common), common_panel_genes=len(common),
        reference_baseline_cache=str(args.reference_baseline_cache), probe_row_normalization='L2 for every readout before train-only StandardScaler',
        input_signatures={directory.name: ev12.sample_inputs(directory) for _, directory, _, _ in metadata})
    contract = json.loads(json.dumps(contract))
    contract_path = args.output_dir / 'control_contract.json'
    if contract_path.exists():
        saved = json.loads(contract_path.read_text())
        if saved != contract:
            raise ValueError(f'Output belongs to a different source/cohort/protocol (changed: {contract_differences(saved, contract)}); '
                             'an edited script or another --batch-size, --reference-baseline-cache, seed, radius or scale '
                             'needs a new --output-dir')
    else:
        if any(args.output_dir.iterdir()):
            raise ValueError('Nonempty output lacks verified control contract')
        atomic_json(contract_path, contract)
    frames, targets, coverage_blocks, qa, tables = [], [], [], [], []
    blocks = {name: [] for name in BASE_READOUTS}
    input_counts, inside_radius = [], []
    for index, directory, obs, split in metadata:
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy(); chosen['split'] = split
        cache = args.output_dir / 'slices' / directory.name
        cache.mkdir(parents=True, exist_ok=True)
        sample_contract = dict(cell_ids_sha256=ev12.ids_hash(chosen.full_cell_id), **ev12.sample_inputs(directory))
        if (cache / 'complete.json').exists():
            entry = json.loads((cache / 'complete.json').read_text())
            if entry['contract'] != sample_contract:
                raise ValueError('Slice cache input mismatch')
            values = {name: np.load(cache / f'{name}.npy', allow_pickle=False) for name in BASE_READOUTS + NEIGHBOR_TABLES}
            item_qa = entry['qa']
        else:
            values, item_qa = process_sample(backbone, cfg, directory, obs, selected, index, common, args)
            for name, value in values.items():
                atomic_array(cache / f'{name}.npy', value)
            atomic_json(cache / 'complete.json', dict(contract=sample_contract, qa=item_qa))
        if not item_qa['encoder_path_check']['passed'] and not args.allow_encoder_path_mismatch:
            raise ValueError(f'{directory.name}: cached slice failed the encoder path check; see encoder_path_check_failed.json')
        for name in BASE_READOUTS:
            value = values[name]
            if value.ndim != 2 or len(value) != len(selected) or not np.isfinite(value).all():
                raise ValueError(f'Invalid control embedding: {name}')
            blocks[name].append(value)
        if (values['real_rows'] == selected[:, None]).any() or (values['shuffled_rows'] == selected[:, None]).any():
            raise ValueError('Center leaked into its own neighbor aggregate')
        target, coverage = ev12.neighbor_composition(obs, selected, classes, args.composition_radius_um)
        targets.append(target); coverage_blocks.append(coverage); frames.append(chosen); qa.append(item_qa)
        tables.append(neighbor_frame(obs, selected, directory.name, values))
        input_counts.append((values['real_rows'] >= 0).sum(axis=1))
        inside_radius.append(((values['real_rows'] >= 0) & (values['real_distance'] <= args.composition_radius_um)).sum(axis=1))
        atomic_json(args.output_dir / 'progress.json', dict(stage='embedding', completed_slices=len(frames),
            total_slices=len(metadata), last_sample=directory.name, seconds=time.monotonic() - started))
        print(f'[nbr-ctl] {directory.name} done {len(frames)}/{len(metadata)} zero-neighbor centers='
              f'{item_qa["zero_neighbor_centers"]} {time.monotonic()-started:.1f}s', flush=True)
    backbone_provenance = dict(backbone.provenance)
    del backbone
    torch.cuda.empty_cache()
    obs = pd.concat(frames, ignore_index=True)
    cohort = ev12.ids_hash(obs.full_cell_id)
    train_donors = sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique().tolist())
    val_donors = sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique().tolist())
    if cohort != CANONICAL_COHORT or len(obs) != 5200 or not obs.full_cell_id.is_unique or len(train_donors) != 10 or len(val_donors) != 3:
        raise ValueError(f'Not canonical train10/val3 cohort: {cohort}')
    train = obs['split'].eq('train').to_numpy()
    features = {name: np.concatenate(value) for name, value in blocks.items()}
    fused, pca = fusion_readouts(features, train)
    features.update(fused)
    dimensions = {name: int(features[name].shape[1]) for name in READOUTS}
    target = np.concatenate(targets)
    coverage = {key: np.concatenate([block[key] for block in coverage_blocks]) for key in coverage_blocks[0]}
    input_counts, inside_radius = np.concatenate(input_counts), np.concatenate(inside_radius)
    ev12.write_frame(args.output_dir / 'cells.parquet', obs)
    ev12.write_frame(args.output_dir / 'neighbor_table.parquet', pd.concat(tables, ignore_index=True))
    atomic_array(args.output_dir / 'neighbor_composition_targets.npy', target)
    atomic_array(args.output_dir / 'common_panel_genes.npy', common)
    atomic_json(args.output_dir / 'neighbor_composition_classes.json', classes)
    for name in READOUTS:
        atomic_array(args.output_dir / f'{name}_embeddings.npy', features[name])
    diagnostics = ev12.representation_diagnostics(obs, {name: features[name] for name in READOUTS})
    atomic_json(args.output_dir / 'diagnostics.json', diagnostics)
    probe_features = {name: ev12.normalize_for_probe(features[name]) for name in READOUTS}
    probe_dir = args.output_dir / 'probes'; probe_dir.mkdir(exist_ok=True)
    baseline_sources = dict(
        official_cache_read_only=sum(bool(item['frozen_baseline_cache'].get('read_only_official_cache')) for item in qa),
        evaluator_path_private_cache=sum(not item['frozen_baseline_cache'].get('read_only_official_cache') for item in qa))
    results, scores = [], {}
    for name in READOUTS:
        for task in TASKS:
            cache = probe_dir / f'{name}_{task}.json'
            if cache.is_file():
                row = json.loads(cache.read_text())
                if row['embedding'] != name or row['task'] != task or not (args.output_dir / f'{name}_{task}_val_predictions.parquet').is_file():
                    raise ValueError('Invalid cached probe')
            else:
                row = (ev12.classification_probe(obs, probe_features[name], name, task, args.output_dir) if task in FIXED_C
                       else ev12.composition_probe(obs, probe_features[name], target, coverage, classes, name, args.output_dir))
                atomic_json(cache, row)
            results.append(row); scores[(name, task)] = score_of(row)
            atomic_json(args.output_dir / 'progress.json', dict(stage='probing', completed_probes=len(results),
                total_probes=len(READOUTS) * len(TASKS), seconds=time.monotonic() - started))
        if name == ANCHOR_READOUTS[-1]:
            # Fail before the controls: an unreproduced anchor voids every comparison.
            check = anchor_check(scores, tolerance=args.anchor_tolerance, source=baseline_sources)
            print(f'[nbr-ctl] anchor check passed={check["passed"]} max|diff|={check["max_abs_difference"]} '
                  f'embedding_source={baseline_sources}', flush=True)
            if not check['passed']:
                atomic_json(args.output_dir / 'anchor_check_failed.json', dict(anchor_check=check, cohort_sha256=cohort,
                    batch_size=args.batch_size, torch_version=torch.__version__, allowed=bool(args.allow_anchor_mismatch),
                    frozen_baseline_cache=[item['frozen_baseline_cache'] for item in qa],
                    hint='This --output-dir (cached slices and probes) can only be continued with --allow-anchor-mismatch. '
                         'Trying another --batch-size or --reference-baseline-cache (the official evaluation used batch 8 '
                         'and its own baseline cache) changes control_contract.json and needs a NEW empty --output-dir.'))
                if not args.allow_anchor_mismatch:
                    atomic_json(args.output_dir / 'progress.json', dict(stage='anchor_mismatch', diagnostic='anchor_check_failed.json'))
                    print(f'[nbr-ctl] ANCHOR MISMATCH {json.dumps(check)}', flush=True)
                    raise SystemExit(3)
    check = anchor_check(scores, tolerance=args.anchor_tolerance, source=baseline_sources)
    per_donor = {(row['embedding'], row['task']): row['fixed_C_result']['per_donor_macro_f1'] for row in results if row['task'] in FIXED_C}
    path_failed = [item['sample'] for item in qa if not item['encoder_path_check']['passed']]
    split_mask = {split: obs['split'].eq(split).to_numpy() for split in ('train', 'val')}
    composition_qa = {split: {key: float(value[mask].mean()) for key, value in coverage.items()} for split, mask in split_mask.items()}
    neighbor_sets = dict(
        model_input=f'k<={qa[0]["neighbors"]} nearest physical cells of the same sample, center identity and duplicates excluded, '
                    f'float64 distance <= {qa[0]["data_radius_um"]} um (data) and float32 squared offset <= model radius '
                    f'{qa[0]["model_radius_um"]} um, cells without counts in the scGPT-mapped panel dropped and not replaced',
        composition_target=f'every cell within {args.composition_radius_um} um, center excluded, any number, '
                           'known train-vocabulary final_CT labels only',
        difference='The target counts all cells inside the radius; the input holds at most k nearest expressed cells. '
                   'They overlap; this is environment readability, not held-out neighbor prediction.',
        mean_input_neighbors={split: float(input_counts[mask].mean()) for split, mask in split_mask.items()},
        mean_composition_radius_cells={split: float(coverage['total'][mask].mean()) for split, mask in split_mask.items()},
        input_neighbors_inside_composition_radius_fraction=float(inside_radius.sum() / max(int(input_counts.sum()), 1)),
        input_share_of_composition_radius_cells=float(inside_radius.sum() / max(int(coverage['total'].sum()), 1)),
        zero_neighbor_centers={split: int((input_counts[mask] == 0).sum()) for split, mask in split_mask.items()},
        zero_neighbor_policy='zero vector for nbr_* readouts; fusions keep only the center part',
        neighbor_count_histogram=np.bincount(input_counts, minlength=qa[0]['neighbors'] + 1).tolist(),
        dropped={key: int(sum(item['dropped'][key] for item in qa)) for key in qa[0]['dropped']})
    metrics_path = args.output_dir / 'metrics.json'
    common_columns = dict(variant=EXPERIMENT, step=None, checkpoint_sha256=None, config_sha256=contract['config_sha256'],
                          cohort_sha256=cohort, metrics_path=str(metrics_path))
    report = dict(status='complete', experiment=EXPERIMENT, variant=EXPERIMENT, step=None, checkpoint=None,
        trained_parameters=0, cells=len(obs), cohort_sha256=cohort, canonical_cohort=True, no_test=True, test_donors_used=False,
        train_donors=train_donors, val_donors=val_donors, fixed_C=FIXED_C, seed=42, shuffle_seed=args.shuffle_seed,
        source_sha256=sources, config=cfg, config_sha256=contract['config_sha256'], backbone_provenance=backbone_provenance,
        baseline_separate_pretrained_encoder=True, protocol_sha256=contract['protocol_sha256'],
        baseline_sources=baseline_sources,
        baseline_cache_policy='Official evaluator cache entries are read (never written) only when schema, sample, obs/genes '
            'hashes, counts sizes, selection, original-backbone hashes, the four packing/backbone/evaluator source hashes and the '
            'tokenization/readout/dtype strings equal the current checkout; absolute paths, counts mtime, batch size and torch '
            'version are not compared (the last two rank several matches, all listed in sample_qa). Otherwise the evaluator '
            'frozen_baselines() path recomputes into a private cache.',
        contract_sha256=ev12.fingerprint(contract), evaluator_sha256=sources['scripts/evaluate_subcellular_v12.py'],
        solver_sha256=sources['scripts/probe_lungpf_embeddings.py'],
        input_signatures_sha256=ev12.fingerprint(contract['input_signatures']), dimensions=dimensions, pca=pca,
        probe_row_normalization=contract['probe_row_normalization'],
        probe_protocol='Reference evaluator functions: common row L2, train-only StandardScaler, balanced logistic fixed C '
                       '(cell_type 0.1, region 0.01), Ridge alpha=1 then simplex and JS distance for neighbor composition; no search',
        readout_protocol=dict(
            scgpt_whole='original ordinary scGPT raw CLS, deterministic-left 51-bin packing of nucleus+cytoplasm counts (evaluator frozen_baselines)',
            scgpt_nucleus='same, nucleus counts only',
            nbr_mean='unweighted mean of frozen scgpt_whole-style CLS over the V12 model-input neighbors',
            nbr_mean_dw=f'same neighbors, weights exp(-d/{args.distance_scale_um} um) normalized per center',
            nbr_mean_shuffled='same count per center, cells drawn uniformly without replacement from the same sample '
                              '(mapped-expression cells, center excluded)',
            nbr_expr_mean=f'no scGPT: pooled raw nucleus+cytoplasm counts of the real neighbors over {len(common)} '
                          'common panel genes (sorted names), library 1e4, log1p',
            whole_plus_nbr='concat[L2(scgpt_whole), L2(nbr_mean)]', nucleus_plus_nbr='concat[L2(scgpt_nucleus), L2(nbr_mean)]',
            pca512='evaluator concat_pca: train-only StandardScaler + PCA(512, seed 42)',
            whole_add_nbr='L2(scgpt_whole) + L2(nbr_mean)', whole_add_nbr_shuffled='L2(scgpt_whole) + L2(nbr_mean_shuffled)'),
        composition_protocol=dict(radius_um=args.composition_radius_um, center_excluded=True, labels='external existing final_CT',
            class_vocabulary='all nonmissing labels in train donors only', classes=classes, coverage=composition_qa),
        neighbor_sets=neighbor_sets, common_panel_genes=int(len(common)),
        inference=dict(mask=False, dtype='bfloat16', batch_size=args.batch_size, unique_neighbor_cells_encoded_once_per_sample=True),
        sample_qa=qa, diagnostics=diagnostics, anchor_check=check, anchor_mismatch_allowed=bool(args.allow_anchor_mismatch),
        encoder_path_check=dict(samples=len(qa), failed_samples=path_failed,
            min_margin=min(item['encoder_path_check']['min_margin'] for item in qa),
            min_top1_fraction=min(item['encoder_path_check']['top1_fraction'] for item in qa),
            max_null_centered_cosine=max(item['encoder_path_check']['null_centered_cosine']['max'] for item in qa),
            criterion=qa[0]['encoder_path_check']['criterion'], per_sample='sample_qa[*].encoder_path_check'),
        encoder_path_mismatch_allowed=bool(args.allow_encoder_path_mismatch),
        comparison_hint=comparison_hint(scores, dimensions, per_donor), results=results, seconds=time.monotonic() - started)
    validate_report(report)
    write_csv(args.output_dir / 'metrics.csv', [flat_row(row, common_columns) for row in results])
    atomic_json(metrics_path, report)
    atomic_json(args.output_dir / 'progress.json', dict(stage='complete', metrics='metrics.json'))
    print(f'[done-nbr-ctl] {len(READOUTS)} readouts x {len(TASKS)} tasks complete; anchors passed={check["passed"]}', flush=True)


if __name__ == '__main__':
    main()
