"""Inference-time input ablations of one trained V12 checkpoint.

Which inputs does the trained model use: real neighbor content, the LR attention
bias, the nucleus/cytoplasm distinction? Same canonical train10/val3 5,200-cell
cohort, frozen ordinary-scGPT baselines, probes and held-out count scoring as
scripts/evaluate_subcellular_v12.py (imported, never re-implemented). Only batch
tensors or the patch rows handed to the unchanged V12 encoder are perturbed;
weights, labels and count targets are untouched. Run on a compute allocation.

none               unperturbed; ANCHOR against the official evaluator.
shuffle_neighbors  every present real neighbor is replaced by a cell drawn
                   uniformly (fixed seed) from the same slide, excluding the
                   center and its real neighbors where possible. Slots keep the
                   real neighbors' relative coordinates and presence. Tokens and
                   both LR matrices are recomputed for the substitutes by the
                   unchanged encode_v12_views path.
drop_neighbors     neighbor_present all False. The model's own presence rule
                   then yields a zero spatial readout and fused == local.
lr_off             neighbor_lr_bias and center_lr_bias zeroed; identical to LR
                   strength 0. Diagnostic-only coverage tensors are kept.
lr_shuffled        LR evidence reassigned by a fixed-seed random cycle within
                   each center, over exactly the entries the model reads:
                   center_lr_bias among its present neighbors, neighbor_lr_bias
                   among the receiver/sender edges of its own graph (present and
                   within spatial_radius of each other). The consumed multiset
                   of evidence is preserved, its edge assignment is destroyed.
                   The ligand-receptor prior itself is not re-paired.
swap_compartments  center nucleus and cytoplasm gene_ids/bins/valid/log_counts/
                   present and the N/C columns of center_statistics exchanged.
                   Center LR uses N+C on jointly visible genes, so it is
                   symmetric and needs no change. Targets stay as-is; scores
                   against exchanged targets are reported additionally.
whole_as_nucleus   center patch row rewritten before the unchanged encoder:
                   nucleus := N+C, cytoplasm := 0. Bins, log-counts, statistics
                   and presence are recomputed by encode_v12_views; cytoplasm
                   becomes absent. Neighbors and LR are unchanged (they use
                   N+C). Count targets are restored from the unperturbed batch.

Two probe readings per perturbed condition. probe_fit=refit refits scaler, probe
and concat PCA on the condition's own train embeddings (information still
linearly readable; a lower bound on dependence, since a refit absorbs systematic
shifts). probe_fit=none_train fits them on the unperturbed train embeddings and
applies them to the perturbed validation embeddings (reliance of the trained
representation on that input). Count rows and embedding_shift_vs_none need no
probe. shuffle_neighbors draws from the same slide, i.e. the same donor, so
slide-level composition survives it; read it next to drop_neighbors.
"""
from __future__ import annotations

import argparse
import csv
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

import evaluate_subcellular_v12 as ev12

ABLATIONS = ('none', 'shuffle_neighbors', 'drop_neighbors', 'lr_off', 'lr_shuffled',
             'swap_compartments', 'whole_as_nucleus')
BASELINE_LABEL = 'frozen_baseline'
REFIT, TRANSFER = 'refit', 'none_train'
LOG_SUFFIXES = ('.log', '.out', '.err')
READOUTS = ev12.MODEL_READOUTS + ('concat_pca512',)
COUNT_TASK = 'masked_count_prediction'
COUNT_METRICS = ('nll', 'total_abs_error', 'nucleus_abs_error', 'cytoplasm_abs_error', 'allocation_abs_error')
PROBE_METRICS = ('val_macro_f1', 'val_balanced_accuracy', 'val_mae', 'val_js_divergence', 'val_js_distance')
HIGHER_IS_BETTER = {'val_macro_f1', 'val_balanced_accuracy'}
PRIMARY_METRIC = dict(cell_type='val_macro_f1', region='val_macro_f1', neighbor_composition='val_js_distance')
COMPARTMENT_FIELDS = ('gene_ids', 'bins', 'valid', 'log_counts', 'present')
LR_BIAS_KEYS = ('neighbor_lr_bias', 'center_lr_bias')
TARGET_KEYS = ('query_nucleus_counts', 'query_cytoplasm_counts', 'query_total_counts')
SOURCES = dict(shuffle_neighbors='shuffled', whole_as_nucleus='whole')
DIAGNOSTIC_KEYS = ('fusion_residual_strength', 'local_nucleus_weight', 'spatial_attention_entropy',
                   'paired_fraction', 'lr_coverage_on_edges', 'center_lr_coverage')
SHUFFLE_TAG, LR_TAG = 1291, 1293
ANCHOR_TOLERANCE = 2e-3
OFFICIAL_VARIANT = 'subcellular_v12_compartment_20260930'
BASELINE_ANCHORS = {
    ('scgpt_whole', 'cell_type'): .545217, ('scgpt_whole', 'region'): .280141,
    ('scgpt_whole', 'neighbor_composition'): .381156,
    ('scgpt_nucleus', 'cell_type'): .566397, ('scgpt_nucleus', 'region'): .212978,
    ('scgpt_nucleus', 'neighbor_composition'): .391247}
STEP_ANCHORS = {
    0: {('spatial', 'cell_type'): .139900, ('spatial', 'region'): .399528,
        ('fused', 'cell_type'): .572920, ('fused', 'region'): .266189,
        ('local', 'cell_type'): .5724, ('local', 'region'): .2671},
    3400: {('nucleus', 'cell_type'): .579559, ('nucleus', 'region'): .255758,
           ('local', 'cell_type'): .588161, ('local', 'region'): .271975,
           ('spatial', 'cell_type'): .137009, ('spatial', 'region'): .391389,
           ('fused', 'cell_type'): .569586, ('fused', 'region'): .373513}}
STEP_COUNT_ANCHORS = {3400: {('val', 'fused', 'nll'): .6006, ('val', 'fused', 'total_abs_error'): .386,
                             ('val', 'fused', 'allocation_abs_error'): .390}}
DESCRIPTIONS = dict(
    none='Unperturbed reference path; anchor for the official evaluator.',
    shuffle_neighbors='Present real neighbors replaced by fixed-seed uniform same-slide cells (center and real neighbors '
                      'excluded where possible); real relative coordinates and presence kept; tokens and both LR matrices '
                      'recomputed for the substitutes by encode_v12_views. A slide is one donor, so slide-level '
                      'composition is kept: deltas bound the neighbor-content contribution from below.',
    drop_neighbors='neighbor_present set False for every slot; model presence handling gives zero spatial and fused == local.',
    lr_off='neighbor_lr_bias and center_lr_bias zeroed (equivalent to LR strength 0); coverage diagnostics untouched.',
    lr_shuffled='Within each center, a fixed-seed random cycle reassigns center_lr_bias among the neighbors the model '
                'treats as present and neighbor_lr_bias among the receiver/sender edges of the model graph (present, '
                'within spatial_radius); evidence the model masks is never moved in or out. LR prior pairs are NOT re-paired.',
    swap_compartments='Center nucleus/cytoplasm token tensors, presence and center_statistics N/C columns exchanged; '
                      'center LR is symmetric in compartments (N+C on jointly visible genes) and is unchanged; targets as-is.',
    whole_as_nucleus='Center patch row rewritten to nucleus=N+C, cytoplasm=0 before encode_v12_views (bins, log-counts, '
                     'statistics and presence recomputed); neighbors/LR unchanged; count targets restored from the real batch.')


def parse_ablations(values):
    """Canonical order, no duplicates, 'none' always first (anchor and delta reference)."""
    names = [name.strip() for value in values for name in str(value).split(',') if name.strip()]
    if 'all' in names:
        names = list(ABLATIONS)
    unknown = sorted(set(names) - set(ABLATIONS))
    if unknown:
        raise ValueError(f'Unknown ablations: {unknown}')
    return ['none'] + [name for name in ABLATIONS if name in names and name != 'none']


# ---- pure batch -> batch perturbations (CPU tensors; inputs are never mutated) ----

def identity(batch):
    return dict(batch)


def drop_neighbors(batch):
    output = dict(batch)
    output['neighbor_present'] = torch.zeros_like(batch['neighbor_present'])
    return output


def lr_off(batch):
    output = dict(batch)
    for key in LR_BIAS_KEYS:
        output[key] = torch.zeros_like(batch[key])
    return output


def random_cycle(count, rng):
    """source[i] is the old position copied to i; one random cycle, so no fixed point if count >= 2."""
    order = rng.permutation(count)
    source = np.empty(count, dtype=np.int64)
    source[order] = np.roll(order, 1)
    return source


def present_pairs(present):
    """[B,K,K] ordered receiver/sender pairs of distinct present neighbors."""
    present = present.bool()
    return present[:, :, None] & present[:, None, :] & ~torch.eye(present.shape[1], dtype=torch.bool)[None]


def lr_domain(batch, present=None, edges=None):
    """Entries whose LR evidence is read: [B,K] neighbors and [B,K,K] receiver-by-sender edges.

    Defaults are batch presence and every ordered present pair; the evaluation passes
    the model's own presence and graph (model_graph), a subset of those pairs.
    """
    present = (batch['neighbor_present'] if present is None else present).bool()
    pairs = present_pairs(present)
    edges = pairs if edges is None else edges.bool()
    if (present.shape != batch['neighbor_present'].shape or edges.shape != pairs.shape
            or bool((edges & ~pairs).any())):
        raise ValueError('LR edges must be ordered pairs of distinct present neighbors')
    return present, edges


def lr_shuffled(batch, seed=42, present=None, edges=None):
    """Random cycles of LR evidence inside the read domain; entries outside it never move.

    Seeded per physical center (cell_identity), hence independent of batch composition.
    """
    present, edges = (value.numpy() for value in lr_domain(batch, present, edges))
    identities = batch['cell_identity'].numpy()
    graph = batch['neighbor_lr_bias'].numpy().copy()
    center = batch['center_lr_bias'].numpy().copy()
    original_graph, original_center = graph.copy(), center.copy()
    for index in range(len(present)):
        slots = np.flatnonzero(present[index])
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(seed), int(identities[index, 0]), int(identities[index, 1]), LR_TAG]))
        center[index, slots] = original_center[index, slots[random_cycle(len(slots), rng)]]
        receiver, sender = np.nonzero(edges[index])
        source = random_cycle(len(receiver), rng)
        graph[index, receiver, sender] = original_graph[index, receiver[source], sender[source]]
    output = dict(batch)
    output['neighbor_lr_bias'], output['center_lr_bias'] = torch.from_numpy(graph), torch.from_numpy(center)
    return output


def lr_reassignment(original, shuffled, present=None, edges=None):
    """Additive per-batch account of the reassignment on the entries the model reads."""
    present, edges = lr_domain(original, present, edges)
    result = dict(centers=len(present), present_neighbors=int(present.sum()), graph_edges=int(edges.sum()),
                  present_pairs_outside_graph=int((present_pairs(present) & ~edges).sum()),
                  centers_without_movable_edges=int((edges.flatten(1).sum(1) < 2).sum()))
    for label, key, mask in (('edge', 'neighbor_lr_bias', edges), ('center', 'center_lr_bias', present)):
        before, after = original[key][mask].double(), shuffled[key][mask].double()
        result.update({f'{label}_nonzero': int((before > 0).sum()), f'{label}_changed': int((before != after).sum()),
                       f'{label}_evidence_before': float(before.sum()), f'{label}_evidence_after': float(after.sum())})
    return result


def sum_counts(items):
    return {key: sum(item[key] for item in items) for key in items[0]} if items else {}


def model_graph(model, batch):
    """The model's own neighbor presence and graph eligibility, evaluated on the CPU batch."""
    present = model._v12_presence(batch)[2]
    return present, model.spatial_encoder.graph(present, batch['relative_coords'])[0]


def swap_compartments(batch):
    expected = {f'{name}_{field}' for name in ('nucleus', 'cytoplasm') for field in COMPARTMENT_FIELDS}
    found = {key for key in batch if key.split('_', 1)[0] in ('nucleus', 'cytoplasm')}
    if found != expected:
        raise ValueError(f'Unexpected per-compartment tensors: {sorted(found ^ expected)}')
    if batch['center_statistics'].ndim != 2 or batch['center_statistics'].shape[1] != 4:
        raise ValueError('center_statistics must be [B,4] = N sum, C sum, N positive, C positive')
    output = dict(batch)
    for field in COMPARTMENT_FIELDS:
        output[f'nucleus_{field}'], output[f'cytoplasm_{field}'] = batch[f'cytoplasm_{field}'], batch[f'nucleus_{field}']
    output['center_statistics'] = batch['center_statistics'][:, [1, 0, 3, 2]]
    return output


def swap_targets(batch):
    """Exchanged N/C targets for the additional swap score; never a model input."""
    output = dict(batch)
    output['query_nucleus_counts'], output['query_cytoplasm_counts'] = (
        batch['query_cytoplasm_counts'], batch['query_nucleus_counts'])
    return output


TRANSFORMS = dict(none=identity, shuffle_neighbors=identity, whole_as_nucleus=identity,
                  drop_neighbors=drop_neighbors, lr_off=lr_off, swap_compartments=swap_compartments)


def apply_ablation(name, batch, seed=42, graph=None):
    """Tensor-level part of an ablation; data-level ones arrive already encoded.

    graph is the (present, edges) domain of lr_shuffled and is ignored otherwise.
    """
    if name == 'lr_shuffled':
        return lr_shuffled(batch, seed, *(graph or ()))
    return TRANSFORMS[name](batch)


def embedding_inputs(batch):
    """No query identity, validity or count target enters the embedding forward."""
    return {key: value for key, value in batch.items() if not key.startswith('query_')}


def restore_targets(perturbed, reference):
    """Perturbed model inputs, count targets of the same held-out queries from the real batch."""
    for key in ('cell_identity', 'query_gene_ids', 'query_valid_mask'):
        if not torch.equal(perturbed[key], reference[key]):
            raise ValueError(f'Ablated held-out batch changed {key}')
    output = dict(perturbed)
    for key in TARGET_KEYS:
        output[key] = reference[key]
    return output


# ---- data-level perturbations through the unchanged V12 encoder ----

def expressed_cells(nuclear, cyto):
    """Cells the V12 encoder would mark present as unmasked neighbors (positive mapped N+C)."""
    return ((np.asarray(nuclear) > 0) | (np.asarray(cyto) > 0)).any(axis=1)


def substitute_neighbors(neighbors, centers, eligible, *, seed=42, slice_seed=0):
    """Replace each present real neighbor by a same-slide cell; [B,K] rows, -1 padding kept.

    Real neighbors with no mapped expression stay (they are absent either way),
    so presence is unchanged. Seeded per center, independent of batch layout.
    """
    neighbors = np.asarray(neighbors, dtype=np.int64)
    centers = np.asarray(centers, dtype=np.int64)
    eligible = np.asarray(eligible, dtype=bool)
    if (neighbors.ndim != 2 or centers.shape != (len(neighbors),) or eligible.ndim != 1
            or (neighbors >= len(eligible)).any() or (centers < 0).any() or (centers >= len(eligible)).any()):
        raise ValueError('Expected [B,K] neighbor rows, [B] centers and a [N] eligibility mask')
    if (neighbors == centers[:, None]).any():
        raise ValueError('Center leaked into neighbors')
    result = neighbors.copy()
    relaxed = retained = 0
    for index, (center, row) in enumerate(zip(centers, neighbors)):
        real = row[row >= 0]
        slots = np.flatnonzero((row >= 0) & eligible[np.maximum(row, 0)])
        retained += len(real) - len(slots)
        if not len(slots):
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(slice_seed), int(center), SHUFFLE_TAG]))
        allowed = eligible.copy()
        allowed[real] = False
        allowed[center] = False
        pool = np.flatnonzero(allowed)
        if len(pool) < len(slots):
            # Too few other cells: real neighbors may be redrawn, the center never.
            allowed[row[slots]] = True
            pool = np.flatnonzero(allowed)
            relaxed += 1
        result[index, slots] = rng.choice(pool, size=len(slots), replace=False)
    replaced = (neighbors >= 0) & eligible[np.maximum(neighbors, 0)]
    return result, dict(centers=len(centers), substituted_slots=int(replaced.sum()), retained_absent_slots=int(retained),
                        relaxed_centers=int(relaxed), unchanged_slots=int((replaced & (result == neighbors)).sum()))


def whole_as_nucleus_counts(nuclear, cyto):
    """Patch rows [1+K,G]: only the center row changes; neighbor N+C is invariant."""
    nuclear, cyto = np.array(nuclear, dtype=np.int64), np.array(cyto, dtype=np.int64)
    nuclear[0] += cyto[0]
    cyto[0] = 0
    return nuclear, cyto


def prepare_batch(nuclear, cyto, genes, centers, content_rows, coord_rows, coords, *, center_counts=None, **options):
    """prepare_view_batch_v12 with neighbor content decoupled from neighbor position.

    Slot j holds the counts/identity of content_rows[:, j] at the coordinates of
    coord_rows[:, j]. Equal arrays and no center_counts reproduce the reference.
    """
    from spatial_pretrain.data.subcellular_v12 import collate_v12, encode_v12_views
    centers = np.asarray(centers, dtype=np.int64)
    content_rows, coord_rows = np.asarray(content_rows, dtype=np.int64), np.asarray(coord_rows, dtype=np.int64)
    if (content_rows.ndim != 2 or content_rows.shape != coord_rows.shape or len(content_rows) != len(centers)
            or ((content_rows >= 0) != (coord_rows >= 0)).any() or not len(centers)):
        raise ValueError('Content and coordinate neighbor rows must share [B,K] shape and padding')
    if (content_rows == centers[:, None]).any():
        raise ValueError('Center leaked into neighbors')
    items = []
    for center, content, place in zip(centers, content_rows, coord_rows):
        nodes = np.concatenate(([center], content[content >= 0]))
        positions = np.concatenate(([center], place[place >= 0]))
        patch_nuclear, patch_cyto = nuclear[nodes], cyto[nodes]
        if center_counts is not None:
            patch_nuclear, patch_cyto = center_counts(patch_nuclear, patch_cyto)
        items.append(encode_v12_views(patch_nuclear, patch_cyto, genes, nodes, coords[positions],
                                      neighbors=content_rows.shape[1], **options))
    return collate_v12(items)


def check_substitution(real, shuffled):
    """Only neighbor content may differ: presence, coordinates and the center are identical."""
    keys = ['neighbor_present', 'relative_coords', 'cell_identity', 'center_statistics']
    keys += [f'{name}_{field}' for name in ('nucleus', 'cytoplasm') for field in COMPARTMENT_FIELDS]
    keys += [key for key in real if key.startswith('query_')]
    for key in keys:
        if not torch.equal(real[key], shuffled[key]):
            raise ValueError(f'Neighbor substitution changed {key}')
    same = shuffled['neighbor_cell_identity'].eq(shuffled['cell_identity'][:, None]).all(-1)
    if bool((same & shuffled['neighbor_present']).any()):
        raise ValueError('Center selected as its own substitute neighbor')


# ---- metrics ----

def embedding_shift(reference, perturbed):
    """Per-cell movement of one readout relative to the unperturbed forward."""
    reference, perturbed = np.asarray(reference, dtype=np.float64), np.asarray(perturbed, dtype=np.float64)
    if reference.shape != perturbed.shape or reference.ndim != 2:
        raise ValueError('Embedding shift requires aligned [cells, dim] matrices')
    first, second = np.linalg.norm(reference, axis=1), np.linalg.norm(perturbed, axis=1)
    both, nonzero = (first > 1e-12) & (second > 1e-12), first > 1e-12
    cosine = (reference[both] * perturbed[both]).sum(axis=1) / (first[both] * second[both])
    relative = np.linalg.norm(perturbed - reference, axis=1)[nonzero] / first[nonzero]
    return dict(cells=len(reference), compared_cells=int(both.sum()),
                mean_cosine=float(cosine.mean()) if both.any() else None,
                min_cosine=float(cosine.min()) if both.any() else None,
                mean_relative_l2=float(relative.mean()) if nonzero.any() else None,
                max_abs_difference=float(np.abs(perturbed - reference).max()) if reference.size else 0.,
                zero_reference_cells=int((~nonzero).sum()), zero_perturbed_cells=int((second <= 1e-12).sum()))


def hybrid_features(reference, perturbed, train):
    """Unperturbed rows for train cells, perturbed rows elsewhere.

    The reference probes fit every statistic on train rows only, so probing this matrix
    applies the scaler/probe/PCA of the unperturbed model to perturbed validation cells.
    """
    reference, perturbed, train = np.asarray(reference), np.asarray(perturbed), np.asarray(train)
    if (reference.ndim != 2 or reference.shape != perturbed.shape or train.dtype != bool
            or train.shape != (len(reference),)):
        raise ValueError('Transfer probing requires aligned [cells, dim] matrices and a boolean [cells] train mask')
    return np.where(train[:, None], reference, perturbed)


def weighted_diagnostics(items):
    """Batch-size weighted means of per-batch model diagnostics (approximate for conditional means)."""
    result = {}
    for key in sorted({key for _, values in items for key in values}):
        pairs = [(cells, values[key]) for cells, values in items if key in values and np.isfinite(values[key])]
        total = sum(cells for cells, _ in pairs)
        result[key] = float(sum(cells * value for cells, value in pairs) / total) if total else None
    return result


def flatten_probe_row(common, result):
    """Same columns as summarize_subcellular_v12.py comparison/metrics.csv, plus 'ablation' and 'probe_fit'."""
    row = dict(**common, ablation=result['ablation'], probe_fit=result.get('probe_fit', REFIT),
               embedding=result['embedding'], task=result['task'],
               dimension=result['embedding_dimension'], train_cells=result.get('train_cells'), val_cells=result.get('val_cells'))
    if 'skipped' in result:
        row['skipped'] = result['skipped']
    elif result['task'] in ev12.FIXED_C:
        score = result['fixed_C_result']
        row.update(C=score['C'], val_macro_f1=score['val_macro_f1'], val_balanced_accuracy=score['val_balanced_accuracy'])
        row.update({f'donor_macro_f1_{donor}': value for donor, value in score['per_donor_macro_f1'].items()})
    else:
        row.update({f'val_{key}': value for key, value in result['val_result'].items()})
        row.update(ridge_alpha=result['ridge_alpha'], mean_val_label_coverage=result['mean_val_label_coverage'])
    return row


def flatten_count_row(common, result):
    return dict(**common, ablation=result['ablation'], embedding=result['mode'], task=COUNT_TASK,
                **{key: value for key, value in result.items() if key != 'ablation'})


def measured_values(flat_rows, fit=REFIT):
    """(ablation, embedding, task, split, metric) -> value; pooled donors, as-is targets only.

    Count rows carry no probe and belong to the default refit table only.
    """
    values = {}
    for row in flat_rows:
        if row.get('donor') not in (None, 'pooled') or row.get('target_orientation') not in (None, 'as_is'):
            continue
        if (row.get('probe_fit') or REFIT) != fit:
            continue
        for metric in PROBE_METRICS + COUNT_METRICS:
            if metric in row:
                values[(row['ablation'], row['embedding'], row['task'], row.get('split'), metric)] = row[metric]
    return values


def delta_vs_none(flat_rows):
    """Every perturbed value minus the unperturbed one; both probe fits share the refit 'none' reference."""
    reference = measured_values(flat_rows)
    rows = []
    for fit in (REFIT, TRANSFER):
        for (ablation, embedding, task, split, metric), value in measured_values(flat_rows, fit).items():
            if ablation in ('none', BASELINE_LABEL):
                continue
            base = reference.get(('none', embedding, task, split, metric))
            rows.append(dict(ablation=ablation, probe_fit=None if task == COUNT_TASK else fit, embedding=embedding,
                             task=task, split=split, metric=metric, none=base, value=value,
                             delta=None if value is None or base is None else float(value) - float(base),
                             higher_is_better=metric in HIGHER_IS_BETTER))
    return rows


def anchor(source, ablation, embedding, task, metric, expected, split=None, enforced=True):
    return dict(source=source, ablation=ablation, embedding=embedding, task=task, split=split, metric=metric,
                expected=float(expected), enforced=bool(enforced))


def builtin_anchors(variant, step):
    """Numbers measured by the official evaluator on the canonical cohort."""
    anchors = [anchor('official_baseline', BASELINE_LABEL, name, task, PRIMARY_METRIC[task], value)
               for (name, task), value in BASELINE_ANCHORS.items()]
    if variant == OFFICIAL_VARIANT:
        anchors += [anchor(f'official_step_{step}', 'none', name, task, PRIMARY_METRIC[task], value)
                    for (name, task), value in STEP_ANCHORS.get(step, {}).items()]
        anchors += [anchor(f'official_step_{step}', 'none', mode, COUNT_TASK, metric, value, split=split)
                    for (split, mode, metric), value in STEP_COUNT_ANCHORS.get(step, {}).items()]
    return anchors


def reference_anchors(report, step, checkpoint_sha256):
    """Every probe macro-F1 and pooled held-out NLL of the official metrics.json is enforced."""
    if report.get('status') != 'complete' or report.get('model_version') != 12:
        raise ValueError('Reference metrics are not a complete V12 evaluation')
    if int(report.get('step', -1)) != step or report.get('checkpoint_sha256') != checkpoint_sha256:
        raise ValueError('Reference metrics belong to a different checkpoint')
    if report.get('cohort_sha256') != ev12.CANONICAL_COHORT:
        raise ValueError('Reference metrics use a different cohort')
    anchors = []
    for row in report['results']:
        ablation = BASELINE_LABEL if row['embedding'] in ev12.BASELINES else 'none'
        if row['task'] in ev12.FIXED_C:
            anchors.append(anchor('reference_metrics', ablation, row['embedding'], row['task'], 'val_macro_f1',
                                  row['fixed_C_result']['val_macro_f1']))
        else:
            anchors += [anchor('reference_metrics', ablation, row['embedding'], row['task'], f'val_{metric}',
                               row['val_result'][metric], enforced=False) for metric in ('js_distance', 'mae')]
    for row in report['masked_count_prediction']['results']:
        if row['donor'] != 'pooled':
            continue
        anchors += [anchor('reference_metrics', 'none', row['mode'], COUNT_TASK, metric, row[metric],
                           split=row['split'], enforced=metric == 'nll')
                    for metric in COUNT_METRICS if row.get(metric) is not None]
    return anchors


def none_anchor_coverage(anchors):
    """Which numbers of the unperturbed model path are held to an enforced official anchor.

    Frozen-baseline anchors validate cohort, probes and data loading, never the V12 forward.
    """
    enforced = [item for item in anchors if item['enforced'] and item['ablation'] == 'none']
    covered = {(item['embedding'], item['task']) for item in enforced}
    wanted = [(name, task) for name in READOUTS for task in ev12.FIXED_C]
    wanted += [(mode, COUNT_TASK) for mode in ev12.COUNT_MODES]
    missing = [f'{name}/{task}' for name, task in wanted if (name, task) not in covered]
    status = 'full' if not missing else 'partial' if len(missing) < len(wanted) else 'unanchored'
    return dict(status=status, enforced_checks=len(enforced), unanchored=missing)


def unexpected_entries(names):
    """Entries of a contract-less output directory that are not launcher logs."""
    return sorted(name for name in names if name != 'logs' and not name.endswith(LOG_SUFFIXES))


def compare_anchors(anchors, measured, tolerance=ANCHOR_TOLERANCE):
    checks = []
    for item in anchors:
        value = measured.get((item['ablation'], item['embedding'], item['task'], item['split'], item['metric']))
        difference = None if value is None else abs(float(value) - item['expected'])
        checks.append(dict(**item, measured=value, abs_diff=difference,
                           within_tolerance=difference is not None and difference <= tolerance))
    failures = [item for item in checks if item['enforced'] and not item['within_tolerance']]
    return dict(passed=not failures, tolerance=tolerance, enforced_checks=sum(item['enforced'] for item in checks),
                failures=len(failures), checks=checks)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


# ---- GPU extraction: the loop of ev12.extract() with a perturbation hook ----

def load_slide(model, cfg, directory, obs, selected):
    """Same coordinates, neighbors, canonical gene order and slice seed as ev12.extract()."""
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    if not np.isfinite(coords).all() or not obs.full_cell_id.is_unique:
        raise ValueError('Invalid coordinates or duplicate cells')
    k = int(cfg.get('v12', {}).get('neighbors', 8))
    if k != model.neighbors:
        raise ValueError('V12 data/model neighbor count mismatch')
    neighbors = ev12.real_neighbors(coords, selected, k)
    panel = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([i for i, gene in enumerate(panel) if gene in model.vocab], dtype=np.int64)
    if not len(keep):
        raise ValueError('No genes map to scGPT vocabulary')
    keep = keep[np.argsort([model.vocab[gene] for gene in panel[keep]], kind='stable')]
    genes = np.array([model.vocab[gene] for gene in panel[keep]], dtype=np.int64)
    nuclear_all = np.load(directory / 'nuclear.npy', mmap_mode='r', allow_pickle=False)
    cyto_all = np.load(directory / 'cytoplasmic.npy', mmap_mode='r', allow_pickle=False)
    if nuclear_all.shape != cyto_all.shape or nuclear_all.shape != (len(obs), len(panel)):
        raise ValueError('Panel/counts/observations shape mismatch')
    return dict(coords=coords, neighbors=neighbors, genes=genes, nuclear=np.asarray(nuclear_all[:, keep]),
                cyto=np.asarray(cyto_all[:, keep]), matched_genes=len(keep), panel_genes=len(panel),
                slice_seed=int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little'))


def extract_conditions(model, cfg, slide, lr, names, batch_size, seed):
    from spatial_pretrain.data.subcellular_v12 import prepare_view_batch_v12
    started = time.monotonic()
    settings = cfg.get('v12', {})
    nuclear, cyto, genes, coords = (slide[key] for key in ('nuclear', 'cyto', 'genes', 'coords'))
    selected, neighbors = slide['selected'], slide['neighbors']
    options = dict(pad_token_id=model.vocab['<pad>'], cls_token_id=model.vocab['<cls>'],
        max_tokens=int(settings.get('max_tokens', 512)), query_genes=int(settings.get('query_genes', 64)),
        seed=42, epoch=0, slice_seed=slide['slice_seed'], slice_index=slide['index'],
        lr_pairs=lr['pairs'], lr_weights=lr['weights'],
        max_neighbor_distance=settings.get('max_neighbor_distance', 100.0))
    # Exactly the two reference passes: unmasked readouts, then fixed held-out queries.
    passes = (dict(mask=False), dict(mask=True, gene_mask_ratio=0., neighbor_dropout=0.))
    sources = {SOURCES.get(name, 'real') for name in names}
    substitutes = substitution = None
    if 'shuffled' in sources:
        substitutes, substitution = substitute_neighbors(neighbors, selected, expressed_cells(nuclear, cyto),
                                                         seed=seed, slice_seed=slide['slice_seed'])
    output = {name: dict(embeddings={readout: [] for readout in ev12.MODEL_READOUTS},
                         available={mode: [] for mode in ev12.COUNT_MODES}, count_rows=[], swapped_target_rows=[],
                         diagnostics=[], lr_reassignment=[]) for name in names}
    for start in range(0, len(selected), batch_size):
        end = min(start + batch_size, len(selected))
        rows, nearby = selected[start:end], neighbors[start:end]
        batches = {'real': tuple(prepare_view_batch_v12(nuclear, cyto, genes, rows, nearby, coords, **options, **extra)
                                 for extra in passes)}
        for key in ('nucleus', 'cytoplasm', 'total'):
            if f'query_{key}_counts' not in batches['real'][1]:
                raise ValueError('Missing raw count targets in V12 evaluation batch')
        if 'shuffled' in sources:
            batches['shuffled'] = tuple(prepare_batch(nuclear, cyto, genes, rows, substitutes[start:end], nearby, coords,
                                                      **options, **extra) for extra in passes)
            for real, shuffled in zip(batches['real'], batches['shuffled']):
                check_substitution(real, shuffled)
        if 'whole' in sources:
            batches['whole'] = tuple(prepare_batch(nuclear, cyto, genes, rows, nearby, nearby, coords,
                                                   center_counts=whole_as_nucleus_counts, **options, **extra)
                                     for extra in passes)
        for name in names:
            item = output[name]
            source, heldout = batches[SOURCES.get(name, 'real')]
            # lr_shuffled permutes only what the model itself reads: its presence and graph rules.
            graphs = [model_graph(model, batch) if name == 'lr_shuffled' else None for batch in (source, heldout)]
            view = apply_ablation(name, source, seed, graphs[0])
            if name == 'lr_shuffled':
                item['lr_reassignment'].append(lr_reassignment(source, view, *graphs[0]))
            heldout = restore_targets(apply_ablation(name, heldout, seed, graphs[1]), batches['real'][1])
            inputs = {key: value.cuda(non_blocking=True) for key, value in embedding_inputs(view).items()}
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                result = model(inputs, return_embeddings=True)
            for readout in ev12.MODEL_READOUTS:
                value = result['embeddings'][readout].float().cpu().numpy()
                if value.ndim != 2 or len(value) != end - start or not np.isfinite(value).all():
                    raise ValueError(f'Invalid V12 {readout} encoding under {name}')
                item['embeddings'][readout].append(value)
            for mode in ev12.COUNT_MODES:
                item['available'][mode].append(result['mode_available'][mode].cpu().numpy().astype(bool))
            item['diagnostics'].append((end - start, {key: float(result['diagnostics'][key]) for key in DIAGNOSTIC_KEYS
                                                      if key in result['diagnostics']}))
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(ev12.count_model_inputs(heldout))
            item['count_rows'].extend(ev12.score_masked_counts(prediction, heldout, rows))
            if name == 'swap_compartments':
                item['swapped_target_rows'].extend(ev12.score_masked_counts(prediction, swap_targets(heldout), rows))
        print(f'[ablate-v12] {slide["directory"].name} {end}/{len(selected)} {",".join(names)} '
              f'{time.monotonic()-started:.1f}s', flush=True)
    for name, item in output.items():
        item['embeddings'] = {readout: np.concatenate(blocks) for readout, blocks in item['embeddings'].items()}
        item['available'] = {mode: np.concatenate(blocks) for mode, blocks in item['available'].items()}
        if name == 'drop_neighbors' and item['available']['spatial'].any():
            raise ValueError('drop_neighbors left an available spatial view')
    return output, substitution


def count_frame(rows, obs, split):
    frame = pd.DataFrame(rows)
    centers = frame.center_row.to_numpy()
    frame['full_cell_id'] = obs.iloc[centers].full_cell_id.to_numpy()
    frame['patient'] = obs.iloc[centers].patient.to_numpy()
    frame['split'] = split
    if frame.full_cell_id.isna().any() or frame.duplicated(['full_cell_id', 'mode']).any():
        raise ValueError('Invalid held-out count cell identities')
    return frame


def run_probes(obs, features, target, coverage, classes, directory, label, constant=(), fit=REFIT):
    """Reference probes on row-L2-normalized readouts; rows gain only 'ablation' and 'probe_fit'."""
    rows = []
    for name, value in features.items():
        normalized = ev12.normalize_for_probe(value)
        for task in ev12.TASKS:
            try:
                row = (ev12.classification_probe(obs, normalized, name, task, directory) if task in ev12.FIXED_C
                       else ev12.composition_probe(obs, normalized, target, coverage, classes, name, directory))
            except Exception as error:
                # Only an information-free constant readout may skip; real failures stay loud.
                if name not in constant:
                    raise
                row = dict(embedding=name, task=task, embedding_dimension=value.shape[1],
                           skipped=f'constant readout: {type(error).__name__}: {error}')
            rows.append(dict(ablation=label, probe_fit=fit, **row))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--expected-step', required=True, type=int)
    parser.add_argument('--output-dir', '--output', required=True, type=Path)
    parser.add_argument('--ablations', nargs='+', default=['all'], help=f'Any of {", ".join(ABLATIONS)} or all')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/benchmarking_data/lungpf/prepared')
    parser.add_argument('--batch-size', default=8, type=int, help='Official V12 evaluations used 8')
    parser.add_argument('--reference-metrics', type=Path, help="Official evaluator metrics.json of this checkpoint")
    parser.add_argument('--allow-anchor-mismatch', action='store_true')
    parser.add_argument('--anchor-tolerance', default=ANCHOR_TOLERANCE, type=float)
    parser.add_argument('--baseline-cache-dir', type=Path, help='Default: recompute inside the output directory')
    parser.add_argument('--composition-radius-um', default=100., type=float)
    parser.add_argument('--ablation-seed', default=42, type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.composition_radius_um <= 0 or args.anchor_tolerance < 0 or args.ablation_seed < 0:
        parser.error('Batch size, radius must be positive; tolerance and seed nonnegative')
    names = parse_ablations(args.ablations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / 'metrics.json').exists():
        raise FileExistsError('Refusing to overwrite completed metrics')
    started = time.monotonic()
    torch.manual_seed(42); np.random.seed(42); torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    checkpoint_sha = ev12.file_sha256(args.checkpoint)
    reference = None
    if args.reference_metrics is not None:
        reference = reference_anchors(json.loads(args.reference_metrics.read_text()), args.expected_step, checkpoint_sha)
    sources = {name: ev12.file_sha256(ROOT / name) for name in (
        'scripts/evaluate_subcellular_v12_ablations.py', 'scripts/evaluate_subcellular_v12.py',
        'scripts/evaluate_subcellular_v2.py', 'scripts/probe_lungpf_embeddings.py',
        'src/spatial_pretrain/data/subcellular_v12.py', 'src/spatial_pretrain/models/subcellular_v12.py',
        'src/spatial_pretrain/models/subcellular_v11.py', 'src/spatial_pretrain/training/subcellular_v12.py',
        'src/spatial_pretrain/data/subcellular_v6.py', 'src/spatial_pretrain/data/subcellular_v6_left.py')}
    protocol_path = args.prepared / 'protocol.json'
    contract = json.loads(json.dumps(dict(schema='v12_inference_ablations_2', checkpoint_sha256=checkpoint_sha,
        step=args.expected_step, ablations=names, ablation_seed=args.ablation_seed, batch_size=args.batch_size,
        prepared=str(args.prepared.resolve()), protocol_sha256=ev12.file_sha256(protocol_path),
        composition_radius_um=args.composition_radius_um, canonical_cohort=ev12.CANONICAL_COHORT)))
    contract_path = args.output_dir / 'run_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Output belongs to a different checkpoint/ablation/protocol run')
    else:
        # Nothing in this directory is ever read back, but unverified leftovers must not sit
        # next to new results. Launcher logs (logs/, *.log, *.out, *.err) are not results.
        stale = unexpected_entries(path.name for path in args.output_dir.iterdir())
        if stale:
            raise ValueError(f'Nonempty output lacks a run contract: {stale[:5]}')
        ev12.atomic_json(contract_path, contract)

    def progress(**values):
        ev12.atomic_json(args.output_dir / 'progress.json', dict(step=args.expected_step, seconds=time.monotonic()-started, **values))

    model, baseline, cfg, step, provenance = ev12.load_model(args.checkpoint, args.expected_step)
    from spatial_pretrain.data.subcellular_v6 import load_lr_prior
    lr = load_lr_prior(cfg, vocab=model.vocab)
    lr_state = dict(enabled=bool(model.spatial_encoder.lr_enabled), prior=lr['metadata'],
                    strength_per_head=model.spatial_encoder.lr_strength.detach().float().cpu().tolist(),
                    max_strength=float(model.spatial_encoder.lr_max_strength))
    print(f'[ablate-v12] step={step} ablations={names} lr_strength={lr_state["strength_per_head"]}', flush=True)
    variant = cfg.get('experiment', 'v12')
    anchors = builtin_anchors(variant, step) + (reference or [])
    none_anchor = none_anchor_coverage(anchors)
    if none_anchor['status'] != 'full':
        print(f'[anchor-v12] WARNING: unperturbed V12 path is {none_anchor["status"].upper()} at step {step}: '
              f'{none_anchor["enforced_checks"]} enforced checks, {len(none_anchor["unanchored"])} numbers without one '
              f'({", ".join(none_anchor["unanchored"][:6])}{" ..." if len(none_anchor["unanchored"]) > 6 else ""}). '
              'Frozen-baseline anchors do not validate the model forward; pass --reference-metrics with the official '
              'metrics.json of this checkpoint.', flush=True)

    # Cohort: the selection code and seed of ev12.main(); no test donor is opened.
    protocol = json.loads(protocol_path.read_text())
    slides, classes = [], set()
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
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400, len(obs)), replace=False))
        slides.append(dict(index=index, directory=directory, obs=obs, split=split, selected=selected))
    classes = sorted(classes)
    if len(classes) < 2:
        raise ValueError('Insufficient training-donor neighbor classes')
    frames, targets, coverage_blocks = [], [], []
    for slide in slides:
        chosen = slide['obs'].iloc[slide['selected']].copy(); chosen['split'] = slide['split']
        frames.append(chosen)
        target, coverage = ev12.neighbor_composition(slide['obs'], slide['selected'], classes, args.composition_radius_um)
        targets.append(target); coverage_blocks.append(coverage)
    obs = pd.concat(frames, ignore_index=True)
    cohort = ev12.ids_hash(obs.full_cell_id)
    train_donors = sorted(obs.loc[obs['split'].eq('train'), 'patient'].unique().tolist())
    val_donors = sorted(obs.loc[obs['split'].eq('val'), 'patient'].unique().tolist())
    if (cohort != ev12.CANONICAL_COHORT or len(obs) != 5200 or not obs.full_cell_id.is_unique
            or len(train_donors) != 10 or len(val_donors) != 3):
        raise ValueError(f'Not canonical train10/val3 cohort: {cohort}')
    train = obs['split'].eq('train').to_numpy()
    target = np.concatenate(targets)
    coverage = {key: np.concatenate([block[key] for block in coverage_blocks]) for key in coverage_blocks[0]}
    ev12.write_frame(args.output_dir / 'cells.parquet', obs)
    ev12.atomic_array(args.output_dir / 'neighbor_composition_targets.npy', target)
    ev12.atomic_json(args.output_dir / 'neighbor_composition_classes.json', classes)
    print(f'[ablate-v12] cohort {cohort} cells={len(obs)}', flush=True)

    baseline_cache = args.baseline_cache_dir or args.output_dir / 'baseline_cache'
    baseline_blocks = {name: [] for name in ev12.BASELINES}
    gene_coverage, baseline_reused = [], []
    for slide in slides:
        slide.update(load_slide(model, cfg, slide['directory'], slide['obs'], slide['selected']))
        frozen, cache_info = ev12.frozen_baselines(baseline, cfg, slide['directory'], slide['selected'], slide['nuclear'],
            slide['cyto'], slide['genes'], slide['slice_seed'], args.batch_size, baseline_cache)
        for name, value in frozen.items():
            baseline_blocks[name].append(value)
        baseline_reused.append(bool(cache_info['reused']))
        gene_coverage.append(dict(sample=slide['directory'].name, selected_cells=len(slide['selected']),
                                  matched_genes=slide['matched_genes'], panel_genes=slide['panel_genes']))
        print(f'[ablate-v12] baselines {slide["directory"].name} reused={cache_info["reused"]}', flush=True)
    del baseline
    baseline_features = {name: np.concatenate(blocks) for name, blocks in baseline_blocks.items()}
    for name, value in baseline_features.items():
        if value.shape != (len(obs), 512):
            raise ValueError('Frozen baseline dimension contract violated')
        ev12.atomic_array(args.output_dir / f'{name}_embeddings.npy', value)

    common = dict(variant=variant, step=step, checkpoint_sha256=checkpoint_sha, config_sha256=provenance['config_sha256'],
                  cohort_sha256=cohort, metrics_path=str((args.output_dir / 'metrics.json').resolve()))
    results, transfer_results, count_results, summaries, substitution = [], [], [], {}, []
    reference_features = {}

    def embed(group):
        store = {name: dict(embeddings={readout: [] for readout in ev12.MODEL_READOUTS},
                            available={mode: [] for mode in ev12.COUNT_MODES}, frames=[], swapped=[], diagnostics=[],
                            lr_reassignment=[])
                 for name in group}
        for position, slide in enumerate(slides):
            values, info = extract_conditions(model, cfg, slide, lr, group, args.batch_size, args.ablation_seed)
            if info is not None:
                substitution.append(dict(sample=slide['directory'].name, **info))
            for name, item in values.items():
                entry = store[name]
                for readout, value in item['embeddings'].items():
                    entry['embeddings'][readout].append(value)
                for mode, value in item['available'].items():
                    entry['available'][mode].append(value)
                entry['frames'].append(count_frame(item['count_rows'], slide['obs'], slide['split']))
                if item['swapped_target_rows']:
                    entry['swapped'].append(count_frame(item['swapped_target_rows'], slide['obs'], slide['split']))
                entry['diagnostics'].extend(item['diagnostics'])
                entry['lr_reassignment'].extend(item['lr_reassignment'])
            progress(stage='embedding', ablations=group, completed_slices=position + 1, total_slices=len(slides))
        return store

    def finish(name, entry):
        directory = args.output_dir / name
        directory.mkdir(exist_ok=True)
        features = {readout: np.concatenate(blocks) for readout, blocks in entry['embeddings'].items()}
        if any(features[readout].shape != (len(obs), 1536 if readout == 'concat' else 512) for readout in features):
            raise ValueError(f'V12 readout dimension contract violated under {name}')
        features['concat_pca512'], pca = ev12.concat_pca(features['concat'], train)
        for readout, value in features.items():
            ev12.atomic_array(directory / f'{readout}_embeddings.npy', value)
        frame = pd.concat(entry['frames'], ignore_index=True)
        ev12.write_frame(directory / 'masked_count_per_cell.parquet', frame)
        counts = [dict(ablation=name, target_orientation='as_is', **row) for row in ev12.aggregate_masked_counts(frame)]
        if entry['swapped']:
            swapped = pd.concat(entry['swapped'], ignore_index=True)
            ev12.write_frame(directory / 'masked_count_per_cell_swapped_targets.parquet', swapped)
            counts += [dict(ablation=name, target_orientation='swapped', **row)
                       for row in ev12.aggregate_masked_counts(swapped)]
        count_results.extend(counts)
        constant = [readout for readout, value in features.items() if float(np.ptp(value, axis=0).max()) == 0.]
        results.extend(run_probes(obs, features, target, coverage, classes, directory, name, constant))
        if name == 'none':
            reference_features.update(features)
        summaries[name] = dict(description=DESCRIPTIONS[name], pca=pca, constant_readouts=constant,
            available_counts={mode: int(np.concatenate(blocks).sum()) for mode, blocks in entry['available'].items()},
            model_diagnostics=weighted_diagnostics(entry['diagnostics']),
            collapse=ev12.representation_diagnostics(obs, features)['collapse'],
            embedding_shift_vs_none={readout: embedding_shift(reference_features[readout], value)
                                     for readout, value in features.items()})
        structural = {}
        if name == 'swap_compartments':
            # One shared encoder/adapter and no compartment embedding: expected ~0 up to BF16 batch effects.
            structural = dict(nucleus_vs_none_cytoplasm=embedding_shift(reference_features['cytoplasm'], features['nucleus']),
                              cytoplasm_vs_none_nucleus=embedding_shift(reference_features['nucleus'], features['cytoplasm']))
        if name == 'drop_neighbors':
            structural = dict(fused_vs_local=embedding_shift(features['local'], features['fused']),
                              spatial_max_abs=float(np.abs(features['spatial']).max()))
        if name == 'whole_as_nucleus':
            structural = dict(nucleus_vs_frozen_scgpt_whole=embedding_shift(baseline_features['scgpt_whole'], features['nucleus']),
                              note='Equal up to BF16 only at step 0; later it measures encoder drift on whole-cell input')
        summaries[name]['structural_checks'] = structural
        if name == 'lr_shuffled':
            # Evidence sums over the model-read entries are equal before/after by construction.
            summaries[name]['lr_reassignment'] = sum_counts(entry['lr_reassignment'])
        if name != 'none':
            # Scaler, probe and concat PCA of the unperturbed model applied to perturbed validation cells.
            hybrid = {readout: hybrid_features(reference_features[readout], features[readout], train)
                      for readout in ev12.MODEL_READOUTS}
            hybrid['concat_pca512'], summaries[name]['transfer_pca'] = ev12.concat_pca(hybrid['concat'], train)
            transfer_dir = directory / TRANSFER
            transfer_dir.mkdir(exist_ok=True)
            transfer_results.extend(run_probes(obs, hybrid, target, coverage, classes, transfer_dir, name, fit=TRANSFER))
        progress(stage='probing', completed_ablations=list(summaries), total_ablations=len(names))

    def flat_rows():
        return ([flatten_probe_row(common, row) for row in results]
                + [flatten_count_row(common, row) for row in count_results])

    # Phase 1: frozen baselines and the unperturbed path must reproduce the official numbers
    # before any GPU time is spent on perturbed conditions.
    baseline_dir = args.output_dir / BASELINE_LABEL
    baseline_dir.mkdir(exist_ok=True)
    results.extend(run_probes(obs, baseline_features, target, coverage, classes, baseline_dir, BASELINE_LABEL))
    finish('none', embed(['none'])['none'])
    anchor_check = compare_anchors(anchors, measured_values(flat_rows()), args.anchor_tolerance)
    anchor_check.update(step=step, variant=variant, checkpoint_sha256=checkpoint_sha, cohort_sha256=cohort,
        none_path_anchor=none_anchor,
        reference_metrics=None if args.reference_metrics is None else str(args.reference_metrics.resolve()),
        allow_anchor_mismatch=bool(args.allow_anchor_mismatch), frozen_baselines_recomputed=not any(baseline_reused),
        step_specific_builtin_anchors=variant == OFFICIAL_VARIANT and (step in STEP_ANCHORS or step in STEP_COUNT_ANCHORS))
    ev12.atomic_json(args.output_dir / 'anchor_check.json', anchor_check)
    for item in anchor_check['checks']:
        if item['enforced']:
            print(f'[anchor-v12] {item["source"]} {item["ablation"]}/{item["embedding"]}/{item["task"]}/{item["metric"]} '
                  f'expected={item["expected"]:.6f} measured={item["measured"]} ok={item["within_tolerance"]}', flush=True)
    if not anchor_check['passed'] and not args.allow_anchor_mismatch:
        progress(stage='anchor_failed', diagnostic='anchor_check.json')
        raise SystemExit(f'ANCHOR MISMATCH: {anchor_check["failures"]} enforced checks exceed {args.anchor_tolerance}; '
                         f'see {args.output_dir / "anchor_check.json"} (use --allow-anchor-mismatch to continue)')

    # Phase 2: perturbed conditions, sharing every encoded batch across ablations.
    rest = names[1:]
    if rest:
        store = embed(rest)
        del model
        torch.cuda.empty_cache()
        for name in rest:
            finish(name, store.pop(name))

    flat = flat_rows()
    transfer_flat = [flatten_probe_row(common, row) for row in transfer_results]
    deltas = delta_vs_none(flat + transfer_flat)
    expected = {(label, name, task) for label in names for name in READOUTS for task in ev12.TASKS}
    expected_transfer = {key for key in expected if key[0] != 'none'}
    expected |= {(BASELINE_LABEL, name, task) for name in ev12.BASELINES for task in ev12.TASKS}
    for rows, wanted in ((results, expected), (transfer_results, expected_transfer)):
        if len(rows) != len(wanted) or {(row['ablation'], row['embedding'], row['task']) for row in rows} != wanted:
            raise ValueError('Every ablation needs every readout and task exactly once per probe fit')
    # metrics.csv keeps one row per (ablation, embedding, task); none-fitted probes live in their own file.
    write_csv(args.output_dir / 'metrics.csv', flat)
    if transfer_flat:
        write_csv(args.output_dir / 'transfer_metrics.csv', transfer_flat)
    write_csv(args.output_dir / 'masked_count_prediction.csv', [dict(**common, **row) for row in count_results])
    write_csv(args.output_dir / 'delta_vs_none.csv', deltas)
    composition_qa = {split: {key: float(value[obs['split'].eq(split).to_numpy()].mean()) for key, value in coverage.items()}
                      for split in ('train', 'val')}
    report = dict(status='complete', experiment='v12_inference_ablations', model_version=12, architecture=ev12.ARCHITECTURE,
        variant=variant, step=step, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=checkpoint_sha, **provenance,
        config=cfg, evaluation_source_sha256=sources, run_contract=contract,
        cells=len(obs), cohort_sha256=cohort, canonical_cohort=True, no_test=True, test_donors_used=False,
        train_donors=train_donors, val_donors=val_donors, fixed_C=ev12.FIXED_C, seed=42, ablation_seed=args.ablation_seed,
        protocol_sha256=contract['protocol_sha256'], probe_row_normalization=ev12.PROBE_ROW_NORMALIZATION,
        probe_protocol='Functions of scripts/evaluate_subcellular_v12.py: row L2, train-only StandardScaler, balanced '
                       'logistic fixed C, Ridge alpha=1 then simplex for neighborhood composition; concat PCA512 fit '
                       'on train cells; no hyperparameter search',
        probe_fits={REFIT: 'results: scaler, probe and concat PCA refit on the train embeddings of the same condition; '
                           'measures what stays linearly readable and is a lower bound on input dependence',
                    TRANSFER: 'transfer_results: the same functions on a matrix holding unperturbed embeddings for train '
                              'cells and perturbed embeddings for validation cells, i.e. the scaler, probe and PCA of the '
                              'unperturbed model applied to perturbed validation cells; measures reliance on the input'},
        composition_protocol=dict(radius_um=args.composition_radius_um, center_excluded=True, labels='external existing final_CT',
            class_vocabulary='all nonmissing labels in train donors only', classes=classes, coverage=composition_qa,
            interpretation='Targets always come from the REAL physical neighborhood, also when model inputs are ablated'),
        baseline_protocol='Separately loaded original ordinary scGPT via ev12.frozen_baselines; never the fine-tuned backbone',
        frozen_baseline_cache=dict(path=str(Path(baseline_cache).resolve()), reused=baseline_reused),
        inference=dict(dtype='bfloat16', batch_size=args.batch_size, embedding_pass='mask=False',
            count_pass='mask=True, gene_mask_ratio=0, neighbor_dropout=0, seed=42, epoch=0',
            target_counts_passed_to_forward=False, perturbation_scope='batch tensors or encoder input rows only'),
        ablation_protocol={name: DESCRIPTIONS[name] for name in names},
        ablation_notes=dict(
            shared_compartment_encoder='Nucleus and cytoplasm use one backbone and one count adapter without a compartment '
                'embedding. Swapped nucleus/cytoplasm readouts are the unswapped cytoplasm/nucleus readouts by '
                'construction; identity can only act through the local gate, fusion, concat order and decoder state.',
            lr_scope='neighbor_lr_bias changes spatial, fused and concat; center_lr_bias only the fused environment twin.',
            degenerate_readouts='An all-zero readout (spatial under drop_neighbors, cytoplasm under whole_as_nucleus) '
                'is probed as-is and gives the constant-predictor floor; listed in constant_readouts.',
            swapped_targets='target_orientation=swapped rows score the same predictions against exchanged N/C targets.',
            reading_order='Whether the trained model uses an input is read from masked-count rows, '
                'embedding_shift_vs_none and probe_fit=none_train deltas; probe_fit=refit deltas only bound it from '
                'below because a refit probe absorbs systematic shifts.',
            shuffle_scope='shuffle_neighbors substitutes come from the same slide, which is one donor: slide-level '
                'composition stays in the spatial input, so region and neighbor_composition do not fall to chance. '
                'Compare with drop_neighbors before concluding anything about neighbor content.',
            lr_shuffle_scope='lr_shuffled permutes evidence only among the neighbors and graph edges the model reads '
                '(ablations.lr_shuffled.lr_reassignment); centers with fewer than two such entries are unchanged.',
            none_anchor='anchor_check.none_path_anchor states which unperturbed numbers were held to an official '
                'anchor; frozen-baseline anchors alone do not validate the V12 forward.'),
        lr=lr_state, neighbor_substitution=substitution, gene_coverage=gene_coverage,
        anchor_check=anchor_check, none_path_anchor=none_anchor['status'], ablations=summaries, results=results,
        transfer_results=transfer_results,
        masked_count_prediction=dict(seed=42, epoch=0, query_genes=int(cfg.get('v12', {}).get('query_genes', 64)),
            both_center_compartments_query_hidden=True, gene_mask_ratio=0., neighbor_dropout=0.,
            target_counts_passed_to_forward=False, decoder='NB(total)*beta-binomial(nuclear|total)', results=count_results),
        delta_vs_none=deltas, seconds=time.monotonic()-started)
    ev12.atomic_json(args.output_dir / 'metrics.json', report)
    progress(stage='complete', metrics='metrics.json')
    print(f'[done-v12-ablations] step={step} {len(names)} conditions x {len(READOUTS)} readouts x {len(ev12.TASKS)} tasks; '
          f'anchors passed={anchor_check["passed"]} unperturbed_path_anchor={none_anchor["status"]}', flush=True)
    if none_anchor['status'] != 'full':
        print(f'[anchor-v12] WARNING: {len(none_anchor["unanchored"])} unperturbed numbers had no enforced official anchor; '
              'see anchor_check.json none_path_anchor', flush=True)


if __name__ == '__main__':
    main()
