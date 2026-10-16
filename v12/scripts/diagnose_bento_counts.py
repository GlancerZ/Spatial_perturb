#!/usr/bin/env python3
"""Compute-node, read-only diagnosis of published Bento counts and geometry.

No source files are changed. Source nucleus labels are compared as literal
categories only; neither they nor spliced/unspliced layers assign compartments.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from shapely import wkt

from prepare_bento_public_corpus import parse_shapes


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def matrix_stats(matrix):
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    finite = np.isfinite(values)
    safe = values[finite]
    return {"shape": list(matrix.shape), "dtype": str(matrix.dtype),
            "nnz": int(matrix.nnz if sp.issparse(matrix) else np.count_nonzero(matrix)),
            "total": float(matrix.sum()) if finite.all() else None,
            "nonfinite_entries": int((~finite).sum()),
            "negative_entries": int((safe < 0).sum()),
            "noninteger_entries": int((safe != np.floor(safe)).sum()),
            "nonzero_min": float(safe.min()) if len(safe) else None,
            "maximum": float(safe.max()) if len(safe) else None}


def compare(observed, reference, cells, genes):
    diff = observed.astype(np.float64) - reference.astype(np.float64)
    diff.eliminate_zeros()
    coo = diff.tocoo()
    order = np.argsort(-np.abs(coo.data))[:12]
    rows = np.asarray(diff.sum(axis=1)).ravel()
    cols = np.asarray(diff.sum(axis=0)).ravel()
    examples = [{"cell": str(cells[coo.row[i]]), "gene": str(genes[coo.col[i]]),
                 "points": float(observed[coo.row[i], coo.col[i]]),
                 "reference": float(reference[coo.row[i], coo.col[i]]),
                 "difference": float(coo.data[i])} for i in order if np.isfinite(coo.data[i])]
    return {"mismatched_entries": int(diff.nnz),
            "absolute_difference": float(np.abs(diff.data).sum()) if np.isfinite(diff.data).all() else None,
            "points_exceed_reference_entries": int((diff.data > 0).sum()),
            "points_below_reference_entries": int((diff.data < 0).sum()),
            "rows_with_net_difference": int(np.count_nonzero(rows)),
            "genes_with_net_difference": int(np.count_nonzero(cols)),
            "largest_entry_differences": examples}


def fingerprint_permutation(observed, reference, labels, axis):
    """Exact content match under just one axis permutation; no nearest matches."""
    left, right = observed.toarray(), reference.toarray()
    if axis == 1:
        left, right = left.T, right.T
    def fingerprint(row):
        return hashlib.sha256(np.asarray(row, dtype=np.float64).tobytes()).hexdigest()
    lookup = defaultdict(list)
    for i, row in enumerate(left):
        lookup[fingerprint(row)].append(i)
    matches = [lookup.get(fingerprint(row), []) for row in right]
    unambiguous = [(i, found[0]) for i, found in enumerate(matches) if len(found) == 1]
    bijective = len(unambiguous) == len(labels) and len({j for _, j in unambiguous}) == len(labels)
    return {"vectors_with_exact_match": sum(bool(found) for found in matches),
            "unambiguous_vectors": len(unambiguous), "complete_unique_permutation": bijective,
            "moved_vectors": sum(i != j for i, j in unambiguous),
            "examples": [{"reference_axis_label": str(labels[i]), "matches_points_label": str(labels[j])}
                         for i, j in unambiguous if i != j][:12]}


def geometry_diagnosis(data):
    reasons = Counter()
    by_batch = defaultdict(Counter)
    outside_ratios, examples = [], []
    for cell_id, row in data.obs.iterrows():
        _, _, reason = parse_shapes(row.get("cell_shape"), row.get("nucleus_shape"))
        reasons[reason] += 1
        by_batch[str(row.get("batch"))][reason] += 1
        if reason == "nucleus_not_contained_in_cell":
            cell, nucleus = wkt.loads(row["cell_shape"]), wkt.loads(row["nucleus_shape"])
            ratio = float(nucleus.difference(cell).area / nucleus.area)
            outside_ratios.append(ratio)
            if len(examples) < 12:
                examples.append({"cell": str(cell_id), "batch": str(row.get("batch")),
                                 "nucleus_area_fraction_outside_cell": ratio})
    return {"reasons": dict(reasons), "by_batch": {k: dict(v) for k, v in by_batch.items()},
            "noncontained_nucleus_outside_fraction_quantiles":
                dict(zip(["min", "p25", "median", "p75", "max"],
                         map(float, np.quantile(outside_ratios, [0, .25, .5, .75, 1])))) if outside_ratios else {},
            "noncontained_examples": examples,
            "policy": "Diagnostic only; no clipping, repair, or change to adapter eligibility"}


def candidate_orders(index, series):
    candidates = {"published_axis": index.tolist(), "lexicographic": sorted(index)}
    if isinstance(series.dtype, pd.CategoricalDtype):
        candidates["point_category_order"] = list(map(str, series.cat.categories))
    candidates["point_first_appearance"] = list(map(str, pd.unique(series)))
    return {key: index.get_indexer(order) for key, order in candidates.items()
            if len(order) == len(index) and set(order) == set(index)}


def diagnose(path, chunk_rows):
    started = time.monotonic()
    path = path.resolve()
    data = ad.read_h5ad(path)
    points = data.uns["points"]
    cells, genes = pd.Index(data.obs_names.astype(str)), pd.Index(data.var_names.astype(str))
    if not cells.is_unique or not genes.is_unique:
        raise ValueError("Nonunique published cell or gene axes")
    raw_x = sp.csr_matrix(data.X)
    references = {"X": raw_x, **{f"layer:{name}": sp.csr_matrix(value) for name, value in data.layers.items()}}
    if "spliced" in data.layers and "unspliced" in data.layers:
        references["layers:spliced_plus_unspliced"] = (sp.csr_matrix(data.layers["spliced"]) + sp.csr_matrix(data.layers["unspliced"]))
    observed = sp.csr_matrix(data.shape, dtype=np.int64)
    label_missing = sp.csr_matrix(data.shape, dtype=np.int64)
    label_present = sp.csr_matrix(data.shape, dtype=np.int64)
    unknown_cell, unknown_gene, batch_mismatch, nucleus_labels = Counter(), Counter(), 0, Counter()
    known_points = 0
    obs_batches = data.obs["batch"].astype(str).to_numpy()
    for start in range(0, len(points), chunk_rows):
        frame = points.iloc[start:start + chunk_rows]
        ci, gi = cells.get_indexer(frame["cell"].astype(str)), genes.get_indexer(frame["gene"].astype(str))
        unknown_cell.update(frame.loc[ci < 0, "cell"].astype(str))
        unknown_gene.update(frame.loc[gi < 0, "gene"].astype(str))
        valid = (ci >= 0) & (gi >= 0)
        known_points += int(valid.sum())
        if "batch" in frame:
            batch_mismatch += int(np.count_nonzero(frame["batch"].astype(str).to_numpy()[valid] != obs_batches[ci[valid]]))
        def aggregate(mask):
            return sp.coo_matrix((np.ones(int(mask.sum()), dtype=np.int64), (ci[mask], gi[mask])), shape=data.shape).tocsr()
        observed += aggregate(valid)
        if "nucleus" in frame:
            labels = frame["nucleus"].astype(str).to_numpy()
            nucleus_labels.update(labels)
            missing = np.isin(labels, ["-1", "-1.0", "None", "nan", "<NA>"])
            label_missing += aggregate(valid & missing)
            label_present += aggregate(valid & ~missing)
        print(f"[bento-count-diagnosis] {path.name} RNA={min(start+chunk_rows, len(points))}/{len(points)}", flush=True)
    comparisons = {name: compare(observed, value, cells, genes) for name, value in references.items()}
    row_points, col_points = np.asarray(observed.sum(axis=1)).ravel(), np.asarray(observed.sum(axis=0)).ravel()
    row_x, col_x = np.asarray(raw_x.sum(axis=1)).ravel(), np.asarray(raw_x.sum(axis=0)).ravel()
    permutations = {}
    if observed.sum() == raw_x.sum():
        permutations["rows"] = fingerprint_permutation(observed, raw_x, cells, axis=0)
        permutations["genes"] = fingerprint_permutation(observed, raw_x, genes, axis=1)
    candidates = []
    for rn, ri in candidate_orders(cells, points["cell"]).items():
        for cn, ci in candidate_orders(genes, points["gene"]).items():
            diff = observed[ri][:, ci].astype(float) - raw_x.astype(float)
            diff.eliminate_zeros()
            candidates.append({"row_order": rn, "gene_order": cn, "mismatched_entries": int(diff.nnz),
                               "absolute_difference": float(np.abs(diff.data).sum()) if np.isfinite(diff.data).all() else None})
    batch_totals = []
    for batch in pd.unique(obs_batches):
        mask = obs_batches == batch
        batch_totals.append({"batch": batch, "cells": int(mask.sum()), "point_total": int(row_points[mask].sum()),
                             "X_total": float(row_x[mask].sum()),
                             "net_difference": float((row_points[mask] - row_x[mask]).sum())})
    return {"source": str(path), "shape": list(data.shape), "points_rows": len(points), "known_identity_points": known_points,
            "unknown_cell_ids": dict(unknown_cell), "unknown_gene_ids": dict(unknown_gene),
            "batch_mismatched_points": batch_mismatch, "obs_axis_head": cells[:20].tolist(), "var_axis_head": genes[:20].tolist(),
            "matrix_stats": {name: matrix_stats(value) for name, value in references.items()},
            "points_matrix_stats": matrix_stats(observed), "all_points_vs_references": comparisons,
            "literal_nucleus_label_counts": dict(nucleus_labels),
            "literal_label_minus1_vs_references": {name: compare(label_missing, value, cells, genes) for name, value in references.items()},
            "literal_label_not_minus1_vs_references": {name: compare(label_present, value, cells, genes) for name, value in references.items()},
            "literal_label_warning": "Diagnostic category comparisons only; no splicing or compartment interpretation",
            "sorted_row_totals_equal": bool(np.array_equal(np.sort(row_points), np.sort(row_x))),
            "sorted_gene_totals_equal": bool(np.array_equal(np.sort(col_points), np.sort(col_x))),
            "exact_axis_permutations": permutations, "candidate_axis_orders": candidates,
            "batch_totals": batch_totals, "geometry": geometry_diagnosis(data),
            "elapsed_seconds": round(time.monotonic()-started, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=250000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not 1 <= args.chunk_rows <= 1000000:
        raise ValueError("chunk-rows must be in [1,1000000]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    for path in args.input:
        result[str(path)] = diagnose(path, args.chunk_rows)
        write_json(args.output, result)
        report = result[str(path)]
        print(json.dumps({"file": str(path), "points": report["points_rows"],
                          "X": report["matrix_stats"]["X"],
                          "comparison": report["all_points_vs_references"]["X"],
                          "geometry": report["geometry"]["reasons"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
