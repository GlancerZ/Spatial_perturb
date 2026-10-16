"""Compare fixed raw-data lung marker gates before/after total-count correction.

This is a descriptive QC, not a molecule-level contamination estimate or an
independent validation of ResolVI (the genes may participate in model fitting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def column(matrix: sp.csr_matrix, names, gene: str) -> np.ndarray:
    if gene not in names:
        raise ValueError(f"Required gene is absent from the panel: {gene}")
    return np.asarray(matrix[:, names.get_loc(gene)].toarray()).ravel()


def group_summary(values: np.ndarray, totals: np.ndarray, selected: np.ndarray) -> dict:
    denominator = float(totals[selected].sum())
    return {
        "cells": int(selected.sum()),
        "marker_sum": float(values[selected].sum()),
        "per_1000_original_rna": float(values[selected].sum() * 1000 / denominator)
        if denominator else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = ad.read_h5ad(args.corrected)
    if "raw_total_counts" not in data.layers or "spatial" not in data.obsm:
        raise ValueError("Expected corrected H5AD with raw total counts and global coordinates")
    corrected = sp.csr_matrix(data.X)
    raw = sp.csr_matrix(data.layers["raw_total_counts"])
    if raw.shape != corrected.shape:
        raise ValueError("Raw and corrected count shapes differ")
    names = data.var_names
    totals = np.asarray(raw.sum(axis=1)).ravel()
    epcam = column(raw, names, "EPCAM")
    cd3e = column(raw, names, "CD3E")
    support_genes = [name for name in ("CD3D", "LCK", "CD8A", "CD8B", "TRAC")
                     if name in names]
    if len(support_genes) < 2:
        raise ValueError("Panel lacks two distinct T-cell support genes")
    support = np.column_stack([column(raw, names, name) > 0 for name in
                               support_genes]).sum(axis=1)
    t_cells = (cd3e >= 2) & (support >= 2) & (epcam <= 1) & (totals >= 20)
    sources = (epcam >= 10) & (totals >= 50)
    if not np.any(t_cells) or not np.any(sources):
        raise ValueError("No fixed-gate T cells or epithelial sources in this slice")
    coordinates = np.asarray(data.obsm["spatial"], dtype=np.float64)[:, :2]
    distance, _ = cKDTree(coordinates[sources]).query(coordinates[t_cells], k=1)
    t_indices = np.flatnonzero(t_cells)
    near = np.zeros(data.n_obs, dtype=bool)
    far = np.zeros(data.n_obs, dtype=bool)
    near[t_indices[distance <= 15.0]] = True
    far[t_indices[distance > 50.0]] = True
    result = {
        "sample": data.uns.get("raw_resolvi_pipeline", {}).get("sample_id", args.corrected.parent.name),
        "definition": "gates and distances use raw counts; rate denominator uses raw total RNA",
        "limitations": "descriptive only; model fit included these genes; near/far not depth-area standardized",
        "t_candidate_cells": int(t_cells.sum()),
        "epcam_source_cells": int(sources.sum()),
        "near_t_cells": int(near.sum()),
        "far_t_cells": int(far.sum()),
        "markers": {},
    }
    for gene in ("EPCAM", "MUC1", "CD3E", *[name for name in ("TRAC", "LCK") if name in names]):
        old = column(raw, names, gene)
        new = column(corrected, names, gene)
        result["markers"][gene] = {
            "raw_near": group_summary(old, totals, near),
            "corrected_near": group_summary(new, totals, near),
            "raw_far": group_summary(old, totals, far),
            "corrected_far": group_summary(new, totals, far),
            "raw_source": group_summary(old, totals, sources),
            "corrected_source": group_summary(new, totals, sources),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
