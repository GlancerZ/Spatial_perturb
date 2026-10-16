"""Exploratory Xenium proximity audit using existing compartment H5AD files.

This is not RCTD or SPLIT: it measures expression patterns associated with
nearby EPCAM-high cells, without changing molecule or cell assignments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree


DISTANCE_BINS_UM = [0.0, 15.0, 30.0, 50.0, 100.0]
READOUT_CANDIDATES = ["EPCAM", "CEACAM6", "MUC1", "KRT20", "SOX2"]
T_SUPPORT = ["LCK", "CD8A", "CD8B", "CD3D", "TRAC"]


def gene_counts(matrix, genes, gene: str) -> np.ndarray:
    if gene not in genes:
        return np.zeros(matrix.shape[0], dtype=np.int64)
    values = matrix[:, genes.get_loc(gene)]
    return np.asarray(values.toarray() if hasattr(values, "toarray") else values).ravel().astype(np.int64)


def summarize(mask: np.ndarray, count: np.ndarray, nuclear: np.ndarray,
              cytoplasmic: np.ndarray, depth: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"cells": 0}
    return {
        "cells": n,
        "positive_cells": int(np.count_nonzero(count[mask])),
        "positive_fraction": float(np.mean(count[mask] > 0)),
        "mean_count_per_cell": float(np.mean(count[mask])),
        "mean_nuclear_count_per_cell": float(np.mean(nuclear[mask])),
        "mean_cytoplasmic_count_per_cell": float(np.mean(cytoplasmic[mask])),
        "total_count_per_1000_rna": float(1000.0 * count[mask].sum() / max(1, depth[mask].sum())),
        "nuclear_count_per_1000_rna": float(1000.0 * nuclear[mask].sum() / max(1, depth[mask].sum())),
        "cytoplasmic_count_per_1000_rna": float(1000.0 * cytoplasmic[mask].sum() / max(1, depth[mask].sum())),
        "median_total_rna": float(np.median(depth[mask])),
    }


def stratified_positive_rate(near: np.ndarray, far: np.ndarray,
                             positive: np.ndarray, depth: np.ndarray,
                             area: np.ndarray) -> dict:
    """Standardize positivity to strata represented in both distance groups."""
    depth_edges = np.quantile(np.log1p(depth), [0.25, 0.5, 0.75])
    area_edges = np.quantile(np.log1p(area), [0.25, 0.5, 0.75])
    depth_bin = np.searchsorted(depth_edges, np.log1p(depth), side="right")
    area_bin = np.searchsorted(area_edges, np.log1p(area), side="right")
    weighted_near = weighted_far = common_weight = 0.0
    strata = 0
    for d in range(4):
        for a in range(4):
            current = (depth_bin == d) & (area_bin == a)
            near_stratum, far_stratum = current & near, current & far
            n_near, n_far = int(near_stratum.sum()), int(far_stratum.sum())
            if n_near < 10 or n_far < 10:
                continue
            weight = min(n_near, n_far)
            weighted_near += weight * float(np.mean(positive[near_stratum]))
            weighted_far += weight * float(np.mean(positive[far_stratum]))
            common_weight += weight
            strata += 1
    if common_weight == 0:
        return {"usable_strata": 0, "overlap_weight": 0}
    near_rate = weighted_near / common_weight
    far_rate = weighted_far / common_weight
    return {
        "usable_strata": strata,
        "overlap_weight": int(common_weight),
        "depth_area_standardized_near_positive_fraction": near_rate,
        "depth_area_standardized_far_positive_fraction": far_rate,
        "depth_area_standardized_risk_ratio": near_rate / far_rate if far_rate else None,
    }


def run(input_h5ad: Path, *, strict_gate: bool = False,
        epcam_max_target: int | None = None) -> dict:
    data = ad.read_h5ad(input_h5ad)
    genes = data.var_names
    if not genes.is_unique:
        raise ValueError("Duplicate gene identifiers")
    if "CD3E" not in genes or "EPCAM" not in genes:
        raise ValueError("CD3E and EPCAM are required for this pilot")
    if not {"nuclear_counts", "extranuclear_counts"}.issubset(data.layers):
        raise ValueError("Missing nuclear/extranuclear count layers")
    xy = np.asarray(data.obsm["spatial"][:, :2], dtype=float)
    if xy.shape != (data.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError("Invalid cell centroids")
    total = data.X
    nuc = data.layers["nuclear_counts"]
    cyto = data.layers["extranuclear_counts"]
    depth = np.asarray(total.sum(axis=1)).ravel().astype(np.int64)
    cd3e = gene_counts(total, genes, "CD3E")
    t_support = sum((gene_counts(total, genes, gene) for gene in T_SUPPORT),
                    np.zeros(data.n_obs, dtype=np.int64))
    t_distinct_support = sum(((gene_counts(total, genes, gene) > 0).astype(np.int64)
                              for gene in T_SUPPORT), np.zeros(data.n_obs, dtype=np.int64))
    epcam = gene_counts(total, genes, "EPCAM")
    # Source and target gates are based on disjoint marker sets. Readouts other
    # than EPCAM are additionally held out from the source gate.
    if strict_gate:
        t_cells = (cd3e >= 2) & (t_distinct_support >= 2) & (depth >= 20)
        epithelial_candidates = (epcam >= 10) & (depth >= 50) & ~t_cells
    else:
        t_cells = (cd3e >= 1) & (t_support >= 1) & (depth >= 20)
        epithelial_candidates = (epcam >= 3) & (depth >= 50) & ~t_cells
    if epcam_max_target is not None:
        t_cells &= epcam <= epcam_max_target
    if t_cells.sum() < 50 or epithelial_candidates.sum() < 50:
        raise ValueError(f"Insufficient marker-gated cells: T={t_cells.sum()} epithelial={epithelial_candidates.sum()}")
    t_index = np.flatnonzero(t_cells)
    nearest_um, nearest_source_index = cKDTree(xy[epithelial_candidates]).query(xy[t_cells], k=1, workers=-1)
    selected_source = np.flatnonzero(epithelial_candidates)[nearest_source_index]
    result = {
        "sample": input_h5ad.stem,
        "input": str(input_h5ad.resolve()),
        "cells": int(data.n_obs),
        "genes": int(data.n_vars),
        "gates": {"strict": strict_gate,
                  "epcam_max_target": epcam_max_target,
                  "T_cells": int(t_cells.sum()),
                  "epithelial_candidates": int(epithelial_candidates.sum()),
                  "T_rule": "CD3E>=2 and >=2 distinct support genes" if strict_gate else "CD3E>=1 and support counts>=1",
                  "source_rule": "EPCAM>=10" if strict_gate else "EPCAM>=3",
                  "T_support_genes_present": [g for g in T_SUPPORT if g in genes]},
        "distance_definition": "Euclidean distance between cell centroids, in micrometers",
        "distance_bins_um": [*DISTANCE_BINS_UM, ">=100"],
        "readouts": {},
        "near_T_cell_ids_example": data.obs_names[t_index[nearest_um <= 15][:10]].tolist(),
        "nearest_source_ids_example": data.obs_names[selected_source[:10]].tolist(),
        "interpretation_limit": "Exploratory marker-gated spatial association; does not establish segmentation error, RCTD mixture, or SPLIT correction.",
    }
    near = nearest_um <= 15.0
    far = nearest_um > 50.0
    area_column = "cell_area" if "cell_area" in data.obs else None
    area = (np.asarray(data.obs[area_column], dtype=float)[t_index]
            if area_column else np.ones(len(t_index)))
    if not np.isfinite(area).all() or np.any(area < 0):
        raise ValueError("Invalid cell area")
    result["near_far"] = {"near_le15um": int(near.sum()), "far_gt50um": int(far.sum())}
    for gene in READOUT_CANDIDATES:
        if gene not in genes:
            continue
        values = gene_counts(total, genes, gene)[t_index]
        nuclear = gene_counts(nuc, genes, gene)[t_index]
        cytoplasmic = gene_counts(cyto, genes, gene)[t_index]
        if not np.array_equal(values, nuclear + cytoplasmic):
            raise ValueError(f"Compartment count mismatch for {gene}")
        record = {
            "source_gate_independent": gene != "EPCAM",
            "bins": {},
            "near": summarize(near, values, nuclear, cytoplasmic, depth[t_index]),
            "far": summarize(far, values, nuclear, cytoplasmic, depth[t_index]),
            "depth_area_standardization": stratified_positive_rate(
                near, far, values > 0, depth[t_index], area),
        }
        for left, right in zip(DISTANCE_BINS_UM, [*DISTANCE_BINS_UM[1:], float("inf")]):
            mask = (nearest_um >= left) & (nearest_um < right)
            record["bins"][f"{left:g}-{right:g}"] = summarize(mask, values, nuclear, cytoplasmic, depth[t_index])
        if near.sum() and far.sum():
            near_rate = record["near"]["positive_fraction"]
            far_rate = record["far"]["positive_fraction"]
            record["near_far_positive_risk_ratio"] = float(near_rate / far_rate) if far_rate else None
            near_rna_rate = record["near"]["total_count_per_1000_rna"]
            far_rna_rate = record["far"]["total_count_per_1000_rna"]
            record["near_far_rna_rate_ratio"] = float(near_rna_rate / far_rna_rate) if far_rna_rate else None
        result["readouts"][gene] = record
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--strict-gate", action="store_true",
                        help="Require multiple T markers and EPCAM>=10 sources for sensitivity analysis")
    parser.add_argument("--epcam-max-target", type=int,
                        help="Sensitivity test: exclude EPCAM-high T candidates; do not interpret EPCAM readout under this filter")
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_json}")
    result = run(args.input_h5ad, strict_gate=args.strict_gate,
                 epcam_max_target=args.epcam_max_target)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"sample": result["sample"], "near_far": result["near_far"],
                      "readout_ratios": {gene: record.get("near_far_positive_risk_ratio")
                                         for gene, record in result["readouts"].items()}}, indent=2))


if __name__ == "__main__":
    main()
