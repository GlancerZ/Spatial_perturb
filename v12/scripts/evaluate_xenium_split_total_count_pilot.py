"""Compare SPLIT-corrected total counts with original Xenium near/far controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.io import mmread


MARKERS = ("MUC1", "CEACAM6", "CD3E", "CD3D", "LCK", "TRAC", "CD8A", "CD8B", "EPCAM")


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _summary(mask: np.ndarray, raw: np.ndarray, corrected: np.ndarray,
             exposure: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"cells": 0}
    raw_sum = float(raw[mask].sum())
    corrected_sum = float(corrected[mask].sum())
    return {
        "cells": n,
        "raw_total": raw_sum,
        "corrected_total": corrected_sum,
        "retained_fraction": corrected_sum / raw_sum if raw_sum else None,
        "raw_positive_fraction": float(np.mean(raw[mask] > 0)),
        "corrected_positive_fraction": float(np.mean(corrected[mask] > 1e-8)),
        "raw_per_1000_original_rna": float(1000 * raw_sum / exposure[mask].sum()),
        "corrected_per_1000_original_rna": float(1000 * corrected_sum / exposure[mask].sum()),
    }


def _standardized(near: np.ndarray, far: np.ndarray, raw: np.ndarray,
                  corrected: np.ndarray, depth: np.ndarray, area: np.ndarray) -> dict:
    pool = near | far
    log_depth = np.log1p(depth)
    log_area = np.log1p(area)
    d_edges = np.quantile(log_depth[pool], [0.25, 0.5, 0.75])
    a_edges = np.quantile(log_area[pool], [0.25, 0.5, 0.75])
    d_bin = np.searchsorted(d_edges, log_depth, side="right")
    a_bin = np.searchsorted(a_edges, log_area, side="right")
    totals = {key: 0.0 for key in ("near_raw", "far_raw", "near_corrected", "far_corrected")}
    weight_sum = 0
    strata = 0
    for d in range(4):
        for a in range(4):
            within = (d_bin == d) & (a_bin == a)
            nmask, fmask = near & within, far & within
            if nmask.sum() < 3 or fmask.sum() < 3:
                continue
            weight = int(min(nmask.sum(), fmask.sum()))
            for group, group_mask in (("near", nmask), ("far", fmask)):
                totals[f"{group}_raw"] += weight * 1000 * raw[group_mask].sum() / depth[group_mask].sum()
                totals[f"{group}_corrected"] += weight * 1000 * corrected[group_mask].sum() / depth[group_mask].sum()
            weight_sum += weight
            strata += 1
    if not weight_sum:
        return {"usable_strata": 0, "overlap_weight": 0}
    rates = {key: value / weight_sum for key, value in totals.items()}
    return {
        "usable_strata": strata, "overlap_weight": weight_sum,
        "rates_per_1000_original_rna": rates,
        "raw_near_far_ratio": rates["near_raw"] / rates["far_raw"] if rates["far_raw"] else None,
        "corrected_near_far_ratio": (
            rates["near_corrected"] / rates["far_corrected"]
            if rates["far_corrected"] else None
        ),
        "raw_near_far_difference": rates["near_raw"] - rates["far_raw"],
        "corrected_near_far_difference": rates["near_corrected"] - rates["far_corrected"],
    }


def run(tile_dir: Path, result_dir: Path, output_json: Path,
        restrict_cells_file: Path | None = None) -> dict:
    if output_json.exists():
        raise FileExistsError(output_json)
    if not (result_dir / "COMPLETE.json").exists():
        raise ValueError("SPLIT output has no COMPLETE.json")
    cells = pd.read_csv(tile_dir / "cells.tsv", sep="\t")
    tile_ids = cells.cell_id.astype(str).tolist()
    tile_genes = _lines(tile_dir / "genes.tsv")
    corrected_genes = _lines(result_dir / "purified_genes.tsv")
    corrected_ids = _lines(result_dir / "purified_cells.tsv")
    if len(corrected_ids) != len(set(corrected_ids)):
        raise ValueError("Duplicate corrected cell ID")
    raw_matrix = mmread(str(tile_dir / "counts_genes_by_cells.mtx.gz")).tocsr()
    corrected_matrix = mmread(str(result_dir / "purified_counts.mtx")).tocsr()
    if corrected_matrix.shape != (len(corrected_genes), len(corrected_ids)):
        raise ValueError("Corrected matrix/labels shape mismatch")
    if restrict_cells_file is not None:
        restricted_ids = _lines(restrict_cells_file)
        restricted_indices = pd.Index(corrected_ids).get_indexer(restricted_ids)
        if (restricted_indices < 0).any():
            raise ValueError("Restricted cell is absent from candidate matrix")
        corrected_matrix = corrected_matrix[:, restricted_indices]
        corrected_ids = restricted_ids
    raw_gene_index = pd.Index(tile_genes).get_indexer(corrected_genes)
    raw_cell_index = pd.Index(tile_ids).get_indexer(corrected_ids)
    if (raw_gene_index < 0).any() or (raw_cell_index < 0).any():
        raise ValueError("SPLIT output gene/cell absent from input")
    raw_matrix = raw_matrix[raw_gene_index][:, raw_cell_index]
    difference = corrected_matrix - raw_matrix
    if (corrected_matrix.data < -1e-8).any() or (difference.data > 1e-6).any():
        raise ValueError("Corrected counts are negative or exceed raw counts")
    cells = cells.set_index("cell_id").loc[corrected_ids]
    source_h5ad = Path(json.loads((tile_dir / "manifest.json").read_text())["source"])
    original = ad.read_h5ad(source_h5ad, backed="r")
    try:
        area = original.obs.loc[corrected_ids, "cell_area"].to_numpy(dtype=float)
    finally:
        original.file.close()
    depth = cells.total_rna.to_numpy(dtype=float)
    if not np.isfinite(area).all() or (area < 0).any() or (depth < 0).any():
        raise ValueError("Invalid cell area/depth")
    interior = cells.interior_15um.to_numpy(dtype=bool)
    strict_t = cells.t_strict_epcam_low.to_numpy(dtype=bool)
    distance = cells.nearest_source_um.to_numpy(dtype=float)
    near = strict_t & interior & (distance <= 15)
    far = strict_t & interior & (distance > 50)
    source = cells.epcam_high_source.to_numpy(dtype=bool) & interior
    assignments_path = result_dir / "rctd_assignments.tsv"
    if not assignments_path.exists():
        source_split_dir = json.loads((result_dir / "COMPLETE.json").read_text()).get("source_split_dir")
        if not source_split_dir:
            raise FileNotFoundError(assignments_path)
        assignments_path = Path(source_split_dir) / "rctd_assignments.tsv"
    assignments = pd.read_csv(assignments_path, sep="\t").set_index("cell_id")
    assignments = assignments.reindex(corrected_ids)
    primary_t = assignments.first_type.fillna("").str.startswith("T cell").to_numpy()
    result = {
        "tile": str(tile_dir), "split_result": str(result_dir),
        "n_cells_input": len(tile_ids), "n_cells_corrected": len(corrected_ids),
        "n_genes_input": len(tile_genes), "n_genes_corrected": len(corrected_genes),
        "omitted_genes": sorted(set(tile_genes) - set(corrected_genes)),
        "strict_t_near_interior": int(near.sum()),
        "strict_t_far_interior": int(far.sum()),
        "epcam_high_source_interior": int(source.sum()),
        "overall_count_retained_fraction": float(corrected_matrix.sum() / raw_matrix.sum()),
        "spot_classes": assignments.spot_class.fillna("not_in_RCTD").value_counts().to_dict(),
        "strict_t_primary_type_counts": assignments.loc[strict_t, "first_type"].fillna("not_in_RCTD").value_counts().to_dict(),
        "strict_t_near_primary_t": int((near & primary_t).sum()),
        "strict_t_far_primary_t": int((far & primary_t).sum()),
        "markers": {},
        "interpretation_limit": (
            "Observational single-tile analysis using a non-patient-matched 10x reference; "
            "readout markers also informed RCTD, so this is not a held-out validation. "
            "SPLIT corrected cell-by-gene total counts only, not molecule or compartment assignment."
        ),
    }
    for gene in MARKERS:
        if gene not in corrected_genes:
            continue
        i = corrected_genes.index(gene)
        raw = raw_matrix.getrow(i).toarray().ravel()
        corrected = corrected_matrix.getrow(i).toarray().ravel()
        result["markers"][gene] = {
            "strict_t_near": _summary(near, raw, corrected, depth),
            "strict_t_far": _summary(far, raw, corrected, depth),
            "strict_t_near_primary_t": _summary(near & primary_t, raw, corrected, depth),
            "strict_t_near_other_primary": _summary(near & ~primary_t, raw, corrected, depth),
            "strict_t_far_primary_t": _summary(far & primary_t, raw, corrected, depth),
            "strict_t_far_other_primary": _summary(far & ~primary_t, raw, corrected, depth),
            "epcam_high_source": _summary(source, raw, corrected, depth),
            "near_far_depth_area_standardized": _standardized(
                near, far, raw, corrected, depth, area
            ),
        }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile_dir", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--restrict-cells-file", type=Path)
    args = parser.parse_args()
    result = run(args.tile_dir, args.result_dir, args.output_json, args.restrict_cells_file)
    print(json.dumps({
        "corrected_cells": result["n_cells_corrected"],
        "overall_count_retained_fraction": result["overall_count_retained_fraction"],
        "strict_t_near_interior": result["strict_t_near_interior"],
        "strict_t_far_interior": result["strict_t_far_interior"],
        "MUC1": result["markers"].get("MUC1"),
        "CD3E": result["markers"].get("CD3E"),
    }, indent=2))


if __name__ == "__main__":
    main()
