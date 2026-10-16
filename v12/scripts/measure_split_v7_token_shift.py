"""Measure hypothetical V7 whole-cell token changes from SPLIT total counts.

This is a sensitivity audit only. V7's production encoder rejects fractional
counts; no model or cache is modified here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import mmread


def bins(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.int16)
    positive = values > 0
    if positive.any():
        selected = values[positive]
        edges = np.quantile(selected, np.linspace(0, 1, 50))
        result[positive] = np.digitize(selected, edges)
    return result


def evaluate(tile_dir: Path, candidate_dir: Path, cell_ids_file: Path,
             gene_ids_file: Path | None = None,
             round_candidate: bool = False) -> dict:
    tile_ids = pd.read_csv(tile_dir / "cells.tsv", sep="\t").cell_id.astype(str).tolist()
    tile_genes = (tile_dir / "genes.tsv").read_text().splitlines()
    candidate_ids = (candidate_dir / "purified_cells.tsv").read_text().splitlines()
    candidate_genes = (candidate_dir / "purified_genes.tsv").read_text().splitlines()
    selected_genes = (candidate_genes if gene_ids_file is None
                      else gene_ids_file.read_text().splitlines())
    selected_ids = cell_ids_file.read_text().splitlines()
    raw = mmread(str(tile_dir / "counts_genes_by_cells.mtx.gz")).tocsc()
    candidate = mmread(str(candidate_dir / "purified_counts.mtx")).tocsc()
    if raw.shape != (len(tile_genes), len(tile_ids)) or candidate.shape != (len(candidate_genes), len(candidate_ids)):
        raise ValueError("Matrix and labels differ")
    raw_gene_indices = pd.Index(tile_genes).get_indexer(selected_genes)
    candidate_gene_indices = pd.Index(candidate_genes).get_indexer(selected_genes)
    raw_cell_indices = pd.Index(tile_ids).get_indexer(selected_ids)
    candidate_cell_indices = pd.Index(candidate_ids).get_indexer(selected_ids)
    if ((raw_gene_indices < 0).any() or (candidate_gene_indices < 0).any()
            or (raw_cell_indices < 0).any() or (candidate_cell_indices < 0).any()):
        raise ValueError("Cannot align pilot matrices")
    raw = raw[raw_gene_indices][:, raw_cell_indices]
    candidate = candidate[candidate_gene_indices][:, candidate_cell_indices]
    result = {"cells": len(selected_ids), "genes": len(selected_genes),
              "candidate_transform": "nearest_integer_sensitivity_only" if round_candidate else "fractional_as_output",
              "original_positive_tokens": 0, "lost_positive_tokens": 0,
              "fractional_candidate_positive_tokens": 0,
              "changed_bins_on_original_positive_tokens": 0,
              "absolute_bin_change_sum": 0,
              "cells_with_any_bin_change": 0,
              "cells_with_any_support_loss": 0,
              "changed_bins_per_cell": []}
    for cell in range(len(selected_ids)):
        original = raw.getcol(cell).toarray().ravel()
        corrected = candidate.getcol(cell).toarray().ravel()
        if round_candidate:
            corrected = np.rint(corrected)
        original_positive = original > 0
        corrected_positive = corrected > 0
        b0, b1 = bins(original), bins(corrected)
        changed = b0 != b1
        lost = original_positive & ~corrected_positive
        result["original_positive_tokens"] += int(original_positive.sum())
        result["lost_positive_tokens"] += int(lost.sum())
        result["fractional_candidate_positive_tokens"] += int((
            corrected_positive & (np.abs(corrected - np.rint(corrected)) > 1e-6)).sum())
        result["changed_bins_on_original_positive_tokens"] += int((changed & original_positive).sum())
        result["absolute_bin_change_sum"] += int(np.abs(b0[original_positive] - b1[original_positive]).sum())
        result["cells_with_any_bin_change"] += int(changed.any())
        result["cells_with_any_support_loss"] += int(lost.any())
        result["changed_bins_per_cell"].append(int((changed & original_positive).sum()))
    positive = result["original_positive_tokens"]
    result["positive_token_support_loss_fraction"] = result["lost_positive_tokens"] / positive
    result["positive_token_bin_change_fraction"] = result["changed_bins_on_original_positive_tokens"] / positive
    result["mean_absolute_bin_change_per_original_positive_token"] = result["absolute_bin_change_sum"] / positive
    result["median_changed_bins_per_cell"] = float(np.median(result.pop("changed_bins_per_cell")))
    result["fraction_cells_any_bin_change"] = result["cells_with_any_bin_change"] / result["cells"]
    result["fraction_cells_any_support_loss"] = result["cells_with_any_support_loss"] / result["cells"]
    result["note"] = (
        "Nearest-integer sensitivity only; rounding is not a validated biological correction."
        if round_candidate else
        "Hypothetical unmasked whole-cell bins with fractional counts; production V7 requires integer inputs."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile_dir", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("matched_cell_ids", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--restrict-genes-file", type=Path)
    parser.add_argument("--round-candidate", action="store_true",
                        help="Sensitivity audit only; nearest-integer rounding is not a validated correction")
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    result = evaluate(args.tile_dir, args.candidate_dir, args.matched_cell_ids,
                      args.restrict_genes_file, args.round_candidate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
