"""Audit nuclear T-marker support when Xenium marker gates disagree with RCTD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from diagnose_xenium_spillover import gene_counts


T_MARKERS = ("CD3E", "CD3D", "LCK", "TRAC", "CD8A", "CD8B")
EPITHELIAL_MARKERS = ("EPCAM", "MUC1", "CEACAM6")


def summarize(mask: np.ndarray, nuclear_t: np.ndarray, nuclear_t_strong: np.ndarray,
              extranuclear_t: np.ndarray,
              nuclear_epithelial: np.ndarray, area: np.ndarray, depth: np.ndarray) -> dict:
    if not mask.any():
        return {"cells": 0}
    return {
        "cells": int(mask.sum()),
        "nuclear_t_positive_fraction": float(np.mean(nuclear_t[mask] > 0)),
        "nuclear_t_strong_fraction": float(np.mean(nuclear_t_strong[mask])),
        "nuclear_t_mean": float(np.mean(nuclear_t[mask])),
        "nuclear_t_median": float(np.median(nuclear_t[mask])),
        "extranuclear_t_mean": float(np.mean(extranuclear_t[mask])),
        "nuclear_epithelial_positive_fraction": float(np.mean(nuclear_epithelial[mask] > 0)),
        "nuclear_epithelial_mean": float(np.mean(nuclear_epithelial[mask])),
        "median_area_um2": float(np.median(area[mask])),
        "median_total_rna": float(np.median(depth[mask])),
    }


def run(tile_dir: Path, split_dir: Path, output_json: Path) -> dict:
    if output_json.exists():
        raise FileExistsError(output_json)
    tile = pd.read_csv(tile_dir / "cells.tsv", sep="\t").set_index("cell_id")
    corrected_ids = (split_dir / "purified_cells.tsv").read_text().splitlines()
    tile = tile.loc[corrected_ids]
    assignments = pd.read_csv(split_dir / "rctd_assignments.tsv", sep="\t").set_index("cell_id")
    assignments = assignments.loc[corrected_ids]
    input_path = Path(json.loads((tile_dir / "manifest.json").read_text())["source"])
    data = ad.read_h5ad(input_path)
    idx = data.obs_names.get_indexer(corrected_ids)
    if (idx < 0).any():
        raise ValueError("Corrected cells absent from original H5AD")
    nuclear = data.layers["nuclear_counts"][idx]
    extranuclear = data.layers["extranuclear_counts"][idx]
    genes = data.var_names
    nuclear_t_matrix = np.column_stack([gene_counts(nuclear, genes, gene)
                                        for gene in T_MARKERS])
    nuclear_t = nuclear_t_matrix.sum(axis=1)
    nuclear_t_strong = (nuclear_t_matrix[:, 0] >= 1) & (
        (nuclear_t_matrix > 0).sum(axis=1) >= 2)
    extranuclear_t = sum((gene_counts(extranuclear, genes, gene) for gene in T_MARKERS),
                         np.zeros(len(idx), dtype=np.int64))
    nuclear_epithelial = sum((gene_counts(nuclear, genes, gene) for gene in EPITHELIAL_MARKERS),
                             np.zeros(len(idx), dtype=np.int64))
    strict_t = tile.t_strict_epcam_low.to_numpy(dtype=bool)
    interior = tile.interior_15um.to_numpy(dtype=bool)
    distance = tile.nearest_source_um.to_numpy(dtype=float)
    primary_t = assignments.first_type.fillna("").str.startswith("T cell").to_numpy()
    area = data.obs.iloc[idx]["cell_area"].to_numpy(dtype=float)
    depth = tile.total_rna.to_numpy(dtype=float)
    groups = {}
    for region, region_mask in (("near", distance <= 15), ("far", distance > 50)):
        for label, label_mask in (("rctd_t_primary", primary_t),
                                  ("rctd_other_primary", ~primary_t)):
            mask = strict_t & interior & region_mask & label_mask
            groups[f"{region}_{label}"] = summarize(
                mask, nuclear_t, nuclear_t_strong, extranuclear_t,
                nuclear_epithelial, area, depth
            )
    result = {"input": str(input_path), "groups": groups,
              "strong_nuclear_t_gate": "nuclear CD3E>=1 and >=2 distinct nuclear T markers",
              "strong_nuclear_t_cells": int(nuclear_t_strong.sum()),
              "strong_nuclear_t_non_t_primary": int((nuclear_t_strong & ~primary_t).sum()),
              "limitation": "Nuclear overlap is supportive but not independent cell identity ground truth."}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile_dir", type=Path)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.tile_dir, args.split_dir, args.output_json), indent=2))


if __name__ == "__main__":
    main()
