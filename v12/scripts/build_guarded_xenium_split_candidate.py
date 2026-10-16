"""Build an experimental full-tile SPLIT count candidate with raw-count fallbacks.

The guard protects cells with strong *nuclear* T evidence if RCTD assigned a
non-T primary type. Uncorrected RCTD rejects and missing reference genes also
retain their original counts. This is a pilot artifact, not training input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite

from diagnose_xenium_spillover import gene_counts


T_MARKERS = ("CD3E", "CD3D", "LCK", "TRAC", "CD8A", "CD8B")


def run(tile_dir: Path, split_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not (split_dir / "COMPLETE.json").exists():
        raise ValueError("SPLIT output is incomplete")
    tile_ids = pd.read_csv(tile_dir / "cells.tsv", sep="\t").cell_id.astype(str).tolist()
    tile_genes = (tile_dir / "genes.tsv").read_text().splitlines()
    split_ids = (split_dir / "purified_cells.tsv").read_text().splitlines()
    split_genes = (split_dir / "purified_genes.tsv").read_text().splitlines()
    tile_cells_index = pd.Index(tile_ids)
    tile_genes_index = pd.Index(tile_genes)
    split_cell_indices = tile_cells_index.get_indexer(split_ids)
    split_gene_indices = tile_genes_index.get_indexer(split_genes)
    if (split_cell_indices < 0).any() or (split_gene_indices < 0).any():
        raise ValueError("SPLIT output contains unknown cells or genes")
    raw = mmread(str(tile_dir / "counts_genes_by_cells.mtx.gz")).tocsr()
    corrected = mmread(str(split_dir / "purified_counts.mtx")).tocsr()
    if raw.shape != (len(tile_genes), len(tile_ids)) or corrected.shape != (len(split_genes), len(split_ids)):
        raise ValueError("Count matrix dimensions disagree with labels")
    assignments = pd.read_csv(split_dir / "rctd_assignments.tsv", sep="\t").set_index("cell_id")
    primary = assignments.reindex(split_ids).first_type.fillna("")
    if (primary == "").any():
        raise ValueError("A SPLIT-corrected cell has no RCTD primary type")
    data_path = Path(json.loads((tile_dir / "manifest.json").read_text())["source"])
    data = ad.read_h5ad(data_path)
    original_idx = data.obs_names.get_indexer(split_ids)
    if (original_idx < 0).any():
        raise ValueError("SPLIT-corrected cell absent from original H5AD")
    nuclear = data.layers["nuclear_counts"][original_idx]
    nuclear_t = np.column_stack([gene_counts(nuclear, data.var_names, gene)
                                 for gene in T_MARKERS])
    strong_nuclear_t = (nuclear_t[:, 0] >= 1) & ((nuclear_t > 0).sum(axis=1) >= 2)
    t_primary = primary.str.startswith("T cell").to_numpy()
    protected = strong_nuclear_t & ~t_primary
    accepted = ~protected

    raw_shared = raw[split_gene_indices][:, split_cell_indices]
    change = (corrected[:, accepted] - raw_shared[:, accepted]).tocoo()
    if (change.data > 1e-6).any():
        raise ValueError("SPLIT output unexpectedly exceeds raw count")
    accepted_original_cells = split_cell_indices[accepted]
    delta = sparse.coo_matrix(
        (change.data, (split_gene_indices[change.row],
                       accepted_original_cells[change.col])),
        shape=raw.shape,
    ).tocsr()
    candidate = raw + delta
    candidate.eliminate_zeros()
    if not np.isfinite(candidate.data).all() or (candidate.data < -1e-6).any():
        raise ValueError("Invalid guarded candidate counts")
    if ((candidate - raw).data > 1e-6).any():
        raise ValueError("Guarded candidate exceeds raw counts")

    statuses = np.full(len(tile_ids), "rctd_rejected_or_filtered_raw", dtype=object)
    statuses[split_cell_indices] = "split_applied"
    statuses[split_cell_indices[protected]] = "nuclear_t_primary_discordance_raw"
    summary = {
        "source_tile": str(tile_dir.resolve()),
        "source_split_dir": str(split_dir.resolve()),
        "scope": "Experimental cell-by-gene total counts only; no molecule/compartment reassignment",
        "guard": "If nuclear CD3E>=1 and >=2 distinct nuclear T markers but RCTD primary is non-T, use raw counts",
        "cells": len(tile_ids), "genes": len(tile_genes),
        "split_applied_cells": int(np.count_nonzero(statuses == "split_applied")),
        "protected_nuclear_t_cells": int(protected.sum()),
        "rctd_rejected_or_filtered_raw_cells": int(np.count_nonzero(statuses == "rctd_rejected_or_filtered_raw")),
        "raw_total_count": float(raw.sum()),
        "candidate_total_count": float(candidate.sum()),
        "count_retained_fraction": float(candidate.sum() / raw.sum()),
        "warning": "Not patient-matched; the nuclear T gate is not independent identity ground truth; fractional SPLIT values need a downstream training design decision.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="guarded-split-", dir=output_dir.parent) as temp:
        temp_dir = Path(temp)
        mmwrite(str(temp_dir / "purified_counts.mtx"), candidate.tocoo())
        (temp_dir / "purified_genes.tsv").write_text("\n".join(tile_genes) + "\n")
        (temp_dir / "purified_cells.tsv").write_text("\n".join(tile_ids) + "\n")
        pd.DataFrame({"cell_id": tile_ids, "correction_status": statuses}).to_csv(
            temp_dir / "cell_correction_status.tsv", sep="\t", index=False)
        (temp_dir / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
        temp_dir.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tile_dir", type=Path)
    parser.add_argument("split_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.tile_dir, args.split_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
