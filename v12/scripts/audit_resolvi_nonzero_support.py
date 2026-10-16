"""Measure how many originally detected cell-gene entries remain positive after ResolVI.

Read-only diagnostic for compute nodes. It distinguishes count shrinkage from
removal of false-positive gene tokens; it does not assess biological truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import scipy.sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = ad.read_h5ad(args.input, backed="r")
    result = {
        "input": str(args.input.resolve()),
        "cells": data.n_obs,
        "genes": data.n_vars,
        "raw_positive_cell_gene_entries": 0,
        "corrected_positive_cell_gene_entries": 0,
        "corrected_entries_below_0_5": 0,
        "corrected_entries_below_0_1": 0,
    }
    try:
        for start in range(0, data.n_obs, 4096):
            stop = min(start + 4096, data.n_obs)
            raw = sp.csr_matrix(data.layers["raw_total_counts"][start:stop])
            corrected = sp.csr_matrix(data.X[start:stop])
            raw.eliminate_zeros()
            corrected.eliminate_zeros()
            new = corrected - corrected.multiply(raw.sign())
            new.eliminate_zeros()
            if new.nnz:
                raise ValueError("Corrected data introduced a cell-gene position absent in raw data")
            result["raw_positive_cell_gene_entries"] += raw.nnz
            result["corrected_positive_cell_gene_entries"] += corrected.nnz
            result["corrected_entries_below_0_5"] += int((corrected.data < 0.5).sum())
            result["corrected_entries_below_0_1"] += int((corrected.data < 0.1).sum())
    finally:
        data.file.close()
    total = result["raw_positive_cell_gene_entries"]
    result["positive_position_retention"] = (
        result["corrected_positive_cell_gene_entries"] / total if total else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
