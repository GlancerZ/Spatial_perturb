"""Compare proportional and nucleus-preserving splits of a fixed corrected total.

This is a read-only diagnostic; it does not assert either split is biologically correct.
Run on a Slurm compute node in the isolated preprocessing environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    data = ad.read_h5ad(args.input, backed="r")
    metrics = {
        "input": str(args.input.resolve()),
        "cells": data.n_obs,
        "genes": data.n_vars,
        "raw_nuclear": 0.0,
        "raw_extranuclear": 0.0,
        "corrected_total": 0.0,
        "proportional_nuclear": 0.0,
        "cytoplasm_first_nuclear": 0.0,
        "positions_where_corrected_total_below_raw_nuclear": 0,
        "shortfall_below_raw_nuclear": 0.0,
    }
    try:
        for start in range(0, data.n_obs, 4096):
            stop = min(start + 4096, data.n_obs)
            raw_nuc = sp.csr_matrix(data.layers["raw_nuclear_counts"][start:stop], dtype=np.float64)
            raw_cyto = sp.csr_matrix(data.layers["raw_extranuclear_counts"][start:stop], dtype=np.float64)
            corr_total = sp.csr_matrix(data.X[start:stop], dtype=np.float64)
            prop_nuc = sp.csr_matrix(data.layers["nuclear_counts"][start:stop], dtype=np.float64)
            cyto_first_nuc = raw_nuc.minimum(corr_total).tocsr()
            shortfall = (raw_nuc - cyto_first_nuc).tocsr()
            shortfall.eliminate_zeros()
            metrics["raw_nuclear"] += float(raw_nuc.sum())
            metrics["raw_extranuclear"] += float(raw_cyto.sum())
            metrics["corrected_total"] += float(corr_total.sum())
            metrics["proportional_nuclear"] += float(prop_nuc.sum())
            metrics["cytoplasm_first_nuclear"] += float(cyto_first_nuc.sum())
            metrics["positions_where_corrected_total_below_raw_nuclear"] += int(shortfall.nnz)
            metrics["shortfall_below_raw_nuclear"] += float(shortfall.sum())
    finally:
        data.file.close()
    metrics["proportional_nuclear_retention"] = (
        metrics["proportional_nuclear"] / metrics["raw_nuclear"]
        if metrics["raw_nuclear"] else None
    )
    metrics["cytoplasm_first_nuclear_retention"] = (
        metrics["cytoplasm_first_nuclear"] / metrics["raw_nuclear"]
        if metrics["raw_nuclear"] else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
