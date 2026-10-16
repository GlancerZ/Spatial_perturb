#!/usr/bin/env python3
"""Check that GEO region raw counts equal merged RDS counts on every scored cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-counts", type=Path, required=True)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--spatial-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing audit")
    merged = ad.read_h5ad(args.merged_counts)
    mapping = pd.read_parquet(args.spatial_index)
    if merged.obs_names.astype(str).tolist() != mapping.full_cell_id.astype(str).tolist():
        raise ValueError("Merged counts and spatial score index differ in cell order")
    reference_genes = merged.var_names.astype(str)
    records = []
    for region in sorted(mapping.region.unique()):
        data = ad.read_h5ad(args.context_dir / f"{region}.h5ad")
        region_rows = np.flatnonzero(mapping.region.eq(region).to_numpy())
        selected = mapping.iloc[region_rows]
        source_rows = data.obs_names.get_indexer(selected.barcode.astype(str))
        gene_rows = data.var_names.get_indexer(reference_genes)
        if (source_rows < 0).any() or (gene_rows < 0).any() or \
                len(data.var_names) != len(reference_genes):
            raise ValueError(f"Missing raw cells or genes in {region}")
        source = sparse.csr_matrix(data.X[source_rows, :][:, gene_rows],
                                   dtype=np.float64)
        expected = sparse.csr_matrix(merged.X[region_rows, :], dtype=np.float64)
        difference = source - expected
        difference.eliminate_zeros()
        row = {"region": region, "cells": len(region_rows),
               "genes": len(reference_genes), "different_entries": int(difference.nnz),
               "maximum_absolute_difference": float(np.max(np.abs(difference.data)))
               if difference.nnz else 0.0}
        records.append(row)
        print(json.dumps(row), flush=True)
    result = {"cells": len(mapping), "genes": len(reference_genes),
              "all_counts_identical": all(row["different_entries"] == 0
                                          for row in records), "regions": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_counts_identical"]:
        raise ValueError("Merged counts differ from GEO region raw matrices")


if __name__ == "__main__":
    main()
