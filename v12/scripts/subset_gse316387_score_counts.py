#!/usr/bin/env python3
"""Recreate the no-pretraining raw-count SVD baseline on the spatial score subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing count subset")
    index = pd.read_parquet(args.index)
    data = ad.read_h5ad(args.counts)
    wanted = index.full_cell_id.astype(str).tolist()
    if data.obs_names.has_duplicates or not set(wanted).issubset(data.obs_names):
        raise ValueError("Selected IDs missing or duplicated in original raw count matrix")
    selected = data[wanted, :].copy()
    if selected.obs_names.astype(str).tolist() != wanted:
        raise ValueError("Count subset differs from frozen score order")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.write_h5ad(args.output, compression="lzf")
    print(f"Wrote raw counts for {selected.n_obs} scored cells and {selected.n_vars} genes")


if __name__ == "__main__":
    main()
