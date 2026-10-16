#!/usr/bin/env python3
"""Train-only SVD baseline for raw spatial cell counts; run on a compute node."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--components", type=int, default=50)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite existing output-dir")
    data = ad.read_h5ad(args.input_h5ad)
    index = pd.read_parquet(args.index)
    if data.n_obs != len(index) or data.obs_names.astype(str).tolist() != \
            index.full_cell_id.astype(str).tolist():
        raise ValueError("Raw counts and benchmark index do not have identical cell IDs/order")
    if "split" not in index or set(index.split) != {"train", "val", "test"}:
        raise ValueError("Benchmark index lacks complete split labels")
    if data.X is None or data.X.shape != data.shape:
        raise ValueError("Raw counts are absent")
    X = sparse.csr_matrix(data.X, dtype=np.float32)
    if not np.isfinite(X.data).all() or (X.data < 0).any() or \
            not np.equal(X.data, np.floor(X.data)).all():
        raise ValueError("Expected nonnegative, finite integer raw counts")
    library_size = np.asarray(X.sum(axis=1)).ravel()
    if (library_size <= 0).any():
        raise ValueError("Cells with zero total counts must be excluded before the benchmark")
    X = sparse.diags(1e4 / library_size) @ X
    X.data = np.log1p(X.data)
    n_train = int(index.split.eq("train").sum())
    n_components = min(args.components, X.shape[1] - 1, n_train - 1)
    if n_components < 2:
        raise ValueError("Not enough genes or training cells for SVD")
    model = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    model.fit(X[index.split.eq("train").to_numpy()])
    embedding = model.transform(X).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "embeddings.npy", embedding, allow_pickle=False)
    (args.output_dir / "cell_ids.txt").write_text(
        "\n".join(index.full_cell_id.astype(str)) + "\n", encoding="utf-8")
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "model_name": "raw_log1p_svd",
        "input_h5ad": str(args.input_h5ad.resolve()),
        "input_semantics": "raw_integer_whole_cell_counts",
        "transform": "normalize_per_cell_to_10000_then_log1p_then_train_only_TruncatedSVD",
        "components": n_components,
        "seed": 42,
        "train_cells": n_train,
        "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(embedding)} baseline embeddings with {n_components} components")


if __name__ == "__main__":
    main()
