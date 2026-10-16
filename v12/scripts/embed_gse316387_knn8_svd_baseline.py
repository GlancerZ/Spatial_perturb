#!/usr/bin/env python3
"""Train-only SVD baseline using each cell and its eight nearest spatial neighbors.

The neighbors are queried from complete GEO regions. Run on a compute node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--score-index", type=Path, required=True)
    parser.add_argument("--score-spatial", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to overwrite existing output directory")

    index = pd.read_parquet(args.score_index)
    spatial_index = pd.read_parquet(args.score_spatial)
    if not index.full_cell_id.equals(spatial_index.full_cell_id):
        raise ValueError("Score index and spatial mapping have different row orders")
    if set(index.split) != {"train", "val", "test"}:
        raise ValueError("Missing train, validation, or test split")

    blocks = []
    positions = []
    reference_genes = None
    total_zero_libraries = 0
    region_records = []
    for region in sorted(spatial_index.region.unique()):
        path = args.context_dir / f"{region}.h5ad"
        data = ad.read_h5ad(path)
        genes = data.var_names.astype(str).tolist()
        if reference_genes is None:
            reference_genes = genes
        elif genes != reference_genes:
            raise ValueError(f"Gene feature order differs in {region}")
        region_positions = np.flatnonzero(spatial_index.region.eq(region).to_numpy())
        selected = spatial_index.iloc[region_positions]
        barcodes = selected.barcode.astype(str).tolist()
        lookup = data.obs_names.get_indexer(barcodes)
        if (lookup < 0).any():
            raise ValueError(f"Scored cells missing from {region} raw counts")
        coords = np.asarray(data.obsm["spatial"], dtype=np.float32)
        if coords.shape != (data.n_obs, 2) or not np.isfinite(coords).all():
            raise ValueError(f"Invalid spatial coordinates in {region}")
        knn = NearestNeighbors(n_neighbors=12, algorithm="kd_tree").fit(coords)
        candidate = knn.kneighbors(coords[lookup], return_distance=False)
        neighbors = np.empty((len(lookup), 8), dtype=np.int64)
        for i, row in enumerate(candidate):
            other = row[row != lookup[i]]
            if len(other) < 8:
                raise ValueError(f"Fewer than eight neighbors for cell {i} in {region}")
            neighbors[i] = other[:8]

        used, inverse = np.unique(np.concatenate([lookup, neighbors.ravel()]),
                                  return_inverse=True)
        X = sparse.csr_matrix(data.X[used], dtype=np.float32)
        if not np.isfinite(X.data).all() or (X.data < 0).any() or \
                not np.equal(X.data, np.floor(X.data)).all():
            raise ValueError(f"Invalid raw counts in {region}")
        library = np.asarray(X.sum(axis=1)).ravel()
        total_zero_libraries += int((library == 0).sum())
        scale = np.zeros_like(library, dtype=np.float32)
        np.divide(1e4, library, out=scale, where=library > 0)
        X = sparse.diags(scale) @ X
        X.data = np.log1p(X.data)
        n_scored = len(lookup)
        own = X[inverse[:n_scored]]
        neighbor_local = inverse[n_scored:].reshape(n_scored, 8)
        W = sparse.csr_matrix(
            (np.full(n_scored * 8, 1 / 8, dtype=np.float32),
             (np.repeat(np.arange(n_scored), 8), neighbor_local.ravel())),
            shape=(n_scored, len(used)),
        )
        neighborhood = W @ X
        blocks.append(sparse.hstack([own, neighborhood], format="csr"))
        positions.append(region_positions)
        region_records.append({"region": region, "full_cells": int(data.n_obs),
                               "scored_cells": n_scored,
                               "context_h5ad_sha256": sha256(path)})
        print(f"{region}: {n_scored} scored cells, {len(used)} count rows",
              flush=True)

    order = np.argsort(np.concatenate(positions))
    if not np.array_equal(np.concatenate(positions)[order], np.arange(len(index))):
        raise ValueError("Score cells were duplicated or omitted")
    features = sparse.vstack(blocks, format="csr")[order]
    train = index.split.eq("train").to_numpy()
    svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
    svd.fit(features[train])
    embedding = svd.transform(features).astype(np.float32)
    if not np.isfinite(embedding).all():
        raise ValueError("Nonfinite embedding")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "embeddings.npy", embedding, allow_pickle=False)
    (args.output_dir / "cell_ids.txt").write_text(
        "\n".join(index.full_cell_id.astype(str)) + "\n", encoding="utf-8")
    provenance = {
        "model_name": "expression_knn8_svd",
        "input": "raw Xenium counts and complete GEO-region centroids",
        "transform": "per-cell 10k-normalized log1p expression; concatenate self and mean of 8 nearest cells; train-only SVD50",
        "neighbors": 8,
        "training_cells": int(train.sum()),
        "scored_cells": len(index),
        "zero_library_context_cells": total_zero_libraries,
        "score_index_sha256": sha256(args.score_index),
        "score_spatial_sha256": sha256(args.score_spatial),
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "regions": region_records,
        "seed": 42,
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(index), "embedding_dim": embedding.shape[1]}),
          flush=True)


if __name__ == "__main__":
    main()
