#!/usr/bin/env python3
"""Frozen Novae human-0 embedding on complete Xenium region graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import novae
import numpy as np
import pandas as pd
import torch


REGIONS = (
    "GSM9451442_R1_2",
    "GSM9451445_R1_4",
    "GSM9451448_R1_6",
    "GSM9451458_R2_6",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to replace existing output directory")
    score_index_path = args.context_dir / "score_index.parquet"
    score_index = pd.read_parquet(score_index_path)
    adatas = [ad.read_h5ad(args.context_dir / f"{region}.h5ad") for region in REGIONS]
    if sum(data.n_obs for data in adatas) != 382346:
        raise ValueError("Full-context cell count differs from the frozen manifest")
    if any("spatial" not in data.obsm or "benchmark_id" not in data.obs
           for data in adatas):
        raise ValueError("Region lacks spatial coordinates or frozen-cell mapping")
    np.random.seed(42)
    torch.manual_seed(42)
    novae.spatial_neighbors(adatas, radius=80)
    graph_edges = {region: int(data.obsp["spatial_connectivities"].nnz)
                   for region, data in zip(REGIONS, adatas, strict=True)}
    print(json.dumps({"graph_edges": graph_edges}), flush=True)
    model = novae.Novae.from_pretrained(str(args.checkpoint_dir))
    model.compute_representations(adatas, zero_shot=True,
                                  accelerator="gpu", num_workers=4)
    vectors = {}
    dimension = None
    for region, data in zip(REGIONS, adatas, strict=True):
        embedding = np.asarray(data.obsm["novae_latent"], dtype=np.float32)
        if embedding.shape[0] != data.n_obs or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid Novae embedding for {region}")
        if dimension is None:
            dimension = embedding.shape[1]
        elif embedding.shape[1] != dimension:
            raise ValueError("Novae embedding width changed across regions")
        mask = data.obs.benchmark_id.astype(str).ne("").to_numpy()
        for cell_id, vector in zip(data.obs.benchmark_id[mask],
                                   embedding[mask], strict=True):
            if cell_id in vectors:
                raise ValueError(f"Duplicate scored cell {cell_id}")
            vectors[cell_id] = vector
        print(json.dumps({"region": region, "selected": int(mask.sum()),
                          "embedding_dim": dimension}), flush=True)
    ids = score_index.full_cell_id.astype(str).to_numpy()
    if set(ids) != set(vectors):
        raise ValueError(f"Scored ID mismatch: expected {len(ids)}, got {len(vectors)}")
    X = np.stack([vectors[cell_id] for cell_id in ids]).astype(np.float32)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "embeddings.npz",
                        cell_ids=ids.astype("U"), embedding=X)
    provenance = {"model_name": "Novae human-0", "software_version": novae.__version__,
                  "model_source": "prism-oncology/novae-human-0",
                  "checkpoint_sha256": sha256(args.checkpoint_dir / "model.safetensors"),
                  "input": "raw Xenium gene counts plus complete-region cell centroids",
                  "spatial_graph": "Delaunay within each GEO region; edges over 80 um pruned",
                  "inference": "official frozen checkpoint; zero_shot=True; no labels supplied",
                  "context_cells": sum(data.n_obs for data in adatas),
                  "score_cells": len(ids), "score_index_sha256": sha256(score_index_path),
                  "context_manifest_sha256": sha256(args.context_dir / "manifest.json"),
                  "graph_edges": graph_edges, "seed": 42,
                  "embedding_dim": int(X.shape[1])}
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(ids), "embedding_dim": X.shape[1]}), flush=True)


if __name__ == "__main__":
    main()
