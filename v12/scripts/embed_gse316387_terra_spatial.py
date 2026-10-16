#!/usr/bin/env python3
"""Frozen TERRA-96M spatial-cell embedding with complete local Xenium neighbors."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import terra
import torch


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
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.cache_dir.exists():
        parser.error("Refusing to replace an existing TERRA result or token cache")
    if not torch.cuda.is_available():
        raise RuntimeError("TERRA requires a visible GPU")
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    torch.manual_seed(42)
    index = pd.read_parquet(args.score_index)
    context_path = args.context_dir / "terra_context.h5ad"
    data = ad.read_h5ad(context_path)
    if not data.obs_names.is_unique or "spatial" not in data.obsm:
        raise ValueError("TERRA context lacks unique cells or real spatial coordinates")
    expected = index.full_cell_id.astype(str).to_numpy()
    selected = data.obs.benchmark_id.astype(str)
    if set(selected[selected.ne("")]) != set(expected):
        raise ValueError("TERRA context differs from the frozen score cohort")
    print(json.dumps({"context_cells": data.n_obs,
                      "score_cells": len(expected),
                      "visible_gpus": torch.cuda.device_count()}), flush=True)
    data = terra.harmonize_tokenize_embed_pipeline(
        adata=data, sample_key="sample", batch_key="batch",
        model_folder_path=str(args.checkpoint_dir),
        cache_directory_path=str(args.cache_dir),
        nproc=4, processing_mode="parallel", batch_size=64,
        pin_memory=True, num_workers=4)
    print(json.dumps({"harmonized_cells": data.n_obs,
                      "obsm_keys": list(data.obsm.keys())}), flush=True)
    if "spatial_cell_emb" not in data.obsm or "benchmark_id" not in data.obs:
        raise ValueError("TERRA did not return a scored spatial-cell embedding")
    mask = data.obs.benchmark_id.astype(str).ne("").to_numpy()
    selected_ids = data.obs.benchmark_id.astype(str).to_numpy()[mask]
    vectors = np.asarray(data.obsm["spatial_cell_emb"], dtype=np.float32)[mask]
    if len(selected_ids) != len(expected) or len(set(selected_ids)) != len(expected):
        raise ValueError("TERRA returned missing or duplicated scored cells")
    lookup = dict(zip(selected_ids, vectors, strict=True))
    X = np.stack([lookup[cell_id] for cell_id in expected]).astype(np.float32)
    if X.ndim != 2 or X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError("Invalid TERRA spatial-cell embeddings")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output_dir / "embeddings.npz",
                        cell_ids=expected.astype("U"), embedding=X)
    provenance = {"model_name": "TERRA-96M spatial-cell embedding",
                  "software_version": terra.__version__,
                  "model_source": "lotfollahi-lab/TERRA-96M",
                  "checkpoint_sha256": sha256(args.checkpoint_dir / "model_checkpoint.pt"),
                  "input": "raw Xenium counts plus real cell centroids",
                  "context": "every scored cell and its 20 nearest full-region cells; official tokenizer uses 10 spatial neighbors",
                  "inference": "official frozen checkpoint; no labels supplied; spatial_cell_emb",
                  "context_cells": data.n_obs, "score_cells": len(expected),
                  "score_index_sha256": sha256(args.score_index),
                  "context_manifest_sha256": sha256(args.context_dir / "manifest.json"),
                  "seed": 42, "embedding_dim": int(X.shape[1])}
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(expected), "embedding_dim": X.shape[1]}),
          flush=True)


if __name__ == "__main__":
    main()
