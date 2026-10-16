#!/usr/bin/env python3
"""Keep every scored cell and its 20 nearest raw-region neighbors for TERRA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.spatial import cKDTree


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to replace existing output directory")
    parts = []
    rows = []
    for region in REGIONS:
        source = args.context_dir / f"{region}.h5ad"
        data = ad.read_h5ad(source)
        spatial = np.asarray(data.obsm["spatial"], dtype=np.float64)
        anchors = np.flatnonzero(data.obs.benchmark_id.astype(str).ne("").to_numpy())
        if len(anchors) < 1:
            raise ValueError(f"{region} lacks scored anchors")
        neighbor_indices = cKDTree(spatial).query(
            spatial[anchors], k=21, workers=4)[1]
        keep = np.unique(neighbor_indices.reshape(-1))
        if not set(anchors).issubset(keep):
            raise ValueError(f"{region} omitted a scored anchor")
        context = data[keep, :].copy()
        context.obs_names = [f"{region}:{barcode}" for barcode in context.obs_names]
        context.obs["sample"] = region
        context.obs["batch"] = region
        parts.append(context)
        row = {"region": region, "full_region_cells": data.n_obs,
               "scored_cells": len(anchors), "context_cells": context.n_obs,
               "source_sha256": sha256(source)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    joined = ad.concat(parts, join="inner", merge="same")
    if joined.obs_names.has_duplicates:
        raise ValueError("Concatenated TERRA cell IDs are duplicated")
    score_ids = joined.obs.benchmark_id.astype(str)
    if score_ids[score_ids.ne("")].duplicated().any():
        raise ValueError("TERRA context duplicates a scored cell")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / "terra_context.h5ad"
    joined.write_h5ad(target, compression="lzf")
    report = {"method": "all scored cells plus 20 nearest full-region cells",
              "rationale": "TERRA uses 10 spatial neighbors; retain 20 to preserve their local context",
              "regions": rows, "context_cells": joined.n_obs,
              "scored_cells": int(score_ids.ne("").sum()),
              "gene_features": joined.n_vars,
              "context_sha256": sha256(target),
              "score_index_sha256": sha256(args.context_dir / "score_index.parquet")}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"context_cells": joined.n_obs,
                      "score_cells": report["scored_cells"],
                      "genes": joined.n_vars}), flush=True)


if __name__ == "__main__":
    main()
