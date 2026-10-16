#!/usr/bin/env python3
"""Prepare four complete Xenium regions plus a frozen patient-split score cohort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


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
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--centroid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Refusing to replace existing output directory")
    index = pd.read_parquet(args.index)
    spatial = pd.read_parquet(args.spatial)
    if not index.full_cell_id.equals(spatial.full_cell_id):
        raise ValueError("Centroids differ from frozen cell order")
    chosen = spatial.region.isin(REGIONS)
    score_index = index.loc[chosen].copy().reset_index(drop=True)
    score_spatial = spatial.loc[chosen].copy().reset_index(drop=True)
    if set(score_index.split) != {"train", "val", "test"}:
        raise ValueError("Chosen regions lack a score split")
    score_spatial["barcode"] = score_spatial.full_cell_id.str.rsplit("_", n=1).str[-1]
    if score_spatial.duplicated(["region", "barcode"]).any():
        raise ValueError("Selected region/barcode pairs are duplicated")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    score_index.to_parquet(args.output_dir / "score_index.parquet", index=False)
    score_spatial.to_parquet(args.output_dir / "score_spatial.parquet", index=False)
    records = []
    for region in REGIONS:
        h5 = args.raw_dir / f"{region}_cell_feature_matrix.h5"
        centroid = args.centroid_dir / f"{region}_cells.parquet.gz"
        adata = sc.read_10x_h5(h5, gex_only=True)
        if adata.obs_names.has_duplicates:
            raise ValueError(f"{region} raw matrix duplicates barcodes")
        adata.var_names_make_unique()
        with gzip.open(centroid, "rb") as stream:
            coords = pd.read_parquet(io.BytesIO(stream.read()))
        if coords.cell_id.duplicated().any():
            raise ValueError(f"{region} centroid table duplicates barcodes")
        coords = coords.set_index("cell_id").reindex(adata.obs_names)
        if coords[["x_centroid", "y_centroid"]].isna().any().any():
            raise ValueError(f"{region} matrix cells lack centroids")
        values = coords[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{region} has nonfinite coordinates")
        adata.obsm["spatial"] = values
        adata.obs["region"] = region
        subset = score_spatial.loc[score_spatial.region.eq(region)]
        lookup = dict(zip(subset.barcode, subset.full_cell_id, strict=True))
        missing = sorted(set(lookup) - set(adata.obs_names))
        if missing:
            raise ValueError(f"{region} lacks {len(missing)} scored matrix cells: {missing[:3]}")
        adata.obs["benchmark_id"] = [lookup.get(name, "") for name in adata.obs_names]
        target = args.output_dir / f"{region}.h5ad"
        adata.write_h5ad(target, compression="lzf")
        row = {"region": region, "raw_cells": adata.n_obs,
               "gene_features": adata.n_vars, "score_cells": len(subset),
               "score_splits": score_index.loc[score_spatial.region.eq(region),
                                                 "split"].value_counts().to_dict(),
               "raw_h5_sha256": sha256(h5),
               "centroid_sha256": sha256(centroid),
               "context_h5ad_sha256": sha256(target)}
        records.append(row)
        print(json.dumps(row), flush=True)
    report = {"regions": records,
              "score_cells": len(score_index),
              "score_splits": score_index.split.value_counts().to_dict(),
              "score_index_sha256": sha256(args.output_dir / "score_index.parquet"),
              "original_index_sha256": sha256(args.index),
              "original_spatial_sha256": sha256(args.spatial)}
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"score_cells": len(score_index),
                      "full_context_cells": sum(row["raw_cells"] for row in records)}),
          flush=True)


if __name__ == "__main__":
    main()
