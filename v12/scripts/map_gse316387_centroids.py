#!/usr/bin/env python3
"""Match GEO Xenium centroids to the frozen GSE316387 cells on a compute node."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--centroid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Output directory already exists; refusing to replace it")
    index = pd.read_parquet(args.index)
    ids = index.full_cell_id.astype(str)
    if not ids.is_unique:
        raise ValueError("Frozen cell IDs are not unique")
    barcodes = ids.str.rsplit("_", n=1).str[-1]
    if (barcodes == ids).any():
        raise ValueError("Frozen cell ID lacks a Patient..._ prefix")
    if not barcodes.is_unique:
        raise ValueError(f"Frozen Xenium barcodes repeat across patients: {barcodes.duplicated().sum()}")
    rows = []
    audit = []
    wanted = set(barcodes)
    for path in sorted(args.centroid_dir.glob("*_cells.parquet.gz")):
        with gzip.open(path, "rb") as stream:
            frame = pd.read_parquet(io.BytesIO(stream.read()))
        required = {"cell_id", "x_centroid", "y_centroid"}
        if not required.issubset(frame):
            raise ValueError(f"{path.name} missing {sorted(required - set(frame))}")
        if frame.cell_id.duplicated().any():
            raise ValueError(f"{path.name} duplicates raw cell IDs")
        matched = frame.loc[frame.cell_id.astype(str).isin(wanted),
                            ["cell_id", "x_centroid", "y_centroid"]].copy()
        matched["region"] = path.name.removesuffix("_cells.parquet.gz")
        rows.append(matched)
        audit.append({"file": path.name, "raw_cells": len(frame),
                      "matched_barcodes": len(matched),
                      "columns": list(frame.columns)})
    if len(audit) != 15:
        raise ValueError(f"Expected 15 GEO region tables, found {len(audit)}")
    matched = pd.concat(rows, ignore_index=True)
    multiplicity = matched.cell_id.value_counts()
    candidates = pd.DataFrame({"full_cell_id": ids.to_numpy(),
                               "sample": index["sample"].astype(str).to_numpy(),
                               "cell_id": barcodes.to_numpy()}).merge(
        matched, on="cell_id", how="left", validate="one_to_many")
    unique_rows = candidates.loc[candidates.cell_id.map(multiplicity).eq(1)]
    sample_counts = unique_rows.groupby(["sample", "region"]).size()
    sample_region = {}
    sample_audit = {}
    for sample, counts in sample_counts.groupby(level=0):
        counts = counts.droplevel(0).sort_values(ascending=False)
        ratio = float(counts.iloc[0] / counts.sum())
        sample_audit[sample] = {"region": str(counts.index[0]),
                                "support": int(counts.iloc[0]),
                                "fraction": ratio}
        if counts.iloc[0] < 10 or ratio < 0.95:
            raise ValueError(f"Cannot identify a reliable GEO region for sample {sample}: {counts.to_dict()}")
        sample_region[sample] = counts.index[0]
    if set(index["sample"].astype(str)) != set(sample_region):
        raise ValueError("Some frozen samples lack a GEO region mapping")
    candidates = candidates.loc[
        candidates.region.eq(candidates["sample"].map(sample_region))].copy()
    target_counts = candidates.full_cell_id.value_counts()
    unique = int(target_counts.eq(1).sum())
    missing_ids = ids.loc[~ids.isin(target_counts.index)]
    missing = len(missing_ids)
    ambiguous = int(target_counts.gt(1).sum())
    report = {"frozen_cells": len(index), "unique_matches": unique,
              "missing": missing, "ambiguous": ambiguous, "regions": audit,
              "sample_region": sample_audit,
              "globally_ambiguous_barcodes": multiplicity[multiplicity > 1].head(20).to_dict(),
              "missing_examples": missing_ids.head(20).tolist()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "centroid_mapping_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("frozen_cells", "unique_matches", "missing", "ambiguous")}), flush=True)
    if missing or ambiguous:
        raise ValueError("Cannot assign every frozen cell to a unique Xenium region")
    selected = candidates.set_index("full_cell_id").loc[ids.to_numpy()]
    out = pd.DataFrame({"full_cell_id": ids.to_numpy(),
                        "region": selected.region.to_numpy(),
                        "x_centroid": selected.x_centroid.to_numpy(dtype=float),
                        "y_centroid": selected.y_centroid.to_numpy(dtype=float)})
    if not np.isfinite(out[["x_centroid", "y_centroid"]].to_numpy()).all():
        raise ValueError("Matched coordinates contain nonfinite values")
    out.to_parquet(args.output_dir / "cell_spatial.parquet", index=False)
    print(f"Wrote {len(out)} spatial coordinates in frozen cell order", flush=True)


if __name__ == "__main__":
    main()
