#!/usr/bin/env python3
"""Match ENACT expert nuclei and CRC domain bins to Space Ranger 4 cells.

Run in a Slurm compute step. The two publications annotate the same physical
10x CRC section but use different processing versions, so neither cell IDs nor
pixel coordinates are assumed to agree without explicit checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
from zipfile import ZipFile


CLASSES = {"epithelial cells", "immune cells", "stromal cells"}


def require_compute_node() -> None:
    host = socket.gethostname().split(".")[0]
    node = os.environ.get("SLURMD_NODENAME", "").split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != node:
        raise RuntimeError("Run inside a verified Slurm compute-node step")


def inside_ring(x: float, y: float, ring) -> bool:
    """Even-odd point-in-polygon for one closed GeoJSON ring."""
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < cross:
                inside = not inside
        previous = current
    return inside


def inside_polygon(x: float, y: float, rings) -> bool:
    return inside_ring(x, y, rings[0]) and not any(
        inside_ring(x, y, hole) for hole in rings[1:]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compartments", type=Path, required=True)
    parser.add_argument("--feature-slice", type=Path, required=True)
    parser.add_argument("--expert-zip", type=Path, required=True)
    parser.add_argument("--region-csv", type=Path, required=True)
    parser.add_argument("--barcode-mappings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-expert-matches", type=int, default=5000)
    parser.add_argument("--min-region-bins", type=int, default=4)
    parser.add_argument("--min-region-purity", type=float, default=0.6)
    args = parser.parse_args()
    require_compute_node()

    import anndata as ad
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    if not 0.5 < args.min_region_purity <= 1:
        raise ValueError("min-region-purity must be in (0.5, 1]")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    data = ad.read_h5ad(args.compartments, backed="r")
    if "nuclear_spatial_um" not in data.obsm or "spatial_um" not in data.obsm:
        raise ValueError("Compartment input needs nuclear and cell centroids in micrometres")
    cell_ids = np.asarray(data.obs_names, dtype=str)
    if len(np.unique(cell_ids)) != len(cell_ids):
        raise ValueError("Duplicate Space Ranger cell IDs")
    nuclear_um = np.asarray(data.obsm["nuclear_spatial_um"], dtype=np.float64)
    if nuclear_um.shape != (len(cell_ids), 2):
        raise ValueError("Invalid nuclear centroid shape")
    with h5py.File(args.feature_slice, "r") as handle:
        metadata = json.loads(handle.attrs["metadata_json"])
    if metadata.get("sample_id") != "Visium_HD_Human_Colon_Cancer":
        raise ValueError("Wrong Space Ranger sample")
    pitch = float(metadata["spot_pitch"])
    if abs(pitch - 2.) > 1e-6:
        raise ValueError("Expected native 2 um grid")
    matrix = np.asarray(
        metadata["transform_matrices"]["spot_colrow_to_microscope_colrow"],
        dtype=np.float64,
    )
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("Invalid spot-to-microscope matrix")
    with ZipFile(args.expert_zip) as archive:
        candidates = [name for name in archive.namelist()
                      if name.endswith(".geojson") and "Human_Colon_Cancer" in name
                      and not name.startswith("__MACOSX/")]
        if len(candidates) != 1:
            raise ValueError(f"Expected one CRC expert GeoJSON, found {candidates}")
        filename = Path(candidates[0]).name
        crop = re.search(r"-wsi-(\d+)_(\d+)_(\d+)_(\d+)\.geojson$", filename)
        if crop is None:
            raise ValueError("Unknown QuPath WSI crop naming convention")
        left, top, right, bottom = (int(crop.group(i)) for i in range(1, 5))
        width, height = right - left, bottom - top
        if width <= 0 or height <= 0:
            raise ValueError("Invalid ENACT crop bounds")
        with archive.open(candidates[0]) as handle:
            expert = json.load(handle)
    if expert.get("type") != "FeatureCollection":
        raise ValueError("Expected GeoJSON FeatureCollection")

    finite = np.isfinite(nuclear_um).all(axis=1)
    grid = nuclear_um[finite] / pitch
    homogeneous = np.column_stack((grid, np.ones(len(grid)))) @ matrix.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2, None]
    pixels -= np.array([left, top], dtype=np.float64)
    valid_rows = np.flatnonzero(finite)
    tree = cKDTree(pixels)
    hits = []
    feature_counts = Counter()
    overlap_counts = Counter()
    feature_centers = []
    for feature in expert["features"]:
        label = feature.get("properties", {}).get("classification", {}).get("name", "")
        feature_counts[label] += 1
        if label not in CLASSES:
            continue
        geometry = feature.get("geometry", {})
        if geometry.get("type") != "Polygon" or not geometry.get("coordinates"):
            raise ValueError("Unexpected expert cell geometry")
        rings = geometry["coordinates"]
        outer = np.asarray(rings[0], dtype=np.float64)
        center = outer.mean(axis=0)
        feature_centers.append((center[0], center[1], label))
        radius = float(np.linalg.norm(outer - center, axis=1).max()) + 1e-6
        nearby = tree.query_ball_point(center, radius)
        matched = [i for i in nearby if inside_polygon(*pixels[i], rings)]
        overlap_counts[len(matched)] += 1
        if len(matched) == 1:
            hits.append((int(valid_rows[matched[0]]), label, feature["id"]))

    matches = pd.DataFrame(hits, columns=["source_row", "cell_type", "expert_polygon_id"])
    if matches.empty:
        raise ValueError("No expert nuclei aligned to Space Ranger nuclei")
    duplicated = matches.duplicated("source_row", keep=False)
    ambiguous_cell_hits = int(duplicated.sum())
    matches = matches.loc[~duplicated].copy()
    matches["cell_id"] = cell_ids[matches.source_row.to_numpy()]
    matches = matches.sort_values("source_row").reset_index(drop=True)
    if len(matches) < args.min_expert_matches:
        raise ValueError(f"Only {len(matches)} unambiguous expert matches; inspect registration")

    # The marker-assisted region annotation is keyed by physical 2 um square ID.
    # Only expert-matched cells enter this pilot; no region label is imputed.
    mapping = pd.read_parquet(args.barcode_mappings)
    columns = {name.lower(): name for name in mapping.columns}
    needed = {"square_002um", "cell_id", "in_cell"}
    if not needed.issubset(columns):
        raise ValueError(f"Missing barcode mapping fields: {needed - set(columns)}")
    mapping = mapping[[columns[k] for k in ("square_002um", "cell_id", "in_cell")]]
    mapping.columns = ["square", "cell_id", "in_cell"]
    mapping = mapping.loc[mapping.in_cell, ["square", "cell_id"]].copy()
    original_id_examples = mapping.head(3).to_dict(orient="records")
    # SR barcode tables omit the -1 suffix in some versions. Only normalize
    # documented syntactic differences; coordinates and numeric IDs stay fixed.
    numeric_ids = mapping.cell_id.astype("string").str.extract(
        r"^cellid_(\d+)(?:-1)?$", expand=False)
    if numeric_ids.isna().any():
        raise ValueError("Unsupported assigned Space Ranger cell ID")
    mapping["cell_id"] = "cellid_" + numeric_ids.str.zfill(9) + "-1"
    mapping = mapping.loc[mapping.cell_id.isin(matches.cell_id)].copy()
    if mapping.empty:
        raise ValueError("No expert-matched cell IDs in the barcode mapping")
    def normalize_squares(values):
        parts = values.astype("string").str.extract(
            r"^s_002um_(\d+)_(\d+)(?:-1)?$")
        if parts.isna().any().any():
            raise ValueError("Unsupported physical 2 um square barcode")
        return "s_002um_" + parts[0].str.zfill(5) + "_" + parts[1].str.zfill(5) + "-1"
    mapping["square"] = normalize_squares(mapping.square)
    if mapping.square.duplicated().any():
        raise ValueError("Duplicated 2 um square in Space Ranger mapping")
    regions = pd.read_csv(args.region_csv, sep="\t", header=None,
                          names=["square", "region"], dtype="string")
    regions["square"] = normalize_squares(regions.square)
    if regions.square.duplicated().any():
        raise ValueError("Duplicated 2 um square in region annotation")
    joined = mapping.merge(regions, on="square", how="inner", validate="one_to_one")
    if joined.empty:
        raise ValueError("No physical 2 um square IDs agree across the two versions")
    votes = joined.groupby(["cell_id", "region"], observed=True).size().rename("votes")
    votes = votes.reset_index()
    totals = votes.groupby("cell_id").votes.sum().rename("labeled_bins")
    best = votes.sort_values(["cell_id", "votes", "region"],
                             ascending=[True, False, True]).drop_duplicates("cell_id")
    best = best.join(totals, on="cell_id")
    best["region_purity"] = best.votes / best.labeled_bins
    best = best.loc[(best.labeled_bins >= args.min_region_bins) &
                    (best.region_purity >= args.min_region_purity)]
    labels = matches.merge(best[["cell_id", "region", "labeled_bins", "region_purity"]],
                           on="cell_id", how="left", validate="one_to_one")
    labels.rename(columns={"region": "region_label"}, inplace=True)
    labels.to_csv(output / "labels.csv", index=False)
    (output / "center_ids.txt").write_text("\n".join(labels.cell_id) + "\n")

    rng = np.random.default_rng(42)
    background = rng.choice(len(pixels), min(len(pixels), 15000), replace=False)
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.scatter(pixels[background, 0], pixels[background, 1], s=.1,
                 color="0.7", alpha=.3, label="Space Ranger nuclei")
    colors = {"epithelial cells": "#a65521", "immune cells": "#2166ac",
              "stromal cells": "#4d9221"}
    for label in sorted(CLASSES):
        points = np.asarray([(x, y) for x, y, cls in feature_centers if cls == label])
        axis.scatter(points[:, 0], points[:, 1], s=1, alpha=.7,
                     color=colors[label], label=f"ENACT {label}")
    axis.set(xlim=(0, width), ylim=(height, 0), xlabel="cropped WSI x (px)",
             ylabel="cropped WSI y (px)", title="CRC expert label registration audit")
    axis.legend(markerscale=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "registration_overlay.png", dpi=160)
    plt.close(fig)

    audit = {
        "status": "LABEL_MATCHES_BUILT", "sample_id": metadata["sample_id"],
        "expert_feature_count": len(expert["features"]),
        "expert_classes": dict(feature_counts),
        "expert_polygon_nucleus_overlap_counts": {str(k): v for k, v in overlap_counts.items()},
        "ambiguous_cell_hits_removed": ambiguous_cell_hits,
        "matched_expert_cells": len(matches),
        "matched_classes": matches.cell_type.value_counts().to_dict(),
        "matched_with_region": int(labels.region_label.notna().sum()),
        "matched_region_counts": labels.region_label.value_counts().to_dict(),
        "region_bins_joined": len(joined),
        "region_purity_threshold": args.min_region_purity,
        "region_min_labeled_bins": args.min_region_bins,
        "crop_origin_px": [left, top], "crop_extent_px": [width, height],
        "crop_bounds_px": [left, top, right, bottom],
        "barcode_original_examples": original_id_examples,
        "barcode_normalization": "numeric cell ID and exact 2um row/column; zero padding and optional -1 suffix only",
        "spot_colrow_to_microscope_colrow": matrix.tolist(),
        "nucleus_input": "segmented In_nucleus square geometric centroid",
        "classification": "expert cell polygons; same physical slide, independent segmentation version",
        "region_provenance": "SpaceHack2023 2 um bin labels; H&E plus marker genes, not histology-only truth",
        "sources": {"compartments": str(args.compartments.resolve()),
                    "feature_slice": str(args.feature_slice.resolve()),
                    "expert_zip": str(args.expert_zip.resolve()),
                    "region_csv": str(args.region_csv.resolve()),
                    "barcode_mappings": str(args.barcode_mappings.resolve())},
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)
    data.file.close()


if __name__ == "__main__":
    main()
