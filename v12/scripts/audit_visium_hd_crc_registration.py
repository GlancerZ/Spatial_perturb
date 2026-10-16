#!/usr/bin/env python3
"""Audit the fixed official coordinate transform without using model scores."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import socket
from zipfile import ZipFile


def inside_ring(x, y, ring):
    inside = False
    previous = ring[-1]
    for current in ring:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y) and x < (x2-x1)*(y-y1)/(y2-y1)+x1:
            inside = not inside
        previous = current
    return inside


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compartments", type=Path, required=True)
    p.add_argument("--feature-slice", type=Path, required=True)
    p.add_argument("--expert-zip", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run in a real Slurm compute-node step")
    import anndata as ad
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree
    a.output_dir.mkdir(parents=True, exist_ok=False)
    cells = ad.read_h5ad(a.compartments, backed="r")
    nuclear = np.asarray(cells.obsm["nuclear_spatial_um"], dtype=float)
    finite = np.isfinite(nuclear).all(1)
    ids = np.asarray(cells.obs_names, dtype=str)[finite]
    with h5py.File(a.feature_slice, "r") as f:
        meta = json.loads(f.attrs["metadata_json"])
    matrix = np.asarray(meta["transform_matrices"]["spot_colrow_to_microscope_colrow"])
    grid = nuclear[finite]/float(meta["spot_pitch"])
    homogeneous = np.column_stack((grid, np.ones(len(grid)))) @ matrix.T
    pixels = homogeneous[:, :2]/homogeneous[:, 2, None]
    with ZipFile(a.expert_zip) as z:
        names = [n for n in z.namelist() if n.endswith(".geojson") and "Human_Colon_Cancer" in n and not n.startswith("__MACOSX/")]
        if len(names) != 1:
            raise ValueError("Expected one expert CRC file")
        bounds = re.search(r"-wsi-(\d+)_(\d+)_(\d+)_(\d+)\.geojson$", names[0])
        if bounds is None:
            raise ValueError("Unknown crop bounds")
        left, top, right, bottom = map(int, bounds.groups())
        pixels -= np.asarray([left, top])
        expert = json.loads(z.read(names[0]))
    polygons = []
    for feature in expert["features"]:
        label = feature.get("properties", {}).get("classification", {}).get("name", "")
        if label not in {"epithelial cells", "immune cells", "stromal cells"}:
            continue
        rings = feature["geometry"]["coordinates"]
        outer = np.asarray(rings[0], dtype=float)
        center = outer.mean(0)
        polygons.append((feature["id"], label, rings, center, float(np.linalg.norm(outer-center, axis=1).max())+1e-6))
    shifts = [(0,0), (150,0), (-150,0), (0,150), (0,-150), (500,0), (0,500)]
    rows = []
    zero_hits = []
    for dx, dy in shifts:
        shifted = pixels + np.asarray([dx, dy])
        tree = cKDTree(shifted)
        hits = []
        for polygon_id, label, rings, center, radius in polygons:
            nearby = tree.query_ball_point(center, radius)
            match = [i for i in nearby if inside_ring(*shifted[i], rings[0]) and not any(inside_ring(*shifted[i], hole) for hole in rings[1:])]
            if len(match) == 1:
                i = match[0]
                hits.append((ids[i], polygon_id, label, float(np.linalg.norm(shifted[i]-center))))
        frame = pd.DataFrame(hits, columns=["cell_id", "expert_polygon_id", "cell_type", "distance_to_polygon_vertex_mean_px"])
        frame = frame.loc[~frame.cell_id.duplicated(keep=False)]
        rows.append({"dx_px":dx, "dy_px":dy, "unambiguous_matches":len(frame), "fraction_of_classified_polygons":len(frame)/len(polygons)})
        if dx == 0 and dy == 0:
            zero_hits = frame
        print(rows[-1], flush=True)
    labels = pd.read_csv(a.labels, dtype={"cell_id":str, "expert_polygon_id":str})
    positions = pd.DataFrame(np.asarray(cells.obsm["spatial_um"]),
                             index=cells.obs_names, columns=["x_um", "y_um"])
    coverage = labels.set_index("cell_id").join(positions, validate="one_to_one")
    coverage["tile"] = (np.floor(coverage.x_um/500).astype(int).astype(str)
                        + "_" + np.floor(coverage.y_um/500).astype(int).astype(str))
    region_tile = coverage.dropna(subset=["region_label"]).groupby(
        ["region_label", "tile"]).size().rename("cells").reset_index()
    region_tile.to_csv(a.output_dir/"region_tile_coverage.csv", index=False)
    left_keys = set(zip(labels.cell_id, labels.expert_polygon_id.astype(str), labels.cell_type))
    right_keys = set(zip(zero_hits.cell_id, zero_hits.expert_polygon_id.astype(str), zero_hits.cell_type))
    if left_keys != right_keys:
        raise ValueError("Independent registration audit does not reproduce the exact frozen label matches")
    zero_hits.to_csv(a.output_dir/"matched_distances.csv", index=False)
    control = pd.DataFrame(rows)
    control.to_csv(a.output_dir/"translation_controls.csv", index=False)
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    # Display the three most populated nonoverlapping 800px windows so the
    # registration can be inspected locally, without selecting by model scores.
    expert_centers = np.asarray([q[3] for q in polygons])
    tile = np.floor(expert_centers/800).astype(int)
    tiles, counts = np.unique(tile, axis=0, return_counts=True)
    chosen = tiles[np.argsort(-counts, kind="stable")[:3]]
    tree = cKDTree(pixels)
    for ax, (tx, ty) in zip(axes, chosen):
        origin = np.asarray([tx,ty])*800
        within = ((pixels >= origin) & (pixels < origin+800)).all(1)
        ax.scatter(pixels[within,0],pixels[within,1],s=8,c="#2166ac",label="SR4 nucleus center")
        for _, _, rings, center, _ in polygons:
            if ((center >= origin) & (center < origin+800)).all():
                ring = np.asarray(rings[0])
                ax.plot(ring[:,0],ring[:,1],c="#a65521",lw=.6)
        ax.set(xlim=(origin[0],origin[0]+800),ylim=(origin[1]+800,origin[1]),aspect="equal",xlabel="cropped WSI x (px)",ylabel="cropped WSI y (px)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Fixed official transform: ENACT expert polygons and SR4 nucleus centers")
    fig.tight_layout()
    fig.savefig(a.output_dir/"registration_local_overlay.png",dpi=160)
    plt.close(fig)
    audit={"status":"REGISTRATION_QA_COMPLETE", "slurm_job_id":os.environ["SLURM_JOB_ID"],"hostname":host,
           "classified_polygons":len(polygons),"matched_labels_exactly_reproduced":True,
           "fixed_transform_zero_offset_matches":rows[0]["unambiguous_matches"],
           "max_shifted_control_matches":max(q["unambiguous_matches"] for q in rows[1:]),
           "region_tile_counts":region_tile.groupby("region_label").tile.nunique().to_dict(),
           "translation_controls":rows,"distance_to_vertex_mean_px_quantiles":zero_hits.distance_to_polygon_vertex_mean_px.quantile([0,.5,.9,.99,1]).to_dict(),
           "crop_bounds_xyxy_px":[left,top,right,bottom],
           "scope":"Geometry audit only; no model scores used to choose transform; distances are to vertex means, not polygon centroids."}
    (a.output_dir/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    cells.file.close()


if __name__ == "__main__":
    main()
