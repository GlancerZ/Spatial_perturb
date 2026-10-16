"""Check whether epithelial-marker RNAs in nearby T cells face EPCAM-high cells.

The analysis reads raw molecule coordinates and existing segmentation-assigned
cell ids. It is a directional diagnostic, not a reassignment algorithm.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from diagnose_xenium_spillover import T_SUPPORT, gene_counts


def run(h5ad: Path, archive: Path) -> dict:
    data = ad.read_h5ad(h5ad)
    genes = data.var_names
    cd3e = gene_counts(data.X, genes, "CD3E")
    support = sum((gene_counts(data.X, genes, gene) for gene in T_SUPPORT),
                  np.zeros(data.n_obs, dtype=np.int64))
    epcam = gene_counts(data.X, genes, "EPCAM")
    depth = np.asarray(data.X.sum(axis=1)).ravel()
    t_mask = (cd3e >= 1) & (support >= 1) & (depth >= 20)
    source_mask = (epcam >= 3) & (depth >= 50) & ~t_mask
    xy = np.asarray(data.obsm["spatial"][:, :2], dtype=float)
    t_index = np.flatnonzero(t_mask)
    distances, source_match = cKDTree(xy[source_mask]).query(xy[t_mask], k=1, workers=-1)
    near_t = t_index[distances <= 15.0]
    near_source = np.flatnonzero(source_mask)[source_match[distances <= 15.0]]
    target_xy = xy[near_t]
    source_vec = xy[near_source] - target_xy
    length = np.linalg.norm(source_vec, axis=1)
    source_vec /= np.maximum(length[:, None], 1e-6)
    area = np.asarray(data.obs["cell_area"], dtype=float)[near_t]
    radius = np.sqrt(np.maximum(area, 1e-6) / np.pi)
    cell_index = pd.Index(data.obs_names[near_t].astype(str))
    readouts = [gene for gene in ("EPCAM", "CEACAM6", "MUC1", "SOX2", "CD3E", "LCK")
                if gene in genes]
    stats = defaultdict(lambda: {"molecules": 0, "cells": set(), "projection_sum_um": 0.0,
                                  "projection_sum_radius": 0.0, "facing": 0, "opposite": 0,
                                  "distance_to_target_sum_um": 0.0})
    per_cell = {gene: {"n": np.zeros(len(near_t), dtype=np.int64),
                       "projection_um": np.zeros(len(near_t), dtype=float),
                       "distance_um": np.zeros(len(near_t), dtype=float)}
                for gene in readouts}
    rows_seen = selected = 0
    with tempfile.TemporaryDirectory(prefix="xenium-geometry-") as temp_dir:
        parquet_path = Path(temp_dir) / "transcripts.parquet"
        with zipfile.ZipFile(archive) as z:
            matching = [name for name in z.namelist() if Path(name).name == "transcripts.parquet"]
            if len(matching) != 1:
                raise ValueError("Expected one transcripts.parquet member")
            with z.open(matching[0]) as src, parquet_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
        parquet = pq.ParquetFile(parquet_path)
        columns = ["cell_id", "feature_name", "qv", "overlaps_nucleus", "x_location", "y_location"]
        if not set(columns).issubset(parquet.schema.names):
            raise ValueError(f"Missing raw transcript columns: {set(columns) - set(parquet.schema.names)}; got {parquet.schema.names}")
        for batch in parquet.iter_batches(columns=columns, batch_size=500_000):
            frame = batch.to_pandas()
            rows_seen += len(frame)
            feature = frame["feature_name"].astype(str).to_numpy()
            keep = np.isin(feature, readouts)
            keep &= pd.to_numeric(frame["qv"], errors="coerce").to_numpy() >= 20
            keep &= frame["overlaps_nucleus"].astype(str).str.lower().isin(["false", "0", "0.0"]).to_numpy()
            if not keep.any():
                continue
            frame = frame.loc[keep]
            t_match = cell_index.get_indexer(frame["cell_id"].astype(str))
            assigned = t_match >= 0
            if not assigned.any():
                continue
            frame = frame.iloc[np.flatnonzero(assigned)]
            t_match = t_match[assigned]
            selected += len(frame)
            coords = frame[["x_location", "y_location"]].to_numpy(dtype=float)
            delta = coords - target_xy[t_match]
            projection = np.einsum("ij,ij->i", delta, source_vec[t_match])
            normalized = projection / radius[t_match]
            separation = np.linalg.norm(delta, axis=1)
            feature = frame["feature_name"].astype(str).to_numpy()
            for gene in readouts:
                mask = feature == gene
                if not mask.any():
                    continue
                current = stats[gene]
                current["molecules"] += int(mask.sum())
                current["cells"].update(t_match[mask].tolist())
                current["projection_sum_um"] += float(projection[mask].sum())
                current["projection_sum_radius"] += float(normalized[mask].sum())
                current["facing"] += int((normalized[mask] >= 0.5).sum())
                current["opposite"] += int((normalized[mask] <= -0.5).sum())
                current["distance_to_target_sum_um"] += float(separation[mask].sum())
                np.add.at(per_cell[gene]["n"], t_match[mask], 1)
                np.add.at(per_cell[gene]["projection_um"], t_match[mask], projection[mask])
                np.add.at(per_cell[gene]["distance_um"], t_match[mask], separation[mask])
    summary = {}
    for gene, record in stats.items():
        count = record["molecules"]
        summary[gene] = {
            "cytoplasmic_molecules": count,
            "cells_with_molecules": len(record["cells"]),
            "mean_projection_toward_source_um": record["projection_sum_um"] / count,
            "mean_projection_in_equivalent_cell_radii": record["projection_sum_radius"] / count,
            "facing_outer_half_fraction": record["facing"] / count,
            "opposite_outer_half_fraction": record["opposite"] / count,
            "mean_distance_to_target_centroid_um": record["distance_to_target_sum_um"] / count,
        }
        if gene != "CD3E" and "CD3E" in per_cell:
            paired = (per_cell[gene]["n"] > 0) & (per_cell["CD3E"]["n"] > 0)
            if paired.any():
                gene_proj = per_cell[gene]["projection_um"][paired] / per_cell[gene]["n"][paired]
                cd3e_proj = per_cell["CD3E"]["projection_um"][paired] / per_cell["CD3E"]["n"][paired]
                gene_radial = per_cell[gene]["distance_um"][paired] / per_cell[gene]["n"][paired]
                cd3e_radial = per_cell["CD3E"]["distance_um"][paired] / per_cell["CD3E"]["n"][paired]
                summary[gene]["paired_with_CD3E"] = {
                    "cells": int(paired.sum()),
                    "median_projection_difference_um": float(np.median(gene_proj - cd3e_proj)),
                    "mean_projection_difference_um": float(np.mean(gene_proj - cd3e_proj)),
                    "fraction_projection_more_source_facing": float(np.mean(gene_proj > cd3e_proj)),
                    "median_radial_distance_difference_um": float(np.median(gene_radial - cd3e_radial)),
                    "fraction_more_peripheral_than_CD3E": float(np.mean(gene_radial > cd3e_radial)),
                }
    return {"sample": h5ad.stem, "archive": str(archive.resolve()),
            "near_T_cells": len(near_t), "raw_rows_seen": rows_seen,
            "cytoplasmic_marker_molecules_selected": selected,
            "source_distance_um": "centroid <=15", "readouts": summary,
            "interpretation_limit": "Directional enrichment supports source-associated boundary contamination but cannot distinguish segmentation spillover from overlapping cells or true expression alone."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("transcript_zip", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    if args.output_json.exists():
        raise FileExistsError(args.output_json)
    result = run(args.input_h5ad, args.transcript_zip)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"sample": result["sample"], "near_T_cells": result["near_T_cells"],
                      "raw_rows_seen": result["raw_rows_seen"], "readouts": result["readouts"]}, indent=2))


if __name__ == "__main__":
    main()
