#!/usr/bin/env python3
"""Align V12 and official scGPT-spatial embeddings to one frozen patient split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import anndata as ad
import numpy as np
import pandas as pd


REGIONS = ("GSM9451442_R1_2", "GSM9451445_R1_4",
           "GSM9451448_R1_6", "GSM9451458_R2_6")
V12 = ("nucleus", "cytoplasm", "local", "spatial", "fused", "concat",
       "scgpt_total_cls")
MODES = V12 + ("scgpt_spatial_v1",)
PRIMARY = ("local", "spatial", "fused", "concat", "scgpt_total_cls", "scgpt_spatial_v1")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-index", type=Path, required=True)
    p.add_argument("--region-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run in a verified Slurm compute-node step")
    if args.output_dir.exists():
        p.error("Refusing to replace output directory")
    index = pd.read_parquet(args.frozen_index)
    if not index.full_cell_id.is_unique or set(index.split) != {"train", "val", "test"}:
        raise ValueError("Frozen index is invalid")
    loaded: dict[str, dict[str, np.ndarray]] = {name: {} for name in MODES}
    availability: dict[str, dict[str, bool]] = {name: {} for name in MODES}
    source_checksums = {}
    for region in REGIONS:
        v12_path = args.region_root / region / "v12" / "embeddings.h5ad"
        official_path = args.region_root / region / "scgpt_spatial" / "embeddings.npz"
        v12 = ad.read_h5ad(v12_path)
        if "benchmark_id" not in v12.obs or not v12.obs_names.is_unique:
            raise ValueError(f"{region}: V12 has no benchmark IDs or duplicate raw IDs")
        full_ids = v12.obs["benchmark_id"].astype(str).to_numpy()
        if np.any(full_ids == "") or len(set(full_ids)) != len(full_ids):
            raise ValueError(f"{region}: incomplete or duplicate scored IDs")
        with np.load(official_path, allow_pickle=False) as npz:
            raw_ids = np.asarray(npz["cell_ids"], dtype=str)
            official_embedding = np.asarray(npz["embeddings"], dtype=np.float32)
        if not np.array_equal(raw_ids, v12.obs_names.astype(str)) or len(official_embedding) != len(raw_ids):
            raise ValueError(f"{region}: V12 and scGPT-spatial cells or order differ")
        for name in V12:
            key = "X_" + name
            if key not in v12.obsm or name + "_available" not in v12.obs:
                raise ValueError(f"{region}: V12 lacks {name} or its availability")
            matrix = np.asarray(v12.obsm[key], dtype=np.float32)
            flags = v12.obs[name + "_available"].to_numpy().astype(bool)
            if matrix.shape[0] != len(full_ids) or not np.isfinite(matrix).all():
                raise ValueError(f"{region}: invalid V12 {name} embedding")
            for i, cell in enumerate(full_ids):
                if cell in loaded[name]:
                    raise ValueError(f"Duplicate full cell ID: {cell}")
                loaded[name][cell] = matrix[i]
                availability[name][cell] = bool(flags[i])
        if not np.isfinite(official_embedding).all():
            raise ValueError(f"{region}: invalid scGPT-spatial embeddings")
        for i, cell in enumerate(full_ids):
            loaded["scgpt_spatial_v1"][cell] = official_embedding[i]
            availability["scgpt_spatial_v1"][cell] = True
        source_checksums[region] = {"v12": sha256(v12_path),
                                    "scgpt_spatial_v1": sha256(official_path)}
        print(f"[align] {region}: {len(full_ids)} scored cells", flush=True)
    ids = index.full_cell_id.astype(str).tolist()
    known = set(loaded["scgpt_spatial_v1"])
    if not known.issubset(ids):
        raise ValueError("Model produced IDs outside frozen benchmark index")
    missing = sorted(set(ids) - known)
    # The initial all-view intersection failed its coverage gate before any
    # test scoring: some cells contain no measured nuclear/cytoplasmic RNA.
    # Preserve ALL frozen cells for the primary comparison. Evaluate each
    # compartment-only view with fused and official baselines on its exact
    # available subset; do not relax the gate and silently shrink the main test.
    common = [cell for cell in ids if all(availability[name].get(cell, False) for name in PRIMARY)]
    if common != ids or missing:
        raise ValueError("Primary comparison must cover every frozen cell")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def write_cohort(cohort_ids, names, root, cohort_name):
        selected = index.set_index("full_cell_id").loc[cohort_ids].reset_index()
        if (set(selected.split) != {"train", "val", "test"} or
                set(selected.patient) != set(index.patient)):
            raise ValueError(f"{cohort_name}: a split or patient is absent")
        root.mkdir(parents=True, exist_ok=True)
        index_path = root / "common_index.parquet"
        selected.to_parquet(index_path, index=False)
        members = set(cohort_ids)
        coverage = {"all": len(cohort_ids)/len(ids), "by_split": {}, "by_patient": {}}
        for column, key in (("split", "by_split"), ("patient", "by_patient")):
            for value, frame in index.groupby(column, sort=True):
                coverage[key][str(value)] = sum(cell in members for cell in frame.full_cell_id.astype(str))/len(frame)
        for name in names:
            target = root / "embeddings" / name
            target.mkdir(parents=True, exist_ok=False)
            matrix = np.stack([loaded[name][cell] for cell in cohort_ids]).astype(np.float32)
            if not np.isfinite(matrix).all() or matrix.shape[0] != len(cohort_ids):
                raise ValueError(f"Invalid assembled matrix: {cohort_name}/{name}")
            np.save(target / "embeddings.npy", matrix, allow_pickle=False)
            (target / "cell_ids.txt").write_text("\n".join(cohort_ids) + "\n")
            provenance = {
                "model_name": "official_scGPT_spatial_V1" if name == "scgpt_spatial_v1" else "V12_step_10234",
                "readout": name, "dataset": "GSE316387", "cells": len(cohort_ids),
                "cohort": cohort_name,
                "input_context": "whole-cell counts" if name in ("scgpt_spatial_v1", "scgpt_total_cls") else "V12 compartment/neighbor readout: " + name,
                "source_sha256": source_checksums,
                "frozen_index_sha256": sha256(args.frozen_index),
                "common_index_sha256": sha256(index_path),
            }
            (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        return {"cells": len(cohort_ids), "split_cells": selected.split.value_counts().to_dict(),
                "coverage": coverage, "readouts": list(names)}

    primary = write_cohort(common, PRIMARY, args.output_dir, "primary_all_frozen_cells")
    supplemental = {}
    for view in ("nucleus", "cytoplasm"):
        available_ids = [cell for cell in ids if availability[view].get(cell, False)]
        supplemental[view] = write_cohort(
            available_ids, (view, "fused", "scgpt_spatial_v1"),
            args.output_dir / "supplementary" / view, view + "_available_cells")
    flags = index[["full_cell_id", "patient", "split"]].copy()
    for name in MODES:
        flags[name + "_available"] = [availability[name].get(cell, False) for cell in ids]
    flags.to_csv(args.output_dir / "availability.csv", index=False)
    report = {
        "status": "COMMON_SAME_CELL_COHORT_COMPLETE",
        "frozen_cells": len(ids), "common_cells": len(common),
        "missing_model_cells": len(missing),
        "primary": primary,
        "supplementary": supplemental,
        "protocol_revision_before_test_scoring": "Preserve complete frozen main cohort; match each compartment-only readout to fused and official baselines on exactly its available cells",
        "unavailable_by_readout": {
            name: sum(not availability[name].get(cell, False) for cell in ids)
            for name in MODES},
        "split_cells": primary["split_cells"],
        "source_sha256": source_checksums,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
