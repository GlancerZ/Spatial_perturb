#!/usr/bin/env python3
"""Rebuild Xenium nuclear/cytoplasmic matrices from official molecules.

One GEO region per invocation; run in a Slurm compute step. The retained
QV>=20 assigned transcripts must reconstruct the frozen whole-cell matrix
exactly. Any discrepancy is a hard error rather than a corrected count.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transcripts-gz", type=Path, required=True)
    p.add_argument("--context-h5ad", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scratch-dir", type=Path, required=True)
    p.add_argument("--batch-rows", type=int, default=500_000)
    args = p.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run in a verified Slurm compute-node step")
    if args.batch_rows < 1 or args.output_dir.exists():
        p.error("Use a positive batch size and a new output directory")

    import anndata as ad
    import numpy as np
    import pyarrow.parquet as pq
    from scipy import sparse
    from spatial_pretrain.data.xenium_compartments import REQUIRED, aggregate_batches

    ref = ad.read_h5ad(args.context_h5ad)
    if not ref.obs_names.is_unique or not ref.var_names.is_unique:
        raise ValueError("Reference cell and gene names must be unique")
    if "spatial" not in ref.obsm or "benchmark_id" not in ref.obs:
        raise ValueError("Expected frozen context coordinates and benchmark IDs")
    coords = np.asarray(ref.obsm["spatial"], dtype=np.float32)
    if coords.shape != (ref.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("Invalid micron coordinates")
    reference_counts = sparse.csr_matrix(ref.X)
    if (not np.isfinite(reference_counts.data).all() or
            (reference_counts.data < 0).any() or
            not np.array_equal(reference_counts.data, np.rint(reference_counts.data))):
        raise ValueError("Reference X is not raw nonnegative integer counts")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    # On the shared filesystem, an open Parquet footer handle can briefly
    # create a hidden delayed-unlink file during TemporaryDirectory cleanup.
    # Release the reader before cleanup; a transient rmdir race must not mask
    # successful molecule aggregation or its subsequent count-conservation QC.
    with tempfile.TemporaryDirectory(prefix="gse316387-molecules-", dir=args.scratch_dir,
                                     ignore_cleanup_errors=True) as tmp:
        parquet_path = Path(tmp) / "transcripts.parquet"
        with gzip.open(args.transcripts_gz, "rb") as source, parquet_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 << 20)
        parquet = pq.ParquetFile(parquet_path)
        missing = set(REQUIRED) - set(parquet.schema.names)
        if missing:
            raise ValueError(f"GEO transcript parquet lacks fields: {sorted(missing)}")
        batches = (batch.to_pandas() for batch in parquet.iter_batches(
            batch_size=args.batch_rows, columns=REQUIRED))
        nuclear, cyto, stats = aggregate_batches(
            batches, ref.obs_names, ref.var_names, qv_min=20.0)
        parquet_rows = parquet.metadata.num_rows
        batches.close()
        del batches, parquet
        gc.collect()
    if stats["rows_seen"] != parquet_rows or not stats["retained"]:
        raise ValueError("Transcript table was incomplete or no molecules were retained")
    total = (nuclear + cyto).tocsr()
    difference = total - reference_counts
    difference.eliminate_zeros()
    stats["reference_mismatched_entries"] = int(difference.nnz)
    stats["reference_absolute_count_difference"] = float(abs(difference).sum())
    if difference.nnz:
        raise ValueError(f"N+C does not reconstruct frozen GEO whole-cell counts: {stats}")
    ids = ref.obs["benchmark_id"].astype(str)
    scored = ids.ne("")
    if not scored.any() or not ids[scored].is_unique:
        raise ValueError("Missing or duplicated scored benchmark IDs")
    result = ad.AnnData(total, obs=ref.obs.copy(), var=ref.var.copy())
    result.layers["nuclear_counts"] = nuclear
    result.layers["cytoplasmic_counts"] = cyto
    result.obsm["spatial_um"] = coords
    result.obs["sample_id"] = args.context_h5ad.stem
    provenance = {
        "method": "direct_molecule_aggregation",
        "definition": "assigned Xenium cell_id; overlaps_nucleus True=nuclear, False=cytoplasmic; qv>=20",
        "transcripts_gz": str(args.transcripts_gz.resolve()),
        "transcripts_gz_sha256": sha256(args.transcripts_gz),
        "reference_h5ad": str(args.context_h5ad.resolve()),
        "reference_h5ad_sha256": sha256(args.context_h5ad),
        "whole_cell_count_conservation": True,
    }
    result.uns["compartment_provenance"] = provenance
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(tempfile.mkdtemp(
        prefix=args.output_dir.name + ".incomplete-", dir=args.output_dir.parent))
    result.write_h5ad(staged_dir / "compartments.h5ad", compression="lzf")
    center_ids = ref.obs_names[scored.to_numpy()].astype(str)
    (staged_dir / "center_ids.txt").write_text("\n".join(center_ids) + "\n", encoding="utf-8")
    audit = {
        "status": "EXACT_COUNT_RECONSTRUCTION_COMPLETE",
        "sample": args.context_h5ad.stem,
        "cells": ref.n_obs, "genes": ref.n_vars,
        "scored_cells": int(scored.sum()),
        "parquet_rows": parquet_rows,
        "stats": stats,
        "provenance": provenance,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "hostname": host,
    }
    (staged_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    staged_dir.rename(args.output_dir)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
