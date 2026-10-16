"""Export the fixed LungPF cohort for frozen foundation-model annotation.

Run on a compute node. Counts are the accepted QV>=20, control-filtered
transcripts of the public *corrected* LungPF release, summed over the existing
nuclear and cytoplasmic arrays. They are not uncorrected imaging raw molecules.

The exported cell_index.parquet fixes cell order for every model. Each sample
H5AD keeps all cells, including those without a pathology-region label.
Nothing in the source ``prepared`` directory is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_INPUT = Path("data/benchmarking_data/lungpf/prepared")
DEFAULT_OUTPUT = Path("outputs/foundation_annotation_lungpf/input")
INDEX_COLUMNS = (
    "full_cell_id", "sample", "patient", "split", "final_CT", "region_label",
    "x_centroid", "y_centroid",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_new_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_source(input_root: Path) -> tuple[dict, list[tuple[str, pd.DataFrame]]]:
    protocol = json.loads((input_root / "protocol.json").read_text())
    samples = protocol["samples"]
    if len(samples) != 16 or len(set(samples)) != 16:
        raise ValueError("Expected 16 distinct LungPF samples")
    if protocol["cell_type_label"] != "final_CT" or protocol["qv_min"] != 20:
        raise ValueError("Unexpected LungPF annotation/count protocol")
    cohort: list[tuple[str, pd.DataFrame]] = []
    all_ids: set[str] = set()
    for sample in samples:
        source = input_root / sample
        obs = pd.read_parquet(source / "obs.parquet")
        missing = set(INDEX_COLUMNS) - set(obs.columns)
        if missing:
            raise ValueError(f"{sample}: missing columns {sorted(missing)}")
        obs = obs.loc[:, list(INDEX_COLUMNS)].copy()
        if obs.empty or obs.full_cell_id.isna().any() or not obs.full_cell_id.is_unique:
            raise ValueError(f"{sample}: empty or invalid cell IDs")
        if not obs["sample"].eq(sample).all():
            raise ValueError(f"{sample}: sample name mismatch")
        if obs.patient.nunique() != 1 or obs.split.nunique() != 1:
            raise ValueError(f"{sample}: mixed donor or split")
        patient = str(obs.patient.iloc[0])
        if protocol["donor_split"].get(patient) != obs.split.iloc[0]:
            raise ValueError(f"{sample}: donor split differs from protocol")
        if not np.isfinite(obs[["x_centroid", "y_centroid"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{sample}: nonfinite coordinates")
        new_ids = set(obs.full_cell_id.astype(str))
        if new_ids & all_ids:
            raise ValueError(f"{sample}: duplicate cell IDs across samples")
        all_ids.update(new_ids)
        qa = json.loads((source / "qa.json").read_text())
        if int(qa["cells"]) != len(obs) or int(qa["genes"]) != 343:
            raise ValueError(f"{sample}: source QA cell/gene count mismatch")
        cohort.append((sample, obs))
    if len(set(protocol["donor_split"].values()) - {"train", "val", "test"}):
        raise ValueError("Unsupported split name")
    return protocol, cohort


def prepare_index(input_root: Path, output_root: Path, cohort: list[tuple[str, pd.DataFrame]],
                  protocol: dict) -> dict:
    index = pd.concat([obs for _, obs in cohort], ignore_index=True)
    if index.patient.nunique() != 16 or len(index) != 455379:
        raise ValueError(f"Unexpected LungPF cohort: {len(index)} cells, {index.patient.nunique()} donors")
    expected = {"train": 305598, "val": 85170, "test": 64611}
    observed = index.loc[index.final_CT.notna()].groupby("split").size().to_dict()
    if observed != expected:
        raise ValueError(f"Cell-type label cohort differs from published protocol: {observed}")
    manifest = {
        "schema": "foundation_annotation_lungpf_v1",
        "source_protocol_sha256": digest(input_root / "protocol.json"),
        "samples": protocol["samples"],
        "donor_split": protocol["donor_split"],
        "index_rows": len(index),
        "cell_id_sha256": hashlib.sha256(
            "\n".join(index.full_cell_id.astype(str)).encode()).hexdigest(),
        "labeled_cell_type_by_split": observed,
        "C_grid": protocol["C_grid"],
        "counts_definition": "sum of QV>=20, control-filtered nuclear and cytoplasmic counts from corrected LungPF transcript release",
        "index_columns": list(INDEX_COLUMNS),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "cell_index.parquet"
    manifest_path = output_root / "manifest.json"
    if index_path.exists() or manifest_path.exists():
        if not (index_path.exists() and manifest_path.exists()):
            raise FileExistsError("Only one of index/manifest exists; refusing to overwrite")
        existing = json.loads(manifest_path.read_text())
        if existing != manifest or not pd.read_parquet(index_path).equals(index):
            raise ValueError("Existing export index differs from current source; refusing overwrite")
    else:
        index.to_parquet(index_path, index=False)
        write_new_json(manifest_path, manifest)
    return manifest


def export_sample(input_root: Path, output_root: Path, sample: str, obs: pd.DataFrame,
                  protocol_hash: str, *, chunk_rows: int, skip_existing: bool) -> None:
    destination = output_root / sample
    destination.mkdir(parents=True, exist_ok=True)
    h5ad_path = destination / "counts.h5ad"
    manifest_path = destination / "manifest.json"
    if h5ad_path.exists() or manifest_path.exists():
        if skip_existing and h5ad_path.exists() and manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if (previous.get("sample") == sample and previous.get("cells") == len(obs)
                    and previous.get("protocol_sha256") == protocol_hash
                    and previous.get("cell_id_sha256") == hashlib.sha256(
                        "\n".join(obs.full_cell_id.astype(str)).encode()).hexdigest()
                    and previous.get("h5ad_sha256") == digest(h5ad_path)):
                print(f"[skip verified] {sample}", flush=True)
                return
        raise FileExistsError(f"{sample}: existing export; refusing overwrite")

    source = input_root / sample
    genes = np.load(source / "genes.npy", allow_pickle=False)
    nuclear = np.load(source / "nuclear.npy", mmap_mode="r", allow_pickle=False)
    cytoplasm = np.load(source / "cytoplasmic.npy", mmap_mode="r", allow_pickle=False)
    if nuclear.shape != cytoplasm.shape or nuclear.shape != (len(obs), len(genes)):
        raise ValueError(f"{sample}: source array shape mismatch")
    if len(genes) != 343 or len(set(map(str, genes))) != len(genes):
        raise ValueError(f"{sample}: invalid gene panel")
    if not (np.issubdtype(nuclear.dtype, np.integer)
            and np.issubdtype(cytoplasm.dtype, np.integer)):
        raise ValueError(f"{sample}: count arrays must be integer")

    blocks = []
    total = 0
    for start in range(0, len(obs), chunk_rows):
        end = min(start + chunk_rows, len(obs))
        values = np.asarray(nuclear[start:end], dtype=np.int64) + np.asarray(
            cytoplasm[start:end], dtype=np.int64)
        if values.min(initial=0) < 0 or values.max(initial=0) > np.iinfo(np.int32).max:
            raise ValueError(f"{sample}: invalid whole-cell count range")
        total += int(values.sum(dtype=np.int64))
        blocks.append(sp.csr_matrix(values.astype(np.int32)))
    qa = json.loads((source / "qa.json").read_text())
    if total != int(qa["accepted_transcripts"]):
        raise ValueError(f"{sample}: whole-cell total differs from source QA")
    matrix = sp.vstack(blocks, format="csr")
    # Keep labels and donor splits only in the protected evaluation index.
    # Model adapters receive counts, coordinates, and stable IDs, never truth.
    source_obs = obs[["full_cell_id", "sample"]].set_index("full_cell_id", drop=False).copy()
    source_obs.index.name = "cell_id"
    var = pd.DataFrame(index=pd.Index(genes.astype(str), name="gene_name"))
    data = ad.AnnData(X=matrix, obs=source_obs, var=var)
    data.obsm["spatial"] = obs[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    data.uns["source_protocol_sha256"] = protocol_hash
    data.uns["counts_definition"] = "accepted QV>=20 corrected-release transcripts; nuclear + cytoplasmic"
    tmp_path = destination / "counts.partial.h5ad"
    if tmp_path.exists():
        raise FileExistsError(f"{sample}: partial H5AD exists; inspect before retry")
    data.write_h5ad(tmp_path, compression="gzip")
    try:
        os.link(tmp_path, h5ad_path)  # atomic, refuses an existing final target
    finally:
        tmp_path.unlink(missing_ok=True)
    write_new_json(manifest_path, {
        "sample": sample, "cells": len(obs), "genes": len(genes),
        "accepted_total_counts": total, "protocol_sha256": protocol_hash,
        "cell_id_sha256": hashlib.sha256(
            "\n".join(obs.full_cell_id.astype(str)).encode()).hexdigest(),
        "h5ad_sha256": digest(h5ad_path),
        "source_nuclear": str(source / "nuclear.npy"),
        "source_cytoplasmic": str(source / "cytoplasmic.npy"),
    })
    print(f"[exported] {sample}: {len(obs)} cells, {len(genes)} genes", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", help="Export only this sample after preparing the shared index")
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse only SHA-verified exports")
    parser.add_argument("--chunk-rows", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive")
    protocol, cohort = load_source(args.input_root)
    manifest = prepare_index(args.input_root, args.output_root, cohort, protocol)
    if args.index_only:
        return
    selected = [(sample, obs) for sample, obs in cohort if args.sample in (None, sample)]
    if not selected:
        parser.error(f"Sample absent from LungPF protocol: {args.sample}")
    for sample, obs in selected:
        export_sample(args.input_root, args.output_root, sample, obs,
                      manifest["source_protocol_sha256"], chunk_rows=args.chunk_rows,
                      skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
