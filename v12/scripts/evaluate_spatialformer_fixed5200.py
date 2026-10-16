"""Re-probe cached official SpatialFormer embeddings on the V7 validation cohort.

The reference table defines the exact cell order. Only its 13 train/validation
sample directories are opened; historical test-donor embeddings are never read.
Run on a compute node with CUDA, using the same probe solver as the V7 evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch

from evaluate_subcellular_v2 import (
    CANONICAL_COHORT, atomic_array, atomic_json, file_sha256, probe_embedding,
)

COLUMNS = ["full_cell_id", "sample", "patient", "split", "final_CT", "region_label"]
EXPECTED_COUNTS = {"cell_type": (4000, 1200), "region": (781, 330)}
FIXED_C = {"cell_type": 0.1, "region": 0.01}
OFFICIAL_COMMIT = "f02e7776e8a99c9abdc5fe5572a1a20369faf159"
OFFICIAL_MD5 = "fb2c33044244bf21a962e106951769f5"
CANONICAL_LABELS_SPLIT = "91d68cb16d322d5c46ae78a3804b1fd4a8f4a6ffa45c3b0644cb72da96db212e"


def cohort_digest(obs):
    return hashlib.sha256("\n".join(obs.full_cell_id.astype(str)).encode()).hexdigest()


def run(args):
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires a CUDA compute-node allocation")
    if (args.output / "metrics.json").exists():
        raise FileExistsError(f"Refusing to replace existing metrics: {args.output}")

    protocol = json.loads(args.protocol.read_text())
    reference = pd.read_parquet(args.reference, columns=COLUMNS).reset_index(drop=True)
    if (len(reference) != 5200 or not reference.full_cell_id.is_unique
            or set(reference["split"]) != {"train", "val"}
            or cohort_digest(reference) != CANONICAL_COHORT):
        raise ValueError("Reference must be the exact canonical 5200-cell train/val cohort")
    labels_split_digest = hashlib.sha256(reference[
        ["full_cell_id", "patient", "split", "final_CT", "region_label"]
    ].fillna("").astype(str).to_csv(index=False).encode()).hexdigest()
    if labels_split_digest != CANONICAL_LABELS_SPLIT:
        raise ValueError("Reference labels/splits differ from the canonical V7 cohort")
    if (reference.groupby("patient").size().ne(400).any()
            or reference.groupby("patient")["split"].nunique().ne(1).any()
            or reference.loc[reference["split"].eq("train"), "patient"].nunique() != 10
            or reference.loc[reference["split"].eq("val"), "patient"].nunique() != 3):
        raise ValueError("Reference donor counts or splits differ from the fixed protocol")

    # Resolve every sample and split before opening any historical sample file.
    samples = list(reference["sample"].unique())
    if len(samples) != 13:
        raise ValueError("Expected 13 distinct train/validation samples")
    for sample in samples:
        if not isinstance(sample, str) or Path(sample).name != sample:
            raise ValueError(f"Invalid sample name: {sample!r}")
        selected = reference.loc[reference["sample"].eq(sample)]
        donors = [donor for donor in protocol["donor_split"] if sample.startswith(donor)]
        if sample not in protocol["samples"] or len(donors) != 1:
            raise ValueError(f"Unknown or ambiguous sample: {sample}")
        donor = donors[0]
        split = protocol["donor_split"][donor]
        if (split not in ("train", "val") or not selected.patient.eq(donor).all()
                or not selected["split"].eq(split).all()):
            raise ValueError(f"Forbidden test donor or donor/split mismatch: {sample}")

    provenance_path = args.historical / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("baseline_name") != "SpatialFormer single71"
            or provenance.get("official_commit") != OFFICIAL_COMMIT
            or provenance.get("checkpoint_md5") != OFFICIAL_MD5):
        raise ValueError("Historical cache is not the verified official single71 baseline")

    features = np.empty((len(reference), 512), dtype=np.float32)
    source_samples = []
    for sample in samples:
        directory = args.historical / "embeddings" / sample
        cells_path = directory / "cells.parquet"
        embeddings_path = directory / "spatialformer.npy"
        metadata_path = directory / "spatialformer.json"
        cached_obs = pd.read_parquet(cells_path, columns=COLUMNS)
        if not cached_obs.full_cell_id.is_unique or not cached_obs["sample"].eq(sample).all():
            raise ValueError(f"Duplicate IDs or incorrect sample in cached rows: {sample}")
        positions = np.flatnonzero(reference["sample"].eq(sample).to_numpy())
        chosen = reference.iloc[positions].reset_index(drop=True)
        if (not cached_obs.patient.eq(chosen.patient.iloc[0]).all()
                or not cached_obs["split"].eq(chosen["split"].iloc[0]).all()):
            raise ValueError(f"Cached donor/split mismatch: {sample}")
        lookup = pd.Index(cached_obs.full_cell_id)
        indices = lookup.get_indexer(chosen.full_cell_id)
        if (indices < 0).any():
            raise ValueError(f"Reference cells missing from historical cache: {sample}")
        matched = cached_obs.iloc[indices].reset_index(drop=True)
        # StringDtype normalizes categorical/object storage while preserving nulls.
        pd.testing.assert_frame_equal(
            chosen[COLUMNS].astype("string"), matched[COLUMNS].astype("string"),
            check_dtype=True, obj=f"reference/cache IDs, labels and splits for {sample}")
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("pooling") != "official final-layer CLS"
                or metadata.get("dimensions") != 512 or metadata.get("frozen") is not True
                or metadata.get("cells") != len(cached_obs)):
            raise ValueError(f"Unverified cached SpatialFormer extraction: {sample}")
        cached_features = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
        if cached_features.shape != (len(cached_obs), 512):
            raise ValueError(f"Cached embedding/observation shape mismatch: {sample}")
        block = np.asarray(cached_features[indices], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"Nonfinite selected embeddings: {sample}")
        features[positions] = block
        info = embeddings_path.stat()
        source_samples.append(dict(sample=sample, selected_cells=len(chosen),
            embedding_path=str(embeddings_path.resolve()), embedding_file_bytes=info.st_size,
            embedding_mtime_ns=info.st_mtime_ns,
            selected_embedding_sha256=hashlib.sha256(block.tobytes()).hexdigest(),
            cells_sha256=file_sha256(cells_path), metadata_sha256=file_sha256(metadata_path),
            extraction_metadata=metadata))
        del cached_features
        print(f"[cached-spatialformer] {sample}: aligned {len(chosen)} cells", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    atomic_array(args.output / "embeddings_cls.npy", features)
    reference.to_parquet(args.output / "cells.parquet", index=False)
    results = probe_embedding(reference, features, "cls", args.output)
    for row in results:
        task = row["task"]
        if (row["train_cells"], row["val_cells"]) != EXPECTED_COUNTS[task]:
            raise ValueError(f"Unexpected {task} labeled-cell counts")
        fixed = [entry for entry in row["grid"] if entry["C"] == FIXED_C[task]]
        if len(fixed) != 1 or fixed[0]["solver"].get("converged_gradient") is not True:
            raise ValueError(f"Invalid or unconverged {task} fixed-C probe")
        row["fixed_C_result"] = fixed[0]
    report = dict(model="Official SpatialFormer single71", baseline_version=1,
        cells=len(reference), cohort_sha256=CANONICAL_COHORT, canonical_cohort=True,
        cohort_labels_split_sha256=labels_split_digest,
        test_donors_used=False, source_commit=OFFICIAL_COMMIT, weights_md5=OFFICIAL_MD5,
        pooling="official final-layer CLS (512D; unnormalized)",
        input="Raw nuclear + cytoplasmic counts; official single-cell gene tokenizer",
        context="Individual cells; official Lung/Xenium and Healthy/Disease condition tokens",
        fixed_C=FIXED_C, results=results,
        train_donors=sorted(reference.loc[reference["split"].eq("train"), "patient"].unique()),
        val_donors=sorted(reference.loc[reference["split"].eq("val"), "patient"].unique()),
        reference_path=str(args.reference.resolve()), reference_sha256=file_sha256(args.reference),
        protocol_path=str(args.protocol.resolve()), protocol_sha256=file_sha256(args.protocol),
        historical_provenance_path=str(provenance_path.resolve()),
        historical_provenance_sha256=file_sha256(provenance_path),
        source_samples=source_samples, source_sha256={
            name: file_sha256(ROOT / name) for name in (
                "scripts/evaluate_spatialformer_fixed5200.py",
                "scripts/evaluate_subcellular_v2.py", "scripts/probe_lungpf_embeddings.py")},
        exact_ids_labels_splits_verified=True, elapsed_seconds=time.monotonic() - started,
        interpretation="Frozen cached official encoder; same V7 train-only scaling and balanced "
            "linear probe. Plot fixed-C validation scores, not selected-best-C or historical test scores. "
            "Baseline pretraining overlap is not fully excluded.")
    atomic_json(args.output / "metrics.json", report)
    print(json.dumps({row["task"]: row["fixed_C_result"] for row in results}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path,
                        default=ROOT / "data/benchmarking_data/lungpf/prepared/protocol.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
