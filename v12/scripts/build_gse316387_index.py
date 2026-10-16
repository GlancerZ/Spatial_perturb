#!/usr/bin/env python3
"""Freeze one patient-disjoint benchmark index from GSE316387 metadata.

Run on a compute node after inspect_gse316387.R. Explicit column arguments
avoid guessing the authors' metadata schema. Sampling ignores type labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--patient-column", required=True)
    parser.add_argument("--sample-column", required=True)
    parser.add_argument("--type-column", required=True)
    parser.add_argument("--group-column", required=True)
    parser.add_argument("--genotype-column", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-cells-per-patient", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_cells_per_patient < 1 or args.output_dir.exists():
        parser.error("Require positive cell cap and a new output directory")
    frame = pd.read_csv(args.metadata_csv, compression="gzip", low_memory=False,
                        dtype=str, keep_default_na=False)
    mapping = {"patient": args.patient_column, "sample": args.sample_column,
               "cell_type": args.type_column, "cell_group": args.group_column,
               "genotype": args.genotype_column}
    missing = {"full_cell_id", *mapping.values()} - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    frame = frame[["full_cell_id", *mapping.values()]].rename(
        columns={source: target for target, source in mapping.items()})
    if not frame.full_cell_id.is_unique or (frame.full_cell_id == "").any():
        raise ValueError("Cell IDs must be unique and nonempty")
    for field in ("patient", "sample", "genotype"):
        if (frame[field] == "").any():
            raise ValueError(f"Required metadata field {field} has empty values")
    cohort = frame[["patient", "genotype"]].drop_duplicates()
    if not cohort.patient.is_unique:
        raise ValueError("A patient has inconsistent genotype annotations")
    if len(cohort) != 29:
        raise ValueError(f"Expected 29 patients in the merged object, observed {len(cohort)}")
    patient_ids = cohort.patient.to_numpy()
    genotype = cohort.genotype.to_numpy()
    remaining, test = train_test_split(
        patient_ids, test_size=6, random_state=args.seed, stratify=genotype)
    by_patient = cohort.set_index("patient").genotype
    train, val = train_test_split(
        remaining, test_size=5, random_state=args.seed + 1,
        stratify=by_patient.loc[remaining].to_numpy())
    split = {patient: part for part, ids in
             (("train", train), ("val", val), ("test", test)) for patient in ids}

    random = np.random.default_rng(args.seed)
    selected_rows = []
    for patient, group in frame.groupby("patient", sort=True):
        group = group.sort_values("full_cell_id")
        keep = random.choice(len(group), min(len(group), args.max_cells_per_patient),
                             replace=False)
        selected_rows.append(group.iloc[np.sort(keep)])
    index = pd.concat(selected_rows, ignore_index=True)
    sampled_before_label_filter = len(index)
    labeled = index.cell_type.ne("") & index.cell_group.ne("")
    index = index.loc[labeled].copy()
    index["split"] = index.patient.map(split)
    index.sort_values(["patient", "sample", "full_cell_id"], inplace=True)
    index.reset_index(drop=True, inplace=True)
    if index.empty or set(index.split) != {"train", "val", "test"}:
        raise ValueError("No labeled cells remain in one or more splits")
    if index.groupby("patient").split.nunique().max() != 1:
        raise ValueError("Patient split leakage")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    index_path = args.output_dir / "cell_index.parquet"
    index.to_parquet(index_path, index=False)
    (args.output_dir / "cell_ids.txt").write_text(
        "\n".join(index.full_cell_id.astype(str)) + "\n", encoding="utf-8")
    report = {
        "dataset": "GSE316387",
        "metadata_source": str(args.metadata_csv.resolve()),
        "metadata_sha256": sha256(args.metadata_csv),
        "column_mapping": mapping,
        "seed": args.seed,
        "max_cells_per_patient": args.max_cells_per_patient,
        "sampled_before_label_filter": sampled_before_label_filter,
        "excluded_missing_author_label": int((~labeled).sum()),
        "index_rows": len(index),
        "index_sha256": sha256(index_path),
        "patient_split": {part: sorted(str(patient) for patient in ids)
                          for part, ids in (("train", train), ("val", val),
                                            ("test", test))},
        "cell_counts": {part: int(index.split.eq(part).sum()) for part in
                        ("train", "val", "test")},
        "labels": {column: {part: index.loc[index.split.eq(part), column]
                            .value_counts().to_dict() for part in
                            ("train", "val", "test")}
                   for column in ("cell_type", "cell_group")},
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(index), "patients": len(cohort),
                      "splits": report["cell_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
