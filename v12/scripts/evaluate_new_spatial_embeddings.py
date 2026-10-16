#!/usr/bin/env python3
"""Evaluate one frozen embedding table on a patient-disjoint spatial cohort.

Input IDs must exactly match the order of cell_index.parquet. Run on a compute
node. The probe uses labeled train/validation patients; K-means sees held-out
test features but never their labels. K is fixed by the protocol, not tuned on
test labels. ARI measures agreement with author annotations, not independent
ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler


C_GRID = (0.01, 0.1, 1.0, 10.0)
KMEANS_SEEDS = (17, 42, 101)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_index(path: Path, label_column: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"full_cell_id", "patient", "split", label_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Index missing columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Index contains no cells or missing required values")
    if not frame.full_cell_id.is_unique:
        raise ValueError("Cell IDs are duplicated")
    if set(frame.split) != {"train", "val", "test"}:
        raise ValueError("Index must contain train, val, and test splits")
    if frame.full_cell_id.astype(str).str.contains(r"[\r\n]").any():
        raise ValueError("Cell IDs cannot contain newlines")
    if frame.groupby("patient").split.nunique().max() != 1:
        raise ValueError("A patient appears in more than one split")
    return frame


def load_embedding(embedding_path: Path, ids_path: Path | None,
                   index: pd.DataFrame) -> np.ndarray:
    expected = index.full_cell_id.astype(str).tolist()
    if embedding_path.suffix == ".npz":
        if ids_path is not None:
            raise ValueError("NPZ carries its own cell_ids; omit --ids")
        with np.load(embedding_path, allow_pickle=False) as archive:
            if not {"cell_ids", "embedding"}.issubset(archive.files):
                raise ValueError("NPZ lacks cell_ids or embedding")
            observed = [str(value) for value in archive["cell_ids"]]
            X = archive["embedding"]
    elif embedding_path.suffix == ".npy":
        if ids_path is None:
            raise ValueError("NPY requires --ids")
        observed = ids_path.read_text(encoding="utf-8").splitlines()
        X = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    else:
        raise ValueError("Expected an NPY or NPZ embedding file")
    if observed != expected:
        raise ValueError("Embedding IDs differ from index IDs or row order")
    if X.ndim != 2 or X.shape[0] != len(expected) or X.shape[1] < 2:
        raise ValueError(f"Invalid embedding shape: {X.shape}")
    if not np.issubdtype(X.dtype, np.floating):
        raise ValueError("Embeddings must have floating dtype")
    for start in range(0, len(X), 8192):
        if not np.isfinite(X[start:start + 8192]).all():
            raise ValueError("Embeddings contain NaN or infinity")
    return X


def probe_score(y: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "cells": int(len(y)),
        "macro_f1": float(f1_score(y, prediction, labels=np.unique(y),
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
    }


def fit_probe(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    model = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs",
                               max_iter=2000, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X, y)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--ids", type=Path)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label-column", choices=("cell_type", "cell_group",
                                                    "meta_niche_name"),
                        required=True)
    parser.add_argument("--exclude-empty-labels", action="store_true")
    parser.add_argument("--clusters", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.clusters < 2 or args.output_dir.exists():
        parser.error("clusters must be >=2 and output-dir must not exist")

    index = load_index(args.index, args.label_column)
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    if not isinstance(provenance, dict) or not \
            provenance.get("model_name", provenance.get("model")):
        raise ValueError("Embedding provenance must identify the model")
    X = load_embedding(args.embedding, args.ids, index)
    excluded_unlabeled = 0
    if args.exclude_empty_labels:
        keep = index[args.label_column].astype(str).ne("").to_numpy()
        excluded_unlabeled = int((~keep).sum())
        index = index.loc[keep].reset_index(drop=True)
        X = X[keep]
    masks = {part: index.split.eq(part).to_numpy() for part in
             ("train", "val", "test")}
    y = index[args.label_column].astype(str).to_numpy()
    y_train, y_val, y_test = (y[masks[part]] for part in
                              ("train", "val", "test"))
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training split needs at least two cell types")
    if len(X[masks["test"]]) <= args.clusters:
        raise ValueError("K-means clusters must be fewer than test cells")

    scaler = StandardScaler().fit(X[masks["train"]])
    scaled = {part: scaler.transform(X[mask]).astype(np.float32) for
              part, mask in masks.items()}

    tuning = []
    best = None
    for C in C_GRID:
        classifier = fit_probe(scaled["train"], y_train, C)
        row = {"C": C, **probe_score(y_val, classifier.predict(scaled["val"]))}
        tuning.append(row)
        if best is None or row["macro_f1"] > best[0]:
            best = (row["macro_f1"], C, classifier)
    assert best is not None
    _, selected_C, classifier = best
    prediction = classifier.predict(scaled["test"])

    clustering = []
    test_patients = index.loc[masks["test"], "patient"].astype(str).to_numpy()
    for seed in KMEANS_SEEDS:
        assigned = KMeans(n_clusters=args.clusters, n_init=10,
                          random_state=seed).fit_predict(scaled["test"])
        clustering.append({"seed": seed,
                           "ari": float(adjusted_rand_score(y_test, assigned)),
                           "ari_by_patient": {
                               patient: float(adjusted_rand_score(
                                   y_test[test_patients == patient],
                                   assigned[test_patients == patient]))
                               for patient in sorted(set(test_patients))}})

    test_index = index.loc[masks["test"]].copy()
    test_index["prediction"] = prediction
    patient_rows = []
    for patient, group in test_index.groupby("patient", sort=True):
        patient_rows.append({"patient": str(patient),
                             **probe_score(group[args.label_column].astype(str).to_numpy(),
                                           group.prediction.to_numpy())})

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    test_index.to_parquet(output / "test_predictions.parquet", index=False)
    pd.DataFrame(patient_rows).to_csv(output / "test_by_patient.csv", index=False)
    report = {
        "model": args.model,
        "dataset": "GSE316387",
        "label_column": args.label_column,
        "annotation_level": "author_expression_derived_silver_standard",
        "excluded_empty_labels": excluded_unlabeled,
        "split_patients": {part: sorted(index.loc[mask, "patient"].astype(str).unique())
                           for part, mask in masks.items()},
        "split_cells": {part: int(mask.sum()) for part, mask in masks.items()},
        "embedding_dimension": int(X.shape[1]),
        "selected_C": selected_C,
        "validation_grid": tuning,
        "test_linear_probe": probe_score(y_test, prediction),
        "test_labels_absent_from_train": sorted(set(y_test) - set(y_train)),
        "test_kmeans": {"clusters": args.clusters, "runs": clustering,
                         "ari_mean": float(np.mean([run["ari"] for run in clustering])),
                         "ari_std": float(np.std([run["ari"] for run in clustering]))},
        "input_sha256": {"index": sha256(args.index),
                         "embedding": sha256(args.embedding),
                         "ids": sha256(args.ids) if args.ids else None,
                         "provenance": sha256(args.provenance)},
        "provenance": provenance,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "label": args.label_column,
                      "linear_probe": report["test_linear_probe"],
                      "ARI": report["test_kmeans"]["ari_mean"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
