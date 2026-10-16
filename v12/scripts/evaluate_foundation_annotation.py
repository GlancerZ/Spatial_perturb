"""Compare frozen single-cell embeddings on identical LungPF spatial cells.

Supported complete per-sample formats in ``--embeddings-root/<model>/``::

    <sample>.npz + <sample>.npz.manifest.json
        NPZ keys: cell_ids, embedding
    <sample>/embeddings.npy + <sample>/cell_ids.txt + <sample>/provenance.json
    <sample>.npy + <sample>.ids.txt + model_manifest.json  # legacy format

The exact per-sample order is the order in the exported ``cell_index.parquet``.
This script refuses missing, extra, reordered, duplicate, or nonfinite rows.
It uses the original donor 10/3/3 split; StandardScaler is fit on training
donors only; balanced multinomial logistic C is selected on validation macro-F1.
Test labels are scored once, after C selection. Run on a compute node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT = Path("outputs/foundation_annotation_lungpf/input")
DEFAULT_EMBEDDINGS = Path("outputs/foundation_annotation_lungpf/embeddings")
DEFAULT_OUTPUT = Path("outputs/foundation_annotation_lungpf/evaluation")
TASK_COLUMNS = {"cell_type": "final_CT", "region": "region_label"}
PROVENANCE_FIELDS = ("model_name", "model_source", "checkpoint_or_version",
                     "embedding_recipe", "input_transform")
EXPECTED_MODEL_IDS = {
    "geneformer": "Geneformer-V2-316M",
    "scgpt": "ordinary_scgpt_whole_human",
    "scfoundation": "scFoundation",
    "state": "ArcInstitute_STATE_SE_600M",
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_index(input_root: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((input_root / "manifest.json").read_text())
    if manifest.get("schema") != "foundation_annotation_lungpf_v1":
        raise ValueError("Unknown exported LungPF input schema")
    index = pd.read_parquet(input_root / "cell_index.parquet")
    if len(index) != manifest["index_rows"] or not index.full_cell_id.is_unique:
        raise ValueError("Cell index row count/IDs differ from manifest")
    if index.full_cell_id.isna().any() or index["sample"].isna().any():
        raise ValueError("Cell index contains missing IDs or samples")
    if sorted(index["sample"].unique()) != sorted(manifest["samples"]):
        raise ValueError("Sample set differs from export manifest")
    if index.patient.nunique() != 16 or set(index.split) != {"train", "val", "test"}:
        raise ValueError("Unexpected donor/split cohort")
    for patient, part in index.groupby("patient", sort=False):
        splits = part.split.unique()
        if len(splits) != 1 or manifest["donor_split"].get(patient) != splits[0]:
            raise ValueError(f"Donor split mismatch: {patient}")
    if [s for s in index["sample"].drop_duplicates()] != manifest["samples"]:
        raise ValueError("Sample order differs from export manifest")
    if manifest.get("cell_id_sha256") != hashlib.sha256(
            "\n".join(index.full_cell_id.astype(str)).encode()).hexdigest():
        raise ValueError("Exported cell ID order/hash mismatch")
    return index, manifest


def read_ids(ids: np.ndarray | Path) -> list[str]:
    if isinstance(ids, Path):
        return ids.read_text(encoding="utf-8").splitlines()
    if ids.ndim != 1 or ids.dtype.kind not in "US":
        raise ValueError("NPZ cell_ids must be a one-dimensional Unicode/bytes array")
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in ids]


def load_source_array(source: dict):
    if source["kind"] == "npz":
        with np.load(source["path"], allow_pickle=False) as archive:
            return archive["embedding"]
    return np.load(source["path"], mmap_mode="r", allow_pickle=False)


def validate_embeddings(model_dir: Path, model: str, index: pd.DataFrame,
                        manifest: dict, input_root: Path, *, chunk_rows: int) -> tuple[int, list[dict], list[dict]]:
    model_manifest = model_dir / "model_manifest.json"
    if model_manifest.exists():
        provenance = json.loads(model_manifest.read_text())
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError(f"{model}: empty model_manifest.json")
        if str(provenance.get("model_name", model)).casefold() != model.casefold():
            raise ValueError(f"{model}: model_manifest.json identity mismatch")
    allowed = {"model_manifest.json"} if model_manifest.exists() else set()
    records: list[dict] = []
    sources: list[dict] = []
    dimension = None
    for sample in manifest["samples"]:
        transform_log = None
        npz_path = model_dir / f"{sample}.npz"
        directory = model_dir / sample
        legacy_path = model_dir / f"{sample}.npy"
        present = [name for name, exists in (
            ("npz", npz_path.exists()), ("directory", directory.exists()),
            ("legacy", legacy_path.exists())) if exists]
        if len(present) != 1:
            raise ValueError(f"{model}/{sample}: expected exactly one embedding format, found {present}")
        kind = present[0]
        transform_log = None
        if kind == "npz":
            provenance_path = model_dir / f"{sample}.npz.manifest.json"
            if not provenance_path.is_file():
                raise FileNotFoundError(f"{model}/{sample}: missing NPZ manifest")
            with np.load(npz_path, allow_pickle=False) as archive:
                if not {"cell_ids", "embedding"}.issubset(archive.files):
                    raise ValueError(f"{model}/{sample}: NPZ lacks cell_ids or embedding")
                observed = read_ids(archive["cell_ids"])
                array = archive["embedding"]
            embedding_path = npz_path
            allowed.update((npz_path.name, provenance_path.name))
        elif kind == "directory":
            if not directory.is_dir():
                raise ValueError(f"{model}/{sample}: expected sample directory")
            embedding_path = directory / "embeddings.npy"
            ids_path = directory / "cell_ids.txt"
            provenance_path = directory / "provenance.json"
            if not all(p.is_file() for p in (embedding_path, ids_path, provenance_path)):
                raise FileNotFoundError(f"{model}/{sample}: incomplete sample directory")
            transform_log = directory / "transform.log"
            unexpected = {p.name for p in directory.iterdir()} - {
                "embeddings.npy", "cell_ids.txt", "provenance.json", "transform.log"}
            if unexpected:
                raise ValueError(f"{model}/{sample}: unexpected files {sorted(unexpected)}")
            if transform_log.exists() and not transform_log.is_file():
                raise ValueError(f"{model}/{sample}: transform.log must be a regular file")
            observed = read_ids(ids_path)
            array = load_source_array({"kind": kind, "path": embedding_path})
            allowed.add(directory.name)
        else:
            embedding_path = legacy_path
            ids_path = model_dir / f"{sample}.ids.txt"
            if not model_manifest.exists() or not ids_path.is_file():
                raise FileNotFoundError(f"{model}/{sample}: legacy format needs IDs and model_manifest.json")
            if any(not isinstance(provenance.get(key), str) or not provenance[key].strip()
                   for key in PROVENANCE_FIELDS):
                raise ValueError(f"{model}: incomplete legacy model_manifest.json")
            provenance_path = model_manifest
            observed = read_ids(ids_path)
            array = load_source_array({"kind": kind, "path": embedding_path})
            allowed.update((embedding_path.name, ids_path.name))
        sample_provenance = json.loads(provenance_path.read_text())
        if not isinstance(sample_provenance, dict) or not sample_provenance:
            raise ValueError(f"{model}/{sample}: empty provenance manifest")
        if (str(sample_provenance.get("model_name", model)).casefold() != model.casefold()
                or sample_provenance.get("sample", sample) != sample):
            raise ValueError(f"{model}/{sample}: provenance identity mismatch")
        if kind != "legacy":
            if sample_provenance.get("model") != EXPECTED_MODEL_IDS.get(model):
                raise ValueError(f"{model}/{sample}: incorrect backbone identity in provenance")
            declared_input = sample_provenance.get("input_h5ad", sample_provenance.get("input"))
            expected_input = (input_root / sample / "counts.h5ad").resolve()
            if not isinstance(declared_input, str) or Path(declared_input).resolve() != expected_input:
                raise ValueError(f"{model}/{sample}: provenance points to a different input H5AD")
            if model in {"geneformer", "scfoundation"}:
                if sample_provenance.get("counts_layer") != "X":
                    raise ValueError(f"{model}/{sample}: LungPF must use raw whole-cell X")
                if model == "scfoundation" and sample_provenance.get(
                        "input_count_semantics") != "raw_integer_whole_cell":
                    raise ValueError(f"{model}/{sample}: unexpected count semantics")
            elif (sample_provenance.get("input_source") != "raw_integer_whole_cell_X"
                  or sample_provenance.get("input_layers") != ["X"]):
                raise ValueError(f"{model}/{sample}: LungPF must use raw whole-cell X")
        expected = index.loc[index["sample"].eq(sample), "full_cell_id"].astype(str).tolist()
        if observed != expected:
            raise ValueError(f"{model}/{sample}: missing, extra, or reordered cell IDs")
        if array.ndim != 2 or array.shape[0] != len(expected) or array.shape[1] < 2:
            raise ValueError(f"{model}/{sample}: invalid embedding shape {array.shape}")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"{model}/{sample}: embeddings must be floating point")
        if dimension is None:
            dimension = array.shape[1]
        elif array.shape[1] != dimension:
            raise ValueError(f"{model}/{sample}: feature dimension mismatch")
        for start in range(0, len(array), chunk_rows):
            if not np.isfinite(array[start:start + chunk_rows]).all():
                raise ValueError(f"{model}/{sample}: nonfinite embedding values")
        records.append({"sample": sample, "cells": len(expected),
                        "features": dimension, "format": kind,
                        "cell_id_sha256": hashlib.sha256(
                            "\n".join(observed).encode()).hexdigest(),
                        "embedding_sha256": digest(embedding_path),
                        "provenance_sha256": digest(provenance_path),
                        "transform_log_sha256": digest(transform_log)
                        if transform_log is not None and transform_log.is_file() else None})
        sources.append({"sample": sample, "kind": kind, "path": embedding_path})
    extra = {p.name for p in model_dir.iterdir()} - allowed
    if extra:
        raise ValueError(f"{model}: unexpected entries in embedding directory: {sorted(extra)}")
    return dimension, records, sources


def source_batches(sources: list[dict], index: pd.DataFrame,
                   *, chunk_rows: int):
    for source in sources:
        sample = source["sample"]
        obs = index.loc[index["sample"].eq(sample)]
        array = load_source_array(source)
        for start in range(0, len(obs), chunk_rows):
            stop = min(start + chunk_rows, len(obs))
            yield obs.iloc[start:stop], np.asarray(array[start:stop], dtype=np.float32)


def fit_scaler(sources: list[dict], index: pd.DataFrame,
               column: str, *, chunk_rows: int) -> StandardScaler:
    scaler = StandardScaler()
    rows = 0
    for obs, values in source_batches(sources, index, chunk_rows=chunk_rows):
        selected = obs["split"].eq("train") & obs[column].notna()
        if selected.any():
            scaler.partial_fit(values[selected.to_numpy()])
            rows += int(selected.sum())
    if rows == 0 or not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError("No finite training data for StandardScaler")
    return scaler


def write_scaled_features(path: Path, sources: list[dict], index: pd.DataFrame,
                          column: str, split: str,
                          scaler: StandardScaler, dimension: int, *, chunk_rows: int):
    mask = index["split"].eq(split) & index[column].notna()
    expected_rows = int(mask.sum())
    if expected_rows == 0:
        raise ValueError(f"No labeled {split} cells")
    feature = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                        shape=(expected_rows, dimension))
    offset = 0
    for obs, values in source_batches(sources, index, chunk_rows=chunk_rows):
        selected = (obs["split"].eq(split) & obs[column].notna()).to_numpy()
        n = int(selected.sum())
        if n:
            feature[offset:offset + n] = scaler.transform(values[selected]).astype(np.float32)
            offset += n
    if offset != expected_rows:
        raise ValueError(f"{split}: feature rows do not match label rows")
    feature.flush()
    labels = index.loc[mask, column].astype(str).to_numpy()
    return feature, labels, index.loc[mask].copy()


def fit_probe(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    classifier = LogisticRegression(
        C=C, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(X, y)
    return classifier


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "cells": int(len(y_true)), "classes": int(len(np.unique(y_true))),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.unique(y_true),
                                   average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def evaluate_task(model_dir: Path, output_dir: Path, index: pd.DataFrame,
                  manifest: dict, model: str, task: str, dimension: int,
                  records: list[dict], sources: list[dict], input_root: Path,
                  *, chunk_rows: int) -> dict:
    column = TASK_COLUMNS[task]
    model_output = output_dir / model
    final_dir = model_output / task
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {final_dir}")
    model_output.mkdir(parents=True, exist_ok=True)
    task_dir = Path(tempfile.mkdtemp(prefix=f".{task}.partial-", dir=model_output))
    scaler = fit_scaler(sources, index, column, chunk_rows=chunk_rows)
    X_train, y_train, train_obs = write_scaled_features(
        task_dir / "train_features.npy", sources, index,
        column, "train", scaler, dimension, chunk_rows=chunk_rows)
    X_val, y_val, val_obs = write_scaled_features(
        task_dir / "val_features.npy", sources, index,
        column, "val", scaler, dimension, chunk_rows=chunk_rows)
    tuning = []
    best = None
    for C in manifest["C_grid"]:
        classifier = fit_probe(X_train, y_train, float(C))
        prediction = classifier.predict(X_val)
        result = {"C": float(C), **score(y_val, prediction)}
        tuning.append(result)
        if best is None or result["macro_f1"] > best[0]:
            best = (result["macro_f1"], float(C), classifier)
    assert best is not None
    _, selected_C, classifier = best

    # Only after C selection do test labels enter the probe evaluation.
    X_test, y_test, test_obs = write_scaled_features(
        task_dir / "test_features.npy", sources, index,
        column, "test", scaler, dimension, chunk_rows=chunk_rows)
    y_pred = classifier.predict(X_test)
    confidence = classifier.predict_proba(X_test).max(axis=1)
    primary = score(y_test, y_pred)
    donor_rows = []
    for patient in sorted(test_obs.patient.unique()):
        mask = test_obs.patient.eq(patient).to_numpy()
        donor_rows.append({"patient": patient, **score(y_test[mask], y_pred[mask])})
    # Match the primary macro-F1 denominator: classes actually present in test.
    test_labels = sorted(set(y_test))
    confusion_labels = sorted(set(y_test) | set(y_pred))
    per_class = classification_report(y_test, y_pred, labels=test_labels,
                                      output_dict=True, zero_division=0)
    prediction_table = test_obs[["full_cell_id", "sample", "patient",
                                 "x_centroid", "y_centroid"]].copy()
    prediction_table["label"] = y_test
    prediction_table["prediction"] = y_pred
    prediction_table["confidence"] = confidence
    prediction_table.to_parquet(task_dir / "test_predictions.parquet", index=False)
    pd.DataFrame(donor_rows).to_csv(task_dir / "test_by_donor.csv", index=False)
    pd.DataFrame(per_class).T.to_csv(task_dir / "test_per_class.csv")
    pd.DataFrame(confusion_matrix(y_test, y_pred, labels=confusion_labels),
                 index=confusion_labels, columns=confusion_labels).to_csv(
                     task_dir / "test_confusion.csv")
    np.savez(task_dir / "probe.npz", coef=classifier.coef_, intercept=classifier.intercept_,
             classes=classifier.classes_, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_)
    report = {
        "model": model, "task": task, "label_column": column,
        "selected_C": selected_C, "test": primary, "validation_grid": tuning,
        "train_cells": int(len(train_obs)), "validation_cells": int(len(val_obs)),
        "train_donors": sorted(train_obs.patient.unique()),
        "validation_donors": sorted(val_obs.patient.unique()),
        "test_donors": sorted(test_obs.patient.unique()),
        "test_labels_unseen_in_train": sorted(set(y_test) - set(y_train)),
        "predicted_labels_absent_in_test": sorted(set(y_pred) - set(y_test)),
        "test_known_label_fraction": float(np.isin(y_test, np.unique(y_train)).mean()),
        "input_manifest_sha256": digest(input_root / "manifest.json"),
        "model_manifest_sha256": digest(model_dir / "model_manifest.json")
        if (model_dir / "model_manifest.json").exists() else None,
        "embedding_files": records,
        "note": "Historical LungPF test donors have been viewed in prior experiments; not a fresh independent generalization test.",
    }
    write_json(task_dir / "metrics.json", report)
    if final_dir.exists():
        raise FileExistsError(f"Another result appeared while evaluating: {final_dir}")
    os.rename(task_dir, final_dir)
    print(json.dumps({"model": model, "task": task, "selected_C": selected_C,
                      "test": primary}, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", choices=["cell_type", "region", "both"], default="both")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args()
    if args.chunk_rows < 1 or not args.model.isidentifier():
        parser.error("Positive --chunk-rows and identifier-safe --model are required")
    index, manifest = validate_index(args.input_root)
    if manifest.get("C_grid") != [0.01, 0.1, 1.0, 10.0]:
        raise ValueError("Unexpected validation C grid")
    model_dir = args.embeddings_root / args.model
    dimension, records, sources = validate_embeddings(
        model_dir, args.model, index, manifest, args.input_root,
        chunk_rows=args.chunk_rows)
    tasks = list(TASK_COLUMNS) if args.task == "both" else [args.task]
    for task in tasks:
        evaluate_task(model_dir, args.output_root, index, manifest, args.model,
                      task, dimension, records, sources, args.input_root,
                      chunk_rows=args.chunk_rows)


if __name__ == "__main__":
    main()
