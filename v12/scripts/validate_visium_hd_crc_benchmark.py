#!/usr/bin/env python3
"""Independently verify published Visium HD CRC counts and benchmark scores.

This validator does not refit probes or select a split. It verifies the frozen
cohorts and recomputes reported test metrics from per-cell predictions. Run in
an actual Slurm compute-node step; successful QA is not a claim of biological
generalization or model superiority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import traceback


PRIMARY = ("v12_scgpt_total_cls", "v12_local", "v12_spatial", "v12_fused",
           "v12_concat", "scgpt_spatial_v1_cls")
READOUTS = ("nucleus", "cytoplasm", "local", "spatial", "fused", "concat",
            "scgpt_total_cls")
EXPERT_CLASSES = {"epithelial cells", "immune cells", "stromal cells"}


def require_compute_node():
    host = socket.gethostname().split(".")[0]
    node = os.environ.get("SLURMD_NODENAME", "").split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID") or host != node:
        raise RuntimeError("Run in a verified Slurm compute-node step, never a login node")


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    ensure(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256(("\n".join(map(str, ids)) + "\n").encode()).hexdigest()


def verify(args, qa):
    # Numerical and project-data imports occur only after the compute guard.
    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                                 f1_score, normalized_mutual_info_score)

    root = args.benchmark_dir.resolve()
    hashes = {}

    def verify_hash(path, expected, label):
        path = path.resolve()
        ensure(path.is_file(), f"Missing hash input {label}: {path}")
        ensure(isinstance(expected, str) and len(expected) == 64, f"Invalid SHA256 for {label}")
        if path not in hashes:
            hashes[path] = file_sha256(path)
        ensure(hashes[path] == expected, f"SHA256 mismatch for {label}: {path}")
        return hashes[path]

    def close_equal(actual, expected, label):
        if pd.isna(actual) or pd.isna(expected):
            ensure(pd.isna(actual) and pd.isna(expected), f"Missing/nonmissing mismatch: {label}")
        else:
            ensure(np.isclose(float(actual), float(expected), rtol=0, atol=1e-10),
                   f"Metric mismatch {label}: actual={actual}, reported={expected}")

    def equal_strings(a, b, label):
        ensure(np.array_equal(np.asarray(a, dtype=str), np.asarray(b, dtype=str)), label)

    def compare_frames(a, b, columns, label):
        ensure(len(a) == len(b), f"Row count mismatch: {label}")
        for column in columns:
            if column in ("x_um", "y_um"):
                ensure(np.allclose(a[column], b[column], rtol=0, atol=1e-6), f"{label}/{column}")
            else:
                equal_strings(a[column].fillna("<MISSING>"), b[column].fillna("<MISSING>"),
                              f"{label}/{column}")

    def compare_metric_tables(a, b, label):
        try:
            pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                          check_dtype=False, check_exact=False, rtol=0, atol=1e-10)
        except AssertionError as error:
            raise ValueError(label + ": " + str(error)) from error

    count_audit = read_json(root / "compartments/audit.json")
    ensure(count_audit.get("status") == "complete", "Compartment stage not complete")
    required_checks = {"partition_conservation", "sparse_aggregation_conservation",
                       "cell_filter_conservation", "X_equals_nuclear_plus_cytoplasmic"}
    count_checks = count_audit.get("checks", {})
    ensure(required_checks.issubset(count_checks) and all(v is True for v in count_checks.values()),
           "Compartment count-conservation audit did not pass")
    totals = {"X": 0, "nuclear_counts": 0, "cytoplasmic_counts": 0}
    compartment_path = root / "compartments/compartments.h5ad"
    with h5py.File(compartment_path, "r") as handle:
        groups = {"X": handle["X"], "nuclear_counts": handle["layers/nuclear_counts"],
                  "cytoplasmic_counts": handle["layers/cytoplasmic_counts"]}
        shapes = {key: tuple(group.attrs["shape"]) for key, group in groups.items()}
        ensure(len(set(shapes.values())) == 1, "Count matrix shapes differ")
        shape = shapes["X"]
        ensure(shape[0] == count_audit["n_cells_retained"] and
               shape[1] == count_audit["features"]["n_measured_genes"], "Count audit shape mismatch")
        for key, group in groups.items():
            encoding = group.attrs.get("encoding-type")
            if isinstance(encoding, bytes):
                encoding = encoding.decode()
            ensure(encoding == "csr_matrix", f"Validator expects row-addressable CSR storage: {key}")
            ensure(len(group["indptr"]) == shape[0] + 1, f"Invalid CSR indptr: {key}")
            ensure(int(group["indptr"][0]) == 0 and int(group["indptr"][-1]) ==
                   len(group["data"]) == len(group["indices"]), f"Invalid CSR endpoint/payload: {key}")
        obs = handle["obs"]
        index_name = obs.attrs["_index"]
        compartment_ids = obs[index_name].asstr()[:]
        ensure(len(compartment_ids) == shape[0] and len(set(compartment_ids)) == shape[0],
               "Invalid compartment cell IDs")
        for start in range(0, shape[0], args.count_chunk_rows):
            stop = min(start + args.count_chunk_rows, shape[0])
            blocks = {}
            for key, group in groups.items():
                pointers = np.asarray(group["indptr"][start:stop + 1], dtype=np.int64)
                ensure(np.all(np.diff(pointers) >= 0), f"Nonmonotone CSR indptr: {key}")
                lo, hi = int(pointers[0]), int(pointers[-1])
                values = np.asarray(group["data"][lo:hi])
                indices = np.asarray(group["indices"][lo:hi])
                ensure(len(values) == len(indices) == hi - lo, f"Invalid sparse payload: {key}")
                ensure(values.dtype.kind in "iu" and np.all(values >= 0), f"Noninteger/negative raw counts: {key}")
                ensure(np.all(indices >= 0) and np.all(indices < shape[1]), f"Invalid sparse column: {key}")
                ensure(np.all(values <= np.iinfo(np.int64).max), f"Counts outside int64: {key}")
                blocks[key] = sparse.csr_matrix((values.astype(np.int64, copy=False), indices,
                                                pointers - lo), shape=(stop - start, shape[1]))
                totals[key] += int(blocks[key].sum(dtype=np.int64))
            difference = blocks["X"] - blocks["nuclear_counts"] - blocks["cytoplasmic_counts"]
            difference.eliminate_zeros()
            ensure(difference.nnz == 0, f"X != nuclear + cytoplasmic in rows {start}:{stop}")
    umi = count_audit["umi"]
    ensure(totals["X"] == totals["nuclear_counts"] + totals["cytoplasmic_counts"] == umi["tissue_assigned_umi"],
           "Recomputed assigned UMI total differs from audit")
    ensure(totals["nuclear_counts"] == umi["nuclear_umi"] and
           totals["cytoplasmic_counts"] == umi["cytoplasmic_umi"], "Recomputed compartment UMI differs")
    ensure(umi["all_measured_gene_umi"] == umi["outside_tissue_umi"] +
           umi["tissue_unassigned_umi"] + umi["tissue_assigned_umi"], "Input UMI partition does not conserve counts")
    qa["counts"] = {"cells": int(shape[0]), "genes": int(shape[1]), "recomputed_umi": totals,
                    "all_rows_X_equals_N_plus_C": True}
    print("[QA] All compartment rows conserve X=N+C", flush=True)

    v12_complete = read_json(root / "v12/COMPLETE.json")
    v12_audit = read_json(root / "v12/input_audit.json")
    ensure(v12_complete.get("status") == "INFERENCE_AND_ARTIFACT_QC_COMPLETE" and
           v12_complete.get("full_compartment_model_evaluated") is True, "V12 completion contract invalid")
    for name, expected in v12_complete["artifacts"].items():
        ensure(Path(name).name == name, "Unsafe V12 artifact filename")
        verify_hash(root / "v12" / name, expected, "V12 artifact " + name)
    verify_hash(compartment_path, v12_audit["input_sha256"], "V12 compartment input")
    official_audit = read_json(root / "scgpt_spatial/audit.json")
    ensure(official_audit.get("status") == "OFFICIAL_V1_FROZEN_EMBEDDINGS_COMPLETE", "Official baseline not complete")
    cells = ad.read_h5ad(root / "v12/embeddings.h5ad", backed="r")
    try:
        cell_ids = np.asarray(cells.obs_names, dtype=str)
        ensure(cells.obs_names.is_unique, "Duplicate V12 cell IDs")
        with np.load(root / "scgpt_spatial/embeddings.npz", allow_pickle=False) as artifact:
            equal_strings(artifact["cell_ids"], cell_ids, "Official/V12 ordered cell IDs differ")
            embedding = artifact["embeddings"]
            ensure(embedding.shape == (len(cell_ids), official_audit["embedding_dim"]) and
                   np.isfinite(embedding).all(), "Invalid official embeddings")
            ensure(np.all(np.linalg.norm(embedding, axis=1) > 1e-8), "All-zero official embedding")
        dimensions = {"scgpt_spatial_v1_cls": int(embedding.shape[1])}
        ensure(official_audit["cells"] == len(cell_ids), "Official cell count mismatch")
        ensure(official_audit["center_ids_sha256"] == hashlib.sha256("\n".join(cell_ids).encode()).hexdigest(),
               "Official ordered cell-ID checksum differs")
        availability = {"scgpt_spatial_v1_cls": np.ones(len(cell_ids), dtype=bool)}
        for name in READOUTS:
            flags = cells.obs[name + "_available"]
            ensure(not flags.isna().any() and flags.isin([True, False]).all(), f"Invalid availability: {name}")
            availability["v12_" + name] = flags.to_numpy(dtype=bool)
            value = np.asarray(cells.obsm["X_" + name])
            ensure(value.ndim == 2 and value.shape[0] == len(cell_ids) and
                   np.isfinite(value).all(), f"Invalid V12 embedding: {name}")
            active = value[availability["v12_" + name]]
            ensure(np.all(np.linalg.norm(active, axis=1) > 1e-8), f"All-zero available V12 embedding: {name}")
            ensure(v12_complete["embedding_qc"][name]["available"] == len(active), "V12 QC availability differs")
            dimensions["v12_" + name] = int(value.shape[1])
        coords = np.asarray(cells.obsm["spatial_um"], dtype=np.float64)
        ensure(coords.shape == (len(cell_ids), 2) and np.isfinite(coords).all(), "Invalid V12 coordinates")
        source_rows = cells.obs["source_row"].to_numpy()
        ensure(source_rows.dtype.kind in "iu" and np.all(np.diff(source_rows) > 0) and
               np.all(source_rows >= 0) and np.all(source_rows < len(compartment_ids)), "V12 source rows invalid")
        equal_strings(compartment_ids[source_rows], cell_ids, "V12 IDs differ from source compartment rows")
    finally:
        cells.file.close()

    label_audit = read_json(root / "labels/audit.json")
    ensure(label_audit.get("status") == "LABEL_MATCHES_BUILT", "Expert label matching not complete")
    labels = pd.read_csv(root / "labels/labels.csv", dtype={"cell_id": str})
    ensure(not labels.cell_id.duplicated().any() and len(labels) == label_audit["matched_expert_cells"],
           "Expert label cell count/uniqueness mismatch")
    ensure(labels.cell_type.isin(EXPERT_CLASSES).all(), "Unexpected expert classes")
    center_ids = (root / "labels/center_ids.txt").read_text().splitlines()
    equal_strings(center_ids, labels.cell_id, "Requested center list differs from expert labels")
    ensure(set(cell_ids).issubset(set(center_ids)), "V12 includes cells outside requested expert cohort")
    ensure(v12_audit["center_selection"] == "requested_cell_ids" and
           v12_audit["mode"] == "compartments", "Wrong V12 inference cohort/mode")
    removed_ids = sorted(set(center_ids) - set(cell_ids))
    ensure(v12_audit["requested_center_ids"] == len(center_ids) and
           v12_audit["selected_centers"] == len(cell_ids) and
           v12_audit["requested_centers_failing_input_qc"] == len(removed_ids), "Input-QC exclusion count mismatch")
    qa["input_qc"] = {"expert_matches": len(center_ids), "inferred_cells": len(cell_ids),
                      "excluded_before_both_encoders": len(removed_ids),
                      "excluded_cell_ids_sha256": ids_sha256(removed_ids),
                      "minimum_mapped_positive_genes": v12_audit["min_mapped_positive_genes"]}

    evaluation = root / "evaluation"
    audit = read_json(evaluation / "audit.json")
    complete = read_json(evaluation / "COMPLETE.json")
    ensure(audit.get("status") == complete.get("status") == "MATCHED_CRC_BENCHMARK_COMPLETE", "Evaluation not complete")
    verify_hash(evaluation / "audit.json", complete["audit_sha256"], "Evaluation audit")
    source_files = {"v12_h5ad": root / "v12/embeddings.h5ad",
                    "spatial_v1_npz": root / "scgpt_spatial/embeddings.npz",
                    "labels_csv": root / "labels/labels.csv"}
    for key, path in source_files.items():
        # A staging directory may have been renamed after publication. Bind
        # this known stage and filename explicitly; never glob for substitutes.
        ensure(Path(audit["source_paths"][key]).name == path.name, f"Unexpected recorded source filename: {key}")
        verify_hash(path, audit["source_sha256"][key], "Evaluation source " + key)
    if args.evaluator_code:
        code_path = args.evaluator_code.resolve()
    else:
        candidates = [parent / "scripts" / filename for parent in root.parents
                      for filename in ("evaluate_enact_crc_embeddings_20261009.py", "evaluate_enact_crc_embeddings.py")]
        code_path = next((path for path in candidates if path.is_file()), None)
        ensure(code_path is not None, "Evaluator code unavailable; pass --evaluator-code")
    verify_hash(code_path, audit["source_sha256"]["evaluator_code"], "Evaluator code")
    for name, expected in audit["output_sha256"].items():
        ensure(Path(name).name == name, "Unsafe evaluator output filename")
        verify_hash(evaluation / name, expected, "Evaluation output " + name)

    frozen = pd.read_csv(evaluation / "frozen_score_cells.csv", dtype={"cell_id": str, "region_label": "string"})
    metrics = pd.read_csv(evaluation / "metrics.csv")
    main_metrics = pd.read_csv(evaluation / "primary_metrics.csv")
    available_csv = pd.read_csv(evaluation / "availability.csv", dtype={"cell_id": str})
    ensure(not frozen.cell_id.duplicated().any(), "Duplicate frozen cells")
    ensure(not metrics.duplicated(["cohort", "task", "readout"]).any(), "Duplicate reported metrics")
    equal_strings(available_csv.cell_id, cell_ids, "Availability order differs from V12 input")
    aligned_labels = labels.set_index("cell_id").loc[cell_ids].copy()
    valid_expert = aligned_labels.cell_type.isin(EXPERT_CLASSES).to_numpy()
    equal_strings(frozen.cell_id, cell_ids[valid_expert], "Frozen main cohort differs from expert input")
    equal_strings(frozen.cell_type, aligned_labels.cell_type.to_numpy()[valid_expert], "Frozen expert truth differs")
    normalized_regions = aligned_labels.region_label.astype("string").str.strip().replace("", pd.NA)
    equal_strings(frozen.region_label.fillna("<MISSING>"),
                  normalized_regions.iloc[np.flatnonzero(valid_expert)].fillna("<MISSING>"), "Frozen region truth differs")
    ensure(np.array_equal(available_csv.expert_label_in_primary.to_numpy(dtype=bool), valid_expert),
           "Expert primary availability flags differ")
    for name, flags in availability.items():
        values = available_csv[name + "_available"]
        ensure(not values.isna().any() and values.isin([True, False]).all() and
               np.array_equal(values.to_numpy(dtype=bool), flags), f"Availability table mismatch: {name}")
        if name in PRIMARY:
            ensure(flags[valid_expert].all(), f"Primary readout missing expert cells: {name}")
    ensure(audit["input_cells"] == len(cell_ids) and audit["primary_cells"] == len(frozen) == complete["primary_cells"] and
           audit["input_cells_without_valid_expert_type"] == int((~valid_expert).sum()), "Evaluation cohort counts mismatch")
    ensure(set(audit["primary_readouts"]) == set(PRIMARY), "Expected six primary readouts")
    ensure(np.allclose(frozen[["x_um", "y_um"]], coords[valid_expert], rtol=0, atol=1e-6), "Frozen coordinates differ")
    grid = np.floor(coords[valid_expert] / audit["tile_um"]).astype(np.int64)
    equal_strings(frozen.tile, [f"{x}_{y}" for x, y in grid], "Frozen tile IDs differ from physical coordinates")
    ensure(set(frozen.split) == {"train", "val", "test"}, "Invalid frozen split membership")
    tile_sets = {part: set(frozen.loc[frozen.split == part, "tile"]) for part in ("train", "val", "test")}
    ensure(all(not tile_sets[a] & tile_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))),
           "Frozen spatial tiles overlap across splits")
    ensure(audit["tile_counts"] == {part: len(values) for part, values in tile_sets.items()}, "Tile counts mismatch")
    primary_rows = metrics.loc[metrics.cohort == "primary_all_expert_cells"].reset_index(drop=True)
    compare_metric_tables(main_metrics, primary_rows, "primary_metrics.csv differs from main metrics subset")
    ensure(len(main_metrics) == complete["primary_metric_rows"] and len(metrics) == complete["all_metric_rows"],
           "Evaluation completion metric row counts differ")

    cohort_specs = {"primary_all_expert_cells": (np.ones(len(frozen), dtype=bool), PRIMARY, evaluation)}
    for view in ("nucleus", "cytoplasm"):
        cohort_specs[view + "_available_cells"] = (availability["v12_" + view][valid_expert],
            ("v12_" + view, "v12_fused", "scgpt_spatial_v1_cls"), evaluation / "supplementary" / view)
    expected_rows = set()
    verified_metrics = []
    task_columns = {"expert_cell_type": "cell_type", "annotated_region": "region_label"}
    for cohort_name, (mask, names, directory) in cohort_specs.items():
        cohort_audit = audit["cohorts"][cohort_name]
        subset = frozen.loc[mask].copy()
        if directory != evaluation:
            supplementary = pd.read_csv(directory / "frozen_score_cells.csv", dtype={"cell_id": str, "region_label": "string"})
            compare_frames(supplementary, subset, list(frozen.columns), "Supplementary frozen cohort " + cohort_name)
            ensure(read_json(directory / "audit.json") == cohort_audit, "Supplementary audit differs from parent audit")
            supplementary_metrics = pd.read_csv(directory / "metrics.csv")
            compare_metric_tables(supplementary_metrics, metrics.loc[metrics.cohort == cohort_name],
                                  "Supplementary metric table differs from parent metrics")
        equal_strings((directory / "cell_ids.txt").read_text().splitlines(), subset.cell_id, "Cohort cell IDs differ")
        ensure(cohort_audit["cells"] == len(subset) and cohort_audit["ordered_cell_ids_sha256"] == ids_sha256(subset.cell_id),
               "Cohort size/ordered checksum mismatch: " + cohort_name)
        close_equal(cohort_audit["coverage_of_primary"], mask.mean(), "Cohort coverage " + cohort_name)
        ensure(cohort_audit["readouts"] == list(names), "Cohort readout contract mismatch")
        for part in ("train", "val", "test"):
            part_mask = frozen.split.to_numpy() == part
            ensure(cohort_audit["split_cells"][part] == int((mask & part_mask).sum()), "Cohort split cell count mismatch")
            close_equal(cohort_audit["coverage_by_split"][part], mask[part_mask].mean(), "Cohort split coverage")
        expected_tasks = {"expert_cell_type"}
        if frozen.region_label.notna().any():
            expected_tasks.add("annotated_region")
        ensure(set(cohort_audit["tasks"]) == expected_tasks, "Cohort omitted an available task")
        for task, task_audit in cohort_audit["tasks"].items():
            ensure(task in task_columns, "Unknown benchmark task")
            column = task_columns[task]
            known = frozen[column].notna().to_numpy()
            # Intentionally the full main-cohort true-label union, including
            # train-unseen and supplementary-subset-absent regional classes.
            classes = np.unique(frozen.loc[known, column].astype(str).to_numpy())
            equal_strings(task_audit["scoring_classes"], classes, "Scoring class union differs")
            part_frames = {part: frozen.loc[mask & known & (frozen.split.to_numpy() == part)]
                           for part in ("train", "val", "test")}
            counts = {part: {str(k): int(v) for k, v in frame[column].value_counts().items()}
                      for part, frame in part_frames.items()}
            ensure(task_audit["class_counts"] == counts, "Task/split class counts differ")
            absent = {part: [str(name) for name in classes if counts[part].get(name, 0) == 0] for part in counts}
            ensure(task_audit["absent_classes_by_split"] == absent, "Absent task classes differ")
            train_classes = set(counts["train"])
            unseen = {part: sum(v for k, v in values.items() if k not in train_classes) for part, values in counts.items()}
            ensure(task_audit["unseen_label_cells_by_split"] == unseen and
                   task_audit["unseen_test_class_counts"] == {k: v for k, v in counts["test"].items() if k not in train_classes},
                   "Train-unseen class accounting differs")
            expected_test = part_frames["test"]
            ensure(task_audit["ordered_test_cell_ids_sha256"] == ids_sha256(expected_test.cell_id), "Task test cohort checksum differs")
            insufficient = (any(counts[part].get(str(name), 0) < 5 for part in counts for name in classes)
                            if task == "expert_cell_type" else
                            len(train_classes) < 2 or any(len(frame) == 0 for frame in part_frames.values()))
            if task_audit["status"] != "ASSESSED":
                ensure(task_audit["status"].startswith("NOT_ASSESSED_"), "Unknown task assessment state")
                ensure(not task_audit["selected_probes"], "Unassessed task contains selected probes")
                ensure(insufficient, "Available task was marked unassessed despite passing coverage gate")
                continue
            ensure(not insufficient, "Assessed task failed frozen coverage gate")
            if task == "expert_cell_type":
                ensure(all(counts[part].get(str(name), 0) >= 5 for part in counts for name in classes), "Assessed expert task failed class gate")
            else:
                ensure(len(train_classes) >= 2 and all(len(frame) for frame in part_frames.values()), "Assessed region failed coverage gate")
            ensure(set(task_audit["selected_probes"]) == set(names), "Missing selected probes")
            if cohort_name == "primary_all_expert_cells":
                ensure(set(names) == set(PRIMARY), "Main cohort lost a readout")
            for name in names:
                key = (cohort_name, task, name)
                expected_rows.add(key)
                match = metrics.loc[(metrics.cohort == cohort_name) & (metrics.task == task) & (metrics.readout == name)]
                ensure(len(match) == 1, "Missing reported metric: " + str(key))
                row = match.iloc[0]
                prediction_path = directory / (task + "__" + name + "__test_predictions.csv")
                predicted = pd.read_csv(prediction_path, dtype={"cell_id": str, "truth": str, "prediction": str})
                ensure(not predicted.cell_id.duplicated().any(), "Duplicate prediction cell IDs")
                compare_frames(predicted, expected_test, ["cell_id", "split", "tile", "x_um", "y_um"], "Predictions " + str(key))
                equal_strings(predicted.truth, expected_test[column].astype(str), "Predicted test truth differs: " + str(key))
                ensure(set(predicted.prediction).issubset(train_classes), "Classifier predicted a class absent from training")
                probabilities = predicted.predicted_probability.to_numpy(dtype=float)
                ensure(np.isfinite(probabilities).all() and np.all((probabilities >= 0) & (probabilities <= 1)), "Invalid prediction confidence")
                for part, frame in part_frames.items():
                    ensure(row[part + "_cells"] == len(frame), "Metric cell count mismatch: " + str(key))
                ensure(row["classes"] == len(classes) and row["ordered_test_cell_ids_sha256"] == ids_sha256(predicted.cell_id),
                       "Metric class-count/test-ID contract differs")
                ensure(row["embedding_dimensions"] == dimensions[name], "Metric embedding dimension differs")
                selection = task_audit["selected_probes"][name]
                candidates = selection["candidates"]
                ensure([item["c"] for item in candidates] == audit["probe_c_grid"], "Validation C grid differs")
                best = max(candidates, key=lambda item: item["val_macro_f1"])
                close_equal(selection["selected_c"], best["c"], "Selected C not validation argmax")
                close_equal(row.selected_c, selection["selected_c"], "Reported selected C")
                close_equal(row.val_macro_f1, best["val_macro_f1"], "Reported validation F1")
                ensure(selection["scaler_fit_cells"] == len(part_frames["train"]), "Scaler training cell count differs")
                truth, prediction = predicted.truth.to_numpy(), predicted.prediction.to_numpy()
                recalculated = {
                    "test_macro_f1": float(f1_score(truth, prediction, labels=classes, average="macro", zero_division=0)),
                    "test_weighted_f1": float(f1_score(truth, prediction, labels=classes, average="weighted", zero_division=0)),
                    "test_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
                }
                if task_audit["kmeans_status"] == "ASSESSED":
                    cluster = predicted.kmeans_cluster.to_numpy(dtype=float)
                    ensure(np.isfinite(cluster).all() and np.all(cluster == np.floor(cluster)) and
                           np.all((cluster >= 0) & (cluster < len(classes))), "Invalid K-means cluster IDs")
                    recalculated["test_ari_k_known"] = float(adjusted_rand_score(truth, cluster.astype(np.int64)))
                    recalculated["test_nmi_k_known"] = float(normalized_mutual_info_score(truth, cluster.astype(np.int64)))
                else:
                    ensure(predicted.kmeans_cluster.isna().all() and len(expected_test) < len(classes), "Unexpected unassessed K-means")
                    recalculated["test_ari_k_known"] = None
                    recalculated["test_nmi_k_known"] = None
                for metric, value in recalculated.items():
                    close_equal(value, row[metric], "/".join(key) + "/" + metric)
                verified_metrics.append({"cohort": cohort_name, "task": task, "readout": name,
                                         "test_cells": len(predicted), **recalculated})
    actual_rows = set(map(tuple, metrics[["cohort", "task", "readout"]].to_numpy()))
    ensure(actual_rows == expected_rows, "Reported metric rows include missing/unexpected tasks or readouts")
    ensure(audit["cohorts"]["primary_all_expert_cells"]["tasks"]["expert_cell_type"]["status"] == "ASSESSED",
           "Primary expert annotation task not assessed")
    qa["evaluation"] = {"primary_readouts": list(PRIMARY), "primary_cells": len(frozen),
                        "primary_test_cell_type_cells": int((frozen.split == "test").sum()),
                        "nonoverlapping_spatial_tiles": True, "metric_rows_verified": len(verified_metrics),
                        "region_assessment_status": audit["region_labels"]["status"],
                        "recomputed_metrics": verified_metrics}
    qa["source_overlap"] = audit["source_overlap"]
    qa["verified_sha256"] = {str(path): digest for path, digest in hashes.items()}
    qa["status"] = "VISIUM_HD_CRC_BENCHMARK_QA_PASSED"
    qa["interpretation"] = "Artifact/count/metric verification passed; not proof of model superiority or held-out-source generalization"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True, help="Published benchmark results directory")
    parser.add_argument("--evaluator-code", type=Path, help="Exact evaluator file if unavailable next to benchmark project")
    parser.add_argument("--output", type=Path, help="Defaults to BENCHMARK_DIR/QA.json; refuses overwrite")
    parser.add_argument("--count-chunk-rows", type=int, default=2048)
    args = parser.parse_args()
    require_compute_node()
    ensure(args.count_chunk_rows > 0, "Count chunk rows must be positive")
    output = (args.output or args.benchmark_dir / "QA.json").resolve()
    ensure(not output.exists(), f"Refusing to overwrite existing QA: {output}")
    ensure(output.parent.is_dir(), "QA output parent directory does not exist")
    qa = {"status": "VALIDATION_FAILED", "benchmark_dir": str(args.benchmark_dir.resolve()),
          "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
          "validator_sha256": file_sha256(Path(__file__))}
    code = 0
    try:
        verify(args, qa)
    except Exception as error:
        code = 1
        qa["error_type"] = type(error).__name__
        qa["error"] = str(error)
        traceback.print_exc()
    # Exclusive creation ensures concurrent validators cannot replace evidence.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(qa, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": qa["status"], "qa_path": str(output), "exit_code": code}), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
