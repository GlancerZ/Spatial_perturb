#!/usr/bin/env python3
"""Same-cell CRC expert-label probe for frozen V12 and scGPT-spatial V1.

This is a single-slide spatial-block technical benchmark. The official
scGPT-spatial pretraining source manifest includes this CRC slide, so this is
not a held-out-patient or held-out-source generalization test. Fit the probe
on training blocks, select C on validation blocks, and score test blocks once.
No author-predicted cell types enter the labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import warnings


READOUTS = ("nucleus", "cytoplasm", "local", "spatial", "fused", "concat",
            "scgpt_total_cls")
PRIMARY = ("v12_scgpt_total_cls", "v12_local", "v12_spatial", "v12_fused",
           "v12_concat", "scgpt_spatial_v1_cls")
EXPERT_CLASSES = ("epithelial cells", "immune cells", "stromal cells")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha256(ids) -> str:
    return hashlib.sha256(("\n".join(map(str, ids)) + "\n").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v12-h5ad", type=Path, required=True)
    parser.add_argument("--spatial-v1-npz", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-score-cells", type=Path, required=True,
                        help="Original CRC benchmark membership; never search a new split")
    parser.add_argument("--tile-um", type=float, default=500.)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run inside a verified Slurm compute-node step")
    if args.tile_um <= 100 or args.seed < 0:
        raise ValueError("Spatial tile size must exceed the 100 um V12 neighbor radius")

    import anndata as ad
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                                 f1_score, normalized_mutual_info_score)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cells = ad.read_h5ad(args.v12_h5ad)
    if not cells.obs_names.is_unique or "spatial_um" not in cells.obsm:
        raise ValueError("V12 cell IDs or coordinates invalid")
    labels = pd.read_csv(args.labels, dtype={"cell_id": str}).set_index("cell_id")
    if not labels.index.is_unique or "cell_type" not in labels:
        raise ValueError("Duplicate expert label IDs or missing cell_type")
    missing = cells.obs_names.difference(labels.index)
    if len(missing):
        raise ValueError(f"V12 score cells lack expert labels: {missing[:5].tolist()}")
    labels = labels.loc[cells.obs_names].copy()
    labels.index.name = "cell_id"
    if "region_label" not in labels:
        labels["region_label"] = pd.NA
    labels["region_label"] = labels["region_label"].astype("string").str.strip().replace("", pd.NA)
    with np.load(args.spatial_v1_npz, allow_pickle=False) as external:
        ext_ids = external["cell_ids"].astype(str)
        official_embedding = np.asarray(external["embeddings"], dtype=np.float32)
    if not np.array_equal(ext_ids, np.asarray(cells.obs_names, dtype=str)):
        raise ValueError("scGPT-spatial and V12 must have identical ordered cell IDs")
    features = {"scgpt_spatial_v1_cls": official_embedding}
    availability = {"scgpt_spatial_v1_cls": np.ones(cells.n_obs, dtype=bool)}
    for name in READOUTS:
        key = "X_" + name
        if key not in cells.obsm or name + "_available" not in cells.obs:
            raise ValueError(f"Missing V12 readout or availability: {name}")
        flags = cells.obs[name + "_available"]
        if flags.isna().any() or not flags.isin([True, False]).all():
            raise ValueError(f"Invalid availability flags: {name}")
        features["v12_" + name] = np.asarray(cells.obsm[key], dtype=np.float32)
        availability["v12_" + name] = flags.to_numpy(dtype=bool)
    for name, value in features.items():
        if value.ndim != 2 or value.shape[0] != cells.n_obs or value.shape[1] < 1:
            raise ValueError(f"Invalid embedding shape: {name}: {value.shape}")
        if not np.isfinite(value[availability[name]]).all():
            raise ValueError(f"Nonfinite available embedding: {name}")
    coords = np.asarray(cells.obsm["spatial_um"], dtype=np.float64)
    if coords.shape != (cells.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError("Invalid physical coordinates")

    # Freeze the main cohort from expert labels, before inspecting probe scores.
    # A missing nuclear/cytoplasmic readout cannot remove a primary score cell.
    expert_known = labels.cell_type.isin(EXPERT_CLASSES).to_numpy()
    for name in PRIMARY:
        if not availability[name][expert_known].all():
            raise ValueError(f"Primary comparison must cover every expert-labeled input cell: {name}")
    if expert_known.sum() < 1000:
        raise ValueError("Too few expert-labeled cells for the primary benchmark")
    all_availability = pd.DataFrame({"cell_id": np.asarray(cells.obs_names, dtype=str),
                                     "expert_label_in_primary": expert_known})
    for name, flags in availability.items():
        all_availability[name + "_available"] = flags
    all_availability.to_csv(output / "availability.csv", index=False)
    input_cells = cells.n_obs
    labels = labels.iloc[np.flatnonzero(expert_known)].copy()
    coords = coords[expert_known]
    features = {name: value[expert_known] for name, value in features.items()}
    availability = {name: value[expert_known] for name, value in availability.items()}
    tile = np.floor(coords / args.tile_um).astype(np.int64)
    groups = np.asarray([f"{x}_{y}" for x, y in tile])
    if len(np.unique(groups)) < 10:
        raise ValueError("Too few spatial tiles for block-separated probe")

    tasks = {"expert_cell_type": labels.cell_type.astype(str).to_numpy()}
    regions = labels.region_label
    region_counts = {str(k): int(v) for k, v in regions.value_counts().items()}
    if regions.notna().any():
        tasks["annotated_region"] = regions.to_numpy(dtype=object)
        region_status = "PENDING_FROZEN_SPLIT_COVERAGE_CHECK"
    else:
        region_status = "NOT_ASSESSED_NO_MATCHED_LABELS"

    # The previous benchmark defines every score cell, label and block.
    # Reject even a harmless-looking cohort/split change before fitting probes.
    reference = pd.read_csv(args.frozen_score_cells, dtype={"cell_id": str})
    required = {"cell_id", "cell_type", "region_label", "split", "tile", "x_um", "y_um"}
    if not required.issubset(reference) or not reference.cell_id.is_unique:
        raise ValueError("Invalid original frozen CRC benchmark")
    if not np.array_equal(reference.cell_id.to_numpy(), labels.index.to_numpy(dtype=str)):
        raise ValueError("Score-cell IDs/order differ from the original CRC benchmark")
    for column in ("cell_type", "region_label"):
        if not np.array_equal(reference[column].fillna("").astype(str).to_numpy(),
                              labels[column].fillna("").astype(str).to_numpy()):
            raise ValueError(f"Frozen CRC labels changed: {column}")
    if not np.array_equal(reference.tile.astype(str).to_numpy(), groups) or not np.allclose(
            reference[["x_um", "y_um"]].to_numpy(dtype=float), coords, rtol=0, atol=1e-6):
        raise ValueError("Frozen CRC geometry or tile definition changed")
    membership = reference.split.astype(str).to_numpy()
    if set(membership) != {"train", "val", "test"}:
        raise ValueError("Original CRC split must contain exactly train/val/test")
    split = {name: np.flatnonzero(membership == name) for name in ("train", "val", "test")}
    selected_seed = None  # Membership is loaded, not newly randomized.
    tile_sets = {part: set(groups[indices]) for part, indices in split.items()}
    if any(tile_sets[a] & tile_sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Spatial tile overlap across splits")
    frozen = labels.reset_index()[["cell_id", "cell_type", "region_label"]].copy()
    frozen["split"] = membership
    frozen["tile"] = groups
    frozen["x_um"] = coords[:, 0]
    frozen["y_um"] = coords[:, 1]
    frozen.to_csv(output / "frozen_score_cells.csv", index=False)

    cohorts = {"primary_all_expert_cells": (np.ones(len(labels), dtype=bool), PRIMARY, output)}
    for view in ("nucleus", "cytoplasm"):
        cohorts[view + "_available_cells"] = (
            availability["v12_" + view],
            ("v12_" + view, "v12_fused", "scgpt_spatial_v1_cls"),
            output / "supplementary" / view)
    cohort_audits = {}
    rows = []

    def fit_probe(matrix, target, indices):
        scaler = StandardScaler().fit(matrix[indices["train"]])
        transformed = {part: scaler.transform(matrix[ids]) for part, ids in indices.items()}
        if not all(np.isfinite(value).all() for value in transformed.values()):
            raise ValueError("Nonfinite standardized embedding")
        classes = np.unique(target[pd.notna(target)].astype(str))
        candidates = []
        best = None
        for c in (.01, .1, 1., 10.):
            # A longer retry solves the same objective; do not silently score
            # an unconverged classifier or change C after inspecting test data.
            for max_iter in (2000, 10000):
                probe = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter,
                                           random_state=args.seed)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        probe.fit(transformed["train"], target[indices["train"]].astype(str))
                    break
                except ConvergenceWarning:
                    if max_iter == 10000:
                        raise
            prediction = probe.predict(transformed["val"])
            score = f1_score(target[indices["val"]].astype(str), prediction,
                             labels=classes, average="macro", zero_division=0)
            candidates.append({"c": c, "val_macro_f1": float(score), "max_iter": max_iter,
                               "iterations": [int(x) for x in probe.n_iter_]})
            if best is None or score > best[0]:
                best = (score, c, probe)
        predicted = best[2].predict(transformed["test"])
        confidence = best[2].predict_proba(transformed["test"]).max(axis=1)
        clusters = (KMeans(n_clusters=len(classes), n_init=10,
                           random_state=args.seed).fit_predict(transformed["test"])
                    if len(indices["test"]) >= len(classes) else None)
        return classes, best, candidates, predicted, confidence, clusters

    for cohort_name, (mask, names, root) in cohorts.items():
        root.mkdir(parents=True, exist_ok=True)
        cohort_frozen = frozen.loc[mask].copy()
        if root != output:
            cohort_frozen.to_csv(root / "frozen_score_cells.csv", index=False)
        (root / "cell_ids.txt").write_text("\n".join(cohort_frozen.cell_id) + "\n")
        cohort_audit = {
            "cells": int(mask.sum()), "coverage_of_primary": float(mask.mean()),
            "ordered_cell_ids_sha256": ids_sha256(cohort_frozen.cell_id),
            "readouts": list(names), "split_cells": {}, "coverage_by_split": {},
            "tasks": {}, "status": "COMPLETE",
        }
        for part, ids in split.items():
            cohort_audit["split_cells"][part] = int(mask[ids].sum())
            cohort_audit["coverage_by_split"][part] = float(mask[ids].mean())
        for task, target in tasks.items():
            known = pd.notna(target)
            classes = np.unique(target[known].astype(str))
            indices = {part: ids[known[ids] & mask[ids]] for part, ids in split.items()}
            class_counts = {part: {str(k): int(v) for k, v in
                                  pd.Series(target[ids]).value_counts().items()}
                            for part, ids in indices.items()}
            task_audit = {
                "class_counts": class_counts,
                "scoring_classes": [str(name) for name in classes],
                "absent_classes_by_split": {
                    part: [str(name) for name in classes if class_counts[part].get(name, 0) == 0]
                    for part in indices},
                "ordered_test_cell_ids_sha256": ids_sha256(frozen.cell_id.iloc[indices["test"]]),
                "selected_probes": {},
            }
            train_classes = set(class_counts["train"])
            task_audit["unseen_label_cells_by_split"] = {
                part: sum(count for name, count in counts.items() if name not in train_classes)
                for part, counts in class_counts.items()}
            task_audit["unseen_test_class_counts"] = {
                name: count for name, count in class_counts["test"].items() if name not in train_classes}
            task_audit["kmeans_status"] = "NOT_ASSESSED_TASK_COVERAGE_GATE"
            cohort_audit["tasks"][task] = task_audit
            insufficient_expert = task == "expert_cell_type" and any(
                class_counts[part].get(name, 0) < 5 for part in indices for name in classes)
            insufficient_region = task == "annotated_region" and (
                len(train_classes) < 2 or any(len(ids) == 0 for ids in indices.values()))
            if insufficient_expert or insufficient_region:
                if cohort_name == "primary_all_expert_cells":
                    if insufficient_expert:
                        raise ValueError("Frozen primary split lost expert cell-type coverage")
                    region_status = "NOT_ASSESSED_INSUFFICIENT_REGION_LABELS_IN_FROZEN_SPLIT"
                task_audit["status"] = (
                    "NOT_ASSESSED_INSUFFICIENT_EXPERT_CLASS_COVERAGE_IN_FROZEN_SPLIT"
                    if insufficient_expert else "NOT_ASSESSED_INSUFFICIENT_REGION_LABELS_IN_FROZEN_SPLIT")
                cohort_audit["status"] = "PARTIAL_OR_NOT_ASSESSED"
                continue
            task_audit["status"] = "ASSESSED"
            task_audit["kmeans_status"] = (
                "ASSESSED" if len(indices["test"]) >= len(classes)
                else "NOT_ASSESSED_TEST_CELLS_FEWER_THAN_K")
            if task == "annotated_region" and cohort_name == "primary_all_expert_cells":
                region_status = "ASSESSED_INCLUDING_TRAIN_UNSEEN_CLASSES"
            for model_name in names:
                matrix = features[model_name]
                classes, best, candidates, predicted, confidence, clusters = fit_probe(matrix, target, indices)
                truth = target[indices["test"]].astype(str)
                row = {
                    "cohort": cohort_name, "task": task, "readout": model_name,
                    "train_cells": len(indices["train"]), "val_cells": len(indices["val"]),
                    "test_cells": len(indices["test"]), "classes": len(classes),
                    "embedding_dimensions": matrix.shape[1],
                    "selected_c": best[1], "val_macro_f1": float(best[0]),
                    "test_macro_f1": f1_score(truth, predicted, labels=classes,
                                               average="macro", zero_division=0),
                    "test_weighted_f1": f1_score(truth, predicted, labels=classes,
                                                  average="weighted", zero_division=0),
                    "test_balanced_accuracy": balanced_accuracy_score(truth, predicted),
                    "test_ari_k_known": adjusted_rand_score(truth, clusters) if clusters is not None else float("nan"),
                    "test_nmi_k_known": normalized_mutual_info_score(truth, clusters) if clusters is not None else float("nan"),
                    "ordered_test_cell_ids_sha256": task_audit["ordered_test_cell_ids_sha256"],
                }
                rows.append(row)
                predictions = frozen.iloc[indices["test"]][["cell_id", "split", "tile", "x_um", "y_um"]].copy()
                predictions["truth"] = truth
                predictions["prediction"] = predicted
                predictions["predicted_probability"] = confidence
                predictions["kmeans_cluster"] = clusters
                predictions.to_csv(root / (task + "__" + model_name + "__test_predictions.csv"), index=False)
                task_audit["selected_probes"][model_name] = {
                    "selected_c": best[1], "val_macro_f1": float(best[0]),
                    "candidates": candidates, "scaler_fit_cells": len(indices["train"]),
                }
                print(f"[probe] {cohort_name}/{task}/{model_name}: test n={len(truth)} macro-F1={row['test_macro_f1']:.6f}", flush=True)
        if root != output:
            pd.DataFrame([row for row in rows if row["cohort"] == cohort_name],
                         columns=list(rows[0])).to_csv(root / "metrics.csv", index=False)
            (root / "audit.json").write_text(json.dumps(cohort_audit, indent=2) + "\n")
        cohort_audits[cohort_name] = cohort_audit

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics.csv", index=False)
    primary_metrics = metrics.loc[metrics.cohort == "primary_all_expert_cells"].copy()
    primary_metrics.to_csv(output / "primary_metrics.csv", index=False)
    assessed_tasks = [name for name, task_audit in cohort_audits["primary_all_expert_cells"]["tasks"].items()
                      if task_audit["status"] == "ASSESSED"]
    if len(primary_metrics) != len(PRIMARY) * len(assessed_tasks):
        raise ValueError("Missing primary readout/task metric")
    for task, frame in primary_metrics.groupby("task"):
        if frame.ordered_test_cell_ids_sha256.nunique() != 1:
            raise ValueError(f"Primary readouts scored different cells: {task}")
    fig, axes = plt.subplots(1, len(assessed_tasks), figsize=(6.8 * len(assessed_tasks), 5), squeeze=False)
    for axis, task in zip(axes[0], assessed_tasks):
        subset = primary_metrics.loc[primary_metrics.task == task].sort_values("test_macro_f1")
        colors = ["#CB8B2A" if name == "scgpt_spatial_v1_cls" else
                  "#8B929B" if name == "v12_scgpt_total_cls" else "#337DB0"
                  for name in subset.readout]
        axis.barh(subset.readout, subset.test_macro_f1, color=colors)
        for i, score in enumerate(subset.test_macro_f1):
            axis.text(min(float(score) + .014, .96), i, f"{score:.3f}", va="center", fontsize=9)
        axis.set(title=task.replace("_", " "), xlabel="spatial-block test macro-F1", xlim=(0, 1))
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Visium HD CRC: frozen embeddings, expert-matched cells", fontsize=12)
    fig.text(.5, .015, "Single slide; scGPT-spatial pretraining source overlap. Regional labels are marker assisted.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(output / "matched_comparison.png", dpi=170)
    fig.savefig(output / "matched_comparison.pdf")
    plt.close(fig)
    audit = {
        "status": "MATCHED_CRC_BENCHMARK_COMPLETE",
        "cohort": "one human CRC Visium HD slide; same segmented cells and original counts",
        "inference": "both encoders frozen; V12 additionally reads nuclear/cytoplasmic bins and unlabeled neighbors",
        "expert_cell_type": "ENACT QuPath cell polygons; 3 broad classes; no model-generated type labels",
        "region_labels": {
            "source": "SpaceHack2023 2 um bins; H&E and marker-assisted annotation",
            "status": region_status, "matched_cells": int(regions.notna().sum()),
            "class_counts": region_counts,
        },
        "source_overlap": {
            "official_scgpt_spatial_pretraining_overlap": True,
            "evidence": "SpatialHuman30M source manifest Visium_HD sheet lists Visium_HD_Human_Colorectal_Cancer_FFPE and the exact 10x v4 CRC dataset URL",
            "dataset_url": "https://www.10xgenomics.com/datasets/visium-hd-cytassist-gene-expression-libraries-of-human-crc-v4",
            "interpretation": "Technical/within-slide frozen-probe benchmark; not a held-out-source zero-shot test or patient-generalization estimate",
        },
        "split": "nonoverlapping spatial tiles; no spatial buffer; not patient-independent; unlabeled inference neighbors may cross split boundaries",
        "tile_um": args.tile_um, "requested_seed": args.seed,
        "selected_coverage_seed": selected_seed, "coverage_seed_search_count": 0,
        "split_method": "external_original_frozen_membership_no_new_split_search",
        "minimum_expert_class_cells_each_split": 5,
        "split_coverage_task": "expert_cell_type only; region labels do not affect seed selection",
        "region_scoring": "Retain every region-labeled cell in the frozen split. Macro-F1 classes are the full true-label union; classes absent from training retain zero recall when unpredicted. Do not remove rare/unseen classes or select a new split.",
        "region_assessment_gate": "At least two training region classes and at least one labeled cell in every split; otherwise explicit NOT_ASSESSED",
        "input_cells": input_cells, "primary_cells": len(labels),
        "input_cells_without_valid_expert_type": int(input_cells - len(labels)),
        "tile_counts": {part: len(values) for part, values in tile_sets.items()},
        "cohorts": cohort_audits,
        "primary_readouts": list(PRIMARY),
        "protocol": "Complete primary expert cohort. Each compartment-only readout is compared to fused and official models on its available subset, retaining the primary block split.",
        "metric": "test macro-F1 primary; C selected on validation only; scaler and balanced logistic regression fit on training cells only",
        "probe_c_grid": [.01, .1, 1., 10.],
        "kmeans": "test-only unsupervised; K fixed to known number of classes; 10 initializations; one random seed, no uncertainty interval",
        "probe_and_kmeans_seed": args.seed,
        "source_sha256": {"v12_h5ad": sha256(args.v12_h5ad),
                          "frozen_score_cells": sha256(args.frozen_score_cells),
                          "spatial_v1_npz": sha256(args.spatial_v1_npz),
                          "labels_csv": sha256(args.labels),
                          "evaluator_code": sha256(Path(__file__))},
        "source_paths": {"v12_h5ad": str(args.v12_h5ad.resolve()),
                         "frozen_score_cells": str(args.frozen_score_cells.resolve()),
                         "spatial_v1_npz": str(args.spatial_v1_npz.resolve()),
                         "labels_csv": str(args.labels.resolve())},
        "output_sha256": {name: sha256(output / name) for name in
                          ("frozen_score_cells.csv", "metrics.csv", "primary_metrics.csv", "availability.csv")},
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": host,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    (output / "COMPLETE.json").write_text(json.dumps({
        "status": "MATCHED_CRC_BENCHMARK_COMPLETE", "primary_metric_rows": len(primary_metrics),
        "all_metric_rows": len(metrics), "primary_cells": len(labels),
        "audit_sha256": sha256(output / "audit.json"),
    }, indent=2) + "\n")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
