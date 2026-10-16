"""Audit paired validation cohorts and fixed-C probes using saved embeddings.

No encoder training, checkpoint changes, or held-out test data are used.
Endpoint alpha variants are exploratory readout diagnostics, not new models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler

from probe_lungpf_embeddings import fit_linear, predict

COLS = ["full_cell_id", "patient", "split", "final_CT", "region_label"]
COHORT = "0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_frame(frame):
    return hashlib.sha256(frame[COLS].fillna("").astype(str).to_csv(index=False).encode()).hexdigest()


def save(path, content):
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def fixed_probe(obs, features, task, name, output):
    column, c = ("final_CT", 0.1) if task == "cell_type" else ("region_label", 0.01)
    labels = obs[column]
    usable = labels.notna() & ~labels.astype(str).isin(["", "nan", "None"])
    train = usable & obs["split"].eq("train")
    val = usable & obs["split"].eq("val")
    classes = np.unique(labels[train].astype(str))
    lookup = {label: i for i, label in enumerate(classes)}
    scaler = StandardScaler().fit(features[train])
    tx = torch.as_tensor(scaler.transform(features[train]), device="cuda", dtype=torch.float32)
    vx = torch.as_tensor(scaler.transform(features[val]), device="cuda", dtype=torch.float32)
    y = torch.tensor([lookup[v] for v in labels[train].astype(str)], device="cuda")
    weights, bias, solver = fit_linear(tx, y, c)
    predicted = predict(vx, weights, bias, classes)
    actual = labels[val].astype(str).to_numpy()
    donors = obs.loc[val, "patient"].to_numpy()
    row = dict(name=name, task=task, C=c, train_cells=int(train.sum()), val_cells=int(val.sum()),
               val_macro_f1=float(f1_score(actual, predicted, labels=np.unique(actual),
                                          average="macro", zero_division=0)), solver=solver,
               per_donor_macro_f1={}, per_class=classification_report(
                   actual, predicted, labels=np.unique(actual), output_dict=True, zero_division=0))
    for donor in np.unique(donors):
        mask = donors == donor
        row["per_donor_macro_f1"][str(donor)] = float(f1_score(
            actual[mask], predicted[mask], labels=np.unique(actual[mask]), average="macro", zero_division=0))
    frame = obs.loc[val, ["full_cell_id", "patient"]].copy()
    frame["label"] = actual
    frame["prediction"] = predicted
    frame.to_parquet(output / f"{name}_{task}_fixed_C_predictions.parquet", index=False)
    print("[fixed-probe]", name, task, row["val_macro_f1"], flush=True)
    return row


def reference_score(report, embedding, task):
    row = next(r for r in report["results"] if r["task"] == task
               and (embedding is None or r.get("embedding") == embedding))
    c = 0.1 if task == "cell_type" else 0.01
    return next(g for g in row["grid"] if g["C"] == c)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--spatial-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Audit requires a new output directory: {args.output}")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    paths = {"spatial": args.spatial_eval,
             "initial": args.root / "outputs/subcellular_lr_v7_a_20260921/downstream_validation/step_00000000"}
    for route in ("A", "B"):
        paths[route] = args.root / f"outputs/subcellular_lr_v7_{route.lower()}_20260921/downstream_validation/step_00010269"
    reports, frames, matrices, sources = {}, {}, {}, {}
    for name, directory in paths.items():
        reports[name] = json.loads((directory / "metrics.json").read_text())
        frames[name] = pd.read_parquet(directory / "cells.parquet").reset_index(drop=True)
        frame = frames[name]
        ids_hash = hashlib.sha256("\n".join(frame.full_cell_id.astype(str)).encode()).hexdigest()
        assert len(frame) == 5200 and frame.full_cell_id.is_unique and ids_hash == COHORT
        assert set(frame["split"]) == {"train", "val"}
        assert reports[name]["test_donors_used"] is False
        assert frame.groupby("patient")["split"].nunique().max() == 1
        sources[name] = dict(directory=str(directory), metrics_sha256=sha(directory / "metrics.json"),
                             ids_sha256=ids_hash, labels_split_sha256=digest_frame(frame))
    obs = frames["initial"]
    comparisons = {}
    for name, frame in frames.items():
        mismatches = {}
        for col in COLS + ["sample"]:
            left, right = obs[col].astype("string").fillna("<NULL>"), frame[col].astype("string").fillna("<NULL>")
            mismatches[col] = int((left != right).sum())
        comparisons[name] = mismatches
        assert sum(mismatches.values()) == 0, (name, mismatches)

    tasks = {}
    for task, col in (("cell_type", "final_CT"), ("region", "region_label")):
        labels = obs[col]
        usable = labels.notna() & ~labels.astype(str).isin(["", "nan", "None"])
        counts = obs.loc[usable].groupby(["split", "patient", col], observed=True).size().rename("n").reset_index()
        tasks[task] = dict(counts=counts.to_dict("records"),
                           train_cells=int((usable & obs["split"].eq("train")).sum()),
                           val_cells=int((usable & obs["split"].eq("val")).sum()))

    variants = [("scGPT_spatial", "spatial", None), ("scGPT_initial", "initial", "base")]
    variants += [(f"{r}_{e}", r, e) for r in ("A", "B")
                 for e in ("base", "intrinsic", "joint", "joint_neighbor_shuffle")]
    report = dict(schema="v7_scgpt_spatial_gap_audit_v1", test_donors_used=False,
                  sources=sources, cohort_comparison=comparisons, task_populations=tasks,
                  fixed_C={"cell_type": 0.1, "region": 0.01}, results=[], reproduction=[],
                  interpretation="Fixed-C validation diagnostics on a previously inspected endpoint; not independent test or matched pretraining ablation.")
    for name, source, embedding in variants:
        path = paths[source] / ("embeddings.npy" if embedding is None else f"{embedding}_embeddings.npy")
        values = np.load(path, allow_pickle=False)
        assert values.ndim == 2 and len(values) == len(obs) and np.isfinite(values).all()
        matrices[name] = values
        sources[source].setdefault("embeddings", {})[name] = dict(path=str(path), sha256=sha(path), shape=list(values.shape))
        for task in tasks:
            result = fixed_probe(obs, values, task, name, args.output)
            expected = reference_score(reports[source], embedding, task)
            delta = abs(result["val_macro_f1"] - expected["val_macro_f1"])
            report["reproduction"].append(dict(name=name, task=task, f1_delta=delta, matches=delta < 1e-6))
            assert delta < 1e-6, (name, task, delta)
            report["results"].append(result)
        save(args.output / "progress.json", report)

    report["alpha_diagnostics"] = []
    report["residual_geometry"] = {}
    for route in ("A", "B"):
        base, joint = matrices[f"{route}_base"], matrices[f"{route}_joint"]
        delta = joint - base
        base_norm = np.linalg.norm(base, axis=1)
        ratio = np.linalg.norm(delta, axis=1) / np.maximum(base_norm, 1e-12)
        report["residual_geometry"][route] = dict(mean_ratio=float(ratio.mean()),
            quantiles=np.quantile(ratio, [0, .25, .5, .75, 1]).tolist())
        for alpha in (.1, .25, .5, .75):
            values = base + alpha * delta
            name = f"{route}_alpha_{alpha:g}"
            rows = [fixed_probe(obs, values, task, name, args.output) for task in tasks]
            report["alpha_diagnostics"].append(dict(route=route, alpha=alpha, results=rows))
            save(args.output / "progress.json", report)

    per_class = []
    scores = {(r["name"], r["task"]): r for r in report["results"]}
    for task in tasks:
        ref = scores["scGPT_spatial", task]["per_class"]
        for label, statistics in ref.items():
            if not isinstance(statistics, dict) or label in ("macro avg", "weighted avg", "micro avg"):
                continue
            row = dict(task=task, label=label, support=statistics["support"], scGPT_spatial=statistics["f1-score"])
            for name in ("scGPT_initial", "A_base", "A_joint", "B_base", "B_joint"):
                row[name] = scores[name, task]["per_class"][label]["f1-score"]
            row["spatial_minus_B_joint"] = row["scGPT_spatial"] - row["B_joint"]
            per_class.append(row)
    pd.DataFrame(per_class).to_csv(args.output / "per_class_f1.csv", index=False)
    report["runtime_seconds"] = time.monotonic() - started
    report["audit_script_sha256"] = sha(Path(__file__))
    report["solver_script_sha256"] = sha(Path(__file__).with_name("probe_lungpf_embeddings.py"))
    save(args.output / "audit.json", report)
    print("[done]", report["runtime_seconds"], args.output, flush=True)


if __name__ == "__main__":
    main()
