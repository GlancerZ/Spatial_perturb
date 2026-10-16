#!/usr/bin/env python3
"""Validate completed same-cell evaluations and export scientific comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np
import pandas as pd

PRIMARY = ("scgpt_total_cls", "local", "spatial", "fused", "concat", "scgpt_spatial_v1")
TASKS = ("cell_type", "cell_group", "meta_niche_name")
LABELS = {"scgpt_total_cls": "scGPT (total CLS)", "local": "V12 local N+C",
          "spatial": "V12 spatial", "fused": "V12 fused",
          "concat": "V12 concat", "scgpt_spatial_v1": "scGPT-spatial V1"}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run in a verified Slurm compute step")
    if args.output_dir.exists():
        p.error("Refusing to overwrite summary")
    audit = json.loads((args.results / "comparison/audit.json").read_text())
    if audit["primary"]["cells"] != audit["frozen_cells"] or audit["frozen_cells"] != 7875:
        raise ValueError("Primary cohort lost frozen cells")
    groups = {"primary": (args.results / "evaluation", PRIMARY)}
    for view in ("nucleus", "cytoplasm"):
        groups[view] = (args.results / "evaluation/supplementary" / view,
                        (view, "fused", "scgpt_spatial_v1"))
    rows, source_hashes = [], {}
    for cohort, (root, names) in groups.items():
        for task in TASKS:
            reference = None
            reference_index_sha = None
            for name in names:
                path = root / name / task / "metrics.json"
                report = json.loads(path.read_text())
                predictions = pd.read_parquet(path.parent / "test_predictions.parquet")
                ids = predictions.full_cell_id.astype(str).tolist()
                if len(ids) != len(set(ids)) or len(ids) != report["split_cells"]["test"]:
                    raise ValueError(f"Invalid scored IDs: {cohort}/{name}/{task}")
                identity = predictions[["full_cell_id", "patient", task]].reset_index(drop=True)
                index_sha = report["input_sha256"]["index"]
                if reference is None:
                    reference, reference_index_sha = identity, index_sha
                elif not identity.equals(reference) or index_sha != reference_index_sha:
                    raise ValueError(f"Different scored cells/labels/cohort: {cohort}/{task}")
                if report["model"] != name or report["label_column"] != task:
                    raise ValueError("Metric identity mismatch")
                if any(set(report["split_patients"][a]) & set(report["split_patients"][b])
                       for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
                    raise ValueError("Patient split leakage")
                probe = report["test_linear_probe"]
                cluster = report["test_kmeans"]
                row = dict(cohort=cohort, model=name, task=task,
                           train_cells=report["split_cells"]["train"],
                           val_cells=report["split_cells"]["val"],
                           test_cells=report["split_cells"]["test"],
                           macro_f1=probe["macro_f1"], balanced_accuracy=probe["balanced_accuracy"],
                           accuracy=probe["accuracy"], ari_mean=cluster["ari_mean"],
                           ari_seed_std=cluster["ari_std"], selected_C=report["selected_C"],
                           embedding_dim=report["embedding_dimension"])
                if not np.isfinite([row[k] for k in ("macro_f1", "balanced_accuracy", "ari_mean")]).all():
                    raise ValueError("Nonfinite metrics")
                rows.append(row)
                source_hashes[str(path.relative_to(args.results))] = sha256(path)
    frame = pd.DataFrame(rows)
    if len(frame) != 36:
        raise ValueError("Expected 18 primary and 18 matched-view evaluations")
    delta_rows = []
    for task in TASKS:
        root = args.results / "evaluation"
        fused = pd.read_csv(root / "fused" / task / "test_by_patient.csv", dtype={"patient": str})
        baseline = pd.read_csv(root / "scgpt_spatial_v1" / task / "test_by_patient.csv", dtype={"patient": str})
        matched = fused.merge(baseline, on="patient", validate="one_to_one", suffixes=("_v12", "_official"))
        if len(matched) != 4 or not matched.cells_v12.equals(matched.cells_official):
            raise ValueError("Different test patient coverage")
        for row in matched.itertuples():
            delta_rows.append(dict(task=task, patient=row.patient, test_cells=row.cells_v12,
                v12_fused_f1=row.macro_f1_v12, scgpt_spatial_f1=row.macro_f1_official,
                f1_delta=row.macro_f1_v12-row.macro_f1_official))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "all_metrics.csv", index=False)
    frame[frame.cohort.eq("primary")].to_csv(args.output_dir / "primary_metrics.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(args.output_dir / "per_patient_fused_vs_official.csv", index=False)
    contract = {
        "question": "Do V12 and official scGPT-spatial differ on same-cell type and author niche annotation?",
        "renderer": "standalone Matplotlib PNG/PDF", "family": "horizontal bar small multiples",
        "data": "six primary readouts, two tasks, macro-F1 and ARI; all use identical ordered test cells per task",
        "palette": "blue V12 readouts, gold official scGPT-spatial, neutral scGPT reference; direct labels and outlines",
        "axis_policy": "zero-based, same F1/ARI domains across task panels",
        "uncertainty": "ARI error bars only SD over 3 fixed K-means seeds; no F1 CI or significance claim",
    }
    (args.output_dir / "chart_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    colors = ["#7d858e", "#91b1d6", "#bbd0e8", "#3779b5", "#6c9dca", "#b7862b"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.1))
    for i, task in enumerate(("cell_type", "meta_niche_name")):
        values = frame[frame.cohort.eq("primary") & frame.task.eq(task)].set_index("model").loc[list(PRIMARY)]
        for j, metric in enumerate(("macro_f1", "ari_mean")):
            axis = axes[i, j]
            x = values[metric].to_numpy()
            error = values.ari_seed_std.to_numpy() if metric == "ari_mean" else None
            axis.barh(np.arange(len(x)), x, color=colors, edgecolor="#46505a", linewidth=.6,
                      xerr=error, error_kw={"ecolor": "#46505a", "capsize": 2, "elinewidth": .8})
            axis.set_yticks(np.arange(len(x)), [LABELS[k] for k in PRIMARY])
            axis.invert_yaxis()
            axis.set_xlim(0, .60 if j == 0 else .45)
            axis.set_xlabel("Test macro-F1" if j == 0 else "Test K-means ARI (mean +/- seed SD)")
            task_title = "Cell type" if i == 0 else "Author meta niche"
            axis.set_title(f"{task_title} | test n = {int(values.test_cells.iloc[0]):,}", loc="left", fontsize=12, pad=10)
            axis.xaxis.grid(True, color="#e4e7eb", linewidth=.5)
            axis.set_axisbelow(True)
            for row, value in enumerate(x):
                offset = (error[row] if error is not None else 0) + .009
                axis.text(value + offset, row, f"{value:.3f}", va="center", fontsize=9)
    fig.suptitle("GSE316387 | frozen embedding comparison", x=.035, ha="left", fontsize=16)
    fig.text(.035, .025, "Same patient-disjoint split; 7,875 frozen cells. Author labels are expression-derived.\n"
             "V12 step 10,234; official scGPT-spatial V1. ARI seed SD is not a patient confidence interval.",
             fontsize=9, color="#56616c", va="bottom")
    fig.tight_layout(rect=(0, .085, 1, .955), h_pad=2.1, w_pad=2.5)
    fig.savefig(args.output_dir / "primary_comparison.png", dpi=180)
    fig.savefig(args.output_dir / "primary_comparison.pdf")
    plt.close(fig)
    report = dict(status="ALL_METRICS_AND_SAME_CELL_VALIDATION_COMPLETE", evaluations=len(frame),
                  primary_cells=audit["primary"]["cells"], matched_view_cohorts=audit["supplementary"],
                  source_metric_sha256=source_hashes,
                  slurm_job_id=os.environ["SLURM_JOB_ID"], hostname=host)
    (args.output_dir / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
