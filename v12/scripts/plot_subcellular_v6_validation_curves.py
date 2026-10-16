"""Plot the checkpoint-wise V6 validation curves from completed metrics.

Chart contract
--------------
Question: how do routes A and B change on the fixed validation protocol?
Takeaway surface: static research figure (PNG and PDF), plus a tidy CSV.
Panels: fixed-C joint macro-F1 for cell type and region, fixed masked CE,
and the region-F1 contribution of real versus shuffled neighbors.
The plot never reads test donors and never substitutes per-step best C.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


COLORS = {"A": "#2463A8", "B": "#D97706"}
STYLES = {
    "A": dict(color=COLORS["A"], linestyle="-", marker="o"),
    "B": dict(color=COLORS["B"], linestyle="--", marker="s"),
}


def fixed_score(report: dict, embedding: str, task: str) -> float:
    rows = [row for row in report["results"]
            if row["embedding"] == embedding and row["task"] == task]
    if len(rows) != 1:
        raise ValueError(f"Expected one {embedding}/{task} row")
    row = rows[0]
    expected_c = 0.1 if task == "cell_type" else 0.01
    fixed = row["fixed_C_result"]
    if fixed["C"] != expected_c:
        raise ValueError(f"Unexpected fixed C for {embedding}/{task}")
    if fixed.get("solver", {}).get("converged_gradient") is not True:
        raise ValueError(f"Probe did not converge for {embedding}/{task}")
    return float(fixed["val_macro_f1"])


def read_route(root: Path, route: str) -> list[dict]:
    run = root / "outputs" / f"subcellular_lr_v6_{route.lower()}_20260920"
    rows = []
    for path in sorted((run / "downstream_validation").glob("step_*/metrics.json")):
        report = json.loads(path.read_text())
        if report.get("test_donors_used") is not False:
            raise ValueError(f"Expected validation-only report: {path}")
        if report.get("canonical_cohort") is not True or report.get("cells") != 5200:
            raise ValueError(f"Unexpected evaluation cohort: {path}")
        joint_cell = fixed_score(report, "joint", "cell_type")
        joint_region = fixed_score(report, "joint", "region")
        shuffled_region = fixed_score(report, "joint_neighbor_shuffle", "region")
        rows.append(dict(
            route=route,
            step=int(report["step"]),
            joint_cell_type_macro_f1=joint_cell,
            joint_region_macro_f1=joint_region,
            base_cell_type_macro_f1=fixed_score(report, "base", "cell_type"),
            base_region_macro_f1=fixed_score(report, "base", "region"),
            masked_ce=float(report["masked_validation"]["loss_mean"]["masked_ce"]),
            real_minus_shuffled_region_macro_f1=joint_region - shuffled_region,
            checkpoint_sha256=report["checkpoint_sha256"],
            metrics_path=str(path),
        ))
    if not rows or rows[0]["step"] != 0:
        raise ValueError(f"Route {route} lacks step-zero validation")
    if len({row["step"] for row in rows}) != len(rows):
        raise ValueError(f"Route {route} contains duplicate steps")
    return rows


def read_scgpt_spatial_baseline(path: Path, expected_cohort: str) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    if (report.get("model") != "Official scGPT-spatial V1"
            or report.get("cohort_sha256") != expected_cohort
            or report.get("test_donors_used") is not False
            or report.get("pooling") != "CLS L2 normalized"):
        raise ValueError("Official scGPT-spatial baseline identity/cohort mismatch")
    expected = {"cell_type": (4000, 1200, 0.1), "region": (781, 330, 0.01)}
    scores = {}
    for task, (train_cells, val_cells, fixed_c) in expected.items():
        rows = [row for row in report["results"] if row["task"] == task]
        if len(rows) != 1 or (rows[0]["train_cells"], rows[0]["val_cells"]) != (train_cells, val_cells):
            raise ValueError(f"Official scGPT-spatial {task} cohort mismatch")
        candidates = [row for row in rows[0]["grid"] if row["C"] == fixed_c]
        if len(candidates) != 1 or candidates[0].get("solver", {}).get("converged_gradient") is not True:
            raise ValueError(f"Official scGPT-spatial {task} fixed-C probe is invalid")
        scores[task] = float(candidates[0]["val_macro_f1"])
    return dict(model=report["model"], source_commit=report["source_commit"],
                weights_md5=report["weights_md5"], metrics_sha256=hashlib.sha256(raw).hexdigest(),
                cohort_sha256=report["cohort_sha256"], test_donors_used=False,
                pooling=report["pooling"], input=report["input"], binning=report["binning"],
                context=report["context"], fixed_C={"cell_type": 0.1, "region": 0.01},
                scores=scores, metrics_path=str(path))


def read_plain_scgpt_baseline(path: Path, expected_cohort: str, expected_tie_policy: str) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    tie_policy = report.get("tie_policy", report.get("collator_provenance", {}).get("tie_policy"))
    if tie_policy is None:
        tie_policy = "random"
    if (report.get("model") != "Official ordinary scGPT whole-human; frozen whole-cell encoder"
            or report.get("cohort_sha256") != expected_cohort
            or report.get("test_donors_read") is not False
            or report.get("cells") != 5200
            or tie_policy != expected_tie_policy):
        raise ValueError(f"Plain scGPT {expected_tie_policy} baseline identity/cohort mismatch")
    scores = {}
    for task, fixed_c in (("cell_type", 0.1), ("region", 0.01)):
        rows = [row for row in report["results"]
                if row["embedding"] == "cls" and row["task"] == task]
        if len(rows) != 1:
            raise ValueError(f"Plain scGPT {expected_tie_policy} lacks one CLS/{task} row")
        candidates = [row for row in rows[0]["grid"] if row["C"] == fixed_c]
        if len(candidates) != 1 or candidates[0].get("solver", {}).get("converged_gradient") is not True:
            raise ValueError(f"Plain scGPT {expected_tie_policy} {task} fixed-C probe is invalid")
        scores[task] = float(candidates[0]["val_macro_f1"])
    return dict(model=report["model"], tie_policy=tie_policy,
                metrics_sha256=hashlib.sha256(raw).hexdigest(),
                cohort_sha256=report["cohort_sha256"], test_donors_used=False,
                pooling=report["pooling"]["cls"], inference_dtype=report["inference_dtype"],
                fixed_C={"cell_type": 0.1, "region": 0.01}, scores=scores,
                metrics_path=str(path))


def setup_axis(axis, title: str, ylabel: str):
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)


def plot_series(axis, rows_by_route: dict[str, list[dict]], key: str):
    for route, rows in rows_by_route.items():
        steps, values = [], []
        previous = None
        for row in rows:
            if previous is not None and row["step"] - previous > 500:
                steps.append(math.nan)
                values.append(math.nan)
            steps.append(row["step"])
            values.append(row[key])
            previous = row["step"]
        axis.plot(steps, values,
                  label=f"Route {route}", linewidth=1.8, markersize=3.6,
                  markerfacecolor="white", markeredgewidth=1.1, **STYLES[route])


def annotate_latest(axis, rows_by_route: dict[str, list[dict]], key: str,
                    *, multiplier: float = 1.0, suffix: str = ""):
    latest_values = {route: rows[-1][key] for route, rows in rows_by_route.items()}
    upper = max(latest_values, key=latest_values.get)
    for route, rows in rows_by_route.items():
        last = rows[-1]
        vertical_offset = 7 if route == upper else -7
        axis.annotate(f"{last[key] * multiplier:.3f}{suffix}",
                      (last["step"], last[key]), xytext=(5, vertical_offset),
                      textcoords="offset points", va="center", fontsize=8,
                      color=COLORS[route], fontweight="bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--scgpt-spatial-metrics", type=Path)
    parser.add_argument("--scgpt-random-metrics", type=Path)
    parser.add_argument("--scgpt-deterministic-metrics", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows_by_route = {route: read_route(root, route) for route in ("A", "B")}
    baseline_path = (args.scgpt_spatial_metrics or
                     root / "resources/baselines/scgpt_spatial_v1_fixed5200_metrics.json").resolve()
    v6_step_zero = json.loads((
        root / "outputs/subcellular_lr_v6_a_20260920/downstream_validation/"
        "step_00000000/metrics.json").read_text())
    spatial_baseline = read_scgpt_spatial_baseline(
        baseline_path, v6_step_zero["cohort_sha256"])
    random_baseline = read_plain_scgpt_baseline(
        (args.scgpt_random_metrics or root / "outputs/subcellular_scgpt_human_4gpu/"
         "plain_whole_cell_baseline/metrics.json").resolve(),
        v6_step_zero["cohort_sha256"], "random")
    deterministic_baseline = read_plain_scgpt_baseline(
        (args.scgpt_deterministic_metrics or root / "outputs/subcellular_scgpt_human_4gpu/"
         "plain_whole_cell_baseline_ties_left/metrics.json").resolve(),
        v6_step_zero["cohort_sha256"], "left")
    all_rows = [row for route in ("A", "B") for row in rows_by_route[route]]
    for row in all_rows:
        row["scgpt_spatial_cell_type_macro_f1"] = spatial_baseline["scores"]["cell_type"]
        row["scgpt_spatial_region_macro_f1"] = spatial_baseline["scores"]["region"]
        row["scgpt_random_cell_type_macro_f1"] = random_baseline["scores"]["cell_type"]
        row["scgpt_random_region_macro_f1"] = random_baseline["scores"]["region"]
        row["scgpt_deterministic_cell_type_macro_f1"] = deterministic_baseline["scores"]["cell_type"]
        row["scgpt_deterministic_region_macro_f1"] = deterministic_baseline["scores"]["region"]
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelcolor": "#26313D",
        "text.color": "#26313D",
        "xtick.color": "#52606D",
        "ytick.color": "#52606D",
    })
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.86, bottom=0.13,
                        wspace=0.20, hspace=0.34)
    fig.patch.set_facecolor("white")
    fig.suptitle("V6 checkpoint-wise validation curves", x=0.06, y=0.98, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(0.06, 0.935,
             "Fixed 5,200-cell cohort; fixed C (cell type 0.1, region 0.01); step 7500 is a priority endpoint; no test donors",
             ha="left", fontsize=9, color="#52606D")

    cell_ax, region_ax, ce_ax, gain_ax = axes.flat
    plot_series(cell_ax, rows_by_route, "joint_cell_type_macro_f1")
    base_cell = rows_by_route["A"][0]["base_cell_type_macro_f1"]
    cell_ax.axhline(base_cell, color="#9CA3AF", linestyle=":", linewidth=1.2,
                    label=f"V6 paired base ({base_cell:.3f})")
    cell_ax.axhline(random_baseline["scores"]["cell_type"], color="#6B7280",
                    linestyle="--", linewidth=1.3,
                    label=f"scGPT official random ({random_baseline['scores']['cell_type']:.3f})")
    cell_ax.axhline(deterministic_baseline["scores"]["cell_type"], color="#111827",
                    linestyle="-", linewidth=1.5,
                    label=f"scGPT deterministic ({deterministic_baseline['scores']['cell_type']:.3f})")
    cell_ax.axhline(spatial_baseline["scores"]["cell_type"], color="#111827",
                    linestyle=(0, (6, 2, 1, 2)), linewidth=1.5,
                    label=f"Official scGPT-spatial V1 ({spatial_baseline['scores']['cell_type']:.3f})")
    setup_axis(cell_ax, "A  Cell-type probe", "Validation macro-F1")
    annotate_latest(cell_ax, rows_by_route, "joint_cell_type_macro_f1")

    plot_series(region_ax, rows_by_route, "joint_region_macro_f1")
    base_region = rows_by_route["A"][0]["base_region_macro_f1"]
    region_ax.axhline(base_region, color="#9CA3AF", linestyle=":", linewidth=1.2,
                      label=f"V6 paired base ({base_region:.3f})")
    region_ax.axhline(random_baseline["scores"]["region"], color="#6B7280",
                      linestyle="--", linewidth=1.3,
                      label=f"scGPT official random ({random_baseline['scores']['region']:.3f})")
    region_ax.axhline(deterministic_baseline["scores"]["region"], color="#111827",
                      linestyle="-", linewidth=1.5,
                      label=f"scGPT deterministic ({deterministic_baseline['scores']['region']:.3f})")
    region_ax.axhline(spatial_baseline["scores"]["region"], color="#111827",
                      linestyle=(0, (6, 2, 1, 2)), linewidth=1.5,
                      label=f"Official scGPT-spatial V1 ({spatial_baseline['scores']['region']:.3f})")
    setup_axis(region_ax, "B  Region probe", "Validation macro-F1")
    annotate_latest(region_ax, rows_by_route, "joint_region_macro_f1")

    plot_series(ce_ax, rows_by_route, "masked_ce")
    ce_ax.set_yscale("log")
    ce_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    setup_axis(ce_ax, "C  Masked-expression reconstruction", "Masked CE (log scale; lower is better)")
    annotate_latest(ce_ax, rows_by_route, "masked_ce")

    plot_series(gain_ax, rows_by_route, "real_minus_shuffled_region_macro_f1")
    gain_ax.axhline(0, color="#4B5563", linewidth=1.0)
    setup_axis(gain_ax, "D  Spatial-neighbor contribution to region probe",
               "Real − shuffled-neighbor macro-F1")
    gain_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f} pp"))
    annotate_latest(gain_ax, rows_by_route, "real_minus_shuffled_region_macro_f1",
                    multiplier=100, suffix=" pp")

    for axis in axes.flat:
        axis.margins(x=0.05, y=0.12)
        axis.legend(frameon=False, fontsize=8, loc="best")
    fig.text(0.06, 0.025,
             "Cell type n=1,200; region n=330. scGPT-spatial has no V6 masked decoder or inference-time neighbor branch (C/D: N/A).",
             ha="left", fontsize=8, color="#52606D")

    metadata = {
        "schema": "subcellular_v6_validation_curve_v1",
        "routes": {route: {"steps": [row["step"] for row in rows],
                            "latest_step": rows[-1]["step"]}
                   for route, rows in rows_by_route.items()},
        "test_donors_used": False,
        "cohort_cells": 5200,
        "cell_type_validation_cells": 1200,
        "region_validation_cells": 330,
        "fixed_C": {"cell_type": 0.1, "region": 0.01},
        "plain_scgpt_random_baseline": random_baseline,
        "plain_scgpt_deterministic_baseline": deterministic_baseline,
        "scgpt_spatial_baseline": spatial_baseline,
        "csv": str(prefix.with_suffix(".csv")),
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig.savefig(prefix.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(json.dumps(metadata, allow_nan=False))


if __name__ == "__main__":
    main()
