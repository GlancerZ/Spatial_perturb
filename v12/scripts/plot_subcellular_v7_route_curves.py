"""Draw separate, checkpoint-wise validation figures for V7 routes A and B.

The figures use the fixed 5,200-cell validation cohort and fixed-C probes.
They never read test donors or choose a per-checkpoint best regularization value.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


ROUTE_COLORS = {"A": "#1769AA", "B": "#D97706"}
BASE_COLOR = "#6B7280"
SHUFFLE_COLOR = "#8B5CF6"
GRID_COLOR = "#D9E0E8"
CANONICAL_COHORT = "0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc"
BASELINE_STYLES = {
    "scGPT-spatial": {"color": "#00856A", "linestyle": "--"},
    "SpatialFormer": {"color": "#AD3379", "linestyle": "-."},
}


def read_baseline(path: Path, name: str, expected_model: str) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    if (report.get("model") != expected_model
            or report.get("cohort_sha256") != CANONICAL_COHORT
            or report.get("test_donors_used") is not False):
        raise ValueError(f"Baseline identity/cohort mismatch: {path}")
    scores = {}
    for task, train_n, val_n, fixed_c in (
        ("cell_type", 4000, 1200, 0.1), ("region", 781, 330, 0.01)
    ):
        matches = [r for r in report["results"] if r["task"] == task]
        if len(matches) != 1:
            raise ValueError(f"Missing or duplicate baseline task: {path}/{task}")
        row = matches[0]
        if (row["train_cells"], row["val_cells"]) != (train_n, val_n):
            raise ValueError(f"Baseline task population mismatch: {path}/{task}")
        candidates = [r for r in row["grid"] if r["C"] == fixed_c]
        if (len(candidates) != 1
                or candidates[0].get("solver", {}).get("converged_gradient") is not True):
            raise ValueError(f"Invalid fixed-C baseline: {path}/{task}")
        value = float(candidates[0]["val_macro_f1"])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid baseline F1: {path}/{task}")
        scores[task] = value
    return dict(name=name, model=expected_model, scores=scores,
                metrics_path=str(path), metrics_sha256=hashlib.sha256(raw).hexdigest(),
                cohort_sha256=CANONICAL_COHORT, test_donors_used=False,
                pooling=report.get("pooling"),
                fixed_C={"cell_type": 0.1, "region": 0.01})


def draw_baselines(axis, baselines: list[dict], task: str) -> None:
    ordered = sorted(baselines, key=lambda b: b["scores"][task])
    previous = None
    for baseline in ordered:
        name, value = baseline["name"], baseline["scores"][task]
        style = BASELINE_STYLES[name]
        axis.axhline(value, linewidth=1.5, zorder=2, **style)
        offset = 4 if previous is None or value - previous > 0.014 else 13
        axis.annotate(
            f"{name}: {value:.4f}", xy=(0.60 if name == "SpatialFormer" else 0.02, value),
            xycoords=axis.get_yaxis_transform(), xytext=(0, offset),
            textcoords="offset points", fontsize=8, color=style["color"],
            fontweight="bold", ha="left", va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
        )
        previous = value


def fixed_score(report: dict, embedding: str, task: str) -> float:
    rows = [
        row
        for row in report["results"]
        if row["embedding"] == embedding and row["task"] == task
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one {embedding}/{task} row")
    expected_c = 0.1 if task == "cell_type" else 0.01
    fixed = rows[0]["fixed_C_result"]
    if fixed["C"] != expected_c:
        raise ValueError(f"Unexpected C for {embedding}/{task}: {fixed['C']}")
    if fixed.get("solver", {}).get("converged_gradient") is not True:
        raise ValueError(f"Probe did not converge for {embedding}/{task}")
    return float(fixed["val_macro_f1"])


def read_route(root: Path, route: str) -> list[dict]:
    run = root / "outputs" / f"subcellular_lr_v7_{route.lower()}_20260921"
    rows: list[dict] = []
    for path in (run / "downstream_validation").glob("step_*/metrics.json"):
        report = json.loads(path.read_text())
        if report.get("test_donors_used") is not False:
            raise ValueError(f"Expected validation-only result: {path}")
        if (report.get("canonical_cohort") is not True or report.get("cells") != 5200
                or report.get("cohort_sha256") != CANONICAL_COHORT):
            raise ValueError(f"Unexpected cohort: {path}")
        rows.append(
            {
                "route": route,
                "step": int(report["step"]),
                "base_cell": fixed_score(report, "base", "cell_type"),
                "joint_cell": fixed_score(report, "joint", "cell_type"),
                "shuffle_cell": fixed_score(
                    report, "joint_neighbor_shuffle", "cell_type"
                ),
                "base_region": fixed_score(report, "base", "region"),
                "joint_region": fixed_score(report, "joint", "region"),
                "shuffle_region": fixed_score(
                    report, "joint_neighbor_shuffle", "region"
                ),
                "masked_ce": float(report["masked_validation"]["loss_mean"]["masked_ce"]),
                "metrics_path": str(path),
            }
        )
    rows.sort(key=lambda row: row["step"])
    if len(rows) != 107 or rows[0]["step"] != 0 or rows[-1]["step"] != 10269:
        raise ValueError(
            f"Route {route} is incomplete: count={len(rows)}, "
            f"range={rows[0]['step'] if rows else None}..{rows[-1]['step'] if rows else None}"
        )
    if len({row["step"] for row in rows}) != len(rows):
        raise ValueError(f"Route {route} contains duplicate steps")
    return rows


def moving_average(values: list[float], width: int = 5) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    radius = width // 2
    result = np.empty_like(array)
    for index in range(len(array)):
        left = max(0, index - radius)
        right = min(len(array), index + radius + 1)
        result[index] = array[left:right].mean()
    return result


def padded_limits(values: list[float], *, minimum_padding: float = 0.01) -> tuple[float, float]:
    low, high = min(values), max(values)
    padding = max((high - low) * 0.10, minimum_padding)
    return low - padding, high + padding


def setup_axis(axis, title: str, ylabel: str) -> None:
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=8)
    axis.set_xlabel("Optimizer update")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(labelsize=8)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))


def plot_raw_and_smooth(axis, steps, values, *, label: str, color: str, marker: str) -> None:
    axis.plot(
        steps,
        values,
        color=color,
        linewidth=0.75,
        alpha=0.32,
        marker=marker,
        markersize=2.2,
        markeredgewidth=0,
    )
    axis.plot(
        steps,
        moving_average(values),
        color=color,
        linewidth=2.1,
        label=f"{label} (5-point mean)",
    )


def annotate_endpoint(axis, step: int, value: float, color: str, *, offset: int = 0) -> None:
    axis.scatter([step], [value], s=24, color=color, edgecolor="white", linewidth=0.7, zorder=5)
    axis.annotate(
        f"{value:.3f}",
        (step, value),
        xytext=(-8, 8 + offset),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=color,
    )


def draw_route(
    route: str,
    rows: list[dict],
    shared_limits: dict[str, tuple[float, float]],
    output_dir: Path,
    baselines: list[dict],
) -> dict:
    color = ROUTE_COLORS[route]
    steps = [row["step"] for row in rows]
    start, end = rows[0], rows[-1]
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.7))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.085, right=0.96, top=0.86, bottom=0.12, wspace=0.23, hspace=0.34)
    fig.suptitle(
        f"Route {route}: checkpoint-wise validation across one epoch",
        x=0.06,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.93,
        "107 checkpoints  |  fixed 5,200-cell validation cohort  |  fixed-C probes  |  no test donors",
        ha="left",
        fontsize=9,
        color="#52606D",
    )

    cell_ax, region_ax, spatial_ax, ce_ax = axes.flat

    plot_raw_and_smooth(
        cell_ax, steps, [row["base_cell"] for row in rows],
        label="Base", color=BASE_COLOR, marker="o"
    )
    plot_raw_and_smooth(
        cell_ax, steps, [row["joint_cell"] for row in rows],
        label="Joint", color=color, marker="o"
    )
    cell_ax.axhline(
        start["base_cell"], color="#111827", linestyle=":", linewidth=1.1,
        label=f"scGPT init ({start['base_cell']:.3f})"
    )
    setup_axis(cell_ax, "A  Cell identity", "Cell-type validation macro-F1")
    cell_ax.set_ylim(*shared_limits["cell"])
    draw_baselines(cell_ax, baselines, "cell_type")
    annotate_endpoint(cell_ax, end["step"], end["base_cell"], BASE_COLOR, offset=8)
    annotate_endpoint(cell_ax, end["step"], end["joint_cell"], color, offset=-22)

    plot_raw_and_smooth(
        region_ax, steps, [row["base_region"] for row in rows],
        label="Base", color=BASE_COLOR, marker="s"
    )
    plot_raw_and_smooth(
        region_ax, steps, [row["joint_region"] for row in rows],
        label="Joint", color=color, marker="s"
    )
    region_ax.axhline(
        start["base_region"], color="#111827", linestyle=":", linewidth=1.1,
        label=f"scGPT init ({start['base_region']:.3f})"
    )
    setup_axis(region_ax, "B  Spatial region", "Region validation macro-F1")
    region_ax.set_ylim(*shared_limits["region"])
    draw_baselines(region_ax, baselines, "region")
    annotate_endpoint(region_ax, end["step"], end["base_region"], BASE_COLOR, offset=-20)
    annotate_endpoint(region_ax, end["step"], end["joint_region"], color)

    plot_raw_and_smooth(
        spatial_ax, steps, [row["joint_region"] for row in rows],
        label="True neighbors", color=color, marker="o"
    )
    plot_raw_and_smooth(
        spatial_ax, steps, [row["shuffle_region"] for row in rows],
        label="Shuffled neighbors", color=SHUFFLE_COLOR, marker="x"
    )
    spatial_ax.fill_between(
        steps,
        [row["shuffle_region"] for row in rows],
        [row["joint_region"] for row in rows],
        color=color,
        alpha=0.08,
        linewidth=0,
        label="Observed neighbor contribution",
    )
    setup_axis(spatial_ax, "C  Does the real neighborhood matter?", "Region validation macro-F1")
    spatial_ax.set_ylim(*shared_limits["spatial"])
    annotate_endpoint(spatial_ax, end["step"], end["joint_region"], color)
    annotate_endpoint(spatial_ax, end["step"], end["shuffle_region"], SHUFFLE_COLOR, offset=-20)

    plot_raw_and_smooth(
        ce_ax, steps, [row["masked_ce"] for row in rows],
        label="Masked CE", color=color, marker="o"
    )
    setup_axis(ce_ax, "D  Masked-expression reconstruction", "Masked CE (lower is better)")
    ce_ax.set_yscale("log")
    ce_ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    annotate_endpoint(ce_ax, end["step"], end["masked_ce"], color)

    for axis in axes.flat:
        axis.set_xlim(-220, 10650)
        axis.legend(frameon=False, fontsize=7.7, loc="best")
    cell_ax.legend(frameon=False, fontsize=7.7, loc="upper left", bbox_to_anchor=(0.01, 0.84))
    region_ax.legend(frameon=False, fontsize=7.7, loc="upper left", bbox_to_anchor=(0.01, 0.87))

    cell_delta = end["joint_cell"] - end["base_cell"]
    region_delta = end["joint_region"] - end["base_region"]
    shuffle_delta = end["joint_region"] - end["shuffle_region"]
    fig.text(
        0.06,
        0.025,
        (
            f"Endpoint step 10,269: joint−base cell {cell_delta:+.3f}; "
            f"joint−base region {region_delta:+.3f}; "
            f"true−shuffled region {shuffle_delta:+.3f}. "
            "Faint marks are raw checkpoints; solid curves are centered 5-point means."
        ),
        ha="left",
        fontsize=8.4,
        color="#435160",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"subcellular_v7_route_{route.lower()}_validation_curves"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "route": route,
        "checkpoints": len(rows),
        "step_range": [rows[0]["step"], rows[-1]["step"]],
        "endpoint": end,
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
        "svg": str(stem.with_suffix(".svg")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scgpt-spatial-metrics", type=Path)
    parser.add_argument("--spatialformer-metrics", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    by_route = {route: read_route(root, route) for route in ("A", "B")}
    baselines = [
        read_baseline(
            (args.scgpt_spatial_metrics or root / "resources/baselines/scgpt_spatial_v1_fixed5200_metrics.json").resolve(),
            "scGPT-spatial", "Official scGPT-spatial V1",
        ),
        read_baseline(
            (args.spatialformer_metrics or root / "resources/baselines/spatialformer_single71_fixed5200_metrics.json").resolve(),
            "SpatialFormer", "Official SpatialFormer single71",
        ),
    ]

    all_rows = [row for route in ("A", "B") for row in by_route[route]]
    shared_limits = {
        "cell": padded_limits(
            [row[key] for row in all_rows for key in ("base_cell", "joint_cell")]
            + [b["scores"]["cell_type"] for b in baselines]
        ),
        "region": padded_limits(
            [row[key] for row in all_rows for key in ("base_region", "joint_region")]
            + [b["scores"]["region"] for b in baselines]
        ),
        "spatial": padded_limits(
            [row[key] for row in all_rows for key in ("joint_region", "shuffle_region")]
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "subcellular_v7_ab_validation_curves.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": "#26313D",
            "text.color": "#26313D",
            "xtick.color": "#52606D",
            "ytick.color": "#52606D",
        }
    )
    figures = [
        draw_route(route, by_route[route], shared_limits, output_dir, baselines)
        for route in ("A", "B")
    ]
    manifest = {
        "schema": "subcellular_v7_separate_route_curves_v2",
        "cohort_cells": 5200,
        "test_donors_used": False,
        "fixed_C": {"cell_type": 0.1, "region": 0.01},
        "shared_limits": shared_limits,
        "csv": str(csv_path),
        "figures": figures,
        "external_baselines": baselines,
        "baseline_panels": "Cell type and region only; masked CE and neighbor shuffle are V7-specific.",
    }
    manifest_path = output_dir / "subcellular_v7_ab_validation_curves.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
