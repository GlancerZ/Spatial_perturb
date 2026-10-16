"""Render a point-in-time V7 training and evaluation status figure.

Chart contract
--------------
Question: are routes A/B making training progress, and is independent
checkpoint evaluation keeping pace?
Takeaway: training is healthy on four ranks, while evaluation coverage is
currently the bottleneck and neither allocation can finish the V7 epoch.
Surface: static research figure (PNG/PDF), tidy CSV, and provenance JSON.
The figure separates minibatch training loss from downstream evaluation
coverage; it never treats training loss as a representation-quality score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


COLORS = {"A": "#2463A8", "B": "#D97706"}
TOTAL_COLOR = "#D9E1EA"
PROJECTED_COLOR = "#AAB8C8"


def parse_duration(value: str) -> int:
    """Parse Slurm D-HH:MM:SS / HH:MM:SS / MM:SS durations."""
    value = value.strip()
    days = 0
    if "-" in value:
        day_text, value = value.split("-", 1)
        days = int(day_text)
    fields = [int(part) for part in value.split(":")]
    if len(fields) == 3:
        hours, minutes, seconds = fields
    elif len(fields) == 2:
        hours, (minutes, seconds) = 0, fields
    else:
        raise ValueError(f"Unsupported Slurm duration: {value}")
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def slurm_snapshot(jobid: str) -> dict:
    result = subprocess.run(
        ["squeue", "-h", "-j", jobid, "-o", "%T|%M|%L|%N"],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one live Slurm row for {jobid}: {rows}")
    state, elapsed, remaining, nodes = rows[0].split("|", 3)
    if state != "RUNNING":
        raise RuntimeError(f"Job {jobid} is not RUNNING: {state}")
    return {
        "jobid": jobid,
        "state": state,
        "elapsed": elapsed,
        "time_left": remaining,
        "remaining_seconds": parse_duration(remaining),
        "nodes": nodes,
    }


def read_train_rows(controller_dir: Path) -> list[dict]:
    by_step: dict[int, dict] = {}
    for path in sorted((controller_dir / "logs").glob("*training_step_*.log")):
        for line in path.read_text(errors="replace").splitlines():
            if not line.startswith("[train] "):
                continue
            row = json.loads(line[len("[train] "):])
            step = int(row["step"])
            loss = float(row["loss"]["masked_ce"])
            if not math.isfinite(loss):
                raise ValueError(f"Non-finite loss in {path} at step {step}")
            by_step[step] = {
                "step": step,
                "masked_ce": loss,
                "gradient_norm": float(row["gradient_norm"]),
                "seconds_per_update": float(row["seconds_per_update"]),
                "physical_world_size": int(row["physical_world_size"]),
                "global_batch_size": int(row["global_batch_size"]),
                "jobid": str(row["jobid"]),
                "source_log": str(path),
            }
    rows = [by_step[step] for step in sorted(by_step)]
    if not rows:
        raise ValueError(f"No training rows found in {controller_dir}")
    return rows


def rolling_mean(values: list[float], window: int) -> list[float]:
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(statistics.fmean(values[start:index + 1]))
    return result


def read_route(root: Path, route: str, eval_status: dict) -> dict:
    run = root / "outputs" / f"subcellular_lr_v7_{route.lower()}_20260921"
    progress = json.loads((run / "progress.json").read_text())
    schedule = json.loads((run / "schedule.json").read_text())
    controller = json.loads((Path(f"{run}.controller") / "status.json").read_text())
    rows = read_train_rows(Path(f"{run}.controller"))
    total_updates = int(schedule["total_updates"])
    current_step = int(progress["step"])
    if total_updates != 10269 or current_step > total_updates:
        raise ValueError(f"Unexpected route {route} progress: {current_step}/{total_updates}")
    if int(progress["physical_world_size"]) != 4 or int(controller["training_gpus"]) != 4:
        raise ValueError(f"Route {route} is not a verified four-rank run")
    if controller["status"] != "training" or controller.get("active_child", {}).get("status") != "running":
        raise ValueError(f"Route {route} controller is not actively training")
    if max(row["step"] for row in rows) < current_step:
        raise ValueError(f"Route {route} log trails checkpoint")
    jobid = str(controller["jobid"])
    slurm = slurm_snapshot(jobid)
    recent_speeds = [row["seconds_per_update"] for row in rows[-20:]
                     if row["seconds_per_update"] > 0]
    seconds_per_update = statistics.median(recent_speeds)
    usable_seconds = max(0, slurm["remaining_seconds"] - 300)
    projected_step = min(total_updates, current_step + int(usable_seconds / seconds_per_update))
    queue = eval_status["routes"][route]
    return {
        "route": route,
        "run": str(run),
        "rows": rows,
        "step": current_step,
        "total_updates": total_updates,
        "fraction": current_step / total_updates,
        "seconds_per_update": seconds_per_update,
        "projected_step": projected_step,
        "projected_fraction": projected_step / total_updates,
        "slurm": slurm,
        "published_steps": [int(value) for value in queue.get("published_steps", [])],
        "evaluated_steps": [int(value) for value in queue.get("evaluated_steps", [])],
        "queued_steps": [int(value) for value in queue.get("queued_steps", [])],
    }


def format_hours(seconds: int) -> str:
    return f"{seconds / 3600:.1f} h"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    eval_path = root / "outputs/subcellular_lr_v7_async_eval_20260921/status.json"
    eval_status = json.loads(eval_path.read_text())
    if eval_status.get("schema") != "subcellular_v7_async_evaluation_queue_v1":
        raise ValueError("Unexpected evaluation status schema")
    routes = {route: read_route(root, route, eval_status) for route in ("A", "B")}
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    tidy_rows = []
    for route, info in routes.items():
        for row in info["rows"]:
            tidy_rows.append({"route": route, **row})
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tidy_rows[0]))
        writer.writeheader()
        writer.writerows(tidy_rows)

    snapshot = {
        "schema": "subcellular_v7_live_figure_v1",
        "generated_at": generated_at,
        "training_loss_semantics": "logged_minibatch_masked_bin_cross_entropy",
        "projection_semantics": "linear_from_recent_median_seconds_per_update_to_allocation_end_minus_5_minutes",
        "evaluation_status": {
            "jobid": eval_status.get("jobid"),
            "status": eval_status.get("status"),
            "active": eval_status.get("active"),
            "updated_at": eval_status.get("updated_at"),
        },
        "routes": {
            route: {key: value for key, value in info.items() if key not in {"rows"}}
            for route, info in routes.items()
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(snapshot, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.labelcolor": "#26313D",
        "text.color": "#26313D",
        "xtick.color": "#52606D",
        "ytick.color": "#52606D",
        "axes.titlesize": 11,
    })
    fig = plt.figure(figsize=(12.2, 7.8), facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.55, 1.0), height_ratios=(1.0, 1.0),
                            left=0.07, right=0.96, top=0.84, bottom=0.17,
                            wspace=0.27, hspace=0.38)
    loss_ax = fig.add_subplot(grid[:, 0])
    progress_ax = fig.add_subplot(grid[0, 1])
    eval_ax = fig.add_subplot(grid[1, 1])

    fig.suptitle("Subcellular V7 live training status", x=0.07, y=0.965,
                 ha="left", fontsize=16, fontweight="bold")
    fig.text(0.07, 0.925,
             "Four-rank training is advancing; one-GPU evaluation is active; rolling two-GPU continuation is armed.",
             ha="left", fontsize=10, color="#52606D")
    fig.text(0.07, 0.892, f"Snapshot: {generated_at.replace('+00:00', 'Z')}  |  one V7 epoch = 10,269 optimizer updates",
             ha="left", fontsize=8.8, color="#6B7280")

    for route, info in routes.items():
        steps = [row["step"] for row in info["rows"]]
        losses = [row["masked_ce"] for row in info["rows"]]
        smoothed = rolling_mean(losses, window=10)
        loss_ax.plot(steps, losses, color=COLORS[route], alpha=0.18, linewidth=0.8)
        loss_ax.plot(steps, smoothed, color=COLORS[route], linewidth=2.1,
                     label=f"Route {route} · 100-step rolling mean")
        loss_ax.scatter([steps[-1]], [smoothed[-1]], color=COLORS[route], s=28, zorder=3)
        loss_ax.annotate(f"{route} {smoothed[-1]:.3f}", (steps[-1], smoothed[-1]),
                         xytext=(7, 7), textcoords="offset points",
                         color=COLORS[route], fontsize=8, fontweight="bold")
    loss_ax.set_title("A  Training objective", loc="left", fontweight="bold")
    loss_ax.set_xlabel("V7 optimizer update")
    loss_ax.set_ylabel("Masked-bin cross-entropy (logged minibatch)")
    loss_ax.grid(axis="both", color="#D8DEE6", linewidth=0.7, alpha=0.8)
    loss_ax.spines[["top", "right"]].set_visible(False)
    loss_ax.legend(frameon=False, loc="upper right", fontsize=8)
    loss_ax.text(0.01, 0.015,
                 "Faint: raw logged minibatches  |  Solid: rolling mean\nLoss measures reconstruction, not downstream representation quality.",
                 transform=loss_ax.transAxes, fontsize=8, color="#6B7280", va="bottom")

    y_positions = [1, 0]
    for y, route in zip(y_positions, ("A", "B")):
        info = routes[route]
        progress_ax.barh(y, 1.0, height=0.46, color=TOTAL_COLOR, edgecolor="none")
        progress_ax.barh(y, info["projected_fraction"], height=0.46,
                         color=PROJECTED_COLOR, edgecolor="none")
        progress_ax.barh(y, info["fraction"], height=0.46,
                         color=COLORS[route], edgecolor="none")
        progress_ax.text(info["fraction"] + 0.012, y,
                         f"{info['step']:,}  ({info['fraction']:.1%})",
                         va="center", fontsize=8.5, color="#26313D", fontweight="bold")
        progress_ax.text(1.0, y + 0.34,
                         f"job {info['slurm']['jobid']} · {info['slurm']['time_left']} left · "
                         f"~{info['seconds_per_update']:.2f} s/update",
                         va="center", ha="right", fontsize=7.5, color="#6B7280")
    progress_ax.set_title("B  V7 epoch completion", loc="left", fontweight="bold")
    progress_ax.set_yticks(y_positions, ["Route A", "Route B"])
    progress_ax.set_xlim(0, 1.0)
    progress_ax.set_ylim(-0.45, 1.58)
    progress_ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    progress_ax.set_xlabel("Completed fraction; gray extension = projected before safe stop")
    progress_ax.grid(axis="x", color="#D8DEE6", linewidth=0.7)
    progress_ax.spines[["top", "right", "left"]].set_visible(False)
    progress_ax.tick_params(axis="y", length=0)

    bar_width = 0.22
    centers = [0, 1]
    for x, route in zip(centers, ("A", "B")):
        info = routes[route]
        values = [len(info["published_steps"]), len(info["evaluated_steps"]), len(info["queued_steps"])]
        offsets = [-bar_width, 0, bar_width]
        colors = [COLORS[route], "#2F855A", "#9CA3AF"]
        hatches = [None, None, "///"]
        for offset, value, color, hatch in zip(offsets, values, colors, hatches):
            eval_ax.bar(x + offset, value, width=bar_width * 0.9, color=color,
                        hatch=hatch, edgecolor="white", linewidth=0.5)
            eval_ax.text(x + offset, value + 0.25, str(value), ha="center", va="bottom",
                         fontsize=8, fontweight="bold", color=color)
    eval_ax.set_title("C  Independent evaluation coverage", loc="left", fontweight="bold")
    eval_ax.set_xticks(centers, ["Route A", "Route B"])
    eval_ax.set_ylabel("Checkpoint count")
    eval_ax.grid(axis="y", color="#D8DEE6", linewidth=0.7)
    eval_ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#5B7FA3"),
               plt.Rectangle((0, 0), 1, 1, color="#2F855A"),
               plt.Rectangle((0, 0), 1, 1, color="#9CA3AF", hatch="///")]
    eval_ax.legend(handles, ["Published", "Evaluated", "Queued"], frameon=False,
                   ncol=3, fontsize=7.5, loc="upper right")
    fig.text(0.07, 0.035,
             "Interpretation: both V6 routes completed one epoch. V7 A/B are stable on 4 GPUs; evaluation is the evidence bottleneck.\n"
             "After each current 4-GPU allocation ends, its route switches to bounded sequential 2-GPU continuation. Projections cover only the current allocation.",
             ha="left", fontsize=8.5, color="#52606D")
    fig.savefig(prefix.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    print(json.dumps({"output_prefix": str(prefix), "generated_at": generated_at,
                      "route_steps": {route: info["step"] for route, info in routes.items()},
                      "evaluation_status": eval_status.get("status")}, indent=2))


if __name__ == "__main__":
    main()
