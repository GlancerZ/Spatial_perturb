"""Plot recorded training batches and validation means without smoothing."""
import argparse
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--log", type=Path, default=Path("logs/compartment_ready48.log"))
parser.add_argument("--output", type=Path, default=Path("outputs/xenium_compartment_ready48/loss.png"))
parser.add_argument("--resume-step", type=int, default=None)
parser.add_argument("--title", default="Xenium pretraining · 48 samples")
args = parser.parse_args()
series = {"train": {}, "val": {}}
for line in args.log.read_text().splitlines():
    match = re.search(r"\[train\].*update=(\d+).*loss=([\d.eE+-]+)", line)
    if match:
        series["train"][int(match[1])] = float(match[2])
    match = re.search(r"\[val\] step=(\d+) loss=([\d.eE+-]+)", line)
    if match:
        series["val"][int(match[1])] = float(match[2])
    match = re.search(r"\[done\] val_loss=([\d.eE+-]+)", line)
    if match and series["train"]:
        series["val"][max(series["train"])] = float(match[1])
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.with_suffix(".csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["optimizer_step", "series", "loss"])
    for name, points in series.items():
        writer.writerows((step, name, value) for step, value in sorted(points.items()))
fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
for name, label, color in [("train", "Training (logged batch)", "#2463b4"),
                           ("val", "Validation (up to 100 batches)", "#d16b24")]:
    points = sorted(series[name].items())
    if points:
        steps, values = zip(*points)
        ax.plot(steps, values, marker="o" if name == "val" else ".",
                linestyle="--" if name == "val" else "-", markersize=4,
                label=label, color=color, linewidth=1.6)
        if name == "val":
            ax.annotate(f"{steps[-1]}: {values[-1]:.4f}", (steps[-1], values[-1]),
                        xytext=(-8, -20), textcoords="offset points", ha="right", color=color)
if args.resume_step is not None:
    ax.axvline(args.resume_step, color="#666666", linestyle=":", linewidth=1,
               label=f"Resume from step {args.resume_step}")
ax.set(title=args.title, xlabel="Optimizer update", ylabel="Masked expression cross-entropy")
ax.grid(alpha=0.2)
ax.legend()
fig.savefig(args.output, dpi=180)
fig.savefig(args.output.with_suffix(".pdf"))
print({name: sorted(points.items()) for name, points in series.items()})
print(args.output)
