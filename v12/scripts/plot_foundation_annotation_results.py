"""Make a source-checked four-model LungPF annotation comparison figure.

Run on a compute node after all four per-model evaluations are complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


MODELS = ("geneformer", "scgpt", "scfoundation", "state")
TASKS = ("cell_type", "region")
METRICS = ("macro_f1", "balanced_accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path,
                        default=Path("outputs/foundation_annotation_lungpf/evaluation"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/foundation_annotation_lungpf/four_model_comparison.png"))
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        raise FileExistsError("Refusing to overwrite an existing comparison artifact")
    records = []
    input_hash = None
    for task in TASKS:
        cell_count = None
        for model in MODELS:
            path = args.evaluation_root / model / task / "metrics.json"
            report = json.loads(path.read_text())
            if report.get("model") != model or report.get("task") != task:
                raise ValueError(f"Metric identity mismatch in {path}")
            digest = report.get("input_manifest_sha256")
            if not digest or (input_hash is not None and digest != input_hash):
                raise ValueError("Models do not share the same LungPF input manifest")
            input_hash = digest
            n = report["test"]["cells"]
            if cell_count is not None and n != cell_count:
                raise ValueError(f"Different test cell counts for {task}")
            cell_count = n
            for metric in METRICS:
                value = report["test"][metric]
                if not isinstance(value, (float, int)) or not 0 <= value <= 1:
                    raise ValueError(f"Invalid {metric} for {model}/{task}")
                records.append({"model": model, "task": task, "metric": metric,
                                "value": float(value), "test_cells": n,
                                "selected_C": report["selected_C"]})
    table = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"geneformer": "#37618a", "scgpt": "#1e9c7c",
              "scfoundation": "#d58d34", "state": "#925eb1"}
    for row, task in enumerate(TASKS):
        for col, metric in enumerate(METRICS):
            axis = axes[row, col]
            part = table.loc[(table.task == task) & (table.metric == metric)]
            values = [float(part.loc[part.model == model, "value"].iloc[0])
                      for model in MODELS]
            axis.bar(MODELS, values, color=[colors[model] for model in MODELS])
            axis.set_ylim(0, min(1.0, max(values) + 0.12))
            axis.set_title(f"{task.replace('_', ' ')} · {metric.replace('_', ' ')}")
            axis.set_ylabel("Test score")
            axis.tick_params(axis="x", rotation=20)
            for index, value in enumerate(values):
                axis.text(index, value + 0.015, f"{value:.3f}",
                          ha="center", va="bottom", fontsize=9)
    fig.suptitle("Frozen single-cell model embeddings · LungPF donor-held-out test",
                 fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    table.to_csv(args.output.with_suffix(".csv"), index=False)
    print(f"Compared four models on shared LungPF input {input_hash}: {args.output}")


if __name__ == "__main__":
    main()
