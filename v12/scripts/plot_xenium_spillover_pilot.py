"""Plot the strict-gate Xenium lung spillover pilot from saved JSON outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("v1_json", type=Path)
    parser.add_argument("prime_json", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    if args.output_png.exists():
        raise FileExistsError(args.output_png)
    datasets = [
        (json.loads(args.v1_json.read_text()), "CEACAM6", "V1 lung", "#2563eb"),
        (json.loads(args.prime_json.read_text()), "MUC1", "Prime 5K lung", "#db2777"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    labels = ["0–15", "15–30", "30–50", "50–100", ">100"]
    for record, gene, sample, color in datasets:
        values = list(record["readouts"][gene]["bins"].values())
        positive = [100 * item["positive_fraction"] if item["cells"] else np.nan for item in values]
        axes[0].plot(range(5), positive, marker="o", linewidth=2.3, color=color,
                     label=f"{sample}: {gene}")
    axes[0].set_xticks(range(5), labels)
    axes[0].set_xlabel("Distance to nearest EPCAM-high cell (µm)")
    axes[0].set_ylabel("T cells positive for held-out marker (%)")
    axes[0].set_title("A  Distance-dependent epithelial signal")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(axis="y", alpha=0.2)

    names, nuclear, cytoplasmic = [], [], []
    for record, gene, sample, _ in datasets:
        for group in ("near", "far"):
            item = record["readouts"][gene][group]
            names.append(f"{sample}\n{gene}\n{group}")
            nuclear.append(item["nuclear_count_per_1000_rna"])
            cytoplasmic.append(item["cytoplasmic_count_per_1000_rna"])
    x = np.arange(len(names))
    axes[1].bar(x, nuclear, color="#6366f1", label="Nuclear")
    axes[1].bar(x, cytoplasmic, bottom=nuclear, color="#f59e0b", label="Extranuclear")
    axes[1].set_xticks(x, names, fontsize=8)
    axes[1].set_ylabel("Marker counts per 1,000 total RNAs")
    axes[1].set_title("B  Signal in both compartments")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    gate_label = ("strict T-cell gate; EPCAM≤1 in T candidates"
                  if datasets[0][0]["gates"].get("epcam_max_target") == 1
                  else "strict T-cell gate")
    fig.suptitle(f"Xenium lung proximity pilot · {gate_label}", fontsize=13)
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220)
    fig.savefig(args.output_png.with_suffix(".pdf"))
    print(args.output_png)


if __name__ == "__main__":
    main()
