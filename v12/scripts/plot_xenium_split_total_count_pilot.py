"""Plot matched-cell SPLIT pilot readouts and the nuclear-T safeguard tradeoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run(default_path: Path, guarded_path: Path, output_png: Path) -> None:
    if output_png.exists():
        raise FileExistsError(output_png)
    default = json.loads(default_path.read_text())
    guarded = json.loads(guarded_path.read_text())
    for key in ("n_cells_corrected", "strict_t_near_interior", "strict_t_far_interior"):
        if default[key] != guarded[key]:
            raise ValueError(f"Matched-cell comparison differs on {key}")
    muc1_default = default["markers"]["MUC1"]["near_far_depth_area_standardized"]["rates_per_1000_original_rna"]
    muc1_guarded = guarded["markers"]["MUC1"]["near_far_depth_area_standardized"]["rates_per_1000_original_rna"]
    cd3e_default = default["markers"]["CD3E"]
    cd3e_guarded = guarded["markers"]["CD3E"]
    labels = ("Original", "SPLIT", "Guarded SPLIT")
    near = [muc1_default["near_raw"], muc1_default["near_corrected"], muc1_guarded["near_corrected"]]
    far = [muc1_default["far_raw"], muc1_default["far_corrected"], muc1_guarded["far_corrected"]]
    retention_near = [1, cd3e_default["strict_t_near"]["retained_fraction"],
                      cd3e_guarded["strict_t_near"]["retained_fraction"]]
    retention_far = [1, cd3e_default["strict_t_far"]["retained_fraction"],
                     cd3e_guarded["strict_t_far"]["retained_fraction"]]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.78, bottom=0.22, wspace=0.25)
    colors = ("#b65b43", "#4472a3")
    width = 0.34
    axes[0].bar(x - width / 2, near, width, color=colors[0], label="T near epithelium ≤15 µm")
    axes[0].bar(x + width / 2, far, width, color=colors[1], label="T far >50 µm")
    axes[0].set_ylabel("MUC1 / 1,000 original RNA")
    axes[0].set_title("Epithelial signal in strict T candidates")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - width / 2, np.array(retention_near) * 100, width,
                color=colors[0], label="Near")
    axes[1].bar(x + width / 2, np.array(retention_far) * 100, width,
                color=colors[1], label="Far")
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("CD3E retained vs original (%)")
    axes[1].set_title("T-marker preservation")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Xenium V1 SPLIT total-count pilot — same 18,574 cells; "
        f"strict T near n={default['strict_t_near_interior']}, far n={default['strict_t_far_interior']}",
        fontsize=11, y=0.94,
    )
    fig.text(0.08, 0.055,
             "Exploratory, non-patient-matched reference. MUC1 participates in RCTD; fractional counts leave positivity unchanged.",
             fontsize=7, color="#555555")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("default_json", type=Path)
    parser.add_argument("guarded_json", type=Path)
    parser.add_argument("output_png", type=Path)
    args = parser.parse_args()
    run(args.default_json, args.guarded_json, args.output_png)


if __name__ == "__main__":
    main()
