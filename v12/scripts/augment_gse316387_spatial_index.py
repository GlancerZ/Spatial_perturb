#!/usr/bin/env python3
"""Add author meta-niche labels to the prespecified spatial score cohort."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to replace existing index")
    index = pd.read_parquet(args.index)
    labels = pd.read_csv(args.labels, compression="gzip", dtype=str,
                         keep_default_na=False)
    if not labels.full_cell_id.is_unique or not index.full_cell_id.is_unique:
        raise ValueError("Duplicated cell IDs")
    merged = index.merge(labels[["full_cell_id", "meta_niche_name"]],
                         on="full_cell_id", how="left", sort=False,
                         validate="one_to_one")
    if not merged.full_cell_id.equals(index.full_cell_id) or \
            merged.meta_niche_name.isna().any():
        raise ValueError("Niche metadata is missing scored cell IDs or reordered them")
    merged.to_parquet(args.output, index=False)
    audit = {"score_cells": len(merged),
             "niche_labels": sorted(merged.loc[merged.meta_niche_name.ne(""),
                                            "meta_niche_name"].unique().tolist()),
             "missing_niche_by_split": merged.loc[merged.meta_niche_name.eq(""),
                                                  "split"].value_counts().to_dict(),
             "niche_by_split": {part: frame.meta_niche_name.value_counts().to_dict()
                                for part, frame in merged.groupby("split")}}
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(merged),
                      "niche_classes": len(audit["niche_labels"]),
                      "missing_niche": sum(audit["missing_niche_by_split"].values())}),
          flush=True)


if __name__ == "__main__":
    main()
