#!/usr/bin/env python3
"""Summarize which GEO regions contain frozen train/val/test cells."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing output")
    index = pd.read_parquet(args.index)
    spatial = pd.read_parquet(args.spatial)
    if not index.full_cell_id.equals(spatial.full_cell_id):
        raise ValueError("Spatial coordinates do not follow frozen cell order")
    index["region"] = spatial.region.to_numpy()
    records = []
    for region, frame in index.groupby("region"):
        records.append({"region": region, "cells": len(frame),
                        "by_split": frame.split.value_counts().to_dict(),
                        "patients_by_split": {part: sorted(group.patient.unique().tolist())
                                              for part, group in frame.groupby("split")},
                        "samples": sorted(frame["sample"].unique().tolist())})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
