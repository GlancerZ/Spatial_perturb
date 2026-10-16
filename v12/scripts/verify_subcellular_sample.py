"""Inspect a bounded official Xenium test ZIP on a compute node, without extraction.

This validates one downloaded file, not every dataset of the same platform.
The test bundle is deliberately excluded from biological training cohorts.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence path")
    if args.bundle.stat().st_size > 32 * 1024 * 1024:
        parser.error("This bounded smoke validator only accepts bundles <= 32 MiB")
    raw = args.bundle.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        transcript_members = [item for item in members if item.filename.endswith("/transcripts.parquet") or item.filename == "transcripts.parquet"]
        if len(transcript_members) != 1:
            raise ValueError("Expected exactly one transcript parquet in the test bundle")
        member = transcript_members[0]
        if member.file_size > 64 * 1024 * 1024:
            raise ValueError("Transcript table exceeds smoke validator uncompressed limit")
        table = pq.read_table(io.BytesIO(archive.read(member)))
        required = {"cell_id", "feature_name", "x_location", "y_location", "overlaps_nucleus", "qv"}
        missing = required - set(table.column_names)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        frame = table.select(sorted(required)).to_pandas()
        if frame.empty or frame[list(required)].isna().any().any():
            raise ValueError("Empty table or null required values")
        if not np.isfinite(frame[["x_location", "y_location", "qv"]].to_numpy()).all():
            raise ValueError("Nonfinite molecule coordinates or QV")
        if not frame.overlaps_nucleus.isin([0, 1, False, True]).all():
            raise ValueError("Nuclear-overlap flags must be binary")
        cell_ids = frame.cell_id.astype(str)
        assigned = ~cell_ids.str.lower().isin(["unassigned", "-1", "0", "", "none", "nan"])
        quality = frame.qv >= 20
        feature = frame.feature_name.astype(str)
        # This is schema/compartment QA, not a final assay-specific feature filter.
        gene = ~feature.str.startswith(("NegControl", "Negative", "Unassigned", "Deprecated", "Blank", "GenomicControl"))
        selected = assigned & quality & gene
        nuclear = selected & frame.overlaps_nucleus.astype(bool)
        cytoplasmic = selected & ~frame.overlaps_nucleus.astype(bool)
        if nuclear.sum() + cytoplasmic.sum() != selected.sum():
            raise AssertionError("N/C assignment does not partition selected transcripts")
        if nuclear.sum() == 0 or cytoplasmic.sum() == 0:
            raise ValueError("Test sample must exercise both nuclear and cytoplasmic assignment")
        result = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "bundle": str(args.bundle.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "md5": hashlib.md5(raw).hexdigest(),
            "bytes": len(raw),
            "transcript_member": member.filename,
            "columns": table.column_names,
            "rows": len(frame),
            "assigned_rows": int(assigned.sum()),
            "unassigned_rows": int((~assigned).sum()),
            "selected_q20_noncontrol_assigned": int(selected.sum()),
            "nuclear_rows": int(nuclear.sum()),
            "cytoplasmic_rows": int(cytoplasmic.sum()),
            "selected_cells": int(cell_ids[selected].nunique()),
            "nucleus_mask_or_boundary_files": [m.filename for m in members if "nucleus_boundaries" in m.filename or "cells.zarr" in m.filename],
            "cell_boundary_files": [m.filename for m in members if "cell_boundaries" in m.filename],
            "status": "file_schema_and_assignment_verified",
            "training_use": "format_validation_only",
            "limitations": [
                "Official artificially trimmed test dataset; not biological evidence or a training cohort.",
                "Nuclear flags are segmentation-derived; this check does not establish segmentation accuracy.",
                "Unassigned RNA is excluded, never relabeled as cytoplasm.",
                "General panel/control-gene validation and donor/test overlap audit are still required before training.",
            ],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
