"""Export a small, reproducible LuCA raw-count reference for an RCTD/SPLIT pilot.

The source is a public, heterogeneous lung atlas, not a patient-matched
reference for any Xenium section. Its labels and assay mixture must be checked
before interpreting RCTD weights or SPLIT corrections. This script never
modifies the source or spatial counts. It creates a new output directory and
refuses to overwrite an existing one.

The Matrix Market file is genes x cells (the orientation used by RCTD), with
matching one-column genes.tsv and cells.tsv. ``cell_metadata.tsv`` retains
both fine and major labels, assay, and donor for audit. ``COMPLETE.json`` is
written only after all output files have been created successfully.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from scipy.io import mmwrite


DEFAULT_REFERENCE = Path("resources/xenium_spillover_refs/luca_extended_33165751.h5ad")
OBS_COLUMNS = ("cell_type", "cell_type_major", "assay", "donor_id")


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_categorical(group: h5py.Group, rows: np.ndarray | None = None) -> np.ndarray:
    categories = np.asarray([_text(v) for v in group["categories"][:]], dtype=object)
    codes = group["codes"][:] if rows is None else group["codes"][rows]
    values = np.full(codes.shape, "", dtype=object)
    valid = (codes >= 0) & (codes < len(categories))
    values[valid] = categories[codes[valid]]
    return values


def _read_string_field(parent: h5py.Group, name: str,
                       rows: np.ndarray | None = None) -> np.ndarray:
    field = parent[name]
    if isinstance(field, h5py.Group):
        if not {"categories", "codes"}.issubset(field):
            raise ValueError(f"{field.name} is not an AnnData categorical field")
        return _read_categorical(field, rows)
    values = field[:] if rows is None else field[rows]
    return np.asarray([_text(v) for v in values], dtype=object)


def _panel_genes(args: argparse.Namespace) -> list[str]:
    if args.panel_h5ad is not None:
        with h5py.File(args.panel_h5ad, "r") as panel:
            if "var" not in panel:
                raise ValueError("Panel H5AD has no /var")
            var = panel["var"]
            key = "gene" if "gene" in var else "_index"
            genes = _read_string_field(var, key).tolist()
    else:
        with args.panel_genes.open("r", encoding="utf-8") as handle:
            genes = [line.rstrip("\n").split("\t", 1)[0].strip()
                     for line in handle if line.strip() and not line.startswith("#")]
        if genes and genes[0].lower() in {"gene", "gene_name", "feature_name"}:
            genes.pop(0)
    genes = [gene for gene in genes if gene]
    if len(genes) != len(set(genes)):
        raise ValueError("Panel gene names are not unique")
    if not genes:
        raise ValueError("Panel gene list is empty")
    return genes


def _choose_cells(labels: np.ndarray, assay: np.ndarray,
                  allowed_assays: list[str] | None, cap: int, seed: int
                  ) -> tuple[np.ndarray, dict[str, int]]:
    eligible = labels != ""
    if allowed_assays:
        eligible &= np.isin(assay, allowed_assays)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    available: dict[str, int] = {}
    for label in sorted(set(labels[eligible])):
        candidates = np.flatnonzero(eligible & (labels == label))
        available[str(label)] = int(candidates.size)
        chosen = (candidates if candidates.size <= cap else
                  rng.choice(candidates, size=cap, replace=False))
        selected.append(chosen)
    if not selected:
        raise ValueError("No labeled cells match the requested assay filter")
    return np.sort(np.concatenate(selected)), available


def _extract_counts(group: h5py.Group, selected: np.ndarray,
                    reference_to_panel: np.ndarray, n_panel_genes: int
                    ) -> sparse.coo_matrix:
    if _text(group.attrs.get("encoding-type", "")) != "csr_matrix":
        raise ValueError("/layers/count is not CSR; refusing to use X instead")
    shape = tuple(int(v) for v in group.attrs["shape"])
    if shape[1] != reference_to_panel.size or selected[-1] >= shape[0]:
        raise ValueError("Raw-count matrix shape does not match obs/var")
    pointers = group["indptr"][:]
    if pointers.size != shape[0] + 1 or int(pointers[-1]) != len(group["data"]):
        raise ValueError("Invalid CSR row pointers")
    if len(group["data"]) != len(group["indices"]):
        raise ValueError("CSR values and gene indices have different lengths")

    gene_parts: list[np.ndarray] = []
    cell_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for output_cell, source_cell in enumerate(selected):
        start, stop = int(pointers[source_cell]), int(pointers[source_cell + 1])
        if start == stop:
            continue
        reference_genes = group["indices"][start:stop]
        if np.any(reference_genes < 0) or np.any(reference_genes >= shape[1]):
            raise ValueError(f"Invalid gene index in source cell {source_cell}")
        mapped = reference_to_panel[reference_genes]
        keep = mapped >= 0
        if not np.any(keep):
            continue
        raw = group["data"][start:stop][keep]
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise ValueError(f"Non-finite or negative count in source cell {source_cell}")
        rounded = np.rint(raw)
        if not np.all(np.abs(raw - rounded) < 1e-4):
            raise ValueError(f"Non-integer raw count in source cell {source_cell}")
        if np.any(rounded > np.iinfo(np.int64).max):
            raise ValueError(f"Raw count exceeds int64 in source cell {source_cell}")
        gene_parts.append(mapped[keep].astype(np.int32, copy=False))
        cell_parts.append(np.full(int(np.count_nonzero(keep)), output_cell,
                                  dtype=np.int32))
        value_parts.append(rounded.astype(np.int64))
    if not value_parts:
        raise ValueError("Selected reference cells have no counts in panel genes")
    matrix = sparse.coo_matrix(
        (np.concatenate(value_parts),
         (np.concatenate(gene_parts), np.concatenate(cell_parts))),
        shape=(n_panel_genes, selected.size), dtype=np.int64,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _write_lines(path: Path, values: list[str] | np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            item = str(value)
            if "\n" in item or "\t" in item:
                raise ValueError(f"Invalid TSV value in {path.name}: {item!r}")
            handle.write(item + "\n")


def export(args: argparse.Namespace) -> dict:
    panel = _panel_genes(args)
    with h5py.File(args.reference_h5ad, "r") as reference:
        for column in OBS_COLUMNS:
            if column not in reference["obs"]:
                raise ValueError(f"Reference /obs/{column} is missing")
        if "feature_name" not in reference["var"] or "count" not in reference["layers"]:
            raise ValueError("Reference needs /var/feature_name and /layers/count")
        labels = _read_string_field(reference["obs"], args.label_column)
        assay = _read_string_field(reference["obs"], "assay")
        selected, available = _choose_cells(labels, assay, args.assay,
                                            args.max_cells_per_type, args.seed)
        names = _read_string_field(reference["var"], "feature_name")
        panel_set = set(panel)
        duplicate_panel_genes = [gene for gene, count in Counter(names).items()
                                 if count > 1 and gene in panel_set]
        if duplicate_panel_genes:
            raise ValueError("Reference has ambiguous panel gene symbols: "
                             f"{duplicate_panel_genes[:10]}")
        lookup = {name: i for i, name in enumerate(names) if name}
        overlapping = [gene for gene in panel if gene in lookup]
        if len(overlapping) < args.min_overlap:
            raise ValueError(f"Only {len(overlapping)} of {len(panel)} panel genes "
                             f"overlap the reference (minimum {args.min_overlap})")
        reference_to_panel = np.full(len(names), -1, dtype=np.int32)
        for panel_index, gene in enumerate(overlapping):
            reference_to_panel[lookup[gene]] = panel_index
        columns = {column: _read_string_field(reference["obs"], column, selected)
                   for column in OBS_COLUMNS}
        cell_ids = _read_string_field(reference["obs"], "_index", selected)
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("Selected reference cell IDs are not unique")

        matrix = _extract_counts(reference["layers"]["count"], selected,
                                 reference_to_panel, len(overlapping))

    # Create only after source validation/extraction; never write over a prior run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    mmwrite(str(args.output_dir / "counts.mtx"), matrix, comment="LuCA raw counts; genes x cells")
    _write_lines(args.output_dir / "genes.tsv", overlapping)
    _write_lines(args.output_dir / "cells.tsv", cell_ids)
    with (args.output_dir / "cell_metadata.tsv").open("w", encoding="utf-8") as handle:
        handle.write("cell_id\tcell_type\tcell_type_major\tassay\tdonor_id\n")
        for i, cell_id in enumerate(cell_ids):
            fields = [cell_id] + [columns[column][i] for column in OBS_COLUMNS]
            if any("\n" in str(v) or "\t" in str(v) for v in fields):
                raise ValueError("Metadata contains a tab or newline")
            handle.write("\t".join(map(str, fields)) + "\n")

    qc = {
        "reference_h5ad": str(args.reference_h5ad.resolve()),
        "panel_source": str((args.panel_h5ad or args.panel_genes).resolve()),
        "label_column": args.label_column,
        "seed": args.seed,
        "max_cells_per_type": args.max_cells_per_type,
        "assay_filter": args.assay,
        "panel_genes": len(panel),
        "overlapping_panel_genes": len(overlapping),
        "missing_panel_genes": [gene for gene in panel if gene not in lookup],
        "selected_cells": int(selected.size),
        "nonzero_counts": int(matrix.nnz),
        "total_raw_counts": int(matrix.sum()),
        "cells_available_by_type": available,
        "cells_selected_by_type": dict(Counter(columns[args.label_column])),
        "cells_selected_by_assay": dict(Counter(columns["assay"])),
        "cells_selected_by_donor": dict(Counter(columns["donor_id"])),
        "limitation": "Public LuCA reference is not patient matched; assay and tumor-state mismatch can bias RCTD/SPLIT.",
    }
    with (args.output_dir / "COMPLETE.json").open("w", encoding="utf-8") as handle:
        json.dump(qc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return qc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-h5ad", type=Path, default=DEFAULT_REFERENCE)
    panel = parser.add_mutually_exclusive_group(required=True)
    panel.add_argument("--panel-h5ad", type=Path,
                       help="Xenium H5AD; reads /var/gene or /var/_index")
    panel.add_argument("--panel-genes", type=Path,
                       help="One-column text/TSV of Xenium gene symbols")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label-column", choices=("cell_type", "cell_type_major"),
                        default="cell_type")
    parser.add_argument("--assay", action="append",
                        help="Optional exact LuCA assay name; repeat to allow several")
    parser.add_argument("--max-cells-per-type", type=int, default=400)
    parser.add_argument("--min-overlap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    if args.max_cells_per_type <= 0 or args.min_overlap <= 0:
        parser.error("--max-cells-per-type and --min-overlap must be positive")
    result = export(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
