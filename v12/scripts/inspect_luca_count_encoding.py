"""Audit whether LuCA H5AD count-like matrices contain integer raw UMI values."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


SOURCE = Path("resources/xenium_spillover_refs/luca_extended_33165751.h5ad")


def sample_matrix(group: h5py.Group, row: int) -> dict:
    start, stop = map(int, group["indptr"][row: row + 2])
    values = group["data"][start:stop]
    return {
        "dtype": str(group["data"].dtype),
        "encoding": str(group.attrs.get("encoding-type")),
        "nnz_row": len(values),
        "min": float(np.min(values)) if len(values) else None,
        "max": float(np.max(values)) if len(values) else None,
        "non_integer": int(np.count_nonzero(np.abs(values - np.rint(values)) > 1e-4)),
        "first_non_integer": values[np.abs(values - np.rint(values)) > 1e-4][:12].tolist(),
    }


with h5py.File(SOURCE, "r") as handle:
    assay_field = handle["obs"]["assay"]
    if isinstance(assay_field, h5py.Group):
        assay_labels = [v.decode() if isinstance(v, bytes) else str(v)
                        for v in assay_field["categories"][:]]
        assay_codes = assay_field["codes"][:]
    else:
        assay_labels = sorted(set(assay_field[:]))
        assay_codes = np.searchsorted(assay_labels, assay_field[:])
    assay_audit = {}
    for code, label in enumerate(assay_labels):
        cells = np.flatnonzero(assay_codes == code)
        if not len(cells):
            continue
        sampled = cells[np.linspace(0, len(cells) - 1, min(40, len(cells)), dtype=int)]
        audited = [sample_matrix(handle["layers"]["count"], int(row)) for row in sampled]
        assay_audit[label] = {
            "cells": int(len(cells)), "sampled": int(len(sampled)),
            "noninteger_values": sum(item["non_integer"] for item in audited),
            "nonzero_values": sum(item["nnz_row"] for item in audited),
        }
    print(json.dumps({
        "source": str(SOURCE),
        "layers": list(handle["layers"].keys()),
        "uns_keys": list(handle["uns"].keys()),
        "row": 263975,
        "row_assay": assay_labels[int(assay_codes[263975])],
        "assay_audit": assay_audit,
        "count": sample_matrix(handle["layers"]["count"], 263975),
        "X": sample_matrix(handle["X"], 263975),
    }, indent=2))
