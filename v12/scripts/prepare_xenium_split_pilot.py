"""Export a contiguous Xenium lung region for an isolated total-count SPLIT pilot."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmwrite
from scipy.spatial import cKDTree

from diagnose_xenium_spillover import T_SUPPORT, gene_counts


def run(input_h5ad: Path, output_dir: Path, max_cells: int) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    data = ad.read_h5ad(input_h5ad)
    genes = data.var_names.astype(str)
    if not genes.is_unique or "CD3E" not in genes or "EPCAM" not in genes:
        raise ValueError("Requires unique gene names, CD3E and EPCAM")
    xy = np.asarray(data.obsm["spatial"][:, :2], dtype=float)
    depth = np.asarray(data.X.sum(axis=1)).ravel().astype(np.int64)
    cd3e = gene_counts(data.X, genes, "CD3E")
    distinct_support = sum(((gene_counts(data.X, genes, g) > 0).astype(np.int64)
                            for g in T_SUPPORT), np.zeros(data.n_obs, dtype=np.int64))
    epcam = gene_counts(data.X, genes, "EPCAM")
    t_base = (cd3e >= 2) & (distinct_support >= 2) & (depth >= 20)
    t_eval = t_base & (epcam <= 1)
    epithelial = (epcam >= 10) & (depth >= 50) & ~t_base
    if not epithelial.any() or not t_eval.any():
        raise ValueError("Insufficient T or epithelial candidate cells")
    distance = np.full(data.n_obs, np.nan)
    distance[t_eval] = cKDTree(xy[epithelial]).query(xy[t_eval], k=1, workers=-1)[0]
    near, far = t_eval & (distance <= 15), t_eval & (distance > 50)

    candidates = []
    origin = xy.min(axis=0)
    for tile_size in (1000.0, 1500.0, 2000.0):
        tile = np.floor((xy - origin) / tile_size).astype(np.int64)
        for i, j in np.unique(tile, axis=0):
            chosen = (tile[:, 0] == i) & (tile[:, 1] == j)
            n = int(chosen.sum())
            if n < 3000 or n > max_cells:
                continue
            n_near, n_far = int((chosen & near).sum()), int((chosen & far).sum())
            n_source = int((chosen & epithelial).sum())
            if n_near < 10 or n_far < 30 or n_source < 50:
                continue
            score = n_near + min(n_far, 200) * 0.1 + min(n_source, 500) * 0.01
            candidates.append((score, tile_size, int(i), int(j), n, n_near, n_far, n_source))
    if not candidates:
        raise ValueError(f"No contiguous tile within {max_cells} cells has enough near/far T candidates")
    score, tile_size, i, j, _, _, _, _ = max(candidates)
    lower = origin + tile_size * np.array([i, j])
    upper = lower + tile_size
    selected = np.flatnonzero(np.all((xy >= lower) & (xy < upper), axis=1))
    if len(selected) > max_cells:
        raise RuntimeError("Tile selection exceeded cell limit")
    counts = sp.csr_matrix(data.X[selected])
    if counts.nnz and (np.any(counts.data < 0) or not np.all(counts.data == np.rint(counts.data))):
        raise ValueError("X must contain nonnegative integer raw counts")
    layers_sum = (sp.csr_matrix(data.layers["nuclear_counts"][selected]) +
                  sp.csr_matrix(data.layers["extranuclear_counts"][selected]))
    delta = counts - layers_sum
    delta.eliminate_zeros()
    if delta.nnz:
        raise ValueError("Nuclear plus extranuclear counts disagree with X")
    margin = np.min(np.minimum(xy[selected] - lower, upper - xy[selected]), axis=1)
    cells = pd.DataFrame({
        "cell_id": data.obs_names[selected].astype(str),
        "x_um": xy[selected, 0], "y_um": xy[selected, 1],
        "total_rna": depth[selected],
        "t_strict_epcam_low": t_eval[selected],
        "epcam_high_source": epithelial[selected],
        "nearest_source_um": distance[selected],
        "interior_15um": margin >= 15,
    })
    summary = {
        "source": str(input_h5ad.resolve()), "sample": input_h5ad.stem,
        "cells": len(selected), "genes": data.n_vars, "total_rna": int(counts.sum()),
        "tile_size_um": tile_size, "tile_ij": [i, j],
        "tile_lower_um": lower.tolist(), "tile_upper_um": upper.tolist(),
        "selection_score": score,
        "near_T_all": int(near[selected].sum()), "far_T_all": int(far[selected].sum()),
        "near_T_interior": int((near[selected] & (margin >= 15)).sum()),
        "far_T_interior": int((far[selected] & (margin >= 15)).sum()),
        "source_cells": int(epithelial[selected].sum()),
        "definition": "Contiguous spatial tile; total counts only; original H5AD/layers unchanged",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="split-input-", dir=output_dir.parent) as tmp:
        tmp_path = Path(tmp)
        with gzip.open(tmp_path / "counts_genes_by_cells.mtx.gz", "wb") as handle:
            mmwrite(handle, counts.T.tocoo())
        pd.Series(genes).to_csv(tmp_path / "genes.tsv", sep="\t", index=False, header=False)
        cells.to_csv(tmp_path / "cells.tsv", sep="\t", index=False)
        (tmp_path / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
        tmp_path.rename(output_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5ad", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-cells", type=int, default=30_000)
    args = parser.parse_args()
    print(json.dumps(run(args.input_h5ad, args.output_dir, args.max_cells), indent=2))


if __name__ == "__main__":
    main()
