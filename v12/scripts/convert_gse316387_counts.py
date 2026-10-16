#!/usr/bin/env python3
"""Convert the frozen RDS Matrix Market count export to raw-count H5AD.

Run on a compute node. The script preserves the exact benchmark cell order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing H5AD")
    ids = (args.matrix_dir / "cell_ids.txt").read_text(encoding="utf-8").splitlines()
    genes = (args.matrix_dir / "genes.txt").read_text(encoding="utf-8").splitlines()
    index = pd.read_parquet(args.index)
    if ids != index.full_cell_id.astype(str).tolist() or len(genes) != len(set(genes)):
        raise ValueError("Matrix cell order or gene names are invalid")
    counts = sparse.csr_matrix(mmread(args.matrix_dir / "counts.mtx"),
                               dtype=np.float32)
    if counts.shape != (len(ids), len(genes)):
        raise ValueError("Matrix dimensions differ from cell/gene index")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or \
            not np.equal(counts.data, np.floor(counts.data)).all():
        raise ValueError("Matrix contains invalid raw counts")
    if (np.asarray(counts.sum(axis=1)).ravel() == 0).any():
        raise ValueError("Matrix contains zero-count cells")
    obs = index.copy().set_index("full_cell_id", verify_integrity=True)
    var = pd.DataFrame(index=pd.Index(genes, name="gene_symbol"))
    data = ad.AnnData(X=counts, obs=obs, var=var)
    data.uns["source"] = "GSE316387_Merged_Object.rds RNA raw counts"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(args.output, compression="gzip")
    print(f"Wrote {data.n_obs} cells x {data.n_vars} genes to {args.output}")


if __name__ == "__main__":
    main()
