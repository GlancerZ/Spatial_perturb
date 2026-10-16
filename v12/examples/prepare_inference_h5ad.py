#!/usr/bin/env python3
"""Copy segmentation-derived Xenium counts into the explicit V12 inference schema."""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse


def checked(matrix, name):
    values = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
    if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.rint(values)).all():
        raise ValueError(f'{name} must be finite nonnegative integer raw counts')
    return sparse.csr_matrix(matrix, dtype=np.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output already exists; choose a new path')
    x = ad.read_h5ad(a.input)
    n = checked(x.layers['nuclear_counts'], 'nuclear_counts')
    c = checked(x.layers['extranuclear_counts'], 'extranuclear_counts')
    total = checked(x.X, 'X')
    if (total != n + c).nnz:
        raise ValueError('X must equal nuclear_counts + extranuclear_counts exactly')
    coords = np.asarray(x.obsm['spatial'], dtype=np.float64)
    if coords.shape != (x.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError('Expected finite [cells,2] Xenium spatial coordinates in micrometers')
    x.layers['cytoplasmic_counts'] = x.layers['extranuclear_counts'].copy()
    x.obsm['spatial_um'] = coords.copy()
    x.uns['compartment_provenance'] = {
        'source': str(a.input.resolve()),
        'method': 'Alias existing segmentation-derived Xenium counts; no counts inferred from total',
        'cytoplasmic_counts_source': 'layers/extranuclear_counts',
        'spatial_um_source': 'Xenium obsm/spatial in micrometers',
        'count_conservation_verified': True,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    x.write_h5ad(a.output, compression='gzip')
    print(a.output)


if __name__ == '__main__':
    main()
