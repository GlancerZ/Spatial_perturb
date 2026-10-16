"""Temporary read-only diagnostic for the Xenium ResolVI pilot input."""
import sys

import anndata as ad
import numpy as np
import scipy.sparse as sp

data = ad.read_h5ad(sys.argv[1])
print("input", type(data.X), getattr(data.X, "dtype", None), data.shape, sp.issparse(data.X))
totals = np.asarray(data.X.sum(axis=1)).ravel()
eligible = np.flatnonzero(totals >= 20)
subset = data[eligible, :].copy()
print("subset", type(subset.X), getattr(subset.X, "dtype", None), subset.shape, sp.issparse(subset.X))
print("subset_3", repr(subset.X[:3, :3]))
converted = sp.csr_matrix(subset.X)
print("converted", converted.dtype, converted.shape)
for name in list(subset.layers.keys()):
    del subset.layers[name]
print("after_layer_drop", type(subset.X), getattr(subset.X, "dtype", None), sp.issparse(subset.X))
converted = sp.csr_matrix(subset.X)
print("after_layer_drop_converted", converted.dtype, converted.shape)
