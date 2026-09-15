"""One explicit binning contract for training and embedding extraction."""
import numpy as np
from pathlib import Path
import warnings


def validate_edges(edges, num_bins):
    edges = np.asarray(edges, dtype=np.float32)
    if edges.shape != (num_bins - 2,) or not np.isfinite(edges).all():
        raise ValueError('Expected num_bins-2 finite positive-expression boundaries')
    if (edges < 0).any() or (np.diff(edges) < 0).any():
        raise ValueError('Expression boundaries must be nonnegative and nondecreasing')
    return edges


def configured_edges(data):
    edges = getattr(data, 'compartment_bin_edges', [])
    return validate_edges(edges, data.num_bins) if len(edges) else None


def positive_bins(values, edges):
    # Match the float16 storage precision of the compartment training caches.
    values = np.asarray(values, dtype=np.float16).astype(np.float32)
    return np.digitize(values, edges, right=False).astype(np.int64) + 1


def evaluation_edges(data):
    edges = configured_edges(data)
    if edges is not None:
        return edges
    warnings.warn('Legacy checkpoint: training used per-slice bins; evaluation uses the historical '
                  'median boundary approximation. Retrain with compartment_bin_edges for a shared contract.',
                  RuntimeWarning)
    return validate_edges(np.median(np.stack([np.load(Path(c) / 'bin_edges.npy')
        for c in data.cache_dirs]), axis=0), data.num_bins)
