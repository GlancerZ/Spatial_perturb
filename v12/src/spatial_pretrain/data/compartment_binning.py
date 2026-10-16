"""One explicit binning contract for training and embedding extraction."""
import numpy as np
from pathlib import Path
import warnings


def uses_scgpt_binning(data):
    mode = getattr(data, 'compartment_binning', 'legacy')
    if mode not in ('legacy', 'scgpt_cell'):
        raise ValueError(f'Unknown compartment_binning: {mode}')
    if mode == 'scgpt_cell' and len(getattr(data, 'compartment_bin_edges', [])):
        raise ValueError('Per-cell binning cannot also specify global boundaries')
    return mode == 'scgpt_cell'


def scgpt_cell_bins(nuclear, cytoplasmic, num_bins, rng):
    """scGPT-spatial quantiles and randomized ties, jointly over one cell.

    Inputs are complete, unpadded compartment rows before token sampling.
    Cached log1p/float16 values are preserved in both training and evaluation.
    """
    if num_bins < 3:
        raise ValueError('At least three bins are required')
    sizes = (len(nuclear), len(cytoplasmic))
    row = np.concatenate((nuclear, cytoplasmic)).astype(np.float16).astype(np.float32)
    if not np.isfinite(row).all() or (row < 0).any():
        raise ValueError('Expression must be finite and nonnegative')
    result = np.zeros(len(row), dtype=np.int64)
    nonzero = row > 0
    values = row[nonzero]
    if len(values):
        edges = np.quantile(values, np.linspace(0, 1, num_bins - 1))
        left = np.digitize(values, edges)
        right = np.digitize(values, edges, right=True)
        result[nonzero] = np.ceil(rng.random(len(values)) * (right - left) + left).astype(np.int64)
    return result[:sizes[0]], result[sizes[0]:]


def count_bin_edges(num_bins):
    """Identity binning on raw molecule counts: bin k means exactly k molecules.

    Per-compartment Xenium counts are small integers dominated by 1, so quantile
    boundaries collapse onto log1p(1), log1p(2), ... Thirty pooled quantile edges
    reduce to four distinct values, leaving 28 of 32 prediction classes
    unreachable. Midpoints between consecutive integers make ``positive_bins``
    return ``min(count, num_bins - 1)`` instead: every class is one count, the
    mapping is identical in every slice and panel, and nothing has to be fitted.
    """
    if num_bins < 3:
        raise ValueError('Binning needs a zero bin and at least two positive bins')
    return np.log1p(np.arange(1, num_bins - 1, dtype=np.float64) + 0.5).astype(np.float32)


def validate_edges(edges, num_bins, *, strict=True):
    edges = np.asarray(edges, dtype=np.float32)
    if edges.shape != (num_bins - 2,) or not np.isfinite(edges).all():
        raise ValueError('Expected num_bins-2 finite positive-expression boundaries')
    if (edges < 0).any() or (np.diff(edges) < 0).any():
        raise ValueError('Expression boundaries must be nonnegative and nondecreasing')
    # Duplicate boundaries retire prediction classes without any error, which is
    # how quantile fitting silently reduced this head to four reachable bins.
    if strict and (np.diff(edges) <= 0).any():
        raise ValueError('Expression boundaries must be strictly increasing; '
                         'duplicates leave prediction classes unreachable')
    return edges


def configured_edges(data):
    edges = getattr(data, 'compartment_bin_edges', [])
    return validate_edges(edges, data.num_bins) if len(edges) else None


def training_edges(data):
    """Boundaries used to build training targets; identical for every slice."""
    if uses_scgpt_binning(data):
        return None
    edges = configured_edges(data)
    return count_bin_edges(data.num_bins) if edges is None else edges


def positive_bins(values, edges):
    # Match the float16 storage precision of the compartment training caches.
    values = np.asarray(values, dtype=np.float16).astype(np.float32)
    return np.digitize(values, edges, right=False).astype(np.int64) + 1


def evaluation_edges(data):
    if uses_scgpt_binning(data):
        return None
    edges = configured_edges(data)
    if edges is not None:
        return edges
    warnings.warn('Legacy checkpoint: training used per-slice bins; evaluation uses the historical '
                  'median boundary approximation. Retrain with compartment_bin_edges for a shared contract.',
                  RuntimeWarning)
    return validate_edges(np.median(np.stack([np.load(Path(c) / 'bin_edges.npy')
        for c in data.cache_dirs]), axis=0), data.num_bins, strict=False)
