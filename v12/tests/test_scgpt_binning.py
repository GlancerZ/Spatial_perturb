import numpy as np
import unittest

from spatial_pretrain.data.compartment_binning import scgpt_cell_bins


def reference(row, n_bins, rng):
    # scGPT-spatial preprocess.binning/_digitize, with an explicit RNG.
    result = np.zeros_like(row, dtype=np.int64)
    nonzero = row.nonzero()
    values = row[nonzero]
    if not len(values):
        return result
    edges = np.quantile(values, np.linspace(0, 1, n_bins - 1))
    left = np.digitize(values, edges)
    right = np.digitize(values, edges, right=True)
    result[nonzero] = np.ceil(rng.random(len(values)) * (right - left) + left).astype(np.int64)
    return result


def check_matches_upstream_quantiles_and_ties(counts):
    values = np.log1p(counts).astype(np.float16).astype(np.float32)
    split = len(values)//2
    actual = np.concatenate(scgpt_cell_bins(values[:split], values[split:], 32, np.random.default_rng(42)))
    expected = reference(values, 32, np.random.default_rng(42))
    np.testing.assert_array_equal(actual, expected)
    assert (actual[values == 0] == 0).all()
    assert ((actual[values > 0] >= 1) & (actual[values > 0] < 32)).all()


def test_full_cell_binning_preserves_compartment_shapes_and_seed():
    nuclear = np.log1p([0, 1, 1, 5])
    cytoplasm = np.log1p([2, 8, 0])
    first = scgpt_cell_bins(nuclear, cytoplasm, 32, np.random.default_rng(123))
    second = scgpt_cell_bins(nuclear, cytoplasm, 32, np.random.default_rng(123))
    assert tuple(map(len, first)) == (4, 3)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_invalid_values_rejected():
    with unittest.TestCase().assertRaises(ValueError):
        scgpt_cell_bins([float('nan')], [1], 32, np.random.default_rng(42))


class ScgptBinningTests(unittest.TestCase):
    def test_reference_parity(self):
        for counts in [[0,0,0,0], [0,1,1,2,3,0], [1]*200, [0,4,0,0]]:
            with self.subTest(counts=counts[:6]):
                check_matches_upstream_quantiles_and_ties(counts)

    def test_shapes_and_seed(self):
        test_full_cell_binning_preserves_compartment_shapes_and_seed()

    def test_invalid(self):
        test_invalid_values_rejected()
