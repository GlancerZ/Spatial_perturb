import unittest
from types import SimpleNamespace

import numpy as np

from spatial_pretrain.data.compartment_binning import (
    count_bin_edges, positive_bins, training_edges, validate_edges,
)


class CountIdentityBinningTests(unittest.TestCase):
    def test_bin_equals_clipped_raw_count(self):
        edges = count_bin_edges(32)
        self.assertEqual(edges.shape, (30,))
        counts = np.arange(1, 60)
        bins = positive_bins(np.log1p(counts.astype(np.float32)), edges)
        np.testing.assert_array_equal(bins, np.minimum(counts, 31))

    def test_float16_cache_precision_does_not_shift_a_bin(self):
        # Training reads float16 caches; extraction re-derives log1p in float32.
        edges = count_bin_edges(32)
        counts = np.arange(1, 60)
        stored = np.log1p(counts.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(positive_bins(stored, edges),
                                      positive_bins(np.log1p(counts.astype(np.float32)), edges))

    def test_every_positive_class_is_reachable(self):
        for num_bins in (4, 8, 32):
            edges = count_bin_edges(num_bins)
            bins = positive_bins(np.log1p(np.arange(1, num_bins + 20).astype(np.float32)), edges)
            self.assertEqual(sorted(set(bins.tolist())), list(range(1, num_bins)))

    def test_duplicate_boundaries_are_rejected(self):
        # Pooled quantiles over Xenium counts collapse to four distinct values.
        collapsed = np.concatenate([np.full(23, .6934), np.full(4, 1.0986), np.full(2, 1.3867), [1.6094]])
        with self.assertRaises(ValueError):
            validate_edges(collapsed, 32)
        validate_edges(collapsed, 32, strict=False)

    def test_unconfigured_runs_share_one_slice_independent_mapping(self):
        data = SimpleNamespace(num_bins=32, compartment_bin_edges=[])
        np.testing.assert_array_equal(training_edges(data), count_bin_edges(32))
        configured = SimpleNamespace(num_bins=32, compartment_bin_edges=count_bin_edges(32).tolist())
        np.testing.assert_array_equal(training_edges(configured), count_bin_edges(32))


if __name__ == '__main__':
    unittest.main()
