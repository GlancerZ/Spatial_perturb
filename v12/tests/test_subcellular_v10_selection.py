"""Teacher-eligibility selection tests; run only inside a compute allocation."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_subcellular_v10 import fixed_subset


class FakeDataset:
    def __init__(self):
        # Physical row order intentionally differs from patch-center order.
        self.slices = [
            dict(nodes=np.array([[3, 1], [0, 2], [2, 0], [1, 3]]),
                 nuclear=np.array([[0., 0.], [0., 2.], [0., 0.], [1., 0.]], dtype=np.float32),
                 cyto=np.array([[0., 0.], [0., 0.], [3., 0.], [0., 0.]], dtype=np.float32)),
            dict(nodes=np.array([[2, 1], [1, 0], [0, 2]]),
                 nuclear=np.array([[0., 0.], [4., 0.], [0., 0.]], dtype=np.float32),
                 cyto=np.zeros((3, 2), dtype=np.float32)),
        ]
        self.ends = [4, 7]

    def __len__(self):
        return self.ends[-1]


def config(count, variant="cosine"):
    return {"data": {"seed": 42}, "v10": {"pilot_centers": count, "pilot_seed": 17},
            "loss": {"type": variant}}


class V10SelectionTests(unittest.TestCase):
    def test_full_selection_excludes_empty_teachers_and_uses_patch_center_mapping(self):
        indices, stats = fixed_subset(FakeDataset(), config(0))
        np.testing.assert_array_equal(indices, [0, 2, 3, 5])
        self.assertEqual(stats["candidate_checked"], 7)
        self.assertEqual(stats["zero_teacher_excluded"], 3)
        self.assertEqual(stats["total_eligible_centers"], 4)
        self.assertTrue(stats["scan_exhausted"])

    def test_variants_share_first_eligible_candidates_and_sort_final_indices(self):
        expected_order = np.random.default_rng(17).permutation(7)
        expected = np.sort([index for index in expected_order if index in {0, 2, 3, 5}][:3])
        for variant in ("cosine", "contrastive", "relational"):
            indices, stats = fixed_subset(FakeDataset(), config(3, variant))
            np.testing.assert_array_equal(indices, expected)
            self.assertEqual(stats["eligible_candidates_observed"], 4)
            self.assertEqual(stats["candidate_checked"], 7)

    def test_insufficient_eligible_centers_raises_without_empty_fallback(self):
        with self.assertRaisesRegex(ValueError, "Insufficient nonempty whole-cell teachers"):
            fixed_subset(FakeDataset(), config(5))

    def test_pilot_stops_scanning_after_sufficient_candidates(self):
        dataset = FakeDataset()
        count = 10000
        dataset.slices = [dict(nodes=np.arange(count).reshape(-1, 1),
                               nuclear=np.ones((count, 2), dtype=np.float32),
                               cyto=np.zeros((count, 2), dtype=np.float32))]
        dataset.ends = [count]
        indices, stats = fixed_subset(dataset, config(128))
        expected = np.sort(np.random.default_rng(17).permutation(count)[:128])
        np.testing.assert_array_equal(indices, expected)
        self.assertEqual(stats["candidate_checked"], 4096)
        self.assertFalse(stats["scan_exhausted"])
        self.assertIsNone(stats["total_eligible_centers"])


if __name__ == "__main__":
    unittest.main()
