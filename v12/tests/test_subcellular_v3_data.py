"""V3's bin/visibility contract, tested without pretrained weights or large data."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA, stable_seed
from spatial_pretrain.data.subcellular_v3 import (
    V3Dataset, collate_v3, encode_context_v3, full_cell_bins_v3,
    sample_paired_queries_v3, sample_uniform_queries_v3,
)


ENCODER_KEYS = {"gene_ids", "input_bins", "compartment_ids", "token_valid_mask", "node_indices"}


def token_bins(encoded, row=0):
    return {(int(comp), int(gene)): int(value)
            for comp, gene, value, valid in zip(
                encoded["compartment_ids"][row], encoded["gene_ids"][row],
                encoded["input_bins"][row], encoded["token_valid_mask"][row])
            if valid and comp in (0, 1)}


class FullBinContextTests(unittest.TestCase):
    def setUp(self):
        self.nuclear = np.log1p(np.array([[0, 1, 2, 3, 0, 1], [4, 1, 3, 1, 1, 2]], dtype=np.float32))
        self.cyto = np.log1p(np.array([[2, 0, 1, 0, 4, 1], [1, 3, 1, 2, 1, 1]], dtype=np.float32))
        self.genes = np.arange(2, 8)

    def encode(self, **kwargs):
        return encode_context_v3(self.nuclear, self.cyto, self.genes,
            node_indices=[21, 50], context_tokens_per_compartment=6,
            pad_token_id=0, cls_token_id=1, **kwargs)

    def test_full_bins_share_one_reference_and_keep_real_zero(self):
        bins = full_cell_bins_v3(np.log1p([1, 8, 0]), np.log1p([10, 3, 0]), node_index=21)
        self.assertEqual(bins.shape, (2, 3))
        self.assertEqual(bins.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(bins[:, 2], [0, 0])
        self.assertLess(bins[0, 0], bins[1, 1])
        self.assertLess(bins[1, 1], bins[0, 1])
        self.assertLess(bins[0, 1], bins[1, 0])
        self.assertTrue(((bins[:, :2] >= 1) & (bins[:, :2] <= 50)).all())

    def test_all_mask_modes_retain_original_visible_bins(self):
        unmasked = self.encode()
        baseline = token_bins(unmasked)
        for mask in ({"masked_genes": [1, 2]}, {"masked_compartment": 0},
                     {"masked_compartment": 1}, {"mask_whole_cell": True}):
            with self.subTest(mask=mask):
                encoded = self.encode(**mask)
                self.assertEqual(set(encoded), ENCODER_KEYS)
                for key, value in token_bins(encoded).items():
                    self.assertEqual(value, baseline[key])
                # Each neighbor owns its reference and is never center-masked.
                for key in encoded:
                    np.testing.assert_array_equal(encoded[key][1], unmasked[key][1])
                if "masked_genes" in mask:
                    self.assertFalse(np.isin(encoded["gene_ids"][0], self.genes[mask["masked_genes"]]).any())
                if mask.get("mask_whole_cell"):
                    self.assertEqual(encoded["token_valid_mask"][0].sum(), 1)

    def test_bins_do_not_depend_on_epoch_patch_position_or_token_budget(self):
        baseline = token_bins(self.encode())
        self.assertEqual(token_bins(self.encode(epoch=19)), baseline)
        reverse = encode_context_v3(self.nuclear[::-1], self.cyto[::-1], self.genes,
            node_indices=[50, 21], epoch=31, context_tokens_per_compartment=6,
            pad_token_id=0, cls_token_id=1)
        self.assertEqual(token_bins(reverse, row=1), baseline)
        shortened = encode_context_v3(self.nuclear, self.cyto, self.genes,
            node_indices=[21, 50], epoch=31, context_tokens_per_compartment=1,
            pad_token_id=0, cls_token_id=1)
        self.assertEqual(len(token_bins(shortened)), 2)
        for key, value in token_bins(shortened).items():
            self.assertEqual(value, baseline[key])

    def test_preserves_sources_and_exposes_no_target_tensors(self):
        original_n, original_c = self.nuclear.copy(), self.cyto.copy()
        encoded = self.encode(masked_genes=[1, 3])
        self.assertEqual(set(encoded), ENCODER_KEYS)
        self.assertFalse(any("target" in key or "count" in key for key in encoded))
        np.testing.assert_array_equal(self.nuclear, original_n)
        np.testing.assert_array_equal(self.cyto, original_c)
        gene_mask = encoded["token_valid_mask"] & (encoded["compartment_ids"] < 2)
        self.assertTrue((encoded["input_bins"][gene_mask] >= 1).all())
        self.assertTrue((encoded["input_bins"][~encoded["token_valid_mask"]] == -2).all())

    def test_empty_compartments_and_padding_rows_keep_only_sentinels(self):
        encoded = encode_context_v3(np.zeros((2, 6)), np.zeros((2, 6)), self.genes,
            node_indices=[21, -1], pad_token_id=0, cls_token_id=1)
        self.assertEqual(encoded["token_valid_mask"].sum(), 2)
        np.testing.assert_array_equal(encoded["gene_ids"][:, 0], [1, 1])
        np.testing.assert_array_equal(encoded["compartment_ids"][:, 0], [2, 2])
        self.assertTrue((encoded["input_bins"] == -2).all())
        bins = full_cell_bins_v3(np.zeros(6), np.zeros(6), node_index=21)
        self.assertTrue((bins == 0).all())

    def test_invalid_panels_masks_and_values_are_rejected(self):
        for values in (np.full((1, 2), -1.), np.full((1, 2), np.nan), np.full((1, 2), np.inf)):
            with self.assertRaises(ValueError):
                encode_context_v3(values, np.zeros((1, 2)), [2, 3], pad_token_id=0, cls_token_id=1)
        for shape in ((0, 2), (2, 0)):
            with self.assertRaises(ValueError):
                encode_context_v3(np.zeros(shape), np.zeros(shape), np.arange(shape[1]),
                                  pad_token_id=0, cls_token_id=1)
        for hidden in ([-1], [6]):
            with self.assertRaisesRegex(ValueError, "measured-panel"):
                self.encode(masked_genes=hidden)
        with self.assertRaisesRegex(ValueError, "empty batch"):
            collate_v3([])


class UniformQueryTests(unittest.TestCase):
    def test_paired_mask_is_independent_of_query_budget(self):
        for panel_size in (1, 6, 343, 5000):
            hidden, selected = sample_paired_queries_v3(panel_size, 128, .5, np.random.default_rng(91))
            other_hidden, other_selected = sample_paired_queries_v3(panel_size, 2, .5, np.random.default_rng(91))
            np.testing.assert_array_equal(hidden, other_hidden)
            self.assertEqual(len(hidden), int(np.ceil(panel_size / 2)))
            self.assertEqual(len(selected), min(128, len(hidden)))
            self.assertEqual(len(np.unique(selected)), len(selected))
            self.assertTrue(np.isin(selected, hidden).all())
            self.assertTrue(np.isin(other_selected, hidden).all())

    def test_queries_use_uniform_choice_over_allowed_panel_not_support_groups(self):
        class RecordingRng:
            def choice(self, eligible, count, replace):
                self.call = (eligible.copy(), count, replace)
                return eligible[:count]
        rng = RecordingRng()
        selected = sample_uniform_queries_v3(100, 2, rng, candidates=[1, 8, 10, 25])
        np.testing.assert_array_equal(rng.call[0], [1, 8, 10, 25])
        self.assertEqual(rng.call[1:], (2, False))
        np.testing.assert_array_equal(selected, [1, 8])
        for candidates in ([], [0, 0], [-1], [100]):
            with self.assertRaises(ValueError):
                sample_uniform_queries_v3(100, 2, rng, candidates=candidates)


class V3CachedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "cache" / "slide"
        self.folder.mkdir(parents=True)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        vocab = {"<pad>": 0, "<cls>": 1, **{f"G{i}": i + 2 for i in range(6)}}
        vocab_path = self.assets / "vocab.json"
        vocab_path.write_text(json.dumps(vocab))
        self.vocab_sha = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
        self.nuclear = np.log1p(np.array([[0, 1, 2, 3, 0, 1], [4, 1, 3, 1, 1, 2]], dtype=np.float32))
        self.cyto = np.log1p(np.array([[2, 0, 1, 0, 4, 1], [1, 3, 1, 2, 1, 1]], dtype=np.float32))
        for filename, value in {
            "nuclear_log1p_float32.npy": self.nuclear,
            "cyto_log1p_float32.npy": self.cyto,
            "gene_ids.npy": np.arange(2, 8),
            "coords.npy": np.array([[0., 0.], [1., 1.]], dtype=np.float32),
            "train_nodes.npy": np.array([[0, 1, -1], [1, 0, -1]], dtype=np.int64),
            "val_nodes.npy": np.array([[0, 1, -1]], dtype=np.int64),
        }.items():
            np.save(self.folder / filename, value)
        self.data = dict(context_cells=3, context_tokens_per_compartment=6,
                         query_genes=8, paired_mask_ratio=.5, seed=42)

    def dataset(self, split="train", query_genes=8):
        data = dict(self.data, query_genes=query_genes)
        manifest = dict(schema=CACHE_SCHEMA, data_config=data, vocab_sha256=self.vocab_sha,
                        slices=[dict(name="slide", cache_dir=str(self.folder))])
        (self.folder.parent / "manifest.json").write_text(json.dumps(manifest))
        return V3Dataset(dict(cache_root=str(self.folder.parent), assets_dir=str(self.assets), data=data), split=split)

    def examples_by_mode(self, dataset):
        found = {}
        for epoch in range(200):
            dataset.set_epoch(epoch)
            item = dataset[0]
            found.setdefault(int(item["task_mode"]), (epoch, item))
            if len(found) == 3:
                return found
        self.fail("Deterministic fixture did not exercise all three mask modes")

    def test_reuses_cache_and_all_modes_share_full_target_table(self):
        dataset = self.dataset()
        self.assertIsInstance(dataset.slices[0]["nuclear"], np.memmap)
        unmasked = encode_context_v3(self.nuclear, self.cyto, np.arange(2, 8),
            node_indices=[0, 1], seed=42, slice_seed=stable_seed("slide"),
            pad_token_id=0, cls_token_id=1, context_tokens_per_compartment=6)
        expected = token_bins(unmasked)
        modes = self.examples_by_mode(dataset)
        for mode, (_, item) in modes.items():
            with self.subTest(mode=mode):
                self.assertNotIn("target_log_counts", item)
                self.assertNotIn("neighbor_query_mask", item)
                self.assertEqual(item["target_bins"].dtype, np.dtype("int64"))
                for gene, comp, target, valid in zip(item["query_gene_ids"], item["query_compartment_ids"],
                                                     item["target_bins"], item["query_valid_mask"]):
                    if valid:
                        self.assertEqual(target, expected.get((int(comp), int(gene)), 0))
                        self.assertNotIn((int(comp), int(gene)), token_bins(item))
                for key, value in token_bins(item).items():
                    self.assertEqual(value, expected[key])
                np.testing.assert_array_equal(item["target_nonzero"], item["target_bins"] > 0)
                for comp in (0, 1):
                    observed = (item["token_valid_mask"][0] & (item["compartment_ids"][0] == comp)).any()
                    self.assertEqual(item["compartment_visible_mask"][comp], observed)
                padding = item["query_gene_ids"] == dataset.pad_id
                self.assertTrue((item["target_bins"][padding] == -2).all())
                self.assertFalse(item["query_valid_mask"][padding].any())
        self.assertFalse(modes[2][1]["compartment_visible_mask"].any())
        self.assertEqual(modes[1][1]["compartment_visible_mask"].sum(), 1)

    def test_query_sampling_is_unchanged_by_target_expression_support(self):
        dataset = self.dataset(query_genes=2)
        epochs = self.examples_by_mode(dataset)
        before = {mode: item for mode, (_, item) in epochs.items()}
        dataset.slices[0]["nuclear"] = np.full_like(self.nuclear, np.log1p(7))
        dataset.slices[0]["cyto"] = np.full_like(self.cyto, np.log1p(3))
        for mode, (epoch, _) in epochs.items():
            dataset.set_epoch(epoch)
            changed = dataset[0]
            for key in ("query_gene_ids", "query_compartment_ids", "query_valid_mask", "task_mode"):
                np.testing.assert_array_equal(changed[key], before[mode][key])

    def test_validation_is_fixed_and_collation_preserves_target_masks(self):
        dataset = self.dataset(split="val")
        before = dataset[0]
        dataset.set_epoch(99)
        after = dataset[0]
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        batch = collate_v3([before, after])
        self.assertEqual(tuple(batch["compartment_visible_mask"].shape), (2, 2))
        self.assertEqual(tuple(batch["target_bins"].shape), (2, 16))
        self.assertEqual(batch["gene_ids"].shape[2], int(before["token_valid_mask"].sum(axis=1).max()))
        np.testing.assert_array_equal(batch["query_valid_mask"].numpy(), np.stack([before["query_valid_mask"]] * 2))
        self.assertTrue(batch["token_valid_mask"][:, :, 0].all())

    def test_whole_cell_mask_requires_real_nonself_cytoplasm_neighbor(self):
        dataset = self.dataset()
        epoch, _ = self.examples_by_mode(dataset)[2]
        dataset.set_epoch(epoch)
        dataset.slices[0]["nodes"] = np.array([[0, 0, -1], [1, 1, -1]], dtype=np.int64)
        self.assertEqual(int(dataset[0]["task_mode"]), 0)
        dataset.slices[0]["nodes"] = np.array([[0, 1, -1], [1, 0, -1]], dtype=np.int64)
        cyto = self.cyto.copy()
        cyto[1] = 0
        dataset.slices[0]["cyto"] = cyto
        self.assertEqual(int(dataset[0]["task_mode"]), 0)

    def test_all_zero_observations_and_missing_patch_centers(self):
        dataset = self.dataset()
        dataset.slices[0]["nuclear"] = np.zeros_like(self.nuclear)
        dataset.slices[0]["cyto"] = np.zeros_like(self.cyto)
        item = dataset[0]
        self.assertFalse(item["compartment_visible_mask"].any())
        self.assertTrue((item["target_bins"][item["query_valid_mask"]] == 0).all())
        self.assertFalse(item["target_nonzero"].any())
        self.assertEqual(item["token_valid_mask"].sum(), 3)
        dataset.slices[0]["nodes"] = np.array([[-1, 1, -1], [1, 0, -1]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "center cannot be padding"):
            dataset[0]
        with self.assertRaises(IndexError):
            dataset[-1]
        np.save(self.folder / "train_nodes.npy", np.empty((0, 3), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "No prepared train patches"):
            self.dataset()


if __name__ == "__main__":
    unittest.main()
