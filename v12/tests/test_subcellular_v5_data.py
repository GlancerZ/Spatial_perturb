"""V5 whole-cell visibility and collation tests; no model weights required."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA, stable_seed
from spatial_pretrain.data.subcellular_v5 import (
    V5Dataset,
    collate_v5,
    collate_whole_cell_v5,
    encode_context_v5,
    encode_whole_cell_v5,
    recover_raw_counts_v5,
)

class WholeCellHelperTests(unittest.TestCase):
    def setUp(self):
        self.nuclear = np.array([0, 1, 2, 2, 7, 5, 0, 3], dtype=np.int64)
        self.cyto = np.array([2, 1, 0, 2, 1, 0, 0, 3], dtype=np.int64)
        self.genes = np.arange(10, 18, dtype=np.int64)

    def encode(self, **kwargs):
        return encode_whole_cell_v5(
            self.nuclear, self.cyto, self.genes, pad_token_id=0,
            cls_token_id=1, **kwargs)

    def test_bins_full_visible_panel_before_sampling(self):
        full = self.encode(seed=91, max_tokens=1200)
        shortened = self.encode(seed=91, max_tokens=5)
        full_bins = dict(zip(full["whole_gene_ids"][1:], full["whole_input_bins"][1:]))
        self.assertEqual(len(shortened["whole_gene_ids"]), 5)
        for gene, value in zip(
                shortened["whole_gene_ids"][1:], shortened["whole_input_bins"][1:]):
            self.assertEqual(value, full_bins[gene])
        self.assertEqual(shortened["whole_gene_ids"][0], 1)
        self.assertEqual(shortened["whole_input_bins"][0], -2)
        self.assertTrue(((full["whole_input_bins"][1:] >= 1)
                         & (full["whole_input_bins"][1:] <= 50)).all())

    def test_every_mask_is_applied_before_sum_and_binning(self):
        cases = (
            dict(masked_genes=np.array([1, 4])),
            dict(masked_compartment=0),
            dict(masked_compartment=1),
            dict(mask_whole_cell=True),
        )
        for mask in cases:
            with self.subTest(mask=mask):
                changed_n, changed_c = self.nuclear.copy(), self.cyto.copy()
                if "masked_genes" in mask:
                    changed_n[mask["masked_genes"]] = [100, 90]
                    changed_c[mask["masked_genes"]] = [80, 70]
                elif mask.get("masked_compartment") == 0:
                    changed_n[:] = np.arange(100, 108)
                elif mask.get("masked_compartment") == 1:
                    changed_c[:] = np.arange(100, 108)
                else:
                    changed_n[:] = np.arange(100, 108)
                    changed_c[:] = np.arange(200, 208)
                before = self.encode(seed=31, **mask)
                after = encode_whole_cell_v5(
                    changed_n, changed_c, self.genes, pad_token_id=0,
                    cls_token_id=1, seed=31, **mask)
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key])
                if "masked_genes" in mask:
                    self.assertFalse(np.isin(
                        before["whole_gene_ids"], self.genes[mask["masked_genes"]]).any())
                if mask.get("mask_whole_cell"):
                    np.testing.assert_array_equal(before["whole_gene_ids"], [1])

    def test_zeros_are_omitted_and_empty_cell_retains_only_cls(self):
        empty = encode_whole_cell_v5(
            np.zeros(8, dtype=np.int64), np.zeros(8, dtype=np.int64), self.genes,
            pad_token_id=0, cls_token_id=1, seed=9)
        np.testing.assert_array_equal(empty["whole_gene_ids"], [1])
        np.testing.assert_array_equal(empty["whole_input_bins"], [-2])
        np.testing.assert_array_equal(empty["whole_token_valid_mask"], [True])
        nonempty = self.encode(seed=9)
        self.assertFalse(np.isin(nonempty["whole_gene_ids"], self.genes[[6]]).any())

    def test_invalid_counts_masks_rng_and_budget_fail_loudly(self):
        for bad in (np.array([-1, 0]), np.array([1.5, 0]), np.array([np.nan, 0])):
            with self.assertRaises(ValueError):
                encode_whole_cell_v5(
                    bad, np.zeros(2), [2, 3], pad_token_id=0, cls_token_id=1)
        with self.assertRaisesRegex(ValueError, "either rng or seed"):
            self.encode(rng=np.random.default_rng(1), seed=1)
        with self.assertRaises(ValueError):
            self.encode(masked_genes=[1, 1])
        with self.assertRaises(ValueError):
            self.encode(max_tokens=0)


class RawRecoveryTests(unittest.TestCase):
    def test_exact_float32_cache_values_recover_discrete_counts(self):
        raw = np.array([0, 1, 2, 19, 100_000], dtype=np.int64)
        cached = np.log1p(raw.astype(np.float32))
        np.testing.assert_array_equal(recover_raw_counts_v5(cached), raw)

    def test_wrong_dtype_noninteger_encoding_and_ambiguous_range_are_rejected(self):
        with self.assertRaises(TypeError):
            recover_raw_counts_v5(np.log1p(np.array([1], dtype=np.float64)))
        with self.assertRaisesRegex(ValueError, "exact float32"):
            recover_raw_counts_v5(np.array([.1], dtype=np.float32))
        # At this log scale adjacent integer counts collapse to one float32 value.
        with self.assertRaisesRegex(ValueError, "uniquely"):
            recover_raw_counts_v5(np.array([20.], dtype=np.float32))


class ContextAndCollationTests(unittest.TestCase):
    def setUp(self):
        self.raw_n = np.array([[1, 2, 3, 4, 5, 6], [2, 1, 0, 3, 1, 2]], dtype=np.int64)
        self.raw_c = np.array([[6, 5, 4, 3, 2, 1], [1, 2, 3, 0, 2, 1]], dtype=np.int64)
        self.nuclear = np.log1p(self.raw_n.astype(np.float32))
        self.cyto = np.log1p(self.raw_c.astype(np.float32))
        self.genes = np.arange(2, 8)

    def encode(self, **kwargs):
        return encode_context_v5(
            self.nuclear, self.cyto, self.genes, node_indices=[21, 50],
            context_tokens_per_compartment=6, pad_token_id=0, cls_token_id=1,
            whole_cell_max_tokens=1200, **kwargs)

    def test_hidden_target_changes_do_not_change_whole_branch(self):
        hidden = np.array([0, 4])
        before = self.encode(masked_genes=hidden)
        changed_n, changed_c = self.raw_n.copy(), self.raw_c.copy()
        changed_n[0, hidden] = [100, 90]
        changed_c[0, hidden] = [80, 70]
        after = encode_context_v5(
            np.log1p(changed_n.astype(np.float32)), np.log1p(changed_c.astype(np.float32)),
            self.genes, node_indices=[21, 50], context_tokens_per_compartment=6,
            pad_token_id=0, cls_token_id=1, masked_genes=hidden)
        for key in ("whole_gene_ids", "whole_input_bins", "whole_token_valid_mask"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(np.isin(before["whole_gene_ids"], self.genes[hidden]).any())
        # V3 compatibility is intentional: its visible ranks use a pre-mask table.
        visible = before["token_valid_mask"][0] & (before["compartment_ids"][0] < 2)
        before_map = dict(zip(
            zip(before["compartment_ids"][0, visible], before["gene_ids"][0, visible]),
            before["input_bins"][0, visible]))
        visible = after["token_valid_mask"][0] & (after["compartment_ids"][0] < 2)
        after_map = dict(zip(
            zip(after["compartment_ids"][0, visible], after["gene_ids"][0, visible]),
            after["input_bins"][0, visible]))
        self.assertTrue(any(before_map[key] != after_map[key] for key in before_map))

    def test_raw_eval_input_matches_checked_cache_recovery(self):
        cached = self.encode(masked_compartment=0)
        direct = self.encode(
            masked_compartment=0, whole_nuclear_counts=self.raw_n[0],
            whole_cyto_counts=self.raw_c[0])
        for key in ("whole_gene_ids", "whole_input_bins", "whole_token_valid_mask"):
            np.testing.assert_array_equal(cached[key], direct[key])

    def test_variable_length_whole_collation_has_independent_padding(self):
        full = self.encode()
        empty = self.encode(mask_whole_cell=True)
        whole = collate_whole_cell_v5([full, empty], pad_token_id=0)
        self.assertEqual(tuple(whole["whole_gene_ids"].shape), (2, len(full["whole_gene_ids"])))
        self.assertEqual(int(whole["whole_token_valid_mask"][1].sum()), 1)
        self.assertTrue((whole["whole_gene_ids"][1, 1:] == 0).all())
        self.assertTrue((whole["whole_input_bins"][1, 1:] == -2).all())
        batch = collate_v5([full, empty], pad_token_id=0)
        self.assertEqual(batch["gene_ids"].ndim, 3)
        self.assertEqual(batch["whole_gene_ids"].ndim, 2)


class V5CachedDatasetTests(unittest.TestCase):
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
        self.raw_n = np.array([[0, 1, 2, 3, 4, 1], [4, 1, 3, 1, 1, 2]], dtype=np.int64)
        self.raw_c = np.array([[2, 4, 1, 2, 1, 1], [1, 3, 1, 2, 1, 1]], dtype=np.int64)
        for filename, value in {
            "nuclear_log1p_float32.npy": np.log1p(self.raw_n.astype(np.float32)),
            "cyto_log1p_float32.npy": np.log1p(self.raw_c.astype(np.float32)),
            "gene_ids.npy": np.arange(2, 8),
            "coords.npy": np.array([[0., 0.], [1., 1.]], dtype=np.float32),
            "train_nodes.npy": np.array([[0, 1, -1], [1, 0, -1]], dtype=np.int64),
            "val_nodes.npy": np.array([[0, 1, -1]], dtype=np.int64),
        }.items():
            np.save(self.folder / filename, value)
        self.cache_data = dict(
            context_cells=3, context_tokens_per_compartment=6,
            query_genes=8, paired_mask_ratio=.5, seed=42)
        manifest = dict(
            schema=CACHE_SCHEMA, data_config=self.cache_data,
            vocab_sha256=self.vocab_sha,
            slices=[dict(name="slide", cache_dir=str(self.folder))])
        (self.folder.parent / "manifest.json").write_text(json.dumps(manifest))

    def dataset(self, split="train"):
        runtime_data = dict(self.cache_data, whole_cell_max_tokens=5)
        return V5Dataset(dict(
            cache_root=str(self.folder.parent), assets_dir=str(self.assets),
            data=runtime_data), split=split)

    def examples_by_mode(self, dataset):
        found = {}
        for epoch in range(200):
            dataset.set_epoch(epoch)
            item = dataset[0]
            found.setdefault(int(item["task_mode"]), item)
            if len(found) == 3:
                return found
        self.fail("Deterministic fixture did not exercise all three mask modes")

    def test_runtime_whole_budget_does_not_change_v2_cache_manifest_contract(self):
        dataset = self.dataset()
        self.assertEqual(dataset.whole_cell_max_tokens, 5)
        self.assertNotIn("whole_cell_max_tokens", dataset.data)
        self.assertIsInstance(dataset.slices[0]["nuclear"], np.memmap)

    def test_all_v3_modes_apply_the_same_center_mask_to_whole_input(self):
        dataset = self.dataset()
        modes = self.examples_by_mode(dataset)
        for mode, item in modes.items():
            with self.subTest(mode=mode):
                self.assertLessEqual(len(item["whole_gene_ids"]), 5)
                self.assertEqual(item["whole_gene_ids"].ndim, 1)
                if mode == 0:
                    selected = np.unique(item["query_gene_ids"][item["query_valid_mask"]])
                    self.assertFalse(np.isin(item["whole_gene_ids"], selected).any())
                elif mode == 1:
                    self.assertEqual(item["compartment_visible_mask"].sum(), 1)
                else:
                    np.testing.assert_array_equal(item["whole_gene_ids"], [dataset.cls_id])
                    self.assertFalse(item["compartment_visible_mask"].any())

    def test_validation_and_whole_cell_seed_are_fixed(self):
        dataset = self.dataset(split="val")
        before = dataset[0]
        dataset.set_epoch(99)
        after = dataset[0]
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        expected = encode_whole_cell_v5(
            self.raw_n[0], self.raw_c[0], np.arange(2, 8),
            pad_token_id=0, cls_token_id=1, max_tokens=5,
            seed=np.random.SeedSequence([42, stable_seed("slide"), 0, 355]),
            masked_genes=np.unique(before["query_gene_ids"][before["query_valid_mask"]]) - 2
            if int(before["task_mode"]) == 0 else None,
            masked_compartment=(int(np.flatnonzero(~before["compartment_visible_mask"])[0])
                                if int(before["task_mode"]) == 1 else None),
            mask_whole_cell=int(before["task_mode"]) == 2)
        for key in expected:
            np.testing.assert_array_equal(before[key], expected[key])

    def test_collate_v5_keeps_patch_and_center_ranks_separate(self):
        dataset = self.dataset()
        items = self.examples_by_mode(dataset)
        batch = collate_v5([items[0], items[2]])
        self.assertEqual(tuple(batch["gene_ids"].shape[:2]), (2, 3))
        self.assertEqual(batch["whole_gene_ids"].shape[0], 2)
        self.assertEqual(batch["whole_gene_ids"].ndim, 2)


if __name__ == "__main__":
    unittest.main()
