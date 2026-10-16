"""Visibility, teacher pairing and immutable-cache tests for V10 data.

Run on a compute node; these tests require no pretrained weights or real data.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA, stable_seed
from spatial_pretrain.data.subcellular_v6_left import encode_whole_cell_v6_left
from spatial_pretrain.data.subcellular_v10 import (
    V10Dataset, collate_v10, encode_v10_views,
)


class V10ViewTests(unittest.TestCase):
    def setUp(self):
        self.nuclear = np.array([[1, 3, 2, 0], [2, 1, 3, 1], [1, 1, 1, 2]])
        self.cyto = np.array([[0, 1, 2, 4], [0, 0, 3, 0], [0, 2, 0, 0]])
        self.genes = np.arange(2, 6)
        self.nodes = np.array([9, 21, 50])
        self.coords = np.array([[1., 1.], [2., 1.], [4., 1.]])

    def encode(self, nuclear=None, cyto=None, **kwargs):
        return encode_v10_views(
            self.nuclear if nuclear is None else nuclear,
            self.cyto if cyto is None else cyto,
            self.genes, self.nodes, self.coords, pad_token_id=0,
            cls_token_id=1, max_tokens=8, neighbors=3, **kwargs)

    def test_teacher_exactly_matches_paired_ordinary_scgpt_and_is_epoch_stable(self):
        before = self.encode(seed=7, slice_seed=123, epoch=0)
        after = self.encode(seed=7, slice_seed=123, epoch=88, gene_mask_ratio=1,
                            neighbor_dropout=1)
        baseline = encode_whole_cell_v6_left(
            self.nuclear[0], self.cyto[0], self.genes,
            pad_token_id=0, cls_token_id=1, max_tokens=8,
            seed=np.random.SeedSequence([7, 123, 9, 355]))
        for key, ref in (("gene_ids", "whole_gene_ids"), ("bins", "whole_input_bins"),
                         ("valid", "whole_token_valid_mask")):
            np.testing.assert_array_equal(before[f"teacher_{key}"], after[f"teacher_{key}"])
            np.testing.assert_array_equal(before[f"teacher_{key}"][:len(baseline[ref])], baseline[ref])
        self.assertFalse(after["nucleus_present"])
        self.assertEqual(int(after["nucleus_valid"].sum()), 1)
        self.assertFalse(after["neighbor_present"].any())
        self.assertFalse(after["neighbor_valid"].any())

    def test_center_cytoplasm_never_enters_nucleus_or_neighbor_student(self):
        before = self.encode(mask=False)
        nuclear, cyto = self.nuclear.copy(), self.cyto.copy()
        cyto[0] = [900, 5, 1, 2]
        nuclear[1:] = 10_000
        after = self.encode(nuclear, cyto, mask=False)
        for prefix in ("nucleus", "neighbor"):
            for suffix in ("gene_ids", "bins", "valid"):
                np.testing.assert_array_equal(before[f"{prefix}_{suffix}"], after[f"{prefix}_{suffix}"])
        self.assertFalse(np.array_equal(before["teacher_bins"], after["teacher_bins"]))

    def test_overlength_teacher_uses_the_same_gene_sampling_as_paired_baseline(self):
        genes = np.arange(2, 66)
        nuclear = np.tile(np.arange(1, 9), 8)[None]
        cyto = np.ones_like(nuclear)
        output = encode_v10_views(
            nuclear, cyto, genes, np.array([8]), np.zeros((1, 2)),
            pad_token_id=0, cls_token_id=1, max_tokens=10,
            seed=19, slice_seed=21, mask=False)
        baseline = encode_whole_cell_v6_left(
            nuclear[0], cyto[0], genes, pad_token_id=0, cls_token_id=1,
            max_tokens=10, seed=np.random.SeedSequence([19, 21, 8, 355]))
        np.testing.assert_array_equal(output["teacher_gene_ids"], baseline["whole_gene_ids"])
        np.testing.assert_array_equal(output["teacher_bins"], baseline["whole_input_bins"])
        self.assertEqual(output["teacher_valid"].sum(), 10)

    def test_hidden_nucleus_values_cannot_change_visible_bins(self):
        genes = np.arange(2, 102)
        nuclear = np.arange(1, 101, dtype=np.int64)[None]
        cyto = np.ones_like(nuclear)
        kwargs = dict(pad_token_id=0, cls_token_id=1, neighbors=2,
                      max_tokens=128, gene_mask_ratio=.5, neighbor_dropout=0,
                      seed=17, epoch=4)
        before = encode_v10_views(nuclear, cyto, genes, np.array([2]), np.zeros((1, 2)), **kwargs)
        visible = before["nucleus_gene_ids"][before["nucleus_valid"]][1:]
        hidden = np.flatnonzero(~np.isin(genes, visible))
        self.assertTrue(0 < len(hidden) < 100)
        changed = nuclear.copy()
        changed[0, hidden] = 100_000
        after = encode_v10_views(changed, cyto, genes, np.array([2]), np.zeros((1, 2)), **kwargs)
        for suffix in ("gene_ids", "bins", "valid"):
            np.testing.assert_array_equal(before[f"nucleus_{suffix}"], after[f"nucleus_{suffix}"])

    def test_duplicate_centers_and_neighbors_are_excluded_before_nearest_selection(self):
        nuclear = np.ones((6, 4), dtype=np.int64)
        cyto = np.array([[9, 9, 9, 9], [90, 90, 90, 90], [0, 1, 0, 0],
                         [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        nodes = np.array([9, 9, 50, 21, 21, -1])
        coords = np.array([[1., 1.], [1., 1.], [4., 1.], [2., 1.], [2., 1.], [0., 0.]])
        output = encode_v10_views(nuclear, cyto, self.genes, nodes, coords,
                                 pad_token_id=0, cls_token_id=1, neighbors=4,
                                 max_tokens=8, mask=False, slice_index=6)
        np.testing.assert_array_equal(output["neighbor_present"], [True, True, False, False])
        np.testing.assert_array_equal(output["relative_coords"], [[1, 0], [3, 0], [0, 0], [0, 0]])
        np.testing.assert_array_equal(output["neighbor_gene_ids"][:2, 1], [4, 3])
        np.testing.assert_array_equal(output["cell_identity"], [6, 9])
        self.assertFalse(output["neighbor_valid"][2:].any())

    def test_eval_disables_augmentations_and_empty_nucleus_is_explicit(self):
        before = self.encode(mask=False, gene_mask_ratio=1, neighbor_dropout=1, epoch=999)
        after = self.encode(mask=False, gene_mask_ratio=0, neighbor_dropout=0, epoch=0)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        nuclear = self.nuclear.copy()
        nuclear[0] = 0
        empty = self.encode(nuclear, mask=False)
        self.assertFalse(empty["nucleus_present"])
        np.testing.assert_array_equal(empty["nucleus_gene_ids"][:2], [1, 0])
        self.assertEqual(empty["nucleus_valid"].sum(), 1)

    def test_invalid_counts_and_empty_teacher_fail(self):
        with self.assertRaisesRegex(ValueError, "integer counts"):
            self.encode(np.log1p(self.nuclear.astype(np.float32)))
        with self.assertRaisesRegex(ValueError, "zero expression"):
            self.encode(np.zeros_like(self.nuclear), np.zeros_like(self.cyto))
        with self.assertRaisesRegex(ValueError, r"\[0,1\]"):
            self.encode(gene_mask_ratio=-.1)

    def test_independent_collation_trims_padding_and_preserves_absent_neighbor(self):
        full = self.encode(mask=False)
        empty = self.encode(gene_mask_ratio=1, neighbor_dropout=1)
        batch = collate_v10([full, empty])
        self.assertEqual(tuple(batch["teacher_gene_ids"].shape), (2, 5))
        self.assertEqual(tuple(batch["nucleus_gene_ids"].shape), (2, 4))
        self.assertEqual(tuple(batch["neighbor_gene_ids"].shape), (2, 3, 2))
        self.assertEqual(int(batch["nucleus_valid"][1].sum()), 1)
        self.assertFalse(bool(batch["neighbor_valid"][1].any()))
        self.assertEqual(tuple(batch["relative_coords"].shape), (2, 3, 2))
        all_empty = collate_v10([empty, empty])
        self.assertEqual(tuple(all_empty["neighbor_gene_ids"].shape), (2, 3, 1))


class V10CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.folder = root / "cache" / "slide"
        self.folder.mkdir(parents=True)
        assets = root / "assets"
        assets.mkdir()
        vocab_path = assets / "vocab.json"
        vocab_path.write_text(json.dumps({"<pad>": 0, "<cls>": 1, "A": 2, "B": 3, "C": 4}))
        self.nuclear = np.array([[2, 0, 3], [0, 2, 0]], dtype=np.int64)
        self.cyto = np.array([[1, 4, 3], [2, 0, 1]], dtype=np.int64)
        self.coords = np.array([[0., 0.], [1., 2.]], dtype=np.float32)
        arrays = {"nuclear_log1p_float32.npy": np.log1p(self.nuclear.astype(np.float32)),
                  "cyto_log1p_float32.npy": np.log1p(self.cyto.astype(np.float32)),
                  "gene_ids.npy": np.array([2, 3, 4]), "coords.npy": self.coords,
                  "train_nodes.npy": np.array([[0, 1, -1], [1, 0, -1]]),
                  "val_nodes.npy": np.array([[0, 1, -1]])}
        for name, values in arrays.items():
            np.save(self.folder / name, values)
        data = {"context_cells": 3, "context_tokens_per_compartment": 256, "seed": 42}
        manifest = {"schema": CACHE_SCHEMA, "data_config": data,
                    "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
                    "slices": [{"cache_dir": str(self.folder), "name": "tiny"}]}
        (root / "cache" / "manifest.json").write_text(json.dumps(manifest))
        self.cfg = {"cache_root": str(root / "cache"), "assets_dir": str(assets),
                    "data": dict(data, whole_cell_max_tokens=1200),
                    "v10": {"neighbors": 2, "max_tokens": 16,
                            "gene_mask_ratio": .5, "neighbor_dropout": .2}}

    def test_checked_integer_recovery_matches_direct_eval_and_cache_signature_survives(self):
        dataset = V10Dataset(self.cfg, split="val")
        self.assertEqual(len(dataset), 1)
        dataset.set_epoch(9)
        self.assertEqual(dataset.epoch, 0)
        encoded = dataset[0]
        direct = encode_v10_views(
            self.nuclear, self.cyto, np.array([2, 3, 4]), np.array([0, 1]), self.coords,
            pad_token_id=0, cls_token_id=1, neighbors=2, max_tokens=16,
            slice_seed=stable_seed("tiny"), mask=False)
        for key in encoded:
            np.testing.assert_array_equal(encoded[key], direct[key])
        self.assertIn("whole_cell_max_tokens", self.cfg["data"])
        self.assertNotIn("whole_cell_max_tokens", dataset.manifest["data_config"])

    def test_training_masks_change_with_epoch_but_teacher_is_fixed(self):
        dataset = V10Dataset(self.cfg, split="train")
        self.assertEqual(len(dataset), 2)
        before = dataset[0]
        dataset.set_epoch(8)
        self.assertEqual(dataset.epoch, 8)
        after = dataset[0]
        for suffix in ("gene_ids", "bins", "valid"):
            np.testing.assert_array_equal(before[f"teacher_{suffix}"], after[f"teacher_{suffix}"])

    def test_noninteger_log_cache_is_rejected(self):
        bad = np.log1p(self.nuclear.astype(np.float32))
        bad[0, 0] = .1
        np.save(self.folder / "nuclear_log1p_float32.npy", bad)
        with self.assertRaisesRegex(ValueError, "exact float32 log1p"):
            V10Dataset(self.cfg, split="val")[0]


if __name__ == "__main__":
    unittest.main()
