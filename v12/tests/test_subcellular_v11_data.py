"""V11 visibility, no-center LR, target and immutable-cache contracts.

Run on a compute node. No pretrained weights or real input data are required.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA
from spatial_pretrain.data.subcellular_v6 import compute_lr_features_v6
from spatial_pretrain.data.subcellular_v6_left import encode_whole_cell_v6_left
from spatial_pretrain.data.subcellular_v11 import (
    V11Dataset, V11_TARGET_METADATA, collate_v11, encode_v11_views, prepare_view_batch_v11,
)


class V11ViewTests(unittest.TestCase):
    def setUp(self):
        self.genes = np.arange(2, 22)
        self.nuclear = np.stack([np.arange(1, 21), np.arange(20, 0, -1), np.ones(20)]).astype(np.int64)
        self.cyto = np.zeros_like(self.nuclear)
        self.cyto[:, [0, 4, 6]] = [[3, 2, 9], [8, 2, 1], [1, 5, 6]]
        self.nodes = np.array([9, 21, 50])
        self.coords = np.array([[1., 1.], [2., 1.], [4., 1.]])
        self.pairs = np.array([[2, 3], [5, 8], [3, 2]])

    def encode(self, nuclear=None, cyto=None, **kwargs):
        options = dict(pad_token_id=0, cls_token_id=1, max_tokens=32,
                       neighbors=3, query_genes=4, lr_pairs=self.pairs)
        options.update(kwargs)
        return encode_v11_views(self.nuclear if nuclear is None else nuclear,
                                self.cyto if cyto is None else cyto,
                                self.genes, self.nodes, self.coords, **options)

    def test_queries_uniform_over_measured_panel_include_zero_and_never_unknown(self):
        nuclear, cyto = self.nuclear.copy(), self.cyto.copy()
        nuclear[0] = 0
        cyto[0] = 0
        nuclear[0, 0] = 10
        output = self.encode(nuclear, cyto, mask=False, query_genes=25)
        valid = output["query_valid_mask"]
        self.assertEqual(valid.sum(), 20)
        self.assertEqual(set(output["query_gene_ids"][valid]), set(self.genes))
        self.assertEqual((output["query_bins"][valid] == 0).sum(), 19)
        self.assertTrue((output["query_bins"][~valid] == -100).all())
        self.assertTrue((output["query_gene_ids"][~valid] == 0).all())
        alternate = self.encode(mask=False, query_genes=25)
        np.testing.assert_array_equal(output["query_gene_ids"], alternate["query_gene_ids"])
        self.assertEqual(V11_TARGET_METADATA["unmeasured"], "excluded_not_zero")

    def test_query_target_perturbation_cannot_change_inputs_or_lr(self):
        options = dict(gene_mask_ratio=.3, neighbor_dropout=0., seed=4, epoch=3)
        before = self.encode(**options)
        query = before["query_gene_ids"][before["query_valid_mask"]]
        nuclear = self.nuclear.copy()
        nuclear[0, np.isin(self.genes, query)] += 100_000
        cyto = self.cyto.copy()
        cyto[0] = np.arange(20, 0, -1) * 100
        after = self.encode(nuclear, cyto, **options)
        for key in before:
            if key != "query_bins":
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        self.assertFalse(np.array_equal(before["query_bins"], after["query_bins"]))
        visible_nuclear = before["nucleus_gene_ids"][before["nucleus_valid"]]
        self.assertFalse(np.isin(query, visible_nuclear).any())

    def test_center_cytoplasm_changes_only_supervision(self):
        before = self.encode(mask=False, query_genes=20)
        cyto = self.cyto.copy()
        cyto[0] = np.arange(20, 0, -1) * 1000
        after = self.encode(cyto=cyto, mask=False, query_genes=20)
        for key in before:
            if key != "query_bins":
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        self.assertFalse(np.array_equal(before["query_bins"], after["query_bins"]))
        self.assertFalse(any("teacher" in key or "whole" in key for key in before))

    def test_neighbor_uses_sum_of_raw_compartments_and_directed_visible_lr(self):
        output = self.encode(mask=False, slice_seed=123, seed=7)
        for slot, row in enumerate([1, 2]):
            expected = encode_whole_cell_v6_left(
                self.nuclear[row], self.cyto[row], self.genes,
                pad_token_id=0, cls_token_id=1, max_tokens=32,
                seed=np.random.SeedSequence([7, 123, int(self.nodes[row]), 355]))
            count = len(expected["whole_gene_ids"])
            np.testing.assert_array_equal(output["neighbor_gene_ids"][slot, :count], expected["whole_gene_ids"])
            np.testing.assert_array_equal(output["neighbor_bins"][slot, :count], expected["whole_input_bins"])
        expected_lr = compute_lr_features_v6(
            self.nuclear[1:] + self.cyto[1:], self.genes,
            node_indices=self.nodes[1:], lr_pairs=self.pairs)
        np.testing.assert_allclose(output["neighbor_lr_bias"][:2, :2], expected_lr["lr_bias"])
        np.testing.assert_allclose(output["neighbor_lr_coverage"][:2, :2], expected_lr["lr_coverage"])
        self.assertTrue((output["neighbor_lr_bias"][2] == 0).all())
        self.assertTrue((output["neighbor_lr_bias"][:, 2] == 0).all())
        self.assertTrue((np.diag(output["neighbor_lr_bias"]) == 0).all())

    def test_hidden_neighbor_genes_cannot_affect_tokens_lr_activity_or_depth(self):
        options = dict(gene_mask_ratio=.5, neighbor_dropout=0., seed=17, epoch=4)
        before = self.encode(**options)
        nuclear = self.nuclear.copy()
        for slot, row in enumerate([1, 2]):
            visible = before["neighbor_gene_ids"][slot, before["neighbor_valid"][slot]]
            hidden = ~np.isin(self.genes, visible)
            self.assertTrue(hidden.any())
            nuclear[row, hidden] += 100_000
        after = self.encode(nuclear, **options)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_duplicate_centers_neighbors_and_distant_cells_do_not_enter_lr_graph(self):
        nodes = np.array([9, 9, 50, 21, 21, 99, -1])
        coords = np.array([[0, 0], [0, 0], [3, 0], [1, 0], [1, 0], [101, 0], [0, 0]])
        nuclear = np.ones((7, 20), dtype=np.int64)
        cyto = np.ones_like(nuclear)
        output = encode_v11_views(
            nuclear, cyto, self.genes, nodes, coords, pad_token_id=0,
            cls_token_id=1, neighbors=6, query_genes=4, max_tokens=32,
            mask=False, lr_pairs=self.pairs, slice_index=6, max_neighbor_distance=100.)
        np.testing.assert_array_equal(output["neighbor_present"], [True, True, False, False, False, False])
        np.testing.assert_array_equal(output["neighbor_cell_identity"][:2], [[6, 21], [6, 50]])
        np.testing.assert_array_equal(output["neighbor_cell_identity"][2:], -np.ones((4, 2)))
        np.testing.assert_array_equal(output["relative_coords"][:2], [[1, 0], [3, 0]])
        self.assertFalse(output["neighbor_valid"][2:].any())
        self.assertFalse(output["neighbor_lr_bias"][2:].any())
        self.assertFalse(output["neighbor_lr_coverage"][:, 2:].any())

    def test_mask_changes_inputs_but_not_fixed_query_targets(self):
        before = self.encode(mask=False, epoch=0)
        after = self.encode(mask=True, epoch=0, gene_mask_ratio=1., neighbor_dropout=1.)
        for key in ("query_gene_ids", "query_bins", "query_valid_mask"):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(after["nucleus_present"])
        self.assertEqual(after["nucleus_valid"].sum(), 1)
        self.assertFalse(after["neighbor_present"].any())
        self.assertFalse(after["neighbor_lr_bias"].any())
        self.assertFalse(after["neighbor_lr_coverage"].any())
        self.assertTrue((after["neighbor_cell_identity"] == -1).all())

    def test_gene_column_order_does_not_change_queries_masks_truncation_or_bins(self):
        permutation = np.array([8, 1, 5, 19, 0, 4, 6, 7, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        options = dict(pad_token_id=0, cls_token_id=1, neighbors=3, max_tokens=6,
                       query_genes=4, seed=16, epoch=2, lr_pairs=self.pairs)
        original = encode_v11_views(self.nuclear, self.cyto, self.genes,
                                    self.nodes, self.coords, **options)
        reordered = encode_v11_views(self.nuclear[:, permutation], self.cyto[:, permutation],
                                     self.genes[permutation], self.nodes, self.coords, **options)
        for key in original:
            np.testing.assert_array_equal(original[key], reordered[key], err_msg=key)

    def test_eval_disables_augmentation_and_epoch_dependence(self):
        before = self.encode(mask=False, gene_mask_ratio=1., neighbor_dropout=1., epoch=99)
        after = self.encode(mask=False, gene_mask_ratio=0., neighbor_dropout=0., epoch=0)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_padding_trim_does_not_trim_lr_and_missing_streams_stay_explicit(self):
        full = self.encode(mask=False)
        empty = self.encode(gene_mask_ratio=1., neighbor_dropout=1.)
        batch = collate_v11([full, empty])
        self.assertEqual(tuple(batch["nucleus_gene_ids"].shape), (2, 21))
        self.assertEqual(tuple(batch["neighbor_gene_ids"].shape), (2, 3, 21))
        self.assertEqual(tuple(batch["neighbor_lr_bias"].shape), (2, 3, 3))
        self.assertEqual(tuple(batch["neighbor_cell_identity"].shape), (2, 3, 2))
        self.assertEqual(tuple(batch["query_bins"].shape), (2, 4))
        self.assertFalse(bool(batch["neighbor_valid"][1].any()))
        all_empty = collate_v11([empty, empty])
        self.assertEqual(tuple(all_empty["neighbor_gene_ids"].shape), (2, 3, 1))
        self.assertFalse(any("teacher" in key for key in batch))

    def test_batch_adapter_selects_requested_centers_without_global_index_guessing(self):
        batch = prepare_view_batch_v11(
            self.nuclear, self.cyto, self.genes, np.array([2, 0]),
            np.array([[1, 0, -1], [2, 1, -1]]), self.coords,
            pad_token_id=0, cls_token_id=1, query_genes=4, mask=False, lr_pairs=self.pairs)
        np.testing.assert_array_equal(batch["cell_identity"].numpy(), [[0, 2], [0, 0]])
        self.assertEqual(tuple(batch["neighbor_lr_bias"].shape), (2, 3, 3))
        with self.assertRaisesRegex(ValueError, "selected centers"):
            prepare_view_batch_v11(self.nuclear, self.cyto, self.genes, np.array([2]),
                                   np.array([[1], [0], [1]]), self.coords,
                                   pad_token_id=0, cls_token_id=1)

    def test_invalid_raw_counts_and_empty_center_fail(self):
        with self.assertRaisesRegex(ValueError, "integer counts"):
            self.encode(np.log1p(self.nuclear.astype(np.float32)))
        with self.assertRaisesRegex(ValueError, "zero expression"):
            self.encode(np.zeros_like(self.nuclear), np.zeros_like(self.cyto))
        with self.assertRaisesRegex(ValueError, "micrometers"):
            self.encode(max_neighbor_distance=-1)


class V11CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        folder = root / "cache" / "slide"
        folder.mkdir(parents=True)
        assets = root / "assets"
        assets.mkdir()
        vocab_path = assets / "vocab.json"
        vocab_path.write_text(json.dumps({"<pad>": 0, "<cls>": 1, "A": 2, "B": 3, "C": 4}))
        nuclear = np.array([[2, 0, 3], [0, 2, 0], [0, 0, 0]], dtype=np.int64)
        cyto = np.array([[1, 4, 3], [2, 0, 1], [0, 0, 0]], dtype=np.int64)
        arrays = {"nuclear_log1p_float32.npy": np.log1p(nuclear.astype(np.float32)),
                  "cyto_log1p_float32.npy": np.log1p(cyto.astype(np.float32)),
                  "gene_ids.npy": np.array([2, 3, 4]),
                  "coords.npy": np.array([[0., 0.], [1., 2.], [10., 10.]], dtype=np.float32),
                  "train_nodes.npy": np.array([[0, 1, -1], [2, 0, 1], [1, 0, 2]]),
                  "val_nodes.npy": np.array([[0, 1, -1], [2, 0, 1]])}
        for name, values in arrays.items():
            np.save(folder / name, values)
        data = {"context_cells": 3, "context_tokens_per_compartment": 256, "seed": 42}
        manifest = {"schema": CACHE_SCHEMA, "data_config": data,
                    "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
                    "slices": [{"cache_dir": str(folder), "name": "tiny"}]}
        (root / "cache" / "manifest.json").write_text(json.dumps(manifest))
        self.cfg = {"cache_root": str(root / "cache"), "assets_dir": str(assets),
                    "data": dict(data, whole_cell_max_tokens=1200),
                    "v11": {"neighbors": 2, "max_tokens": 16, "query_genes": 1,
                            "gene_mask_ratio": .1, "neighbor_dropout": .2}}

    def test_checked_raw_cache_and_auditable_filter_preserve_input_configuration(self):
        original = copy.deepcopy(self.cfg)
        dataset = V11Dataset(self.cfg, split="train")
        self.assertEqual(self.cfg, original)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.center_filter_metadata["original_centers"], 3)
        self.assertEqual(dataset.center_filter_metadata["excluded_centers"], 1)
        self.assertEqual(dataset.center_filter_metadata["slices"][0]["excluded_cell_ids"], [2])
        self.assertEqual(dataset[0]["cell_identity"].tolist(), [0, 0])
        self.assertEqual(dataset[1]["cell_identity"].tolist(), [0, 1])
        self.assertEqual(dataset.target_metadata["expression"], "relative_expression_deterministic_51_bins")
        self.assertEqual(dataset.lr_metadata["graph"], "neighbor_to_neighbor_center_excluded")
        self.assertFalse(any("teacher" in key for key in dataset[0]))

    def test_validation_epoch_and_center_indices_are_stable(self):
        dataset = V11Dataset(self.cfg, split="val")
        before = dataset[0]
        self.assertEqual(len(dataset), 1)
        dataset.set_epoch(99)
        self.assertEqual(dataset.epoch, 0)
        after = dataset[0]
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        with self.assertRaises(IndexError):
            dataset[1]


if __name__ == "__main__":
    unittest.main()
