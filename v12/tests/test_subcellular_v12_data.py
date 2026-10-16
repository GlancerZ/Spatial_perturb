"""V12 raw-count alignment, held-out visibility and LR direction contracts.

Run only on a compute node. Synthetic fixtures need no pretrained checkpoint.
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
from spatial_pretrain.data.subcellular_v12 import (
    V12Dataset, V12_TARGET_METADATA, collate_v12, encode_v12_views,
    prepare_view_batch_v12, prepare_view_v12,
)


TARGET_FIELDS = {"query_nucleus_counts", "query_cytoplasm_counts", "query_total_counts"}


class V12ViewTests(unittest.TestCase):
    def setUp(self):
        self.genes = np.arange(2, 22)
        self.nuclear = np.stack([np.arange(1, 21), np.arange(20, 0, -1), np.ones(20)]).astype(np.int64)
        self.cyto = np.stack([np.arange(20, 0, -1), np.ones(20), np.arange(1, 21)]).astype(np.int64)
        self.nodes = np.array([9, 21, 50])
        self.coords = np.array([[1., 1.], [2., 1.], [4., 1.]])
        self.pairs = np.array([[2, 3], [5, 8], [3, 2]])

    def encode(self, nuclear=None, cyto=None, **kwargs):
        options = dict(pad_token_id=0, cls_token_id=1, max_tokens=32,
                       neighbors=3, query_genes=4, lr_pairs=self.pairs)
        options.update(kwargs)
        return encode_v12_views(self.nuclear if nuclear is None else nuclear,
                                self.cyto if cyto is None else cyto,
                                self.genes, self.nodes, self.coords, **options)

    def test_raw_paired_targets_include_observed_zeros_not_absent_genes(self):
        nuclear, cyto = self.nuclear.copy(), self.cyto.copy()
        nuclear[0, :3], cyto[0, :3] = 0, 0
        result = self.encode(nuclear, cyto, query_genes=25, mask=False)
        valid = result["query_valid_mask"]
        self.assertEqual(valid.sum(), 20)
        self.assertEqual(set(result["query_gene_ids"][valid]), set(self.genes))
        columns = np.searchsorted(self.genes, result["query_gene_ids"][valid])
        np.testing.assert_array_equal(result["query_nucleus_counts"][valid], nuclear[0, columns])
        np.testing.assert_array_equal(result["query_cytoplasm_counts"][valid], cyto[0, columns])
        np.testing.assert_array_equal(result["query_total_counts"],
                                      result["query_nucleus_counts"] + result["query_cytoplasm_counts"])
        self.assertEqual((result["query_total_counts"][valid] == 0).sum(), 3)
        for field in TARGET_FIELDS:
            self.assertEqual(result[field].dtype, np.int64)
            self.assertFalse(result[field][~valid].any())
        self.assertEqual(V12_TARGET_METADATA["unmeasured"], "excluded_not_zero")
        self.assertFalse(V12_TARGET_METADATA["temporal_target"])

    def test_query_perturbation_cannot_change_either_view_statistics_or_lr(self):
        options = dict(gene_mask_ratio=.3, neighbor_dropout=0., seed=4, epoch=3)
        before = self.encode(**options)
        hidden = np.isin(self.genes, before["query_gene_ids"][before["query_valid_mask"]])
        nuclear, cyto = self.nuclear.copy(), self.cyto.copy()
        nuclear[0, hidden] += 100_000
        cyto[0, hidden] += 3_000_000
        after = self.encode(nuclear, cyto, **options)
        for key in before:
            if key not in TARGET_FIELDS:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        for key in TARGET_FIELDS:
            self.assertFalse(np.array_equal(before[key], after[key]))
        for prefix in ("nucleus", "cytoplasm"):
            visible = before[f"{prefix}_gene_ids"][before[f"{prefix}_valid"]]
            self.assertFalse(np.isin(self.genes[hidden], visible).any())

    def test_independent_random_masks_do_not_leak_via_count_features_or_lr(self):
        options = dict(gene_mask_ratio=.5, neighbor_dropout=0., seed=17, epoch=4)
        before = self.encode(**options)
        nuclear, cyto = self.nuclear.copy(), self.cyto.copy()
        for prefix, values in (("nucleus", nuclear), ("cytoplasm", cyto)):
            visible = before[f"{prefix}_gene_ids"][before[f"{prefix}_valid"]]
            hidden = ~np.isin(self.genes, visible)
            self.assertTrue(hidden.any())
            values[0, hidden] += 100_000
        for slot, row in enumerate([1, 2]):
            visible = before["neighbor_gene_ids"][slot, before["neighbor_valid"][slot]]
            hidden = ~np.isin(self.genes, visible)
            nuclear[row, hidden] += 1_000_000
        after = self.encode(nuclear, cyto, **options)
        for key in before:
            if key not in TARGET_FIELDS:
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_permutation_keeps_query_mask_and_token_count_alignment(self):
        permutation = np.array([8, 1, 5, 19, 0, 4, 6, 7, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        options = dict(pad_token_id=0, cls_token_id=1, neighbors=3, max_tokens=6,
                       query_genes=4, seed=16, epoch=2, lr_pairs=self.pairs)
        before = encode_v12_views(self.nuclear, self.cyto, self.genes,
                                  self.nodes, self.coords, **options)
        after = encode_v12_views(self.nuclear[:, permutation], self.cyto[:, permutation],
                                 self.genes[permutation], self.nodes, self.coords, **options)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        for prefix, raw in (("nucleus", self.nuclear), ("cytoplasm", self.cyto)):
            valid = before[f"{prefix}_valid"] & (before[f"{prefix}_gene_ids"] != 1)
            columns = np.searchsorted(self.genes, before[f"{prefix}_gene_ids"][valid])
            np.testing.assert_allclose(before[f"{prefix}_log_counts"][valid], np.log1p(raw[0, columns]), rtol=1e-6)
            self.assertEqual(before[f"{prefix}_log_counts"][0], 0.)
            self.assertFalse(before[f"{prefix}_log_counts"][~before[f"{prefix}_valid"]].any())

    def test_visible_statistics_use_count_sums_not_bins_or_hidden_targets(self):
        output = self.encode(mask=True, gene_mask_ratio=0., neighbor_dropout=0.)
        visible = ~np.isin(self.genes, output["query_gene_ids"][output["query_valid_mask"]])
        expected = np.log1p([self.nuclear[0, visible].sum(), self.cyto[0, visible].sum(),
                            np.count_nonzero(self.nuclear[0, visible]),
                            np.count_nonzero(self.cyto[0, visible])])
        np.testing.assert_allclose(output["center_statistics"], expected, rtol=1e-6)

    def test_unmasked_views_retain_ordinary_scgpt_tokenization(self):
        output = self.encode(mask=False, slice_seed=123, seed=7)
        for prefix, nuclear, cyto, row, slot in (
            ("nucleus", self.nuclear[0], np.zeros(20, dtype=np.int64), 0, None),
            ("cytoplasm", self.cyto[0], np.zeros(20, dtype=np.int64), 0, None),
            ("neighbor", self.nuclear[1], self.cyto[1], 1, 0),
        ):
            expected = encode_whole_cell_v6_left(
                nuclear, cyto, self.genes, pad_token_id=0, cls_token_id=1,
                max_tokens=32, seed=np.random.SeedSequence([7, 123, int(self.nodes[row]), 355]))
            length = len(expected["whole_gene_ids"])
            index = slice(0, length) if slot is None else (slot, slice(0, length))
            np.testing.assert_array_equal(output[f"{prefix}_gene_ids"][index], expected["whole_gene_ids"])
            np.testing.assert_array_equal(output[f"{prefix}_bins"][index], expected["whole_input_bins"])

    def test_center_lr_is_receiver_center_sender_neighbor_not_reverse(self):
        genes = np.array([2, 3, 4])  # ligand, receptor, other
        nuclei = np.array([[0, 10, 1], [20, 0, 1], [0, 30, 1]])
        result = encode_v12_views(
            nuclei, np.zeros_like(nuclei), genes, self.nodes, self.coords,
            pad_token_id=0, cls_token_id=1, max_tokens=8, neighbors=2,
            query_genes=1, mask=False, lr_pairs=np.array([[2, 3]]))
        self.assertGreater(result["center_lr_bias"][0], 0.)
        self.assertEqual(result["center_lr_bias"][1], 0.)
        self.assertGreater(result["neighbor_lr_bias"][1, 0], 0.)
        self.assertEqual(result["neighbor_lr_bias"][0, 1], 0.)
        np.testing.assert_array_equal(result["center_lr_coverage"], [1., 1.])

    def test_query_all_genes_has_no_center_lr_evidence_but_keeps_neighbors(self):
        output = self.encode(query_genes=20, gene_mask_ratio=0., neighbor_dropout=0.)
        for prefix in ("nucleus", "cytoplasm"):
            self.assertFalse(output[f"{prefix}_present"])
            self.assertEqual(output[f"{prefix}_valid"].sum(), 1)
            self.assertFalse(output[f"{prefix}_log_counts"].any())
        self.assertFalse(output["center_lr_bias"].any())
        self.assertFalse(output["center_lr_coverage"].any())
        self.assertFalse(output["center_statistics"].any())
        self.assertTrue(output["neighbor_present"][:2].all())
        self.assertGreater(output["neighbor_lr_bias"].sum(), 0.)

    def test_center_perturbation_never_changes_neighbor_inputs_or_graph(self):
        before = self.encode(mask=False)
        nuclear, cyto = self.nuclear.copy(), self.cyto.copy()
        nuclear[0] *= 100
        cyto[0] *= 200
        after = self.encode(nuclear, cyto, mask=False)
        for key in before:
            if key.startswith("neighbor_") or key == "relative_coords":
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_adding_center_to_lr_computation_preserves_isolated_neighbor_graph(self):
        output = self.encode(mask=False)
        isolated = compute_lr_features_v6(
            self.nuclear[1:] + self.cyto[1:], self.genes,
            node_indices=self.nodes[1:], lr_pairs=self.pairs)
        np.testing.assert_array_equal(output["neighbor_lr_bias"][:2, :2], isolated["lr_bias"])
        np.testing.assert_array_equal(output["neighbor_lr_coverage"][:2, :2], isolated["lr_coverage"])

    def test_duplicate_and_distant_centers_cannot_be_neighbor_evidence(self):
        nodes = np.array([9, 9, 50, 21, 21, 99, -1])
        coords = np.array([[0, 0], [0, 0], [3, 0], [1, 0], [1, 0], [101, 0], [0, 0]])
        nuclei = np.ones((7, 20), dtype=np.int64)
        output = encode_v12_views(nuclei, nuclei, self.genes, nodes, coords,
                                 pad_token_id=0, cls_token_id=1, neighbors=6,
                                 max_tokens=32, mask=False, lr_pairs=self.pairs)
        np.testing.assert_array_equal(output["neighbor_cell_identity"][:2, 1], [21, 50])
        self.assertFalse(output["neighbor_present"][2:].any())
        self.assertFalse(output["center_lr_bias"][2:].any())
        self.assertFalse(output["center_lr_coverage"][2:].any())
        self.assertFalse(output["neighbor_lr_bias"][2:].any())

    def test_collation_keeps_raw_tokens_aligned_and_zero_neighbors_supported(self):
        full = self.encode(mask=False)
        empty = self.encode(gene_mask_ratio=1., neighbor_dropout=1.)
        batch = collate_v12([full, empty])
        for prefix in ("nucleus", "cytoplasm"):
            self.assertEqual(tuple(batch[f"{prefix}_gene_ids"].shape), (2, 21))
            self.assertEqual(batch[f"{prefix}_log_counts"].shape, batch[f"{prefix}_gene_ids"].shape)
        self.assertEqual(tuple(batch["neighbor_lr_bias"].shape), (2, 3, 3))
        self.assertEqual(tuple(batch["center_lr_bias"].shape), (2, 3))
        self.assertEqual(tuple(batch["center_statistics"].shape), (2, 4))
        self.assertEqual(tuple(batch["query_total_counts"].shape), (2, 4))
        no_neighbors = collate_v12([self.encode(neighbors=0)])
        self.assertEqual(tuple(no_neighbors["neighbor_gene_ids"].shape), (1, 0, 1))
        self.assertEqual(tuple(no_neighbors["center_lr_bias"].shape), (1, 0))

    def test_embedding_extraction_is_epoch_independent_and_alias_exists(self):
        self.assertIs(prepare_view_v12, encode_v12_views)
        before = self.encode(mask=False, epoch=0, gene_mask_ratio=0., neighbor_dropout=0.)
        after = self.encode(mask=False, epoch=99, gene_mask_ratio=1., neighbor_dropout=1.)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)

    def test_batch_adapter_uses_explicit_requested_center_order(self):
        batch = prepare_view_batch_v12(
            self.nuclear, self.cyto, self.genes, np.array([2, 0]),
            np.array([[1, 0, -1], [2, 1, -1]]), self.coords,
            pad_token_id=0, cls_token_id=1, query_genes=20, mask=False)
        np.testing.assert_array_equal(batch["cell_identity"].numpy(), [[0, 2], [0, 0]])
        for item, row in enumerate([2, 0]):
            columns = np.searchsorted(self.genes, batch["query_gene_ids"][item].numpy())
            np.testing.assert_array_equal(batch["query_nucleus_counts"][item].numpy(), self.nuclear[row, columns])
        with self.assertRaisesRegex(ValueError, "selected centers"):
            prepare_view_batch_v12(self.nuclear, self.cyto, self.genes, np.array([2]),
                                   np.array([[1], [0], [1]]), self.coords,
                                   pad_token_id=0, cls_token_id=1)

    def test_invalid_counts_fail_instead_of_rounding_or_clipping(self):
        for value in (np.nan, np.inf, -1., .5):
            values = self.nuclear.astype(np.float64)
            values[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "integer counts"):
                self.encode(values)
        values = self.nuclear.astype(np.uint64)
        values[0, 0] = np.iinfo(np.uint64).max
        with self.assertRaisesRegex(ValueError, "integer counts"):
            self.encode(values)
        with self.assertRaisesRegex(ValueError, "zero expression"):
            self.encode(np.zeros_like(self.nuclear), np.zeros_like(self.cyto))


class V12CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        folder = root / "cache" / "slide"
        folder.mkdir(parents=True)
        assets = root / "assets"
        assets.mkdir()
        vocab = assets / "vocab.json"
        vocab.write_text(json.dumps({"<pad>": 0, "<cls>": 1, "A": 2, "B": 3, "C": 4}))
        self.nuclei = np.array([[2, 0, 3], [0, 0, 0], [0, 0, 0]], dtype=np.int64)
        self.cyto = np.array([[0, 0, 0], [2, 0, 1], [0, 0, 0]], dtype=np.int64)
        arrays = {"nuclear_log1p_float32.npy": np.log1p(self.nuclei.astype(np.float32)),
                  "cyto_log1p_float32.npy": np.log1p(self.cyto.astype(np.float32)),
                  "gene_ids.npy": np.array([2, 3, 4]),
                  "coords.npy": np.array([[0., 0.], [1., 2.], [10., 10.]], dtype=np.float32),
                  "train_nodes.npy": np.array([[0, 1, -1], [2, 0, 1], [1, 0, 2]]),
                  "val_nodes.npy": np.array([[0, 1, -1], [2, 0, 1], [1, 0, 2]])}
        for name, values in arrays.items():
            np.save(folder / name, values)
        self.cache_files = list(folder.glob("*.npy"))
        data = {"context_cells": 3, "context_tokens_per_compartment": 256, "seed": 42}
        manifest = {"schema": CACHE_SCHEMA, "data_config": data,
                    "vocab_sha256": hashlib.sha256(vocab.read_bytes()).hexdigest(),
                    "slices": [{"cache_dir": str(folder), "name": "tiny"}]}
        (root / "cache" / "manifest.json").write_text(json.dumps(manifest))
        self.cfg = {"cache_root": str(root / "cache"), "assets_dir": str(assets),
                    "data": dict(data, whole_cell_max_tokens=1200),
                    "v11": {"query_genes": 2},
                    "v12": {"neighbors": 2, "max_tokens": 16, "query_genes": 3,
                            "gene_mask_ratio": .1, "neighbor_dropout": .2}}

    def test_filter_keeps_single_compartment_cells_cache_and_config_unchanged(self):
        original = copy.deepcopy(self.cfg)
        hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in self.cache_files}
        dataset = V12Dataset(self.cfg, split="val")
        self.assertEqual(self.cfg, original)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.center_filter_metadata["original_centers"], 3)
        self.assertEqual(dataset.center_filter_metadata["excluded_centers"], 1)
        self.assertEqual(dataset.center_filter_metadata["slices"][0]["excluded_cell_ids"], [2])
        for i in range(2):
            item = dataset[i]
            self.assertEqual(item["cell_identity"].tolist(), [0, i])
            columns = item["query_gene_ids"] - 2
            np.testing.assert_array_equal(item["query_nucleus_counts"], self.nuclei[i, columns])
            np.testing.assert_array_equal(item["query_cytoplasm_counts"], self.cyto[i, columns])
        self.assertTrue(dataset[0]["nucleus_present"])
        self.assertFalse(dataset[0]["cytoplasm_present"])
        self.assertFalse(dataset[1]["nucleus_present"])
        self.assertTrue(dataset[1]["cytoplasm_present"])
        for path, digest in hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(dataset.target_metadata["expression"], "joint_raw_nuclear_cytoplasmic_counts")
        self.assertEqual(dataset.manifest["schema"], CACHE_SCHEMA)

    def test_validation_epoch_and_indices_stable_training_targets_hidden(self):
        dataset = V12Dataset(self.cfg, split="val")
        before = dataset[0]
        dataset.set_epoch(99)
        self.assertEqual(dataset.epoch, 0)
        for key in before:
            np.testing.assert_array_equal(before[key], dataset[0][key], err_msg=key)
        with self.assertRaises(IndexError):
            dataset[2]
        training = V12Dataset(self.cfg, split="train")
        training.set_epoch(7)
        self.assertEqual(training.epoch, 7)
        item = training[0]
        self.assertFalse(item["nucleus_present"])
        self.assertFalse(item["cytoplasm_present"])
        self.assertFalse(item["center_statistics"].any())
        self.assertEqual(int(item["query_total_counts"].sum()), 5)


if __name__ == "__main__":
    unittest.main()
