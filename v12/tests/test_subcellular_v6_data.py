"""V6 visibility, directed LR, runtime-prior and paired-task regression tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA
from spatial_pretrain.data.subcellular_v3 import encode_context_v3
from spatial_pretrain.data.subcellular_v5 import V5Dataset, WHOLE_KEYS, encode_context_v5
from spatial_pretrain.data.subcellular_v6 import (
    V6Dataset, collate_v6, compute_lr_features_v6, encode_context_v6, load_lr_prior,
)


class VisibleContextTests(unittest.TestCase):
    def setUp(self):
        self.raw_n = np.array([[1, 2, 3, 4, 5, 6], [2, 1, 0, 3, 1, 2],
                               [1, 2, 3, 4, 5, 6], [8, 8, 8, 8, 8, 8]])
        self.raw_c = np.array([[6, 5, 4, 3, 2, 1], [1, 2, 3, 0, 2, 1],
                               [6, 5, 4, 3, 2, 1], [9, 9, 9, 9, 9, 9]])
        self.genes = np.arange(2, 8)
        self.options = dict(node_indices=[21, 50, 21, -1],
                            context_tokens_per_compartment=6, pad_token_id=0,
                            cls_token_id=1, lr_pairs=np.array([[2, 3], [4, 5]]))

    def encode(self, n=None, c=None, **kwargs):
        options = dict(self.options, **kwargs)
        return encode_context_v6(
            np.log1p((self.raw_n if n is None else n).astype(np.float32)),
            np.log1p((self.raw_c if c is None else c).astype(np.float32)),
            self.genes, **options)

    def test_hidden_values_cannot_change_any_input_in_all_mask_modes(self):
        masks = (dict(masked_genes=[0, 4]), dict(masked_compartment=0),
                 dict(masked_compartment=1), dict(mask_whole_cell=True))
        for mask in masks:
            with self.subTest(mask=mask):
                n, c = self.raw_n.copy(), self.raw_c.copy()
                center_rows = [0, 2]
                if "masked_genes" in mask:
                    n[np.ix_(center_rows, mask["masked_genes"])] = [[100, 90], [80, 70]]
                    c[np.ix_(center_rows, mask["masked_genes"])] = [[60, 50], [40, 30]]
                elif mask.get("masked_compartment") == 0:
                    n[center_rows] = np.arange(100, 112).reshape(2, 6)
                elif mask.get("masked_compartment") == 1:
                    c[center_rows] = np.arange(100, 112).reshape(2, 6)
                else:
                    n[center_rows] = np.arange(100, 112).reshape(2, 6)
                    c[center_rows] = np.arange(200, 212).reshape(2, 6)
                before, after = self.encode(**mask), self.encode(n, c, **mask)
                for key in before:
                    np.testing.assert_array_equal(before[key], after[key], err_msg=key)
                for key in ("gene_ids", "input_bins", "compartment_ids", "token_valid_mask"):
                    np.testing.assert_array_equal(before[key][0], before[key][2])
                if mask.get("mask_whole_cell") or mask.get("masked_compartment") == 1:
                    self.assertFalse(before["lr_bias"][[0, 2]].any())
                    self.assertFalse(before["lr_bias"][:, [0, 2]].any())
                    self.assertFalse(before["lr_coverage"][[0, 2]].any())

    def test_whole_branch_is_exactly_v5_paired_for_every_mask(self):
        for mask in ({}, dict(masked_genes=[0, 4]), dict(masked_compartment=0),
                     dict(masked_compartment=1), dict(mask_whole_cell=True)):
            options = {key: value for key, value in self.options.items() if key != "lr_pairs"}
            baseline = encode_context_v5(
                np.log1p(self.raw_n.astype(np.float32)),
                np.log1p(self.raw_c.astype(np.float32)), self.genes, **options, **mask)
            actual = self.encode(**mask)
            for key in WHOLE_KEYS:
                np.testing.assert_array_equal(actual[key], baseline[key])

    def test_unmasked_compartment_encoding_matches_v3_prepacking(self):
        options = {key: value for key, value in self.options.items() if key != "lr_pairs"}
        baseline = encode_context_v3(
            np.log1p(self.raw_n.astype(np.float32)),
            np.log1p(self.raw_c.astype(np.float32)), self.genes, **options)
        actual = self.encode()
        for key in baseline:
            np.testing.assert_array_equal(actual[key], baseline[key])

    def test_pad_rows_are_ignored_and_compartments_remain_decomposable(self):
        encoded = self.encode()
        n, c = self.raw_n.copy(), self.raw_c.copy()
        n[-1], c[-1] = 10_000, 0
        changed = self.encode(n, c)
        for key in encoded:
            np.testing.assert_array_equal(encoded[key], changed[key])
        self.assertEqual(encoded["token_valid_mask"][-1].sum(), 1)
        self.assertFalse(encoded["lr_bias"][-1].any())
        self.assertFalse(encoded["lr_coverage"][:, -1].any())
        for compartment in (0, 1):
            valid = encoded["token_valid_mask"][0] & (encoded["compartment_ids"][0] == compartment)
            self.assertEqual(set(encoded["gene_ids"][0, valid]), set(self.genes))
            self.assertTrue(((encoded["input_bins"][0, valid] >= 1)
                             & (encoded["input_bins"][0, valid] <= 50)).all())

    def test_lr_is_visible_full_panel_side_channel_independent_of_token_budget(self):
        full = self.encode()
        short = self.encode(context_tokens_per_compartment=1, whole_cell_max_tokens=1)
        for key in ("lr_bias", "lr_coverage"):
            np.testing.assert_array_equal(full[key], short[key])
        self.assertEqual(short["lr_coverage"][0, 1], 1)
        self.assertEqual(len(short["whole_gene_ids"]), 1)

    def test_collate_preserves_lr_node_axes_and_whole_padding(self):
        full, hidden = self.encode(), self.encode(mask_whole_cell=True)
        batch = collate_v6([full, hidden], pad_token_id=0)
        self.assertEqual(tuple(batch["lr_bias"].shape), (2, 4, 4))
        self.assertEqual(tuple(batch["lr_coverage"].shape), (2, 4, 4))
        np.testing.assert_array_equal(batch["lr_bias"][0].numpy(), full["lr_bias"])
        self.assertEqual(batch["whole_token_valid_mask"][1].sum().item(), 1)
        self.assertEqual(batch["gene_ids"].ndim, 3)
        with self.assertRaisesRegex(ValueError, "nodes, nodes"):
            collate_v6([dict(full, lr_bias=np.zeros((4, 3)))], pad_token_id=0)


class DirectedLRTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.array([[0, 10, 5], [10, 0, 5], [0, 0, 0], [4, 5, 6]])
        self.genes = np.array([10, 11, 12])
        self.options = dict(node_indices=[20, 30, 40, -1], lr_pairs=np.array([[10, 11]]))

    def compute(self, **kwargs):
        return compute_lr_features_v6(self.raw, self.genes, **dict(self.options, **kwargs))

    def test_direction_measured_zero_unknown_and_pad_are_distinct(self):
        result = self.compute()
        self.assertGreater(result["lr_bias"][0, 1], .19)
        self.assertEqual(result["lr_bias"][1, 0], 0)
        self.assertEqual(result["lr_bias"][2, 1], 0)
        self.assertEqual(result["lr_coverage"][2, 1], 1)  # measured receptor zero
        visible = np.ones(self.raw.shape, dtype=bool)
        visible[2, 1] = False
        unknown = self.compute(visible_mask=visible)
        self.assertEqual(unknown["lr_bias"][2, 1], 0)
        self.assertEqual(unknown["lr_coverage"][2, 1], 0)  # masked receptor unknown
        missing = self.compute(lr_pairs=np.array([[10, 999]]))
        self.assertFalse(missing["lr_bias"].any())
        self.assertFalse(missing["lr_coverage"].any())
        self.assertFalse(result["lr_bias"][-1].any())
        self.assertFalse(result["lr_coverage"][:, -1].any())
        self.assertFalse(result["lr_coverage"].diagonal().any())

    def test_confidence_reliability_is_distinct_from_full_prior_coverage(self):
        base = self.compute()
        mixed = self.compute(lr_pairs=np.array([[10, 11], [10, 999]]),
                             lr_weights=np.array([.25, .75]))
        self.assertEqual(mixed["lr_coverage"][0, 1], .25)
        self.assertAlmostEqual(mixed["lr_bias"][0, 1], base["lr_bias"][0, 1] * (.25 * 5 / 4.25))
        larger_prior = self.compute(lr_pairs=np.array([[10, 11], [10, 999]]))
        self.assertEqual(larger_prior["lr_coverage"][0, 1], .5)
        self.assertEqual(larger_prior["lr_bias"][0, 1], base["lr_bias"][0, 1])
        # A visible measured-zero second pair enters the reliability denominator.
        known_zero = self.compute(lr_pairs=np.array([[10, 11], [11, 10]]))
        self.assertEqual(known_zero["lr_coverage"][0, 1], 1)
        self.assertAlmostEqual(known_zero["lr_bias"][0, 1], base["lr_bias"][0, 1] * 5 / 6)

    def test_visible_depth_ignores_hidden_counts_and_scales_with_library_size(self):
        visible = np.ones(self.raw.shape, dtype=bool)
        visible[:, 2] = False
        before = self.compute(visible_mask=visible)
        changed_raw = self.raw.copy()
        changed_raw[:, 2] = 1_000_000
        after = compute_lr_features_v6(changed_raw, self.genes, visible_mask=visible, **self.options)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        scaled = compute_lr_features_v6(self.raw * 10, self.genes, **self.options)
        np.testing.assert_allclose(scaled["lr_bias"], self.compute()["lr_bias"])

    def test_gene_permutation_preserves_evidence_and_pair_reversal_reverses_direction(self):
        before = self.compute()
        perm = [2, 0, 1]
        after = compute_lr_features_v6(self.raw[:, perm], self.genes[perm], **self.options)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        reverse = self.compute(lr_pairs=np.array([[11, 10]]))
        np.testing.assert_array_equal(before["lr_bias"].T, reverse["lr_bias"])


class PriorAndCachedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "cache" / "slide"
        self.folder.mkdir(parents=True)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.vocab = {"<pad>": 0, "<cls>": 1, **{f"G{i}": i + 2 for i in range(6)}}
        vocab_path = self.assets / "vocab.json"
        vocab_path.write_text(json.dumps(self.vocab))
        self.raw_n = np.array([[0, 1, 2, 3, 4, 1], [4, 1, 3, 1, 1, 2]])
        self.raw_c = np.array([[2, 4, 1, 2, 1, 1], [1, 3, 1, 2, 1, 1]])
        arrays = {
            "nuclear_log1p_float32.npy": np.log1p(self.raw_n.astype(np.float32)),
            "cyto_log1p_float32.npy": np.log1p(self.raw_c.astype(np.float32)),
            "gene_ids.npy": np.arange(2, 8),
            "coords.npy": np.array([[0., 0.], [1., 1.]], dtype=np.float32),
            "train_nodes.npy": np.array([[0, 1, 0, -1], [1, 0, 1, -1]]),
            "val_nodes.npy": np.array([[0, 1, 0, -1]]),
        }
        for filename, value in arrays.items():
            np.save(self.folder / filename, value)
        self.cache_data = dict(context_cells=4, context_tokens_per_compartment=6,
                               query_genes=8, paired_mask_ratio=.5, seed=42)
        manifest = dict(schema=CACHE_SCHEMA, data_config=self.cache_data,
                        vocab_sha256=hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
                        slices=[dict(name="slide", cache_dir=str(self.folder))])
        (self.folder.parent / "manifest.json").write_text(json.dumps(manifest))
        self.config = dict(cache_root=str(self.folder.parent), assets_dir=str(self.assets),
                           data=dict(self.cache_data, whole_cell_max_tokens=5))
        self.set_prior("ligand,receptor,weight\nG0,G1,1\nG2,G3,0.5\n")

    def set_prior(self, content):
        path = self.root / "lr.csv"
        path.write_text(content)
        self.config["lr"] = dict(enabled=True, path=str(path), source="unit-test synthetic table",
                                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_metadata_hash_and_vocabulary_mapping(self):
        self.set_prior("ligand,receptor\nG0,G1\nUNKNOWN,G2\n")
        prior = load_lr_prior(self.config)
        np.testing.assert_array_equal(prior["pairs"], [[2, 3]])
        np.testing.assert_array_equal(prior["weights"], [1])
        self.assertEqual(prior["metadata"]["unmapped_pair_count"], 1)
        self.assertEqual(prior["metadata"]["sha256"], self.config["lr"]["sha256"])
        self.config["lr"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_lr_prior(self.config, self.vocab)

    def test_compounds_duplicates_bad_weights_and_missing_provenance_are_rejected(self):
        for row in ("G0_G1,G2,1", "G0+G1,G2,1", "G0,G1,0", "G0,G1,nan",
                    "G0,G1,1\nG0,G1,1"):
            self.set_prior("ligand,receptor,weight\n" + row + "\n")
            with self.subTest(row=row), self.assertRaises(ValueError):
                load_lr_prior(self.config, self.vocab)
        self.set_prior("ligand,receptor\nG0,G1\n")
        del self.config["lr"]["source"]
        with self.assertRaisesRegex(ValueError, "source"):
            load_lr_prior(self.config)
        disabled = load_lr_prior({"lr": {"enabled": False}})
        self.assertEqual(disabled["pairs"].shape, (0, 2))

    def test_runtime_metadata_preserves_strict_v2_cache_config_validation(self):
        dataset = V6Dataset(self.config)
        self.assertEqual(dataset.data, self.cache_data)
        self.assertEqual(dataset.lr_metadata["loaded_pairs"], 2)
        self.assertIsInstance(dataset.slices[0]["nuclear"], np.memmap)
        changed = dict(self.config, data=dict(self.config["data"], query_genes=7))
        with self.assertRaisesRegex(ValueError, "Data configuration differs"):
            V6Dataset(changed)

    def test_v5_and_v6_have_identical_query_target_and_whole_cell_streams(self):
        old, new = V5Dataset(self.config), V6Dataset(self.config)
        modes = set()
        for epoch in range(50):
            old.set_epoch(epoch)
            new.set_epoch(epoch)
            before, after = old[0], new[0]
            modes.add(int(after["task_mode"]))
            for key in (*WHOLE_KEYS, "query_gene_ids", "query_compartment_ids",
                        "query_valid_mask", "target_bins", "target_nonzero", "task_mode"):
                np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        self.assertEqual(modes, {0, 1, 2})

    def test_hidden_target_perturbation_preserves_dataset_inputs_and_task_in_all_modes(self):
        dataset = V6Dataset(self.config)
        modes = {}
        for epoch in range(200):
            dataset.set_epoch(epoch)
            item = dataset[0]
            modes.setdefault(int(item["task_mode"]), (epoch, item))
            if len(modes) == 3:
                break
        self.assertEqual(set(modes), {0, 1, 2})
        for mode, (epoch, before) in modes.items():
            altered = V6Dataset(self.config)
            altered.set_epoch(epoch)
            n, c = self.raw_n.copy(), self.raw_c.copy()
            if mode == 0:
                hidden = np.unique(before["query_gene_ids"][before["query_valid_mask"]]) - 2
                n[0, hidden] = 0
                c[0, hidden] = 0
            elif mode == 1:
                hidden_comp = int(before["query_compartment_ids"][before["query_valid_mask"]][0])
                (n, c)[hidden_comp][0] = 0
            else:
                n[0], c[0] = 0, 0
            altered.slices[0]["nuclear"] = np.log1p(n.astype(np.float32))
            altered.slices[0]["cyto"] = np.log1p(c.astype(np.float32))
            after = altered[0]
            for key in before:
                if key not in ("target_bins", "target_nonzero"):
                    np.testing.assert_array_equal(before[key], after[key], err_msg=f"mode={mode}, {key}")
            self.assertFalse(np.array_equal(before["target_bins"], after["target_bins"]))

    def test_validation_remains_fixed_and_lr_off_is_neutral(self):
        dataset = V6Dataset(self.config, split="val")
        before = dataset[0]
        dataset.set_epoch(91)
        after = dataset[0]
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        disabled = V6Dataset(dict(self.config, lr={"enabled": False}))[0]
        self.assertFalse(disabled["lr_bias"].any())
        self.assertFalse(disabled["lr_coverage"].any())


if __name__ == "__main__":
    unittest.main()
