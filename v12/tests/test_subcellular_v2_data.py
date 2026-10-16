import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import sparse

from spatial_pretrain.data.subcellular_v2 import (
    CACHE_SCHEMA, V2Dataset, collate_v2, encode_context, sample_paired_queries, sample_query_genes,
    visible_quantile_bins,
)


_prepare_spec = importlib.util.spec_from_file_location(
    "prepare_subcellular_v2", Path(__file__).resolve().parents[1] / "scripts" / "prepare_subcellular_v2.py")
prepare = importlib.util.module_from_spec(_prepare_spec)
_prepare_spec.loader.exec_module(prepare)


class VisibleContextTests(unittest.TestCase):
    def setUp(self):
        self.nuclear = np.log1p(np.array([[1, 2, 0, 7, 3], [2, 1, 4, 0, 0]], dtype=np.float32))
        self.cyto = np.log1p(np.array([[3, 0, 2, 1, 0], [1, 3, 1, 2, 1]], dtype=np.float32))
        self.genes = np.arange(10, 15)

    def encode(self, nuclear=None, cyto=None, **kwargs):
        return encode_context(self.nuclear if nuclear is None else nuclear,
                              self.cyto if cyto is None else cyto, self.genes,
                              node_indices=[21, 50], context_tokens_per_compartment=5,
                              pad_token_id=0, cls_token_id=1, **kwargs)

    def test_paired_mask_hidden_values_cannot_change_encoder_input(self):
        selected = [1, 3]
        before = self.encode(masked_genes=selected)
        changed_n, changed_c = self.nuclear.copy(), self.cyto.copy()
        changed_n[0, selected] = [0, 99]
        changed_c[0, selected] = [77, 0]
        after = self.encode(changed_n, changed_c, masked_genes=selected)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(np.isin(before["gene_ids"][0], self.genes[selected]).any())

    def test_whole_compartment_and_whole_cell_have_no_target_dependent_flags(self):
        for arguments in ({"masked_compartment": 1}, {"mask_whole_cell": True}):
            before = self.encode(**arguments)
            changed_n, changed_c = self.nuclear.copy(), self.cyto.copy()
            changed_c[0] = [0, 100, 99, 1, 0]
            if arguments.get("mask_whole_cell"):
                changed_n[0] = [100, 0, 100, 1, 99]
            after = self.encode(changed_n, changed_c, **arguments)
            for key in before:
                np.testing.assert_array_equal(before[key], after[key])
            if arguments.get("mask_whole_cell"):
                self.assertEqual(before["token_valid_mask"][0].sum(), 1)
                self.assertEqual(before["compartment_ids"][0, 0], 2)

    def test_overlapping_patch_cell_encoding_does_not_depend_on_patch_position(self):
        before = self.encode()
        after = encode_context(self.nuclear[::-1], self.cyto[::-1], self.genes,
                               node_indices=[50, 21], context_tokens_per_compartment=5,
                               pad_token_id=0, cls_token_id=1)
        for key in before:
            np.testing.assert_array_equal(before[key][::-1], after[key])

    def test_only_positive_gene_tokens_and_nonempty_sentinel(self):
        output = self.encode()
        gene_tokens = output["token_valid_mask"] & (output["compartment_ids"] < 2)
        self.assertTrue((output["input_bins"][gene_tokens] >= 1).all())
        self.assertTrue((output["input_bins"][gene_tokens] <= 50).all())
        self.assertTrue(output["token_valid_mask"][:, 0].all())
        empty = encode_context(np.zeros((2, 5)), np.zeros((2, 5)), self.genes,
                               node_indices=[1, -1], pad_token_id=0, cls_token_id=1)
        self.assertEqual(empty["token_valid_mask"].sum(), 2)

    def test_quantiles_preserve_float32_resolution(self):
        # Historical float16 would merge these; V2 must retain their ordering.
        values = np.array([1., 1.0001, 1.0002], dtype=np.float32)
        bins, _ = visible_quantile_bins(values, np.zeros(3), np.random.default_rng(42))
        self.assertEqual(len(np.unique(bins)), 3)

    def test_query_sampling_never_invents_unmeasured_genes(self):
        for counts in (np.zeros(5), np.ones(5), np.array([1, 0, 1, 0, 0])):
            selected = sample_query_genes(counts, np.zeros(5), 128, np.random.default_rng(42))
            np.testing.assert_array_equal(np.sort(selected), np.arange(5))

    def test_uniform_paired_mask_fraction_is_independent_of_query_budget_and_targets(self):
        for panel_size in (5, 343, 5000):
            n = np.arange(panel_size, dtype=np.float32) % 3
            c = np.arange(panel_size, dtype=np.float32) % 5
            hidden, queries = sample_paired_queries(n, c, 128, .5, np.random.default_rng(42))
            changed_hidden, changed_queries = sample_paired_queries(
                np.zeros_like(n), np.ones_like(c), 7, .5, np.random.default_rng(42))
            self.assertEqual(len(hidden), int(np.ceil(panel_size * .5)))
            np.testing.assert_array_equal(hidden, changed_hidden)
            self.assertTrue(np.isin(queries, hidden).all())
            self.assertTrue(np.isin(changed_queries, hidden).all())
            self.assertEqual(len(queries), min(128, len(hidden)))

    def test_paired_queries_are_absent_from_both_visible_compartments(self):
        hidden, queries = sample_paired_queries(
            self.nuclear[0], self.cyto[0], 128, .5, np.random.default_rng(91))
        context = self.encode(masked_genes=hidden)
        self.assertTrue(np.isin(queries, hidden).all())
        self.assertFalse(np.isin(context["gene_ids"][0], self.genes[queries]).any())
        changed_n, changed_c = self.nuclear.copy(), self.cyto.copy()
        changed_n[0, hidden] = 100
        changed_c[0, hidden] = 0
        modified = self.encode(changed_n, changed_c, masked_genes=hidden)
        for key in context:
            np.testing.assert_array_equal(context[key], modified[key])

    def test_collation_removes_only_trailing_padding(self):
        full, empty = self.encode(), self.encode(mask_whole_cell=True)
        batch = collate_v2([full, empty])
        self.assertEqual(batch["gene_ids"].shape[:2], (2, 2))
        self.assertTrue(batch["token_valid_mask"][:, :, 0].all())
        expected = int(full["token_valid_mask"].sum(axis=1).max())
        self.assertEqual(batch["gene_ids"].shape[-1], expected)


class CachePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "slice.h5ad"
        self.old = self.root / "old"
        self.old.mkdir()
        self.cache = self.root / "new"
        self.cache.mkdir()
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.vocab = {"<pad>": 0, "<cls>": 1, "A": 2, "B": 3}
        (self.assets / "vocab.json").write_text(json.dumps(self.vocab))
        self.vocab_sha = hashlib.sha256((self.assets / "vocab.json").read_bytes()).hexdigest()
        self.data = dict(context_cells=4, context_tokens_per_compartment=5, query_genes=3,
                         neighbors=2, seed=42, train_fraction=.75, split_blocks_x=4,
                         split_blocks_y=4, center_stride=3, paired_mask_ratio=.5)
        coords = np.stack(np.meshgrid(np.arange(4), np.arange(4)), axis=-1).reshape(-1, 2).astype(np.float32)
        distances = ((coords[:, None] - coords[None]) ** 2).sum(axis=-1)
        knn = np.argsort(distances, axis=1)[:, 1:].astype(np.int32)
        np.save(self.old / "coords.npy", coords)
        np.save(self.old / "knn_indices.npy", knn)
        self.nuclear = (np.arange(48).reshape(16, 3) % 7).astype(np.int32)
        self.cyto = (np.arange(48).reshape(16, 3) % 3).astype(np.int32)
        self.nuclear[0, 0] = 100_000

        def write_csr(parent, name, value):
            csr = sparse.csr_matrix(value)
            group = parent.create_group(name)
            group.attrs["encoding-type"] = "csr_matrix"
            group.create_dataset("data", data=csr.data)
            group.create_dataset("indices", data=csr.indices)
            group.create_dataset("indptr", data=csr.indptr)

        with h5py.File(self.source, "w") as handle:
            var = handle.create_group("var")
            var.attrs["_index"] = "_index"
            var.create_dataset("_index", data=np.array([b"A", b"B", b"unmatched"]))
            handle.create_group("obsm").create_dataset("spatial", data=coords)
            layers = handle.create_group("layers")
            write_csr(layers, "nuclear_counts", self.nuclear)
            write_csr(layers, "extranuclear_counts", self.cyto)
            write_csr(handle, "X", self.nuclear + self.cyto)

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self):
        return prepare.prepare_slice(self.source, self.old, self.cache / "slice",
                                     self.vocab, self.vocab_sha, self.data)

    def test_cache_reads_original_counts_and_explicitly_reports_unmatched_panel(self):
        metadata = self.prepare()
        values = np.load(self.cache / "slice" / "nuclear_log1p_float32.npy")
        self.assertEqual(values.dtype, np.dtype("float32"))
        np.testing.assert_array_equal(values, np.log1p(self.nuclear[:, :2].astype(np.float32)))
        self.assertEqual(metadata["excluded_unmatched_genes"], ["unmatched"])
        self.assertEqual(metadata["num_cells"], 16)
        train = np.load(self.cache / "slice" / "train_nodes.npy")
        val = np.load(self.cache / "slice" / "val_nodes.npy")
        train_members, val_members = set(train[train >= 0]), set(val[val >= 0])
        self.assertFalse(train_members & val_members)
        self.assertEqual(train_members | val_members, set(range(16)))
        self.assertEqual(self.prepare(), metadata)  # valid resume

    def test_cache_signature_rejects_changed_settings(self):
        self.prepare()
        self.data["context_cells"] = 5
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            self.prepare()

    def test_dataset_queries_cover_only_measured_vocab_and_validation_is_fixed(self):
        metadata = self.prepare()
        manifest = {"schema": CACHE_SCHEMA, "data_config": self.data,
                    "vocab_sha256": self.vocab_sha, "slices": [metadata]}
        (self.cache / "manifest.json").write_text(json.dumps(manifest))
        config = dict(cache_root=str(self.cache), assets_dir=str(self.assets), data=self.data)
        dataset = V2Dataset(config, split="val")
        before = dataset[0]
        dataset.set_epoch(99)
        after = dataset[0]
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
        real_queries = before["query_gene_ids"][before["query_valid_mask"]]
        self.assertTrue(np.isin(real_queries, [2, 3]).all())
        self.assertEqual(len(before["query_gene_ids"]), 6)
        self.assertEqual(collate_v2([before])["gene_ids"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
