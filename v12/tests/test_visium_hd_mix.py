"""Sparse HD panel, physical-context and immutable-cache contract tests.

Execute on a compute node. The fixtures require no model weights or GPU, but
exercise the real V12 encoder and the real unchanged Xenium V12 dataset.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from prepare_visium_hd_v12_cache import (
    SparseCountWriter, mapped_measured_panel, prepare_sample, read_inputs_manifest,
)
from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA
from spatial_pretrain.data.subcellular_v12 import collate_v12
from spatial_pretrain.data.visium_hd_mix import (
    VISIUM_CACHE_SCHEMA, SparseRawCountRows, V12MixedDataset, VisiumHDV12Dataset,
    assert_not_crc_source, file_sha256,
)


class SparsePanelTests(unittest.TestCase):
    def test_duplicate_symbols_sum_compartments_and_keep_measured_zero(self):
        names = ["A", "B", "A", "ZERO", "ABSENT"]
        vocab = {"<pad>": 0, "<cls>": 1, "A": 2, "B": 3, "ZERO": 4, "NOT_MEASURED": 5}
        genes, symbols, projection, columns, audit = mapped_measured_panel(names, vocab)
        nucleus = sparse.csr_matrix([[2, 3, 5, 0, 99], [1, 0, 8, 0, 55]], dtype=np.int64)
        cyto = sparse.csr_matrix([[11, 7, 13, 0, 2], [4, 9, 6, 0, 7]], dtype=np.int64)
        np.testing.assert_array_equal(genes, [2, 3, 4])
        self.assertEqual(symbols, ["A", "B", "ZERO"])
        np.testing.assert_array_equal((nucleus @ projection).toarray(), [[7, 3, 0], [9, 0, 0]])
        np.testing.assert_array_equal((cyto @ projection).toarray(), [[24, 7, 0], [10, 9, 0]])
        np.testing.assert_array_equal(np.asarray(((nucleus + cyto) @ projection).sum(axis=1)),
                                      np.asarray((nucleus + cyto)[:, columns].sum(axis=1)))
        self.assertEqual(audit["duplicate_mapped_features_summed"], 1)
        self.assertFalse(audit["full_vocab_zero_filling"])
        self.assertNotIn(5, genes)

    def test_sparse_partial_row_reads_preserve_large_integer_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = sparse.csr_matrix([[0, 2**24 + 1, 0, 7], [4, 0, 0, 0], [0, 0, 8, 0]], dtype=np.int64)
            writer = SparseCountWriter(folder, "nuclear", raw.shape)
            writer.append(raw[:1])
            writer.append(raw[1:])
            writer.close()
            cache = SparseRawCountRows(folder, "nuclear", raw.shape, writer.nnz)
            np.testing.assert_array_equal(cache.dense_rows(np.array([2, 0, 2])), raw[[2, 0, 2]].toarray())
            self.assertEqual(cache.dense_rows(np.array([0]))[0, 1], 2**24 + 1)
            with self.assertRaises(ValueError):
                cache.dense_rows(np.array([-1]))

    def test_sparse_zero_compartment_is_supported_without_fabrication(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = SparseCountWriter(temporary, "cytoplasmic", (3, 5))
            writer.append(sparse.csr_matrix((3, 5), dtype=np.int64))
            writer.close()
            cache = SparseRawCountRows(temporary, "cytoplasmic", (3, 5), 0)
            self.assertFalse(cache.dense_rows(np.array([0, 2])).any())

    def test_crc_aliases_and_slide_metadata_are_rejected(self):
        for value in ("Visium_HD_Human_Colon_Cancer", "visium-hd-human-colorectal-cancer-ffpe",
                      {"slide": "H1-VM2JXXK", "well": "A1"}, "H1-VM2JXXK/A1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                assert_not_crc_source(value)
        assert_not_crc_source({"sample_id": "human_kidney", "physical_sample_id": "H2/A1"})

    def test_manifest_does_not_allow_renamed_crc_or_duplicate_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "counts.h5ad").touch()
            manifest = root / "inputs.tsv"
            header = "sample_id\tphysical_sample_id\tcompartments_h5ad\n"
            manifest.write_text(header + "kidney\tH1-VM2JXXK/A1\tcounts.h5ad\n")
            with self.assertRaises(ValueError):
                read_inputs_manifest(manifest)
            (root / "reprocessed.h5ad").touch()
            manifest.write_text(header + "kidney_v4\tH2/A1\tcounts.h5ad\n"
                                "kidney_v5\th2_a1\treprocessed.h5ad\n")
            with self.assertRaisesRegex(ValueError, "physical sample ID"):
                read_inputs_manifest(manifest)
            manifest.write_text(header + "kidney_A\tH2/A1\tcounts.h5ad\n"
                                "kidney_B\tH3/A1\tcounts.h5ad\n")
            with self.assertRaises(ValueError):
                read_inputs_manifest(manifest)


class DatasetIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.vocab = {"<pad>": 0, "<cls>": 1, "A": 2, "B": 3, "ZERO": 4, "NOT_MEASURED": 5}
        assets = self.root / "assets"
        assets.mkdir()
        (assets / "vocab.json").write_text(json.dumps(self.vocab))
        self.config = dict(assets_dir=str(assets),
            data=dict(context_cells=4, context_tokens_per_compartment=8, query_genes=2,
                      paired_mask_ratio=.5, neighbors=2, seed=42, train_fraction=.7,
                      split_blocks_x=2, split_blocks_y=2, center_stride=2),
            v12=dict(neighbors=2, max_tokens=16, query_genes=2, gene_mask_ratio=0.,
                     neighbor_dropout=0., max_neighbor_distance=100., pilot_centers=0),
            binning=dict(num_bins=51, tie_policy="left"), lr=dict(enabled=False))
        self.cache = self.root / "hd_cache"
        self.cache.mkdir()
        self.original_counts = {}
        first = self.make_source("liver_01", 1)
        metadata = prepare_sample(first, self.cache / "sample_000", self.config, self.vocab)
        self.write_manifest([metadata])
        self.config["visium_cache_root"] = str(self.cache)

    def tearDown(self):
        self.temporary.cleanup()

    def make_source(self, name, multiplier):
        source_dir = self.root / (name + "_source")
        source_dir.mkdir()
        # The ZERO feature is truly measured but never observed. NOT_MEASURED
        # belongs only to the vocabulary. A occurs under two distinct IDs.
        names = ["A", "B", "A", "ZERO", "UNMATCHED"]
        nuclear = np.tile(np.array([1, 3, 5, 0, 7], dtype=np.int64), (16, 1)) * multiplier
        nuclear[:, 1] += np.arange(16)
        cyto = np.tile(np.array([11, 13, 17, 0, 19], dtype=np.int64), (16, 1)) * multiplier
        coords = np.asarray([(float(x * 10), float(y * 10)) for y in range(4) for x in range(4)], dtype=np.float32)
        obs = pd.DataFrame(dict(sample_id=[name] * 16), index=[f"cell_{i}" for i in range(16)])
        var = pd.DataFrame(dict(gene_name=names, measured=[True] * len(names)),
                           index=[f"ENSG_{i}" for i in range(len(names))])
        adata = ad.AnnData(sparse.csr_matrix(nuclear + cyto), obs=obs, var=var)
        adata.layers["nuclear_counts"] = sparse.csr_matrix(nuclear)
        adata.layers["cytoplasmic_counts"] = sparse.csr_matrix(cyto)
        adata.obsm["spatial_um"] = coords
        adata.uns["input_schema"] = "visium_hd_segmented_compartments_v1"
        adata.uns["compartment_provenance"] = dict(sample_id=name, coordinate_units="micrometres",
            coordinates="unweighted geometric centroid of assigned 2um capture squares")
        source = source_dir / "compartments.h5ad"
        adata.write_h5ad(source)
        (source_dir / "audit.json").write_text(json.dumps(dict(status="complete", output_h5ad=str(source),
            sample_id=name, checks=dict(X_equals_nuclear_plus_cytoplasmic=True))))
        self.original_counts[name] = (np.column_stack([nuclear[:, 0] + nuclear[:, 2], nuclear[:, 1], nuclear[:, 3]]),
                                      np.column_stack([cyto[:, 0] + cyto[:, 2], cyto[:, 1], cyto[:, 3]]))
        return dict(sample_id=name, physical_sample_id="physical_" + name,
                    compartments_h5ad=str(source), species="Homo sapiens")

    def write_manifest(self, metadata):
        manifest = dict(schema=VISIUM_CACHE_SCHEMA, status="validated", data_config=self.config["data"],
                        vocab_sha256=file_sha256(Path(self.config["assets_dir"]) / "vocab.json"), slices=metadata)
        (self.cache / "manifest.json").write_text(json.dumps(manifest))

    def make_xenium_cache(self):
        root = self.root / "xenium_cache"
        folder = root / "xenium_slice"
        folder.mkdir(parents=True)
        np.save(folder / "gene_ids.npy", np.array([2, 3, 4], dtype=np.int64))
        np.save(folder / "coords.npy", np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]], dtype=np.float32))
        np.save(folder / "nuclear_log1p_float32.npy", np.log1p(np.array([[1, 2, 0]] * 4, dtype=np.float32)))
        np.save(folder / "cyto_log1p_float32.npy", np.log1p(np.array([[3, 4, 0]] * 4, dtype=np.float32)))
        for split in ("train", "val"):
            np.save(folder / f"{split}_nodes.npy", np.array([[0, 1, 2, 3], [2, 3, 1, 0]], dtype=np.int32))
        metadata = dict(name="xenium_slice", cache_dir=str(folder))
        (root / "manifest.json").write_text(json.dumps(dict(schema=CACHE_SCHEMA,
            data_config=self.config["data"], vocab_sha256=file_sha256(Path(self.config["assets_dir"]) / "vocab.json"),
            slices=[metadata])))
        self.config["cache_root"] = str(root)

    def test_real_loader_targets_match_raw_subset_and_no_unmeasured_gene(self):
        config = copy.deepcopy(self.config)
        config["v12"]["query_genes"] = 5
        dataset = VisiumHDV12Dataset(config, split="val")
        nuclear, cyto = self.original_counts["liver_01"]
        for index in range(len(dataset)):
            view = dataset[index]
            center = int(view["cell_identity"][1])
            valid = view["query_valid_mask"]
            self.assertEqual(int(valid.sum()), 3)
            self.assertEqual(set(view["query_gene_ids"][valid]), {2, 3, 4})
            columns = view["query_gene_ids"][valid] - 2
            np.testing.assert_array_equal(view["query_nucleus_counts"][valid], nuclear[center, columns])
            np.testing.assert_array_equal(view["query_cytoplasm_counts"][valid], cyto[center, columns])
            np.testing.assert_array_equal(view["query_total_counts"],
                                          view["query_nucleus_counts"] + view["query_cytoplasm_counts"])
            self.assertFalse(view["query_total_counts"][~valid].any())

    def test_context_uses_full_graph_and_is_split_isolated(self):
        folder = self.cache / "sample_000"
        coords = np.load(folder / "coords.npy")
        self.assertEqual(len(coords), 16)
        graph = np.load(folder / "knn_indices.npy")
        self.assertEqual(graph.shape, (16, 255))
        mask = np.load(folder / "train_cell_mask.npy")
        for split in ("train", "val"):
            nodes = np.load(folder / f"{split}_nodes.npy")
            centers = nodes[:, 0]
            self.assertLess(len(centers), len(coords))
            for patch in nodes:
                real = patch[patch >= 0]
                self.assertTrue((mask[real] == (split == "train")).all())
                self.assertFalse((patch[1:] == patch[0]).any())
                self.assertTrue(set(real[1:]).issubset(set(graph[patch[0]])))

    def test_same_coordinates_on_two_slides_never_make_cross_sample_neighbors(self):
        second = self.make_source("kidney_02", 11)
        metadata = json.loads((self.cache / "sample_000/metadata.json").read_text())
        other = prepare_sample(second, self.cache / "sample_001", self.config, self.vocab)
        self.write_manifest([metadata, other])
        dataset = VisiumHDV12Dataset(self.config, split="val", slice_index_offset=48)
        self.assertEqual(set(dataset.indices_by_sample), {"liver_01", "kidney_02"})
        for name, rows in dataset.indices_by_sample.items():
            view = dataset[int(rows[0])]
            identity = 48 + (name == "kidney_02")
            self.assertEqual(int(view["cell_identity"][0]), identity)
            present = view["neighbor_present"]
            self.assertTrue((view["neighbor_cell_identity"][present, 0] == identity).all())

    def test_cache_byte_change_is_rejected_before_encoding(self):
        target = self.cache / "sample_000/nuclear.data.bin"
        with target.open("r+b") as handle:
            handle.write(np.asarray([999], dtype="<i8").tobytes())
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            VisiumHDV12Dataset(self.config)

    def test_source_audit_does_not_hide_broken_nuclear_cytoplasmic_identity(self):
        source = self.make_source("liver_bad", 7)
        with h5py.File(source["compartments_h5ad"], "r+") as handle:
            handle["X/data"][0] += 1
        with self.assertRaisesRegex(ValueError, r"Source N \+ C differs from X"):
            prepare_sample(source, self.cache / "bad_source", self.config, self.vocab)

    def test_mix_indices_domain_ids_and_v12_collation_match(self):
        self.make_xenium_cache()
        mixed = V12MixedDataset(self.config, split="val")
        self.assertEqual(len(mixed.xenium_indices), 2)
        self.assertEqual(len(mixed.visium_indices), len(mixed.visium))
        np.testing.assert_array_equal(mixed.visium_indices_by_sample["liver_01"], mixed.visium_indices)
        x_view, hd_view = mixed[0], mixed[int(mixed.visium_indices[0])]
        self.assertEqual(int(x_view["domain_id"]), 0)
        self.assertEqual(int(hd_view["domain_id"]), 1)
        self.assertEqual(int(hd_view["cell_identity"][0]), 1)
        batch = collate_v12([x_view, hd_view])
        self.assertEqual(batch["domain_id"].tolist(), [0, 1])
        mixed.set_epoch(8)
        self.assertEqual(mixed.xenium.epoch, 0)  # Frozen validation queries.
        self.assertEqual(mixed.visium.epoch, 0)
        control = V12MixedDataset(self.config, split="train", include_visium=False)
        self.assertEqual(len(control), 2)
        self.assertFalse(len(control.visium_indices))
        control.set_epoch(5)
        self.assertEqual(control.xenium.epoch, 5)


if __name__ == "__main__":
    unittest.main()
