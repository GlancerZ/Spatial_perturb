"""Small HDF5 fixtures for compartment ingestion QA; run on compute nodes."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import sparse


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_existing_compartment_corpus.py"
SPEC = importlib.util.spec_from_file_location("audit_existing_compartment_corpus", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def matrix(parent, name, value, csr):
    if not csr:
        parent.create_dataset(name, data=value)
        return
    value = sparse.csr_matrix(value)
    group = parent.create_group(name)
    group.attrs["encoding-type"] = "csr_matrix"
    group.attrs["shape"] = value.shape
    for key, data in (("data", value.data), ("indices", value.indices), ("indptr", value.indptr)):
        group.create_dataset(key, data=data)


def fixture(path, *, csr=True, bad_total=False, fractional=False, coords=None,
            duplicate_obs=False, duplicate_var=False):
    n = np.array([[1, 0, 2], [0, 4, 0]], dtype=float)
    c = np.array([[0, 2, 1], [3, 0, 1]], dtype=float)
    if fractional:
        n[0, 0] = 1.5
    x = n + c
    if bad_total:
        x[1, 2] += 1
    with h5py.File(path, "w") as handle:
        for name, names in (("obs", ["cell1", "cell1" if duplicate_obs else "cell2"]),
                            ("var", ["A", "A" if duplicate_var else "B", "C"])):
            group = handle.create_group(name)
            group.attrs["_index"] = "_index"
            group.create_dataset("_index", data=np.array(names, dtype=h5py.string_dtype()))
        matrix(handle, "X", x, csr)
        layers = handle.create_group("layers")
        matrix(layers, "nuclear_counts", n, csr)
        matrix(layers, "extranuclear_counts", c, csr)
        handle.create_group("obsm").create_dataset(
            "spatial", data=np.array([[0., 0.], [1., 1.]]) if coords is None else coords)


class CorpusAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "slice.h5ad"

    def tearDown(self):
        self.tmp.cleanup()

    def test_dense_and_csr_streamed_counts(self):
        for csr in (False, True):
            fixture(self.path, csr=csr)
            self.path.with_suffix(".qc.json").write_text(json.dumps(
                {"nuclear": 7, "cytoplasmic": 7, "retained": 14}))
            row = audit.audit_file(self.path, chunk_rows=1, vocab={"A", "B"}, species="human")
            self.assertTrue(row["schema_verified"])
            self.assertFalse(row["training_split_ready"])
            self.assertFalse(row["donor_audited"])
            self.assertEqual((row["cells"], row["genes"]), (2, 3))
            self.assertEqual(row["retained_counts"], 14)
            self.assertEqual(row["vocab_coverage"], 2 / 3)
            self.assertEqual(row["vocab_unmatched_genes"], ["C"])
            self.assertEqual(row["species_evidence"], "explicit_collection_label")

    def test_crosscount_corruption_rejected(self):
        fixture(self.path, bad_total=True)
        with self.assertRaisesRegex(ValueError, "!= X"):
            audit.audit_file(self.path, chunk_rows=1)

    def test_noninteger_rejected(self):
        fixture(self.path, fractional=True)
        with self.assertRaisesRegex(ValueError, "integer counts"):
            audit.audit_file(self.path)

    def test_coordinate_mismatch_and_nan_rejected(self):
        for coords in (np.zeros((1, 2)), np.zeros((2, 3)), np.array([[0, np.nan], [1, 1]])):
            fixture(self.path, coords=coords)
            with self.assertRaisesRegex(ValueError, "spatial coordinates"):
                audit.audit_file(self.path)

    def test_duplicate_axes_rejected(self):
        for argument in ("duplicate_obs", "duplicate_var"):
            fixture(self.path, **{argument: True})
            with self.assertRaisesRegex(ValueError, "unique"):
                audit.audit_file(self.path)

    def test_sidecar_mismatch_rejected(self):
        fixture(self.path)
        self.path.with_suffix(".qc.json").write_text('{"nuclear": 100}')
        with self.assertRaisesRegex(ValueError, "sidecar nuclear"):
            audit.audit_file(self.path)

    def test_bad_csr_column_rejected(self):
        fixture(self.path)
        with h5py.File(self.path, "a") as handle:
            handle["layers"]["nuclear_counts"]["indices"][0] = 99
        with self.assertRaisesRegex(ValueError, "out of bounds"):
            audit.audit_file(self.path)

    def test_incremental_catalog_and_failure_exit(self):
        fixture(self.path)
        fixture(self.root / "corrupt.h5ad", bad_total=True)
        output = self.root / "audit"
        self.assertEqual(audit.main(["--source-dir", str(self.root), "--output", str(output)]), 2)
        catalog = json.loads((output / "catalog.json").read_text())
        self.assertEqual(catalog["summary"]["verified_files"], 1)
        self.assertEqual(catalog["summary"]["failed_files"], 1)
        self.assertEqual(len(list((output / "files").glob("*.json"))), 2)
        self.assertEqual(catalog["summary"]["cells"], 2)
        with self.assertRaises(FileExistsError):
            audit.main(["--source-dir", str(self.root), "--output", str(output)])

    def test_no_species_inference_and_mapping_not_verified(self):
        fixture(self.path)
        row = audit.audit_file(self.path, mapping={"/other/slice.h5ad": "/other/graph"})
        self.assertEqual(row["species"], "unknown")
        self.assertEqual(row["geometry_cache_mapping"], "basename_only_unverified")
        self.assertFalse(row["geometry_cache_verified"])


if __name__ == "__main__":
    unittest.main()
