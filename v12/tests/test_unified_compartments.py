"""Meaningful conversion/data-integrity tests; execute only on compute nodes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd

from spatial_pretrain.data.unified_compartments import convert_manifest, example_manifest


class UnifiedCompartmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cells = pd.DataFrame({"cell_id": ["001", "001", "002"], "fov": ["a", "b", "b"],
                                   "x": [0, 1000, 1010], "y": [0, 1000, 1010]})
        self.molecules = pd.DataFrame({
            "cell_id": ["001", "001", "001", "002", "UNASSIGNED", "001", "001"],
            "fov": ["a", "a", "b", "b", "a", "a", "a"],
            "gene": ["G1", "G2", "G1", "G2", "G1", "NegControl1", "G1"],
            "label": ["Nuclear", "Cytoplasm", "Cytoplasm", "Membrane", "Nuclear", "Nuclear", "Nuclear"],
            "qv": [30, 30, 30, 30, 30, 30, 5], "x": [1, 2, 3, 4, 5, 6, 7], "y": [1, 2, 3, 4, 5, 6, 7]})
        self.manifest = example_manifest("cosmx")
        self.manifest.update(dataset_id="fixture", slide_id="slide/1")
        self.manifest["cells"] = {
            "path": "cells.csv", "columns": {"cell_id": "cell_id", "fov": "fov", "x": "x", "y": "y"},
            "coordinates": {"unit": "pixel", "microns_per_unit": 0.2, "frame": "slide_global",
                            "registration_verified": True, "evidence": "Fixture global coordinates"}}
        self.manifest["molecules"] = {
            "path": "molecules.csv", "record_unit": "molecule", "columns": {"cell_id": "cell_id", "fov": "fov", "gene": "gene",
                                                      "compartment": "label", "quality": "qv", "x": "x", "y": "y"},
            "coordinates": {"unit": "pixel", "frame": "fov_local"}}
        self.manifest["panel"] = {"genes": ["G1", "G2", "G3"], "controls_excluded": True, "evidence": "Fixture measured panel"}
        self.manifest["compartments"]["label_semantics_verified"] = True
        self.manifest["filters"].update(unassigned_cell_ids=["UNASSIGNED"], quality_min=20,
                                         control_prefixes=["NegControl"], control_genes=[], quality_evidence="Fixture Q20")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, parquet=False):
        self.cells.to_csv(self.root / "cells.csv", index=False)
        if parquet:
            self.molecules.to_parquet(self.root / "molecules.parquet", index=False)
            self.manifest["molecules"]["path"] = "molecules.parquet"
        else:
            self.molecules.to_csv(self.root / "molecules.csv", index=False)
        path = self.root / "manifest.json"
        path.write_text(json.dumps(self.manifest))
        return path

    def run_conversion(self, chunk_rows=2, parquet=False, output="converted"):
        destination = self.root / output
        stats = convert_manifest(self.write(parquet), destination, chunk_rows=chunk_rows)
        ready = json.loads((destination / "READY.json").read_text())
        result = ad.read_h5ad(destination / ready["data"])
        return stats, result, ready

    def test_panel_preserved_compartments_conserve_and_membrane_stays_separate(self):
        stats, result, ready = self.run_conversion()
        self.assertEqual(stats["rows_seen"], 7)
        self.assertEqual(stats["retained"], 4)
        self.assertEqual(stats["training_transcripts"], 3)
        self.assertEqual((stats["nuclear"], stats["cytoplasmic"], stats["membrane"]), (1, 2, 1))
        self.assertEqual((stats["unassigned"], stats["control"], stats["below_quality"]), (1, 1, 1))
        self.assertEqual(list(result.var_names), ["G1", "G2", "G3"])
        self.assertTrue(result.var["measured"].all())
        np.testing.assert_array_equal(result.X.toarray(), [[1, 1, 0], [1, 0, 0], [0, 0, 0]])
        self.assertEqual(result.layers["membrane_counts"].sum(), 1)
        np.testing.assert_array_equal(result.X.toarray(), (result.layers["nuclear_counts"] + result.layers["extranuclear_counts"]).toarray())
        np.testing.assert_allclose(result.obsm["spatial"], [[0, 0], [200, 200], [202, 202]])
        self.assertEqual(len(set(result.obs_names)), 3)
        self.assertEqual(result.obs["original_cell_id"].tolist(), ["001", "001", "002"])
        self.assertTrue(ready["data"].startswith("fixture--slide%2F1"))
        self.assertEqual(ready["data_sha256"], hashlib.sha256((self.root / "converted" / ready["data"]).read_bytes()).hexdigest())

    def test_chunk_size_and_parquet_do_not_change_counts(self):
        stats1, first, _ = self.run_conversion(chunk_rows=1, output="one")
        stats2, second, _ = self.run_conversion(chunk_rows=100, parquet=True, output="two")
        np.testing.assert_array_equal(first.X.toarray(), second.X.toarray())
        self.assertEqual(stats1["retained"], stats2["retained"])

    def test_single_fov_rejects_cross_fov_coordinates(self):
        self.manifest["cells"]["coordinates"]["frame"] = "single_fov"
        with self.assertRaisesRegex(ValueError, "multiple FOVs"):
            self.run_conversion()

    def test_local_cell_centroids_rejected_without_transform(self):
        self.manifest["cells"]["coordinates"]["frame"] = "fov_local"
        with self.assertRaisesRegex(ValueError, "transforms are not inferred"):
            self.run_conversion()

    def test_unknown_compartment_not_silently_cytoplasmic(self):
        self.molecules.loc[0, "label"] = "unresolved"
        with self.assertRaisesRegex(ValueError, "Unknown compartment"):
            self.run_conversion()
        self.assertFalse((self.root / "converted").exists())

    def test_assigned_cell_mismatch_fails(self):
        self.molecules.loc[0, "fov"] = "other_slide"
        with self.assertRaisesRegex(ValueError, "missing from cell metadata"):
            self.run_conversion()

    def test_duplicate_cell_key_fails(self):
        self.cells.loc[1, "fov"] = "a"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_conversion()

    def test_gene_panel_mismatch_fails(self):
        self.molecules.loc[0, "gene"] = "unmeasured_gene"
        with self.assertRaisesRegex(ValueError, "absent from measured panel"):
            self.run_conversion()

    def test_unverified_compartment_or_registration_fails(self):
        for section, field in (("compartments", "label_semantics_verified"), ("coordinates", "registration_verified")):
            with self.subTest(field=field):
                target = self.manifest["compartments"] if section == "compartments" else self.manifest["cells"]["coordinates"]
                target[field] = False
                with self.assertRaisesRegex(ValueError, "verified"):
                    self.run_conversion()
                target[field] = True

    def test_existing_output_not_overwritten(self):
        _, _, ready = self.run_conversion()
        path = self.root / "converted" / ready["data"]
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_conversion()
        self.assertEqual(original, path.read_bytes())

    def test_source_checksum_failure_has_no_published_output(self):
        self.manifest["molecules"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_conversion()
        self.assertFalse((self.root / "converted").exists())

    def test_nonfinite_cell_and_molecule_coordinates_rejected(self):
        self.cells.loc[0, "x"] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_conversion()
        self.cells.loc[0, "x"] = 0
        self.molecules.loc[0, "y"] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_conversion()

    def test_xenium_string_false_is_cytoplasmic(self):
        self.manifest["platform"] = "xenium"
        self.manifest["compartments"].pop("map")
        self.molecules["label"] = ["True", "False", "0.0", "0", "1", "1", "1"]
        stats, result, _ = self.run_conversion()
        self.assertEqual(stats["nuclear"], 1)
        self.assertEqual(stats["cytoplasmic"], 3)
        self.assertNotIn("membrane_counts", result.layers)

    def test_panel_file_preserves_zero_observed_gene(self):
        pd.DataFrame({"gene": ["G1", "G2", "G3", "G4"]}).to_csv(self.root / "panel.csv", index=False)
        self.manifest["panel"].pop("genes")
        self.manifest["panel"].update(path="panel.csv", gene_column="gene")
        stats, result, _ = self.run_conversion()
        self.assertEqual(result.n_vars, 4)
        self.assertEqual(stats["zero_observed_measured_genes"], 2)
        self.assertIn("panel", stats["source_files"])

    def test_aggregated_count_rows_are_not_accepted_as_individual_molecules(self):
        self.manifest["molecules"]["columns"]["count"] = "count"
        self.molecules["count"] = 1
        self.molecules.loc[0, "count"] = 4
        with self.assertRaisesRegex(ValueError, "count=1"):
            self.run_conversion()


if __name__ == "__main__":
    unittest.main()
