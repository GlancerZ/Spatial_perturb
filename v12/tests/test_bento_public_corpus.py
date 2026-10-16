"""Bento geometry/scientific boundary tests. Run only on compute nodes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from shapely.geometry import Polygon, box


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_bento_public_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_bento_public_corpus", SCRIPT)
bento = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bento)


class BentoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "published.h5ad"
        obs = pd.DataFrame({"cell_shape": [box(0, 0, 10, 10).wkt] * 3,
                            "nucleus_shape": [box(2, 2, 4, 4).wkt] * 2 + ["None"],
                            "batch": ["A", "B", "B"]}, index=["a", "b", "c"])
        self.data = ad.AnnData(sparse.csr_matrix([[2, 1, 1], [1, 1, 0], [1, 0, 0]]),
                               obs=obs, var=pd.DataFrame(index=["G1", "G2", "NegProbe"]))
        self.data.uns["points"] = pd.DataFrame({"cell": ["a", "a", "a", "b", "b", "c", "a"],
                                               "gene": ["G1", "G2", "NegProbe", "G1", "G2", "G1", "G1"],
                                               "x": [3., 8., 3., 3., 8., 5., 12.],
                                               "y": [3., 8., 3., 3., 8., 5., 12.]})
        # Deliberately misleading layers prove classification comes from geometry.
        self.data.layers["spliced"] = sparse.csr_matrix(np.full((3, 3), 99))
        self.data.layers["unspliced"] = sparse.csr_matrix((3, 3))
        self.spec = {"schema_version": 1, "dataset_id": "bento_fixture", "platform": "merfish",
                     "species": "Homo sapiens", "tissue": "fixture", "source": {
                         "path": str(self.input), "url": "https://example.org/published", "license": "MIT"},
                     "panel": {"complete_original_measured_panel": False, "evidence": "Published retained genes"},
                     "geometry_evidence": "Published matched WKT nucleus and cell polygons",
                     "batch_evidence": "Independent FOVs; source batch is retained",
                     "controls": {"genes": ["NegProbe"], "prefixes": [], "expected_count": 1, "evidence": "Fixture codebook"},
                     "coordinates": {"unit": "native", "microns_per_unit": None}}

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        self.data.write_h5ad(self.input)
        self.spec["source"]["expected_md5"] = hashlib.md5(self.input.read_bytes()).hexdigest()
        path = self.root / "spec.json"
        path.write_text(json.dumps(self.spec))
        return path

    def calibration(self, **updates):
        report = {"status": "passed", "microns_per_pixel": .2,
                  "published_h5ad_sha256": hashlib.sha256(self.input.read_bytes()).hexdigest(),
                  "evidence": "Synthetic fixture calibration, not a real platform scale",
                  "all_published_batches_verified": True, "verified_batch_count": 2,
                  "published_batch_ids": ["A", "B"]}
        report.update(updates)
        path = self.root / "calibration.json"
        path.write_text(json.dumps(report))
        return path

    def test_unknown_units_stage_without_claiming_training_ready(self):
        result = bento.stage_bento(self.write(), self.root / "staged", chunk_rows=2)
        self.assertEqual(result["status"], "staged_native_coordinates_need_calibration")
        self.assertFalse(result["training_split_ready"])
        self.assertEqual(result["published_X_mismatched_entries"], 0)
        self.assertEqual(result["eligible_geometry_cells"], 2)
        self.assertEqual(result["excluded_geometry_points"], 1)
        self.assertEqual(result["outside_cell_points"], 1)
        self.assertEqual(result["control_points"], 1)
        self.assertEqual((result["nuclear_points"], result["cytoplasmic_points"]), (2, 2))
        self.assertFalse(result["spliced_unspliced_used"])
        self.assertFalse(list((self.root / "staged").glob("**/*.h5ad")))
        for row in result["batches"]:
            frame = pd.read_csv(Path(row["directory"]) / "molecules.csv")
            self.assertEqual(frame["fov"].nunique(), 1)

    def test_calibrated_batches_convert_separately(self):
        spec_path = self.write()
        result = bento.stage_bento(spec_path, self.root / "converted", chunk_rows=3,
                                  microns_per_pixel=.2, calibration_evidence=self.calibration())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(list((self.root / "converted").glob("batch_*/bundle/*.h5ad"))), 2)
        for path in (self.root / "converted").glob("batch_*/bundle/*.h5ad"):
            data = ad.read_h5ad(path)
            self.assertEqual(data.layers["nuclear_counts"].sum(), 1)
            self.assertEqual(data.layers["extranuclear_counts"].sum(), 1)
            self.assertEqual(data.n_vars, 2)
            np.testing.assert_allclose(data.obsm["spatial"], [[1, 1]])
            ready = json.loads((path.parent / "READY.json").read_text())
            self.assertEqual(ready["training_eligibility"], "validated_nuclear_geometry_and_counts")
            self.assertFalse(ready["training_split_ready"])

    def test_calibration_must_pass_and_match_source_batches_and_scale(self):
        spec_path = self.write()
        cases = [{"status": "failed"}, {"published_h5ad_sha256": "0" * 64},
                 {"all_published_batches_verified": False}, {"verified_batch_count": 1},
                 {"published_batch_ids": ["A", "C"]}, {"microns_per_pixel": .108}]
        for updates in cases:
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                bento.stage_bento(spec_path, self.root / "bad", microns_per_pixel=.2,
                                  calibration_evidence=self.calibration(**updates))
        self.assertFalse((self.root / "bad").exists())

    def test_scale_without_passed_evidence_is_rejected(self):
        self.spec["coordinates"] = {"unit": "pixel", "microns_per_unit": .2, "evidence": "Unsupported claim"}
        with self.assertRaisesRegex(ValueError, "passed calibration evidence"):
            bento.stage_bento(self.write(), self.root / "bad")

    def test_mouse_preserves_gene_names_but_does_not_enter_human_corpus(self):
        self.spec["species"] = "Mus musculus"
        self.data.var_names = ["G1", "G2", "NegProbe"]
        spec_path = self.write()
        result = bento.stage_bento(spec_path, self.root / "mouse", microns_per_pixel=.2,
                                  calibration_evidence=self.calibration())
        self.assertEqual(result["status"], "staged_species_requires_matching_model")
        self.assertFalse(result["human_model_compatible"])
        self.assertEqual(pd.read_csv(self.root / "mouse/batch_0000/panel.csv")["gene"].tolist(), ["G1", "G2"])
        self.assertFalse(list((self.root / "mouse").glob("**/*.h5ad")))

    def test_reference_count_corruption_blocks_bundles(self):
        self.data.X[0, 0] = 5
        with self.assertRaisesRegex(ValueError, "disagree with published X"):
            bento.stage_bento(self.write(), self.root / "bad")
        self.assertEqual(json.loads((self.root / "bad/status.json").read_text())["status"], "failed")
        self.assertFalse(list((self.root / "bad").glob("**/READY.json")))

    def test_cross_batch_point_mismatch_fails(self):
        self.data.uns["points"]["batch"] = ["B"] * 7
        with self.assertRaisesRegex(ValueError, "RNA batch disagrees"):
            bento.stage_bento(self.write(), self.root / "bad")

    def test_polygon_holes_and_boundary_policy(self):
        cell = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)],
                       holes=[[(6, 6), (8, 6), (8, 8), (6, 8)]])
        labels = bento.classify_points(cell, box(2, 2, 4, 4), [[3, 3], [2, 2], [7, 7], [0, 0]])
        self.assertEqual(list(labels), ["nuclear", "nuclear", "outside_cell", "cytoplasmic"])

    def test_bad_or_outside_nucleus_not_repaired(self):
        self.assertEqual(bento.parse_shapes(box(0, 0, 10, 10).wkt, "None")[2], "missing_nucleus_shape")
        self.assertEqual(bento.parse_shapes(box(0, 0, 10, 10).wkt, box(9, 9, 12, 12).wkt)[2], "nucleus_not_contained_in_cell")
        bowtie = "POLYGON ((0 0, 4 4, 0 4, 4 0, 0 0))"
        self.assertEqual(bento.parse_shapes(bowtie, box(1, 1, 2, 2).wkt)[2], "invalid_cell_shape")

    def test_guessed_native_scale_rejected(self):
        self.spec["coordinates"].update(microns_per_unit=.103, evidence="Unknown OME scale")
        with self.assertRaisesRegex(ValueError, "guessing"):
            bento.stage_bento(self.write(), self.root / "bad")

    def test_unverified_control_count_rejected(self):
        self.spec["controls"]["expected_count"] = 5
        with self.assertRaisesRegex(ValueError, "control feature count"):
            bento.stage_bento(self.write(), self.root / "bad")

    def test_checksum_failure_and_no_overwrite(self):
        spec_path = self.write()
        self.spec["source"]["expected_md5"] = "0" * 32
        spec_path.write_text(json.dumps(self.spec))
        with self.assertRaisesRegex(ValueError, "MD5"):
            bento.stage_bento(spec_path, self.root / "bad")
        good = self.write()
        bento.stage_bento(good, self.root / "same")
        with self.assertRaises(FileExistsError):
            bento.stage_bento(good, self.root / "same")


if __name__ == "__main__":
    unittest.main()
