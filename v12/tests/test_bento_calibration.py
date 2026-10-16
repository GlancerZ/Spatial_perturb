"""Synthetic calibration and identity tests; compute-node execution only."""
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import h5py
import numpy as np
import pandas as pd

from scripts.verify_bento_merfish_calibration import raw_inputs, verify


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def fixtures(self, *, y_scale=0.108, corrupt_identity=False, outlier=False,
                 duplicate_published=False, cross_fov_collision=False, ambiguous_batch=False):
        rows, published = [], []
        for fov in range(2):
            for index in range(50):
                x = 11.25 + 17 * index + 0.17 * fov
                y = 27.125 + (index * 23 % 97) + 0.27 * fov
                gene = "GeneA" if index % 2 else "GeneB"
                rows.append({"x": x, "y": y, "global_x": -10000 + 220 * fov + 0.108 * x,
                             "global_y": -3000 + 190 * fov + y_scale * y,
                             "global_z": fov, "fov": fov, "barcode_id": index % 2})
                # Intentionally reverse the published batch order vs raw FOV.
                published.append({"x": x + (0.5 if corrupt_identity and fov == 0 else 0),
                                  "y": y, "gene": gene, "batch": str(1 - fov)})
        if duplicate_published:
            published.append(dict(published[0]))
        if cross_fov_collision or ambiguous_batch:
            # Raw RNA can share a local coordinate across independent FOVs.
            # These extra raw points need not be inside a published cell.
            for original in rows[:50 if ambiguous_batch else 1]:
                repeated = dict(original)
                repeated.update(fov=1, global_z=1,
                                global_x=-10000 + 220 + 0.108 * repeated["x"],
                                global_y=-3000 + 190 + y_scale * repeated["y"])
                rows.append(repeated)
        if outlier:
            rows[-1]["global_x"] += 0.5
        zipped = io.BytesIO()
        with zipfile.ZipFile(zipped, "w") as archive:
            archive.writestr("barcodes.csv", pd.DataFrame(rows).to_csv(index=False))
        raw = self.root / "raw.tar.gz"
        with tarfile.open(raw, "w:gz") as archive:
            for name, value in [("raw/barcodes.zip", zipped.getvalue()),
                                ("raw/codebook_alex_130gene.csv", b"name\nGeneB\nGeneA\n"),
                                ("raw/.ipynb_checkpoints/codebook_alex_130gene-checkpoint.csv",
                                 b"name\nWrongB\nWrongA\n")]:
                info = tarfile.TarInfo(name)
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
        h5ad = self.root / "published.h5ad"
        with h5py.File(h5ad, "w") as handle:
            points = handle.create_group("uns/points")
            for key in ("x", "y"):
                points.create_dataset(key, data=[row[key] for row in published])
            for key, categories in [("batch", ["0", "1"]), ("gene", ["GeneA", "GeneB"])]:
                column = points.create_group(key)
                column.create_dataset("codes", data=[categories.index(row[key]) for row in published])
                column.create_dataset("categories", data=np.asarray(categories, dtype="S"))
            handle.create_group("obs").create_dataset("batch", data=np.asarray(["0", "1"], dtype="S"))
        return raw, h5ad

    def test_all_fovs_calibrated_and_batch_mapping_inferred(self):
        raw, h5ad = self.fixtures()
        report = verify(raw, h5ad, expected_fov_count=2, samples_per_batch=12, chunk_rows=17)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["raw_rna_rows"], 100)
        self.assertEqual(report["published_batch_to_raw_fov"], {"0": "1", "1": "0"})
        self.assertTrue(report["all_published_batches_verified"])
        self.assertFalse(report["all_published_molecules_identity_verified"])
        self.assertFalse(report["global_z"]["physical_depth_verified"])
        self.assertEqual(report["global_z"]["equal_to_fov_rows"], 100)
        self.assertEqual(len(report["published_h5ad_sha256"]), 64)
        for fit in report["fov_fits"].values():
            self.assertAlmostEqual(fit["axes"]["x"]["fitted_slope_um_per_pixel"], 0.108)

    def test_anisotropic_scale_rejected(self):
        raw, h5ad = self.fixtures(y_scale=0.12)
        report = verify(raw, h5ad, expected_fov_count=2, samples_per_batch=12)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["all_raw_fovs_calibrated"])

    def test_last_raw_row_outlier_rejected(self):
        raw, h5ad = self.fixtures(outlier=True)
        report = verify(raw, h5ad, expected_fov_count=2, samples_per_batch=12, chunk_rows=13)
        self.assertEqual(report["status"], "failed")
        self.assertGreater(report["fov_fits"]["1"]["axes"]["x"]["target_scale_max_abs_residual_um"], 0.4)

    def test_unrelated_published_coordinates_rejected(self):
        raw, h5ad = self.fixtures(corrupt_identity=True)
        report = verify(raw, h5ad, expected_fov_count=2, samples_per_batch=12)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["all_published_batches_verified"])

    def test_duplicate_published_rows_each_receive_a_match(self):
        raw, h5ad = self.fixtures(duplicate_published=True)
        report = verify(raw, h5ad, expected_fov_count=2, samples_per_batch=100, chunk_rows=17)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["identity_by_batch"]["1"]["sampled_rows"], 51)
        self.assertEqual(report["identity_by_batch"]["1"]["matched_rows"], 51)

    def test_single_coordinate_collision_preserves_unique_batch_identity(self):
        raw, h5ad = self.fixtures(cross_fov_collision=True)
        report = verify(raw, h5ad, expected_fov_count=2, samples_per_batch=100, chunk_rows=17)
        self.assertEqual(report["status"], "passed")
        batch = report["identity_by_batch"]["1"]
        self.assertEqual(batch["pointwise_ambiguous_rows"], 1)
        self.assertEqual(batch["pointwise_unique_rows"], 49)
        self.assertEqual(batch["candidate_raw_fovs"], ["0"])
        self.assertEqual(batch["pointwise_ambiguity_examples"][0]["raw_fovs"], ["0", "1"])

    def test_entire_batch_matching_two_fovs_still_rejected(self):
        raw, h5ad = self.fixtures(ambiguous_batch=True)
        report = verify(raw, h5ad, expected_fov_count=2, samples_per_batch=100, chunk_rows=17)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["all_published_batches_verified"])
        self.assertEqual(report["identity_by_batch"]["1"]["candidate_raw_fovs"], ["0", "1"])

    def test_archive_traversal_rejected_without_extraction(self):
        path = self.root / "bad.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            entry = tarfile.TarInfo("../barcodes.csv")
            value = b"x,y\n1,2\n"
            entry.size = len(value)
            archive.addfile(entry, io.BytesIO(value))
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            with raw_inputs(path):
                pass
        self.assertFalse((self.root.parent / "barcodes.csv").exists())


if __name__ == "__main__":
    unittest.main()
