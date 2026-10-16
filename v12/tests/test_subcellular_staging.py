"""Synthetic format/annotation-safety tests; run on a compute node."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pyarrow.parquet as pq
from scipy.io import savemat

from scripts.stage_unlabeled_subcellular import (
    SEQFISH_NAMES, SEQFISH_RUNS, VIZGEN_MEMBER, stage_seqfish, stage_vizgen,
)


def _mat_bytes(values):
    stream = io.BytesIO()
    savemat(stream, values)
    return stream.getvalue()


class StagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def seqfish_zip(self):
        path = self.root / "seqfish.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(SEQFISH_NAMES, _mat_bytes({"allNames": np.array(
                [["GeneA", "GeneB", "NeverExpressed"]], dtype=object)}))
            for index, shape in enumerate(((2, 3, 3), (3, 2, 3))):
                points = np.empty(shape, dtype=object)
                for location in np.ndindex(shape):
                    points[location] = np.empty((0, 3))
                points[0, 0, 0] = np.array([[11.5, 22.25, 1.0], [12, 23, 1]])
                points[shape[0] - 1, shape[1] - 1, 1] = np.array([[99, 101, 1]])
                archive.writestr(SEQFISH_RUNS[index], _mat_bytes({"tot": points}))
        return path

    def vizgen_zip(self, content, member=VIZGEN_MEMBER):
        path = self.root / "vizgen.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(member, content)
        return path

    def test_seqfish_preserves_panel_assignment_and_raw_coordinates(self):
        output = self.root / "staged"
        manifest = stage_seqfish(self.seqfish_zip(), output, expected_gene_count=3,
                                 rows_per_shard=2)
        rows = pq.read_table(output / "rna").to_pylist()
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["z"] is None for row in rows))
        self.assertTrue(all(row["auxiliary_label"] == 1 for row in rows))
        self.assertTrue(all(row["compartment"] == "unknown" for row in rows))
        self.assertTrue(all(row["coordinate_unit"] == "pixel" for row in rows))
        self.assertEqual((rows[0]["x"], rows[0]["y"]), (11.5, 22.25))
        self.assertEqual(len({row["cell_id"] for row in rows}), 4)
        self.assertTrue(any(row["cell_id"].endswith("run1:fov1:cell2") for row in rows))
        self.assertTrue(any(row["cell_id"].endswith("run2:fov2:cell1") for row in rows))
        panel = pq.read_table(output / "panel.parquet").to_pylist()
        self.assertEqual(panel[-1], {"gene": "NeverExpressed", "source_gene_index": 2,
                                     "assayed": True, "observed": False})
        self.assertFalse(manifest["training_ready"])
        self.assertEqual(manifest["coordinate_transform"], "identity")
        self.assertEqual(manifest["measured_gene_count"], 3)
        self.assertTrue(manifest["panel_complete"])
        self.assertEqual([part["rows"] for part in manifest["rna_shards"]], [2, 2, 2])
        self.assertEqual(len(manifest["input"]["sha256"]), 64)

    def test_vizgen_global_coordinates_unassigned_and_panel_unknown(self):
        path = self.vizgen_zip("gene,fov,global_x,global_y,global_z,x,y\n"
                               "GeneA,2,7.125,8.5,3,999,888\nGeneB,3,9,10,4,777,666\n")
        output = self.root / "staged"
        manifest = stage_vizgen(path, output, rows_per_shard=1)
        rows = pq.read_table(output / "rna").to_pylist()
        self.assertEqual((rows[0]["x"], rows[0]["y"], rows[0]["z"]), (7.125, 8.5, 3))
        self.assertEqual(rows[0]["z_unit"], "unverified_source_global_z")
        self.assertTrue(all(row["cell_id"] is None for row in rows))
        self.assertFalse(manifest["training_ready"])
        self.assertFalse(manifest["panel_complete"])
        self.assertIsNone(manifest["measured_panel"])
        self.assertFalse((output / "panel.parquet").exists())
        self.assertTrue(all(row["assayed"] is None for row in
                            pq.read_table(output / "observed_genes.parquet").to_pylist()))

    def test_vizgen_rejects_missing_rna_columns(self):
        path = self.vizgen_zip("gene,fov,x,y\nGeneA,0,1,2\n")
        output = self.root / "staged"
        with self.assertRaisesRegex(ValueError, "Missing RNA columns"):
            stage_vizgen(path, output)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["training_ready"])

    def test_nonfinite_coordinate_rejected(self):
        path = self.vizgen_zip("gene,fov,global_x,global_y\nGeneA,0,nan,2\n")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            stage_vizgen(path, self.root / "staged")

    def test_member_byte_cap_and_sha_checked(self):
        path = self.vizgen_zip("gene,fov,global_x,global_y\nGeneA,0,1,2\n")
        with self.assertRaisesRegex(ValueError, "exceeds byte cap"):
            stage_vizgen(path, self.root / "too_large", max_member_bytes=10)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            stage_vizgen(path, self.root / "bad_sha", expected_sha256="0" * 64)
        self.assertFalse((self.root / "bad_sha").exists())

    def test_unsafe_member_and_existing_output_rejected(self):
        path = self.vizgen_zip("gene,fov,global_x,global_y\nGeneA,0,1,2\n", "../rna.csv")
        with self.assertRaisesRegex(ValueError, "Unsafe archive"):
            stage_vizgen(path, self.root / "unsafe", member="../rna.csv")
        self.assertFalse((self.root.parent / "rna.csv").exists())
        with self.assertRaises(FileExistsError):
            stage_vizgen(path, self.root / "unsafe")


if __name__ == "__main__":
    unittest.main()
