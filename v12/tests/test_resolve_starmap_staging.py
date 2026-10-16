import gzip
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location("resolve_starmap", Path(__file__).resolve().parents[1] / "scripts/stage_resolve_starmap.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
COLS = {"x": 0, "y": 1, "z": 2, "gene": 3, "cell_id": 4}


class ResolveStarmapStagingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make_tar(self, mismatch=False, gene_alias=False, extra_original=False, extra_assigned=False, bad_original=False):
        path = self.root / "raw.tar"
        with tarfile.open(path, "w") as archive:
            for gsm in ("GSM9616841", "GSM9616862"):
                original = "91\t211\t15\tCx3cl1\n1\t2\t3\tGeneX\n"
                assigned = "91\t211\t15\tCx3cl1\tCell87_73_184\n1\t2\t3\tGeneX\tNotAssigned\n"
                if gene_alias:
                    original = original.replace("Cx3cl1", "FP 48")
                    assigned = assigned.replace("Cx3cl1", "FP_48")
                if mismatch:
                    assigned = assigned.replace("91\t", "92\t", 1)
                if extra_original:
                    original += "4\t5\t6\tOriginalOnly\n"
                if extra_assigned:
                    assigned += "7\t8\t9\tAssignedOnly\tNotAssigned\n"
                if bad_original:
                    original = original.replace("91\t", "nan\t", 1)
                for suffix, text in (("_results.txt.gz", original), ("_results_assigned.txt.gz", assigned)):
                    payload = gzip.compress(text.encode())
                    member = tarfile.TarInfo(gsm + "_sample" + suffix)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
        return path

    def stage(self, **kwargs):
        return module.stage_resolve(self.make_tar(), self.root / "out", columns=COLS, has_header=False,
                                    expected_samples=2, rows_per_shard=2, **kwargs)

    def test_stages_real_rows_once_preserves_unknown_species_compartments_units(self):
        manifest = self.stage()
        rows = pq.read_table(self.root / "out/rna").to_pylist()
        self.assertEqual(len(rows), 4)
        self.assertEqual(manifest["unknown_species_samples"], 2)
        self.assertFalse(manifest["training_ready"])
        self.assertFalse(manifest["panel_complete"])
        self.assertEqual(rows[0]["x"], 91)
        self.assertEqual(rows[0]["cell_id"], "GSM9616841:Cell87_73_184")
        self.assertIsNone(rows[1]["cell_id"])
        self.assertTrue(all(row["compartment"] == "unknown" for row in rows))
        self.assertTrue(all(row["coordinate_unit"] == "unverified_source_units" for row in rows))
        self.assertNotEqual(rows[0]["fov"], rows[2]["fov"])
        self.assertTrue(all(sample["original_assigned_rows_identical"] for sample in manifest["samples"]))

    def test_species_only_from_explicit_geo_metadata(self):
        metadata = self.root / "family.soft"
        metadata.write_text("^SAMPLE = GSM9616841\n!Sample_organism_ch1 = Mus musculus\n^SAMPLE = GSM9616862\n!Sample_organism_ch1 = Homo sapiens\n")
        manifest = self.stage(geo_soft=metadata)
        self.assertEqual([sample["species"] for sample in manifest["samples"]], ["Mus musculus", "Homo sapiens"])
        self.assertEqual(manifest["unknown_species_samples"], 0)

    def test_mismatched_original_assignment_fails(self):
        with self.assertRaisesRegex(ValueError, "table mismatch"):
            module.stage_resolve(self.make_tar(mismatch=True), self.root / "out", columns=COLS, has_header=False)
        self.assertEqual(json.loads((self.root / "out/manifest.json").read_text())["status"], "failed")

    def test_explicit_gene_alias_preserves_assigned_label_and_counts(self):
        manifest = module.stage_resolve(self.make_tar(gene_alias=True), self.root / "out", columns=COLS,
            has_header=False, original_gene_aliases={"FP 48": "FP_48"})
        self.assertEqual(manifest["explicit_gene_alias_rows"], 2)
        self.assertEqual(manifest["gene_alias_matches"]["FP 48"], {"assigned_gene": "FP_48", "rows": 2})
        self.assertTrue(all(sample["explicit_gene_alias_rows"] == 1 for sample in manifest["samples"]))
        self.assertTrue(all(not sample["original_assigned_rows_identical"] for sample in manifest["samples"]))
        rows = pq.read_table(self.root / "out/rna").to_pylist()
        self.assertEqual(rows[0]["gene"], "FP_48")
        self.assertEqual(len(rows), 4)

    def test_gene_alias_does_not_allow_unlisted_names_or_coordinate_changes(self):
        with self.assertRaisesRegex(ValueError, "table mismatch"):
            module.stage_resolve(self.make_tar(gene_alias=True), self.root / "unlisted", columns=COLS,
                has_header=False, original_gene_aliases={"FP 47": "FP_47"})
        with self.assertRaisesRegex(ValueError, "table mismatch"):
            module.stage_resolve(self.make_tar(gene_alias=True, mismatch=True), self.root / "moved", columns=COLS,
                has_header=False, original_gene_aliases={"FP 48": "FP_48"})

    def test_report_records_differences_and_keeps_assigned_without_claiming_validation(self):
        manifest = module.stage_resolve(self.make_tar(mismatch=True, gene_alias=True, extra_original=True), self.root / "report",
            columns=COLS, has_header=False, reference_validation_mode="report")
        self.assertFalse(manifest["training_ready"])
        self.assertTrue(manifest["original_assigned_not_identical"])
        self.assertEqual(manifest["reference_validation_status"], "reported_not_identical")
        self.assertEqual(manifest["source_variant"], "author assigned molecule table")
        stats = manifest["reference_comparison"]
        self.assertEqual((stats["differing_pair_count"], stats["coordinate_diff"], stats["gene_diff"], stats["original_extra_rows"], stats["assigned_extra_rows"]), (2, 2, 2, 2, 0))
        self.assertTrue(all(not sample["original_assigned_rows_match_with_explicit_gene_aliases"] for sample in manifest["samples"]))
        rows = pq.read_table(self.root / "report/rna").to_pylist()
        self.assertEqual((len(rows), rows[0]["x"], rows[0]["gene"]), (4, 92, "FP_48"))
        self.assertEqual(manifest["explicit_gene_alias_rows"], 0)

    def test_report_keeps_assigned_extra_rows_and_still_rejects_nonfinite_reference(self):
        manifest = module.stage_resolve(self.make_tar(extra_assigned=True), self.root / "extra",
            columns=COLS, has_header=False, reference_validation_mode="report")
        self.assertEqual(manifest["reference_comparison"]["assigned_extra_rows"], 2)
        self.assertEqual(manifest["rna_rows"], 6)
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            module.stage_resolve(self.make_tar(bad_original=True), self.root / "bad",
                columns=COLS, has_header=False, reference_validation_mode="report")

    def test_coordinate_units_require_evidence(self):
        with self.assertRaisesRegex(ValueError, "require source evidence"):
            self.stage(coordinate_unit="micron")

    def test_decompressed_cap_checked(self):
        with self.assertRaisesRegex(ValueError, "byte cap"):
            self.stage(max_member_bytes=10)

    def test_starmap_appledouble_is_not_rna_mat(self):
        path = self.root / "starmap.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("sample/._goodPoints.mat", b"AppleDouble metadata")
            archive.writestr("sample/scaled/merged_dots.tif", b"image only")
        manifest = module.audit_starmap(path, self.root / "out")
        self.assertEqual(manifest["status"], "blocked_missing_molecule_points")
        self.assertEqual(len(manifest["appledouble_point_sidecars"]), 1)
        self.assertEqual(manifest["rna_rows"], 0)
        self.assertFalse(manifest["training_ready"])

    def test_starmap_real_points_are_unverified_until_schema_mapping(self):
        path = self.root / "starmap.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("sample/goodPoints.mat", b"schema not verified")
        manifest = module.audit_starmap(path, self.root / "out")
        self.assertEqual(manifest["status"], "blocked_unverified_point_schema")
        self.assertEqual(manifest["rna_rows"], 0)


if __name__ == "__main__":
    unittest.main()
