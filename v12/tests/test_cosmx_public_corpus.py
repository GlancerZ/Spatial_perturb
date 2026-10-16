import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import pandas as pd


spec = importlib.util.spec_from_file_location("prepare_cosmx_public_corpus", Path(__file__).resolve().parents[1] / "scripts" / "prepare_cosmx_public_corpus.py")
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


class CosMxCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = {
            "Lung12_metadata_file.csv": pd.DataFrame({"fov": [1, 2], "cell_ID": [1, 1], "CenterX_global_px": [10, 1000], "CenterY_global_px": [20, 1000], "CenterX_local_px": [10, 10], "CenterY_local_px": [20, 20]}),
            "Lung12_fov_positions_file.csv": pd.DataFrame({"fov": [1, 2], "x_global_px": [0, 990], "y_global_px": [0, 980]}),
            "Lung12_tx_file.csv": pd.DataFrame({"fov": [1, 1, 2, 2, 1, 1], "cell_ID": [1, 1, 1, 1, 0, 1],
                "x_global_px": [1, 2, 3, 4, 5, 6], "y_global_px": [1, 2, 3, 4, 5, 6],
                "target": ["G1", "G2", "G1", "G2", "G1", "NegPrb1"],
                "CellComp": ["Nuclear", "Cytoplasm", "Cytoplasmic", "Membrane", "0", "Nuclear"]}),
            "Lung12_exprMat_file.csv": pd.DataFrame({"fov": [1, 2], "cell_ID": [1, 1], "G1": [1, 1], "G2": [1, 1], "G3": [0, 0], "NegPrb1": [1, 0]})}

    def tearDown(self):
        self.temp.cleanup()

    def archive(self):
        path = self.root / "Lung12.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name, frame in self.frames.items():
                payload = frame.to_csv(index=False).encode()
                member = tarfile.TarInfo("vendor/" + name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        record = {"sample_id": "Lung12", "donor_id": "Lung12", "source_url": "https://example.test/source", "url": "https://example.test/archive", "size_bytes": path.stat().st_size, "sha256": corpus.digest_file(path)}
        return path, record

    def process(self):
        archive, record = self.archive()
        return corpus.process_sample(record, archive, self.root / "output", microns_per_pixel=0.18,
                                     coordinate_evidence=corpus.OFFICIAL_README, chunk_rows=2)

    def test_real_gene_reconstruction_includes_membrane_and_zero_genes(self):
        result = self.process()
        self.assertEqual(result["status"], "validated")
        self.assertEqual((result["cells"], result["genes"]), (2, 3))
        self.assertEqual((result["nuclear"], result["cytoplasmic"], result["membrane"]), (1, 2, 1))
        self.assertEqual(result["reference_validation"]["reference_transcripts"], 4)
        self.assertEqual(result["reference_validation"]["mismatched_entries"], 0)
        self.assertTrue((self.root / "output" / "COMPLETE.json").is_file())

    def test_reference_mismatch_never_marks_complete(self):
        self.frames["Lung12_exprMat_file.csv"].loc[0, "G1"] = 2
        with self.assertRaisesRegex(ValueError, "reconstruction failed"):
            self.process()
        self.assertFalse((self.root / "output" / "COMPLETE.json").exists())
        ready = json.loads((self.root / "output" / "bundle" / "READY.json").read_text())
        self.assertEqual(ready["training_eligibility"], "rejected_vendor_expression_mismatch")

    def test_missing_rna_columns_rejects_metadata_masquerading_as_transcripts(self):
        self.frames["Lung12_tx_file.csv"] = self.frames["Lung12_metadata_file.csv"]
        with self.assertRaisesRegex(ValueError, "actual RNA/cell fields"):
            self.process()

    def test_global_local_registration_mismatch_rejects_sample(self):
        self.frames["Lung12_metadata_file.csv"].loc[0, "CenterX_global_px"] = 100
        with self.assertRaisesRegex(ValueError, "disagree with local"):
            self.process()

    def test_tampered_archive_rejected_before_extraction(self):
        archive, record = self.archive()
        record["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA256"):
            corpus.process_sample(record, archive, self.root / "output", microns_per_pixel=.18, coordinate_evidence="official")
        self.assertFalse((self.root / "output").exists())

    def test_unsafe_members_and_extraction_budget(self):
        for case in ("traversal", "symlink", "budget"):
            with self.subTest(case=case):
                path = self.root / (case + ".tar.gz")
                with tarfile.open(path, "w:gz") as archive:
                    member = tarfile.TarInfo("../escape.csv" if case == "traversal" else "file.csv")
                    if case == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "/etc/passwd"
                        archive.addfile(member)
                    else:
                        member.size = 4
                        archive.addfile(member, io.BytesIO(b"data"))
                with self.assertRaises(ValueError):
                    corpus.safe_extract(path, self.root / case, max_unpacked_bytes=3)
        self.assertFalse((self.root.parent / "escape.csv").exists())

    def test_duplicate_flattened_paths_rejected(self):
        path = self.root / "duplicate.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for name in ("one/data.csv", "two/data.csv"):
                member = tarfile.TarInfo(name)
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            corpus.safe_extract(path, self.root / "extract", max_unpacked_bytes=100)

    def test_corpus_continues_after_missing_archive_and_reports_partial(self):
        _, record = self.archive()
        missing = {**record, "sample_id": "Lung6", "donor_id": "Lung6"}
        registry = self.root / "registry.json"
        registry.write_text(json.dumps({"cosmx_nsclc_flat_files": [missing, record]}))
        result = corpus.run_corpus(registry, self.root, self.root / "corpus", microns_per_pixel=.18,
                                   coordinate_evidence=corpus.OFFICIAL_README, chunk_rows=2)
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual((result["failed"], result["validated"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
