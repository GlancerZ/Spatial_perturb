"""Cache migration preserves counts while explicitly checking/remapping genes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from spatial_pretrain.data.subcellular_v2 import vocabulary_path


_spec = importlib.util.spec_from_file_location(
    "v2_cache_test_fixtures", Path(__file__).with_name("test_subcellular_v2_data.py"))
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
prepare = _fixtures.prepare


class CacheReuseTests(unittest.TestCase):
    setUp = _fixtures.CachePreparationTests.setUp
    tearDown = _fixtures.CachePreparationTests.tearDown
    prepare = _fixtures.CachePreparationTests.prepare

    def migrate(self, vocab, sha=None):
        if sha is None:
            sha = hashlib.sha256(json.dumps(vocab, indent=4).encode()).hexdigest()
        return prepare.prepare_slice(
            self.source, self.old, self.root / "migrated", vocab, sha, self.data,
            reuse_folder=self.cache / "slice")

    def test_identical_vocabulary_with_new_json_format_reuses_large_arrays(self):
        self.prepare()
        migrated = self.migrate(self.vocab)
        self.assertFalse(migrated["cache_reuse"]["gene_ids_remapped"])
        self.assertNotEqual(migrated["signature"]["vocab_sha256"], self.vocab_sha)
        self.assertEqual((self.root / "migrated" / "nuclear_log1p_float32.npy").stat().st_ino,
                         (self.cache / "slice" / "nuclear_log1p_float32.npy").stat().st_ino)
        self.assertEqual(migrated["cache_dir"], str((self.root / "migrated").resolve()))

    def test_changed_vocabulary_ids_are_explicitly_remapped(self):
        self.prepare()
        vocab = dict(self.vocab, A=11, B=7)
        migrated = self.migrate(vocab)
        self.assertTrue(migrated["cache_reuse"]["gene_ids_remapped"])
        np.testing.assert_array_equal(np.load(self.root / "migrated" / "gene_ids.npy"), [11, 7])
        np.testing.assert_array_equal(np.load(self.cache / "slice" / "gene_ids.npy"), [2, 3])
        np.testing.assert_array_equal(np.load(self.root / "migrated" / "nuclear_log1p_float32.npy"),
                                      np.log1p(self.nuclear[:, :2].astype(np.float32)))

    def test_changed_panel_coverage_rebuilds_from_raw_source(self):
        self.prepare()
        migrated = self.migrate(dict(self.vocab, unmatched=9))
        self.assertNotIn("cache_reuse", migrated)
        self.assertEqual(migrated["num_genes"], 3)
        np.testing.assert_array_equal(np.load(self.root / "migrated" / "nuclear_log1p_float32.npy"),
                                      np.log1p(self.nuclear.astype(np.float32)))

    def test_broken_names_cannot_silently_reuse_expression_columns(self):
        self.prepare()
        (self.cache / "slice" / "gene_names.txt").write_text("A\n")
        with self.assertRaisesRegex(ValueError, "Gene names/count mismatch"):
            self.migrate(self.vocab)
        self.assertFalse((self.root / "migrated").exists())

    def test_plain_scgpt_assets_and_ambiguous_roots(self):
        (self.assets / "vocab.json").unlink()
        nested = self.assets / "scGPT_human"
        nested.mkdir()
        (nested / "vocab.json").write_text(json.dumps(self.vocab))
        self.assertEqual(vocabulary_path(self.assets), nested / "vocab.json")
        spatial = self.assets / "scGPT_spatial_v1"
        spatial.mkdir()
        (spatial / "vocab.json").write_text(json.dumps(self.vocab))
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            vocabulary_path(self.assets)


if __name__ == "__main__":
    unittest.main()
