import importlib.util
from pathlib import Path
import unittest
import numpy as np
import hashlib
import json
import tempfile
import h5py

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("unified_geometry", ROOT / "scripts/prepare_unified_training_cache.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeometryTest(unittest.TestCase):
    def test_overlapping_fovs_never_connect(self):
        coords = np.array([[0, 0], [1, 0], [0, 0], [1, 0], [0, 0]], dtype=float)
        groups = np.array(["a", "a", "b", "b", "c"])
        neighbors = module.build_geometry(coords, groups, 3)
        np.testing.assert_array_equal(neighbors[:, 0], [1, 0, 3, 2, -1])
        self.assertTrue((neighbors[:, 1:] == -1).all())

    def test_coordinate_ties_remove_actual_self(self):
        coords = np.zeros((4, 2))
        neighbors = module.build_geometry(coords, np.array(["a"] * 4), 3)
        for i, row in enumerate(neighbors):
            self.assertEqual(set(row), set(range(4)) - {i})

    def test_bad_domain_and_coordinates_fail(self):
        with self.assertRaises(ValueError):
            module.build_geometry(np.array([[np.nan, 0]]), np.array(["a"]))
        with self.assertRaises(ValueError):
            module.build_geometry(np.zeros((1, 2)), np.array([""]))

    def test_verified_package_rejects_pixel_units_and_modified_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "example.h5ad"
            with h5py.File(source, "w") as handle:
                handle["uns/unified_compartment_schema_version"] = 1
                handle["uns/compartment_source/spatial_unit"] = "pixel"
            ready = {"data": source.name, "status": "validated_compartment_counts",
                     "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            (root / "READY.json").write_text(json.dumps(ready))
            with self.assertRaisesRegex(ValueError, "micrometer"):
                module.verify_bundle(source)
            with source.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                module.verify_bundle(source)

    def test_cosmx_pending_reference_is_not_cache_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "example.h5ad"
            with h5py.File(source, "w") as handle:
                handle["uns/unified_compartment_schema_version"] = 1
                handle["uns/compartment_source/spatial_unit"] = "micron"
                handle["uns/compartment_source/manifest_json"] = json.dumps({"platform": "cosmx",
                    "cells": {"coordinates": {"registration_verified": True}}})
            (root / "READY.json").write_text(json.dumps({"data": source.name,
                "status": "validated_compartment_counts",
                "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
            (root / "source_manifest.json").write_text(json.dumps({"platform": "cosmx",
                "cells": {"coordinates": {"registration_verified": True}}}))
            with self.assertRaisesRegex(ValueError, "vendor-expression"):
                module.verify_bundle(source)

    def test_manifest_binding_and_eligibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "example.h5ad"
            manifest = {"platform": "other", "cells": {"coordinates": {"registration_verified": True}}}
            with h5py.File(source, "w") as handle:
                handle["uns/unified_compartment_schema_version"] = 1
                handle["uns/compartment_source/spatial_unit"] = "micron"
                handle["uns/compartment_source/manifest_json"] = json.dumps(manifest)
            ready = {"data": source.name, "status": "validated_compartment_counts",
                     "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                     "training_eligibility": "source_policy_review_required"}
            (root / "READY.json").write_text(json.dumps(ready))
            (root / "source_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "eligibility"):
                module.verify_bundle(source)
            ready["training_eligibility"] = "validated_nuclear_geometry_and_counts"
            (root / "READY.json").write_text(json.dumps(ready))
            module.verify_bundle(source)
            (root / "source_manifest.json").write_text(json.dumps(dict(manifest, platform="cosmx")))
            with self.assertRaisesRegex(ValueError, "differs"):
                module.verify_bundle(source)

    def test_geometry_fingerprint_changes_with_grouping_or_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "coords.npy", np.zeros((2, 2)))
            np.save(root / "knn_indices.npy", np.array([[1], [0]]))
            first = module.geometry_fingerprint(root, np.array(["a", "a"]), "fov")
            second = module.geometry_fingerprint(root, np.array(["a", "b"]), "fov")
            self.assertNotEqual(first, second)
            np.save(root / "knn_indices.npy", np.array([[-1], [-1]]))
            self.assertNotEqual(first, module.geometry_fingerprint(root, np.array(["a", "a"]), "fov"))
