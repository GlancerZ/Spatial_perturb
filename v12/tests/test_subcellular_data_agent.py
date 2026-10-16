"""Unit tests for the stdlib-only subcellular data collection agent.

These tests use temporary local files and mocked responses; they require no
network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "subcellular_data_agent.py"
SPEC = importlib.util.spec_from_file_location("subcellular_data_agent", SCRIPT)
assert SPEC and SPEC.loader
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)


def source_record(**overrides):
    value = {
        "id": "sample",
        "title": "Sample",
        "platform": "Xenium",
        "species": ["human"],
        "tissue": "lung",
        "source_url": "https://example.org/study",
        "discovery_urls": [],
        "accession": "GSE12345",
        "license": "unknown",
        "evidence": [
            {"claim": "molecule_coordinates", "url": "https://example.org/docs", "detail": "documented"},
            {"claim": "cell_assignment", "url": "https://example.org/docs", "detail": "documented"},
            {"claim": "nucleus_assignment", "url": "https://example.org/docs", "detail": "documented"},
        ],
        "capabilities": {
            "molecule_coordinates": "yes",
            "nucleus_assignment": "yes",
            "nucleus_boundaries": "unknown",
            "cell_boundaries": "unknown",
            "registered_images": "unknown",
            "coordinate_registration": "unknown",
            "cell_assignment": "yes",
        },
        "evidence_level": "documented",
        "compartment_method": "explicit_assignment",
        "priority": 1,
        "notes": "",
        "assets": [{"url": "https://example.org/data/sample.h5ad", "role": "matrix"}],
        "source_kind": "dataset",
    }
    value.update(overrides)
    return value


def collect_args(registry: Path, output: Path, **overrides):
    values = {
        "registry": str(registry), "output": str(output), "online": False,
        "max_pages": 40, "max_page_bytes": 2 * 1024 * 1024, "timeout": 10,
        "watch_seconds": None, "inventory": None, "page_cache": None,
        "emit_requests": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class FakeResponse:
    def __init__(self, body: bytes, *, url: str = "https://example.org/file", headers=None):
        self._stream = io.BytesIO(body)
        self._url = url
        self.headers = headers or {}
        self.status = 200

    def read(self, size=-1):
        return self._stream.read(size)

    def geturl(self):
        return self._url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class ClassificationTests(unittest.TestCase):
    def test_negative_or_uncited_claim_is_not_positive_support(self):
        for claim in ("nucleus_assignment unavailable", "[nucleus_assignment=no]", "no_nucleus_assignment"):
            row = source_record()
            row["evidence"] = [item for item in row["evidence"] if item["claim"] != "nucleus_assignment"]
            row["evidence"].append({"claim": claim, "url": "https://example.org/docs", "detail": "not available"})
            self.assertEqual(agent.classify_source(row)[0], "candidate")
        row = source_record()
        row["evidence"] = [{"claim": name} for name in ("molecule_coordinates", "cell_assignment", "nucleus_assignment")]
        self.assertEqual(agent.classify_source(row)[0], "candidate")

    def test_processed_matrix_and_landing_page_are_not_training_inputs(self):
        row = source_record()
        row["capabilities"]["molecule_coordinates"] = "no"
        self.assertFalse(agent._asset_candidate(row, {"url": "https://example.org/data.h5ad", "role": "matrix"})["training_candidate"])
        self.assertFalse(agent._asset_candidate(source_record(), {"url": "https://example.org/study", "role": "dataset_landing_and_download"})["training_candidate"])
        self.assertTrue(agent._asset_candidate(source_record(), {"url": "https://example.org/raw.zip", "role": "raw_output_bundle"})["training_candidate"])

    def test_documented_direct_requires_claims_and_yes_capabilities(self):
        classification, _ = agent.classify_source(source_record())
        self.assertEqual(classification, "documented_direct")

        missing_claim = source_record(evidence=[
            {"claim": "molecule_coordinates", "url": "https://example.org/docs", "detail": "only one"}
        ])
        classification, reasons = agent.classify_source(missing_claim)
        self.assertEqual(classification, "candidate")
        self.assertIn("missing_supported_cell_assignment", reasons)

        negative = source_record()
        negative["capabilities"]["nucleus_assignment"] = "no"
        self.assertEqual(agent.classify_source(negative)[0], "candidate")

    def test_geometry_unknown_and_exclusions_are_conservative(self):
        geometry = source_record(
            compartment_method="registered_boundaries",
            evidence=[
                {"claim": name, "url": "https://example.org/docs", "detail": "documented"}
                for name in ("molecule_coordinates", "nucleus_boundaries", "cell_boundaries", "coordinate_registration")
            ],
            capabilities={
                "molecule_coordinates": "yes", "nucleus_assignment": "unknown",
                "nucleus_boundaries": "yes", "cell_boundaries": "yes",
                "registered_images": "yes", "coordinate_registration": "yes",
                "cell_assignment": "unknown",
            },
        )
        self.assertEqual(agent.classify_source(geometry)[0], "documented_geometry")
        geometry["capabilities"]["coordinate_registration"] = "unknown"
        self.assertEqual(agent.classify_source(geometry)[0], "candidate")
        self.assertEqual(agent.classify_source(source_record(source_kind="catalog"))[0], "discovery_only")
        self.assertEqual(
            agent.classify_source(source_record(training_use="format_validation_only"))[0],
            "format_validation_only",
        )


class SafetyAndFormattingTests(unittest.TestCase):
    def test_url_policy_rejects_non_http_and_non_public_literals(self):
        for value in (
            "file:///etc/passwd", "http://127.0.0.1/a", "http://10.1.2.3/a",
            "http://169.254.169.254/latest/meta-data", "http://localhost/a",
            "https://user:pass@example.org/a",
        ):
            with self.subTest(value=value), self.assertRaises(agent.UnsafeURLError):
                agent.validate_public_url(value, resolve=False)
        self.assertEqual(
            agent.validate_public_url("https://example.org/a", resolve=False),
            "https://example.org/a",
        )

    def test_resolved_private_address_is_rejected(self):
        answer = [(2, 1, 6, "", ("192.168.1.4", 443))]
        with mock.patch.object(agent.socket, "getaddrinfo", return_value=answer):
            with self.assertRaises(agent.UnsafeURLError):
                agent.validate_public_url("https://example.org/a")

    def test_tsv_formula_protection_and_quoting(self):
        rendered = agent.render_tsv(
            [{"a": "=2+2", "b": "line\tvalue\nnext", "c": "@cmd"}],
            ("a", "b", "c"),
        )
        row = list(csv.DictReader(io.StringIO(rendered), delimiter="\t"))[0]
        self.assertEqual(row["a"], "'=2+2")
        self.assertEqual(row["b"], "line\tvalue\nnext")
        self.assertEqual(row["c"], "'@cmd")

    def test_path_traversal_is_rejected(self):
        for value in ("../x", "a/b", "a\\b", ".."):
            with self.assertRaises(ValueError):
                agent.safe_filename(value, "https://example.org/x")
        with self.assertRaises(ValueError):
            agent.safe_component("../source")

    def test_declared_page_size_is_enforced(self):
        response = FakeResponse(b"small", headers={"Content-Length": "100"})
        with self.assertRaises(agent.PageTooLargeError):
            agent.read_bounded(response, 10)


class RegistryAndCollectTests(unittest.TestCase):
    def write_registry(self, directory: Path, sources) -> Path:
        path = directory / "registry.json"
        path.write_text(json.dumps({"schema_version": 1, "sources": sources}), encoding="utf-8")
        return path

    def test_duplicate_ids_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = self.write_registry(root, [source_record(), source_record(title="duplicate")])
            with self.assertRaisesRegex(agent.RegistryError, "duplicate source id"):
                agent.load_registry(path)

    def test_cache_collection_outputs_links_geo_requests_and_incremental_state(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "out"
            cache = root / "cache"
            cache.mkdir()
            requests_path = root / "requests.tsv"
            registry = self.write_registry(root, [source_record()])
            url = "https://example.org/study"
            body = (
                b'<html><body>GSE9988 <a href="/dataset/new">new</a>'
                b'<a href="https://files.example.org/data.tsv">file</a></body></html>'
            )
            (cache / f"{agent.cache_key(url)}.body").write_bytes(body)
            (cache / f"{agent.cache_key(url)}.headers").write_text(
                "HTTP/1.1 200 OK\nContent-Type: text/html\n", encoding="utf-8"
            )
            args = collect_args(
                registry, output, page_cache=str(cache), emit_requests=str(requests_path)
            )
            first = agent.collect_once(args)
            second = agent.collect_once(args)
            self.assertNotEqual(first, second)
            self.assertTrue((first / "catalog.json").is_file())
            self.assertTrue((first / "catalog.tsv").is_file())
            self.assertTrue((first / "report.md").is_file())
            self.assertTrue((first / "download_candidates.tsv").is_file())
            self.assertTrue((first / "discovered_links.tsv").is_file())
            self.assertTrue((first / "run.json").is_file())
            catalog = json.loads((first / "catalog.json").read_text(encoding="utf-8"))
            self.assertIn("GSE9988", catalog["sources"][0]["discovered_accessions"])
            self.assertEqual(catalog["sources"][0]["availability"], "not_checked")
            self.assertEqual(catalog["pages"][0]["retrieval"], "page_cache")
            self.assertEqual(catalog["pages"][0]["sha256"], hashlib.sha256(body).hexdigest())
            with (first / "discovered_links.tsv").open(encoding="utf-8", newline="") as handle:
                links = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual({row["kind"] for row in links}, {"download", "dataset_or_catalog"})
            with requests_path.open(encoding="utf-8", newline="") as handle:
                request_rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(list(request_rows[0]), ["cache_key", "url"])
            self.assertEqual(request_rows[0]["cache_key"], agent.cache_key(url))
            latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["run_dir"], second.name)
            self.assertTrue(latest["changes"]["baseline_available"])
            self.assertEqual(latest["changes"]["changed_source_ids"], [])

    def test_missing_cache_is_recorded_without_aborting(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            registry = self.write_registry(root, [source_record()])
            cache = root / "empty-cache"
            cache.mkdir()
            run_dir = agent.collect_once(collect_args(registry, root / "out", page_cache=str(cache)))
            catalog = json.loads((run_dir / "catalog.json").read_text(encoding="utf-8"))
            self.assertEqual(catalog["pages"][0]["status"], "not_fetched")
            self.assertEqual(len(catalog["errors"]), 1)
            self.assertEqual(json.loads((run_dir / "run.json").read_text())["status"], "completed_with_errors")

    def test_inventory_never_marks_processed_h5ad_ready(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            registry = self.write_registry(root, [source_record(inventory_keys=["pancreas_xenium"])])
            inventory = root / "dataset_inventory.tsv"
            inventory.write_text(
                "dataset_key\tdataset_family\tlocal_path\tnote\n"
                "pancreas_xenium\tprocessed_h5ad\t/data/example.h5ad\tlocal only\n",
                encoding="utf-8",
            )
            run_dir = agent.collect_once(
                collect_args(registry, root / "out", inventory=str(inventory))
            )
            source = json.loads((run_dir / "catalog.json").read_text())["sources"][0]
            self.assertEqual(source["inventory_status"], "registered_unvalidated")
            self.assertEqual(source["file_validation_status"], "pending_file_validation")
            self.assertNotIn("train_ready", json.dumps(source))


class FetchTests(unittest.TestCase):
    def test_checksum_mismatch_and_short_http_body_preserve_partial_files(self):
        base = {"source_id": "sample", "url": "https://example.org/a.bin", "role": "transcripts", "filename": "a.bin", "evidence_level": "documented"}
        for extra, headers in (({"expected_sha256": "0" * 64}, {"Content-Length": "3"}), ({}, {"Content-Length": "4"})):
            with self.subTest(extra=extra, headers=headers), tempfile.TemporaryDirectory() as raw:
                with mock.patch.object(agent, "open_public_url", return_value=FakeResponse(b"abc", headers=headers)):
                    record, written = agent._download_one({**base, **extra}, Path(raw), 10, 10, 10)
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["error_type"], "IntegrityError")
                self.assertEqual(written, 3)
                self.assertFalse((Path(raw) / record["path"]).exists())
                self.assertEqual((Path(raw) / record["part_path"]).read_bytes(), b"abc")

    def test_all_failed_fetch_has_nonzero_exit_and_failed_summary(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "selected.tsv"
            manifest.write_text("source_id\turl\trole\tfilename\tevidence_level\nsample\thttps://example.org/a.bin\ttranscripts\ta.bin\tdocumented\n")
            with mock.patch.object(agent, "open_public_url", side_effect=OSError("network unavailable")):
                code = agent.main(["fetch", "--manifest", str(manifest), "--output", str(root / "out"), "--max-total-bytes", "10", "--max-file-bytes", "10"])
            self.assertEqual(code, 3)
            latest = json.loads((root / "out" / "latest.json").read_text())
            report = json.loads((root / "out" / latest["run_dir"] / "run.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed"], 1)

    def test_stream_budget_is_strict_and_partial_file_is_preserved(self):
        row = {
            "source_id": "sample", "url": "https://example.org/large.bin",
            "role": "matrix", "filename": "large.bin", "evidence_level": "candidate",
        }
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            response = FakeResponse(b"0123456789", headers={})
            with mock.patch.object(agent, "open_public_url", return_value=response):
                record, written = agent._download_one(row, run_dir, 10, 5, 100)
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_type"], "DownloadLimitError")
            self.assertEqual(written, 5)
            part = run_dir / record["part_path"]
            self.assertEqual(part.read_bytes(), b"01234")
            self.assertFalse((run_dir / record["path"]).exists())

    def test_success_hash_and_pending_validation_availability(self):
        row = {
            "source_id": "sample", "url": "https://example.org/small.bin",
            "role": "matrix", "filename": "small.bin", "evidence_level": "documented",
        }
        with tempfile.TemporaryDirectory() as raw:
            run_dir = Path(raw)
            response = FakeResponse(b"abc", headers={"Content-Length": "3"})
            with mock.patch.object(agent, "open_public_url", return_value=response):
                record, written = agent._download_one(row, run_dir, 10, 5, 5)
            self.assertEqual(written, 3)
            self.assertEqual(record["status"], "downloaded")
            self.assertEqual(record["availability"], "downloaded_pending_validation")
            self.assertEqual(record["sha256"], hashlib.sha256(b"abc").hexdigest())
            self.assertEqual((run_dir / record["path"]).read_bytes(), b"abc")
            self.assertFalse((run_dir / record["part_path"]).exists())


class ShellGatewayTests(unittest.TestCase):
    def fake_curl_env(self, root: Path):
        bin_dir = root / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "curl"
        fake.write_text('''#!/usr/bin/env bash
set -e
output=""
headers=""
head_only=0
while (($#)); do
  case "$1" in
    --head) head_only=1; shift ;;
    --output) output="$2"; shift 2 ;;
    --dump-header) headers="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if ((head_only)); then
  printf 'HTTP/1.1 200 OK\\r\\nContent-Length: 1024\\r\\n\\r\\n'
else
  [[ -z "$headers" ]] || printf 'HTTP/1.1 200 OK\\r\\nContent-Type: text/html\\r\\n\\r\\n' > "$headers"
  head -c 1024 /dev/zero > "$output"
  printf 200
fi
''')
        fake.chmod(0o755)
        return {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"]}

    def test_raw_download_budget_exhaustion_records_every_row(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env = self.fake_curl_env(root)
            manifest = root / "raw.tsv"
            manifest.write_text("source_id\turl\trole\tfilename\tevidence_level\n" + "".join(
                f"sample{i}\thttps://cf.10xgenomics.com/sample{i}.bin\ttranscripts\tsample{i}.bin\tdocumented\n" for i in range(3)
            ))
            output = root / "download"
            result = subprocess.run(["bash", str(SCRIPT.parent / "fetch_subcellular_manifest.sh"), str(manifest), str(output), "1024", "1024"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            with (output / "downloads.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["status"] for row in rows], ["downloaded_unvalidated", "budget_deferred", "budget_deferred"])
            self.assertEqual(sum(int(row["bytes"]) for row in rows), 1024)

    def test_metadata_page_limit_preserves_deferred_requests(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env = self.fake_curl_env(root)
            manifest = root / "pages.tsv"
            urls = [f"https://www.10xgenomics.com/page{i}" for i in range(3)]
            manifest.write_text("cache_key\turl\n" + "".join(f"{agent.cache_key(url)}\t{url}\n" for url in urls))
            output = root / "cache"
            result = subprocess.run(["bash", str(SCRIPT.parent / "cache_subcellular_source_pages.sh"), str(manifest), str(output), "1"], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            with (output / "download_status.tsv").open() as handle:
                rows = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual([row["status"] for row in rows], ["downloaded", "page_limit_deferred", "page_limit_deferred"])


if __name__ == "__main__":
    unittest.main()
