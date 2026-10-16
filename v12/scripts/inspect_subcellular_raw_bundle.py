#!/usr/bin/env python3
"""Inspect downloaded archives without extracting files; compute-node only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile


def inspect(path: Path, expected_md5: str = "") -> dict:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    result = {"path": str(path.resolve()), "bytes": path.stat().st_size,
              "md5": md5.hexdigest(), "sha256": sha.hexdigest(),
              "biological_training_ready": False}
    if expected_md5 and result["md5"] != expected_md5:
        raise ValueError(f"Published MD5 mismatch: {path}")
    result["published_md5_verified"] = bool(expected_md5)
    entries = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            result["archive_entries"] = len(infos)
            result["uncompressed_bytes"] = sum(i.file_size for i in infos)
            for info in infos:
                item = {"name": info.filename, "bytes": info.file_size}
                if info.filename.lower().endswith((".csv", ".txt", ".json")):
                    with archive.open(info) as source:
                        item["text_preview"] = source.read(4096).decode("utf-8", errors="replace")
                elif info.filename.lower().endswith(".mat") and info.file_size < 512 * 1024**2:
                    from scipy.io import whosmat
                    try:
                        with archive.open(info) as source:
                            item["mat_variables"] = whosmat(source)
                    except (ValueError, NotImplementedError, OSError) as exc:
                        item["mat_inspection_error"] = str(exc)
                entries.append(item)
    elif tarfile.is_tarfile(path):
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                item = {"name": member.name, "bytes": member.size}
                if member.isfile() and member.name.lower().endswith((".csv", ".txt", ".json")):
                    source = archive.extractfile(member)
                    if source is not None:
                        with source:
                            item["text_preview"] = source.read(2048).decode("utf-8", errors="replace")
                entries.append(item)
        result["archive_entries"] = len(entries)
        result["uncompressed_bytes"] = sum(i["bytes"] for i in entries)
    else:
        with path.open("rb") as source:
            result["text_preview"] = source.read(4096).decode("utf-8", errors="replace")
    result["entries"] = entries
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True,
                        help="TSV path, optional published_md5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    with args.manifest.open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            path = Path(row["path"])
            try:
                result = inspect(path, row.get("published_md5", ""))
                result["status"] = "inspected_not_compartment_validated"
            except Exception as exc:
                result = {"path": str(path), "status": "failed", "error": str(exc)}
            key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
            (args.output / f"{key}.json").write_text(json.dumps(result, indent=2))
            summary = {k: v for k, v in result.items() if k != "entries"}
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    return int(any(r["status"] == "failed" for r in summaries))


if __name__ == "__main__":
    raise SystemExit(main())
