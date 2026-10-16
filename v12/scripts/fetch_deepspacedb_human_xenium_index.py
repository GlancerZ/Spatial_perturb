from __future__ import annotations

import csv
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "pretrain_data" / "deepspacedb" / "manifests"
INVENTORY_TSV = MANIFEST_DIR / "deepspacedb_human_xenium_inventory.tsv"
SOURCE_SUMMARY_TSV = MANIFEST_DIR / "deepspacedb_human_xenium_source_summary.tsv"


def fetch_rows() -> list[dict]:
    base = "https://genomics.virus.kyoto-u.ac.jp/deepspacedb/get_data"
    params = {
        "search": "",
        "filters[]": ["Platform:Xenium", "Organism:Human"],
        "start": 0,
        "length": 2000,
    }
    url = base + "?" + urllib.parse.urlencode(params, doseq=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = json.load(response)
    return payload["data"]


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rows = fetch_rows()

    fieldnames = [
        "ID",
        "source",
        "sample_id",
        "series_id",
        "URL",
        "platform",
        "organism",
        "Organ.organ",
        "Organ.organ_detailed",
        "Condition.condition",
        "Condition.condition_detailed",
        "description",
        "date_published",
        "PMID",
        "has_annotation",
        "quality",
    ]

    with INVENTORY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    counts = Counter(row["source"] for row in rows)
    unique_urls = {
        source: sorted({(row.get("series_id") or "", row["URL"]) for row in rows if row["source"] == source})
        for source in counts
    }

    with SOURCE_SUMMARY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source", "sample_rows", "unique_links"])
        for source, count in sorted(counts.items()):
            writer.writerow([source, count, len(unique_urls[source])])

    print(f"rows={len(rows)}")
    for source, count in sorted(counts.items()):
        print(f"{source}={count} unique_links={len(unique_urls[source])}")
    print(f"inventory={INVENTORY_TSV}")
    print(f"summary={SOURCE_SUMMARY_TSV}")


if __name__ == "__main__":
    main()
