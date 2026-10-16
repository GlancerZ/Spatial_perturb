from __future__ import annotations

import csv
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "pretrain_data" / "deepspacedb" / "manifests"
TENX_ROOT = ROOT / "data" / "pretrain_data" / "10x_xenium" / "raw_human_public"
DEEPSPACE_TENX_ROOT = ROOT / "data" / "pretrain_data" / "deepspacedb" / "raw_10x_xenium"
GEO_EXISTING_GSE250346 = ROOT / "data" / "pretrain_data" / "geo" / "raw_xenium_gse250346"
GEO_DEEPSPACE_ROOT = ROOT / "data" / "pretrain_data" / "geo" / "raw_xenium_deepspacedb"
DBKERO_ROOT = ROOT / "data" / "pretrain_data" / "dbkero" / "raw_xenium"

INDEX_TSV = MANIFEST_DIR / "deepspacedb_human_xenium_inventory.tsv"
TENX_PAGES_TSV = MANIFEST_DIR / "deepspacedb_10x_pages.tsv"
TENX_ALL_URLS_TXT = MANIFEST_DIR / "deepspacedb_10x_all_file_urls.txt"
TENX_MISSING_URLS_TXT = MANIFEST_DIR / "deepspacedb_10x_missing_file_urls.txt"
TENX_MINIMAL_ALL_TSV = MANIFEST_DIR / "deepspacedb_10x_minimal_all.tsv"
TENX_MINIMAL_URLS_TSV = MANIFEST_DIR / "deepspacedb_10x_minimal_urls.tsv"
GEO_URLS_TSV = MANIFEST_DIR / "deepspacedb_geo_download_urls.tsv"
GEO_URLS_TXT = MANIFEST_DIR / "deepspacedb_geo_download_urls.txt"
GEO_MINIMAL_ALL_TSV = MANIFEST_DIR / "deepspacedb_geo_minimal_all.tsv"
GEO_MINIMAL_URLS_TSV = MANIFEST_DIR / "deepspacedb_geo_minimal_urls.tsv"
DBKERO_URLS_TSV = MANIFEST_DIR / "deepspacedb_dbkero_download_urls.tsv"
DBKERO_URLS_TXT = MANIFEST_DIR / "deepspacedb_dbkero_download_urls.txt"
DBKERO_MINIMAL_ALL_TSV = MANIFEST_DIR / "deepspacedb_dbkero_minimal_all.tsv"
DBKERO_MINIMAL_URLS_TSV = MANIFEST_DIR / "deepspacedb_dbkero_minimal_urls.tsv"


DBKERO_SOURCE_PAGES = [
    "https://kero.hgc.jp/Ad-SpatialAnalysis_2024.html",
    "https://kero.hgc.jp/Breast_Cancer_Spatial.html",
]


def fetch_inventory() -> list[dict]:
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


def parse_accession(url: str) -> str | None:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.parse_qs(parsed.query).get("acc", [None])[0]


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8", "replace")


def existing_10x_local_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lstrip("/")
    return TENX_ROOT / path


def deepspace_10x_local_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lstrip("/")
    return DEEPSPACE_TENX_ROOT / path


def existing_deepspace_10x(url: str) -> bool:
    return existing_10x_local_path(url).exists() or deepspace_10x_local_path(url).exists()


def scrape_10x_dataset_urls(page_url: str) -> list[str]:
    html = fetch_text(page_url)
    urls = sorted(
        {
            line.strip()
            for line in html.replace('"', "\n").splitlines()
            if "samples/xenium/" in line
        }
    )
    return urls


def tenx_provider_relpath(url: str) -> str:
    return urllib.parse.urlparse(url).path.lstrip("/")


def tenx_sample_key(url: str) -> str:
    relpath = tenx_provider_relpath(url)
    marker = "samples/xenium/"
    if marker in relpath:
        relpath = relpath.split(marker, 1)[1]
    parts = relpath.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2])
    if len(parts) >= 2:
        return "/".join(parts[:-1])
    return relpath


def prefer_10x_url(urls: list[str]) -> str:
    def key(url: str) -> tuple[int, str]:
        host = urllib.parse.urlparse(url).netloc
        priority = 0 if host == "cf.10xgenomics.com" else 1
        return (priority, url)

    return sorted(set(urls), key=key)[0]


def select_minimal_10x_urls(urls: list[str]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for url in urls:
        grouped[tenx_sample_key(url)].append(url)

    selected: list[str] = []
    for sample_urls in grouped.values():
        outs_urls = [url for url in sample_urls if urllib.parse.urlparse(url).path.lower().endswith("_outs.zip")]
        if outs_urls:
            selected.append(prefer_10x_url(outs_urls))
            continue
        direct_urls = []
        for url in sample_urls:
            base = Path(urllib.parse.urlparse(url).path).name.lower()
            if base in {"cell_feature_matrix.h5", "cells.csv.gz", "gene_panel.json"}:
                direct_urls.append(url)
        bases = {Path(urllib.parse.urlparse(url).path).name.lower() for url in direct_urls}
        if {"cell_feature_matrix.h5", "cells.csv.gz"}.issubset(bases):
            by_base = defaultdict(list)
            for url in direct_urls:
                by_base[Path(urllib.parse.urlparse(url).path).name.lower()].append(url)
            selected.append(prefer_10x_url(by_base["cell_feature_matrix.h5"]))
            selected.append(prefer_10x_url(by_base["cells.csv.gz"]))
            if "gene_panel.json" in by_base:
                selected.append(prefer_10x_url(by_base["gene_panel.json"]))
    return sorted(set(selected))


def fetch_geo_supplementary_urls(accession: str) -> list[str]:
    quick_url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&targ=self&form=text&view=quick"
    text = fetch_text(quick_url)
    urls = []
    for line in text.splitlines():
        if line.startswith("!Series_supplementary_file = ") or line.startswith("!Sample_supplementary_file = "):
            url = line.split(" = ", 1)[1].strip()
            if not url or url.upper() == "NONE":
                continue
            if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
                url = "https://ftp.ncbi.nlm.nih.gov/" + url.removeprefix("ftp://ftp.ncbi.nlm.nih.gov/")
            urls.append(url)
    return sorted(set(urls))


def geo_target_name(accession: str, url: str) -> str:
    filename = Path(urllib.parse.urlparse(url).path).name
    return f"{accession}/{filename}"


def geo_exists(accession: str, url: str) -> bool:
    filename = Path(urllib.parse.urlparse(url).path).name
    if accession == "GSE250346" and (GEO_EXISTING_GSE250346 / filename).exists():
        return True
    return (GEO_DEEPSPACE_ROOT / accession / filename).exists()


def select_minimal_geo_urls(urls: list[str]) -> list[str]:
    def basename(url: str) -> str:
        return Path(urllib.parse.urlparse(url).path).name

    direct_matrix = defaultdict(list)
    raw_archives = []
    output_archives = []
    fallback_archives = []

    for url in urls:
        base = basename(url)
        base_lower = base.lower()
        if base_lower in {"cell_feature_matrix.h5", "cells.csv.gz", "gene_panel.json"}:
            direct_matrix[base_lower].append(url)
            continue
        if re.search(r"_raw\.tar(?:\.gz)?$", base, re.IGNORECASE):
            raw_archives.append(url)
            continue
        if base_lower.endswith("_outs.zip") or re.search(r"output-.*\.tar\.gz$", base, re.IGNORECASE):
            output_archives.append(url)
            continue
        if base_lower.endswith((".tar.gz", ".zip", ".tar")):
            noisy = (
                "transcript" in base_lower
                or "annotation" in base_lower
                or "readme" in base_lower
                or "seurat" in base_lower
                or "coordinate" in base_lower
                or "nuclei" in base_lower
                or "metadata" in base_lower
                or "analysis_summary" in base_lower
                or "image" in base_lower
                or ".svs" in base_lower
                or ".tif" in base_lower
                or ".tiff" in base_lower
                or base_lower.endswith((".rds", ".xlsx", ".html", ".html.gz"))
            )
            if not noisy:
                fallback_archives.append(url)

    if raw_archives:
        return sorted(set(raw_archives))
    if output_archives:
        return sorted(set(output_archives))
    if fallback_archives:
        return sorted(set(fallback_archives))

    bases = set(direct_matrix)
    if {"cell_feature_matrix.h5", "cells.csv.gz"}.issubset(bases):
        selected = [
            sorted(direct_matrix["cell_feature_matrix.h5"])[0],
            sorted(direct_matrix["cells.csv.gz"])[0],
        ]
        if "gene_panel.json" in direct_matrix:
            selected.append(sorted(direct_matrix["gene_panel.json"])[0])
        return selected
    return []


def scrape_dbkero_urls(page_url: str) -> list[str]:
    text = fetch_text(page_url)
    return sorted(
        {
            match.group(0)
            for match in re.finditer(r"https://kero\.hgc\.jp/cgi-bin/download/[^\"' <]+", text)
            if "/Xenium/" in match.group(0) or "output-XETG" in match.group(0)
        }
    )


def dbkero_target_name(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    parts = parsed.path.strip("/").split("/")
    tail = parts[-1]
    page_bucket = parts[-3] if len(parts) >= 3 else "unknown"
    return f"{page_bucket}/{tail}"


def dbkero_exists(url: str) -> bool:
    return (DBKERO_ROOT / dbkero_target_name(url)).exists()


def write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


def write_plain(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    rows = fetch_inventory()
    with INDEX_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    tenx_pages = sorted({row["URL"] for row in rows if row["source"] == "10X Genomics website"})
    tenx_page_rows: list[list[str]] = []
    all_tenx_urls: list[str] = []
    missing_tenx_urls: list[str] = []
    minimal_tenx_urls: list[str] = []
    for page in tenx_pages:
        urls = scrape_10x_dataset_urls(page)
        tenx_page_rows.append([page, str(len(urls))])
        all_tenx_urls.extend(urls)
        missing_tenx_urls.extend([url for url in urls if not existing_10x_local_path(url).exists()])
        minimal_tenx_urls.extend(select_minimal_10x_urls(urls))
    write_tsv(TENX_PAGES_TSV, ["page_url", "file_count"], tenx_page_rows)
    write_plain(TENX_ALL_URLS_TXT, sorted(set(all_tenx_urls)))
    write_plain(TENX_MISSING_URLS_TXT, sorted(set(missing_tenx_urls)))
    write_tsv(
        TENX_MINIMAL_ALL_TSV,
        ["name", "url", "already_present"],
        [
            [tenx_provider_relpath(url), url, "yes" if existing_deepspace_10x(url) else "no"]
            for url in sorted(set(minimal_tenx_urls))
        ],
    )
    write_tsv(
        TENX_MINIMAL_URLS_TSV,
        ["name", "url", "already_present"],
        [
            [tenx_provider_relpath(url), url, "yes" if existing_deepspace_10x(url) else "no"]
            for url in sorted(set(minimal_tenx_urls))
            if not existing_deepspace_10x(url)
        ],
    )

    geo_accessions = sorted({parse_accession(row["URL"]) for row in rows if row["source"] == "GEO" and parse_accession(row["URL"])})
    geo_rows: list[list[str]] = []
    geo_urls_plain: list[str] = []
    geo_minimal_all_rows: list[list[str]] = []
    geo_minimal_rows: list[list[str]] = []
    for acc in geo_accessions:
        accession_urls = fetch_geo_supplementary_urls(acc)
        for url in accession_urls:
            geo_rows.append([geo_target_name(acc, url), acc, url, "yes" if geo_exists(acc, url) else "no"])
            if not geo_exists(acc, url):
                geo_urls_plain.append(url)
        for url in select_minimal_geo_urls(accession_urls):
            geo_minimal_all_rows.append([geo_target_name(acc, url), acc, url, "yes" if geo_exists(acc, url) else "no"])
            if not geo_exists(acc, url):
                geo_minimal_rows.append([geo_target_name(acc, url), acc, url, "no"])
    write_tsv(GEO_URLS_TSV, ["name", "accession", "url", "already_present"], geo_rows)
    write_plain(GEO_URLS_TXT, sorted(set(geo_urls_plain)))
    write_tsv(GEO_MINIMAL_ALL_TSV, ["name", "accession", "url", "already_present"], geo_minimal_all_rows)
    write_tsv(GEO_MINIMAL_URLS_TSV, ["name", "accession", "url", "already_present"], geo_minimal_rows)

    dbkero_rows: list[list[str]] = []
    dbkero_urls_plain: list[str] = []
    dbkero_minimal_all_rows: list[list[str]] = []
    dbkero_minimal_rows: list[list[str]] = []
    for page in DBKERO_SOURCE_PAGES:
        for url in scrape_dbkero_urls(page):
            row = [dbkero_target_name(url), page, url, "yes" if dbkero_exists(url) else "no"]
            dbkero_rows.append(row)
            dbkero_minimal_all_rows.append(row)
            if not dbkero_exists(url):
                dbkero_urls_plain.append(url)
                dbkero_minimal_rows.append([dbkero_target_name(url), page, url, "no"])
    write_tsv(DBKERO_URLS_TSV, ["name", "source_page", "url", "already_present"], dbkero_rows)
    write_plain(DBKERO_URLS_TXT, sorted(set(dbkero_urls_plain)))
    write_tsv(DBKERO_MINIMAL_ALL_TSV, ["name", "source_page", "url", "already_present"], dbkero_minimal_all_rows)
    write_tsv(DBKERO_MINIMAL_URLS_TSV, ["name", "source_page", "url", "already_present"], dbkero_minimal_rows)

    print(f"inventory_rows={len(rows)}")
    print(f"tenx_pages={len(tenx_pages)} missing_tenx_urls={len(set(missing_tenx_urls))}")
    print(f"geo_accessions={len(geo_accessions)} geo_missing_urls={len(set(geo_urls_plain))}")
    print(f"dbkero_missing_urls={len(set(dbkero_urls_plain))}")
    print(f"index={INDEX_TSV}")
    print(f"tenx_missing={TENX_MISSING_URLS_TXT}")
    print(f"tenx_minimal_all={TENX_MINIMAL_ALL_TSV}")
    print(f"tenx_minimal={TENX_MINIMAL_URLS_TSV}")
    print(f"geo_urls={GEO_URLS_TSV}")
    print(f"geo_minimal_all={GEO_MINIMAL_ALL_TSV}")
    print(f"geo_minimal={GEO_MINIMAL_URLS_TSV}")
    print(f"dbkero_urls={DBKERO_URLS_TSV}")
    print(f"dbkero_minimal_all={DBKERO_MINIMAL_ALL_TSV}")
    print(f"dbkero_minimal={DBKERO_MINIMAL_URLS_TSV}")


if __name__ == "__main__":
    main()
