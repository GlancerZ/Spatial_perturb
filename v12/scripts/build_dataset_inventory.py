from __future__ import annotations

import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = ROOT / "data" / "pretrain_data"
MANIFEST_ROOT = PRETRAIN_ROOT / "manifests"
INVENTORY_TSV = MANIFEST_ROOT / "dataset_inventory.tsv"
OVERLAP_TSV = MANIFEST_ROOT / "dataset_overlap_report.tsv"


@dataclass
class InventoryRow:
    dataset_key: str
    dataset_family: str
    tech: str
    source_root: str
    local_path: str
    note: str = ""


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def family_from_stem(stem: str) -> str:
    family = stem
    family = re.sub(r"_vizgen$", "", family)
    family = re.sub(r"_10x_xenium$", "", family)
    family = re.sub(r"_slice\d+$", "", family)
    family = re.sub(r"_(racostain|rocostain)$", "", family)
    return slugify(family)


def inventory_10x_xenium() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    root = PRETRAIN_ROOT / "10x_xenium" / "raw_human_public" / "samples" / "xenium"
    if not root.exists():
        return rows
    for version_dir in sorted(root.iterdir()):
        if not version_dir.is_dir():
            continue
        for sample_dir in sorted(version_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            rows.append(
                InventoryRow(
                    dataset_key=f"xenium:{sample_dir.name}",
                    dataset_family=family_from_stem(sample_dir.name),
                    tech="xenium",
                    source_root="10x_xenium/raw_human_public",
                    local_path=str(sample_dir.relative_to(ROOT)),
                    note=version_dir.name,
                )
            )
    return rows


def inventory_10x_visium_hd() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    root = PRETRAIN_ROOT / "10x_visium_hd" / "raw_public" / "samples" / "spatial-exp"
    if not root.exists():
        return rows
    for version_dir in sorted(root.iterdir()):
        if not version_dir.is_dir():
            continue
        for sample_dir in sorted(version_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            rows.append(
                InventoryRow(
                    dataset_key=f"visium_hd:{sample_dir.name}",
                    dataset_family=family_from_stem(sample_dir.name),
                    tech="visium_hd",
                    source_root="10x_visium_hd/raw_public",
                    local_path=str(sample_dir.relative_to(ROOT)),
                    note=version_dir.name,
                )
            )
    return rows


def inventory_seaad_xenium() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    root = PRETRAIN_ROOT / "sea_ad" / "raw_spatial"
    if not root.exists():
        return rows
    for path in sorted(root.glob("*/*/*/xenium_output_and_alignment/output-*")):
        if not path.is_dir():
            continue
        rows.append(
            InventoryRow(
                dataset_key=f"seaad_xenium:{path.name}",
                dataset_family=family_from_stem(path.name),
                tech="xenium",
                source_root="sea_ad/raw_spatial",
                local_path=str(path.relative_to(ROOT)),
                note="minimal",
            )
        )
    return rows


def inventory_seaad_merfish() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    root = PRETRAIN_ROOT / "sea_ad" / "raw_merfish"
    if not root.exists():
        return rows
    for file_path in sorted(root.iterdir()):
        if not file_path.is_file():
            continue
        rows.append(
            InventoryRow(
                dataset_key=f"merfish:{file_path.stem}",
                dataset_family=family_from_stem(file_path.stem),
                tech="merfish",
                source_root="sea_ad/raw_merfish",
                local_path=str(file_path.relative_to(ROOT)),
            )
        )
    return rows


def inventory_scp2510_merfish() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    root = PRETRAIN_ROOT / "single_cell_portal" / "raw_merfish_scp2510"
    if not root.exists():
        return rows
    for file_path in sorted(root.iterdir()):
        if not file_path.is_file():
            continue
        rows.append(
            InventoryRow(
                dataset_key=f"merfish:{file_path.stem}",
                dataset_family=family_from_stem(file_path.stem),
                tech="merfish",
                source_root="single_cell_portal/raw_merfish_scp2510",
                local_path=str(file_path.relative_to(ROOT)),
                note="scp2510",
            )
        )
    return rows


def inventory_geo_xenium() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    root = PRETRAIN_ROOT / "geo" / "raw_xenium_gse250346"
    if not root.exists():
        return rows
    raw_tar = root / "GSE250346_RAW.tar"
    if raw_tar.exists():
        rows.append(
            InventoryRow(
                dataset_key="geo_xenium:gse250346_raw",
                dataset_family="gse250346_raw",
                tech="xenium_archive",
                source_root="geo/raw_xenium_gse250346",
                local_path=str(raw_tar.relative_to(ROOT)),
                note="aggregate_archive",
            )
        )
    return rows


def inventory_vizgen_processed() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    root = PRETRAIN_ROOT / "vizgen" / "h5ad"
    if not root.exists():
        return rows
    for file_path in sorted(root.glob("*.h5ad")):
        rows.append(
            InventoryRow(
                dataset_key=f"vizgen_h5ad:{file_path.stem}",
                dataset_family=family_from_stem(file_path.stem),
                tech="merfish_processed",
                source_root="vizgen/h5ad",
                local_path=str(file_path.relative_to(ROOT)),
            )
        )
    return rows


def build_rows() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    rows.extend(inventory_10x_xenium())
    rows.extend(inventory_10x_visium_hd())
    rows.extend(inventory_seaad_xenium())
    rows.extend(inventory_seaad_merfish())
    rows.extend(inventory_scp2510_merfish())
    rows.extend(inventory_geo_xenium())
    rows.extend(inventory_vizgen_processed())
    return rows


def write_inventory(rows: list[InventoryRow]) -> None:
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    with INVENTORY_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["dataset_key", "dataset_family", "tech", "source_root", "local_path", "note"])
        for row in rows:
            writer.writerow([row.dataset_key, row.dataset_family, row.tech, row.source_root, row.local_path, row.note])


def write_overlap(rows: list[InventoryRow]) -> None:
    by_key: dict[str, list[InventoryRow]] = defaultdict(list)
    by_family: dict[str, list[InventoryRow]] = defaultdict(list)
    for row in rows:
        by_key[row.dataset_key].append(row)
        by_family[row.dataset_family].append(row)

    with OVERLAP_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["overlap_type", "key", "count", "techs", "source_roots", "paths"])
        for dataset_key, group in sorted(by_key.items()):
            if len(group) < 2:
                continue
            writer.writerow(
                [
                    "dataset_key",
                    dataset_key,
                    len(group),
                    ",".join(sorted({item.tech for item in group})),
                    ",".join(sorted({item.source_root for item in group})),
                    " | ".join(item.local_path for item in group),
                ]
            )
        for dataset_family, group in sorted(by_family.items()):
            if len(group) < 2:
                continue
            writer.writerow(
                [
                    "dataset_family",
                    dataset_family,
                    len(group),
                    ",".join(sorted({item.tech for item in group})),
                    ",".join(sorted({item.source_root for item in group})),
                    " | ".join(item.local_path for item in group),
                ]
            )


def main() -> None:
    rows = build_rows()
    write_inventory(rows)
    write_overlap(rows)

    by_source: dict[str, int] = defaultdict(int)
    for row in rows:
        by_source[row.source_root] += 1

    print(f"inventory_rows={len(rows)}")
    for source_root, count in sorted(by_source.items()):
        print(f"{source_root}={count}")
    overlap_lines = sum(1 for _ in OVERLAP_TSV.open("r", encoding="utf-8")) - 1
    print(f"overlap_rows={max(0, overlap_lines)}")
    print(f"inventory={INVENTORY_TSV}")
    print(f"overlap={OVERLAP_TSV}")


if __name__ == "__main__":
    main()
