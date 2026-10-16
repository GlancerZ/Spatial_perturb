from __future__ import annotations

import argparse
import hashlib
import os
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate identical pretraining files by replacing duplicates with hardlinks."
    )
    parser.add_argument(
        "--root",
        default="data/pretrain_data",
        help="Root directory to scan. Defaults to data/pretrain_data.",
    )
    parser.add_argument(
        "--report",
        default="data/pretrain_data/manifests/dedupe_report.tsv",
        help="Output TSV report path.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Replace duplicate files with hardlinks to the canonical file.",
    )
    parser.add_argument(
        "--min-size-bytes",
        type=int,
        default=1,
        help="Ignore files smaller than this size.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_canonical_key(path: Path) -> tuple[int, str]:
    # Prefer files that live under the source-native path rather than mirrored provider aliases.
    text = path.as_posix()
    penalty = 0 if "/samples/" in text else 1
    return (penalty, text)


def collect_duplicates(root: Path, min_size_bytes: int) -> list[dict]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size < min_size_bytes:
            continue
        by_size[size].append(path)

    duplicate_groups: list[dict] = []
    for size, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            by_hash[sha256_file(path)].append(path)
        for digest, group_paths in by_hash.items():
            if len(group_paths) < 2:
                continue
            group_paths = sorted(group_paths, key=stable_canonical_key)
            canonical = group_paths[0]
            duplicates = group_paths[1:]
            duplicate_groups.append(
                {
                    "size": size,
                    "sha256": digest,
                    "canonical": canonical,
                    "duplicates": duplicates,
                    "saved_bytes_if_deduped": size * len(duplicates),
                }
            )
    duplicate_groups.sort(key=lambda item: (-item["saved_bytes_if_deduped"], item["canonical"].as_posix()))
    return duplicate_groups


def write_report(report_path: Path, groups: list[dict], root: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("group_index\tsize_bytes\tsaved_bytes_if_deduped\tsha256\trole\tpath\n")
        for index, group in enumerate(groups, start=1):
            canonical_rel = group["canonical"].relative_to(root.parent)
            handle.write(
                f"{index}\t{group['size']}\t{group['saved_bytes_if_deduped']}\t{group['sha256']}\tcanonical\t{canonical_rel.as_posix()}\n"
            )
            for duplicate in group["duplicates"]:
                duplicate_rel = duplicate.relative_to(root.parent)
                handle.write(
                    f"{index}\t{group['size']}\t{group['saved_bytes_if_deduped']}\t{group['sha256']}\tduplicate\t{duplicate_rel.as_posix()}\n"
                )


def apply_hardlink_dedupe(groups: list[dict]) -> tuple[int, int]:
    files_relinked = 0
    bytes_saved = 0
    for group in groups:
        canonical = group["canonical"]
        canonical_stat = canonical.stat()
        for duplicate in group["duplicates"]:
            duplicate_stat = duplicate.stat()
            if duplicate_stat.st_ino == canonical_stat.st_ino and duplicate_stat.st_dev == canonical_stat.st_dev:
                continue
            tmp_path = duplicate.with_name(f".{duplicate.name}.dedupe_tmp")
            if tmp_path.exists():
                tmp_path.unlink()
            os.link(canonical, tmp_path)
            tmp_path.replace(duplicate)
            files_relinked += 1
            bytes_saved += group["size"]
    return files_relinked, bytes_saved


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    report_path = Path(args.report).resolve()

    groups = collect_duplicates(root, args.min_size_bytes)
    write_report(report_path, groups, root)

    group_count = len(groups)
    duplicate_file_count = sum(len(group["duplicates"]) for group in groups)
    total_savings = sum(group["saved_bytes_if_deduped"] for group in groups)

    print(f"duplicate_groups={group_count}")
    print(f"duplicate_files={duplicate_file_count}")
    print(f"potential_saved_bytes={total_savings}")
    print(f"report={report_path}")

    if args.apply:
        files_relinked, bytes_saved = apply_hardlink_dedupe(groups)
        print(f"files_relinked={files_relinked}")
        print(f"bytes_saved={bytes_saved}")


if __name__ == "__main__":
    main()
