from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

from standardize_spatial_h5ad import ARCHIVE_SUFFIXES, PRETRAIN_ROOT, is_hidden_member, strip_archive_suffix


MANIFEST_ROOT = PRETRAIN_ROOT / "manifests"


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique = sorted(dict.fromkeys(lines))
    path.write_text("\n".join(unique) + ("\n" if unique else ""), encoding="utf-8")
    print(f"[write] {path} {len(unique)}")


def build_skin_merfish() -> list[str]:
    return [f"skin_anatomy__{path.stem}.h5ad" for path in sorted((PRETRAIN_ROOT / "skin_anatomy" / "raw_merfish").glob("*.h5ad"))]


def build_deepspacedb_10x() -> list[str]:
    root = PRETRAIN_ROOT / "deepspacedb" / "raw_10x_xenium"
    return [f"deepspacedb_10x__{strip_archive_suffix(path.name)}.h5ad" for path in sorted(root.rglob("*_outs.zip"))]


def build_dbkero() -> list[str]:
    root = PRETRAIN_ROOT / "dbkero" / "raw_xenium"
    outputs = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES):
            outputs.append(f"dbkero_xenium__{strip_archive_suffix(path.name)}.h5ad")
    return outputs


def list_tar_names(archive_path: Path) -> list[str]:
    result = subprocess.run(
        ["tar", "-tf", str(archive_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"tar failed for {archive_path}")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def collect_nested_archive_targets(archive_path: Path) -> list[str]:
    try:
        member_names = [name for name in list_tar_names(archive_path) if not is_hidden_member(name)]
    except Exception:
        return [strip_archive_suffix(archive_path.name)]

    basenames = [Path(name).name for name in member_names]
    targets: list[str] = []
    if "cell_feature_matrix.h5" in basenames and "cells.csv.gz" in basenames:
        targets.append(strip_archive_suffix(archive_path.name))

    nested_archives = [
        strip_archive_suffix(Path(name).name)
        for name in member_names
        if name.endswith(ARCHIVE_SUFFIXES)
    ]
    if nested_archives:
        targets.extend(nested_archives)
    elif not targets:
        targets.append(strip_archive_suffix(archive_path.name))
    return targets


def build_deepspacedb_geo() -> list[str]:
    root = PRETRAIN_ROOT / "geo" / "raw_xenium_deepspacedb"
    outputs: list[str] = []
    zarr_pairs: dict[str, dict[str, Path]] = defaultdict(dict)

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_cell_feature_matrix.zarr.zip"):
            zarr_pairs[strip_archive_suffix(name)]["matrix"] = path
            continue
        if name.endswith("_cells.zarr.zip"):
            zarr_pairs[strip_archive_suffix(name)]["cells"] = path
            continue
        if name.endswith("_analysis.zarr.zip"):
            continue
        if name.endswith(".zip"):
            outputs.append(f"deepspacedb_geo__{strip_archive_suffix(name)}.h5ad")
            continue
        if name.endswith((".tar", ".tar.gz", ".tgz")):
            outputs.extend(f"deepspacedb_geo__{sample_id}.h5ad" for sample_id in collect_nested_archive_targets(path))

    for sample_id, pair in sorted(zarr_pairs.items()):
        if "matrix" in pair and "cells" in pair:
            outputs.append(f"deepspacedb_geo__{sample_id}.h5ad")
    return outputs


def main() -> None:
    write_lines(MANIFEST_ROOT / "standardization_targets_skin_anatomy_merfish.txt", build_skin_merfish())
    write_lines(MANIFEST_ROOT / "standardization_targets_deepspacedb_10x.txt", build_deepspacedb_10x())
    write_lines(MANIFEST_ROOT / "standardization_targets_dbkero_xenium.txt", build_dbkero())
    write_lines(MANIFEST_ROOT / "standardization_targets_deepspacedb_geo.txt", build_deepspacedb_geo())


if __name__ == "__main__":
    main()
