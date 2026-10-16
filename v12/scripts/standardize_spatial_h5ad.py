from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    import zarr
except ImportError:  # pragma: no cover - optional dependency for zipped zarr Xenium exports
    zarr = None


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_ROOT = ROOT / "data" / "pretrain_data"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".zip", ".tar")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize downloaded spatial datasets into scGPT-spatial-like AnnData files."
    )
    parser.add_argument(
        "--output-root",
        default=str(PRETRAIN_ROOT / "processed_h5ad"),
        help="Output root for standardized h5ad files.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=[],
        choices=[
            "10x_xenium",
            "deepspacedb_10x",
            "sea_ad_xenium",
            "10x_visium_hd",
            "geo_xenium_gse250346",
            "deepspacedb_geo",
            "dbkero_xenium",
            "sea_ad_merfish",
            "skin_anatomy_merfish",
            "skin_anatomy_visium",
            "single_cell_portal_merfish",
            "vizgen_h5ad",
        ],
        help="Optional source filters.",
    )
    parser.add_argument(
        "--visium-hd-bin",
        default="square_008um",
        help="Visium HD bin directory to standardize.",
    )
    parser.add_argument(
        "--scp-segmentation",
        default="baysor_segmentation",
        choices=["baysor_segmentation", "cellpose_segmentation"],
        help="Segmentation column to aggregate SCP2510 transcript-level MERFISH files into cells.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing standardized h5ad files.",
    )
    return parser.parse_args()


def decode_strings(values) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def strip_archive_suffix(name: str) -> str:
    for suffix in [
        ".cell_feature_matrix.zarr.zip",
        ".cells.zarr.zip",
        ".analysis.zarr.zip",
        ".zarr.zip",
        ".tar.gz",
        ".tgz",
        ".zip",
        ".tar",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def is_hidden_member(name: str) -> bool:
    parts = Path(name).parts
    return any(part.startswith("._") for part in parts) or Path(name).name.startswith("._")


def find_matching_name(names: list[str], target_name: str) -> str:
    candidates = [
        name for name in names
        if not is_hidden_member(name)
        and (Path(name).name == target_name or name.endswith("/" + target_name))
    ]
    if not candidates:
        raise FileNotFoundError(target_name)
    return sorted(candidates, key=lambda item: (item.count("/"), len(item)))[0]


def extract_fileobj_to_path(src, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with dst_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)


def read_gzip_csv_from_bytes(raw: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw), compression="gzip")


def load_10x_h5(path: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    with h5py.File(path, "r") as handle:
        matrix_group = handle["matrix"]
        data = matrix_group["data"][:]
        indices = matrix_group["indices"][:]
        indptr = matrix_group["indptr"][:]
        shape = tuple(int(x) for x in matrix_group["shape"][:])
        matrix = sparse.csc_matrix((data, indices, indptr), shape=shape).transpose().tocsr()

        barcodes = decode_strings(matrix_group["barcodes"][:])
        obs = pd.DataFrame(index=pd.Index(barcodes, name="obs_name"))

        features = matrix_group["features"]
        names = decode_strings(features["name"][:]) if "name" in features else decode_strings(features["gene_names"][:])
        ids = decode_strings(features["id"][:]) if "id" in features else names
        var = pd.DataFrame(index=pd.Index(names, name="feature_name"))
        var["gene_name"] = names
        var["feature_id"] = ids
        if "feature_type" in features:
            var["feature_type"] = decode_strings(features["feature_type"][:])
        elif "feature_types" in features:
            var["feature_type"] = decode_strings(features["feature_types"][:])
        if "genome" in features:
            var["genome"] = decode_strings(features["genome"][:])

    return matrix, obs, var


def align_obs(obs: pd.DataFrame | None, obs_names: pd.Index) -> pd.DataFrame:
    if obs is None:
        return pd.DataFrame(index=obs_names.copy())
    aligned = obs.copy()
    aligned.index = aligned.index.astype(str)
    return aligned.reindex(obs_names)


def set_spatial(adata: ad.AnnData, x_col: str, y_col: str) -> None:
    adata.obsm["spatial"] = adata.obs[[x_col, y_col]].astype("float32").to_numpy()


def normalize_adata(
    adata: ad.AnnData,
    *,
    dataset_id: str,
    source: str,
    tech: str,
    sample_id: str,
    batch_id: str,
) -> ad.AnnData:
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.var_names_make_unique()
    adata.var.index.name = "feature_name"

    if "gene_name" not in adata.var.columns:
        adata.var["gene_name"] = adata.var_names.astype(str)

    if "spatial" not in adata.obsm:
        for x_col, y_col in [
            ("x_centroid", "y_centroid"),
            ("global_x", "global_y"),
            ("x", "y"),
            ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ]:
            if x_col in adata.obs.columns and y_col in adata.obs.columns:
                set_spatial(adata, x_col, y_col)
                break
        else:
            for key in adata.obsm.keys():
                if "spatial" in key.lower():
                    adata.obsm["spatial"] = np.asarray(adata.obsm[key], dtype="float32")
                    break

    adata.obs["dataset_id"] = dataset_id
    adata.obs["source"] = source
    adata.obs["tech"] = tech
    adata.obs["sample_id"] = sample_id
    adata.obs["batch_id"] = batch_id
    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    adata.uns["standard_schema_version"] = "scgpt_spatial_like_v1"
    return adata


def write_h5ad(adata: ad.AnnData, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}", flush=True)
        return
    adata.write_h5ad(output_path, compression="gzip")
    print(f"[write] {output_path}", flush=True)


def convert_xenium_from_parts(
    *,
    h5_path: Path,
    cells_df: pd.DataFrame,
    output_path: Path,
    dataset_id: str,
    source: str,
    sample_id: str,
    batch_id: str,
    overwrite: bool,
) -> None:
    matrix, obs_stub, var = load_10x_h5(h5_path)
    obs = cells_df.set_index("cell_id") if "cell_id" in cells_df.columns else cells_df.copy()
    obs = align_obs(obs, obs_stub.index)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    if "x_centroid" in adata.obs.columns and "y_centroid" in adata.obs.columns:
        set_spatial(adata, "x_centroid", "y_centroid")
    normalize_adata(
        adata,
        dataset_id=dataset_id,
        source=source,
        tech="xenium",
        sample_id=sample_id,
        batch_id=batch_id,
    )
    write_h5ad(adata, output_path, overwrite)


def convert_xenium_zip_file(
    zip_path: Path,
    output_path: Path,
    *,
    dataset_id: str,
    source: str,
    sample_id: str,
    batch_id: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}", flush=True)
        return
    print(f"[start] {output_path}", flush=True)
    with zipfile.ZipFile(zip_path) as archive, tempfile.TemporaryDirectory(prefix="xenium_zip_") as tmp_dir:
        member_names = archive.namelist()
        h5_member = find_matching_name(member_names, "cell_feature_matrix.h5")
        cells_member = find_matching_name(member_names, "cells.csv.gz")
        tmp_h5 = Path(tmp_dir) / "cell_feature_matrix.h5"
        with archive.open(h5_member) as src:
            extract_fileobj_to_path(src, tmp_h5)
        cells_df = read_gzip_csv_from_bytes(archive.read(cells_member))
        convert_xenium_from_parts(
            h5_path=tmp_h5,
            cells_df=cells_df,
            output_path=output_path,
            dataset_id=dataset_id,
            source=source,
            sample_id=sample_id,
            batch_id=batch_id,
            overwrite=True,
        )


def find_xenium_members_in_tar(archive: tarfile.TarFile) -> tuple[tarfile.TarInfo, tarfile.TarInfo] | None:
    file_members = [m for m in archive.getmembers() if m.isfile() and not is_hidden_member(m.name)]
    h5_members = [m for m in file_members if Path(m.name).name == "cell_feature_matrix.h5"]
    cells_members = [m for m in file_members if Path(m.name).name == "cells.csv.gz"]
    if not h5_members or not cells_members:
        return None
    h5_member = sorted(h5_members, key=lambda m: (m.name.count("/"), len(m.name)))[0]
    h5_parent = str(Path(h5_member.name).parent)
    matching_cells = [m for m in cells_members if str(Path(m.name).parent) == h5_parent]
    cells_member = sorted(matching_cells or cells_members, key=lambda m: (m.name.count("/"), len(m.name)))[0]
    return h5_member, cells_member


def convert_xenium_tar_file(
    tar_path: Path,
    output_path: Path,
    *,
    dataset_id: str,
    source: str,
    sample_id: str,
    batch_id: str,
    overwrite: bool,
) -> bool:
    mode = "r:gz" if tar_path.name.endswith((".tar.gz", ".tgz")) else "r"
    try:
        with tarfile.open(tar_path, mode) as archive:
            members = find_xenium_members_in_tar(archive)
            if members is None:
                return False
            h5_member, cells_member = members
            with tempfile.TemporaryDirectory(prefix="xenium_tar_") as tmp_dir:
                tmp_h5 = Path(tmp_dir) / "cell_feature_matrix.h5"
                with archive.extractfile(h5_member) as src:
                    extract_fileobj_to_path(src, tmp_h5)
                with archive.extractfile(cells_member) as src:
                    cells_df = read_gzip_csv_from_bytes(src.read())
                convert_xenium_from_parts(
                    h5_path=tmp_h5,
                    cells_df=cells_df,
                    output_path=output_path,
                    dataset_id=dataset_id,
                    source=source,
                    sample_id=sample_id,
                    batch_id=batch_id,
                    overwrite=overwrite,
                )
                return True
    except (tarfile.ReadError, EOFError, OSError, zlib.error) as exc:
        print(f"[skip] unreadable archive {tar_path}: {exc}", flush=True)
        return False


def convert_xenium_nested_tar_file(
    tar_path: Path,
    output_root: Path,
    *,
    output_dir: str,
    output_prefix: str,
    source: str,
    overwrite: bool,
) -> int:
    mode = "r:gz" if tar_path.name.endswith((".tar.gz", ".tgz")) else "r"
    converted = 0
    try:
        with tarfile.open(tar_path, mode) as outer:
            direct_members = find_xenium_members_in_tar(outer)
            if direct_members is not None:
                sample_id = strip_archive_suffix(tar_path.name)
                output_path = output_root / output_dir / f"{output_prefix}{sample_id}.h5ad"
                if convert_xenium_tar_file(
                    tar_path,
                    output_path,
                    dataset_id=sample_id,
                    source=source,
                    sample_id=sample_id,
                    batch_id=sample_id,
                    overwrite=overwrite,
                ):
                    return 1

            for member in outer.getmembers():
                if not (member.isfile() and not is_hidden_member(member.name)):
                    continue
                if not member.name.endswith(ARCHIVE_SUFFIXES):
                    continue
                sample_id = strip_archive_suffix(Path(member.name).name)
                output_path = output_root / output_dir / f"{output_prefix}{sample_id}.h5ad"
                if output_path.exists() and not overwrite:
                    print(f"[skip] {output_path}", flush=True)
                    continue
                with tempfile.TemporaryDirectory(prefix="nested_xenium_") as tmp_dir:
                    tmp_inner = Path(tmp_dir) / Path(member.name).name
                    with outer.extractfile(member) as src:
                        extract_fileobj_to_path(src, tmp_inner)
                    if tmp_inner.name.endswith(".zip"):
                        try:
                            convert_xenium_zip_file(
                                tmp_inner,
                                output_path,
                                dataset_id=sample_id,
                                source=source,
                                sample_id=sample_id,
                                batch_id=sample_id,
                                overwrite=overwrite,
                            )
                            converted += 1
                            continue
                        except (FileNotFoundError, zipfile.BadZipFile):
                            pass
                    if tmp_inner.name.endswith((".tar", ".tar.gz", ".tgz")):
                        if convert_xenium_tar_file(
                            tmp_inner,
                            output_path,
                            dataset_id=sample_id,
                            source=source,
                            sample_id=sample_id,
                            batch_id=sample_id,
                            overwrite=overwrite,
                        ):
                            converted += 1
        return converted
    except (tarfile.ReadError, EOFError, OSError, zlib.error) as exc:
        print(f"[skip] unreadable nested archive {tar_path}: {exc}", flush=True)
        return 0


def decode_cell_id_array(values: np.ndarray) -> list[str]:
    values = np.asarray(values)
    if values.ndim == 1:
        if values.dtype.kind in {"U", "S"}:
            return [str(x) for x in values.tolist()]
        return [str(int(x)) for x in values.tolist()]
    if values.ndim == 2 and values.shape[1] == 2:
        return [f"{int(left)}_{int(right)}" for left, right in values.tolist()]
    decoded = []
    for row in values:
        chars = [chr(int(x)) for x in row.tolist() if int(x) != 0]
        decoded.append("".join(chars) if chars else "_".join(str(int(x)) for x in row.tolist()))
    return decoded


def load_xenium_zarr_pair(matrix_zip: Path, cells_zip: Path) -> tuple[sparse.csr_matrix, pd.DataFrame, pd.DataFrame]:
    if zarr is None:
        raise RuntimeError("zarr is required for zipped Xenium zarr inputs")

    cells_root = zarr.open(zarr.ZipStore(str(cells_zip), mode="r"), mode="r")
    matrix_root = zarr.open(zarr.ZipStore(str(matrix_zip), mode="r"), mode="r")

    cell_summary = np.asarray(cells_root["cell_summary"][:])
    summary_cols = list(cells_root["cell_summary"].attrs["column_names"])
    obs = pd.DataFrame(cell_summary, columns=summary_cols)
    obs["cell_id"] = decode_cell_id_array(cells_root["cell_id"][:])
    obs.index = pd.Index(obs["cell_id"].astype(str), name="cell_id")

    feature_group = matrix_root["cell_features"]
    feature_names = [str(x) for x in feature_group.attrs["feature_names"]]
    feature_ids = [str(x) for x in feature_group.attrs.get("feature_ids", feature_names)]
    feature_types = [str(x) for x in feature_group.attrs.get("feature_types", ["Gene Expression"] * len(feature_names))]
    n_cells = len(obs)
    n_features = len(feature_names)

    data = np.asarray(feature_group["data"][:])
    indices = np.asarray(feature_group["indices"][:])
    indptr = np.asarray(feature_group["indptr"][:])
    if indptr.shape[0] == n_cells + 1:
        matrix = sparse.csr_matrix((data, indices, indptr), shape=(n_cells, n_features))
    elif indptr.shape[0] == n_features + 1:
        matrix = sparse.csc_matrix((data, indices, indptr), shape=(n_cells, n_features)).tocsr()
    else:
        csc_group = feature_group["csc"]
        csc_data = np.asarray(csc_group["data"][:])
        csc_indices = np.asarray(csc_group["indices"][:])
        csc_indptr = np.asarray(csc_group["indptr"][:])
        matrix = sparse.csc_matrix((csc_data, csc_indices, csc_indptr), shape=(n_cells, n_features)).tocsr()

    var = pd.DataFrame(index=pd.Index(feature_names, name="feature_name"))
    var["gene_name"] = feature_names
    var["feature_id"] = feature_ids
    var["feature_type"] = feature_types
    return matrix, obs, var


def convert_xenium_zarr_pair(
    matrix_zip: Path,
    cells_zip: Path,
    output_path: Path,
    *,
    dataset_id: str,
    source: str,
    sample_id: str,
    batch_id: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}", flush=True)
        return
    print(f"[start] {output_path}", flush=True)
    matrix, obs, var = load_xenium_zarr_pair(matrix_zip, cells_zip)
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    if "cell_centroid_x" in adata.obs.columns and "cell_centroid_y" in adata.obs.columns:
        adata.obs["x_centroid"] = adata.obs["cell_centroid_x"]
        adata.obs["y_centroid"] = adata.obs["cell_centroid_y"]
        set_spatial(adata, "x_centroid", "y_centroid")
    normalize_adata(
        adata,
        dataset_id=dataset_id,
        source=source,
        tech="xenium",
        sample_id=sample_id,
        batch_id=batch_id,
    )
    write_h5ad(adata, output_path, overwrite=True)


def convert_10x_xenium_zip(sample_dir: Path, output_root: Path, overwrite: bool) -> None:
    version_id = sample_dir.parent.name
    sample_id = sample_dir.name
    candidate_dirs = [
        sample_dir,
        PRETRAIN_ROOT / "10x_xenium" / "raw_human_public" / "10x.files" / "samples" / "xenium" / version_id / sample_id,
    ]
    zip_path = None
    for candidate_dir in candidate_dirs:
        zip_candidates = sorted(
            path for path in candidate_dir.glob("*_outs.zip")
            if not path.name.endswith("_xe_outs.zip")
        )
        if zip_candidates:
            zip_path = zip_candidates[0]
            break
    if zip_path is None:
        print(f"[skip] no standard outs zip for {sample_id}", flush=True)
        return
    output_path = output_root / "xenium" / f"10x_xenium__{sample_id}.h5ad"
    convert_xenium_zip_file(
        zip_path,
        output_path,
        dataset_id=sample_id,
        source="10x_xenium",
        sample_id=sample_id,
        batch_id=sample_id,
        overwrite=overwrite,
    )


def convert_xenium_dir(sample_dir: Path, output_root: Path, overwrite: bool, source_name: str) -> None:
    h5_path = sample_dir / "cell_feature_matrix.h5"
    cells_path = sample_dir / "cells.csv.gz"
    if not h5_path.exists() or not cells_path.exists():
        return
    sample_id = sample_dir.name
    cells_df = pd.read_csv(cells_path, compression="gzip")
    output_path = output_root / "xenium" / f"{source_name}__{sample_id}.h5ad"
    convert_xenium_from_parts(
        h5_path=h5_path,
        cells_df=cells_df,
        output_path=output_path,
        dataset_id=sample_id,
        source=source_name,
        sample_id=sample_id,
        batch_id=sample_id,
        overwrite=overwrite,
    )


def read_tissue_positions_from_tar(archive: tarfile.TarFile, bin_name: str) -> pd.DataFrame:
    member = next(
        m for m in archive.getmembers()
        if m.isfile()
        and (
            m.name.endswith(f"{bin_name}/spatial/tissue_positions.csv")
            or m.name.endswith(f"{bin_name}/spatial/tissue_positions_list.csv")
            or m.name.endswith(f"{bin_name}/spatial/tissue_positions.parquet")
        )
    )
    if member.name.endswith(".parquet"):
        with archive.extractfile(member) as src:
            raw = src.read()
        df = pd.read_parquet(io.BytesIO(raw))
    else:
        with archive.extractfile(member) as src:
            df = pd.read_csv(src)
        if "barcode" not in df.columns:
            with archive.extractfile(member) as src:
                df = pd.read_csv(
                    src,
                    header=None,
                    names=[
                        "barcode",
                        "in_tissue",
                        "array_row",
                        "array_col",
                        "pxl_row_in_fullres",
                        "pxl_col_in_fullres",
                    ],
                )
    df["barcode"] = df["barcode"].astype(str)
    return df.set_index("barcode")


def convert_visium_hd(sample_dir: Path, output_root: Path, overwrite: bool, bin_name: str) -> None:
    tar_path = next(iter(sample_dir.glob("*_binned_outputs.tar.gz")), None)
    if tar_path is None:
        return
    sample_id = sample_dir.name
    output_path = output_root / "visium_hd" / f"10x_visium_hd__{sample_id}__{bin_name}.h5ad"
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}", flush=True)
        return
    print(f"[start] {output_path}", flush=True)
    with tarfile.open(tar_path, "r:gz") as archive, tempfile.TemporaryDirectory(prefix="visium_hd_") as tmp_dir:
        h5_member = next(
            m for m in archive.getmembers()
            if m.isfile() and m.name.endswith(f"{bin_name}/filtered_feature_bc_matrix.h5")
        )
        positions_df = read_tissue_positions_from_tar(archive, bin_name)
        tmp_h5 = Path(tmp_dir) / "filtered_feature_bc_matrix.h5"
        with archive.extractfile(h5_member) as src, tmp_h5.open("wb") as dst:
            shutil.copyfileobj(src, dst)

        matrix, obs_stub, var = load_10x_h5(tmp_h5)
        obs = align_obs(positions_df, obs_stub.index)
        adata = ad.AnnData(X=matrix, obs=obs, var=var)
        if "pxl_col_in_fullres" in adata.obs.columns and "pxl_row_in_fullres" in adata.obs.columns:
            set_spatial(adata, "pxl_col_in_fullres", "pxl_row_in_fullres")
        normalize_adata(
            adata,
            dataset_id=f"{sample_id}__{bin_name}",
            source="10x_visium_hd",
            tech="visium_hd",
            sample_id=sample_id,
            batch_id=f"{sample_id}__{bin_name}",
        )
        write_h5ad(adata, output_path, overwrite=True)


def convert_geo_xenium(raw_tar_path: Path, output_root: Path, overwrite: bool) -> None:
    with tarfile.open(raw_tar_path, "r") as outer:
        for member in outer.getmembers():
            if not (member.isfile() and member.name.endswith(".tar.gz")):
                continue
            sample_id = Path(member.name).name.removesuffix(".tar.gz")
            output_path = output_root / "xenium_archive" / f"geo_xenium_gse250346__{sample_id}.h5ad"
            if output_path.exists() and not overwrite:
                print(f"[skip] {output_path}", flush=True)
                continue
            print(f"[start] {output_path}", flush=True)
            with tempfile.TemporaryDirectory(prefix="geo_xenium_") as tmp_dir:
                tmp_inner = Path(tmp_dir) / Path(member.name).name
                with outer.extractfile(member) as src, tmp_inner.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                with tarfile.open(tmp_inner, "r:gz") as inner:
                    h5_member = next(m for m in inner.getmembers() if m.isfile() and m.name.endswith("outs/cell_feature_matrix.h5"))
                    cells_member = next(m for m in inner.getmembers() if m.isfile() and m.name.endswith("outs/cells.csv.gz"))
                    tmp_h5 = Path(tmp_dir) / "cell_feature_matrix.h5"
                    with inner.extractfile(h5_member) as src, tmp_h5.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                    with inner.extractfile(cells_member) as src:
                        cells_df = pd.read_csv(src, compression="gzip")
                convert_xenium_from_parts(
                    h5_path=tmp_h5,
                    cells_df=cells_df,
                    output_path=output_path,
                    dataset_id=sample_id,
                    source="geo_xenium_gse250346",
                    sample_id=sample_id,
                    batch_id=sample_id,
                    overwrite=True,
                )


def convert_deepspacedb_10x_zip(zip_path: Path, output_root: Path, overwrite: bool) -> None:
    sample_id = strip_archive_suffix(zip_path.name)
    output_path = output_root / "xenium" / f"deepspacedb_10x__{sample_id}.h5ad"
    convert_xenium_zip_file(
        zip_path,
        output_path,
        dataset_id=sample_id,
        source="deepspacedb_10x",
        sample_id=sample_id,
        batch_id=sample_id,
        overwrite=overwrite,
    )


def convert_deepspacedb_geo_archives(root: Path, output_root: Path, overwrite: bool) -> None:
    zarr_pairs: dict[str, dict[str, Path]] = defaultdict(dict)
    archive_files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_cell_feature_matrix.zarr.zip"):
            zarr_pairs[strip_archive_suffix(name)]["matrix"] = path
        elif name.endswith("_cells.zarr.zip"):
            zarr_pairs[strip_archive_suffix(name)]["cells"] = path
        elif name.endswith("_analysis.zarr.zip"):
            continue
        elif name.endswith(ARCHIVE_SUFFIXES):
            archive_files.append(path)

    for archive_path in archive_files:
        converted = convert_xenium_nested_tar_file(
            archive_path,
            output_root,
            output_dir="xenium_archive",
            output_prefix="deepspacedb_geo__",
            source="deepspacedb_geo",
            overwrite=overwrite,
        )
        if converted:
            continue
        if archive_path.name.endswith(".zip"):
            sample_id = strip_archive_suffix(archive_path.name)
            output_path = output_root / "xenium_archive" / f"deepspacedb_geo__{sample_id}.h5ad"
            try:
                convert_xenium_zip_file(
                    archive_path,
                    output_path,
                    dataset_id=sample_id,
                    source="deepspacedb_geo",
                    sample_id=sample_id,
                    batch_id=sample_id,
                    overwrite=overwrite,
                )
            except (FileNotFoundError, zipfile.BadZipFile):
                pass

    for sample_id, pair in sorted(zarr_pairs.items()):
        if "matrix" not in pair or "cells" not in pair:
            continue
        output_path = output_root / "xenium_archive" / f"deepspacedb_geo__{sample_id}.h5ad"
        convert_xenium_zarr_pair(
            pair["matrix"],
            pair["cells"],
            output_path,
            dataset_id=sample_id,
            source="deepspacedb_geo",
            sample_id=sample_id,
            batch_id=sample_id,
            overwrite=overwrite,
        )


def convert_dbkero_archives(root: Path, output_root: Path, overwrite: bool) -> None:
    for archive_path in sorted(root.rglob("*")):
        if not (archive_path.is_file() and archive_path.name.endswith(ARCHIVE_SUFFIXES)):
            continue
        sample_id = strip_archive_suffix(archive_path.name)
        output_path = output_root / "xenium_archive" / f"dbkero_xenium__{sample_id}.h5ad"
        if archive_path.name.endswith(".zip"):
            try:
                convert_xenium_zip_file(
                    archive_path,
                    output_path,
                    dataset_id=sample_id,
                    source="dbkero_xenium",
                    sample_id=sample_id,
                    batch_id=sample_id,
                    overwrite=overwrite,
                )
                continue
            except (FileNotFoundError, zipfile.BadZipFile):
                pass
        convert_xenium_tar_file(
            archive_path,
            output_path,
            dataset_id=sample_id,
            source="dbkero_xenium",
            sample_id=sample_id,
            batch_id=sample_id,
            overwrite=overwrite,
        )


def convert_h5ad_passthrough(
    input_path: Path,
    output_path: Path,
    *,
    dataset_id: str,
    source: str,
    tech: str,
    sample_id: str,
    batch_id: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}", flush=True)
        return
    print(f"[start] {output_path}", flush=True)
    adata = ad.read_h5ad(input_path)
    normalize_adata(
        adata,
        dataset_id=dataset_id,
        source=source,
        tech=tech,
        sample_id=sample_id,
        batch_id=batch_id,
    )
    write_h5ad(adata, output_path, overwrite=True)


def convert_scp_merfish(csv_path: Path, output_root: Path, overwrite: bool, seg_key: str) -> None:
    sample_id = csv_path.stem
    output_path = output_root / "merfish" / f"single_cell_portal__{sample_id}__{seg_key}.h5ad"
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path}", flush=True)
        return
    print(f"[start] {output_path}", flush=True)
    df = pd.read_csv(csv_path, usecols=["global_x", "global_y", "gene", seg_key])
    df = df[df[seg_key].notna()].copy()
    df = df[df[seg_key] > 0].copy()
    df["cell_id"] = df[seg_key].astype(int).astype(str)
    cell_codes, cell_ids = pd.factorize(df["cell_id"], sort=True)
    gene_codes, gene_names = pd.factorize(df["gene"], sort=True)
    matrix = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.int32), (cell_codes, gene_codes)),
        shape=(len(cell_ids), len(gene_names)),
    )
    obs = df.groupby("cell_id", sort=True)[["global_x", "global_y"]].mean()
    obs = obs.reindex(cell_ids)
    obs.index = pd.Index(cell_ids, name="cell_id")
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=pd.Index(gene_names.astype(str), name="gene_name")))
    adata.var["gene_name"] = adata.var.index.astype(str)
    set_spatial(adata, "global_x", "global_y")
    normalize_adata(
        adata,
        dataset_id=f"{sample_id}__{seg_key}",
        source="single_cell_portal_scp2510",
        tech="merfish",
        sample_id=sample_id,
        batch_id=f"{sample_id}__{seg_key}",
    )
    write_h5ad(adata, output_path, overwrite=True)


def run_all(output_root: Path, only: set[str], overwrite: bool, visium_hd_bin: str, scp_segmentation: str) -> None:
    if not only or "10x_xenium" in only:
        root = PRETRAIN_ROOT / "10x_xenium" / "raw_human_public" / "samples" / "xenium"
        for sample_dir in sorted(root.glob("*/*")):
            if sample_dir.is_dir():
                convert_10x_xenium_zip(sample_dir, output_root, overwrite)

    if not only or "deepspacedb_10x" in only:
        root = PRETRAIN_ROOT / "deepspacedb" / "raw_10x_xenium"
        for zip_path in sorted(root.rglob("*_outs.zip")):
            if zip_path.is_file():
                convert_deepspacedb_10x_zip(zip_path, output_root, overwrite)

    if not only or "sea_ad_xenium" in only:
        for sample_dir in sorted((PRETRAIN_ROOT / "sea_ad" / "raw_spatial").glob("*/*/*/xenium_output_and_alignment/output-*")):
            if sample_dir.is_dir():
                convert_xenium_dir(sample_dir, output_root, overwrite, "sea_ad_xenium")

    if not only or "10x_visium_hd" in only:
        root = PRETRAIN_ROOT / "10x_visium_hd" / "raw_public" / "samples" / "spatial-exp"
        for sample_dir in sorted(root.glob("*/*")):
            if sample_dir.is_dir():
                convert_visium_hd(sample_dir, output_root, overwrite, visium_hd_bin)

    if not only or "geo_xenium_gse250346" in only:
        raw_tar = PRETRAIN_ROOT / "geo" / "raw_xenium_gse250346" / "GSE250346_RAW.tar"
        if raw_tar.exists():
            convert_geo_xenium(raw_tar, output_root, overwrite)

    if not only or "deepspacedb_geo" in only:
        root = PRETRAIN_ROOT / "geo" / "raw_xenium_deepspacedb"
        if root.exists():
            convert_deepspacedb_geo_archives(root, output_root, overwrite)

    if not only or "dbkero_xenium" in only:
        root = PRETRAIN_ROOT / "dbkero" / "raw_xenium"
        if root.exists():
            convert_dbkero_archives(root, output_root, overwrite)

    if not only or "sea_ad_merfish" in only:
        for file_path in sorted((PRETRAIN_ROOT / "sea_ad" / "raw_merfish").glob("*.h5ad")):
            convert_h5ad_passthrough(
                file_path,
                output_root / "merfish" / f"sea_ad__{file_path.stem}.h5ad",
                dataset_id=file_path.stem,
                source="sea_ad_merfish",
                tech="merfish",
                sample_id=file_path.stem,
                batch_id=file_path.stem,
                overwrite=overwrite,
            )

    if not only or "skin_anatomy_merfish" in only:
        for file_path in sorted((PRETRAIN_ROOT / "skin_anatomy" / "raw_merfish").glob("*.h5ad")):
            convert_h5ad_passthrough(
                file_path,
                output_root / "merfish" / f"skin_anatomy__{file_path.stem}.h5ad",
                dataset_id=file_path.stem,
                source="skin_anatomy_merfish",
                tech="merfish",
                sample_id=file_path.stem,
                batch_id=file_path.stem,
                overwrite=overwrite,
            )

    if not only or "skin_anatomy_visium" in only:
        for file_path in sorted((PRETRAIN_ROOT / "skin_anatomy" / "raw_visium").glob("*.h5ad")):
            convert_h5ad_passthrough(
                file_path,
                output_root / "visium" / f"skin_anatomy__{file_path.stem}.h5ad",
                dataset_id=file_path.stem,
                source="skin_anatomy_visium",
                tech="visium",
                sample_id=file_path.stem,
                batch_id=file_path.stem,
                overwrite=overwrite,
            )

    if not only or "single_cell_portal_merfish" in only:
        for file_path in sorted((PRETRAIN_ROOT / "single_cell_portal" / "raw_merfish_scp2510").glob("tx_merfish_*.csv")):
            convert_scp_merfish(file_path, output_root, overwrite, scp_segmentation)

    if not only or "vizgen_h5ad" in only:
        root = PRETRAIN_ROOT / "vizgen" / "h5ad"
        for file_path in sorted(root.glob("*.h5ad")):
            convert_h5ad_passthrough(
                file_path,
                output_root / "merfish_processed" / f"vizgen__{file_path.stem}.h5ad",
                dataset_id=file_path.stem,
                source="vizgen_h5ad",
                tech="merfish",
                sample_id=file_path.stem,
                batch_id=file_path.stem,
                overwrite=overwrite,
            )


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    run_all(
        output_root=output_root,
        only=set(args.only),
        overwrite=args.overwrite,
        visium_hd_bin=args.visium_hd_bin,
        scp_segmentation=args.scp_segmentation,
    )


if __name__ == "__main__":
    main()
