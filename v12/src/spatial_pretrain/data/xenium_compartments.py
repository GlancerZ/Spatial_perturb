"""Stream decoded Xenium molecules into cell × gene compartment counts."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


REQUIRED = ["cell_id", "feature_name", "overlaps_nucleus", "qv"]


@contextmanager
def transcript_batches(archive, scratch_dir=None, chunk_rows=500_000):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        csv_names = [n for n in names if Path(n).name == "transcripts.csv.gz"]
        parquet_names = [n for n in names if Path(n).name == "transcripts.parquet"]
        if len(csv_names) > 1 or (not csv_names and len(parquet_names) != 1):
            raise ValueError(f"Cannot identify one transcript table in {archive}")
        if csv_names:
            with z.open(csv_names[0]) as handle:
                reader = pd.read_csv(handle, compression="gzip", usecols=REQUIRED,
                                     dtype={"cell_id": "string", "feature_name": "string"},
                                     chunksize=chunk_rows)
                try:
                    yield reader
                finally:
                    reader.close()
        else:
            import pyarrow.parquet as pq
            # ZIP Deflate cannot seek to Parquet footer efficiently. Extract just
            # this member to a temporary file; never extract the whole outs ZIP.
            with tempfile.TemporaryDirectory(prefix="xenium-parquet-", dir=scratch_dir) as tmp:
                path = Path(tmp) / "transcripts.parquet"
                with z.open(parquet_names[0]) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                parquet = pq.ParquetFile(path)
                if not set(REQUIRED).issubset(parquet.schema.names):
                    raise ValueError(f"Missing transcript fields: {set(REQUIRED) - set(parquet.schema.names)}")
                yield (batch.to_pandas() for batch in parquet.iter_batches(batch_size=chunk_rows, columns=REQUIRED))


def aggregate_batches(batches, cell_names, gene_names, qv_min=20.0):
    cells, genes = pd.Index(cell_names).astype(str), pd.Index(gene_names).astype(str)
    if not cells.is_unique or not genes.is_unique:
        raise ValueError("Cell and gene identifiers must be unique")
    shape = (len(cells), len(genes))
    # Binary merge levels avoid repeatedly copying the entire accumulated
    # sparse matrix for every transcript batch in very large samples.
    levels = []
    stats = dict(rows_seen=0, retained=0, nuclear=0, cytoplasmic=0,
                 unmatched_cell=0, unmatched_gene=0, below_quality=0, invalid_compartment=0)
    for batch in batches:
        stats["rows_seen"] += len(batch)
        cell_index = cells.get_indexer(batch["cell_id"].astype(str))
        gene_index = genes.get_indexer(batch["feature_name"].astype(str))
        quality = pd.to_numeric(batch["qv"], errors="coerce").to_numpy(dtype=float)
        # Explicitly map the supported encodings; never turn a nonempty string
        # such as "False" into True with astype(bool).
        raw_flag = batch["overlaps_nucleus"].astype(str).str.lower()
        flag = raw_flag.map({"1": 1, "1.0": 1, "true": 1, "0": 0, "0.0": 0, "false": 0}).to_numpy()
        valid_flag = np.isin(flag, [0, 1])
        selected = (cell_index >= 0) & (gene_index >= 0) & np.isfinite(quality) & (quality >= qv_min) & valid_flag
        stats["unmatched_cell"] += int((cell_index < 0).sum())
        stats["unmatched_gene"] += int((gene_index < 0).sum())
        stats["below_quality"] += int((~np.isfinite(quality) | (quality < qv_min)).sum())
        stats["invalid_compartment"] += int((~valid_flag).sum())
        additions = []
        for compartment in (0, 1):
            keep = selected & (flag == compartment)
            addition = sp.coo_matrix((np.ones(int(keep.sum()), dtype=np.int64),
                                      (cell_index[keep], gene_index[keep])), shape=shape).tocsr()
            additions.append(addition)
            if compartment:
                stats["nuclear"] += int(keep.sum())
            else:
                stats["cytoplasmic"] += int(keep.sum())
        stats["retained"] += int(selected.sum())
        level = 0
        while level < len(levels) and levels[level] is not None:
            additions = [a + b for a, b in zip(additions, levels[level])]
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(additions)
        else:
            levels[level] = additions
        print(f"[transcripts] rows={stats['rows_seen']} retained={stats['retained']}", flush=True)
    cyto, nuclear = [sp.csr_matrix(shape, dtype=np.int64) for _ in range(2)]
    for item in levels:
        if item is not None:
            cyto += item[0]
            nuclear += item[1]
    if int(nuclear.sum() + cyto.sum()) != stats["retained"]:
        raise RuntimeError("Compartment aggregation failed count conservation")
    return nuclear, cyto, stats


def convert_xenium_compartments(archive, reference_h5ad, output, *, max_cells=0,
                               chunk_rows=500_000, qv_min=20.0, scratch_dir=None):
    archive, reference_h5ad, output = Path(archive), Path(reference_h5ad), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reference = ad.read_h5ad(reference_h5ad, backed="r")
    try:
        n = min(max_cells, reference.n_obs) if max_cells > 0 else reference.n_obs
        ref = reference[:n].to_memory()
    finally:
        reference.file.close()
    # Retain only real gene features when the reference includes control probes.
    if "feature_type" in ref.var:
        ref = ref[:, ref.var["feature_type"].astype(str).eq("Gene Expression")].copy()
    if not ref.var_names.is_unique:
        raise ValueError("Reference gene names must be unique")
    with transcript_batches(archive, scratch_dir, chunk_rows) as batches:
        nuclear, cyto, stats = aggregate_batches(batches, ref.obs_names, ref.var_names, qv_min)
    if stats["retained"] == 0:
        raise ValueError("No retained transcripts; check gene names, cell IDs and quality fields")
    total = nuclear + cyto
    reference_counts = sp.csr_matrix(ref.X)
    difference = total - reference_counts
    difference.eliminate_zeros()
    stats["reference_mismatched_entries"] = int(difference.nnz)
    stats["reference_absolute_count_difference"] = float(abs(difference).sum())
    if difference.nnz:
        raise ValueError(f"Aggregated counts disagree with reference h5ad: {stats}; check source pairing")
    result = ad.AnnData(total, obs=ref.obs.copy(), var=ref.var.copy())
    result.obsm["spatial"] = np.asarray(ref.obsm["spatial"], dtype=np.float32)[:, :2]
    result.layers["nuclear_counts"] = nuclear
    result.layers["extranuclear_counts"] = cyto
    result.uns["compartment_qc"] = stats
    result.uns["compartment_source"] = {
        "archive": str(archive.resolve()), "reference_h5ad": str(reference_h5ad.resolve()),
        "qv_min": qv_min, "max_cells": max_cells,
        "definition": "assigned_cell_and_overlaps_nucleus; extranuclear_is_segmentation_based",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compartment-write-", dir=output.parent) as tmp:
        staged = Path(tmp) / output.name
        result.write_h5ad(staged, compression="gzip")
        staged.rename(output)
    output.with_suffix(".qc.json").write_text(json.dumps(stats, indent=2))
    print(f"[compartment-write] {output} cells={result.n_obs} genes={result.n_vars} qc={stats}", flush=True)
    return stats
