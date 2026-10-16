"""Make an independent, cell-level ResolVI correction of one Xenium slice.

This does not reassign molecules, cell boundaries, or nuclear/extranuclear RNA.
Run on a Slurm compute node in the isolated scviva-tools environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
import torch
import scviva
from scviva.model import ResolVI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--posterior-chunk", type=int, default=1024)
    parser.add_argument("--posterior-samples", type=int, default=30)
    parser.add_argument("--min-counts", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cells", type=int, default=0,
                        help="Only for a separate smoke run; 0 processes the full slice")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def posterior_summary(model, data, indices: np.ndarray, pyro_model, site: str,
                      statistic: str, args: argparse.Namespace) -> np.ndarray:
    aggregate = np.median if statistic == "median" else np.mean
    samples = model.sample_posterior(
        adata=data,
        indices=indices,
        model=pyro_model,
        return_sites=[site],
        summary_fun={"posterior_summary": aggregate},
        num_samples=args.posterior_samples,
        summary_frequency=1,
        batch_size=args.batch_size,
    )
    values = np.asarray(samples["posterior_summary"][site], dtype=np.float64)
    expected = (len(indices), data.n_vars)
    if values.shape != expected or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid posterior {site} {statistic}: shape={values.shape}, expected={expected}")
    return values


def validate_input(data: ad.AnnData, args: argparse.Namespace) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique")
    raw = sp.csr_matrix(data.X)
    raw.sort_indices()
    if not np.isfinite(raw.data).all() or (raw.data < 0).any():
        raise ValueError("Input counts contain negative or nonfinite values")
    if not np.all(raw.data == np.rint(raw.data)):
        raise ValueError("ResolVI input must be raw integer counts")
    if "spatial" not in data.obsm:
        raise ValueError("Input must contain global cell coordinates in obsm['spatial']")
    coordinates = np.asarray(data.obsm["spatial"], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[0] != data.n_obs or coordinates.shape[1] < 2:
        raise ValueError("Invalid spatial coordinates")
    coordinates = coordinates[:, :2].copy()
    if not np.isfinite(coordinates).all():
        raise ValueError("Spatial coordinates contain nonfinite values")
    if not {"nuclear_counts", "extranuclear_counts"}.issubset(data.layers):
        raise ValueError("Expected the original Xenium nuclear and extranuclear count layers")
    discrepancy = raw - sp.csr_matrix(data.layers["nuclear_counts"])
    discrepancy -= sp.csr_matrix(data.layers["extranuclear_counts"])
    discrepancy.eliminate_zeros()
    if discrepancy.nnz:
        raise ValueError("Original nuclear + extranuclear counts do not equal X")
    totals = np.asarray(raw.sum(axis=1)).ravel()
    eligible = np.flatnonzero(totals >= args.min_counts)
    if len(eligible) <= args.neighbors + 5:
        raise ValueError("Too few eligible cells for the requested spatial neighbors")
    return raw, coordinates, eligible


def gene_retention(raw: sp.csr_matrix, corrected: sp.csr_matrix, names) -> dict:
    result = {}
    for gene in ("CD3E", "EPCAM", "MUC1", "CEACAM6", "TRAC"):
        if gene in names:
            col = names.get_loc(gene)
            original = float(raw[:, col].sum())
            after = float(corrected[:, col].sum())
            result[gene] = {"raw": original, "corrected": after,
                            "retained_fraction": after / original if original else None}
    return result


def main(args: argparse.Namespace) -> None:
    if scviva.__version__ != "0.1.7":
        raise RuntimeError(f"Expected scviva-tools 0.1.7, got {scviva.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("This prepared run requires one visible GPU on a compute node")
    if args.max_cells < 0 or min(args.epochs, args.batch_size, args.posterior_chunk,
                                 args.posterior_samples, args.min_counts, args.neighbors) <= 0:
        raise ValueError("Run parameters must be positive; max-cells must be nonnegative")
    input_path = args.input.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")

    data = ad.read_h5ad(input_path)
    if args.max_cells:
        data = data[:args.max_cells].copy()
    raw, coordinates, eligible = validate_input(data, args)
    source_hash = sha256(input_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running", "input": str(input_path), "input_sha256": source_hash,
        "cells": data.n_obs, "genes": data.n_vars, "eligible_cells": len(eligible),
        "min_counts": args.min_counts, "neighbors": args.neighbors,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "posterior_chunk": args.posterior_chunk, "posterior_samples": args.posterior_samples,
        "seed": args.seed, "max_cells": args.max_cells,
        "scviva_version": str(scviva.__version__), "torch_version": str(torch.__version__),
        "formula": "observed * corrected_px_rate / (corrected_px_rate + residual_mean_poisson + 1.0)",
        "formula_source": "YosefLab/resolvi-reproducibility training_scripts/run_model.py",
        "true_rate_statistic": "posterior_median",
        "residual_rate_statistic": "posterior_mean",
        "scope": "cell_x_gene_total_only; no molecule reassignment; no compartment correction",
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))

    scviva.settings.seed = args.seed
    fit_data = data[eligible, :].copy()
    # AnnData 0.13 can set X to None when deleting layers one by one from a
    # sliced copy. Keep the source layers and add the raw-count view explicitly.
    fit_data.layers["counts"] = sp.csr_matrix(fit_data.X).copy()
    fit_data.obsm["X_spatial"] = coordinates[eligible]
    ResolVI.setup_anndata(
        fit_data, layer="counts",
        prepare_data_kwargs={"n_neighbors": args.neighbors, "spatial_rep": "X_spatial"},
    )
    model = ResolVI(fit_data)
    model.train(max_epochs=args.epochs, batch_size=args.batch_size)
    model.save(str(output_dir / "model"), save_anndata=False)
    print("[resolvi] model trained and saved", flush=True)

    corrected_data = raw.data.astype(np.float32, copy=True)
    for start in range(0, len(eligible), args.posterior_chunk):
        stop = min(start + args.posterior_chunk, len(eligible))
        local_indices = np.arange(start, stop, dtype=np.int64)
        true_rate = posterior_summary(model, fit_data, local_indices,
                                      model.module.model_corrected, "px_rate", "median", args)
        residual_rate = posterior_summary(model, fit_data, local_indices,
                                          model.module.model_residuals, "mean_poisson", "mean", args)
        for local, original_row in enumerate(eligible[start:stop]):
            lo, hi = raw.indptr[original_row:original_row + 2]
            gene_index = raw.indices[lo:hi]
            numerator = true_rate[local, gene_index]
            denominator = numerator + residual_rate[local, gene_index] + 1.0
            corrected_data[lo:hi] *= np.asarray(numerator / denominator, dtype=np.float32)
        print(f"[resolvi] posterior cells={stop}/{len(eligible)}", flush=True)

    corrected = sp.csr_matrix((corrected_data, raw.indices.copy(), raw.indptr.copy()),
                              shape=raw.shape)
    if not np.isfinite(corrected.data).all() or (corrected.data < 0).any():
        raise ValueError("Corrected counts contain invalid values")
    if (corrected.data > raw.data.astype(np.float32) + 1e-5).any():
        raise ValueError("Corrected counts exceed original counts")
    result = ad.AnnData(X=corrected, obs=data.obs.copy(), var=data.var.copy())
    result.layers["raw_total_counts"] = raw
    result.obsm["spatial"] = coordinates
    result.obs["resolvi_corrected"] = False
    result.obs.iloc[eligible, result.obs.columns.get_loc("resolvi_corrected")] = True
    result.uns["resolvi_correction"] = metadata | {"status": "complete"}
    result_path = output_dir / "corrected_total_counts.h5ad"
    staged_path = output_dir / "corrected_total_counts.partial.h5ad"
    result.write_h5ad(staged_path, compression="gzip")
    staged_path.rename(result_path)
    reloaded = ad.read_h5ad(result_path, backed="r")
    try:
        if reloaded.shape != data.shape or not reloaded.obs_names.equals(data.obs_names):
            raise ValueError("Written H5AD differs from the input cell ordering or shape")
        if not reloaded.var_names.equals(data.var_names) or "raw_total_counts" not in reloaded.layers:
            raise ValueError("Written H5AD lost gene ordering or the raw count layer")
        if (sp.csr_matrix(reloaded.X[: min(128, data.n_obs)]) !=
                corrected[: min(128, data.n_obs)]).nnz:
            raise ValueError("Written H5AD does not round-trip the leading rows")
    finally:
        reloaded.file.close()

    raw_sum = float(raw.sum())
    corrected_sum = float(corrected.sum())
    qc = {
        "cells": data.n_obs, "genes": data.n_vars, "corrected_cells": len(eligible),
        "unchanged_low_count_cells": data.n_obs - len(eligible),
        "raw_total": raw_sum, "corrected_total": corrected_sum,
        "total_retained_fraction": corrected_sum / raw_sum if raw_sum else None,
        "no_new_nonzero_positions_by_construction": bool(
            np.array_equal(corrected.indptr, raw.indptr)
            and np.array_equal(corrected.indices, raw.indices)),
        "per_gene_retention": gene_retention(raw, corrected, data.var_names),
        "result": str(result_path),
    }
    (output_dir / "qc.json").write_text(json.dumps(qc, indent=2))
    metadata["status"] = "complete"
    metadata["result"] = str(result_path)
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2))
    print(f"[resolvi] complete: {result_path}", flush=True)


if __name__ == "__main__":
    main(parse_args())
