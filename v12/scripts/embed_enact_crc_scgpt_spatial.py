#!/usr/bin/env python3
"""Freeze official scGPT-spatial V1 CLS on V12's exact CRC center cells.

The official V1 encoder embeds one whole cell at a time. Its original 51-bin
collator, 1200-token limit, gene statistics, and encoder weights are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compartments", type=Path, required=True)
    parser.add_argument("--centers-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--official-asset-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "resources/scgpt_official")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run inside a verified Slurm compute-node step")
    if args.batch_size < 1 or args.seed < 0:
        raise ValueError("Invalid batch size or seed")

    import anndata as ad
    import numpy as np
    import torch
    from scipy import sparse
    import evaluate_official_scgpt as official
    from infer_visium_hd_v12 import checked_counts

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("Official scGPT-spatial V1 inference requires a GPU")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    centers = ad.read_h5ad(args.centers_h5ad, backed="r")
    center_ids = np.asarray(centers.obs_names, dtype=str)
    centers.file.close()
    if len(center_ids) == 0 or len(np.unique(center_ids)) != len(center_ids):
        raise ValueError("V12 center IDs must be nonempty and unique")
    adata = ad.read_h5ad(args.compartments)
    rows = adata.obs_names.get_indexer(center_ids)
    if (rows < 0).any():
        raise ValueError("V12 centers are missing from the same raw-count input")
    official.ASSET = args.official_asset_dir.resolve()
    model, vocab, cfg, stats, collator = official.load_official()
    symbols = np.asarray(adata.var.get("gene_name", adata.var_names), dtype=str)
    columns = np.asarray([i for i, gene in enumerate(symbols) if gene in vocab
                          and not gene.startswith("<")], dtype=np.int64)
    if len(columns) < 50:
        raise ValueError("Too few official vocabulary genes")
    token_ids = np.asarray([vocab[symbols[i]] for i in columns], dtype=np.int64)
    unique_ids, inverse = np.unique(token_ids, return_inverse=True)
    projection = sparse.csr_matrix((np.ones(len(columns), dtype=np.int64),
                                    (columns, inverse)),
                                   shape=(adata.n_vars, len(unique_ids)))
    raw = checked_counts(adata.X, "whole-cell X")
    mapped = (raw @ projection).tocsr()
    mapped.eliminate_zeros()
    if not mapped.nnz:
        raise ValueError("No observed counts map to the official vocabulary")
    slide_mean = float(mapped.data.mean())
    gene_sum = np.asarray(mapped.sum(axis=0)).ravel().astype(np.float64)
    gene_nnz = mapped.getnnz(axis=0)
    means = np.empty(len(unique_ids), dtype=np.float32)
    fallback = 0
    for i, token in enumerate(unique_ids):
        if int(token) in stats.index:
            means[i] = float(stats.loc[int(token), "mean"])
        elif gene_nnz[i]:
            means[i] = gene_sum[i] / gene_nnz[i] / slide_mean
            fallback += 1
        else:
            means[i] = 1.
            fallback += 1
    if not np.isfinite(means).all() or (means <= 0).any():
        raise ValueError("Official or fallback gene means contain nonpositive values")

    embedded = []
    pad_id = vocab[cfg["pad_token"]]
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch_rows = rows[start:start + args.batch_size]
            items = []
            for item_id, row_id in enumerate(batch_rows):
                row = mapped.getrow(int(row_id))
                expressed = row.indices
                values = row.data.astype(np.float32) / slide_mean / means[expressed]
                items.append({
                    "id": item_id,
                    "genes": torch.as_tensor(np.r_[vocab["<cls>"], unique_ids[expressed]], dtype=torch.long),
                    "expressions": torch.as_tensor(np.r_[cfg["pad_value"], values], dtype=torch.float32),
                })
            batch = collator(items)
            genes = batch["gene"].cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                hidden = model._encode(genes, batch["expr"].cuda(), genes.eq(pad_id))
                value = hidden[:, 0].float()
                value = value / value.norm(dim=1, keepdim=True).clamp_min(1e-12)
            embedded.append(value.cpu().numpy())
            if start % (args.batch_size * 20) == 0:
                print(f"[scGPT-spatial] {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    x = np.concatenate(embedded)
    if x.shape[0] != len(center_ids) or not np.isfinite(x).all():
        raise ValueError("Invalid official embedding output")
    np.savez_compressed(output / "embeddings.npz", cell_ids=center_ids,
                        embeddings=x.astype(np.float32))
    audit = {
        "status": "OFFICIAL_V1_FROZEN_EMBEDDINGS_COMPLETE",
        "model": "scGPT-spatial V1", "readout": "L2-normalized CLS",
        "inference_context": "one whole-cell count vector; no test-slide neighbors",
        "scgpt_spatial_zip_md5": "470d40fb77535afc2b4e7cffc0d665ef",
        "center_ids_sha256": hashlib.sha256("\n".join(center_ids).encode()).hexdigest(),
        "cells": len(center_ids), "embedding_dim": x.shape[1],
        "mapped_feature_columns": len(columns), "mapped_unique_tokens": len(unique_ids),
        "duplicate_token_columns_summed": len(columns) - len(unique_ids),
        "genes_with_fallback_means": fallback, "slide_nonzero_mean": slide_mean,
        "max_tokens": 1200, "bins": 51, "seed": args.seed,
        "source_h5ad": str(args.compartments.resolve()),
        "official_asset_dir": str(args.official_asset_dir.resolve()),
        "center_h5ad": str(args.centers_h5ad.resolve()),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": host,
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
