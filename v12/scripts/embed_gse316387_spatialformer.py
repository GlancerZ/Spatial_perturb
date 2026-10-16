#!/usr/bin/env python3
"""Frozen official SpatialFormer single-71 CLS embeddings on raw Xenium counts.

Run on a GPU compute node with SpatialFormer dependencies in the environment.
This reuses the verified single-input path from extract_lungpf_embeddings.py,
with Skin/Disease/Xenium metadata tokens for the melanoma cohort.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "resources/spatialformer/upstream"
sys.path.insert(0, str(UPSTREAM))

import anndata as ad
import numpy as np
from scipy import sparse
import torch

from spatialformer.tools.get_embeddings import GeneTokenizer, collate_fn, manual_train_fm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "resources/spatialformer/single_71.ckpt")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output_dir.exists():
        parser.error("Require positive batch size and a new output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("SpatialFormer inference requires a GPU compute node")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    data = ad.read_h5ad(args.input_h5ad)
    ids = list(map(str, data.obs_names))
    genes = list(map(str, data.var_names))
    if not ids or len(ids) != len(set(ids)) or len(genes) != len(set(genes)):
        raise ValueError("Cell IDs and genes must be present and unique")
    torch.set_num_threads(4)
    pkg = UPSTREAM / "spatialformer"
    config = json.loads((pkg / "config/_config_train_large_pair.json").read_text())
    config["embedding_path"] = str(
        pkg / "spatial_embeddings/xenium_5k_pandavid_gene_embeddings.pkl")
    model = manual_train_fm(config, use_flash_attn=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.eval().requires_grad_(False).cuda()
    tokenizer = GeneTokenizer(
        str(pkg / "tokenizer/tokenv5.json"), mode="single", tissue="Skin",
        condition="Disease", assay="Xenium",
        gene_median_path=str(pkg / "data/gene_median.pkl"))
    tokenizer.genes = np.asarray(genes)
    output = np.empty((len(ids), config["dim_model"]), dtype=np.float32)

    for start in range(0, len(ids), args.batch_size):
        stop = min(len(ids), start + args.batch_size)
        block = data.X[start:stop]
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(block).all() or (block < 0).any() or \
                not np.equal(block, np.floor(block)).all():
            raise ValueError("SpatialFormer received noninteger or invalid raw counts")
        encoded = [tokenizer.encode(row) for row in block]
        sequences = [prefix + tokens + end for tokens, prefix, end in encoded]
        length = max(map(len, sequences))
        indices = torch.zeros((len(sequences), length), dtype=torch.long)
        mask = torch.zeros_like(indices, dtype=torch.bool)
        lengths = torch.zeros((len(sequences), 2), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            indices[row, :len(sequence)] = torch.tensor(sequence)
            mask[row, :len(sequence)] = True
            lengths[row, 0] = len(sequence)
        batch = {"indices": indices, "attention_mask": mask,
                 "token_type_ids": torch.ones_like(indices),
                 "sequence_length": lengths}
        if start == 0 and len(encoded[0][0]) != 2:
            reference_batch = collate_fn(encoded)
            for key in batch:
                torch.testing.assert_close(batch[key], reference_batch[key])
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.inference_mode():
            hidden = model.embeddings(batch["indices"])
            hidden += model.token_type_embeddings(batch["token_type_ids"])
            hidden += model.spatialembeds(batch["indices"])
            vector = model.encoder(hidden, False, batch["attention_mask"])[0][-1][:, 0, :]
            if start == 0:
                official = model.get_embeddings(
                    batch, layers=[-1], pair_prediction=False,
                    co_prediction=False)[0][:, 0, :]
                torch.testing.assert_close(vector, official, rtol=1e-5, atol=1e-6)
            output[start:stop] = vector.float().cpu().numpy()
        if start % (args.batch_size * 100) == 0:
            print(f"SpatialFormer {stop}/{len(ids)}", flush=True)
    if not np.isfinite(output).all():
        raise ValueError("SpatialFormer embedding contains nonfinite values")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "embeddings.npy", output, allow_pickle=False)
    (args.output_dir / "cell_ids.txt").write_text(
        "\n".join(ids) + "\n", encoding="utf-8")
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "model_name": "SpatialFormer_single71_Xenium5K",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_source": "https://doi.org/10.6084/m9.figshare.31146238.v1",
        "official_source": "https://github.com/TerminatorJ/Spatialformer",
        "official_source_commit": "f02e7776e8a99c9abdc5fe5572a1a20369faf159",
        "input_h5ad": str(args.input_h5ad.resolve()),
        "input_source": "raw_integer_whole_cell_X",
        "tissue_token": "Skin", "condition_token": "Disease",
        "assay_token": "Xenium", "pooling": "official_final_layer_CLS",
        "panel_genes": len(genes),
        "recognized_genes": sum(gene in tokenizer.token_to_id for gene in genes),
        "cells": len(ids), "embedding_dim": output.shape[1],
        "official_first_batch_equivalence": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(ids)} SpatialFormer embeddings")


if __name__ == "__main__":
    main()
