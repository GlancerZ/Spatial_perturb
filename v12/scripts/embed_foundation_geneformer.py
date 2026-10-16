#!/usr/bin/env python3
"""Extract Geneformer V2-316M CLS embeddings from raw whole-cell H5AD counts.

Run on a compute node with anndata, scipy, torch, transformers and safetensors.
This is a sparse-panel adaptation of the official rank-value tokenizer. It uses
the official V2 gene dictionaries, median normalization and CLS convention, but
Xenium panels are out of the full-transcriptome pretraining distribution.

Example:
  python scripts/embed_foundation_geneformer.py \
    --input raw_counts.h5ad --output geneformer_embeddings.npz

For corrected_compartments.h5ad use ``--counts-layer raw_total_counts``;
ResolVI-corrected floating-point .X is intentionally rejected, never rounded.
The output is an NPZ with ``cell_ids`` and ``embedding`` in input row order,
plus a JSON manifest at <output>.manifest.json. Existing outputs are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "resources/foundation_annotation/geneformer/Geneformer-V2-316M"
DEFAULT_DICT = ROOT / "resources/foundation_annotation/geneformer/geneformer"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pickle(path: Path) -> dict:
    # Geneformer dictionaries are official pickle assets; use only trusted files.
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected dict in {path}")
    return value


def resolve_features(adata, name_to_id, collapse_map, token_dict, median_dict, id_column):
    if id_column == "auto":
        id_column = "ensembl_id" if "ensembl_id" in adata.var else "symbol"
    if id_column != "symbol" and id_column not in adata.var:
        raise ValueError(f"adata.var lacks requested gene ID column: {id_column}")
    symbols = (adata.var["gene_name"].astype(str).tolist()
               if "gene_name" in adata.var else list(map(str, adata.var_names)))
    raw_ids = (symbols if id_column == "symbol" else
               adata.var[id_column].astype(str).tolist())
    gene_ids = []
    statuses = []
    for value in raw_ids:
        if id_column == "symbol":
            mapped = name_to_id.get(value)
        else:
            mapped = value.split(".", 1)[0] if value.startswith("ENSG") else None
        if isinstance(mapped, (list, tuple, set)):
            if len(mapped) != 1:
                mapped = None
            else:
                mapped = next(iter(mapped))
        if not isinstance(mapped, str):
            gene_ids.append(None)
            statuses.append("unmapped")
            continue
        mapped = collapse_map.get(mapped, mapped)
        if mapped in token_dict and mapped in median_dict and float(median_dict[mapped]) > 0:
            gene_ids.append(mapped)
            statuses.append("matched")
        else:
            gene_ids.append(None)
            statuses.append("not_in_v2_vocabulary")
    unique_ids = list(dict.fromkeys(g for g in gene_ids if g is not None))
    if not unique_ids:
        raise ValueError("No input genes map to Geneformer V2 vocabulary")
    index = {gene: i for i, gene in enumerate(unique_ids)}
    feature_to_unique = [index.get(g) for g in gene_ids]
    mapping = [dict(input_index=i, symbol=symbols[i], input_gene_id=raw_ids[i],
                    resolved_ensembl_id=gene_ids[i], status=statuses[i])
               for i in range(len(symbols))]
    return unique_ids, feature_to_unique, mapping, id_column


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help=".npz output")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--dict-dir", type=Path, default=DEFAULT_DICT)
    p.add_argument("--counts-layer", default="X", help="Raw whole-cell counts layer")
    p.add_argument("--gene-id-column", default="auto",
                   help="auto, symbol, or an adata.var Ensembl ID column")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import torch
    from transformers import BertForMaskedLM

    if args.output.suffix != ".npz" or args.batch_size < 1:
        raise ValueError("--output must end in .npz; --batch-size must be positive")
    manifest_path = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite output or manifest")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    files = {
        "tokens": args.dict_dir / "token_dictionary_gc104M.pkl",
        "medians": args.dict_dir / "gene_median_dictionary_gc104M.pkl",
        "names": args.dict_dir / "gene_name_id_dict_gc104M.pkl",
        "collapse": args.dict_dir / "ensembl_mapping_dict_gc104M.pkl",
        "model": args.model_dir / "model.safetensors",
        "config": args.model_dir / "config.json",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    tokens, medians = load_pickle(files["tokens"]), load_pickle(files["medians"])
    names, collapse = load_pickle(files["names"]), load_pickle(files["collapse"])
    for special in ("<cls>", "<eos>", "<pad>"):
        if special not in tokens:
            raise ValueError(f"Geneformer V2 token dictionary lacks {special}")
    data = ad.read_h5ad(args.input, backed="r")
    try:
        if not data.obs_names.is_unique or data.n_obs == 0:
            raise ValueError("Cell IDs must be unique and dataset nonempty")
        ids = list(map(str, data.obs_names))
        unique_genes, feature_to_unique, gene_mapping, id_source = resolve_features(
            data, names, collapse, tokens, medians, args.gene_id_column)
        matrix = data.X if args.counts_layer == "X" else data.layers[args.counts_layer]
        if matrix.shape != data.shape:
            raise ValueError("Counts layer shape differs from adata.shape")
        mapped = np.array([i for i, j in enumerate(feature_to_unique) if j is not None], dtype=int)
        unique_index = np.array([feature_to_unique[i] for i in mapped], dtype=int)
        median = np.array([float(medians[g]) for g in unique_genes], dtype=np.float64)
        gene_tokens = np.array([int(tokens[g]) for g in unique_genes], dtype=np.int64)
        config = json.loads(files["config"].read_text())
        if config.get("max_position_embeddings") != 4096:
            raise ValueError("Expected Geneformer V2 maximum sequence length 4096")
        model = BertForMaskedLM.from_pretrained(str(args.model_dir), local_files_only=True)
        model.eval().to(args.device)
        output = np.empty((data.n_obs, int(config["hidden_size"])), dtype=np.float32)
        expressed = np.empty(data.n_obs, dtype=np.int32)
        for start in range(0, data.n_obs, args.batch_size):
            stop = min(start + args.batch_size, data.n_obs)
            raw = matrix[start:stop]
            if sp.issparse(raw):
                raw = raw.toarray()
            raw = np.asarray(raw)
            if not np.isfinite(raw).all() or (raw < 0).any() or not np.array_equal(raw, np.rint(raw)):
                raise ValueError(f"Input rows {start}:{stop} are not finite nonnegative raw counts")
            collapsed = np.zeros((stop - start, len(unique_genes)), dtype=np.float64)
            for source_col, target_col in zip(mapped, unique_index):
                collapsed[:, target_col] += raw[:, source_col]
            sequences = []
            for offset, row in enumerate(collapsed):
                present = np.flatnonzero(row > 0)
                expressed[start + offset] = len(present)
                if not len(present):
                    raise ValueError(f"Cell {ids[start + offset]} has no expressed mapped genes")
                # Official ranking is (count / n_counts * 10,000) / gene median.
                # The cell-wide scale is positive and cannot change rank order.
                # Match the official tokenizer's np.argsort(-gene_vector),
                # including NumPy's default behavior for exact ties.
                order = np.argsort(-(row[present] / median[present]))
                ranked = gene_tokens[present[order]][:4094]
                sequences.append(np.r_[tokens["<cls>"], ranked, tokens["<eos>"]])
            width = max(map(len, sequences))
            padded = np.full((len(sequences), width), int(tokens["<pad>"]), dtype=np.int64)
            mask = np.zeros((len(sequences), width), dtype=np.int64)
            for i, sequence in enumerate(sequences):
                padded[i, :len(sequence)] = sequence
                mask[i, :len(sequence)] = 1
            with torch.inference_mode():
                hidden = model.bert(
                    input_ids=torch.as_tensor(padded, device=args.device),
                    attention_mask=torch.as_tensor(mask, device=args.device),
                    output_hidden_states=True, return_dict=True,
                ).hidden_states[-2][:, 0, :]
            output[start:stop] = hidden.float().cpu().numpy()
        if not np.isfinite(output).all() or output.shape[0] != len(ids):
            raise ValueError("Embedding/ID order or finiteness validation failed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, cell_ids=np.asarray(ids, dtype=str),
                            embedding=output, expressed_model_genes=expressed)
        manifest = {
            "model": "Geneformer-V2-316M", "embedding": "second_to_last_layer_cls",
            "source": "https://huggingface.co/ctheodoris/Geneformer",
            "input": str(args.input.resolve()), "counts_layer": args.counts_layer,
            "gene_id_source": id_source, "n_cells": len(ids),
            "n_input_features": data.n_vars, "n_mapped_input_features": len(mapped),
            "n_unique_model_genes": len(unique_genes),
            "median_expressed_model_genes_per_cell": float(np.median(expressed)),
            "gene_mapping_status": dict(Counter(x["status"] for x in gene_mapping)),
            "gene_mapping": gene_mapping,
            "duplicate_resolved_ensembl_policy": "sum_raw_counts_before_ranking",
            "rank_tie_policy": "numpy_argsort_default; tied order depends on input mapped feature order",
            "files_sha256": {key: sha256(path) for key, path in files.items()},
            "warning": "Sparse spatial panel is outside full-transcriptome pretraining distribution.",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    finally:
        data.file.close()


if __name__ == "__main__":
    main()
