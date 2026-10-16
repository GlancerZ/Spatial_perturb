#!/usr/bin/env python3
"""Extract scFoundation cell embeddings from raw whole-cell H5AD counts.

Run on a CUDA compute node with the official scFoundation ``model`` source tree,
PyTorch and anndata. This follows official get_embedding.py cell/ce/all/t4
inference after alignment to its fixed 19,264-gene order. Batching uses the
official gatherData and per-cell valid lengths to exclude padding from pooling.

Example:
  python scripts/embed_foundation_scfoundation.py \
    --input raw_counts.h5ad --output scfoundation_embeddings.npz \
    --source-model-dir resources/foundation_annotation/scfoundation_source/model \
    --allow-unverified-checkpoint

The current local checkpoint is a mirror whose identity has NOT been checked
against a publisher-provided digest. The flag explicitly acknowledges that
fact. Unlike the official loader, this adapter only permits PyTorch's
``weights_only=True`` checkpoint deserialization and fails closed otherwise.
For corrected_compartments.h5ad, use ``--counts-layer raw_total_counts`` for
raw integer counts, or explicitly request ``--corrected-total-x`` for the
ResolVI-corrected floating-point .X sensitivity arm. These are distinct inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (ROOT / "resources/foundation_annotation"
                      / "scfoundation_mirror_unverified/models.ckpt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_gene_order(path: Path) -> list[str]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or "gene_name" not in reader.fieldnames:
            raise ValueError("Official gene-index TSV must have gene_name column")
        order = [row["gene_name"] for row in reader]
    if len(order) != 19264 or len(set(order)) != 19264:
        raise ValueError("Expected exactly 19,264 unique official genes")
    return order


def safe_load_cell_model(checkpoint: Path, torch, convertconfig, select_model):
    """Reproduce official MMF construction without unsafe pickle fallback."""
    try:
        checkpoint_data = torch.load(
            checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint cannot be loaded with torch.load(weights_only=True); "
            "refusing unsafe pickle fallback"
        ) from exc
    if not isinstance(checkpoint_data, dict) or "cell" not in checkpoint_data:
        raise ValueError("Checkpoint has no MMF 'cell' section")
    cell_data = checkpoint_data["cell"]
    if (not isinstance(cell_data, dict) or
            not isinstance(cell_data.get("config"), dict) or
            not isinstance(cell_data.get("state_dict"), dict)):
        raise ValueError("Malformed MMF cell checkpoint")
    # Official load.py converts MMF config/state keys before select_model.
    converted = convertconfig(cell_data)
    config = converted["config"]
    if config.get("model") != "mae_autobin":
        if "qv_dim" not in config:
            config["qv_dim"] = config.get("dim_head", 64)
    config.setdefault("ppi_edge", None)
    model = select_model(config)
    model.load_state_dict(converted["model_state_dict"], strict=True)
    return model.cuda(), config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help=".npz output")
    p.add_argument("--source-model-dir", type=Path, required=True,
                   help="Official biomap-research/scFoundation/model directory")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--counts-layer", default="X", help="Raw whole-cell counts layer")
    p.add_argument("--corrected-total-x", action="store_true",
                   help="Accept finite nonnegative float .X as corrected counts; "
                        "never silently round them")
    p.add_argument("--gene-symbol-column", default="auto",
                   help="auto, index, or an adata.var symbol column")
    p.add_argument("--target-resolution", type=float, default=4.0,
                   help="Official t4 target-resolution number")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--allow-unverified-checkpoint", action="store_true")
    args = p.parse_args()
    if args.output.suffix != ".npz":
        raise ValueError("--output must end in .npz")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.corrected_total_x and args.counts_layer != "X":
        raise ValueError("--corrected-total-x requires --counts-layer X")
    manifest_path = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite output or manifest")
    if not args.allow_unverified_checkpoint:
        raise RuntimeError("Checkpoint source/hash is unverified; pass "
                           "--allow-unverified-checkpoint only if you accept that provenance")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    source = args.source_model_dir.resolve()
    load_py = source / "load.py"
    gene_index = source / "OS_scRNA_gene_index.19264.tsv"
    for required in (load_py, gene_index, source / "pretrainmodels"):
        if not required.exists():
            raise FileNotFoundError(required)
    gene_order = read_gene_order(gene_index)
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Official scFoundation loader uses CUDA; GPU is required")
    if not np.isfinite(args.target_resolution) or args.target_resolution <= 0:
        raise ValueError("Target resolution must be positive")
    # Official load.py imports pretrainmodels from its own source directory.
    sys.path.insert(0, str(source))
    from load import convertconfig, gatherData  # type: ignore
    from pretrainmodels import select_model  # type: ignore

    data = ad.read_h5ad(args.input, backed="r")
    try:
        if not data.obs_names.is_unique or data.n_obs == 0:
            raise ValueError("Cell IDs must be unique and dataset nonempty")
        ids = list(map(str, data.obs_names))
        if args.gene_symbol_column == "auto":
            symbol_source = "gene_name" if "gene_name" in data.var else "index"
        else:
            symbol_source = args.gene_symbol_column
        if symbol_source == "index":
            symbols = list(map(str, data.var_names))
        elif symbol_source in data.var:
            symbols = data.var[symbol_source].astype(str).tolist()
        else:
            raise ValueError(f"Missing gene symbol column: {symbol_source}")
        index = {gene: i for i, gene in enumerate(gene_order)}
        target = [index.get(gene) for gene in symbols]
        mapped = [(i, col) for i, col in enumerate(target) if col is not None]
        if not mapped:
            raise ValueError("No input gene symbols match official 19,264-gene order")
        matrix = data.X if args.counts_layer == "X" else data.layers[args.counts_layer]
        if matrix.shape != data.shape:
            raise ValueError("Counts layer shape differs from adata.shape")
        model, config = safe_load_cell_model(
            args.checkpoint, torch, convertconfig, select_model)
        model.eval()
        if int(config.get("gene_num", 19266)) != 19266:
            raise ValueError("Checkpoint does not describe 19,266 positions")
        output = None
        matched_expression = np.empty(data.n_obs, dtype=np.float32)
        for start in range(0, data.n_obs, args.batch_size):
            stop = min(start + args.batch_size, data.n_obs)
            raw = matrix[start:stop]
            if sp.issparse(raw):
                raw = raw.toarray()
            raw = np.asarray(raw)
            if (raw.shape != (stop - start, data.n_vars) or
                    not np.isfinite(raw).all() or (raw < 0).any()):
                raise ValueError(f"Input rows {start}:{stop} contain invalid counts")
            if not args.corrected_total_x and not np.array_equal(raw, np.rint(raw)):
                raise ValueError(f"Input rows {start}:{stop} are not integer raw counts; "
                                 "use --corrected-total-x only for corrected .X")
            aligned = np.zeros((stop - start, 19264), dtype=np.float32)
            for source_col, target_col in mapped:
                aligned[:, target_col] += raw[:, source_col]
            total = aligned.sum(axis=1, dtype=np.float64)
            if (total <= 1).any():
                first_bad = start + int(np.flatnonzero(total <= 1)[0])
                raise ValueError(f"Cell {ids[first_bad]} has <=1 matched count; "
                                 "official log10(total) special token would be nonpositive")
            matched_expression[start:stop] = total
            # get_embedding.py: pre_normalized=F, singlecell, tgthighres=t4.
            log_expression = np.log1p(aligned / total[:, None] * 1e4)
            model_input = np.column_stack((log_expression,
                                           np.full(stop - start, args.target_resolution),
                                           np.log10(total))).astype(np.float32)
            pretrain_gene_x = torch.as_tensor(model_input, device="cuda")
            gene_ids = torch.arange(19266, device="cuda").unsqueeze(0).expand(stop - start, -1)
            labels = pretrain_gene_x > 0
            valid_lengths = labels.sum(dim=1).tolist()
            x, padding = gatherData(pretrain_gene_x, labels, config["pad_token_id"])
            positions, _ = gatherData(gene_ids, labels, config["pad_token_id"])
            with torch.inference_mode():
                encoded = model.token_emb(x.unsqueeze(2).float(), output_weight=0)
                encoded = encoded + model.pos_emb(positions)
                encoded = model.encoder(encoded, padding)
                vectors = []
                for offset, length in enumerate(valid_lengths):
                    # Two appended resolution/count positions precede padding.
                    if length < 3:
                        raise ValueError(f"Cell {ids[start + offset]} has no model genes")
                    genes = encoded[offset, :length - 2, :]
                    vectors.append(torch.cat((encoded[offset, length - 1, :],
                                              encoded[offset, length - 2, :],
                                              genes.max(dim=0).values, genes.mean(dim=0))))
                vectors = torch.stack(vectors).float().cpu().numpy()
            if output is None:
                output = np.empty((data.n_obs, vectors.shape[1]), dtype=np.float32)
            output[start:stop] = vectors
        if output is None or output.shape[0] != len(ids) or not np.isfinite(output).all():
            raise ValueError("Embedding/ID order or finiteness validation failed")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, cell_ids=np.asarray(ids, dtype=str),
                            embedding=output, matched_raw_counts=matched_expression)
        mapping = [dict(input_index=i, symbol=symbols[i],
                        target_index=target[i],
                        status="matched" if target[i] is not None else "unmapped")
                   for i in range(data.n_vars)]
        manifest = {
            "model": "scFoundation", "embedding": f"cell_ce_all_t{args.target_resolution:g}",
            "source": "https://github.com/biomap-research/scFoundation/tree/main/model",
            "checkpoint_source_verified": False,
            "checkpoint_provenance": "local_mirror_unverified_against_publisher_hash",
            "checkpoint_deserialization": "torch_load_weights_only_true_no_fallback",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256_observed": sha256(args.checkpoint),
            "official_source_load_py_sha256": sha256(load_py),
            "official_gene_index_sha256": sha256(gene_index),
            "input": str(args.input.resolve()), "counts_layer": args.counts_layer,
            "input_count_semantics": ("caller_asserted_resolvi_corrected_total_float"
                                      if args.corrected_total_x else "raw_integer_whole_cell"),
            "gene_symbol_source": symbol_source, "n_cells": len(ids),
            "n_input_features": data.n_vars, "n_mapped_input_features": len(mapped),
            "n_unique_model_genes": len(set(col for _, col in mapped)),
            "target_resolution": args.target_resolution,
            "gene_mapping_status": dict(Counter(x["status"] for x in mapping)),
            "gene_mapping": mapping,
            "warning": "19,264-gene input was mostly zero-filled for a sparse spatial panel; "
                       "out-of-domain embeddings require downstream validation.",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    finally:
        data.file.close()


if __name__ == "__main__":
    main()
