"""Extract STATE SE-600M cell embeddings through the official `state emb transform` CLI.

Only the State Embedding (SE) command is called; this script never invokes
`state tx` or a perturbation-response checkpoint.  Run on a GPU compute node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]


def validate_names(values, kind: str) -> list[str]:
    names = [str(value) for value in values]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"{kind} must be nonempty and unique")
    if any(not name.strip() or "\n" in name or "\r" in name for name in names):
        raise ValueError(f"{kind} contains an empty or multiline value")
    return names


def dense_rows(matrix, start: int, stop: int) -> np.ndarray:
    block = matrix[start:stop]
    return block.toarray() if sparse.issparse(block) else np.asarray(block)


def validate_counts(block: np.ndarray, source: str) -> None:
    if not np.issubdtype(block.dtype, np.number) or not np.isfinite(block).all():
        raise ValueError(f"{source} contains nonnumeric or nonfinite counts")
    if (block < 0).any() or not np.equal(block, np.floor(block)).all():
        raise ValueError(f"{source} must contain nonnegative raw integer counts")


def validate_corrected(block: np.ndarray) -> None:
    if not np.issubdtype(block.dtype, np.number) or not np.isfinite(block).all():
        raise ValueError("Corrected X contains nonnumeric or nonfinite expression")
    if (block < 0).any():
        raise ValueError("Corrected X contains negative expression")


def model_genes(model_folder: Path, specified_list: Path | None) -> tuple[set[str] | None, str]:
    if specified_list is not None:
        if not specified_list.is_file():
            raise FileNotFoundError(specified_list)
        names = [line.strip().split("\t", 1)[0]
                 for line in specified_list.read_text().splitlines() if line.strip()]
        if len(names) != len(set(names)) or not names:
            raise ValueError("Model gene list is empty or duplicated")
        return set(names), str(specified_list.resolve())
    gene_embeddings = model_folder / "protein_embeddings.pt"
    if not gene_embeddings.is_file():
        return None, "unverified: protein_embeddings.pt absent; official CLI must validate genes"
    try:
        mapping = torch.load(gene_embeddings, map_location="cpu", weights_only=True)
    except Exception:
        return None, "unverified: protein_embeddings.pt could not be safely parsed as a gene dictionary"
    if not isinstance(mapping, dict) or not mapping or not all(
            isinstance(key, str) for key in mapping):
        return None, "unverified: protein_embeddings.pt has an unsupported gene-profile format"
    return set(mapping), str(gene_embeddings.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-folder", type=Path,
                        default=ROOT / "resources/foundation_annotation/state_se600m")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional explicit SE checkpoint understood by the installed State CLI")
    parser.add_argument("--model-gene-list", type=Path,
                        help="One gene symbol per line, only if protein_embeddings.pt cannot be read")
    parser.add_argument("--state-bin", default="state")
    parser.add_argument("--nuclear-layer", default="raw_nuclear_counts")
    parser.add_argument("--cytoplasm-layer", default="raw_extranuclear_counts")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--whole-cell-x", action="store_true",
                        help="Use raw integer whole-cell X (canonical LungPF exporter)")
    source.add_argument("--corrected-total-x", action="store_true",
                        help="Use ResolVI-corrected finite nonnegative float X (experimental)")
    parser.add_argument("--read-chunk-size", type=int, default=2048)
    parser.add_argument("--min-gene-overlap", type=float, default=0.8,
                        help="Fraction of input panel genes present in SE gene embeddings")
    args = parser.parse_args()
    if args.read_chunk_size < 1 or not 0 < args.min_gene_overlap <= 1:
        parser.error("read-chunk-size must be positive and min-gene-overlap in (0,1]")
    if not args.input_h5ad.is_file():
        raise FileNotFoundError(args.input_h5ad)
    if not torch.cuda.is_available():
        raise RuntimeError("STATE SE embedding requires a GPU compute node")
    model_folder = args.model_folder.resolve()
    if not (model_folder / "config.yaml").is_file():
        raise FileNotFoundError(f"STATE SE config.yaml is absent from {model_folder}")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.checkpoint is None and not any(
            (model_folder / name).is_file() for name in
            ("model.safetensors", "se600m_epoch16.ckpt", "se600m_epoch4.ckpt")):
        raise FileNotFoundError("No recognizable SE-600M weights in model folder")
    command_path = shutil.which(args.state_bin)
    if command_path is None:
        raise FileNotFoundError(f"Official State CLI {args.state_bin!r} is not on PATH")
    gene_set, gene_source = model_genes(model_folder, args.model_gene_list)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in
               ("embeddings.npy", "cell_ids.txt", "provenance.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing embedding result")

    data = ad.read_h5ad(args.input_h5ad)
    cells = validate_names(data.obs_names, "cell IDs")
    genes = validate_names(data.var_names, "gene symbols")
    if gene_set is None:
        matched, coverage = None, None
        matched_index = np.arange(len(genes), dtype=np.int64)
        print("[state-se] model gene overlap unavailable; official transform must check mapping", flush=True)
    else:
        matched = [gene for gene in genes if gene in gene_set]
        coverage = len(matched) / len(genes)
        if not matched or coverage < args.min_gene_overlap:
            raise ValueError(f"Insufficient STATE SE gene coverage: {len(matched)}/{len(genes)}")
        matched_index = np.asarray([i for i, gene in enumerate(genes) if gene in gene_set],
                                   dtype=np.int64)
    if args.whole_cell_x or args.corrected_total_x:
        if data.X is None or data.X.shape != data.shape:
            raise ValueError("Whole-cell X is absent or has the wrong shape")
        nuclear = cytoplasm = None
    else:
        for layer in (args.nuclear_layer, args.cytoplasm_layer):
            if layer not in data.layers:
                raise KeyError(f"Required raw count layer {layer!r} is absent")
            if data.layers[layer].shape != data.shape:
                raise ValueError(f"Layer {layer!r} has the wrong shape")
        nuclear = data.layers[args.nuclear_layer]
        cytoplasm = data.layers[args.cytoplasm_layer]
    for start in range(0, len(cells), args.read_chunk_size):
        stop = min(start + args.read_chunk_size, len(cells))
        if args.whole_cell_x or args.corrected_total_x:
            raw = dense_rows(data.X, start, stop)
            if args.corrected_total_x:
                validate_corrected(raw)
            else:
                validate_counts(raw, "X")
        else:
            nuc = dense_rows(nuclear, start, stop)
            cyto = dense_rows(cytoplasm, start, stop)
            validate_counts(nuc, args.nuclear_layer)
            validate_counts(cyto, args.cytoplasm_layer)
            raw = nuc + cyto
            if "raw_total_counts" in data.layers:
                expected = dense_rows(data.layers["raw_total_counts"], start, stop)
                if not np.array_equal(raw, expected):
                    raise ValueError(f"N+C disagrees with raw_total_counts at rows {start}:{stop}")
        if not np.all(np.any(raw[:, matched_index] > 0, axis=1)):
            raise ValueError(f"A cell at rows {start}:{stop} lacks any expressed "
                             f"{'SE-matched' if gene_set is not None else 'input'} gene")
    if args.whole_cell_x or args.corrected_total_x:
        raw_matrix = data.X.copy()
    else:
        raw_matrix = nuclear + cytoplasm
    if sparse.issparse(raw_matrix):
        raw_matrix = raw_matrix.tocsr()
    staged = ad.AnnData(X=raw_matrix,
                        obs=pd.DataFrame(index=pd.Index(cells, name="cell_id")),
                        var=pd.DataFrame(index=pd.Index(genes, name="gene_symbol")))
    cell_digest = hashlib.sha256("\n".join(cells).encode()).hexdigest()
    gene_digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()

    with tempfile.TemporaryDirectory(prefix="state-se-", dir=output_dir) as staging:
        stage = Path(staging)
        official_input = stage / "whole_cell_input.h5ad"
        official_output = stage / "state_output.npy"
        staged.write_h5ad(official_input)
        check_input = ad.read_h5ad(official_input, backed="r")
        staged_cells = list(map(str, check_input.obs_names))
        if staged_cells != cells or list(map(str, check_input.var_names)) != genes:
            raise ValueError("Staged STATE input changed cell or gene order")
        check_input.file.close()
        command = [command_path, "emb", "transform", "--model-folder",
                   str(model_folder), "--input", str(official_input),
                   "--output", str(official_output)]
        if args.checkpoint is not None:
            command.extend(["--checkpoint", str(args.checkpoint.resolve())])
        with (output_dir / "transform.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                           cwd=ROOT, check=True)
        if not official_output.is_file():
            raise FileNotFoundError("STATE SE CLI returned without an output NPY")
        embedding = np.load(official_output, mmap_mode="r", allow_pickle=False)
        if embedding.ndim != 2 or embedding.shape[0] != len(staged_cells) or embedding.shape[1] < 2:
            raise ValueError("STATE SE embedding has invalid dimensions")
        if not np.issubdtype(embedding.dtype, np.floating):
            raise ValueError("STATE SE output is not a floating-point embedding matrix")
        if not np.isfinite(embedding).all():
            raise FloatingPointError("STATE SE embedding contains nonfinite values")
        embedding_dim, embedding_dtype = embedding.shape[1], str(embedding.dtype)
        del embedding
        os.replace(official_output, stage / "embeddings.npy")
        (stage / "cell_ids.txt").write_text("\n".join(staged_cells) + "\n")
        metadata = {
            "model": "ArcInstitute_STATE_SE_600M", "embedding": "official_SE_transform_npy",
            "official_command": command, "official_cli": command_path,
            "input_h5ad": str(args.input_h5ad.resolve()),
            "input_layers": ["X"] if args.whole_cell_x or args.corrected_total_x else
                            [args.nuclear_layer, args.cytoplasm_layer],
            "input_source": "resolvi_corrected_total_float_X" if args.corrected_total_x else
                            "raw_integer_whole_cell_X" if args.whole_cell_x else
                            "raw_integer_nuclear_plus_cytoplasmic_layers",
            "input_transform": "whole-cell expression matrix passed to official STATE SE transform",
            "model_folder": str(model_folder), "checkpoint":
                str(args.checkpoint.resolve()) if args.checkpoint else None,
            "model_gene_source": gene_source,
            "model_gene_count": len(gene_set) if gene_set is not None else None,
            "cells": len(cells), "genes": len(genes),
            "matched_genes": len(matched) if matched is not None else None,
            "input_gene_coverage": coverage, "matched_gene_symbols": matched,
            "gene_coverage_verified": gene_set is not None,
            "embedding_dim": embedding_dim, "embedding_dtype": embedding_dtype,
            "cell_ids_sha256": cell_digest,
            "genes_sha256": gene_digest,
            "cell_order": "staged input order verified; NPY row order relies on official transform contract; output row count verified",
            "official_reference": "https://github.com/ArcInstitute/state#state-embedding-model-se",
        }
        (stage / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for name in ("embeddings.npy", "cell_ids.txt", "provenance.json"):
            os.replace(stage / name, output_dir / name)
    print(f"[state-se] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
