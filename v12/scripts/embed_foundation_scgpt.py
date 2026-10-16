"""Extract frozen ordinary-scGPT whole-human CLS embeddings from raw N+C H5AD.

Run on a GPU compute node.  The asset root must contain the verified ordinary
scGPT source, source_manifest.json, and scGPT_human/{args,vocab,best_model}.
This deliberately refuses scGPT-spatial assets.  Corrected expression must be
selected explicitly; the default remains raw nuclear plus cytoplasmic counts.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import anndata as ad
import numpy as np
from scipy import sparse
import torch

from spatial_pretrain.models.scgpt_backbone import ScGPTBackbone


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ordinary_collator(assets: Path, vocab, cfg, tie_policy: str = "left"):
    """Load only the official ordinary scGPT collator, without its package imports."""
    code = assets / "code/scgpt"
    package_name = "_gse316387_scgpt_upstream"
    package = types.ModuleType(package_name)
    package.__path__ = [str(code)]
    sys.modules[package_name] = package
    preprocess_path = code / "preprocess.py"
    preprocess = types.ModuleType(package_name + ".preprocess")
    preprocess.__dict__.update(np=np, torch=torch, Union=Union,
                               logger=logging.getLogger(package_name))
    tree = ast.parse(preprocess_path.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ("_digitize", "binning")]
    if len(definitions) != 2:
        raise ValueError("Ordinary upstream preprocess lacks binning functions")
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 str(preprocess_path), "exec"), preprocess.__dict__)
    if tie_policy != "left":
        raise ValueError("Only deterministic left tie handling is supported")
    preprocess._digitize = lambda values, bins, side="both": np.digitize(values, bins)
    sys.modules[preprocess.__name__] = preprocess
    collator_path = code / "data_collator.py"
    specification = importlib.util.spec_from_file_location(
        package_name + ".data_collator", collator_path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    collator = module.DataCollator(
        do_padding=True, pad_token_id=vocab[cfg["pad_token"]],
        pad_value=cfg["pad_value"], do_mlm=False, do_binning=True,
        max_length=1200, sampling=True, keep_first_n_tokens=1)
    provenance = {
        "collator_source": str(collator_path),
        "collator_sha256": file_sha256(collator_path),
        "preprocessing_source": str(preprocess_path),
        "preprocessing_sha256": file_sha256(preprocess_path),
        "n_bins": 51, "max_length": 1200, "tie_policy": tie_policy,
    }
    return collator, provenance


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path,
                        default=ROOT / "resources/scgpt_human")
    parser.add_argument("--nuclear-layer", default="raw_nuclear_counts")
    parser.add_argument("--cytoplasm-layer", default="raw_extranuclear_counts")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--whole-cell-x", action="store_true",
                        help="Use raw integer whole-cell X (canonical LungPF exporter)")
    source.add_argument("--corrected-total-x", action="store_true",
                        help="Use ResolVI-corrected finite nonnegative float X (experimental)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--read-chunk-size", type=int, default=2048)
    parser.add_argument("--min-gene-overlap", type=float, default=0.8,
                        help="Fraction of input panel genes present in official vocabulary")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.read_chunk_size < args.batch_size:
        parser.error("batch-size must be positive and read-chunk-size >= batch-size")
    if not 0 < args.min_gene_overlap <= 1:
        parser.error("min-gene-overlap must be in (0,1]")
    if not args.input_h5ad.is_file():
        raise FileNotFoundError(args.input_h5ad)
    if not torch.cuda.is_available():
        raise RuntimeError("scGPT embedding requires a GPU compute node")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / name for name in
               ("embeddings.npy", "cell_ids.txt", "provenance.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite an existing embedding result")

    data = ad.read_h5ad(args.input_h5ad)
    cells = validate_names(data.obs_names, "cell IDs")
    genes = validate_names(data.var_names, "gene symbols")
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

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.mha.set_fastpath_enabled(False)
    model = ScGPTBackbone(args.assets_root, checkpointing=False,
                          backbone_kind="scgpt")
    cfg, vocab = model.config, model.vocab
    if cfg.get("n_bins", 51) != 51 or cfg.get("input_emb_style") != "continuous":
        raise ValueError("Only the verified ordinary 51-bin continuous checkpoint is supported")
    keep = np.asarray([i for i, gene in enumerate(genes) if gene in vocab],
                      dtype=np.int64)
    coverage = len(keep) / len(genes)
    if not len(keep) or coverage < args.min_gene_overlap:
        raise ValueError(f"Insufficient scGPT vocabulary coverage: {len(keep)}/{len(genes)}")
    ids = np.asarray([vocab[genes[i]] for i in keep], dtype=np.int64)
    collator, collator_provenance = load_ordinary_collator(
        args.assets_root, vocab, cfg, tie_policy="left")
    model = model.eval().requires_grad_(False).cuda()
    lengths: list[int] = []
    truncated_cells = 0

    with tempfile.TemporaryDirectory(prefix="scgpt-", dir=output_dir) as staging:
        stage = Path(staging)
        matrix_path = stage / "embeddings.npy"
        result = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32,
                                           shape=(len(cells), model.dim))
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
                    total = dense_rows(data.layers["raw_total_counts"], start, stop)
                    if not np.array_equal(raw, total):
                        raise ValueError(f"N+C disagrees with raw_total_counts at rows {start}:{stop}")
            raw = raw[:, keep].astype(np.float32, copy=False)
            examples = []
            for local_row, values in enumerate(raw):
                positive = np.flatnonzero(values > 0)
                if not len(positive):
                    raise ValueError(f"No matched expressed genes for cell {cells[start + local_row]}")
                lengths.append(len(positive))
                truncated_cells += len(positive) > 1199
                examples.append({
                    "id": start + local_row,
                    "genes": torch.from_numpy(np.r_[vocab["<cls>"], ids[positive]]).long(),
                    "expressions": torch.from_numpy(
                        np.r_[cfg["pad_value"], values[positive]]).float(),
                })
            for offset in range(0, len(examples), args.batch_size):
                batch = collator(examples[offset:offset + args.batch_size])
                token_ids = batch["gene"].cuda()
                padding = token_ids.eq(model.pad_id)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    hidden = model(token_ids, batch["expr"].cuda(), padding,
                                   torch.zeros((*token_ids.shape, model.dim),
                                               device="cuda"))
                cls = torch.nn.functional.normalize(hidden[:, 0].float(), p=2, dim=-1)
                values = cls.cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite scGPT CLS embedding")
                first = start + offset
                result[first:first + len(values)] = values
            print(f"[scgpt] {stop}/{len(cells)} cells", flush=True)
        result.flush()
        del result
        check = np.load(matrix_path, mmap_mode="r")
        if check.shape != (len(cells), model.dim) or not np.isfinite(check).all():
            raise ValueError("Stored embeddings failed shape/finite verification")
        (stage / "cell_ids.txt").write_text("\n".join(cells) + "\n")
        cell_digest = hashlib.sha256("\n".join(cells).encode()).hexdigest()
        gene_digest = hashlib.sha256("\n".join(genes).encode()).hexdigest()
        metadata = {
            "model": "ordinary_scgpt_whole_human", "embedding": "frozen_cls_l2",
            "input_h5ad": str(args.input_h5ad.resolve()),
            "input_layers": ["X"] if args.whole_cell_x or args.corrected_total_x else
                            [args.nuclear_layer, args.cytoplasm_layer],
            "input_source": "resolvi_corrected_total_float_X" if args.corrected_total_x else
                            "raw_integer_whole_cell_X" if args.whole_cell_x else
                            "raw_integer_nuclear_plus_cytoplasmic_layers",
            "input_transform": "matched positive genes; upstream 51-bin collator with deterministic left ties",
            "cells": len(cells), "genes": len(genes), "matched_genes": len(keep),
            "input_gene_coverage": coverage, "matched_gene_symbols": [genes[i] for i in keep],
            "expressed_matched_genes_min": min(lengths),
            "expressed_matched_genes_max": max(lengths),
            "truncated_cells": truncated_cells, "embedding_dim": model.dim,
            "cell_ids_sha256": cell_digest, "genes_sha256": gene_digest,
            "cell_order": "identical to input H5AD obs_names; one output row per cell",
            "seed": args.seed, "backbone_provenance": model.provenance,
            "collator_provenance": collator_provenance,
        }
        (stage / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
        for name in ("embeddings.npy", "cell_ids.txt", "provenance.json"):
            os.replace(stage / name, output_dir / name)
    print(f"[scgpt] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
