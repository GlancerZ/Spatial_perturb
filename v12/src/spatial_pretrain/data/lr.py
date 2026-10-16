from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


LIGAND_CANDIDATES = [
    "ligand",
    "ligand_gene_symbol",
    "ligand_symbol",
    "source",
    "gene_a",
]
RECEPTOR_CANDIDATES = [
    "receptor",
    "receptor_gene_symbol",
    "receptor_symbol",
    "target",
    "gene_b",
]


def normalize_gene_name(name: str) -> str:
    return str(name).strip().upper()


def _find_column(columns: list[str], candidates: list[str]) -> str:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise KeyError(f"Could not find any of {candidates} in columns {columns}")


def load_lr_pairs(path: str | Path) -> list[tuple[str, str]]:
    table = pd.read_csv(path)
    ligand_col = _find_column(table.columns.tolist(), LIGAND_CANDIDATES)
    receptor_col = _find_column(table.columns.tolist(), RECEPTOR_CANDIDATES)
    pairs = (
        table[[ligand_col, receptor_col]]
        .dropna()
        .drop_duplicates()
        .assign(
            **{
                ligand_col: lambda df: df[ligand_col].map(normalize_gene_name),
                receptor_col: lambda df: df[receptor_col].map(normalize_gene_name),
            }
        )
    )
    return list(pairs.itertuples(index=False, name=None))


def map_lr_pairs_to_genes(
    gene_names: list[str],
    lr_pairs: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    gene_to_idx = {normalize_gene_name(name): idx for idx, name in enumerate(gene_names)}
    selected: list[tuple[int, int]] = []
    receptor_gene_ids: set[int] = set()

    for ligand, receptor in lr_pairs:
        if ligand in gene_to_idx and receptor in gene_to_idx:
            lig_idx = gene_to_idx[ligand]
            rec_idx = gene_to_idx[receptor]
            selected.append((lig_idx, rec_idx))
            receptor_gene_ids.add(rec_idx)

    if selected:
        pair_array = np.asarray(selected, dtype=np.int32)
    else:
        pair_array = np.zeros((0, 2), dtype=np.int32)

    receptor_array = np.asarray(sorted(receptor_gene_ids), dtype=np.int32)
    return pair_array, receptor_array


def compute_pairwise_lr_scores(
    expr_subgraph: np.ndarray,
    lr_pairs_idx: np.ndarray,
) -> np.ndarray:
    if lr_pairs_idx.size == 0:
        num_cells = expr_subgraph.shape[0]
        return np.zeros((num_cells, num_cells), dtype=np.float32)

    lig_idx = lr_pairs_idx[:, 0]
    rec_idx = lr_pairs_idx[:, 1]
    ligand_expr = expr_subgraph[:, lig_idx]
    receptor_expr = expr_subgraph[:, rec_idx]
    scores = (ligand_expr[:, None, :] * receptor_expr[None, :, :]).sum(axis=-1)
    return scores.astype(np.float32, copy=False)


def compute_center_candidate_lr_scores(
    center_expr: np.ndarray,
    candidate_expr: np.ndarray,
    lr_pairs_idx: np.ndarray,
) -> np.ndarray:
    if lr_pairs_idx.size == 0 or len(candidate_expr) == 0:
        return np.zeros((len(candidate_expr),), dtype=np.float32)

    lig_idx = lr_pairs_idx[:, 0]
    rec_idx = lr_pairs_idx[:, 1]

    forward = (center_expr[lig_idx][None, :] * candidate_expr[:, rec_idx]).sum(axis=-1)
    backward = (candidate_expr[:, lig_idx] * center_expr[rec_idx][None, :]).sum(axis=-1)
    return (forward + backward).astype(np.float32, copy=False)
