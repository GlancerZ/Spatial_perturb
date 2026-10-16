"""V6 token/mask sampling with V7 deterministic-left bin values.

V6's randomized tie assignment consumes random numbers before whole-cell
truncation. Preserve that generator stream so the controlled comparison changes
only bin values, even for panels with more genes than the whole-cell budget.
"""
from __future__ import annotations

import numpy as np

from .subcellular_v3 import sample_paired_queries_v3, sample_uniform_queries_v3
from .subcellular_v5 import encode_whole_cell_v5, recover_raw_counts_v5
from .subcellular_v7 import V7Dataset, collate_v7


def encode_whole_cell_v6_left(
    nuclear_counts, cyto_counts, global_ids, *, pad_token_id, cls_token_id,
    max_tokens=1200, rng=None, seed=None, masked_genes=None,
    masked_compartment=None, mask_whole_cell=False,
):
    """Keep V6's selected gene IDs/order and replace only their bin values."""
    output = encode_whole_cell_v5(
        nuclear_counts, cyto_counts, global_ids,
        pad_token_id=pad_token_id, cls_token_id=cls_token_id,
        max_tokens=max_tokens, rng=rng, seed=seed, masked_genes=masked_genes,
        masked_compartment=masked_compartment, mask_whole_cell=mask_whole_cell)
    nuclear = np.asarray(nuclear_counts, dtype=np.int64).copy()
    cyto = np.asarray(cyto_counts, dtype=np.int64).copy()
    if masked_genes is not None:
        nuclear[masked_genes] = 0
        cyto[masked_genes] = 0
    if masked_compartment is not None:
        (nuclear, cyto)[masked_compartment][:] = 0
    if mask_whole_cell:
        nuclear[:] = 0
        cyto[:] = 0
    total = nuclear + cyto
    expressed = np.flatnonzero(total > 0)
    if len(expressed):
        values = total[expressed]
        edges = np.quantile(values, np.linspace(0, 1, 50))
        deterministic = np.digitize(values, edges)
        by_gene = dict(zip(np.asarray(global_ids)[expressed].tolist(), deterministic.tolist()))
        output['whole_input_bins'][1:] = [by_gene[int(gene)] for gene in output['whole_gene_ids'][1:]]
    return output


class V6LeftDataset(V7Dataset):
    """Use V7's masks/targets and restore V6's whole-cell token selection."""

    def __getitem__(self, index):
        output = super().__getitem__(index)
        slice_index = int(np.searchsorted(self.ends, index, side='right'))
        entry = self.slices[slice_index]
        local = index - (self.ends[slice_index - 1] if slice_index else 0)
        nodes = np.asarray(entry['nodes'][local], dtype=np.int64)
        real = nodes >= 0
        rows = np.maximum(nodes, 0)
        nuclear = np.asarray(entry['nuclear'][rows], dtype=np.float32)
        cyto = np.asarray(entry['cyto'][rows], dtype=np.float32)
        cyto[~real] = 0
        seed = int(self.data.get('seed', 42))
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, self.epoch, entry['slice_seed'], int(nodes[0]), 831]))
        task_mode = int(rng.choice(3, p=[.7, .2, .1]))
        neighbor_rows = real & (nodes != nodes[0])
        if task_mode == 2 and not (cyto[neighbor_rows] > 0).any():
            task_mode = 0
        if task_mode != int(output['task_mode']):
            raise AssertionError('V6-left task sampling differs from V7')
        masked_compartment = int(rng.integers(2)) if task_mode == 1 else None
        if task_mode == 0:
            hidden_genes, _ = sample_paired_queries_v3(
                len(entry['genes']), int(self.data['query_genes']),
                self.data.get('paired_mask_ratio', .5), rng)
        else:
            hidden_genes = None
            sample_uniform_queries_v3(len(entry['genes']), int(self.data['query_genes']), rng)
        output.update(encode_whole_cell_v6_left(
            recover_raw_counts_v5(nuclear[0], name='nuclear center'),
            recover_raw_counts_v5(cyto[0], name='cytoplasmic center'),
            entry['genes'], pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            max_tokens=self.whole_cell_max_tokens,
            seed=np.random.SeedSequence([seed, entry['slice_seed'], int(nodes[0]), 355]),
            masked_genes=hidden_genes, masked_compartment=masked_compartment,
            mask_whole_cell=task_mode == 2))
        return output


collate_v6_left = collate_v7
