"""Check that V6-left changes bin values but not training examples or tokens.

Run only on a compute node with the staged full V6 cache.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import yaml

from spatial_pretrain.data.subcellular_v6 import V6Dataset
from spatial_pretrain.data.subcellular_v6_left import V6LeftDataset, encode_whole_cell_v6_left
from spatial_pretrain.data.subcellular_v5 import encode_whole_cell_v5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    original = V6Dataset(yaml.safe_load(args.original.read_text()), split='train')
    candidate = V6LeftDataset(yaml.safe_load(args.candidate.read_text()), split='train')
    if len(original) != len(candidate) or not np.array_equal(original.ends, candidate.ends):
        raise AssertionError('Selected training centers changed')
    allowed = {'input_bins', 'whole_input_bins', 'target_bins'}
    compared = changed = 0
    modes = set()
    for start, end in zip(np.r_[0, original.ends[:-1]], original.ends):
        for index in sorted({int(start), int(min(start + 1, end - 1)), int(min(start + 17, end - 1))}):
            old, new = original[index], candidate[index]
            if old.keys() != new.keys():
                raise AssertionError(f'Fields differ at center {index}')
            for key in old:
                if key not in allowed and not np.array_equal(old[key], new[key]):
                    raise AssertionError(f'Non-bin field changed at center {index}: {key}')
            changed += any(not np.array_equal(old[key], new[key]) for key in allowed)
            modes.add(int(old['task_mode']))
            compared += 1
    if changed == 0:
        raise AssertionError('The new tie policy did not change any tested bin')

    # A synthetic overlength panel checks the V6 RNG stream used for truncation.
    rng = np.random.default_rng(1729)
    nuclear = rng.integers(1, 8, size=1600)
    cyto = rng.integers(1, 8, size=1600)
    ids = np.arange(100, 1700)
    kwargs = dict(pad_token_id=0, cls_token_id=1, max_tokens=1200, seed=42)
    old = encode_whole_cell_v5(nuclear, cyto, ids, **kwargs)
    new = encode_whole_cell_v6_left(nuclear, cyto, ids, **kwargs)
    if not np.array_equal(old['whole_gene_ids'], new['whole_gene_ids']):
        raise AssertionError('Overlength whole-cell token selection changed')
    if not np.array_equal(old['whole_token_valid_mask'], new['whole_token_valid_mask']):
        raise AssertionError('Overlength whole-cell validity changed')
    if np.array_equal(old['whole_input_bins'], new['whole_input_bins']):
        raise AssertionError('Overlength whole-cell bins did not change')
    print(f'PASS: {compared} centers across {len(original.ends)} slices; '
          f'{changed} changed-bin examples; task modes {sorted(modes)}; '
          'overlength token order preserved', flush=True)


if __name__ == '__main__':
    main()
