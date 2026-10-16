"""V8 pilot trainer: the unchanged V7 training loop with V8 data/model/loss.

V8 exposes three switches on top of V7 route A/B (see the config):
``binning.tie_policy`` (left = V7, random = V6), ``model.residual_cap``
(soft norm cap of the CLS residual) and ``loss.nonzero_weight``. Everything
else, including optimizer partition, schedule, sampler, checkpoint and resume
contracts, is the V7 implementation.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_subcellular_v7 as v7  # noqa: E402
from spatial_pretrain.data.subcellular_v8 import V8Dataset, collate_v8, tie_policy  # noqa: E402
from spatial_pretrain.models.subcellular_v8 import SubcellularV8  # noqa: E402
from spatial_pretrain.training.subcellular_v8 import losses, nonzero_weight  # noqa: E402

MODEL_VERSION = 8
ARCHITECTURE = 'subcellular_capped_routes_v8'
BINS_PROTOCOL = {
    'left': 'deterministic_left_visible_inputs_full_panel_51_bin_targets',
    'random': 'random_ties_visible_inputs_full_panel_51_bin_targets',
}
SOURCE_FILES = (*v7.SOURCE_FILES,
                'scripts/train_subcellular_v8.py',
                'src/spatial_pretrain/data/subcellular_v8.py',
                'src/spatial_pretrain/models/subcellular_v8.py',
                'src/spatial_pretrain/training/subcellular_v8.py')
_validate_v7 = v7.validate_config


def validate_config(cfg):
    policy = tie_policy(cfg)
    nonzero_weight(cfg)
    cap = cfg['model'].get('residual_cap')
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0):
        raise ValueError('model.residual_cap must be positive or null')
    if cfg['protocol'].get('bins') != BINS_PROTOCOL[policy]:
        raise ValueError(f'protocol.bins must be {BINS_PROTOCOL[policy]!r} for tie_policy={policy}')
    # Reuse every other V7 check on a copy expressed in V7 vocabulary.
    normalized = copy.deepcopy(cfg)
    normalized['binning'] = {'num_bins': 51, 'tie_policy': 'left'}
    normalized['loss'] = {'type': 'masked_bin_cross_entropy'}
    normalized['protocol']['bins'] = BINS_PROTOCOL['left']
    if cfg['binning'].get('num_bins') != 51:
        raise ValueError('V8 keeps 51 bins')
    _validate_v7(normalized)


v7.MODEL_VERSION = MODEL_VERSION
v7.ARCHITECTURE = ARCHITECTURE
v7.SOURCE_FILES = SOURCE_FILES
v7.validate_config = validate_config
v7.V7Dataset = V8Dataset
v7.collate_v7 = collate_v8
v7.SubcellularV7 = SubcellularV8
v7.losses = losses


if __name__ == '__main__':
    v7.main()
