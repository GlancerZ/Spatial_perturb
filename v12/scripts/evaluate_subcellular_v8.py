"""Evaluate a V8 pilot checkpoint with the unchanged V7 evaluator.

Same fixed 5,200-cell LungPF cohort, fixed-C probes, neighbor shuffle and
masked validation as V7. Input binning follows the checkpoint's
``binning.tie_policy`` so each arm is evaluated on the representation it
was trained with.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import train_subcellular_v8 as trainer  # noqa: E402  (patches the V7 trainer module)
import evaluate_subcellular_v7 as ev  # noqa: E402
from spatial_pretrain.data.subcellular_v8 import (  # noqa: E402
    V8Dataset, collate_v8, compartment_encoder, tie_policy, whole_cell_encoder)
from spatial_pretrain.models.subcellular_v8 import SubcellularV8  # noqa: E402
from spatial_pretrain.training.subcellular_v8 import losses  # noqa: E402

_load_model = ev.load_model


def load_model(path, expected_step=None, config_path=None):
    model, cfg, vocab, step, provenance = _load_model(path, expected_step, config_path)
    policy = tie_policy(cfg)
    ev.encode_compartment_context_v7 = compartment_encoder(policy)
    ev.encode_whole_cell_v7 = whole_cell_encoder(policy)
    ev.BINNING_PROTOCOL = f'V8 tie_policy={policy}; ' + trainer.BINS_PROTOCOL[policy]
    provenance['tie_policy'] = policy
    return model, cfg, vocab, step, provenance


ev.MODEL_VERSION = trainer.MODEL_VERSION
ev.ARCHITECTURE = trainer.ARCHITECTURE
ev.SOURCE_FILES = trainer.SOURCE_FILES
ev.validate_config = trainer.validate_config
ev.SubcellularV7 = SubcellularV8
ev.V7Dataset = V8Dataset
ev.collate_v7 = collate_v8
ev.training_losses = losses
ev.load_model = load_model


if __name__ == '__main__':
    ev.main()
