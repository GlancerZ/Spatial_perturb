"""Read-only A/B pretrained-backbone and optimizer initialization check on GPU."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
import yaml

from spatial_pretrain.models.subcellular_v6 import SubcellularV6
from spatial_pretrain.training.subcellular_v6_optimizer import build_optimizer
from train_subcellular_v6_left import validate_config


def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    provenance = None
    for route in ('a', 'b'):
        config_path = ROOT / f'configs/subcellular_lr_v6_left_{route}_20260925.yaml'
        cfg = yaml.safe_load(config_path.read_text())
        validate_config(cfg)
        model = SubcellularV6(cfg['assets_dir'], **cfg['model']).cuda()
        model.set_backbone_trainable_layers(0)
        if any(parameter.requires_grad for parameter in model.backbone.parameters()):
            raise AssertionError('Ordinary scGPT backbone is not frozen')
        optimizer = build_optimizer(model, cfg['training'])
        current = model.backbone.provenance
        if current['weights_sha256'] != '6cb5d451ab5c4b33eb673adbe4fddc61d2389df1b89b7651a9fe2e557572b922':
            raise AssertionError('Unexpected scGPT checkpoint')
        if provenance is not None and current != provenance:
            raise AssertionError('A/B initialized from different scGPT weights')
        provenance = current
        print(json.dumps(dict(route=route.upper(), backbone_provenance=current,
                              optimizer_manifest=optimizer.manifest,
                              trainable_parameters=sum(p.numel() for p in model.parameters()
                                                       if p.requires_grad),
                              allocated_gib=torch.cuda.memory_allocated() / 2**30)), flush=True)
        del optimizer, model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
