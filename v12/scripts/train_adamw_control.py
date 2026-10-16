"""Run an AdamW control with explicitly identical starting model weights."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from spatial_pretrain.config import load_config
from spatial_pretrain.training import runner

original_build = runner._build_optimizer

def build_control_optimizer(model, config):
    initial = torch.load(ROOT / 'outputs/cytoplasm_mean_full_muon_v1/initial.pt',
                         map_location='cpu', weights_only=False)
    assert initial['step'] == 0
    model.load_state_dict(initial['model'], strict=True)
    assert all(torch.equal(value.cpu(), initial['model'][key])
               for key, value in model.state_dict().items())
    print('[control] Exact Muon step-0 weights loaded and verified; fresh AdamW optimizer.', flush=True)
    return original_build(model, config)

def main():
    cfg = load_config(ROOT / 'configs/cytoplasm_mean_full_adamw_v1.yaml')
    assert cfg.optimizer.name == 'adamw'
    resume = Path(cfg.experiment.output_dir) / 'resume.pt'
    if resume.exists():
        previous = torch.load(resume, map_location='cpu', weights_only=False)
        if previous['step'] >= 700:
            raise SystemExit('AdamW comparison horizon already reached')
        cfg.training.run_update_limit = 700 - previous['step']
        del previous
    runner._build_optimizer = build_control_optimizer
    runner.run_pretraining(cfg)

if __name__ == '__main__':
    main()
