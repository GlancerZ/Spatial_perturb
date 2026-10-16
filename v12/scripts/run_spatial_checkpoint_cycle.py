"""Train one full epoch in resumable 100-update segments, evaluating each archive."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from spatial_pretrain.config import load_config


def main():
    os.chdir(ROOT)
    config = 'configs/cytoplasm_mean_scgpt_spatial_adamw_v1.yaml'
    cfg = load_config(config)
    assert cfg.training.epochs == 1 and cfg.training.max_train_steps == 0
    run = Path(cfg.experiment.output_dir)
    out = run/'downstream_validation'
    out.mkdir(parents=True, exist_ok=True)
    deadline = float(os.environ['SPATIAL_TRAIN_DEADLINE'])
    while time.time() < deadline:
        if (run/'STOP_REQUESTED').exists():
            print('[stopped-by-user] Do not resume this experiment; checkpoints preserved.', flush=True)
            return
        points = [(0, run/'initial.pt')]
        points += [(int(p.stem.split('_')[-1]), p) for p in sorted((run/'checkpoints').glob('step_*.pt'))]
        for step, checkpoint in points:
            target = out/f'step_{step:08d}'
            if not checkpoint.exists() or (target/'metrics.json').exists():
                continue
            if deadline-time.time() < 900:
                print(f'[cycle-pause] Save remaining evaluation step={step} for next allocation', flush=True)
                return
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES','0,1').split(',')[0]
            started = time.monotonic()
            print(f'[downstream-start] step={step} at={time.time()}', flush=True)
            subprocess.run([sys.executable, '-u', 'scripts/evaluate_cytoplasm_pilot.py',
                '--checkpoint', str(checkpoint), '--expected-step', str(step), '--output', str(target),
                '--cells-per-donor', '400', '--batch-size', '8'], env=env, check=True)
            print(f'[downstream-done] step={step} seconds={time.monotonic()-started:.1f}', flush=True)
            records = [json.loads(p.read_text()) for p in sorted(out.glob('step_*/metrics.json'))]
            assert len({r['cohort_sha256'] for r in records}) == 1
            assert len({r['binning'] for r in records}) == 1
            lines = ['# 每100步固定子集验证', '', '5200中心；每中心256细胞邻域；非独立测试集。', '',
                     '| Step | Cell type macro-F1 | Region macro-F1 |', '|---|---:|---:|']
            for r in records:
                scores = {v['task']: v['best']['val_macro_f1'] for v in r['results']}
                lines.append(f"| {r['step']} | {scores['cell_type']:.6f} | {scores['region']:.6f} |")
            (out/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
        step = 0
        if (run/'resume.pt').exists():
            state = torch.load(run/'resume.pt', map_location='cpu', weights_only=False)
            step, complete = state['step'], state['complete']
            del state
            if complete:
                print('[cycle-complete] One full epoch and all archived checkpoint evaluations finished', flush=True)
                return
        if deadline-time.time() < 300:
            return
        if (run/'STOP_REQUESTED').exists():
            print('[stopped-by-user] Do not resume this experiment; checkpoints preserved.', flush=True)
            return
        updates = 100-step % 100
        print(f'[cycle-train] resume_step={step} next_updates={updates} at={time.time()}', flush=True)
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            'scripts/train_pretrain.py', '--config', config, '--run-update-limit', str(updates)], check=True)


if __name__ == '__main__':
    main()
