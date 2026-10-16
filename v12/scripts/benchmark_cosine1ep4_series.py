"""Two GPU workers, eight immutable checkpoints each, fixed frozen-probe protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

RUN = Path('outputs/xenium_compartment_ready48_cosine1ep_4gpu')
OUT = Path('outputs/xenium_cosine1ep4_benchmarks')
STEPS = list(range(100, 1501, 100)) + [1543]


def summarize():
    rows = []
    for step in STEPS:
        directory = OUT / f'step_{step:08d}'
        if not (directory / 'comparison_qa.json').exists():
            continue
        for r in json.loads((directory / 'metrics.json').read_text()):
            rows.append(dict(step=step, model=r['model'], task=r['task'],
                macro_f1=r['macro_f1'], balanced_accuracy=r['balanced_accuracy'],
                test_cells=r['test_cells']))
    if not rows:
        return
    table = pd.DataFrame(rows).sort_values(['step', 'model', 'task'])
    table.to_csv(OUT / 'metrics_by_step.csv', index=False)
    lines = ['# 四卡单 epoch cosine：检查点评测', '',
        '冻结模型，使用相同供体划分、预处理和线性分类器协议；SpatialFormer 复用既有基线。',
        '同一测试集的重复评测用于探索训练趋势；超参数仅按验证集选择。', '',
        '| Step | 细胞类型 macro-F1 | Balanced accuracy | 区域 macro-F1 |',
        '|---|---:|---:|---:|']
    for step, group in table[table.model == 'ours'].groupby('step'):
        r = group.set_index('task')
        lines.append(f'| {step} | {r.loc["cell_type", "macro_f1"]:.4f} | '
                     f'{r.loc["cell_type", "balanced_accuracy"]:.4f} | '
                     f'{r.loc["region", "macro_f1"]:.4f} |')
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')


def worker(rank):
    # Child inherits a single visible GPU. The extractor must retain its identity.
    for step in STEPS[rank * 8:(rank + 1) * 8]:
        checkpoint = RUN / 'checkpoints' / f'step_{step:08d}.pt'
        directory = OUT / f'step_{step:08d}'
        qa = directory / 'comparison_qa.json'
        assert checkpoint.exists(), checkpoint
        if qa.exists():
            result = json.loads(qa.read_text())
            assert result['checkpoint'] == str(checkpoint.resolve())
            assert result['step'] == step and result['all_classifiers_converged']
        else:
            print(f'[worker {rank}] evaluating step={step}', flush=True)
            with (OUT / f'step_{step:08d}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'scripts/benchmark_step2300.py'],
                    env=dict(os.environ, BENCHMARK_STEP=str(step),
                        BENCHMARK_CHECKPOINT=str(checkpoint), BENCHMARK_OUTPUT_DIR=str(directory),
                        BENCHMARK_EXTRACTION_GPUS='1'),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            assert json.loads(qa.read_text())['step'] == step
        print(f'[worker {rank}] complete step={step}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=[0, 1])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        worker(args.worker)
        return
    schedule = json.loads((RUN / 'training_schedule.json').read_text())
    assert schedule['total_updates'] == 1543 and schedule['global_batch_size'] == 32
    assert all((RUN / 'checkpoints' / f'step_{s:08d}.pt').exists() for s in STEPS)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    assert len(visible) >= 2
    (OUT / 'plan.json').write_text(json.dumps(dict(checkpoints=STEPS,
        workers={visible[i]: STEPS[i*8:(i+1)*8] for i in range(2)},
        training_schedule=schedule), indent=2))
    workers = [subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(i)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=visible[i])) for i in range(2)]
    while any(p.poll() is None for p in workers):
        summarize()
        import time
        time.sleep(15)
    statuses = [p.wait() for p in workers]
    summarize()
    assert statuses == [0, 0], statuses
    (OUT / 'complete.json').write_text(json.dumps(dict(steps=STEPS, complete=True)))
    print('[series] all 16 checkpoints evaluated', flush=True)


if __name__ == '__main__':
    main()
