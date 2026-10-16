"""Evaluate archived checkpoints after training releases the existing GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path('outputs')
ARCHIVE = ROOT / 'xenium_compartment_ready48/checkpoints'
SUMMARY = ROOT / 'xenium_checkpoint_benchmarks'


def summarize():
    rows = []
    for directory in ROOT.glob('xenium_step*_benchmark'):
        step = int(directory.name.split('step')[1].split('_')[0])
        if step != 1500 and not (directory / 'comparison_qa.json').exists():
            continue
        metrics = directory / 'metrics.json'
        if not metrics.exists():
            continue
        for result in json.loads(metrics.read_text()):
            rows.append(dict(step=step, model=result['model'], task=result['task'],
                             macro_f1=result['macro_f1'],
                             balanced_accuracy=result['balanced_accuracy'],
                             test_cells=result['test_cells']))
    table = pd.DataFrame(rows).sort_values(['step', 'model', 'task'])
    table.to_csv(SUMMARY / 'metrics_by_step.csv', index=False)
    lines = ['# Checkpoint benchmark progress', '',
             'Frozen encoders; identical donor split, cells and linear-probe protocol.',
             'SpatialFormer baseline reused. Repeated test evaluation is exploratory.', '',
             '| Step | Cell type macro-F1 | Cell type balanced accuracy | Region macro-F1 |',
             '|---|---:|---:|---:|']
    ours = table[table.model == 'ours']
    for step, group in ours.groupby('step'):
        by_task = group.set_index('task')
        lines.append(f'| {step} | {by_task.loc["cell_type", "macro_f1"]:.4f} | '
                     f'{by_task.loc["cell_type", "balanced_accuracy"]:.4f} | '
                     f'{by_task.loc["region", "macro_f1"]:.4f} |')
    (SUMMARY / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    SUMMARY.mkdir(exist_ok=True)
    steps = list(range(2400, 3001, 100)) + [3086]
    summarize()
    for step in steps:
        checkpoint = ARCHIVE / f'step_{step:08d}.pt'
        if not checkpoint.exists():
            raise FileNotFoundError(f'Required checkpoint missing: {checkpoint}')
        output = ROOT / f'xenium_step{step}_benchmark'
        qa = output / 'comparison_qa.json'
        if qa.exists():
            status = json.loads(qa.read_text())
            assert status['step'] == step and status['all_classifiers_converged']
            print(f'[series] verified existing step={step}', flush=True)
        else:
            print(f'[series] evaluating step={step}', flush=True)
            with (SUMMARY / f'step_{step}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', 'scripts/benchmark_step2300.py'],
                               env=dict(os.environ, BENCHMARK_STEP=str(step)),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            assert json.loads(qa.read_text())['step'] == step
        summarize()
        print(f'[series] complete step={step}', flush=True)
    (SUMMARY / 'complete.json').write_text(json.dumps({'steps': steps, 'complete': True}))
    print('[series] all requested checkpoints evaluated', flush=True)


if __name__ == '__main__':
    main()
