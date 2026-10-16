"""Single-GPU evaluator queue, run within an existing compute tmux/Slurm step.

Never submits resources. A failed checkpoint is recorded and skipped, not
silently retried; other variants can still produce results.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('cosine', 'contrastive', 'relational')


def atomic(path, payload):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deadline', required=True, type=int)
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Evaluator must run inside a compute step')
    root = ROOT / 'outputs/subcellular_v10_evaluation_20260929'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    lock = (root / '.watcher.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = root / 'watcher.json'
    failures = [json.loads(path.read_text()) for path in root.glob('*/step_*/failure.json')]
    while time.time() < args.deadline - 300:
        pending = []
        for variant in args.variants:
            source = ROOT / f'outputs/subcellular_v10_{variant}_20260929'
            for checkpoint in (source / 'checkpoints').glob('step_*.pt'):
                step = int(checkpoint.stem.removeprefix('step_'))
                destination = root / variant / checkpoint.stem
                if not (destination / 'metrics.json').exists() and not (destination / 'failure.json').exists():
                    pending.append((step, variant, checkpoint, destination))
        if not pending:
            training_done = all((ROOT / f'outputs/subcellular_v10_{v}_20260929/TRAIN_COMPLETE.json').is_file()
                                for v in args.variants)
            if training_done:
                for variant in args.variants:
                    source = ROOT / f'outputs/subcellular_v10_{variant}_20260929'
                    completed = json.loads((source / 'TRAIN_COMPLETE.json').read_text())
                    last_step = completed['total_updates']
                    expected = set(range(0, last_step + 1, 100)) | {last_step}
                    accounted = {int(p.parent.name.removeprefix('step_'))
                                 for pattern in ('step_*/metrics.json', 'step_*/failure.json')
                                 for p in (root / variant).glob(pattern)}
                    if not expected.issubset(accounted):
                        training_done = False
            atomic(state, dict(status='finished_with_failures' if training_done and failures else
                               'complete' if training_done else 'waiting_for_checkpoints',
                               updated=time.time(), failures=failures, jobid=os.environ.get('SLURM_JOB_ID')))
            if training_done:
                return 1 if failures else 0
            time.sleep(15)
            continue
        step, variant, checkpoint, destination = min(pending)
        destination.mkdir(parents=True, exist_ok=True)
        atomic(state, dict(status='evaluating', variant=variant, step=step, started=time.time(),
                           pending=len(pending), checkpoint=str(checkpoint), failures=failures))
        command = [sys.executable, '-u', 'scripts/evaluate_subcellular_v10.py',
                   '--checkpoint', str(checkpoint), '--expected-step', str(step),
                   '--output-dir', str(destination), '--batch-size', '16']
        print(f'[evaluate] {variant} step={step} output={destination}', flush=True)
        log_path = root / 'logs' / f'{variant}_{checkpoint.stem}.log'
        with log_path.open('a') as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode or not (destination / 'metrics.json').is_file():
            failure = dict(variant=variant, step=step, returncode=result.returncode,
                           log=str(log_path), time=time.time())
            atomic(destination / 'failure.json', failure)
            failures.append(failure)
            print(f'[evaluation_failed] {failure}', flush=True)
        else:
            print(f'[evaluation_complete] {variant} step={step}', flush=True)
            summary = subprocess.run([sys.executable, 'scripts/summarize_subcellular_v10.py',
                                      '--inputs', str(root), '--output-dir', str(root / 'comparison')], cwd=ROOT)
            if summary.returncode:
                atomic(root / 'summary_failure.json', dict(returncode=summary.returncode, time=time.time()))
    atomic(state, dict(status='allocation_deadline', updated=time.time(), failures=failures))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
