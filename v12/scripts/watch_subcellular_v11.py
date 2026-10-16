"""Sequential independent-GPU evaluation queue; never requests resources.

Every stable step checkpoint is evaluated, including step zero and final. Only
verified all-readout/all-task results count. Deadlines and idle limits yield
resumable incomplete state, never a false completion. Run in compute tmux.
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
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluate_subcellular_v11 import file_sha256, validate_completion


def atomic(path, payload):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checkpoint_step(path):
    suffix = path.stem.removeprefix('step_')
    return int(suffix) if path.suffix == '.pt' and path.stem.startswith('step_') and suffix.isdigit() else None


def expected_steps(completion, interval=100):
    final = int(completion['total_updates'])
    interval = int(completion.get('checkpoint_every', interval))
    if final < 0 or interval < 1:
        raise ValueError('Invalid training completion update count/interval')
    return set(range(0, final + 1, interval)) | {final}


def verified_result(path, step, checkpoint=None):
    report = json.loads(path.read_text())
    validate_completion(report, path.parent)
    if int(report['step']) != step:
        raise ValueError('Evaluation step does not match filename')
    if checkpoint is not None and report.get('checkpoint_sha256') != file_sha256(checkpoint):
        raise ValueError('Completed evaluation belongs to different checkpoint bytes')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/benchmarking_data/lungpf/prepared')
    parser.add_argument('--baseline-cache-dir', type=Path, default=ROOT / 'outputs/subcellular_v11_original_baseline_cache')
    parser.add_argument('--training-complete', type=Path)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--deadline', type=int, help='Absolute Unix allocation deadline')
    parser.add_argument('--idle-timeout-seconds', type=float, default=1800.)
    parser.add_argument('--poll-seconds', type=float, default=15.)
    parser.add_argument('--minimum-start-seconds', type=float, default=1200.)
    parser.add_argument('--deadline-margin-seconds', type=float, default=180.)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--composition-radius-um', type=float, default=100.)
    args = parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run evaluator inside a compute Slurm step')
    if min(args.poll_seconds, args.idle_timeout_seconds, args.minimum_start_seconds) <= 0:
        parser.error('Watcher intervals must be positive')
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    (root / 'logs').mkdir(exist_ok=True)
    lock = (root / '.watcher.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    done_path = args.training_complete or args.checkpoint_dir.parent / 'TRAIN_COMPLETE.json'
    state = root / 'watcher.json'
    last_progress = time.time()
    durations = []
    failures = []
    verified_cache = {}
    while True:
        now = time.time()
        checkpoints = sorted((step, path) for path in args.checkpoint_dir.glob('step_*.pt')
                             if (step := checkpoint_step(path)) is not None)
        seen, pending = {}, []
        for step, path in checkpoints:
            if step in seen:
                raise ValueError(f'Duplicate step checkpoint: {step}')
            seen[step] = path
            destination = root / f'step_{step:06d}'
            complete = destination / 'metrics.json'
            if complete.exists():
                signature = (path.stat().st_size, path.stat().st_mtime_ns, complete.stat().st_mtime_ns)
                if verified_cache.get(step) != signature:
                    verified_result(complete, step, path)
                    verified_cache[step] = signature
                continue
            if (destination / 'failure.json').exists():
                failure = json.loads((destination / 'failure.json').read_text())
                if failure not in failures:
                    failures.append(failure)
                continue
            pending.append((step, path, destination))
        complete_steps = {step for step in seen if (root / f'step_{step:06d}/metrics.json').exists()}
        expected = expected_steps(json.loads(done_path.read_text()), args.checkpoint_every) if done_path.exists() else None
        missing = sorted(expected - complete_steps) if expected is not None else None
        common = dict(updated=now, jobid=os.environ.get('SLURM_JOB_ID'), completed_steps=sorted(complete_steps),
                      expected_steps=sorted(expected) if expected is not None else None, missing_steps=missing, failures=failures)
        if failures:
            atomic(state, dict(status='failed_requires_review', **common))
            return 2
        if expected is not None and not missing and not pending and not failures:
            atomic(state, dict(status='complete', **common))
            return 0
        if not pending and expected is not None:
            # Training is finished but checkpoints missing or evaluations failed.
            atomic(state, dict(status='finished_incomplete', **common))
            return 2
        if args.deadline is not None:
            # The first estimate is conservative; observed durations guide later launches.
            estimate = max(args.minimum_start_seconds, max(durations, default=0.) * 1.2)
            remaining = args.deadline - now
            if remaining < args.deadline_margin_seconds + (estimate if pending else 0):
                atomic(state, dict(status='insufficient_remaining_time', remaining_seconds=remaining,
                                   estimated_next_evaluation_seconds=estimate, **common))
                return 3
        if not pending:
            if now - last_progress >= args.idle_timeout_seconds:
                atomic(state, dict(status='waiting_timeout_incomplete', **common))
                return 3
            atomic(state, dict(status='waiting_for_checkpoints', **common))
            time.sleep(min(args.poll_seconds, 60.))
            continue
        step, checkpoint, destination = pending[0]
        destination.mkdir(parents=True, exist_ok=True)
        atomic(state, dict(status='evaluating', step=step, checkpoint=str(checkpoint), **common))
        command = [sys.executable, '-u', str(ROOT / 'scripts/evaluate_subcellular_v11.py'),
            '--checkpoint', str(checkpoint), '--expected-step', str(step), '--output-dir', str(destination),
            '--prepared', str(args.prepared), '--batch-size', str(args.batch_size),
            '--baseline-cache-dir', str(args.baseline_cache_dir), '--composition-radius-um', str(args.composition_radius_um)]
        log_path = root / 'logs' / f'step_{step:06d}.log'
        print(f'[evaluate-v11] step={step} checkpoint={checkpoint} log={log_path}', flush=True)
        started = time.time()
        with log_path.open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            interrupted = False
            while process.poll() is None:
                if args.deadline is not None and time.time() >= args.deadline - args.deadline_margin_seconds:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    interrupted = True
                    break
                time.sleep(min(args.poll_seconds, 15.))
        if interrupted:
            # Partial stage caches are valid for the next allocation. Do not
            # persist failure.json, which would block a legitimate resume.
            atomic(state, dict(status='allocation_deadline_partial', step=step, log=str(log_path), **common))
            return 3
        error = None
        try:
            if process.returncode:
                raise RuntimeError(f'Evaluator exit code {process.returncode}')
            verified_result(destination / 'metrics.json', step, checkpoint)
        except (OSError, ValueError, KeyError, RuntimeError) as caught:
            error = str(caught)
        if error:
            failure = dict(step=step, returncode=process.returncode, error=error, log=str(log_path), time=time.time())
            atomic(destination / 'failure.json', failure)
            failures.append(failure)
            print(f'[evaluation_failed-v11] {failure}', flush=True)
            atomic(state, dict(status='failed_requires_review', **(common | {'failures': failures})))
            return 2
        else:
            durations.append(time.time()-started)
            print(f'[evaluation_complete-v11] step={step} seconds={durations[-1]:.1f}', flush=True)
            command = [sys.executable, str(ROOT / 'scripts/summarize_subcellular_v11.py'), '--inputs', str(root),
                       '--output-dir', str(root / 'comparison')]
            summary = subprocess.run(command, cwd=ROOT)
            if summary.returncode:
                atomic(root / 'summary_failure.json', dict(returncode=summary.returncode, time=time.time()))
        last_progress = time.time()


if __name__ == '__main__':
    raise SystemExit(main())
