"""Run at most four independent one-GPU V6-left evaluation workers.

Each worker runs inside its own Slurm compute step. Checkpoint locks protect
the existing fixed5200 outputs; only w0 publishes the canonical queue status.
Stop the legacy queue before starting these workers, and do not restart it
while parallel workers are active.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

from watch_subcellular_v6_left_evaluations import (
    EVAL, committed, load_route, sha256, verified,
)

DEFAULT_PRIORITIES = ('A:3800', 'B:4900', 'A:10269', 'B:10269')
WORKERS = ('w0', 'w1', 'w2', 'w3')


def require_compute_step():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run the evaluator queue inside a live Slurm compute step')


def atomic_json(path: Path, value: dict) -> None:
    """Publish one complete JSON document using a process-unique sibling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f'.{path.name}.{os.getpid()}.',
                                        suffix='.partial', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(path: Path):
    """Yield the owner handle (or False); keep the inode after releasing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            try:
                yield handle
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def worker_lock_path(output: Path, worker_id: str) -> Path:
    if worker_id not in WORKERS:
        raise ValueError(f'Unknown worker ID: {worker_id}')
    return output / 'parallel_workers' / worker_id / 'owner.lock'


def checkpoint_lock_path(output: Path, route: str, step: int) -> Path:
    if route not in ('A', 'B') or not isinstance(step, int) or step < 0:
        raise ValueError('Invalid checkpoint lock identity')
    return output / 'parallel_workers' / 'checkpoint_locks' / f'{route}_step_{step:08d}.lock'


def queue_identity(entries: dict) -> dict:
    # Exactly the legacy identity: its existing metrics remain reusable.
    return dict(schema='subcellular_v6_left_fixed5200_queue_v1',
                evaluator_sha256=sha256(EVAL),
                routes={route: dict(config=str(entry['config']),
                                    config_sha256=entry['fingerprint'], run=str(entry['run']))
                        for route, entry in entries.items()})


def validate_manifest(output: Path, identity: dict) -> None:
    """Exclude the legacy worker for the complete identity check/publication."""
    retry_until = time.monotonic() + 2
    while True:
        with exclusive_lock(output / 'owner.lock') as owned:
            if owned:
                manifest = output / 'identity.json'
                if manifest.is_file():
                    if json.loads(manifest.read_text()) != identity:
                        raise ValueError('Queue output belongs to another evaluator/configuration')
                elif any(path.name not in ('owner.lock', 'parallel_workers') for path in output.iterdir()):
                    raise ValueError('Nonempty evaluation output lacks queue identity')
                else:
                    atomic_json(manifest, identity)
                return
        # Another parallel worker can briefly hold this startup lock. Give it
        # time to finish, while retaining a bounded refusal of a live legacy queue.
        remaining = retry_until - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Legacy evaluator owns owner.lock; stop it before parallel handoff')
        time.sleep(min(.1, remaining))


def publish_status(output: Path, worker_id: str, identity: dict, value: dict) -> None:
    status = dict(value, worker_id=worker_id, queue_identity=identity,
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  slurm_step_id=os.environ.get('SLURM_STEP_ID'), pid=os.getpid())
    atomic_json(output / 'parallel_workers' / worker_id / 'status.json', status)
    if worker_id == 'w0':
        atomic_json(output / 'status.json', status)


def evaluation_command(checkpoint: Path, config: Path, target: Path, step: int) -> list[str]:
    # Batch size remains the evaluator's existing default of 2; fixed-C probes
    # and model/evaluation version 61 are unchanged in the imported evaluator.
    return [sys.executable, str(EVAL), '--checkpoint', str(checkpoint),
            '--config', str(config), '--output-dir', str(target),
            '--cohort', 'fixed5200', '--cells-per-donor', '400',
            '--mask-validation-batches', '16', '--expected-step', str(step)]


def run_queue(args, entries: dict, output: Path, identity: dict, priorities: dict):
    digest_cache = {}

    def archive_digest(path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in digest_cache:
            digest_cache[key] = sha256(path)
        return digest_cache[key]

    def status(snapshots, pending, state, active=None):
        publish_status(output, args.worker_id, identity,
                       dict(snapshots=snapshots,
                            queued=[dict(route=route, step=step) for step, route, *_ in pending],
                            priority_checkpoints=args.priority_checkpoint,
                            all_evaluated=state == 'complete', state=state,
                            active=active, updated_at=time.time()))

    while True:
        snapshots, pending = {}, []
        for route, entry in entries.items():
            snapshot, tasks = committed(route, entry)
            snapshots[route] = snapshot
            for _, step, checkpoint in tasks:
                target = output / route / f'step_{step:08d}'
                digest = archive_digest(checkpoint)
                if not verified(target / 'metrics.json', step, digest, entry['fingerprint']):
                    pending.append((step, route, checkpoint, digest, target))
        pending.sort(key=lambda task: (0, priorities[(task[1], task[0])])
                     if (task[1], task[0]) in priorities else (1, task[0], task[1]))
        done = all(snapshot and snapshot['complete'] for snapshot in snapshots.values()) and not pending
        status(snapshots, pending, 'complete' if done else 'waiting')
        if done:
            print(f'[{args.worker_id}] A/B complete; every required checkpoint verified', flush=True)
            return
        if time.time() + args.minimum_launch_seconds >= args.deadline_epoch:
            status(snapshots, pending, 'deadline')
            print(f'[{args.worker_id}] Allocation deadline reached; queue is resumable', flush=True)
            return
        evaluated = False
        for step, route, checkpoint, digest, target in pending:
            with exclusive_lock(checkpoint_lock_path(output, route, step)) as owned:
                if not owned:
                    continue
                # Another worker may have finished since this queue scan.
                digest = archive_digest(checkpoint)
                if verified(target / 'metrics.json', step, digest, entries[route]['fingerprint']):
                    continue
                if time.time() + args.minimum_launch_seconds >= args.deadline_epoch:
                    status(snapshots, pending, 'deadline')
                    return
                target.parent.mkdir(parents=True, exist_ok=True)
                active = dict(route=route, step=step, checkpoint=str(checkpoint), target=str(target))
                status(snapshots, pending, 'evaluating', active)
                command = evaluation_command(checkpoint, entries[route]['config'], target, step)
                print(f'[{args.worker_id}] {route} step {step}: {target}', flush=True)
                try:
                    # Inherit the checkpoint descriptor so an orphan evaluator
                    # retains ownership if the queue parent is abruptly killed.
                    subprocess.run(command, check=True,
                                   timeout=max(1, args.deadline_epoch - time.time()),
                                   pass_fds=(owned.fileno(),))
                    if not verified(target / 'metrics.json', step, digest, entries[route]['fingerprint']):
                        raise RuntimeError(f'Evaluator returned without valid metrics: {target}')
                except subprocess.TimeoutExpired:
                    status(snapshots, pending, 'deadline', active)
                    print(f'[{args.worker_id}] Evaluator stopped at allocation deadline; queue is resumable',
                          flush=True)
                    return
                except Exception:
                    status(snapshots, pending, 'failed', active)
                    raise
                evaluated = True
                break
        if not evaluated:
            time.sleep(min(args.poll_seconds, max(0, args.deadline_epoch - time.time())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-a', type=Path, required=True)
    parser.add_argument('--config-b', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--worker-id', choices=WORKERS, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--minimum-launch-seconds', type=int, default=600)
    parser.add_argument('--priority-checkpoint', action='append', default=None)
    args = parser.parse_args()
    require_compute_step()
    if args.poll_seconds < 1 or args.minimum_launch_seconds < 1:
        parser.error('Polling and launch margin must be positive')
    if not math.isfinite(args.deadline_epoch):
        parser.error('Deadline must be finite')
    args.priority_checkpoint = args.priority_checkpoint or list(DEFAULT_PRIORITIES)
    priorities = {}
    for token in args.priority_checkpoint:
        if re.fullmatch(r'[AB]:\d+', token) is None:
            parser.error(f'Invalid priority checkpoint: {token}')
        route, step = token.split(':')
        priorities.setdefault((route, int(step)), len(priorities))
    entries = {route: load_route(path, route) for route, path in
               (('A', args.config_a), ('B', args.config_b))}
    if entries['A']['run'] == entries['B']['run']:
        raise ValueError('A/B training output directories must differ')
    output = args.output_dir.resolve()
    if any(output == entry['run'] or output.is_relative_to(entry['run']) for entry in entries.values()):
        raise ValueError('Evaluation queue must be outside training output directories')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(worker_lock_path(output, args.worker_id)) as owned:
        if not owned:
            raise RuntimeError(f'Worker {args.worker_id} already owns this queue')
        identity = queue_identity(entries)
        validate_manifest(output, identity)
        run_queue(args, entries, output, identity, priorities)


if __name__ == '__main__':
    main()
