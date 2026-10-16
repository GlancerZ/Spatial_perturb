"""Evaluate every committed V6-left A/B archive on one dedicated compute GPU.

This worker never requests resources or changes training state. Run inside a
Slurm compute step; pass the actual allocation end as --deadline-epoch.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from train_subcellular_v6_left import ARCHITECTURE, MODEL_VERSION, config_hash, validate_config


EVAL = Path(__file__).with_name('evaluate_subcellular_v6_left.py')
ARCHIVE = re.compile(r'step_(\d{8})\.pt\Z')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_route(config_path: Path, route: str) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    validate_config(cfg)
    if cfg['model']['route'] != route:
        raise ValueError(f'Expected route {route}: {config_path}')
    return dict(config=config_path.resolve(), cfg=cfg, run=Path(cfg['output_dir']).resolve(),
                fingerprint=config_hash(cfg))


def committed(route: str, entry: dict):
    run = entry['run']
    progress_path = run / 'progress.json'
    schedule_path = run / 'schedule.json'
    if not progress_path.is_file() or not schedule_path.is_file():
        return None, []
    progress = json.loads(progress_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    if (progress.get('config_sha256') != entry['fingerprint']
            or progress.get('architecture') != ARCHITECTURE
            or progress.get('model_version') != MODEL_VERSION
            or schedule.get('source_sha256') is None
            or schedule.get('global_batch_size') != 64):
        raise ValueError(f'{route}: progress/schedule identity differs from V6-left config')
    step, total = progress['step'], schedule['total_updates']
    if not isinstance(step, int) or not isinstance(total, int) or not 0 <= step <= total:
        raise ValueError(f'{route}: invalid committed step/epoch length')
    if total != 10269:
        raise ValueError(f'{route}: expected the unchanged V6 10,269-step epoch, got {total}')
    archives = {}
    for path in (run / 'checkpoints').glob('step_*.pt'):
        match = ARCHIVE.fullmatch(path.name)
        if match is None or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'{route}: invalid archive {path}')
        archived_step = int(match[1])
        if archived_step <= step:
            archives[archived_step] = path
    required = {0, *range(100, step + 1, 100)}
    if progress.get('complete'):
        if step != total:
            raise ValueError(f'{route}: completion cursor differs from epoch length')
        required.add(total)
    missing = required - set(archives)
    if missing:
        raise ValueError(f'{route}: committed steps lack archives: {sorted(missing)}')
    # TRAIN_COMPLETE is published immediately after progress.json; tolerate
    # that narrow publication window without reporting a false failure.
    confirmed_complete = bool(progress.get('complete') and (run / 'TRAIN_COMPLETE').is_file())
    return dict(step=step, total=total, complete=confirmed_complete,
                required=sorted(required)), [(route, n, archives[n]) for n in sorted(required)]


def verified(metrics_path: Path, step: int, checkpoint_sha: str, config_sha: str) -> bool:
    if not metrics_path.is_file():
        return False
    report = json.loads(metrics_path.read_text())
    if (report.get('model_version') != MODEL_VERSION
            or report.get('architecture') != ARCHITECTURE
            or report.get('step') != step
            or report.get('checkpoint_sha256') != checkpoint_sha
            or report.get('config_sha256') != config_sha
            or report.get('canonical_cohort') is not True
            or report.get('cells') != 5200
            or report.get('test_donors_used') is not False):
        raise ValueError(f'Evaluation output has a different identity/protocol: {metrics_path}')
    for row in report.get('results', []):
        expected = .1 if row['task'] == 'cell_type' else .01
        if row['fixed_C_result']['C'] != expected:
            raise ValueError(f'Fixed-C probe mismatch: {metrics_path}')
    if not report.get('results'):
        raise ValueError(f'Evaluation contains no probe results: {metrics_path}')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-a', type=Path, required=True)
    parser.add_argument('--config-b', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--poll-seconds', type=int, default=30)
    parser.add_argument('--minimum-launch-seconds', type=int, default=600)
    parser.add_argument('--priority-checkpoint', action='append', default=[],
                        help='Evaluate this committed archive first, as A:3800 or B:4900')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run the evaluator queue inside a live Slurm compute step')
    if args.poll_seconds < 1 or args.minimum_launch_seconds < 1:
        parser.error('Polling and launch margin must be positive')
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
    lock = (output / 'owner.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(schema='subcellular_v6_left_fixed5200_queue_v1',
                    evaluator_sha256=sha256(EVAL),
                    routes={route: dict(config=str(entry['config']),
                                        config_sha256=entry['fingerprint'], run=str(entry['run']))
                            for route, entry in entries.items()})
    manifest = output / 'identity.json'
    if manifest.is_file():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError('Queue output belongs to another evaluator/configuration')
    elif any(path.name != 'owner.lock' for path in output.iterdir()):
        raise ValueError('Nonempty evaluation output lacks queue identity')
    else:
        atomic_json(manifest, identity)
    digest_cache = {}

    def archive_digest(path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in digest_cache:
            digest_cache[key] = sha256(path)
        return digest_cache[key]

    while True:
        snapshots, pending = {}, []
        for route, entry in entries.items():
            snapshot, tasks = committed(route, entry)
            snapshots[route] = snapshot
            for task in tasks:
                _, step, checkpoint = task
                target = output / route / f'step_{step:08d}'
                digest = archive_digest(checkpoint)
                if not verified(target / 'metrics.json', step, digest, entry['fingerprint']):
                    pending.append((step, route, checkpoint, digest, target))
        pending.sort(key=lambda task: (0, priorities[(task[1], task[0])])
                     if (task[1], task[0]) in priorities else (1, task[0], task[1]))
        done = all(snapshot and snapshot['complete'] for snapshot in snapshots.values()) and not pending
        atomic_json(output / 'status.json', dict(snapshots=snapshots,
            queued=[dict(route=route, step=step) for step, route, *_ in pending],
            priority_checkpoints=args.priority_checkpoint,
            all_evaluated=bool(done), updated_at=time.time()))
        if done:
            print('[v6-left-eval] A/B complete; every required checkpoint verified', flush=True)
            return
        if time.time() + args.minimum_launch_seconds >= args.deadline_epoch:
            print('[v6-left-eval] Allocation deadline reached; queue is resumable', flush=True)
            return
        if pending:
            step, route, checkpoint, digest, target = pending[0]
            target.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(EVAL), '--checkpoint', str(checkpoint),
                       '--config', str(entries[route]['config']), '--output-dir', str(target),
                       '--cohort', 'fixed5200', '--cells-per-donor', '400',
                       '--mask-validation-batches', '16', '--expected-step', str(step)]
            print(f'[v6-left-eval] {route} step {step}: {target}', flush=True)
            subprocess.run(command, check=True)
            if not verified(target / 'metrics.json', step, digest, entries[route]['fingerprint']):
                raise RuntimeError(f'Evaluator returned without valid metrics: {target}')
        else:
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
