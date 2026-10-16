"""One-GPU, checkpoint-complete evaluation queue for the frozen V7 runs.

The training-only controllers publish execution_mode.json while owning their
existing controller leases. This worker never prepares a training run, changes
its controller status, deletes an archive, or selects only the latest snapshot.
Use one persistent --output-dir for this shared queue across allocations.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

from run_subcellular_v7 import ROOT, Runner, atomic_json


MODE_SCHEMA = 'subcellular_v7_execution_mode_v1'
QUEUE_SCHEMA = 'subcellular_v7_async_evaluation_queue_v1'
PAUSE_MARKERS = ('STOP_REQUESTED', 'PAUSE_REQUESTED', 'HANDOFF_TO_4GPU_REQUESTED')


class RequiredAttention(RuntimeError):
    """A failed training run or conflicting identity needs operator diagnosis."""


class QueuePause(RuntimeError):
    """A deliberate stop preserves incomplete evaluator output for later use."""


@dataclass(frozen=True)
class EvaluationTask:
    route: str
    step: int
    checkpoint: Path

    def record(self):
        return dict(route=self.route, step=self.step, checkpoint=str(self.checkpoint))


def execution_mode(runner):
    return dict(schema=MODE_SCHEMA, mode='training_only',
                config_sha256=runner.fingerprint, output_dir=str(runner.run))


def mode_ready(runner):
    path = runner.control / 'execution_mode.json'
    if not path.exists():
        return False
    if json.loads(path.read_text()) != execution_mode(runner):
        raise RequiredAttention(f'Conflicting training-only execution mode: {path}')
    return True


def required_steps(step, total):
    required = {0, *range(100, step + 1, 100)}
    if step == total:
        required.add(total)
    return required


def discover_checkpoints(route, runner):
    """Read parent-run metadata and defer the archive-before-progress window."""
    resume = runner.run / 'resume.pt'
    if not resume.is_file() or resume.stat().st_size == 0:
        raise RequiredAttention(f'{route}: formal resume.pt is absent or empty')
    progress, total = runner.progress()
    published, deferred, evaluated, tasks = [], [], [], []
    for checkpoint in sorted((runner.run / 'checkpoints').glob('step_*.pt')):
        match = re.fullmatch(r'step_(\d{8,})\.pt', checkpoint.name)
        if match is None:
            raise RequiredAttention(f'{route}: unknown archive name: {checkpoint}')
        step = int(match[1])
        if step > progress['step']:
            # save() publishes archive, then resume.pt, then progress.json.
            # This normal interval must never be mistaken for corruption.
            deferred.append(step)
            continue
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RequiredAttention(f'{route}: empty/non-file published archive: {checkpoint}')
        if step in published:
            raise RequiredAttention(f'{route}: duplicate archive step {step}')
        published.append(step)
        if runner.evaluated(step):
            evaluated.append(step)
        else:
            tasks.append(EvaluationTask(route, step, checkpoint))
    missing = required_steps(progress['step'], total) - set(published)
    if missing:
        raise RequiredAttention(f'{route}: committed progress lacks required archives: {sorted(missing)}')
    snapshot = dict(ready=True, training_step=progress['step'], total_updates=total,
                    training_complete=progress['complete'], published_steps=published,
                    deferred_publication_steps=deferred, evaluated_steps=evaluated,
                    queued_steps=[task.step for task in tasks],
                    required_steps=sorted(required_steps(progress['step'], total)))
    return snapshot, tasks


def fair_order(tasks):
    """Every old archive first, alternating A then B within a shared step."""
    return sorted(tasks, key=lambda task: (task.step, task.route))


def make_runner(config, deadline, eval_min_seconds):
    # Only use read-only methods on this instance. In particular, never call
    # prepare(), pending(), status(), child(), or run_foreground().
    return Runner(SimpleNamespace(config=config, output_dir=None, resume=True,
                                 deadline=deadline, training_gpus=4, nodes=1,
                                 gpus_per_node=4, eval_min_seconds=eval_min_seconds,
                                 train_min_seconds=120))


class EvaluationQueue:
    def __init__(self, runners, output, deadline, *, poll_seconds=30, eval_min_seconds=600):
        if set(runners) not in ({'A'}, {'B'}, {'A', 'B'}) or any(
                runner.cfg['model']['route'] != route for route, runner in runners.items()):
            raise ValueError('Require the approved A config, B config, or both')
        if len({runner.run for runner in runners.values()}) != len(runners):
            raise ValueError('Evaluation routes must have distinct formal run directories')
        if (not math.isfinite(deadline) or not math.isfinite(poll_seconds) or poll_seconds <= 0
                or not math.isfinite(eval_min_seconds) or eval_min_seconds <= 0):
            raise ValueError('Require a finite deadline and positive finite polling/launch margins')
        self.runners = runners
        self.output = Path(output).resolve()
        for runner in runners.values():
            if any(self.output == protected or self.output.is_relative_to(protected)
                   for protected in (runner.run, runner.control,
                                     runner.run.with_name(runner.run.name + '.evaluation'))):
                raise ValueError('Queue output must be separate from formal/controller/per-run lock directories')
        self.deadline = deadline
        self.poll_seconds = poll_seconds
        self.eval_min_seconds = eval_min_seconds
        self.lock = None
        self.route_locks = {}
        self.identity = dict(schema=QUEUE_SCHEMA, routes={
            route: dict(config=str(runner.config_path), controller_identity=runner.identity)
            for route, runner in sorted(runners.items())})
        self.state = dict(schema=QUEUE_SCHEMA, identity=self.identity,
                          jobid=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(),
                          pid=os.getpid(), deadline=deadline, started_at=time.time(),
                          active=None, routes={}, queued=[], all_evaluated=False)

    def prepare(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / 'owner.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        manifest = self.output / 'identity.json'
        if manifest.exists():
            if json.loads(manifest.read_text()) != self.identity:
                raise RequiredAttention('Queue identity differs from the requested A/B configs')
        elif any(path.name != 'owner.lock' for path in self.output.iterdir()):
            raise RequiredAttention('Existing nonempty queue output lacks its identity')
        else:
            atomic_json(manifest, self.identity)
        status_path = self.output / 'status.json'
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if previous.get('status') in ('failed', 'required_attention'):
                raise RequiredAttention('Previous queue failure requires diagnosis; no automatic retry')
        (self.output / 'logs').mkdir(exist_ok=True)

    def close(self):
        for handle in self.route_locks.values():
            handle.close()
        self.route_locks.clear()
        if self.lock is not None:
            # Never LOCK_UN an inherited open description: an evaluator still
            # alive after a parent crash must retain its inherited leases.
            self.lock.close()
            self.lock = None

    def status(self, status, **extra):
        self.state.update(status=status, updated_at=time.time(), **extra)
        atomic_json(self.output / 'status.json', self.state)
        print('[v7-evaluation-queue]', json.dumps(dict(
            status=status, queued=len(self.state['queued']), active=self.state['active'],
            reason=self.state.get('reason'), all_evaluated=self.state['all_evaluated']),
            allow_nan=False), flush=True)

    def training_status(self, route):
        runner = self.runners[route]
        path = runner.control / 'status.json'
        if not path.exists():
            return None
        status = json.loads(path.read_text())
        if status.get('identity') != runner.identity:
            raise RequiredAttention(f'{route}: controller status has a different identity: {path}')
        if status.get('status') == 'failed':
            raise RequiredAttention(f'{route}: training controller failed: {status.get("error", path)}')
        return status

    def require_run_identity(self, route):
        runner = self.runners[route]
        manifest = runner.control / 'identity.json'
        if not manifest.is_file() or json.loads(manifest.read_text()) != runner.identity:
            raise RequiredAttention(f'{route}: missing or conflicting formal controller identity')

    def lock_route(self, route):
        if route in self.route_locks:
            return
        runner = self.runners[route]
        # A persistent per-run lease prevents duplication even if another
        # worker is accidentally given a different global --output-dir.
        directory = runner.run.with_name(runner.run.name + '.evaluation')
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / 'owner.lock').open('a')
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.require_run_identity(route)
            if not mode_ready(runner):
                raise RequiredAttention(f'{route}: execution mode disappeared while acquiring evaluator lease')
        except BaseException:
            handle.close()
            raise
        self.route_locks[route] = handle

    def pause_markers(self, directory):
        return [name for name in PAUSE_MARKERS if (directory / name).exists()]

    def collect(self):
        snapshots, tasks = {}, []
        # Check both training controllers before enqueueing either route.
        statuses = {route: self.training_status(route) for route in self.runners}
        for route, runner in self.runners.items():
            if not mode_ready(runner):
                snapshots[route] = dict(ready=False, reason='awaiting_training_only_handoff',
                                        training_complete=False, queued_steps=[])
                continue
            self.require_run_identity(route)
            if statuses[route] is None:
                raise RequiredAttention(f'{route}: published execution mode lacks controller status')
            markers = self.pause_markers(runner.run)
            if markers:
                snapshots[route] = dict(ready=False, reason='training_pause_marker', markers=markers,
                                        training_complete=False, queued_steps=[])
                continue
            self.lock_route(route)
            snapshots[route], pending = discover_checkpoints(route, runner)
            snapshots[route]['controller_status'] = statuses[route].get('status')
            tasks.extend(pending)
        tasks = fair_order(tasks)
        complete = all(snapshot.get('ready') and snapshot.get('training_complete')
                       and not snapshot['queued_steps'] and not snapshot['deferred_publication_steps']
                       and set(snapshot['required_steps']).issubset(snapshot['evaluated_steps'])
                       for snapshot in snapshots.values())
        self.state.update(routes=snapshots, queued=[task.record() for task in tasks],
                          training_complete=all(snapshot.get('training_complete', False)
                                                for snapshot in snapshots.values()),
                          all_evaluated=complete)
        return tasks, complete

    def active_guard(self, route):
        for current in self.runners:
            self.training_status(current)
        if self.pause_markers(self.output):
            raise QueuePause('queue_stop_marker')
        if self.pause_markers(self.runners[route].run):
            raise QueuePause(f'{route}: training_pause_marker')
        if not mode_ready(self.runners[route]):
            raise RequiredAttention(f'{route}: training-only execution mode disappeared during evaluation')

    @staticmethod
    def terminate_child(process, entry):
        """Terminate only this child's dedicated process group and record it."""
        entry.setdefault('termination_signals', [])
        for sig, grace in ((signal.SIGTERM, 15), (signal.SIGKILL, None)):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, sig)
                    entry['termination_signals'].append(dict(
                        process_group=process.pid, signal=sig.name, at=time.time()))
                except ProcessLookupError:
                    pass
            try:
                return process.wait(timeout=grace) if grace is not None else process.wait()
            except subprocess.TimeoutExpired:
                continue
        raise RuntimeError('Unreachable child termination state')

    def evaluate(self, task):
        runner = self.runners[task.route]
        self.active_guard(task.route)
        if self.deadline - time.time() < self.eval_min_seconds:
            return False
        command = runner.evaluation_command(task.step, task.checkpoint)
        tag = f'{time.time_ns()}_{task.route}_step_{task.step:08d}'
        log_path = self.output / 'logs' / f'{tag}.log'
        entry = dict(**task.record(), command=command, cwd=str(ROOT), log=str(log_path),
                     started_at=time.time(), status='starting', deadline=self.deadline,
                     exit_code=None, returncode=None)
        self.state['queued'] = [row for row in self.state['queued'] if row != task.record()]
        self.status('evaluating', active=entry)
        process, completed = None, False
        try:
            with log_path.open('x', buffering=1) as log:
                process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True,
                                           pass_fds=(self.lock.fileno(),
                                                     *(handle.fileno() for handle in self.route_locks.values())))
                entry.update(pid=process.pid, process_group=process.pid, status='running')
                self.status('evaluating', active=entry)
                while True:
                    self.active_guard(task.route)
                    remaining = self.deadline - time.time()
                    if remaining <= 0:
                        entry['status'] = 'deadline_interrupted'
                        entry['exit_code'] = self.terminate_child(process, entry)
                        break
                    try:
                        entry['exit_code'] = process.wait(timeout=min(remaining, self.poll_seconds, 60))
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if entry['status'] != 'deadline_interrupted':
                    if entry['exit_code'] != 0:
                        entry['status'] = 'failed'
                        raise subprocess.CalledProcessError(entry['exit_code'], command)
                    entry['status'] = 'verifying_metrics'
                    if not runner.evaluated(task.step):
                        raise RuntimeError('Evaluator returned zero without complete valid metrics')
                    entry['status'], completed = 'passed', True
        except BaseException as error:
            if process is not None:
                polled = process.poll()
                if polled is None:
                    entry['exit_code'] = self.terminate_child(process, entry)
                elif entry['exit_code'] is None:
                    entry['exit_code'] = polled
            if isinstance(error, QueuePause):
                entry['status'] = 'pause_interrupted'
            elif isinstance(error, RequiredAttention):
                entry['status'] = 'required_attention'
            elif entry['status'] == 'verifying_metrics':
                entry['status'] = 'invalid_metrics'
            elif entry['status'] != 'failed':
                entry['status'] = 'launch_failed' if process is None else 'interrupted'
            entry['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            entry.update(finished_at=time.time(), returncode=entry['exit_code'])
            with (self.output / 'commands.jsonl').open('a') as handle:
                handle.write(json.dumps(entry, allow_nan=False) + '\n')
            atomic_json(log_path.with_suffix('.json'), entry)
            self.state.update(active=None, last_command=entry)
        return completed

    def run_foreground(self):
        prepared = False
        try:
            self.prepare()
            prepared = True
            while True:
                if self.pause_markers(self.output):
                    raise QueuePause('queue_stop_marker')
                tasks, complete = self.collect()
                if complete:
                    self.status('all_evaluated', active=None)
                    return
                if self.deadline - time.time() < self.eval_min_seconds:
                    self.status('paused', active=None, reason='evaluation_time_budget')
                    return
                if tasks:
                    if not self.evaluate(tasks[0]):
                        self.status('paused', active=None, reason='evaluation_deadline')
                        return
                else:
                    self.status('idle', active=None, reason='awaiting_training_or_handoff')
                    time.sleep(min(self.poll_seconds, max(0., self.deadline - time.time()), 60))
        except QueuePause as error:
            if prepared:
                self.status('paused', active=None, reason=str(error))
        except RequiredAttention as error:
            if prepared:
                self.status('required_attention', active=None, error=f'{type(error).__name__}: {error}')
            raise
        except BaseException as error:
            if prepared:
                self.status('failed', active=None, error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-a', type=Path)
    parser.add_argument('--config-b', type=Path)
    parser.add_argument('--route', choices=('A', 'B'),
                        help='Evaluate only one route; used by the guarded two-GPU drain')
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--eval-min-seconds', type=float, default=600)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run only inside a verified one-GPU Slurm compute step')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('The shared evaluator requires exactly one visible GPU')
    os.chdir(ROOT)
    configs = {'A': args.config_a, 'B': args.config_b}
    selected = (args.route,) if args.route else ('A', 'B')
    missing = [route for route in selected if configs[route] is None]
    if missing:
        parser.error(f'Missing config for selected route(s): {missing}')
    runners = {route: make_runner(configs[route], args.deadline, args.eval_min_seconds)
               for route in selected}
    queue = EvaluationQueue(runners, args.output_dir, args.deadline,
                            poll_seconds=args.poll_seconds, eval_min_seconds=args.eval_min_seconds)
    def pause_signal(signum, _frame):
        raise QueuePause(f'worker_signal_{signal.Signals(signum).name}')
    signal.signal(signal.SIGTERM, pause_signal)
    signal.signal(signal.SIGINT, pause_signal)
    queue.run_foreground()


if __name__ == '__main__':
    main()
