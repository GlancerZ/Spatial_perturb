"""Compute-node coordinator: one-GPU bridge/eval plus four-GPU training.

The shared training lock is inherited by child processes, so coordinator death
cannot start a second trainer. Four-GPU intent requests a saved update boundary
from the bridge; the single card subsequently stays dedicated to evaluation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
COHORT = '0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc'


def atomic_json(path, payload):
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Runner:
    def __init__(self, args):
        self.args = args
        self.config_path = args.config.resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.fingerprint = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        self.run = Path(self.cfg['output_dir'])
        self.run.mkdir(parents=True, exist_ok=True)
        (self.run / 'logs').mkdir(exist_ok=True)
        self.role = args.role
        self.deadline = float(os.environ.get('V3_TRAIN_DEADLINE', 'inf'))
        self.eval_deadline = float(os.environ.get('SPATIAL_EVAL_DEADLINE', str(self.deadline)))
        self.jobid = os.environ['SLURM_JOB_ID']
        self.intent = self.run / 'four_gpu_intent.json'
        self.stop_markers = ('PAUSE_REQUESTED', 'STOP_REQUESTED')
        self.lock = None
        self.owner = None
        self.eval_lock = None
        if torch.cuda.device_count() < (4 if self.role == 'train' else 1):
            raise ValueError('Selected allocation has insufficient visible GPUs')

    def status(self, status, **extra):
        payload = dict(status=status, role=self.role, jobid=self.jobid, pid=os.getpid(),
                       updated=datetime.now(timezone.utc).isoformat(), **extra)
        atomic_json(self.run / f'runner_{self.role}.json', payload)
        print('[v3-runner]', json.dumps(payload), flush=True)

    def stopped(self):
        return any((self.run / name).exists() for name in self.stop_markers)

    def child(self, command, kind, deadline):
        self.status(kind, command=command)
        inherited = tuple(handle.fileno() for handle in (self.owner, self.lock, self.eval_lock) if handle is not None)
        with (self.run / 'logs' / f'{kind}_{self.role}.log').open('a', buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=inherited, start_new_session=True)
            while process.poll() is None:
                # Trainers observe their own deadline/stop marker and save. Eval
                # has no mutable optimizer state and may safely be interrupted.
                if kind == 'evaluation' and (time.time() >= deadline or self.stopped()):
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                    self.status('evaluation_interrupted_safely')
                    return False
                time.sleep(5)
            if process.returncode:
                self.status('action_required', returncode=process.returncode, kind=kind)
                if process.returncode not in (-9, -15, 137, 143):
                    (self.run / 'PAUSE_REQUESTED').write_text(f'V3 {kind} failed: {process.returncode}; {log.name}\n')
                raise RuntimeError(f'{kind} failed with {process.returncode}; inspect {log.name}')
        return True

    def trainer_command(self, four=False, limit=0, initialize=False):
        command = [sys.executable, '-u']
        if four:
            command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
        command += ['scripts/train_subcellular_v3.py', '--config', str(self.config_path)]
        if limit: command += ['--run-update-limit', str(limit)]
        if initialize: command += ['--initialize-only']
        return command

    def evaluated(self, step):
        metric = self.run / 'downstream_validation' / f'step_{step:08d}' / 'metrics.json'
        if not metric.exists(): return False
        report = json.loads(metric.read_text())
        if (report.get('step') != step or report.get('evaluation_version', 0) < 3
                or not report.get('masked_validation', {}).get('enabled')
                or report.get('config_sha256') != self.fingerprint
                or report.get('cohort_sha256') != COHORT
                or report.get('test_donors_used') is not False
                or report.get('backbone_kind') != 'scgpt'):
            raise ValueError(f'Incomplete/incompatible evaluation: {metric}')
        return True

    def pending(self):
        points = []
        for checkpoint in sorted((self.run / 'checkpoints').glob('step_*.pt')):
            step = int(checkpoint.stem.removeprefix('step_'))
            if not self.evaluated(step):
                points.append((step, checkpoint))
        return points

    def has_four_gpu_intent(self):
        """Clear stale handoff only after scheduler-confirmed allocation end."""
        try: claim = json.loads(self.intent.read_text())
        except FileNotFoundError: return False
        job = str(claim['jobid'])
        if not job.isdigit(): raise ValueError('Invalid four-GPU intent job ID')
        try:
            result = subprocess.run(['squeue', '-h', '-j', job, '-o', '%T'],
                                    text=True, capture_output=True, timeout=30)
            if result.returncode or result.stdout.strip(): return True
            accounting = subprocess.run(['sacct', '-X', '-n', '-j', job,
                                         '--format=JobIDRaw,State', '--parsable2'],
                                        text=True, capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            self.status('scheduler_uncertain_preserving_handoff', error=str(error))
            return True
        if accounting.returncode: return True
        terminal = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
                    'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
        for row in accounting.stdout.splitlines():
            fields = row.strip().split('|')
            if len(fields) >= 2 and fields[0] == job and fields[1].split()[0].rstrip('+') in terminal:
                # A later allocation can publish its intent while sacct runs.
                try:
                    if json.loads(self.intent.read_text()) == claim:
                        self.intent.unlink()
                        self.status('cleared_terminal_four_gpu_intent', ended_jobid=job, slurm_state=fields[1])
                except FileNotFoundError:
                    pass
                return self.intent.exists()
        return True  # Accounting lag or scheduler uncertainty is not terminal.

    def evaluate_pending(self):
        with (self.run / 'evaluation.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                except BlockingIOError:
                    if self.stopped() or time.time() >= self.eval_deadline - 120: return False
                    time.sleep(5)
            self.eval_lock = lock
            try:
                result = self._evaluate_pending_locked()
            finally:
                self.eval_lock = None
        if result and (self.run / 'downstream_validation/step_00000000/metrics.json').exists():
            subprocess.run(['bash', 'scripts/watch_subcellular_v3_eval.sh', '--run-dir', str(self.run),
                            '--config', str(self.config_path), '--batch-size', str(self.cfg['evaluation']['batch_size']),
                            '--summary-only'], cwd=ROOT, check=True)
        return result

    def _evaluate_pending_locked(self):
        for step, checkpoint in self.pending():
            if self.stopped() or time.time() >= self.eval_deadline - 120:
                return False
            output = self.run / 'downstream_validation' / f'step_{step:08d}'
            command = [sys.executable, '-u', 'scripts/evaluate_subcellular_v3.py',
                       '--config', str(self.config_path), '--checkpoint', str(checkpoint),
                       '--output-dir', str(output), '--expected-step', str(step),
                       '--cells-per-donor', '400', '--batch-size', str(self.cfg['evaluation']['batch_size']),
                       '--mask-validation-batches', str(self.cfg['evaluation']['mask_validation_batches'])]
            if not self.child(command, 'evaluation', self.eval_deadline):
                return False
        return True

    def train_four(self):
        # Evaluator/bridge sees this before the blocking lock, and the bridge
        # trainer checkpoints at its next optimizer boundary before releasing.
        atomic_json(self.intent, dict(jobid=self.jobid, pid=os.getpid(), requested=time.time()))
        try:
            with (self.run / 'training.lock').open('a') as lock:
                self.lock = lock
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                    except BlockingIOError:
                        if self.stopped() or time.time() >= self.deadline: return
                        time.sleep(5)
                if self.stopped() or time.time() >= self.deadline or (self.run / 'TRAIN_COMPLETE').exists(): return
                if not (self.run / 'resume.pt').exists():
                    self.child(self.trainer_command(four=True, initialize=True), 'initialization', self.deadline)
                # The initial checkpoint must be evaluated before optimization.
                fcntl.flock(lock, fcntl.LOCK_UN)
                self.lock = None
                while not self.evaluated(0):
                    if self.stopped() or time.time() >= self.deadline: return
                    self.status('waiting_for_initial_evaluation')
                    time.sleep(15)
                self.lock = lock
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                    except BlockingIOError:
                        if self.stopped() or time.time() >= self.deadline: return
                        time.sleep(5)
                if self.stopped() or time.time() >= self.deadline: return
                self.child(self.trainer_command(four=True), 'training', self.deadline)
        finally:
            self.lock = None
            if self.intent.exists() and json.loads(self.intent.read_text()).get('jobid') == self.jobid:
                self.intent.unlink()

    def bridge_and_evaluate(self):
        while not self.stopped() and time.time() < self.eval_deadline - 120:
            if not self.evaluate_pending(): return
            if (self.run / 'TRAIN_COMPLETE').exists():
                self.status('all_training_and_evaluation_complete'); return
            if self.has_four_gpu_intent():
                self.status('waiting_for_next_checkpoint_four_gpu_training')
                time.sleep(30); continue
            if time.time() >= self.deadline - 60: return
            with (self.run / 'training.lock').open('a') as lock:
                try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(10); continue
                self.lock = lock
                try:
                    if self.intent.exists() or self.stopped(): continue
                    if not (self.run / 'resume.pt').exists():
                        self.child(self.trainer_command(initialize=True), 'initialization', self.deadline)
                    else:
                        # resume.pt is authoritative; progress.json is a report
                        # and can lag if preemption occurred between renames.
                        resume = torch.load(self.run / 'resume.pt', map_location='cpu', weights_only=False)
                        if resume['config_sha256'] != self.fingerprint:
                            raise ValueError('Bridge resume/config mismatch')
                        step = int(resume['step'])
                        del resume
                        interval = self.cfg['training']['checkpoint_every']
                        limit = interval - step % interval
                        self.child(self.trainer_command(limit=limit), 'training', self.deadline)
                finally:
                    self.lock = None

    def run_foreground(self):
        with (self.run / f'coordinator_{self.role}.lock').open('a') as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.owner = owner
            self.status('started')
            if self.role == 'train': self.train_four()
            else: self.bridge_and_evaluate()
            self.status('clean_boundary_exit', training_complete=(self.run / 'TRAIN_COMPLETE').exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--role', choices=['train', 'eval'], required=True)
    Runner(parser.parse_args()).run_foreground()


if __name__ == '__main__':
    main()
