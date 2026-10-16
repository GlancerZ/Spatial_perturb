"""Run one unchanged V6 epoch, evaluating step zero and each saved slice.

This compute-node controller neither requests resources nor changes the training
configuration. Its sibling ``<output>.controller`` directory keeps orchestration
logs out of the trainer's protected new-run directory. Resume is always explicit.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
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

import yaml

ROOT = Path(__file__).resolve().parents[1]
ARCHITECTURE = 'subcellular_lr_routes_v6'
COHORT = '0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc'


def atomic_json(path, payload):
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Runner:
    def __init__(self, args):
        self.args = args
        args.nodes = getattr(args, 'nodes', 1)
        args.gpus_per_node = getattr(args, 'gpus_per_node', None) or args.training_gpus // args.nodes
        if args.nodes * args.gpus_per_node != args.training_gpus:
            raise ValueError('nodes * gpus_per_node must match the authorized training GPU count')
        self.config_path = args.config.resolve()
        self.cfg = yaml.safe_load(self.config_path.read_text())
        self.run = Path(self.cfg['output_dir']).resolve()
        if args.output_dir is not None and args.output_dir.resolve() != self.run:
            raise ValueError('--output-dir must equal config output_dir; controller never rewrites config')
        training, evaluation = self.cfg['training'], self.cfg['evaluation']
        if training['epochs'] != 1 or training['checkpoint_every'] != 100 or evaluation['interval'] != 100:
            raise ValueError('This controller requires the agreed one epoch and 100-update interval')
        if (evaluation['cells_per_donor'] != 400 or evaluation['mask_validation_batches'] != 16
                or evaluation['test_donors'] is not False):
            raise ValueError('This controller requires fixed5200, mask16 and no test donors')
        if (args.training_gpus not in (1, 2, 4)
                or training['global_batch_size'] % (training['batch_size'] * args.training_gpus)):
            raise ValueError('Physical GPU count must divide the unchanged global batch')
        if (not math.isfinite(args.deadline) or args.eval_min_seconds <= 0
                or args.train_min_seconds <= 0):
            raise ValueError('Require finite deadline and positive launch margins')
        self.fingerprint = hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest()
        self.control = self.run.with_name(self.run.name + '.controller')
        self.identity = dict(schema='subcellular_v6_controller_v1', output_dir=str(self.run),
                             config_sha256=self.fingerprint, interval=100, cohort='fixed5200',
                             neighbor_shuffle=True, neighbor_shuffle_seed=815,
                             mask_validation_batches=16)
        self.lock = None
        self.state = {}
        self.checkpoint_digests = {}

    def status(self, status, **extra):
        self.state.update(status=status, updated_at=time.time(), deadline=self.args.deadline,
                          jobid=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(),
                          pid=os.getpid(), **extra)
        atomic_json(self.control / 'status.json', self.state)
        print('[v6-controller]', json.dumps(self.state, allow_nan=False), flush=True)

    def prepare(self):
        manifest = self.control / 'identity.json'
        if self.args.resume:
            if not manifest.is_file() or json.loads(manifest.read_text()) != self.identity:
                raise ValueError('--resume requires the matching controller identity')
            if not (self.run / 'resume.pt').is_file():
                raise FileNotFoundError('--resume requires an existing formal resume.pt')
        else:
            if self.run.exists() and any(self.run.iterdir()):
                raise FileExistsError('Formal output is not empty; explicit --resume required')
            if self.control.exists():
                raise FileExistsError('Controller directory already exists; explicit --resume required')
            self.control.mkdir(parents=True)
        self.lock = (self.control / 'owner.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            self.lock = None
            raise
        if not self.args.resume:
            atomic_json(manifest, self.identity)
        (self.control / 'logs').mkdir(exist_ok=True)
        self.state = dict(identity=self.identity, started_at=time.time(), resumed=self.args.resume,
                          training_gpus=self.args.training_gpus)

    def stopped(self):
        return any((self.run / marker).exists() for marker in
                   ('STOP_REQUESTED', 'PAUSE_REQUESTED', 'HANDOFF_TO_4GPU_REQUESTED'))

    def progress(self):
        progress = json.loads((self.run / 'progress.json').read_text())
        schedule = json.loads((self.run / 'schedule.json').read_text())
        step, total = progress.get('step'), schedule.get('total_updates')
        if (isinstance(step, bool) or not isinstance(step, int)
                or isinstance(total, bool) or not isinstance(total, int) or not 0 <= step <= total
                or total < 1 or progress.get('complete') is not (step == total)
                or progress.get('model_version') != 6 or progress.get('architecture') != ARCHITECTURE
                or progress.get('config_sha256') != self.fingerprint
                or schedule.get('model_version') != 6 or schedule.get('architecture') != ARCHITECTURE):
            raise ValueError('Invalid or incompatible formal training progress/schedule')
        return progress, total

    def evaluated(self, step):
        metrics = self.run / 'downstream_validation' / f'step_{step:08d}' / 'metrics.json'
        if not metrics.exists():
            return False
        report = json.loads(metrics.read_text())
        checkpoint = self.run / 'checkpoints' / f'step_{step:08d}.pt'
        info = checkpoint.stat()
        signature = (str(checkpoint), info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if signature not in self.checkpoint_digests:
            with checkpoint.open('rb') as handle:
                self.checkpoint_digests[signature] = hashlib.file_digest(handle, 'sha256').hexdigest()
        if report.get('checkpoint_sha256') != self.checkpoint_digests[signature]:
            raise ValueError(f'Evaluation checkpoint identity mismatch: {metrics}')
        if (report.get('step') != step or report.get('model_version') != 6
                or report.get('architecture') != ARCHITECTURE or report.get('evaluation_version') != 6
                or report.get('config_sha256') != self.fingerprint or report.get('cohort_sha256') != COHORT
                or report.get('cells') != 5200 or report.get('canonical_cohort') is not True
                or report.get('test_donors_used') is not False or report.get('backbone_kind') != 'scgpt'
                or report.get('neighbor_shuffle') is not True or report.get('neighbor_shuffle_seed') != 815
                or report.get('diagnostic_schema_version') != 1
                or not report.get('masked_validation', {}).get('enabled')
                or report['masked_validation'].get('requested_minimum_batches') != 16):
            raise ValueError(f'Incomplete/incompatible evaluation: {metrics}')
        width = self.cfg['model']['slots_per_compartment'] * self.cfg['model']['slot_dim']
        if report.get('arch_config', {}).get('backbone_dim') != 512:
            raise ValueError('Expected the verified ordinary scGPT 512-dimensional backbone')
        if report['arch_config'].get('route') != self.cfg['model']['route']:
            raise ValueError('Evaluation route differs from this experiment')
        dimensions = {name: 512 if name in ('base', 'intrinsic', 'joint') else width
                      for name in ('base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
                                   'nucleus_local', 'cytoplasm_local')}
        dimensions.update({f'{name}_neighbor_shuffle': dimensions[name]
                           for name in ('joint', 'nucleus', 'cytoplasm')})
        expected = {(name, task) for name in dimensions for task in ('cell_type', 'region')}
        results = report.get('results', [])
        actual = [(row.get('embedding'), row.get('task')) for row in results]
        if report.get('embedding_dimensions') != dimensions or set(actual) != expected or len(actual) != len(expected):
            raise ValueError(f'Incomplete readouts/fixed probes: {metrics}')
        for row in results:
            fixed = row.get('fixed_C_result', {})
            score = fixed.get('val_macro_f1')
            if (fixed.get('C') != (.1 if row['task'] == 'cell_type' else .01)
                    or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError(f'Invalid fixed-C probe: {metrics}')
        if report.get('lr_prior', {}).get('sha256') != self.cfg['lr']['sha256']:
            raise ValueError('Evaluation LR prior differs from this experiment')
        return True

    def pending(self, step):
        checkpoints = []
        for checkpoint in sorted((self.run / 'checkpoints').glob('step_*.pt')):
            saved_step = int(checkpoint.stem.removeprefix('step_'))
            if saved_step > step:
                raise ValueError('Archived checkpoint is ahead of reported progress; inspect interrupted save')
            if not self.evaluated(saved_step):
                checkpoints.append((saved_step, checkpoint))
        required = {0} | set(range(100, step + 1, 100))
        if step == json.loads((self.run / 'schedule.json').read_text())['total_updates']:
            required.add(step)
        existing = {int(path.stem.removeprefix('step_'))
                    for path in (self.run / 'checkpoints').glob('step_*.pt')}
        if not required.issubset(existing):
            raise FileNotFoundError(f'Missing formal checkpoints: {sorted(required - existing)}')
        return checkpoints

    def trainer_command(self, *, initialize=False, limit=None):
        command = ['bash', 'scripts/launch_subcellular_v6_distributed.sh', str(self.config_path),
                   str(self.args.nodes), str(self.args.gpus_per_node)]
        if initialize:
            command += ['--initialize-only']
        else:
            command += ['--resume', '--run-update-limit', str(limit)]
        return command

    def evaluation_command(self, step, checkpoint):
        return [sys.executable, '-u', 'scripts/evaluate_subcellular_v6.py',
                '--config', str(self.config_path), '--checkpoint', str(checkpoint),
                '--expected-step', str(step), '--output-dir',
                str(self.run / 'downstream_validation' / f'step_{step:08d}'),
                '--cohort', 'fixed5200', '--batch-size', str(self.cfg['evaluation']['batch_size']),
                '--cells-per-donor', '400', '--mask-validation-batches', '16',
                '--neighbor-shuffle', '--neighbor-shuffle-seed', '815']

    def prelaunch_handoff(self, log_path, step):
        """Recognize only the trainer's explicit startup refusal during A/2->4.

        A request may arrive after the controller's stop check but before the
        trainer acquires its lock. The trainer must still refuse that startup;
        its precise error plus an unchanged saved boundary distinguishes this
        normal handoff from an actual training/launcher failure.
        """
        if self.args.training_gpus != 2 or self.cfg['model']['route'] != 'A':
            return False
        try:
            request_path = self.run.with_name(self.run.name + '.coordination') / 'upgrade_request.json'
            request = json.loads(request_path.read_text())
            marker = json.loads((self.run / 'HANDOFF_TO_4GPU_REQUESTED').read_text())
            if (not isinstance(request, dict) or marker != request
                    or request.get('schema') != 'subcellular_v6_upgrade_request_v1'
                    or request.get('route') != 'A' or request.get('output_dir') != str(self.run)
                    or request.get('target_world_size') != 4
                    or not isinstance(request.get('target_jobid'), str)
                    or not request['target_jobid'].isdigit()):
                return False
            resume = self.run / 'resume.pt'
            if not resume.is_file() or resume.stat().st_size == 0:
                return False
            progress, _ = self.progress()
            if progress['step'] != step or progress['complete']:
                return False
            log = log_path.read_text()
            if any(tag in log for tag in ('[schedule]', '[train]', '[checkpoint]')):
                return False
            refusal = re.compile(
                r'(?:\[rank\d+\]: )?RuntimeError: RuntimeError: '
                r'Run paused for review; inspect the marker before resuming')
            lines = [line.strip() for line in log.splitlines()]
            for line in lines:
                error = re.match(r'(?:\[rank\d+\]: )?(?:[A-Za-z_]\w*\.)*([A-Za-z_]\w*):', line)
                if (error and error[1].endswith(('Error', 'Exception'))
                        and error[1] != 'ChildFailedError' and not refusal.fullmatch(line)):
                    return False
            return any(refusal.fullmatch(line) for line in lines)
        except (OSError, ValueError, KeyError, TypeError):
            # Invalid or missing evidence must preserve the original failure.
            return False

    def child(self, command, kind, step):
        tag = f'{time.time_ns()}_{kind}_step_{step:08d}'
        log_path = self.control / 'logs' / f'{tag}.log'
        entry = dict(kind=kind, step=step, command=command, cwd=str(ROOT), log=str(log_path),
                     started_at=time.time(), status='running', deadline=self.args.deadline)
        self.status(kind, active_child=entry)
        environment = os.environ.copy()
        environment['V6_TRAIN_DEADLINE'] = str(self.args.deadline)
        interrupted = False
        with log_path.open('x', buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(self.lock.fileno(),))
            try:
                timeout = max(.001, self.args.deadline - time.time()) if kind == 'evaluation' else None
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                interrupted = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    returncode = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    returncode = process.wait()
        handoff = (kind == 'training' and returncode != 0 and not interrupted
                   and self.prelaunch_handoff(log_path, step))
        entry.update(returncode=returncode, finished_at=time.time(),
                     status=('deadline_interrupted' if interrupted else 'handoff_before_trainer_start'
                             if handoff else 'passed' if returncode == 0 else 'failed'))
        with (self.control / 'commands.jsonl').open('a') as handle:
            handle.write(json.dumps(entry, allow_nan=False) + '\n')
        self.status(entry['status'], active_child=None, last_child=entry)
        if interrupted or handoff:
            return False
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return True

    def evaluate_pending(self, step):
        points = self.pending(step)
        for saved_step, checkpoint in points:
            if self.stopped() or self.args.deadline - time.time() < self.args.eval_min_seconds:
                self.status('paused', reason='stop_marker' if self.stopped() else 'evaluation_time_budget',
                            step=step, pending_evaluation_steps=[s for s, _ in self.pending(step)])
                return False
            if not self.child(self.evaluation_command(saved_step, checkpoint), 'evaluation', saved_step):
                self.status('paused', reason='evaluation_deadline', step=step,
                            pending_evaluation_steps=[s for s, _ in self.pending(step)])
                return False
            if not self.evaluated(saved_step):
                raise RuntimeError(f'Evaluator exited successfully without valid metrics at step {saved_step}')
        return True

    def run_foreground(self):
        # Do not claim a new output directory if there is no time to initialize
        # its optimizer checkpoint; a later invocation can still start cleanly.
        if not self.args.resume and self.args.deadline - time.time() < self.args.train_min_seconds:
            print('[v6-controller] insufficient initialization time; no run created', flush=True)
            return
        self.prepare()
        try:
            if not self.args.resume:
                if self.stopped() or self.args.deadline - time.time() < self.args.train_min_seconds:
                    self.status('paused', reason='initialization_time_budget', step=None)
                    return
                self.child(self.trainer_command(initialize=True), 'initialization', 0)
                progress, _ = self.progress()
                if progress['step'] != 0:
                    raise ValueError('Initialization must save optimizer step zero')
            while True:
                progress, total = self.progress()
                step = progress['step']
                if not self.evaluate_pending(step):
                    return
                if progress['complete']:
                    self.status('complete', step=step, total_updates=total, pending_evaluation_steps=[])
                    return
                if self.stopped() or self.args.deadline - time.time() < self.args.train_min_seconds:
                    self.status('paused', reason='stop_marker' if self.stopped() else 'training_time_budget',
                                step=step, total_updates=total, pending_evaluation_steps=[])
                    return
                target = min(total, step + 100 - step % 100)
                if not self.child(self.trainer_command(limit=target - step), 'training', step):
                    self.status('paused', reason='handoff_before_trainer_start', step=step,
                                total_updates=total, pending_evaluation_steps=[])
                    return
                advanced, new_total = self.progress()
                if new_total != total or not step < advanced['step'] <= target:
                    raise ValueError('Training progress must increase without skipping its evaluation boundary')
        except Exception as error:
            self.status('failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.lock.close()
            self.lock = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, help='Optional assertion; must match config output_dir')
    parser.add_argument('--resume', action='store_true', help='Explicitly continue this formal run')
    parser.add_argument('--training-gpus', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--nodes', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--gpus-per-node', type=int, choices=(1, 2, 4))
    parser.add_argument('--deadline', type=float, required=True, help='Unix seconds; already subtract allocation safety buffer')
    parser.add_argument('--eval-min-seconds', type=float, default=600)
    parser.add_argument('--train-min-seconds', type=float, default=120)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run inside a verified live Slurm compute step')
    import torch
    visible_required = args.gpus_per_node or args.training_gpus // args.nodes
    if not torch.cuda.is_available() or torch.cuda.device_count() < visible_required:
        raise RuntimeError('Requested physical GPUs are not visible in this compute step')
    try:
        Runner(args).run_foreground()
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode if error.returncode > 0 else 128 - error.returncode) from error


if __name__ == '__main__':
    main()
