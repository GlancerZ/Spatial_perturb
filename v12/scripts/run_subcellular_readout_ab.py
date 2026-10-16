"""Resumable single-GPU A/B coordinator; resource allocation belongs to the caller."""
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
ARMS = ('control', 'direct_niche')
COHORT = '0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc'
STOP_MARKERS = ('AB_PAUSE_REQUESTED', 'STOP_REQUESTED', 'PAUSE_REQUESTED')
STOP_PREFIX = 'readout_ab_runner:'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Runner:
    def __init__(self, args):
        self.run = args.run_root.resolve()
        self.start = args.start_step
        self.target = self.start + args.extra_updates
        self.deadline = float(os.environ.get('V2_TRAIN_DEADLINE', 'inf'))
        self.eval_deadline = float(os.environ.get('SPATIAL_EVAL_DEADLINE', str(self.deadline)))
        physical = torch.cuda.device_count()
        if physical == 1:
            self.mode = 'serial_logical_ranks'
            self.train_prefix = [sys.executable, '-u', 'scripts/train_subcellular_serial.py']
        elif physical >= 4:
            self.mode = 'ddp_four_gpus'
            self.train_prefix = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                                 '--nproc_per_node=4', 'scripts/train_subcellular_v2.py']
        else:
            raise ValueError(f'A/B runner requires one or at least four visible GPUs, found {physical}')
        self.child = None
        self.child_kind = None
        self.child_arm = None
        self.lock_fd = None
        self.headers = {}
        self.configs = {}
        self.fingerprints = {}
        self.state = dict(created=now(), pid=os.getpid(), jobid=os.environ.get('SLURM_JOB_ID'),
                          start_step=self.start, target_step=self.target,
                          physical_world_size=1 if physical == 1 else 4, logical_world_size=4,
                          execution_mode=self.mode,
                          interpretation='Fixed-budget readout A/B; validation diagnostics, no success claim.')
        if not self.run.is_dir():
            raise ValueError('Prepare both experiment forks before starting the runner')
        experiment = json.loads((self.run / 'experiment.json').read_text())
        if experiment['parent_step'] != self.start or experiment['target_step'] != self.target:
            raise ValueError('Requested start/target differs from the prepared experiment')
        previous_path = self.run / 'runner_state.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        self.state['execution_history'] = previous.get('execution_history', []) + [dict(
            started=now(), jobid=self.state['jobid'], mode=self.mode,
            physical_world_size=self.state['physical_world_size'])]
        for arm in ARMS:
            directory = self.run / arm
            if not (directory / 'fork_metadata.json').is_file():
                raise ValueError(f'{arm} is missing fork_metadata.json')
            cfg = yaml.safe_load((directory / 'config.yaml').read_text())
            if Path(cfg['output_dir']).resolve() != directory:
                raise ValueError(f'{arm} config output_dir does not identify its own fork')
            if cfg['model'].get('backbone_kind') != 'scgpt':
                raise ValueError('Only the ordinary scGPT experiment is authorized')
            self.configs[arm] = cfg
            self.fingerprints[arm] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
            metadata = json.loads((directory / 'fork_metadata.json').read_text())
            if (metadata['parent_step'] != self.start or metadata['target_step'] != self.target
                    or metadata['config_sha256'] != self.fingerprints[arm]
                    or experiment['arms'][arm]['config_sha256'] != self.fingerprints[arm]):
                raise ValueError(f'{arm} fork metadata disagrees with its configuration/budget')
            if not (directory / 'checkpoints' / f'step_{self.start:08d}.pt').is_file():
                raise ValueError(f'{arm} is missing the common starting checkpoint {self.start}')

    def update(self, status, **values):
        self.state.update(status=status, updated=now(), **values)
        atomic_json(self.run / 'runner_state.json', self.state)
        self.write_report()
        try:
            print('[readout-ab]', json.dumps(self.state), flush=True)
        except (BrokenPipeError, OSError):
            pass  # Console lifetime must not determine training lifetime.

    def progress(self, arm):
        path = self.run / arm / 'resume.pt'
        stamp = (path.stat().st_mtime_ns, path.stat().st_size)
        cached = self.headers.get(arm)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['config_sha256'] != self.fingerprints[arm] or saved['world_size'] != 4:
            raise ValueError(f'{arm} resume config or logical world size mismatch')
        step = int(saved['step'])
        if not self.start <= step <= self.target:
            raise ValueError(f'{arm} step {step} outside [{self.start}, {self.target}]')
        result = dict(step=step, complete=bool(saved.get('complete')),
                      epoch=int(saved['epoch']), micro_step=int(saved['micro_step']))
        del saved
        self.headers[arm] = stamp, result
        return result

    def stop_reason(self):
        for name in STOP_MARKERS:
            if (self.run / name).exists():
                return name
        for arm in ARMS:
            for name in ('STOP_REQUESTED', 'PAUSE_REQUESTED', 'STOP_EVALUATION'):
                if (self.run / arm / name).exists():
                    return f'{arm}/{name}'
        return None

    def metrics(self, arm, step):
        path = self.run / arm / 'downstream_validation' / f'step_{step:08d}' / 'metrics.json'
        if not path.exists():
            return None
        report = json.loads(path.read_text())
        if (report.get('step') != step or report.get('config_sha256') != self.fingerprints[arm]
                or report.get('cohort_sha256') != COHORT or report.get('test_donors_used') is not False
                or report.get('backbone_kind') != 'scgpt' or report.get('evaluation_version', 0) < 2
                or not report.get('masked_validation', {}).get('enabled')):
            raise ValueError(f'{arm} step {step}: incompatible/incomplete evaluation provenance')
        for row in report['results']:
            if not all(item['solver']['converged_gradient'] for item in row['grid']):
                raise ValueError(f'{arm} step {step}: unconverged probe')
        return report

    def published(self, arm):
        return sorted(int(path.stem.removeprefix('step_'))
                      for path in (self.run / arm / 'checkpoints').glob('step_*.pt'))

    def pending(self, arm):
        return [step for step in self.published(arm) if self.metrics(arm, step) is None]

    def archive_target(self, arm):
        target = self.run / arm / 'checkpoints' / f'step_{self.target:08d}.pt'
        if target.exists():
            return
        saved = torch.load(self.run / arm / 'resume.pt', map_location='cpu', weights_only=False)
        if int(saved['step']) != self.target or saved['config_sha256'] != self.fingerprints[arm]:
            raise ValueError('Target archive can only publish the matching saved optimizer boundary')
        payload = {key: value for key, value in saved.items() if key not in ('optimizer', 'rng_by_rank')}
        temporary = target.with_name(target.name + '.partial')
        torch.save(payload, temporary)
        temporary.replace(target)
        del saved, payload

    def write_report(self):
        by_arm = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for path in sorted((self.run / arm / 'downstream_validation').glob('step_*/metrics.json')):
                report = json.loads(path.read_text())
                by_arm[arm][int(report['step'])] = report
        paired = []
        lines = ['# 显式 niche 末端读出：短预算 A/B', '',
                 f'起点 {self.start}；每臂目标 {self.target}；状态：{self.state.get("status", "initializing")}。',
                 f'本次执行：{self.state.get("execution_mode", "unknown")}；'
                 f'实际 GPU 数：{self.state.get("physical_world_size", "unknown")}；逻辑 world size 保持 4。',
                 'control 保留原读出；direct_niche 仅增加显式 niche 残差读出。'
                 '同一固定供体验证集，不读取 test，不按结果自动宣布成功。', '',
                 '| Step | 任务 | Control best | Direct best | Control 固定C | Direct 固定C | 固定C差值 |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for step in sorted(set(by_arm['control']) & set(by_arm['direct_niche'])):
            for task, fixed_c in (('cell_type', .1), ('region', .01)):
                comparisons = {}
                for arm in ARMS:
                    # Validate every result that is used in the paired report.
                    report = self.metrics(arm, step)
                    row = next(r for r in report['results'] if r['embedding'] == 'joint' and r['task'] == task)
                    fixed = next(item for item in row['grid'] if item['C'] == fixed_c)
                    comparisons[arm] = dict(best=row['best'], fixed_C=fixed_c, fixed=fixed)
                a, b = comparisons['control'], comparisons['direct_niche']
                delta = b['fixed']['val_macro_f1'] - a['fixed']['val_macro_f1']
                donor_delta = {donor: score - a['fixed']['per_donor_macro_f1'][donor]
                               for donor, score in b['fixed']['per_donor_macro_f1'].items()}
                paired.append(dict(step=step, task=task, **comparisons,
                                   fixed_C_delta=delta, fixed_C_per_donor_delta=donor_delta))
                lines.append(f'| {step} | {task} | {a["best"]["val_macro_f1"]:.6f} | '
                             f'{b["best"]["val_macro_f1"]:.6f} | {a["fixed"]["val_macro_f1"]:.6f} | '
                             f'{b["fixed"]["val_macro_f1"]:.6f} | {delta:+.6f} |')
        lines += ['', '固定 C：细胞类型 0.1，区域 0.01。best C 仍按原验证协议选择。'
                  '需同时检查任务取舍与供体差异；这些是探索性对照，不是独立测试。']
        for execution in self.state.get('execution_history', []):
            lines.append(f'执行记录：{execution["started"]}，job={execution["jobid"]}，'
                         f'mode={execution["mode"]}，GPU={execution["physical_world_size"]}。')
        for row in paired:
            lines += ['', f'步骤 {row["step"]}，{row["task"]}，固定 C 的逐供体差值（direct−control）：',
                      ', '.join(f'{donor}: {delta:+.6f}' for donor, delta in row['fixed_C_per_donor_delta'].items())]
        atomic_json(self.run / 'paired_metrics.json', dict(start_step=self.start, target_step=self.target,
                    cohort_sha256=COHORT, test_donors_used=False, comparisons=paired))
        path = self.run / 'REPORT.zh.md'
        temporary = path.with_name(path.name + '.partial')
        temporary.write_text('\n'.join(lines) + '\n')
        temporary.replace(path)

    def terminate_eval(self):
        if self.child is None:
            return
        try:
            os.killpg(self.child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.child.wait()

    def execute(self, arm, kind, command):
        directory = self.run / arm
        logs = directory / 'logs'
        logs.mkdir(exist_ok=True)
        marker = directory / 'STOP_REQUESTED'
        self.child_kind, self.child_arm = kind, arm
        interrupted = None
        with (logs / f'ab_{kind}.log').open('a', buffering=1) as log:
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                          pass_fds=(self.lock_fd,), start_new_session=True)
            while self.child.poll() is None:
                interrupted = interrupted or self.stop_reason()
                if time.time() >= (self.deadline if kind == 'train' else self.eval_deadline):
                    interrupted = 'allocation_deadline'
                if interrupted:
                    if kind == 'train':
                        marker.write_text(f'{STOP_PREFIX} save completed update: {interrupted}\n')
                    else:
                        self.terminate_eval()
                        break
                time.sleep(10)
            code = self.child.wait()
            self.child = None
        if marker.exists() and marker.read_text().startswith(STOP_PREFIX):
            marker.unlink()
        if code and not (kind == 'eval' and interrupted):
            raise RuntimeError(f'{arm} {kind} failed with exit {code}; inspect {logs}')
        return interrupted

    def run_locked(self, lock):
        self.lock_fd = lock.fileno()
        # An inherited lock prevents a new coordinator while an old trainer
        # survives. Once acquired, our own stale stop markers are safe to clear.
        for arm in ARMS:
            for name in ('SERIAL_STOP_REQUESTED', 'STOP_REQUESTED'):
                marker = self.run / arm / name
                if marker.exists() and marker.read_text().startswith(STOP_PREFIX):
                    marker.unlink()
        while True:
            reason = self.stop_reason()
            if reason:
                self.update('paused', reason=reason)
                return
            progress = {arm: self.progress(arm) for arm in ARMS}
            self.state['progress'] = progress
            for arm in ARMS:
                if progress[arm]['step'] == self.target:
                    self.archive_target(arm)
            waiting = {arm: self.pending(arm) for arm in ARMS}
            if all(progress[arm]['step'] == self.target and not waiting[arm] for arm in ARMS):
                self.update('complete')
                atomic_json(self.run / 'AB_COMPLETE', dict(completed=now(), target_step=self.target,
                            progress=progress, all_published_checkpoints_evaluated=True))
                return
            # Published checkpoints are evaluated immediately after each
            # segment, before spending the card on another training segment.
            candidates = [arm for arm in ARMS if waiting[arm]]
            if not candidates:
                candidates = [arm for arm in ARMS if progress[arm]['step'] < self.target]
            arm = min(candidates, key=lambda name: (progress[name]['step'], ARMS.index(name)))
            next_deadline = self.eval_deadline if waiting[arm] else self.deadline
            if time.time() + (300 if waiting[arm] else 120) >= next_deadline:
                self.update('allocation_deadline', next_arm=arm, pending_evaluations=waiting)
                return
            if waiting[arm]:
                self.update('evaluating', arm=arm, pending_steps=waiting[arm])
                command = ['bash', 'scripts/watch_subcellular_v2_eval.sh', '--run-dir', str(self.run / arm),
                           '--batch-size', '2', '--cells-per-donor', '400', '--once']
                reason = self.execute(arm, 'eval', command)
                if reason:
                    self.update('paused' if reason != 'allocation_deadline' else 'allocation_deadline', reason=reason)
                    return
                if self.pending(arm):
                    if time.time() + 600 >= self.eval_deadline:
                        self.update('evaluation_deferred_allocation_deadline', arm=arm)
                        return
                    raise RuntimeError(f'{arm} one-pass evaluation left required checkpoints incomplete')
                self.update('evaluated', arm=arm)
                continue
            step = progress[arm]['step']
            if progress[arm]['complete']:
                raise RuntimeError(f'{arm} epoch ended before the requested A/B target')
            stop = min(self.target, (step // 100 + 1) * 100)
            self.update('training', arm=arm, from_step=step, requested_step=stop)
            reason = self.execute(arm, 'train', self.train_prefix + [
                                  '--config', str(self.run / arm / 'config.yaml'),
                                  '--run-update-limit', str(stop - step)])
            saved = self.progress(arm)
            after = saved['step']
            if after < step:
                raise RuntimeError(f'{arm} trainer moved its saved update backwards')
            self.state['progress'][arm] = saved
            self.update('saved', arm=arm, saved_step=after)
            # Model/optimizer initialization can consume the last available
            # time. A trainer may then save the unchanged boundary and exit 0
            # before its first update; this is an ordinary resumable stop.
            reason = reason or self.stop_reason()
            if reason is None and time.time() >= self.deadline:
                reason = 'allocation_deadline'
            if reason:
                self.update('paused' if reason != 'allocation_deadline' else 'allocation_deadline', reason=reason)
                return
            if after == step:
                raise RuntimeError(f'{arm} trainer exited without advancing its saved update')
            if after != stop:
                raise RuntimeError(f'{arm} saved unexpected step {after}, expected {stop}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--extra-updates', type=int, default=300)
    parser.add_argument('--start-step', type=int, default=3652)
    args = parser.parse_args()
    if args.extra_updates <= 0 or args.start_step < 0:
        parser.error('extra updates must be positive and start step nonnegative')
    os.chdir(ROOT)
    runner = Runner(args)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f'A/B coordinator received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    with (runner.run / 'experiment.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another A/B coordinator or inherited trainer owns experiment.lock')
        try:
            runner.run_locked(lock)
        except BaseException as error:
            marker = runner.run / 'PAUSE_REQUESTED'
            if not marker.exists():
                atomic_json(marker, dict(source='readout_ab_runner_failure', timestamp=now(),
                                        reason=str(error), parameters_changed=False))
            if runner.child is not None and runner.child.poll() is None:
                if runner.child_kind == 'train':
                    (runner.run / runner.child_arm / 'STOP_REQUESTED').write_text(
                        f'{STOP_PREFIX} coordinator failure; save complete update and stop\n')
                    runner.child.wait()
                else:
                    runner.terminate_eval()
            runner.update('action_required', error=str(error))
            raise


if __name__ == '__main__':
    main()
