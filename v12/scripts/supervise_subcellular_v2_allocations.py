"""Foreground Slurm allocation supervisor; Python standard library only.

This program performs scheduler/file checks on the login node. Project compute
is always launched through slurm_job_exec into the existing compute-node tmux
holder. It never cancels allocations and never adopts an unrelated user's job.
The initially supplied job is already executing: the initial action is monitor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TASK_FAMILY = 'subcellular_v2'
LOG_PREFIX = 'v2'
TMUX_PREFIX = 'subcell_v2'
TERMINAL = {'COMPLETED', 'CANCELLED', 'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
RECOVERABLE_INTERRUPTION = {'TIMEOUT', 'PREEMPTED', 'NODE_FAIL'}
INTERRUPTION_EXIT_CODES = {137, 143, -9, -15}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def scheduler(command):
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=45, check=False, cwd=ROOT)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed ({result.returncode}): {result.stderr.strip()}')
    return result.stdout


def queue_snapshot():
    # Always check all of our current allocations before requesting any more.
    output = scheduler(['squeue', '--me', '--noheader', '--format=%i|%T|%j'])
    jobs = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = line.strip().split('|')
        if len(fields) != 3:
            raise RuntimeError(f'Unexpected squeue output: {line!r}')
        jobid, state, name = fields
        jobs[jobid] = {'jobid': jobid, 'state': state, 'name': name}
    return jobs


def terminal_state(jobid):
    output = scheduler(['sacct', '--noheader', '--allocations', '--jobs', str(jobid),
                        '--format=JobIDRaw,State,ExitCode', '--parsable2'])
    for line in output.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 3 and fields[0] == str(jobid):
            state = fields[1].split()[0].rstrip('+')
            return {'state': state, 'slurm_exit_code': fields[2], 'terminal': state in TERMINAL}
    # Accounting can lag squeue. An absent record is not evidence of completion.
    return {'state': 'ACCOUNTING_PENDING', 'terminal': False}


def completed_evaluation(path, step):
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (report.get('step') == step and report.get('evaluation_version', 0) >= 2
            and bool(report.get('config_sha256'))
            and report.get('masked_validation', {}).get('enabled', False))


def pending_checkpoints(run):
    points = {0: run / 'initial.pt'} if (run / 'initial.pt').exists() else {}
    for path in (run / 'checkpoints').glob('step_*.pt'):
        match = re.fullmatch(r'step_(\d+)\.pt', path.name)
        if match:
            points[int(match.group(1))] = path
    pending = [step for step in sorted(points) if not completed_evaluation(
        run / 'downstream_validation' / f'step_{step:08d}' / 'metrics.json', step)]
    return points, pending


def allocation_command(role, job_name):
    if role == 'train':
        gpu, cpu, memory, duration, hold = 4, 64, '460G', '04:00:00', 14400
    else:
        gpu, cpu, memory, duration, hold = 1, 16, '120G', '06:00:00', 21600
    return ['sbatch', '--parsable', '--account=aip-junding', '--partition=gpubase_bygpu_b2',
            f'--job-name={job_name}', '--nodes=1', f'--gpus-per-node={gpu}',
            f'--cpus-per-task={cpu}', f'--mem={memory}', f'--time={duration}',
            '--mail-type=ALL', '--mail-user=bowen.zhao@mail.mcgill.ca',
            f'--wrap=sleep {hold}']


def launch_command(role, jobid):
    return [sys.executable, 'tools/slurm_job_exec.py', '--jobid', str(jobid), '--cwd', str(ROOT),
            '--env', 'source scripts/vulcan_compartment_env.sh',
            '--command', f'bash scripts/hold_{TASK_FAMILY}_task.sh {role}']


def exit_status(role, jobid):
    path = ROOT / 'logs' / f'{LOG_PREFIX}_{role}_{jobid}.exit'
    if not path.exists():
        return None
    try:
        value = int(path.read_text().strip())
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Invalid task exit record {path}: {error}') from error
    return value


def terminal_recovery_allowed(final_state, task_exit, holder_exit, resume_possible):
    """Scheduler evidence, not a signal exit code alone, authorizes recovery."""
    if task_exit not in (None, 0) and task_exit not in INTERRUPTION_EXIT_CODES:
        return False  # Python/algorithm errors never become timeout retries.
    if final_state in RECOVERABLE_INTERRUPTION:
        return task_exit == 0 or resume_possible
    return final_state == 'COMPLETED' and task_exit == 0 and holder_exit in (None, 0)


def compute_session_present(role, jobid):
    """Read-only tmux reconciliation inside the existing allocation."""
    command = launch_command(role, jobid)
    command[-1] = (f'if tmux -L {TMUX_PREFIX}_{role}_{jobid} has-session -t {role} 2>/dev/null; '
                   'then printf "V2_TASK_RUNNING\\n"; else printf "V2_TASK_ABSENT\\n"; fi')
    output = scheduler(command)
    markers = [line.strip() for line in output.splitlines()
               if line.strip() in ('V2_TASK_RUNNING', 'V2_TASK_ABSENT')]
    if len(markers) != 1:
        raise RuntimeError('Cannot reconcile the compute-node tmux session')
    return markers[0] == 'V2_TASK_RUNNING'


class Supervisor:
    def __init__(self, args):
        self.args = args
        self.run = args.run_dir.resolve()
        self.path = self.run / f'supervisor_{args.role}.json'
        self.log_path = self.run / f'supervisor_{args.role}.log'
        run_hash = hashlib.sha256(str(self.run).encode()).hexdigest()[:10]
        self.job_name = f'subcellv2_{args.role}_{run_hash}'
        self.child = None
        self.child_log = None
        self.console_enabled = not getattr(args, 'quiet_console', False)
        self.reconciled_jobs = set()
        self.absent_session_checks = {}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get('task_family', 'subcellular_v2') != TASK_FAMILY:
                raise RuntimeError('Existing supervisor state uses a different task family')
            if (self.state.get('role') != args.role or self.state.get('run_dir') != str(self.run)
                    or str(args.jobid) not in self.state.get('known_job_ids', [])):
                raise RuntimeError('Existing supervisor state belongs to another run/initial allocation')
            if self.state.get('status') == 'action_required' or self.state.get('submission_pending'):
                raise RuntimeError('Prior supervisor requires inspection; resolve its state before restarting')
        else:
            self.state = dict(role=args.role, run_dir=str(self.run), created=now(),
                              initial_jobid=str(args.jobid), current_jobid=str(args.jobid),
                              known_job_ids=[str(args.jobid)], new_allocations=0,
                              jobs={str(args.jobid): {'source': 'initial_launch_authorized' if getattr(args, 'launch_initial', False) else 'initial_already_running',
                                                     'launch_attempted': not getattr(args, 'launch_initial', False)}})
        self.state.update(pid=os.getpid(), max_new_allocations=args.max_new_allocations, task_family=TASK_FAMILY,
                          job_name=self.job_name)

    def update(self, status, **fields):
        previous = (self.state.get('status'), self.state.get('current_jobid'))
        self.state.update(status=status, updated=now(), **fields)
        atomic_json(self.path, self.state)
        current = (status, self.state.get('current_jobid'))
        if current != previous or status in {'action_required', 'allocation_submitted', 'compute_launched'}:
            line = json.dumps(dict(timestamp=now(), role=self.args.role, status=status,
                                   jobid=self.state.get('current_jobid'), **fields))
            # State and durable logs must succeed independently of an SSH/stdout
            # consumer. Only the optional console write may be discarded.
            with self.log_path.open('a') as handle:
                handle.write(line + '\n')
            if self.console_enabled:
                try:
                    print(line, flush=True)
                except (BrokenPipeError, ValueError):
                    # ValueError is raised by a closed TextIO stream. Disable
                    # further console writes without changing scheduler state.
                    self.console_enabled = False
                except OSError as error:
                    if error.errno != errno.EBADF:
                        raise
                    self.console_enabled = False

    def stopping_reason(self):
        if TASK_FAMILY == 'subcellular_readout_ab':
            for name in ('AB_COMPLETE', 'AB_PAUSE_REQUESTED'):
                if (self.run / name).exists():
                    return name
            for arm in ('control', 'direct_niche'):
                for name in ('STOP_REQUESTED', 'PAUSE_REQUESTED', 'STOP_EVALUATION'):
                    marker = self.run / arm / name
                    if marker.exists():
                        # The coordinator owns transient deadline markers and
                        # removes them once the completed update is saved.
                        if name == 'STOP_REQUESTED':
                            try:
                                owned = marker.read_text().startswith('readout_ab_runner:')
                            except FileNotFoundError:
                                continue  # The coordinator cleared its completed deadline marker.
                            if owned:
                                continue
                        return f'{arm}/{name}'
        for name in ('STOP_REQUESTED', 'PAUSE_REQUESTED'):
            if (self.run / name).exists():
                return name
        if self.args.role == 'eval' and (self.run / 'STOP_EVALUATION').exists():
            return 'STOP_EVALUATION'
        if (self.run / 'TRAIN_COMPLETE').exists():
            if self.args.role == 'train':
                return 'TRAIN_COMPLETE'
            points, pending = pending_checkpoints(self.run)
            if points and not pending:
                return 'TRAIN_COMPLETE_AND_ALL_CHECKPOINTS_EVALUATED'
        return None

    def launch(self, jobid):
        info = self.state['jobs'][jobid]
        # Persist before launch so a supervisor restart cannot accidentally start
        # a duplicate workload. An interrupted ambiguous launch needs inspection.
        info['launch_attempted'] = True
        info['launch_time'] = now()
        self.update('launching_compute')
        log_path = ROOT / 'logs' / f'v2_supervisor_{self.args.role}_launch_{jobid}.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.child_log = log_path.open('a', buffering=1)
        self.child = subprocess.Popen(launch_command(self.args.role, jobid), cwd=ROOT,
                                      stdout=self.child_log, stderr=subprocess.STDOUT)
        self.reconciled_jobs.add(jobid)
        info['holder_process_pid'] = self.child.pid
        info['launch_log'] = str(log_path)
        self.update('compute_launched', launch_log=str(log_path))

    def renew(self, jobs):
        # Only adopt allocations already registered for this exact role/run.
        # Other RUNNING jobs belong to their existing workloads and are retained.
        owned_active = [jobid for jobid in self.state['known_job_ids'] if jobid in jobs]
        if owned_active:
            if len(owned_active) > 1:
                self.update('action_required', reason='Multiple registered allocations are active; refusing ambiguous reuse')
                return False
            jobid = owned_active[0]
            self.update('reusing_registered_allocation', current_jobid=jobid)
            return True
        # A matching name not in our registry could indicate a crash after sbatch
        # accepted the request but before the job ID was saved. Never submit twice.
        unregistered = [jobid for jobid, job in jobs.items() if job['name'] == self.job_name]
        if unregistered:
            self.update('action_required', reason='Unregistered matching-name allocation; inspect before adopting',
                        matching_jobids=unregistered)
            return False
        # Explicitly authorized, previously allocated idle jobs may be reused
        # after the current allocation ends. Never adopt a busy or pending job.
        for candidate in getattr(self.args, 'reuse_idle_job', []):
            if candidate not in jobs or jobs[candidate]['state'] != 'RUNNING':
                continue
            steps = scheduler(['squeue', '--steps', '--noheader', '--jobs', candidate,
                               '--format=%i']).splitlines()
            active = [step.strip() for step in steps if step.strip()
                      and step.strip() not in (candidate + '.batch', candidate + '.extern')]
            if active:
                continue
            reason = self.stopping_reason()
            if reason:
                self.update('finished', reason=reason)
                return False
            self.state['known_job_ids'].append(candidate)
            self.state['jobs'][candidate] = dict(source='explicitly_authorized_idle_reuse',
                                                registered=now(), launch_attempted=False)
            self.update('reusing_authorized_idle_allocation', current_jobid=candidate)
            return True
        if self.state['new_allocations'] >= self.args.max_new_allocations:
            self.update('action_required', reason='Authorized automatic renewal bound exhausted')
            return False
        # Re-check stop/completion flags immediately before any scheduler mutation.
        reason = self.stopping_reason()
        if reason:
            self.update('finished', reason=reason)
            return False
        command = allocation_command(self.args.role, self.job_name)
        self.update('submitting_allocation', submission_command=command, submission_pending=True)
        output = scheduler(command).strip()
        jobid = output.split(';')[0]
        if not re.fullmatch(r'\d+', jobid):
            self.update('action_required', reason='Cannot parse sbatch response; do not resubmit without inspection',
                        submission_response=output)
            return False
        self.state['new_allocations'] += 1
        self.state['known_job_ids'].append(jobid)
        self.state['jobs'][jobid] = {'source': 'renewal', 'submitted': now(), 'launch_attempted': False}
        self.update('allocation_submitted', current_jobid=jobid, submission_pending=False)
        return True

    def run_foreground(self):
        self.update('monitoring_existing_allocation')
        while True:
            reason = self.stopping_reason()
            if reason:
                self.update('finished', reason=reason)
                return
            jobid = self.state['current_jobid']
            try:
                task_exit = exit_status(self.args.role, jobid)
                if self.child is not None and self.child.poll() is not None:
                    child_exit = self.child.returncode
                    self.child = None
                    self.child_log.close()
                    self.child_log = None
                    self.state['jobs'][jobid]['holder_exit_code'] = child_exit
                jobs = queue_snapshot()
                self.state['visible_own_allocations'] = list(jobs.values())
                info = self.state['jobs'][jobid]
                holder_exit = info.get('holder_exit_code')
                if jobid in jobs:
                    state = jobs[jobid]['state']
                    if task_exit not in (None, 0) or holder_exit not in (None, 0):
                        # Slurm may still display COMPLETING just after a walltime
                        # kill. Wait briefly for terminal evidence; never resubmit
                        # based merely on a 137/143 or failed srun return code.
                        failure_began = info.setdefault('failure_observed_at', time.time())
                        algorithm_failure = task_exit not in (None, 0) and task_exit not in INTERRUPTION_EXIT_CODES
                        if algorithm_failure or (state != 'COMPLETING' and time.time() - failure_began >= 120):
                            self.update('action_required', reason='Task/holder failed while allocation is active; no renewal',
                                        task_exit_code=task_exit, holder_exit_code=holder_exit, slurm_state=state)
                            return
                        self.update('waiting_failure_scheduler_confirmation', task_exit_code=task_exit,
                                    holder_exit_code=holder_exit, slurm_state=state)
                        time.sleep(self.args.poll_seconds)
                        continue
                    if state == 'RUNNING' and not info['launch_attempted']:
                        self.launch(jobid)
                    elif (state == 'RUNNING' and task_exit is None and self.child is None
                          and jobid not in self.reconciled_jobs):
                        # On restart, a persisted launch_attempted flag does not
                        # prove Popen reached tmux. Confirm the actual session;
                        # never relaunch automatically when that is ambiguous.
                        if compute_session_present(self.args.role, jobid):
                            self.reconciled_jobs.add(jobid)
                        else:
                            self.absent_session_checks[jobid] = self.absent_session_checks.get(jobid, 0) + 1
                            task_exit = exit_status(self.args.role, jobid)
                            if task_exit is None and self.absent_session_checks[jobid] >= 2:
                                self.update('action_required', reason='Launch was recorded but two compute-node checks found no tmux task or exit marker; refusing duplicate launch')
                                return
                            self.update('reconciling_ambiguous_launch', absent_checks=self.absent_session_checks[jobid])
                            time.sleep(self.args.poll_seconds)
                            continue
                    self.update('waiting_allocation_end' if task_exit == 0 else 'allocation_' + state.lower(),
                                task_exit_code=task_exit, slurm_state=state)
                else:
                    final = terminal_state(jobid)
                    if not final['terminal']:
                        self.update('waiting_accounting_confirmation', slurm_state=final['state'])
                    else:
                        self.state['jobs'][jobid]['terminal'] = final
                        resume_possible = self.args.role == 'eval' or (self.run / 'resume.pt').exists()
                        if not terminal_recovery_allowed(final['state'], task_exit, holder_exit, resume_possible):
                            self.update('action_required', reason='Terminal allocation is not a confirmed safe/resumable completion; no renewal',
                                        slurm_terminal=final, task_exit_code=task_exit, holder_exit_code=holder_exit)
                            return
                        self.update('allocation_ended', slurm_terminal=final, task_exit_code=task_exit,
                                    holder_exit_code=holder_exit,
                                    recovery_reason='confirmed_scheduler_interruption' if final['state'] in RECOVERABLE_INTERRUPTION else 'clean_completion')
                        # queue_snapshot just succeeded; no submission is possible
                        # while the previous allocation still appears R/PD/etc.
                        if not self.renew(jobs):
                            return
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                # A scheduler failure never means the allocation has ended.
                # An uncertain sbatch result requires manual inspection, since it
                # might have submitted a job even when its response was lost.
                if self.state.get('status') == 'submitting_allocation':
                    self.update('action_required', reason='Allocation submission result uncertain; refusing retry',
                                error=str(error))
                    return
                self.update('scheduler_or_monitor_error', error=str(error))
            time.sleep(self.args.poll_seconds)


def main():
    global TASK_FAMILY, LOG_PREFIX, TMUX_PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['train', 'eval'], required=True)
    parser.add_argument('--jobid', required=True, help='Existing already-running allocation to monitor first')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--max-new-allocations', type=int, default=3)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--quiet-console', action='store_true',
                        help='Suppress live console updates; keep atomic state and durable log writes')
    parser.add_argument('--reuse-idle-job', action='append', default=[],
                        help='Explicitly authorized job ID to reuse only if RUNNING with no compute steps')
    parser.add_argument('--launch-initial', action='store_true',
                        help='Launch the explicitly supplied idle/pending allocation once RUNNING')
    parser.add_argument('--task-family', choices=['subcellular_v2', 'subcellular_scgpt', 'subcellular_readout_ab', 'subcellular_v3'], default='subcellular_v2')
    args = parser.parse_args()
    TASK_FAMILY = args.task_family
    if TASK_FAMILY == 'subcellular_scgpt':
        LOG_PREFIX, TMUX_PREFIX = 'scgpt', 'subcell_scgpt'
    elif TASK_FAMILY == 'subcellular_readout_ab':
        LOG_PREFIX, TMUX_PREFIX = 'readout_ab', 'subcell_readout_ab'
    elif TASK_FAMILY == 'subcellular_v3':
        LOG_PREFIX, TMUX_PREFIX = 'v3', 'subcell_v3'
    if (not args.jobid.isdigit() or any(not job.isdigit() for job in args.reuse_idle_job)
            or args.max_new_allocations < 0 or args.poll_seconds < 1):
        parser.error('jobid must be numeric, renewal bound nonnegative, poll interval >=1 second')
    run = args.run_dir.resolve()
    if args.dry_run:
        run_hash = hashlib.sha256(str(run).encode()).hexdigest()[:10]
        report = dict(dry_run=True, task_family=TASK_FAMILY, initial_action='monitor_only', initial_jobid=args.jobid,
                      role=args.role, run_dir=str(run), max_new_allocations=args.max_new_allocations,
                      renewal_command=allocation_command(args.role, f'subcellv2_{args.role}_{run_hash}'),
                      compute_launch_command=launch_command(args.role, 'NEW_JOB_ID'),
                      stop_markers=['STOP_REQUESTED', 'PAUSE_REQUESTED', 'TRAIN_COMPLETE'],
                      requirement='Prior allocation must be confirmed terminal; re-check squeue --me; never cancel allocations')
        try:
            report['read_only_queue_snapshot'] = queue_snapshot()
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            report['queue_read_error'] = str(error)
        print(json.dumps(report, indent=2))
        return
    run.mkdir(parents=True, exist_ok=True)
    with (run / f'supervisor_{args.role}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(f'Another {args.role} allocation supervisor already holds the lock') from error
        supervisor = Supervisor(args)
        try:
            supervisor.run_foreground()
        except KeyboardInterrupt:
            supervisor.update('supervisor_interrupted', reason='Foreground supervisor interrupted; allocations preserved')


if __name__ == '__main__':
    main()
