"""Compute-node coordinator: serial four-rank updates, eval, then four-GPU handoff."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary-jobid', required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/subcellular_scgpt_human_4gpu.yaml'))
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/subcellular_scgpt_human_4gpu'))
    args = parser.parse_args()
    if not args.primary_jobid.isdigit():
        parser.error('primary job ID must be numeric')
    os.chdir(ROOT)
    run = args.run_dir.resolve()
    state_path = run / 'single_gpu_bridge.json'
    stop_path = run / 'SERIAL_STOP_REQUESTED'
    deadline = float(os.environ['V2_TRAIN_DEADLINE'])
    state = dict(created=now(), physical_jobid=os.environ['SLURM_JOB_ID'],
                 primary_jobid=args.primary_jobid, physical_world_size=1,
                 logical_world_size=4, pid=os.getpid())

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Bridge received signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)

    def update(status, **values):
        state.update(status=status, updated=now(), **values)
        write_json(state_path, state)
        print('[bridge]', json.dumps(state), flush=True)

    def primary_ready():
        result = subprocess.run(['squeue', '--me', '--noheader', '--format=%i|%T'],
                                text=True, capture_output=True, timeout=30, check=True)
        for line in result.stdout.splitlines():
            jobid, status = line.strip().split('|')
            if jobid == args.primary_jobid:
                return status == 'RUNNING'
        return False

    def progress():
        saved = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
        result = int(saved['step']), bool(saved.get('complete'))
        del saved
        return result

    def published_evaluated(step):
        checkpoint = run / 'checkpoints' / f'step_{step:08d}.pt'
        if not checkpoint.exists():
            return True
        path = run / 'downstream_validation' / f'step_{step:08d}' / 'metrics.json'
        if not path.exists():
            return False
        report = json.loads(path.read_text())
        return (report.get('step') == step and report.get('evaluation_version', 0) >= 2
                and bool(report.get('config_sha256'))
                and report.get('masked_validation', {}).get('enabled', False))

    if (run / 'STOP_EVALUATION').exists() or stop_path.exists():
        raise RuntimeError('Inspect existing evaluation/serial stop marker before starting bridge')
    with (run / 'training.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            update('handoff_training_lock_owned')
            return
        child = None
        child_kind = None
        try:
            while time.time() + 300 < deadline:
                if any((run / name).exists() for name in ('STOP_REQUESTED', 'PAUSE_REQUESTED')):
                    update('training_guardrail_stop')
                    return
                try:
                    ready = primary_ready()
                except (subprocess.SubprocessError, OSError) as error:
                    update('waiting_scheduler_confirmation', detail=str(error))
                    time.sleep(15)
                    continue
                if ready:
                    update('handoff_four_gpu_ready')
                    return
                step, complete = progress()
                if complete:
                    update('training_complete', step=step)
                    return
                backup = run / 'resume_before_single_gpu_bridge.pt'
                if not backup.exists():
                    # Atomic checkpoint replacement leaves this original inode
                    # intact, without first copying a multi-GB optimizer state.
                    os.link(run / 'resume.pt', backup)
                updates = 100 - step % 100
                update('training', start_step=step, next_evaluation_step=step + updates)
                command = [sys.executable, '-u', 'scripts/train_subcellular_serial.py',
                           '--config', str(args.config), '--run-update-limit', str(updates)]
                with (ROOT / 'logs/scgpt_serial_train.log').open('a', buffering=1) as log:
                    child_kind = 'train'
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock.fileno(),))
                    handoff = False
                    while child.poll() is None:
                        try:
                            if primary_ready() or time.time() >= deadline:
                                handoff = True
                                stop_path.write_text('single_gpu_bridge: save completed update and hand off\n')
                        except (subprocess.SubprocessError, OSError) as error:
                            # A failed scheduler query is not evidence of an idle/ready job.
                            update('training_scheduler_check_unavailable', detail=str(error))
                        time.sleep(15)
                    if child.returncode:
                        raise RuntimeError(f'Serial trainer failed with exit code {child.returncode}')
                    child = None
                step, complete = progress()
                update('saved', step=step, complete=complete)
                # Handoff was checked while the child was active; another
                # readiness query occurs at the next loop boundary.
                if handoff:
                    update('handoff_saved', step=step)
                    return
                if time.time() + 300 >= deadline:
                    update('allocation_deadline_saved', step=step)
                    return
                update('evaluating', step=step)
                child_kind = 'eval'
                child = subprocess.Popen(['bash', 'scripts/watch_subcellular_v2_eval.sh',
                    '--run-dir', str(run), '--batch-size', '2', '--cells-per-donor', '400',
                    '--neighbor-shuffle-every', '500', '--once'], start_new_session=True)
                result_code = child.wait()
                child = None
                if not result_code and not published_evaluated(step) and time.time() + 600 >= deadline:
                    update('evaluation_deferred_allocation_deadline', step=step)
                    return
                if result_code or not published_evaluated(step):
                    raise RuntimeError(f'Checkpoint {step} evaluation did not complete; inspect eval logs')
            update('allocation_deadline')
        except BaseException as error:
            # Do not let the four-GPU writer proceed after an unresolved failure.
            marker = run / 'PAUSE_REQUESTED'
            if not marker.exists():
                write_json(marker, dict(source='single_gpu_bridge_failure', timestamp=now(),
                                        reason=str(error), parameters_changed=False))
            if child is not None and child.poll() is None:
                if child_kind == 'train':
                    stop_path.write_text('single_gpu_bridge: coordinator failure; save and stop\n')
                    child.wait()  # Retain the lock until the checkpoint writer stops.
                else:
                    # Evaluations are reproducible and can restart; terminate
                    # this owned process group, including its GPU subprocess.
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            update('action_required', error=str(error))
            raise
        finally:
            if stop_path.exists() and stop_path.read_text().startswith('single_gpu_bridge:'):
                stop_path.unlink()


if __name__ == '__main__':
    main()
