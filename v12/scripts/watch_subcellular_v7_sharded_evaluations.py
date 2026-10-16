"""Work-stealing V7 checkpoint evaluator for independent one-GPU allocations.

This is intentionally separate from the legacy serial queue.  Training must be
complete and the legacy queue must be paused before workers start.  Workers use
one advisory lease per route/step, so staggered allocations can safely claim
the next unfinished checkpoint without assigning work to a pending job.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from run_subcellular_v7 import ROOT, atomic_json
from watch_subcellular_v7_evaluations import (
    EvaluationTask,
    fair_order,
    make_runner,
    mode_ready,
)


SCHEMA = "subcellular_v7_sharded_evaluation_v1"


def terminate(process):
    signals = []
    for sig, timeout in ((signal.SIGTERM, 15), (signal.SIGKILL, None)):
        if process.poll() is None:
            try:
                os.killpg(process.pid, sig)
                signals.append(sig.name)
            except ProcessLookupError:
                pass
        try:
            return process.wait(timeout=timeout), signals
        except subprocess.TimeoutExpired:
            continue
    raise RuntimeError("Unable to terminate evaluator process group")


class Worker:
    def __init__(self, runners, coordination, legacy_queue, worker_id, deadline,
                 poll_seconds, eval_min_seconds):
        self.runners = runners
        self.coordination = coordination
        self.legacy_queue = legacy_queue
        self.worker_id = worker_id
        self.deadline = deadline
        self.poll_seconds = poll_seconds
        self.eval_min_seconds = eval_min_seconds
        self.worker_lock = None
        self.legacy_evaluated = {}
        self.published_steps = {}
        self.status_path = coordination / "workers" / f"{worker_id}.json"
        self.state = dict(schema=SCHEMA, worker_id=worker_id,
                          jobid=os.environ.get("SLURM_JOB_ID"),
                          hostname=os.uname().nodename, pid=os.getpid(),
                          deadline=deadline, started_at=time.time(), active=None)

    @property
    def identity(self):
        return dict(schema=SCHEMA, routes={route: dict(
            config=str(runner.config_path), controller_identity=runner.identity,
            published_steps=self.published_steps[route],
            legacy_evaluated_steps=sorted(self.legacy_evaluated[route]))
            for route, runner in sorted(self.runners.items())})

    def publish(self, status, **extra):
        self.state.update(status=status, updated_at=time.time(), **extra)
        atomic_json(self.status_path, self.state)
        print("[v7-shard]", json.dumps(dict(
            worker=self.worker_id, status=status, active=self.state.get("active"),
            reason=self.state.get("reason")), allow_nan=False), flush=True)

    def prepare(self):
        self.coordination.mkdir(parents=True, exist_ok=True)
        for name in ("workers", "logs"):
            (self.coordination / name).mkdir(exist_ok=True)
        self.verify_runs()
        self.verify_handoff()
        owner = (self.coordination / "identity.lock").open("a")
        try:
            fcntl.flock(owner, fcntl.LOCK_EX)
            manifest = self.coordination / "identity.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != self.identity:
                    raise RuntimeError("Sharded evaluation identity mismatch")
            else:
                atomic_json(manifest, self.identity)
        finally:
            owner.close()
        self.worker_lock = (self.coordination / "workers" / f"{self.worker_id}.lock").open("a")
        fcntl.flock(self.worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.publish("ready")

    def verify_runs(self):
        for route, runner in self.runners.items():
            if not mode_ready(runner):
                raise RuntimeError(f"{route}: training-only execution mode is absent")
            manifest = runner.control / "identity.json"
            if not manifest.is_file() or json.loads(manifest.read_text()) != runner.identity:
                raise RuntimeError(f"{route}: controller identity mismatch")
            status = json.loads((runner.control / "status.json").read_text())
            if (status.get("identity") != runner.identity
                    or status.get("status") != "training_complete"):
                raise RuntimeError(f"{route}: invalid training controller status")
            progress, total = runner.progress()
            if not progress.get("complete") or progress.get("step") != total:
                raise RuntimeError(f"{route}: training is not complete")
            checkpoints = sorted((runner.run / "checkpoints").glob("step_*.pt"))
            steps = [int(path.stem.removeprefix("step_")) for path in checkpoints]
            required = {0, *range(100, total + 1, 100), total}
            if len(steps) != len(set(steps)) or not required.issubset(steps):
                raise RuntimeError(f"{route}: incomplete or duplicate checkpoint archive")
            self.published_steps[route] = steps

    def verify_handoff(self):
        """Require the legacy serial queue to be durably paused and unlocked."""
        marker = self.legacy_queue / "PAUSE_REQUESTED"
        status_path = self.legacy_queue / "status.json"
        if not marker.is_file() or not status_path.is_file():
            raise RuntimeError("Legacy evaluation queue lacks its pause handoff evidence")
        status = json.loads(status_path.read_text())
        if status.get("status") != "paused" or status.get("active") is not None:
            raise RuntimeError("Legacy evaluation queue is not fully paused")
        for route, runner in self.runners.items():
            route_status = status.get("routes", {}).get(route, {})
            evaluated = route_status.get("evaluated_steps")
            if (not isinstance(evaluated, list) or any(
                    isinstance(step, bool) or not isinstance(step, int) for step in evaluated)
                    or not set(evaluated).issubset(self.published_steps[route])):
                raise RuntimeError(f"{route}: invalid legacy evaluated-step snapshot")
            for step in evaluated:
                metrics = runner.run / "downstream_validation" / f"step_{step:08d}" / "metrics.json"
                if not metrics.is_file():
                    raise RuntimeError(f"{route}: legacy completed metric is absent: {step}")
            self.legacy_evaluated[route] = set(evaluated)
            directory = runner.run.with_name(runner.run.name + ".evaluation")
            directory.mkdir(exist_ok=True)
            handle = (directory / "owner.lock").open("a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError(f"{route}: legacy route lease is still held") from error
            handle.close()

    def completion_path(self, task):
        directory = self.runners[task.route].run.with_name(
            self.runners[task.route].run.name + ".evaluation") / "sharded_completed"
        directory.mkdir(exist_ok=True)
        return directory / f"step_{task.step:08d}.json"

    def blocked_path(self, task):
        directory = self.runners[task.route].run.with_name(
            self.runners[task.route].run.name + ".evaluation") / "sharded_blocked"
        directory.mkdir(exist_ok=True)
        return directory / f"step_{task.step:08d}.json"

    def task_done(self, task):
        if task.step in self.legacy_evaluated[task.route]:
            return True
        marker = self.completion_path(task)
        if not marker.is_file():
            return False
        payload = json.loads(marker.read_text())
        runner = self.runners[task.route]
        expected = dict(schema=SCHEMA, route=task.route, step=task.step,
                        checkpoint=str(task.checkpoint), config_sha256=runner.fingerprint)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Invalid sharded completion marker: {marker}")
        if not (runner.run / "downstream_validation" / f"step_{task.step:08d}" / "metrics.json").is_file():
            raise RuntimeError(f"Completion marker lacks metrics: {marker}")
        return True

    def pending(self):
        self.verify_handoff()
        tasks = []
        for route, runner in self.runners.items():
            for step in self.published_steps[route]:
                task = EvaluationTask(
                    route, step, runner.run / "checkpoints" / f"step_{step:08d}.pt")
                if not self.task_done(task):
                    tasks.append(task)
        return fair_order(tasks)

    def try_claim(self, task):
        directory = self.runners[task.route].run.with_name(
            self.runners[task.route].run.name + ".evaluation") / "task_locks"
        directory.mkdir(exist_ok=True)
        path = directory / f"step_{task.step:08d}.lock"
        handle = path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        try:
            if self.blocked_path(task).exists() or self.task_done(task):
                handle.close()
                return None
        except BaseException:
            handle.close()
            raise
        return handle

    def evaluate(self, task, lease):
        runner = self.runners[task.route]
        command = runner.evaluation_command(task.step, task.checkpoint)
        tag = f"{task.route}_step_{task.step:08d}_{self.worker_id}_{time.time_ns()}"
        log_path = self.coordination / "logs" / f"{tag}.log"
        record_path = log_path.with_suffix(".json")
        entry = dict(route=task.route, step=task.step, checkpoint=str(task.checkpoint),
                     command=command, cwd=str(ROOT), log=str(log_path),
                     worker_id=self.worker_id, started_at=time.time(),
                     deadline=self.deadline, status="starting")
        self.publish("evaluating", active=entry)
        process = None
        deadline_interrupted = False
        try:
            with log_path.open("x", buffering=1) as log:
                process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                           stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=(lease.fileno(),))
                entry.update(pid=process.pid, process_group=process.pid, status="running")
                self.publish("evaluating", active=entry)
                while True:
                    remaining = self.deadline - time.time()
                    if remaining <= 0:
                        deadline_interrupted = True
                        entry["returncode"], entry["termination_signals"] = terminate(process)
                        entry["status"] = "deadline_interrupted"
                        break
                    try:
                        entry["returncode"] = process.wait(timeout=min(remaining, self.poll_seconds, 60))
                        break
                    except subprocess.TimeoutExpired:
                        continue
            if deadline_interrupted:
                return False
            if entry["returncode"] != 0:
                entry["status"] = "failed"
                raise subprocess.CalledProcessError(entry["returncode"], command)
            if not runner.evaluated(task.step):
                entry["status"] = "invalid_metrics"
                raise RuntimeError("Evaluator exited successfully without valid metrics")
            entry["status"] = "passed"
            atomic_json(self.completion_path(task), dict(
                schema=SCHEMA, route=task.route, step=task.step,
                checkpoint=str(task.checkpoint), config_sha256=runner.fingerprint,
                worker_id=self.worker_id, verified_at=time.time()))
            return True
        except BaseException as error:
            if process is not None and process.poll() is None:
                entry["returncode"], entry["termination_signals"] = terminate(process)
            entry["error"] = f"{type(error).__name__}: {error}"
            if not deadline_interrupted:
                atomic_json(self.blocked_path(task), entry)
            raise
        finally:
            entry["finished_at"] = time.time()
            atomic_json(record_path, entry)
            self.state["active"] = None
            lease.close()

    def run(self):
        self.prepare()
        try:
            while self.deadline - time.time() >= self.eval_min_seconds:
                tasks = self.pending()
                if not tasks:
                    self.publish("all_evaluated", active=None, reason=None)
                    return 0
                claimed = None
                for task in tasks:
                    lease = self.try_claim(task)
                    if lease is not None:
                        claimed = task, lease
                        break
                if claimed is None:
                    if tasks and all(self.blocked_path(task).exists() for task in tasks):
                        raise RuntimeError("All remaining evaluation tasks are blocked")
                    self.publish("waiting", active=None, reason="unfinished_tasks_owned_by_other_workers")
                    time.sleep(min(self.poll_seconds, max(0, self.deadline - time.time())))
                    continue
                if not self.evaluate(*claimed):
                    self.publish("paused", active=None, reason="deadline_interrupted")
                    return 0
            self.publish("paused", active=None, reason="evaluation_time_budget")
            return 0
        except BaseException as error:
            self.publish("failed", active=None, error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if self.worker_lock is not None:
                self.worker_lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-a", type=Path, required=True)
    parser.add_argument("--config-b", type=Path, required=True)
    parser.add_argument("--coordination-dir", type=Path, required=True)
    parser.add_argument("--legacy-queue-dir", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--eval-min-seconds", type=float, default=600)
    args = parser.parse_args()
    visible = [item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
    if (not re.fullmatch(r"[A-Za-z0-9_.-]+", args.worker_id)
            or not all(math.isfinite(value) for value in
                       (args.poll_seconds, args.eval_min_seconds, args.deadline))
            or args.poll_seconds <= 0 or args.eval_min_seconds <= 0
            or args.deadline - time.time() < args.eval_min_seconds
            or not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID")
            or len(visible) != 1):
        parser.error("Invalid worker ID, polling interval, or deadline")
    runners = {
        "A": make_runner(args.config_a.resolve(), args.deadline, args.eval_min_seconds),
        "B": make_runner(args.config_b.resolve(), args.deadline, args.eval_min_seconds),
    }
    worker = Worker(runners, args.coordination_dir.resolve(), args.legacy_queue_dir.resolve(),
                    args.worker_id, args.deadline, args.poll_seconds, args.eval_min_seconds)
    raise SystemExit(worker.run())


if __name__ == "__main__":
    main()
