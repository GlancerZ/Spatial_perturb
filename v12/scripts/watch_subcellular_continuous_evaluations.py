"""Alternate one checkpoint at a time between the V6 and V7 evaluation queues.

This process runs on one dedicated GPU.  V6 remains eligible until its finite
archive is complete; V7 becomes eligible as soon as either training-only route
publishes checkpoints.  Alternating generations prevents the historical V6
backlog from delaying early V7 stability measurements indefinitely.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

from run_subcellular_v6 import ROOT
from watch_subcellular_v6_evaluations import (
    EvaluationQueue as V6Queue,
    QueuePause as V6QueuePause,
    make_runner as make_v6_runner,
)
from watch_subcellular_v7_evaluations import (
    EvaluationQueue as V7Queue,
    QueuePause as V7QueuePause,
    make_runner as make_v7_runner,
)


def make_queue(generation, args):
    if generation == "v6":
        factory, queue_type = make_v6_runner, V6Queue
        configs = (
            ROOT / "configs/subcellular_lr_v6_a_20260920.yaml",
            ROOT / "configs/subcellular_lr_v6_b_20260920.yaml",
        )
        output = ROOT / "outputs/subcellular_lr_v6_async_eval_20260920"
    else:
        factory, queue_type = make_v7_runner, V7Queue
        configs = (
            ROOT / "configs/subcellular_lr_v7_a_20260921.yaml",
            ROOT / "configs/subcellular_lr_v7_b_20260921.yaml",
        )
        output = ROOT / "outputs/subcellular_lr_v7_async_eval_20260921"
    runners = {
        route: factory(config, args.deadline, args.eval_min_seconds)
        for route, config in zip(("A", "B"), configs)
    }
    return queue_type(
        runners, output, args.deadline,
        poll_seconds=args.poll_seconds, eval_min_seconds=args.eval_min_seconds,
    )


def run_one(generation, args):
    """Evaluate at most one checkpoint and release every generation lock."""
    queue = make_queue(generation, args)
    prepared = False
    try:
        queue.prepare()
        prepared = True
        tasks, complete = queue.collect()
        if complete:
            queue.status("all_evaluated", active=None)
            return "complete"
        if args.deadline - time.time() < args.eval_min_seconds:
            queue.status("paused", active=None, reason="evaluation_time_budget")
            return "deadline"
        if not tasks:
            queue.status("idle", active=None, reason="awaiting_training_or_handoff")
            return "idle"
        if not queue.evaluate(tasks[0]):
            queue.status("paused", active=None, reason="evaluation_deadline")
            return "deadline"
        queue.status("idle", active=None, reason="continuous_generation_rotation")
        return "evaluated"
    except (V6QueuePause, V7QueuePause) as error:
        if prepared:
            queue.status("paused", active=None, reason=str(error))
        return "paused"
    except BaseException as error:
        if prepared:
            queue.status(
                "required_attention", active=None,
                error=f"{type(error).__name__}: {error}",
            )
        raise
    finally:
        queue.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--eval-min-seconds", type=float, default=600)
    args = parser.parse_args()
    if (
        not math.isfinite(args.deadline)
        or not math.isfinite(args.poll_seconds)
        or args.poll_seconds <= 0
        or not math.isfinite(args.eval_min_seconds)
        or args.eval_min_seconds <= 0
    ):
        raise ValueError("Require a finite deadline and positive polling margins")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_STEP_ID"):
        raise RuntimeError("Run only inside a verified one-GPU Slurm compute step")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Continuous evaluation requires exactly one visible GPU")
    os.chdir(ROOT)

    stopped = False

    def request_stop(signum, _frame):
        nonlocal stopped
        stopped = True
        print(f"[continuous-eval] stop={signal.Signals(signum).name}", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    generations = ("v6", "v7")
    while not stopped and args.deadline - time.time() >= args.eval_min_seconds:
        cycle = {}
        for generation in generations:
            if stopped or args.deadline - time.time() < args.eval_min_seconds:
                break
            cycle[generation] = run_one(generation, args)
        print("[continuous-eval] " + json.dumps(cycle, sort_keys=True), flush=True)
        if cycle and all(value == "complete" for value in cycle.values()):
            return
        if not any(value == "evaluated" for value in cycle.values()):
            time.sleep(min(args.poll_seconds, max(0., args.deadline - time.time()), 60))


if __name__ == "__main__":
    main()
