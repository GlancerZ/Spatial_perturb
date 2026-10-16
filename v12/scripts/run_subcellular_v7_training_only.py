"""Train the V7 epoch while a separate worker evaluates checkpoints.

Uses the original controller identity and exclusive owner lock. This controller
never evaluates or deletes a checkpoint and never claims evaluation completion.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from run_subcellular_v7 import Runner as InlineRunner, atomic_json


EXECUTION_MODE_SCHEMA = 'subcellular_v7_execution_mode_v1'


class Runner(InlineRunner):
    def reserve_execution_mode(self):
        """Publish only while holding the original controller's owner lock."""
        expected = dict(schema=EXECUTION_MODE_SCHEMA, mode='training_only',
                        config_sha256=self.fingerprint, output_dir=str(self.run))
        path = self.control / 'execution_mode.json'
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError('Existing execution mode belongs to a different configuration or controller mode')
        else:
            atomic_json(path, expected)
        self.state.update(execution_mode='training_only', evaluation_completion_verified=False)

    def training_complete(self, progress, total):
        if (not (self.run / 'TRAIN_COMPLETE').is_file() or progress.get('epoch') != 1
                or progress.get('next_update') != 0):
            raise ValueError('Training completion lacks its complete epoch checkpoint boundary')
        self.status('training_complete', step=progress['step'], training_step=progress['step'],
                    total_updates=total, active_child=None, evaluation_completion_verified=False)

    def pause(self, reason, step, total):
        self.status('paused', reason=reason, step=step, training_step=step,
                    total_updates=total, active_child=None, evaluation_completion_verified=False)

    def run_foreground(self):
        if not self.args.resume and self.args.deadline - time.time() < self.args.train_min_seconds:
            print('[v7-training-only] insufficient initialization time; no run created', flush=True)
            return
        self.prepare()
        mode_reserved = False
        try:
            self.reserve_execution_mode()
            mode_reserved = True
            if not self.args.resume:
                if self.stopped() or self.args.deadline - time.time() < self.args.train_min_seconds:
                    self.pause('stop_marker' if self.stopped() else 'initialization_time_budget', None, None)
                    return
                self.child(self.trainer_command(initialize=True), 'initialization', 0)
                progress, _ = self.progress()
                if progress['step'] != 0:
                    raise ValueError('Initialization must save optimizer step zero')
            progress, total = self.progress()
            step = progress['step']
            if self.stopped():
                self.pause('stop_marker', step, total)
                return
            if progress['complete']:
                self.training_complete(progress, total)
                return
            if self.args.deadline - time.time() < self.args.train_min_seconds:
                self.pause('training_time_budget', step, total)
                return
            # One trainer process receives the entire remaining epoch budget.
            # Its unchanged 100/25 checkpoint/resume cadence remains active.
            if not self.child(self.trainer_command(limit=total - step), 'training', step):
                self.pause('handoff_before_trainer_start', step, total)
                return
            advanced, new_total = self.progress()
            if new_total != total or not step < advanced['step'] <= total:
                raise ValueError('Training progress must increase within the unchanged full epoch budget')
            if advanced['complete']:
                self.training_complete(advanced, total)
            elif self.stopped():
                self.pause('stop_marker', advanced['step'], total)
            elif time.time() >= self.args.deadline:
                self.pause('training_time_budget', advanced['step'], total)
            else:
                raise ValueError('Trainer exited before epoch completion without a stop request or deadline')
        except Exception as error:
            if mode_reserved:
                self.status('failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            self.lock.close()
            self.lock = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--training-gpus', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--nodes', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--gpus-per-node', type=int, choices=(1, 2, 4))
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--eval-min-seconds', type=float, default=600,
                        help='Accepted for launcher compatibility; no evaluation runs here')
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
