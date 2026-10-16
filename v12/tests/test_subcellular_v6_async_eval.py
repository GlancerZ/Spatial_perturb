"""Async V6 queue boundaries; run only on the authorized compute allocation.

Small filesystem fixtures exercise the actual legacy Runner validators. Child
processes and waits are mocked; no Slurm job, CUDA model or evaluator is started.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import watch_subcellular_v6_evaluations as worker


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class V6AsyncEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.deadline = time.time() + 3600
        self.runners = {}
        for route in ('A', 'B'):
            config = dict(
                output_dir=str(self.root / route),
                training=dict(epochs=1, checkpoint_every=100, global_batch_size=64, batch_size=16),
                evaluation=dict(interval=100, cells_per_donor=400, mask_validation_batches=16,
                                test_donors=False, batch_size=16),
                model=dict(route=route, slots_per_compartment=8, slot_dim=64),
                lr=dict(sha256='f' * 64))
            path = self.root / f'{route}.yaml'
            path.write_text(yaml.safe_dump(config))
            runner = worker.make_runner(path, self.deadline, 1)
            self.runners[route] = runner
            runner.run.mkdir()
            (runner.run / 'checkpoints').mkdir()
            (runner.run / 'resume.pt').write_bytes(b'optimizer resume boundary fixture')
            runner.control.mkdir()
            write_json(runner.control / 'identity.json', runner.identity)
            write_json(runner.control / 'execution_mode.json', worker.execution_mode(runner))
            self.progress(route, 100, 200)
            self.archive(route, 0)
            self.archive(route, 100)
        self.print_patch = mock.patch('builtins.print')
        self.print_patch.start()
        self.addCleanup(self.print_patch.stop)

    def queue(self, *, prepare=False, name='queue', **kwargs):
        queue = worker.EvaluationQueue(self.runners, self.root / name, self.deadline,
                                       poll_seconds=1, eval_min_seconds=1, **kwargs)
        self.addCleanup(queue.close)
        if prepare:
            queue.prepare()
        return queue

    def progress(self, route, step, total):
        runner = self.runners[route]
        complete = step == total
        write_json(runner.run / 'progress.json', dict(
            model_version=6, architecture='subcellular_lr_routes_v6',
            config_sha256=runner.fingerprint, step=step, complete=complete,
            epoch=1 if complete else 0, next_update=0 if complete else step))
        write_json(runner.run / 'schedule.json', dict(
            model_version=6, architecture='subcellular_lr_routes_v6',
            total_updates=total, updates_per_epoch=total))
        write_json(runner.control / 'status.json', dict(
            identity=runner.identity, status='training_complete' if complete else 'training',
            step=step, training_step=step, total_updates=total, active_child=None))

    def archive(self, route, step):
        path = self.runners[route].run / 'checkpoints' / f'step_{step:08d}.pt'
        path.write_bytes(f'atomic published checkpoint {route}/{step}'.encode())
        return path

    def metrics(self, route, step):
        runner = self.runners[route]
        checkpoint = runner.run / 'checkpoints' / f'step_{step:08d}.pt'
        dimensions = {name: 512 for name in (
            'base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm', 'nucleus_local', 'cytoplasm_local',
            'joint_neighbor_shuffle', 'nucleus_neighbor_shuffle', 'cytoplasm_neighbor_shuffle')}
        value = dict(
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            step=step, model_version=6, architecture='subcellular_lr_routes_v6', evaluation_version=6,
            config_sha256=runner.fingerprint,
            cohort_sha256='0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc',
            cells=5200, canonical_cohort=True, test_donors_used=False, backbone_kind='scgpt',
            neighbor_shuffle=True, neighbor_shuffle_seed=815, diagnostic_schema_version=1,
            masked_validation=dict(enabled=True, requested_minimum_batches=16),
            arch_config=dict(backbone_dim=512, route=route), embedding_dimensions=dimensions,
            results=[dict(embedding=name, task=task, fixed_C_result=dict(
                C=.1 if task == 'cell_type' else .01, val_macro_f1=.4))
                for name in dimensions for task in ('cell_type', 'region')],
            lr_prior=dict(sha256='f' * 64))
        path = runner.run / 'downstream_validation' / f'step_{step:08d}' / 'metrics.json'
        write_json(path, value)
        return path

    def command_records(self, queue):
        return [json.loads(line) for line in (queue.output / 'commands.jsonl').read_text().splitlines()]

    def test_archive_before_progress_is_deferred_and_partial_filename_ignored(self):
        future = self.archive('A', 200)
        partial = future.with_name('step_00000300.pt.partial')
        partial.write_bytes(b'incomplete unpublished write')
        queue = self.queue(prepare=True)
        tasks, complete = queue.collect()
        self.assertFalse(complete)
        self.assertNotIn(('A', 200), [(task.route, task.step) for task in tasks])
        self.assertEqual(queue.state['routes']['A']['deferred_publication_steps'], [200])
        self.progress('A', 200, 200)
        tasks, _ = queue.collect()
        self.assertIn(('A', 200), [(task.route, task.step) for task in tasks])
        self.assertTrue(future.exists())
        self.assertEqual(partial.read_bytes(), b'incomplete unpublished write')

    def test_valid_old_metrics_are_skipped_and_pending_order_is_fair(self):
        self.metrics('A', 0)
        queue = self.queue(prepare=True)
        tasks, _ = queue.collect()
        self.assertEqual([(task.step, task.route) for task in tasks], [(0, 'B'), (100, 'A'), (100, 'B')])
        self.assertEqual(queue.state['routes']['A']['evaluated_steps'], [0])
        self.assertEqual(queue.state['routes']['A']['queued_steps'], [100])

    def test_all_archives_including_nonperiodic_handoff_steps_are_retained(self):
        self.archive('A', 25)
        self.archive('B', 50)
        before = {path: path.read_bytes() for runner in self.runners.values()
                  for path in (runner.run / 'checkpoints').iterdir()}
        queue = self.queue(prepare=True)
        tasks, _ = queue.collect()
        self.assertEqual([(task.step, task.route) for task in tasks],
                         [(0, 'A'), (0, 'B'), (25, 'A'), (50, 'B'), (100, 'A'), (100, 'B')])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_all_evaluated_requires_both_complete_progress_and_every_checkpoint(self):
        queue = self.queue(prepare=True)
        for route in self.runners:
            for step in (0, 100):
                self.metrics(route, step)
        tasks, complete = queue.collect()
        self.assertEqual(tasks, [])
        self.assertFalse(complete)
        for route in self.runners:
            self.archive(route, 200)
            self.progress(route, 200, 200)
        self.metrics('A', 200)
        tasks, complete = queue.collect()
        self.assertFalse(complete)
        self.assertEqual([(task.route, task.step) for task in tasks], [('B', 200)])
        self.metrics('B', 200)
        self.assertEqual(queue.collect(), ([], True))

    def test_missing_required_archive_or_wrong_metric_checkpoint_hash_is_rejected(self):
        self.progress('A', 200, 200)
        queue = self.queue(prepare=True)
        with self.assertRaisesRegex(worker.RequiredAttention, 'required archives'):
            queue.collect()
        self.archive('A', 200)
        path = self.metrics('A', 0)
        report = json.loads(path.read_text())
        report['checkpoint_sha256'] = '0' * 64
        write_json(path, report)
        with self.assertRaisesRegex(ValueError, 'checkpoint identity mismatch'):
            queue.collect()

    def test_absent_mode_waits_without_locking_or_evaluating_that_route(self):
        (self.runners['A'].control / 'execution_mode.json').unlink()
        queue = self.queue(prepare=True)
        tasks, complete = queue.collect()
        self.assertFalse(complete)
        self.assertTrue(all(task.route == 'B' for task in tasks))
        self.assertNotIn('A', queue.route_locks)
        self.assertEqual(queue.state['routes']['A']['reason'], 'awaiting_training_only_handoff')

    def test_mode_config_mismatch_and_global_queue_identity_conflict_fail_closed(self):
        runner = self.runners['A']
        conflicting = worker.execution_mode(runner)
        conflicting['config_sha256'] = 'wrong'
        write_json(runner.control / 'execution_mode.json', conflicting)
        queue = self.queue(prepare=True)
        with self.assertRaisesRegex(worker.RequiredAttention, 'execution mode'):
            queue.collect()
        queue.close()
        write_json(queue.output / 'identity.json', {'schema': 'foreign'})
        another = self.queue()
        with self.assertRaisesRegex(worker.RequiredAttention, 'Queue identity differs'):
            another.prepare()

    def test_global_and_per_run_leases_prevent_duplicate_workers(self):
        first = self.queue(prepare=True)
        with self.assertRaises(BlockingIOError):
            self.queue().prepare()
        first.collect()
        independent_output = self.queue(name='other_queue', prepare=True)
        with self.assertRaises(BlockingIOError):
            independent_output.collect()

    def test_training_failure_requires_attention_without_modifying_controller_status(self):
        runner = self.runners['B']
        path = runner.control / 'status.json'
        status = json.loads(path.read_text())
        status.update(status='failed', error='fixture training failure')
        write_json(path, status)
        before = path.read_bytes()
        queue = self.queue()
        with self.assertRaisesRegex(worker.RequiredAttention, 'B: training controller failed'):
            queue.run_foreground()
        self.assertEqual(path.read_bytes(), before)
        report = json.loads((queue.output / 'status.json').read_text())
        self.assertEqual(report['status'], 'required_attention')
        self.assertIn('fixture training failure', report['error'])

    def test_pause_markers_remain_owned_by_their_creators(self):
        marker = self.runners['A'].run / 'PAUSE_REQUESTED'
        marker.write_text('operator-owned marker')
        queue = self.queue(prepare=True)
        tasks, _ = queue.collect()
        self.assertTrue(all(task.route == 'B' for task in tasks))
        self.assertEqual(marker.read_text(), 'operator-owned marker')
        self.assertEqual(queue.state['routes']['A']['reason'], 'training_pause_marker')

    def test_deadline_avoids_new_child_and_never_writes_training_controller_status(self):
        before = {route: (runner.control / 'status.json').read_bytes()
                  for route, runner in self.runners.items()}
        queue = self.queue()
        queue.deadline = time.time() - 1
        with mock.patch.object(worker.subprocess, 'Popen') as popen:
            queue.run_foreground()
        popen.assert_not_called()
        self.assertEqual(json.loads((queue.output / 'status.json').read_text())['status'], 'paused')
        self.assertEqual(before, {route: (runner.control / 'status.json').read_bytes()
                                  for route, runner in self.runners.items()})

    def test_timeout_kills_only_owned_process_group_and_retains_partial_output(self):
        queue = self.queue(prepare=True)
        tasks, _ = queue.collect()
        partial = self.runners['A'].run / 'downstream_validation/step_00000000/cache.partial'
        partial.parent.mkdir(parents=True)
        partial.write_bytes(b'reusable evaluator partial')
        process = mock.Mock(pid=73125)
        process.poll.return_value = None
        calls = []

        def wait(timeout=None):
            calls.append(timeout)
            if len(calls) == 1:
                queue.deadline = time.time() - 1
                raise subprocess.TimeoutExpired('evaluator', timeout)
            if len(calls) == 2:
                raise subprocess.TimeoutExpired('evaluator', timeout)
            return -9

        process.wait.side_effect = wait
        with mock.patch.object(worker.subprocess, 'Popen', return_value=process) as popen, \
             mock.patch.object(worker.os, 'killpg') as killpg:
            self.assertFalse(queue.evaluate(tasks[0]))
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(set(popen.call_args.kwargs['pass_fds']),
                         {queue.lock.fileno(), *(handle.fileno() for handle in queue.route_locks.values())})
        self.assertEqual(killpg.call_args_list, [mock.call(73125, signal.SIGTERM), mock.call(73125, signal.SIGKILL)])
        entry = self.command_records(queue)[0]
        self.assertEqual((entry['status'], entry['exit_code']), ('deadline_interrupted', -9))
        self.assertTrue(Path(entry['log']).exists())
        self.assertTrue(Path(entry['log']).with_suffix('.json').exists())
        self.assertEqual(partial.read_bytes(), b'reusable evaluator partial')
        self.assertIsNone(queue.state['active'])

    def test_child_nonzero_is_logged_and_queue_stops_without_retry(self):
        queue = self.queue()
        process = mock.Mock(pid=73126)
        process.wait.return_value = process.poll.return_value = 7
        with mock.patch.object(worker.subprocess, 'Popen', return_value=process) as popen:
            with self.assertRaises(subprocess.CalledProcessError):
                queue.run_foreground()
        self.assertEqual(popen.call_count, 1)
        entry = self.command_records(queue)[0]
        self.assertEqual((entry['status'], entry['exit_code'], entry['returncode']), ('failed', 7, 7))
        status = json.loads((queue.output / 'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertIsNone(status['active'])
        retry = self.queue()
        with self.assertRaisesRegex(worker.RequiredAttention, 'Previous queue failure'):
            retry.prepare()

    def test_zero_exit_requires_real_valid_metrics_and_success_records_exact_command(self):
        queue = self.queue(prepare=True)
        tasks, _ = queue.collect()
        process = mock.Mock(pid=73127)
        process.wait.return_value = process.poll.return_value = 0
        with mock.patch.object(worker.subprocess, 'Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'without complete valid metrics'):
                queue.evaluate(tasks[0])
        self.assertEqual(self.command_records(queue)[0]['status'], 'invalid_metrics')
        self.metrics('B', 0)
        with mock.patch.object(worker.subprocess, 'Popen', return_value=process):
            self.assertTrue(queue.evaluate(tasks[1]))
        entry = self.command_records(queue)[1]
        self.assertEqual(entry['status'], 'passed')
        self.assertEqual(entry['command'], self.runners['B'].evaluation_command(0, tasks[1].checkpoint))
        self.assertEqual(entry['exit_code'], 0)

    def test_training_failure_during_child_terminates_only_that_child_with_provenance(self):
        queue = self.queue()
        process = mock.Mock(pid=73128)
        process.poll.return_value = None
        waits = []

        def wait(timeout=None):
            waits.append(timeout)
            if len(waits) == 1:
                runner = self.runners['B']
                write_json(runner.control / 'status.json', dict(
                    identity=runner.identity, status='failed', error='failure while A is evaluating'))
                raise subprocess.TimeoutExpired('evaluator', timeout)
            return -15

        process.wait.side_effect = wait
        with mock.patch.object(worker.subprocess, 'Popen', return_value=process) as popen, \
             mock.patch.object(worker.os, 'killpg') as killpg:
            with self.assertRaisesRegex(worker.RequiredAttention, 'B: training controller failed'):
                queue.run_foreground()
        popen.assert_called_once()
        killpg.assert_called_once_with(73128, signal.SIGTERM)
        entry = self.command_records(queue)[0]
        self.assertEqual((entry['route'], entry['status'], entry['exit_code']), ('A', 'required_attention', -15))
        self.assertEqual(json.loads((queue.output / 'status.json').read_text())['status'], 'required_attention')

    def test_complete_runs_exit_all_evaluated_without_spawning_any_child(self):
        for route in self.runners:
            self.progress(route, 100, 100)
            self.metrics(route, 0)
            self.metrics(route, 100)
        queue = self.queue()
        with mock.patch.object(worker.subprocess, 'Popen') as popen:
            queue.run_foreground()
        popen.assert_not_called()
        status = json.loads((queue.output / 'status.json').read_text())
        self.assertEqual(status['status'], 'all_evaluated')
        self.assertTrue(status['all_evaluated'])
        self.assertTrue(status['training_complete'])

    def test_idle_before_handoff_waits_instead_of_claiming_completion(self):
        for runner in self.runners.values():
            (runner.control / 'execution_mode.json').unlink()
        queue = self.queue()

        def end_budget(_seconds):
            queue.deadline = time.time() - 1

        with mock.patch.object(worker.time, 'sleep', side_effect=end_budget) as sleep, \
             mock.patch.object(worker.subprocess, 'Popen') as popen:
            queue.run_foreground()
        sleep.assert_called_once()
        popen.assert_not_called()
        status = json.loads((queue.output / 'status.json').read_text())
        self.assertFalse(status['all_evaluated'])
        self.assertTrue(all(not route['ready'] for route in status['routes'].values()))


if __name__ == '__main__':
    unittest.main()
