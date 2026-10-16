"""Temporary-file concurrency checks; run only inside a Slurm compute step."""
from __future__ import annotations

from argparse import Namespace
import json
import multiprocessing
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import watch_subcellular_v6_left_parallel_evaluations as worker


def lock_holder(path, connection):
    with worker.exclusive_lock(Path(path)) as owned:
        connection.send(bool(owned))
        if owned:
            connection.recv()
    connection.close()


def status_writer(output, worker_id, identity):
    for number in range(30):
        worker.publish_status(Path(output), worker_id, identity, {'sequence': number})


class ParallelEvaluationChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v6-left-parallel-check-')
        self.output = Path(self.temporary.name)
        self.context = multiprocessing.get_context('fork')
        self.children = []

    def tearDown(self):
        for process, connection in self.children:
            if process.is_alive():
                connection.send('release')
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()  # Only this test's own dummy process.
                process.join(timeout=10)
            connection.close()
        self.temporary.cleanup()

    def hold(self, path):
        parent, child = self.context.Pipe()
        process = self.context.Process(target=lock_holder, args=(str(path), child))
        process.start()
        child.close()
        self.children.append((process, parent))
        self.assertTrue(parent.poll(15), 'Dummy lock holder did not report ownership')
        return process, parent, parent.recv()

    def test_checkpoint_exclusion_and_release(self):
        path = worker.checkpoint_lock_path(self.output, 'A', 3800)
        first, channel, owned = self.hold(path)
        self.assertTrue(owned)
        second, _, owned = self.hold(path)
        self.assertFalse(owned)
        second.join(timeout=10)
        self.assertEqual(second.exitcode, 0)
        channel.send('release')
        first.join(timeout=10)
        self.assertEqual(first.exitcode, 0)
        with worker.exclusive_lock(path) as owned:
            self.assertTrue(owned)

    def test_distinct_checkpoints_can_run_together(self):
        _, _, first = self.hold(worker.checkpoint_lock_path(self.output, 'A', 3800))
        _, _, second = self.hold(worker.checkpoint_lock_path(self.output, 'B', 4900))
        self.assertTrue(first and second)

    def test_worker_ownership_rejects_duplicate(self):
        path = worker.worker_lock_path(self.output, 'w1')
        _, _, owned = self.hold(path)
        self.assertTrue(owned)
        with worker.exclusive_lock(path) as duplicate:
            self.assertFalse(duplicate)
        with worker.exclusive_lock(worker.worker_lock_path(self.output, 'w2')) as other:
            self.assertTrue(other)

    def test_legacy_handoff_and_identity(self):
        identity = {'schema': 'dummy-v6-identity', 'routes': {'A': {}, 'B': {}}}
        first, channel, owned = self.hold(self.output / 'owner.lock')
        self.assertTrue(owned)
        with self.assertRaisesRegex(RuntimeError, 'Legacy evaluator'):
            worker.validate_manifest(self.output, identity)
        channel.send('release')
        first.join(timeout=10)
        worker.validate_manifest(self.output, identity)
        self.assertEqual(json.loads((self.output / 'identity.json').read_text()), identity)
        worker.validate_manifest(self.output, identity)
        with self.assertRaisesRegex(ValueError, 'another evaluator'):
            worker.validate_manifest(self.output, {'schema': 'different'})
        with worker.exclusive_lock(self.output / 'owner.lock') as owned:
            self.assertTrue(owned, 'Legacy startup lock was not released')

    def test_brief_parallel_manifest_ownership_retried(self):
        identity = {'schema': 'dummy-v6-identity', 'routes': {'A': {}, 'B': {}}}
        process, channel, owned = self.hold(self.output / 'owner.lock')
        self.assertTrue(owned)
        release = threading.Timer(.2, lambda: channel.send('release'))
        release.start()
        try:
            worker.validate_manifest(self.output, identity)
        finally:
            release.join(timeout=5)
            process.join(timeout=10)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(json.loads((self.output / 'identity.json').read_text()), identity)

    def test_concurrent_atomic_status_and_canonical_owner(self):
        identity = {'schema': 'dummy-v6-identity', 'routes': {'A': {}, 'B': {}}}
        processes = [self.context.Process(target=status_writer,
                     args=(str(self.output), worker_id, identity)) for worker_id in worker.WORKERS]
        try:
            for process in processes:
                process.start()
            while any(process.is_alive() for process in processes):
                for path in self.output.rglob('status.json'):
                    status = json.loads(path.read_text())
                    self.assertEqual(status['queue_identity'], identity)
                    if path.parent == self.output:
                        self.assertEqual(status['worker_id'], 'w0')
                time.sleep(.005)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)
        canonical = json.loads((self.output / 'status.json').read_text())
        worker.publish_status(self.output, 'w3', identity, {'sequence': 99})
        self.assertEqual(json.loads((self.output / 'status.json').read_text()), canonical)
        for worker_id in worker.WORKERS:
            status = json.loads((self.output / 'parallel_workers' / worker_id / 'status.json').read_text())
            self.assertEqual(status['worker_id'], worker_id)
            self.assertEqual(status['queue_identity'], identity)
        self.assertFalse(list(self.output.rglob('*.partial')))

    def test_command_protocol(self):
        command = worker.evaluation_command(Path('checkpoint.pt'), Path('config.yaml'), Path('result'), 3800)
        self.assertEqual(command[1:], [str(worker.EVAL), '--checkpoint', 'checkpoint.pt',
            '--config', 'config.yaml', '--output-dir', 'result', '--cohort', 'fixed5200',
            '--cells-per-donor', '400', '--mask-validation-batches', '16', '--expected-step', '3800'])

    def queue_fixture(self):
        checkpoint = self.output / 'dummy.pt'
        checkpoint.write_bytes(b'dummy checkpoint')
        entries = {'A': dict(config=Path('a.yaml'), fingerprint='a'),
                   'B': dict(config=Path('b.yaml'), fingerprint='b')}
        args = Namespace(worker_id='w0', priority_checkpoint=list(worker.DEFAULT_PRIORITIES),
                         deadline_epoch=time.time() + 120, minimum_launch_seconds=1, poll_seconds=1)
        complete = dict(step=3800, total=10269, complete=True)
        return checkpoint, entries, args, complete

    def test_reverify_after_lock_avoids_duplicate(self):
        checkpoint, entries, args, complete = self.queue_fixture()
        calls = []

        def fake_verified(*unused):
            calls.append(None)
            return len(calls) >= 2

        def tasks(route, entry):
            return complete, [('A', 3800, checkpoint)] if route == 'A' else []

        with patch.object(worker, 'committed', side_effect=tasks), \
             patch.object(worker, 'verified', side_effect=fake_verified), \
             patch.object(worker.subprocess, 'run') as run:
            worker.run_queue(args, entries, self.output, {'schema': 'dummy'}, {})
        run.assert_not_called()
        self.assertTrue(json.loads((self.output / 'status.json').read_text())['all_evaluated'])

    def test_failed_subprocess_releases_checkpoint_lock(self):
        checkpoint, entries, args, complete = self.queue_fixture()
        def tasks(route, entry):
            return complete, [('A', 3800, checkpoint)] if route == 'A' else []
        with patch.object(worker, 'committed', side_effect=tasks), \
             patch.object(worker, 'verified', return_value=False), \
             patch.object(worker.subprocess, 'run', side_effect=subprocess.CalledProcessError(7, ['dummy'])):
            with self.assertRaises(subprocess.CalledProcessError):
                worker.run_queue(args, entries, self.output, {'schema': 'dummy'}, {})
        status = json.loads((self.output / 'status.json').read_text())
        self.assertEqual(status['state'], 'failed')
        self.assertFalse(status['all_evaluated'])
        with worker.exclusive_lock(worker.checkpoint_lock_path(self.output, 'A', 3800)) as owned:
            self.assertTrue(owned)

    def test_busy_checkpoint_skipped_for_other_work(self):
        checkpoint, entries, args, complete = self.queue_fixture()
        _, _, owned = self.hold(worker.checkpoint_lock_path(self.output, 'A', 3800))
        self.assertTrue(owned)
        evaluated = []

        def tasks(route, entry):
            step = 3800 if route == 'A' else 4900
            return complete, [(route, step, checkpoint)]

        def fake_run(command, **options):
            evaluated.append(command[command.index('--expected-step') + 1])
            self.assertEqual(len(options['pass_fds']), 1)

        with patch.object(worker, 'committed', side_effect=tasks), \
             patch.object(worker, 'verified', side_effect=lambda *unused: bool(evaluated)), \
             patch.object(worker.subprocess, 'run', side_effect=fake_run):
            worker.run_queue(args, entries, self.output, {'schema': 'dummy'}, {})
        self.assertEqual(evaluated, ['4900'])

    def test_deadline_without_starting_work(self):
        checkpoint, entries, args, complete = self.queue_fixture()
        args.deadline_epoch = time.time()
        def tasks(route, entry):
            return complete, [('A', 3800, checkpoint)] if route == 'A' else []
        with patch.object(worker, 'committed', side_effect=tasks), \
             patch.object(worker, 'verified', return_value=False), \
             patch.object(worker.subprocess, 'run') as run:
            worker.run_queue(args, entries, self.output, {'schema': 'dummy'}, {})
        run.assert_not_called()
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['state'], 'deadline')

    def test_deadline_timeout_releases_checkpoint(self):
        checkpoint, entries, args, complete = self.queue_fixture()
        def tasks(route, entry):
            return complete, [('A', 3800, checkpoint)] if route == 'A' else []
        with patch.object(worker, 'committed', side_effect=tasks), \
             patch.object(worker, 'verified', return_value=False), \
             patch.object(worker.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['dummy'], 1)):
            worker.run_queue(args, entries, self.output, {'schema': 'dummy'}, {})
        self.assertEqual(json.loads((self.output / 'status.json').read_text())['state'], 'deadline')
        with worker.exclusive_lock(worker.checkpoint_lock_path(self.output, 'A', 3800)) as owned:
            self.assertTrue(owned)


if __name__ == '__main__':
    worker.require_compute_step()
    unittest.main(verbosity=2)
