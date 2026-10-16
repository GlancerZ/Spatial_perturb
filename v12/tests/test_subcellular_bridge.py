"""Bridge concurrency QA; no real scheduler, model, CUDA, or training commands."""
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location('subcellular_bridge_under_test',
    Path(__file__).resolve().parents[1] / 'scripts/bridge_subcellular_single_gpu.py')
module = importlib.util.module_from_spec(SPEC)
# Only torch.load is used by the coordinator. Avoid importing torch entirely;
# these tests can run with the Python standard library and cannot touch CUDA.
with patch.dict(sys.modules, {'torch': SimpleNamespace(load=Mock())}):
    SPEC.loader.exec_module(module)


def assert_lock_busy(test, path):
    with path.open('a') as contender:
        with test.assertRaises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


class InheritedLockTests(unittest.TestCase):
    def test_child_inherits_lock_after_parent_descriptor_closes(self):
        """Actual exec inheritance, not a mocked assertion about pass_fds."""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'training.lock'
            owner = path.open('a')
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # This child does no work: acknowledge its inherited descriptor,
            # then wait on stdin so the test controls its lifetime without sleeps.
            code = ('import os,sys; os.fstat(int(sys.argv[1])); '
                    'print("READY",flush=True); sys.stdin.buffer.read(1)')
            child = subprocess.Popen([sys.executable, '-c', code, str(owner.fileno())],
                pass_fds=(owner.fileno(),), stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                readable, _, _ = select.select([child.stdout], [], [], 10)
                self.assertTrue(readable, 'Lock-holder child did not become ready')
                self.assertEqual(child.stdout.readline().strip(), 'READY')
                owner.close()
                assert_lock_busy(self, path)
                child.communicate(input='x', timeout=10)
                self.assertEqual(child.returncode, 0)
                with path.open('a') as contender:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                owner.close()
                if child.poll() is None:
                    child.kill()
                    child.communicate(timeout=10)


class BridgeCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / 'outputs/run'
        self.run.mkdir(parents=True)
        (self.root / 'logs').mkdir()
        (self.run / 'resume.pt').write_text('original checkpoint 2255')
        self.saved_step = 2255
        self.original_cwd = Path.cwd()
        self.patches = [
            patch.object(module, 'ROOT', self.root),
            patch.object(module.signal, 'signal'),
            patch.dict(os.environ, SLURM_JOB_ID='101', V2_TRAIN_DEADLINE='10000'),
            patch.object(sys, 'argv', ['bridge', '--primary-jobid', '202',
                                      '--run-dir', str(self.run)]),
            patch.object(module.time, 'time', return_value=1000.),
            patch.object(module.time, 'sleep'),
            patch.object(module.torch, 'load', side_effect=lambda *a, **k:
                         dict(step=self.saved_step, complete=False)),
            patch.object(module.subprocess, 'run',
                         side_effect=AssertionError('Unmocked scheduler command')),
            patch.object(module.subprocess, 'Popen',
                         side_effect=AssertionError('Unmocked process launch')),
            patch.object(sys, 'stdout', new_callable=io.StringIO),
        ]
        started = [item.start() for item in self.patches]
        self.sleep, self.load, self.scheduler, self.popen = started[5:9]

    def tearDown(self):
        os.chdir(self.original_cwd)
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def queue(self, status):
        return SimpleNamespace(stdout=f'202|{status}\n', returncode=0)

    def final_state(self):
        return json.loads((self.run / 'single_gpu_bridge.json').read_text())

    def assert_lock_released(self):
        with (self.run / 'training.lock').open('a') as contender:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_running_primary_waits_for_complete_update_before_handoff(self):
        self.scheduler.side_effect = [self.queue('PENDING'), self.queue('RUNNING')]
        child = Mock(returncode=0)
        child.poll.side_effect = [None, 0]
        self.popen.side_effect = None
        self.popen.return_value = child

        def save_update_before_exit(_seconds):
            self.assertTrue((self.run / 'SERIAL_STOP_REQUESTED').exists())
            assert_lock_busy(self, self.run / 'training.lock')
            self.saved_step = 2256
            replacement = self.run / 'resume.pt.partial'
            replacement.write_text('saved complete optimizer update 2256')
            replacement.replace(self.run / 'resume.pt')

        self.sleep.side_effect = save_update_before_exit
        module.main()
        self.assertEqual(self.final_state()['status'], 'handoff_saved')
        self.assertEqual(self.final_state()['step'], 2256)
        self.assertEqual(self.load.call_count, 2)
        self.assertEqual(self.popen.call_count, 1, 'No evaluator may start before handoff')
        launch = self.popen.call_args
        self.assertIn('--run-update-limit', launch.args[0])
        self.assertEqual(launch.args[0][-1], '45')
        self.assertEqual(len(launch.kwargs['pass_fds']), 1)
        self.assertEqual((self.run / 'resume_before_single_gpu_bridge.pt').read_text(),
                         'original checkpoint 2255')
        self.assertFalse((self.run / 'SERIAL_STOP_REQUESTED').exists())
        self.assertFalse((self.run / 'PAUSE_REQUESTED').exists())
        self.assert_lock_released()

    def test_trainer_failure_pauses_instead_of_handing_off(self):
        self.scheduler.side_effect = [self.queue('PENDING')]
        child = Mock(returncode=7)
        child.poll.return_value = 7
        self.popen.side_effect = None
        self.popen.return_value = child
        with self.assertRaisesRegex(RuntimeError, 'Serial trainer failed with exit code 7'):
            module.main()
        pause = json.loads((self.run / 'PAUSE_REQUESTED').read_text())
        self.assertEqual(pause['source'], 'single_gpu_bridge_failure')
        self.assertEqual(self.final_state()['status'], 'action_required')
        self.assertEqual(self.saved_step, 2255)
        self.assertEqual(self.popen.call_count, 1)
        self.assert_lock_released()

    def test_short_remaining_deadline_exits_without_launch_or_false_pause(self):
        with patch.dict(os.environ, V2_TRAIN_DEADLINE='1100'):
            module.main()
        self.assertEqual(self.final_state()['status'], 'allocation_deadline')
        self.scheduler.assert_not_called()
        self.popen.assert_not_called()
        self.load.assert_not_called()
        self.assertFalse((self.run / 'PAUSE_REQUESTED').exists())
        self.assert_lock_released()


if __name__ == '__main__':
    unittest.main()
