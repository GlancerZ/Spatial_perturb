"""Pure coordinator scheduling tests; no torch, CUDA, Slurm or real training."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location('readout_ab_runner_under_test',
    Path(__file__).resolve().parents[1] / 'scripts/run_subcellular_readout_ab.py')
module = importlib.util.module_from_spec(SPEC)
with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'yaml': SimpleNamespace()}):
    SPEC.loader.exec_module(module)


class RunnerSchedulingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for arm in module.ARMS:
            (self.root / arm).mkdir()
        self.runner = module.Runner.__new__(module.Runner)
        r = self.runner
        r.run, r.start, r.target = self.root, 3652, 3952
        r.deadline = r.eval_deadline = float('inf')
        r.state = {}
        r.train_prefix = ['fake-python', 'fake-trainer']
        self.steps = {arm: 3652 for arm in module.ARMS}
        self.archives = {arm: {3652} for arm in module.ARMS}
        self.evaluated = {arm: set() for arm in module.ARMS}
        self.events = []
        r.progress = lambda arm: dict(step=self.steps[arm], complete=False)
        r.pending = lambda arm: sorted(self.archives[arm] - self.evaluated[arm])
        r.archive_target = lambda arm: self.archives[arm].add(r.target)
        r.update = lambda status, **values: r.state.update(status=status, **values)
        r.execute = self.execute
        # Any accidental process launch is a test failure, never a real job.
        self.run_patch = patch.object(module.subprocess, 'run', side_effect=AssertionError('Unexpected subprocess.run'))
        self.popen_patch = patch.object(module.subprocess, 'Popen', side_effect=AssertionError('Unexpected Popen'))
        self.run_patch.start()
        self.popen_patch.start()

    def tearDown(self):
        self.popen_patch.stop()
        self.run_patch.stop()
        self.temporary.cleanup()

    def execute(self, arm, kind, command):
        self.events.append((arm, kind, self.steps[arm]))
        if kind == 'train':
            self.steps[arm] += int(command[-1])
            if self.steps[arm] % 100 == 0:
                self.archives[arm].add(self.steps[arm])
        else:
            self.evaluated[arm].update(self.archives[arm])
        return None

    def run_locked(self):
        with (self.root / 'experiment.lock').open('a') as lock:
            self.runner.run_locked(lock)

    def test_balanced_segments_are_evaluated_before_next_training_and_target_is_archived(self):
        self.run_locked()
        self.assertEqual(self.events[:2], [('control', 'eval', 3652), ('direct_niche', 'eval', 3652)])
        self.assertEqual(self.steps, {'control': 3952, 'direct_niche': 3952})
        for arm in module.ARMS:
            self.assertEqual(self.archives[arm], {3652, 3700, 3800, 3900, 3952})
            self.assertEqual(self.evaluated[arm], self.archives[arm])
        training = [(arm, step) for arm, kind, step in self.events if kind == 'train']
        self.assertEqual(training, [('control', 3652), ('direct_niche', 3652),
                                    ('control', 3700), ('direct_niche', 3700),
                                    ('control', 3800), ('direct_niche', 3800),
                                    ('control', 3900), ('direct_niche', 3900)])
        for index, (arm, kind, _) in enumerate(self.events):
            if kind == 'train':
                self.assertEqual(self.events[index + 1][:2], (arm, 'eval'))
        complete = json.loads((self.root / 'AB_COMPLETE').read_text())
        self.assertTrue(complete['all_published_checkpoints_evaluated'])
        self.assertEqual(complete['target_step'], 3952)

    def test_deadline_mid_segment_saves_progress_and_reentry_prioritizes_lagging_arm(self):
        for arm in module.ARMS:
            self.evaluated[arm].add(3652)

        def interrupted_segment(arm, kind, command):
            self.assertEqual((arm, kind), ('control', 'train'))
            self.steps[arm] = 3660
            return 'allocation_deadline'

        self.runner.execute = interrupted_segment
        self.run_locked()
        self.assertEqual(self.runner.state['status'], 'allocation_deadline')
        self.assertEqual(self.steps, {'control': 3660, 'direct_niche': 3652})
        self.assertFalse((self.root / 'AB_COMPLETE').exists())
        self.assertFalse((self.root / 'PAUSE_REQUESTED').exists())
        self.runner.execute = self.execute
        self.run_locked()
        self.assertEqual(self.events[0], ('direct_niche', 'train', 3652))
        self.assertEqual(self.steps, {'control': 3952, 'direct_niche': 3952})
        self.assertTrue((self.root / 'AB_COMPLETE').exists())

    def test_explicit_experiment_pause_prevents_training_and_completion(self):
        (self.root / 'AB_PAUSE_REQUESTED').write_text('review requested')
        self.run_locked()
        self.assertEqual(self.runner.state['status'], 'paused')
        self.assertEqual(self.events, [])
        self.assertFalse((self.root / 'AB_COMPLETE').exists())

    def test_deadline_during_initialization_allows_unchanged_saved_boundary(self):
        for arm in module.ARMS:
            self.evaluated[arm].add(3652)
        self.runner.execute = lambda arm, kind, command: 'allocation_deadline'
        self.run_locked()
        self.assertEqual(self.runner.state['status'], 'allocation_deadline')
        self.assertEqual(self.runner.state['saved_step'], 3652)
        self.assertEqual(self.steps, {'control': 3652, 'direct_niche': 3652})
        self.assertFalse((self.root / 'PAUSE_REQUESTED').exists())
        self.assertFalse((self.root / 'AB_COMPLETE').exists())

    def test_stop_seen_by_trainer_before_first_update_is_not_false_failure(self):
        for arm in module.ARMS:
            self.evaluated[arm].add(3652)

        def stopped_during_initialization(arm, kind, command):
            # Trainer observes the marker and exits between coordinator polls.
            (self.root / 'STOP_REQUESTED').write_text('requested while model loaded')
            return None

        self.runner.execute = stopped_during_initialization
        self.run_locked()
        self.assertEqual(self.runner.state['status'], 'paused')
        self.assertEqual(self.runner.state['reason'], 'STOP_REQUESTED')
        self.assertEqual(self.runner.state['saved_step'], 3652)
        self.assertFalse((self.root / 'PAUSE_REQUESTED').exists())
        self.assertFalse((self.root / 'AB_COMPLETE').exists())

    def test_no_progress_without_stop_reason_remains_an_error(self):
        for arm in module.ARMS:
            self.evaluated[arm].add(3652)
        self.runner.execute = lambda arm, kind, command: None
        with self.assertRaisesRegex(RuntimeError, 'without advancing'):
            self.run_locked()


if __name__ == '__main__':
    unittest.main()
