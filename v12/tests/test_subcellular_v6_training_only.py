"""Training-only orchestration contracts; run only on an authorized node.

Temporary JSON checkpoints and mocked child calls exercise orchestration without
starting a trainer, evaluator, subprocess or Slurm operation.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_subcellular_v6 import ARCHITECTURE, Runner as InlineRunner
from run_subcellular_v6_training_only import EXECUTION_MODE_SCHEMA, Runner


class V6TrainingOnlyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        self.config = dict(
            output_dir=str(self.run), model={'route': 'A'},
            training=dict(epochs=1, checkpoint_every=100, resume_every=25,
                          global_batch_size=64, batch_size=16),
            evaluation=dict(interval=100, cells_per_donor=400,
                            mask_validation_batches=16, test_donors=False))
        self.config_path = self.root / 'config.yaml'
        self.config_path.write_text(yaml.safe_dump(self.config))
        self.args = SimpleNamespace(
            config=self.config_path, output_dir=None, resume=True,
            training_gpus=4, nodes=1, gpus_per_node=4, deadline=time.time() + 3600,
            eval_min_seconds=600, train_min_seconds=120)
        self.runner = Runner(self.args)
        self.runner.control.mkdir()
        (self.runner.control / 'identity.json').write_text(json.dumps(self.runner.identity))
        (self.runner.control / 'status.json').write_text(json.dumps({'status': 'paused', 'original': True}))
        (self.run / 'resume.pt').write_bytes(b'preserve original optimizer state')
        (self.run / 'checkpoints').mkdir()
        (self.run / 'checkpoints/step_00000100.pt').write_bytes(b'preserve archived checkpoint')
        self.progress(175)
        for name in ('evaluation_command', 'evaluate_pending', 'evaluated', 'pending'):
            patch = mock.patch.object(self.runner, name, side_effect=AssertionError('Training must not evaluate'))
            patch.start()
            self.addCleanup(patch.stop)

    def progress(self, step, total=450, *, complete=False):
        (self.run / 'schedule.json').write_text(json.dumps(dict(
            model_version=6, architecture=ARCHITECTURE, total_updates=total, updates_per_epoch=total)))
        (self.run / 'progress.json').write_text(json.dumps(dict(
            model_version=6, architecture=ARCHITECTURE, config_sha256=self.runner.fingerprint,
            step=step, epoch=1 if complete else 0, next_update=0 if complete else step,
            complete=complete)))
        if complete:
            (self.run / 'TRAIN_COMPLETE').write_text(f'step={step}\n')

    def status(self):
        return json.loads((self.runner.control / 'status.json').read_text())

    def test_resume_uses_same_identity_and_one_full_remaining_budget_without_evaluation(self):
        identity = copy.deepcopy(self.runner.identity)
        self.assertEqual(identity, InlineRunner(self.args).identity)

        def finish(command, kind, step):
            self.assertEqual(kind, 'training')
            self.assertEqual(step, 175)
            self.assertEqual(command[-3:], ['--resume', '--run-update-limit', '275'])
            self.assertEqual(command[3:5], ['1', '4'])
            self.progress(450, complete=True)
            return True

        with mock.patch.object(self.runner, 'child', side_effect=finish) as child:
            self.runner.run_foreground()
        self.assertEqual(child.call_count, 1)
        state = self.status()
        self.assertEqual(state['status'], 'training_complete')
        self.assertEqual((state['training_step'], state['step'], state['total_updates']), (450, 450, 450))
        self.assertFalse(state['evaluation_completion_verified'])
        self.assertNotEqual(state['status'], 'complete')
        self.assertFalse((self.run / 'downstream_validation').exists())
        self.assertEqual(json.loads((self.runner.control / 'identity.json').read_text()), identity)
        self.assertEqual((self.run / 'checkpoints/step_00000100.pt').read_bytes(), b'preserve archived checkpoint')
        self.assertEqual((self.run / 'resume.pt').read_bytes(), b'preserve original optimizer state')
        self.assertEqual(yaml.safe_load(self.config_path.read_text()), self.config)
        self.assertIsNone(self.runner.lock)
        expected_mode = dict(schema=EXECUTION_MODE_SCHEMA, mode='training_only',
                             config_sha256=self.runner.fingerprint, output_dir=str(self.run))
        self.assertEqual(json.loads((self.runner.control / 'execution_mode.json').read_text()), expected_mode)

    def test_stop_markers_pause_without_launch_and_are_never_cleared(self):
        for name in ('STOP_REQUESTED', 'PAUSE_REQUESTED', 'HANDOFF_TO_4GPU_REQUESTED'):
            with self.subTest(marker=name):
                marker = self.run / name
                marker.write_text('caller owns this marker')
                with mock.patch.object(self.runner, 'child') as child:
                    self.runner.run_foreground()
                child.assert_not_called()
                self.assertEqual(self.status()['reason'], 'stop_marker')
                self.assertEqual(self.status()['training_step'], 175)
                self.assertEqual(marker.read_text(), 'caller owns this marker')
                marker.unlink()  # Test fixture only; production never clears it.

    def test_foreign_execution_mode_is_rejected_without_replacing_status_or_marker(self):
        mode = dict(schema=EXECUTION_MODE_SCHEMA, mode='inline_evaluation',
                    config_sha256=self.runner.fingerprint, output_dir=str(self.run))
        path = self.runner.control / 'execution_mode.json'
        path.write_text(json.dumps(mode))
        original_status = (self.runner.control / 'status.json').read_bytes()
        with mock.patch.object(self.runner, 'child') as child:
            with self.assertRaisesRegex(ValueError, 'Existing execution mode'):
                self.runner.run_foreground()
        child.assert_not_called()
        self.assertEqual(json.loads(path.read_text()), mode)
        self.assertEqual((self.runner.control / 'status.json').read_bytes(), original_status)
        self.assertIsNone(self.runner.lock)
        with (self.runner.control / 'owner.lock').open('a') as handle:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_existing_inline_controller_lock_blocks_training_only(self):
        owner = InlineRunner(self.args)
        owner.prepare()
        self.addCleanup(owner.lock.close)
        with mock.patch.object(self.runner, 'child') as child:
            with self.assertRaises(BlockingIOError):
                self.runner.run_foreground()
        child.assert_not_called()
        self.assertFalse((self.runner.control / 'execution_mode.json').exists())

    def test_safety_deadline_pauses_at_new_saved_boundary_with_original_jobid(self):
        def stop_at_deadline(command, kind, step):
            self.progress(200)
            self.args.deadline = time.time() - 1
            return True

        with (mock.patch.dict('os.environ', {'SLURM_JOB_ID': '123456'}),
              mock.patch.object(self.runner, 'child', side_effect=stop_at_deadline) as child):
            self.runner.run_foreground()
        self.assertEqual(child.call_count, 1)
        state = self.status()
        self.assertEqual(state['status'], 'paused')
        self.assertEqual(state['reason'], 'training_time_budget')
        self.assertEqual(state['jobid'], '123456')
        self.assertEqual((state['training_step'], state['total_updates']), (200, 450))

    def test_short_launch_budget_does_not_start_trainer(self):
        self.args.deadline = time.time() + 10
        with mock.patch.object(self.runner, 'child') as child:
            self.runner.run_foreground()
        child.assert_not_called()
        self.assertEqual(self.status()['reason'], 'training_time_budget')

    def test_real_child_failure_is_preserved_and_not_retried(self):
        failure = subprocess.CalledProcessError(17, ['unchanged-trainer'])
        with mock.patch.object(self.runner, 'child', side_effect=failure) as child:
            with self.assertRaises(subprocess.CalledProcessError) as caught:
                self.runner.run_foreground()
        self.assertIs(caught.exception, failure)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(self.status()['status'], 'failed')

    def test_unexplained_zero_exit_or_changed_epoch_budget_is_rejected(self):
        for changed_total in (450, 451):
            with self.subTest(total=changed_total):
                self.progress(175)

                def unexpected_exit(command, kind, step):
                    self.progress(200, total=changed_total)
                    return True

                with mock.patch.object(self.runner, 'child', side_effect=unexpected_exit) as child:
                    with self.assertRaises(ValueError):
                        self.runner.run_foreground()
                self.assertEqual(child.call_count, 1)
                self.assertEqual(self.status()['status'], 'failed')

    def test_complete_training_does_not_require_or_claim_completed_evaluations(self):
        self.progress(450, complete=True)
        with mock.patch.object(self.runner, 'child') as child:
            self.runner.run_foreground()
        child.assert_not_called()
        self.assertEqual(self.status()['status'], 'training_complete')
        self.assertFalse(self.status()['evaluation_completion_verified'])

    def test_missing_complete_marker_cannot_claim_training_finished(self):
        self.progress(450, complete=True)
        (self.run / 'TRAIN_COMPLETE').unlink()
        with self.assertRaisesRegex(ValueError, 'complete epoch checkpoint boundary'):
            self.runner.run_foreground()
        self.assertEqual(self.status()['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
