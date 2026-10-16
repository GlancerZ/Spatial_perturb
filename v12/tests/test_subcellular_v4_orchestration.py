"""Controller sequencing and failure safety without GPUs or project data."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('controller_v4', ROOT / 'scripts/run_subcellular_v4.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = self.root / 'config.yaml'
        self.run = self.root / 'formal'
        cfg = yaml.safe_load((ROOT / 'configs/subcellular_geneset_v4_20260919.yaml').read_text())
        cfg['output_dir'] = str(self.run)
        self.config.write_text(yaml.safe_dump(cfg))
        self.args = argparse.Namespace(config=self.config, output_dir=None, resume=False,
                                       training_gpus=2, deadline=10000.,
                                       eval_min_seconds=600., train_min_seconds=120.)
        self.total = 100
        self.clock = 1000.
        self.commands = []
        self.on_train = None
        self.failed_kind = None
        self.no_evaluation_output = False
        self.advance = True
        self.runner = controller.Runner(self.args)
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(controller.time, 'time', side_effect=lambda: self.clock).start()
        mock.patch('builtins.print').start()
        self.popen = mock.patch.object(controller.subprocess, 'Popen', side_effect=self.fake_process).start()

    def save_progress(self, step):
        self.run.mkdir(exist_ok=True)
        (self.run / 'checkpoints').mkdir(exist_ok=True)
        (self.run / 'resume.pt').write_bytes(b'mocked optimizer checkpoint')
        (self.run / 'checkpoints' / f'step_{step:08d}.pt').write_bytes(b'mocked model checkpoint')
        controller.atomic_json(self.run / 'progress.json', dict(
            step=step, complete=step == self.total, model_version=4,
            architecture=controller.ARCHITECTURE, config_sha256=self.runner.fingerprint))
        controller.atomic_json(self.run / 'schedule.json', dict(
            total_updates=self.total, model_version=4, architecture=controller.ARCHITECTURE))

    def save_evaluation(self, step):
        output = self.run / 'downstream_validation' / f'step_{step:08d}'
        output.mkdir(parents=True, exist_ok=True)
        width = self.runner.cfg['model']['slots_per_compartment'] * self.runner.cfg['model']['slot_dim']
        dimensions = {name: width * (2 if name in ('base', 'intrinsic', 'joint') else 1)
                      for name in ('base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
                                   'nucleus_local', 'cytoplasm_local')}
        dimensions.update({f'{name}_neighbor_shuffle': dimensions[name]
                           for name in ('joint', 'nucleus', 'cytoplasm')})
        controller.atomic_json(output / 'metrics.json', dict(
            step=step, model_version=4, architecture=controller.ARCHITECTURE,
            checkpoint_sha256=hashlib.sha256((self.run / 'checkpoints' / f'step_{step:08d}.pt').read_bytes()).hexdigest(),
            evaluation_version=4, config_sha256=self.runner.fingerprint,
            cohort_sha256=controller.COHORT, cells=5200, canonical_cohort=True,
            test_donors_used=False, backbone_kind='scgpt', neighbor_shuffle=True,
            neighbor_shuffle_seed=815, diagnostic_schema_version=1,
            embedding_dimensions=dimensions,
            masked_validation={'enabled': True, 'requested_minimum_batches': 16},
            results=[dict(embedding=name, task=task,
                          fixed_C_result=dict(C=.1 if task == 'cell_type' else .01, val_macro_f1=.4))
                     for name in dimensions for task in ('cell_type', 'region')]))

    def fake_process(self, command, **kwargs):
        if '--initialize-only' in command:
            kind, step = 'initialization', 0
        elif 'scripts/train_subcellular_v4.py' in command:
            kind = 'training'
            previous = json.loads((self.run / 'progress.json').read_text())['step']
            limit = int(command[command.index('--run-update-limit') + 1])
            step = min(self.total, previous + limit) if self.advance else previous
        else:
            kind = 'evaluation'
            step = int(command[command.index('--expected-step') + 1])
        self.commands.append((kind, step, command, kwargs))
        process = mock.Mock(pid=987654)
        if kind == self.failed_kind:
            process.wait.return_value = 7
            return process
        if kind in ('initialization', 'training'):
            self.save_progress(step)
            if kind == 'training' and self.on_train:
                self.on_train()
        elif not self.no_evaluation_output:
            self.save_evaluation(step)
        process.wait.return_value = 0
        return process

    def status(self):
        return json.loads((self.runner.control / 'status.json').read_text())

    def run_controller(self, resume=False):
        self.args.resume = resume
        self.runner = controller.Runner(self.args)
        self.runner.run_foreground()

    def test_step_zero_then_100_and_complete_keep_original_config(self):
        original = self.config.read_bytes()
        self.run_controller()
        self.assertEqual([(kind, step) for kind, step, _, _ in self.commands],
                         [('initialization', 0), ('evaluation', 0), ('training', 100), ('evaluation', 100)])
        self.assertEqual(self.status()['status'], 'complete')
        self.assertEqual(self.config.read_bytes(), original)
        training = self.commands[2]
        self.assertIn('--resume', training[2])
        self.assertIn('--nproc_per_node=2', training[2])
        self.assertEqual(training[3]['env']['V4_TRAIN_DEADLINE'], '10000.0')
        self.assertTrue(training[3]['pass_fds'])
        for kind, _, command, _ in self.commands:
            if kind == 'evaluation':
                self.assertIn('--neighbor-shuffle', command)
                self.assertEqual(command[command.index('--mask-validation-batches') + 1], '16')
        history = [json.loads(line) for line in (self.runner.control / 'commands.jsonl').read_text().splitlines()]
        self.assertEqual([entry['returncode'] for entry in history], [0, 0, 0, 0])
        self.assertTrue(all(Path(entry['log']).is_file() for entry in history))

    def test_resume_evaluates_pending_before_training(self):
        self.total = 200
        self.on_train = lambda: setattr(self, 'clock', 9500.)
        self.run_controller()
        self.assertEqual(self.status()['status'], 'paused')
        self.assertEqual(self.status()['pending_evaluation_steps'], [100])
        self.assertEqual([(kind, step) for kind, step, _, _ in self.commands][-1], ('training', 100))
        self.commands.clear()
        self.clock = 1000.
        self.on_train = None
        self.run_controller(resume=True)
        self.assertEqual([(kind, step) for kind, step, _, _ in self.commands],
                         [('evaluation', 100), ('training', 200), ('evaluation', 200)])
        self.assertEqual(self.status()['status'], 'complete')

    def test_final_nonmultiple_of_100_is_evaluated(self):
        self.total = 125
        self.run_controller()
        self.assertEqual([(kind, step) for kind, step, _, _ in self.commands],
                         [('initialization', 0), ('evaluation', 0), ('training', 100),
                          ('evaluation', 100), ('training', 125), ('evaluation', 125)])
        self.assertEqual(self.status()['step'], 125)

    def test_resume_from_nonarchived_25_boundary_does_not_require_archive_25(self):
        self.clock = 9500.
        self.run_controller()
        self.save_progress(25)
        (self.run / 'checkpoints/step_00000025.pt').unlink()
        self.clock = 1000.
        self.commands.clear()
        self.run_controller(resume=True)
        self.assertEqual([(kind, step) for kind, step, _, _ in self.commands],
                         [('evaluation', 0), ('training', 100), ('evaluation', 100)])
        self.assertEqual(self.status()['status'], 'complete')

    def test_evaluation_failure_stops_without_starting_training(self):
        self.failed_kind = 'evaluation'
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.run_controller()
        self.assertEqual(failure.exception.returncode, 7)
        self.assertEqual([(kind, step) for kind, step, _, _ in self.commands],
                         [('initialization', 0), ('evaluation', 0)])
        self.assertEqual(self.status()['status'], 'failed')
        self.assertEqual(self.status()['last_child']['returncode'], 7)

    def test_success_exit_without_evaluation_output_is_failure(self):
        self.no_evaluation_output = True
        with self.assertRaisesRegex(RuntimeError, 'without valid metrics'):
            self.run_controller()
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.status()['status'], 'failed')

    def test_training_progress_must_increase(self):
        self.advance = False
        with self.assertRaisesRegex(ValueError, 'must increase'):
            self.run_controller()
        self.assertEqual(self.status()['status'], 'failed')
        self.assertEqual(len(self.commands), 3)

    def test_insufficient_evaluation_time_leaves_step_zero_pending(self):
        self.clock = 9500.
        self.run_controller()
        self.assertEqual([(kind, step) for kind, step, _, _ in self.commands], [('initialization', 0)])
        self.assertEqual(self.status()['status'], 'paused')
        self.assertEqual(self.status()['pending_evaluation_steps'], [0])
        self.assertTrue((self.run / 'checkpoints/step_00000000.pt').is_file())

    def test_existing_run_requires_explicit_resume_and_matching_identity(self):
        self.run.mkdir()
        existing = self.run / 'user-file'
        existing.write_text('keep')
        with self.assertRaisesRegex(FileExistsError, 'not empty'):
            self.run_controller()
        self.assertEqual(existing.read_text(), 'keep')
        self.assertFalse(self.runner.control.exists())
        with self.assertRaisesRegex(ValueError, 'matching controller identity'):
            self.run_controller(resume=True)

    def test_insufficient_initialization_time_does_not_claim_directories(self):
        self.clock = 9999.
        self.run_controller()
        self.assertFalse(self.run.exists())
        self.assertFalse(self.runner.control.exists())
        self.assertFalse(self.commands)

    def test_evaluation_is_bound_to_actual_checkpoint_bytes(self):
        self.run_controller()
        (self.run / 'checkpoints/step_00000100.pt').write_bytes(b'different model at same step')
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, 'checkpoint identity mismatch'):
            self.run_controller(resume=True)
        self.assertEqual(self.commands, [])

    def test_duplicate_controller_and_output_override_are_rejected(self):
        self.run_controller()
        with self.assertRaisesRegex(FileExistsError, 'explicit --resume'):
            self.run_controller()
        self.args.output_dir = self.root / 'different'
        with self.assertRaisesRegex(ValueError, 'must equal'):
            controller.Runner(self.args)

    def test_completed_run_resume_starts_no_process(self):
        self.run_controller()
        self.commands.clear()
        self.run_controller(resume=True)
        self.assertEqual(self.commands, [])
        self.assertEqual(self.status()['status'], 'complete')

    def test_wrong_cohort_metrics_are_not_accepted_as_complete(self):
        self.run_controller()
        metric = self.run / 'downstream_validation/step_00000100/metrics.json'
        report = json.loads(metric.read_text())
        report['cohort_sha256'] = 'different'
        controller.atomic_json(metric, report)
        self.commands.clear()
        with self.assertRaisesRegex(ValueError, 'Incompatible|incompatible'):
            self.run_controller(resume=True)
        self.assertEqual(self.commands, [])

    def test_evaluation_timeout_records_actual_code_and_keeps_pending(self):
        original = self.fake_process

        def timeout_evaluation(command, **kwargs):
            if 'scripts/evaluate_subcellular_v4.py' not in command:
                return original(command, **kwargs)
            process = mock.Mock(pid=987654)
            process.wait.side_effect = [subprocess.TimeoutExpired(command, 1), -signal.SIGTERM]
            self.commands.append(('evaluation', 0, command, kwargs))
            return process

        self.popen.side_effect = timeout_evaluation
        with mock.patch.object(controller.os, 'killpg') as kill:
            self.run_controller()
        kill.assert_called_once_with(987654, signal.SIGTERM)
        self.assertEqual(self.status()['status'], 'paused')
        self.assertEqual(self.status()['reason'], 'evaluation_deadline')
        self.assertEqual(self.status()['pending_evaluation_steps'], [0])
        self.assertEqual(self.status()['last_child']['returncode'], -signal.SIGTERM)
        self.assertEqual(len(self.commands), 2)


if __name__ == '__main__':
    unittest.main()
