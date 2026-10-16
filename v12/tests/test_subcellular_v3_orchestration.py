"""Stdlib-only orchestration regressions; no real CUDA, Slurm or children.

PyTorch/YAML imports in the compute coordinator are stubbed for these tests.
Temporary JSON is also valid YAML. All process creation is mocked and blocked
unless a test explicitly supplies its expected fake process.
"""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename, imports=None):
    specification = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(specification)
    with patch.dict(sys.modules, imports or {}):
        specification.loader.exec_module(module)
    return module


runner_module = load_script('v3_orchestration_runner', 'run_subcellular_v3.py', {
    'torch': SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4),
                             load=lambda path, **kwargs: json.loads(Path(path).read_text())),
    'yaml': SimpleNamespace(safe_load=json.loads),
})
supervisor_module = load_script('v3_orchestration_supervisor', 'supervise_subcellular_v2_allocations.py')


class EndPoll(BaseException):
    """Bound a foreground loop without simulating an algorithm failure."""


def process_result(stdout='', returncode=0):
    return SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / 'run'
        self.run.mkdir()
        (self.run / 'checkpoints').mkdir()
        (self.root / 'logs').mkdir()
        self.cfg = dict(experiment='dual_cls_binned_v3', output_dir=str(self.run),
                        training={'checkpoint_every': 100},
                        evaluation={'batch_size': 2, 'mask_validation_batches': 4})
        self.config_path = self.root / 'config.yaml'
        self.config_path.write_text(json.dumps(self.cfg))
        self.summary_commands = []

        def controlled_subprocess(command, **kwargs):
            if command[:2] == ['bash', 'scripts/watch_subcellular_v3_eval.sh'] and '--summary-only' in command:
                self.summary_commands.append(command)
                return process_result()
            raise AssertionError(f'Unmocked subprocess command: {command}')

        self.patches = [patch.object(runner_module, 'ROOT', self.root),
                        patch.object(supervisor_module, 'ROOT', self.root),
                        patch.object(supervisor_module, 'TASK_FAMILY', 'subcellular_v3'),
                        patch.object(supervisor_module, 'LOG_PREFIX', 'v3'),
                        patch.object(supervisor_module, 'TMUX_PREFIX', 'subcell_v3'),
                        patch.dict('os.environ', {'SLURM_JOB_ID': '101', 'V3_TRAIN_DEADLINE': 'inf',
                                                  'SPATIAL_EVAL_DEADLINE': 'inf'}),
                        patch.object(subprocess, 'Popen', side_effect=AssertionError('Unmocked child launch')),
                        patch.object(subprocess, 'run', side_effect=controlled_subprocess)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def runner(self, role='eval'):
        return runner_module.Runner(SimpleNamespace(config=self.config_path, role=role))

    def checkpoint(self, step):
        checkpoint = self.run / 'checkpoints' / f'step_{step:08d}.pt'
        checkpoint.write_bytes(f'fake checkpoint {step}'.encode())
        (self.run / 'resume.pt').write_text(json.dumps(dict(
            step=step, config_sha256=hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest())))
        (self.run / 'progress.json').write_text(json.dumps({'step': step}))
        return checkpoint

    def metric(self, step, **changes):
        output = self.run / 'downstream_validation' / f'step_{step:08d}'
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = self.run / 'checkpoints' / f'step_{step:08d}.pt'
        report = dict(step=step, evaluation_version=3, architecture='dual_cls_binned_v3',
                      model='dual_cls_binned_v3', backbone_kind='scgpt',
                      cohort_sha256='0ba9d43399081eebfd67a138c0d9659bc3c64552b5fe8d6c1578bf282d9ac9dc',
                      test_donors_used=False,
                      backbone_provenance={'backbone_kind': 'scgpt'},
                      config_sha256=hashlib.sha256(json.dumps(self.cfg, sort_keys=True).encode()).hexdigest(),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      masked_validation={'enabled': True})
        report.update(changes)
        (output / 'metrics.json').write_text(json.dumps(report))

    def write_intent(self, jobid='202'):
        claim = dict(jobid=jobid, pid=999, requested=1.)
        (self.run / 'four_gpu_intent.json').write_text(json.dumps(claim))
        return claim

    def test_all_children_inherit_role_lock_and_training_inherits_both_locks(self):
        runner = self.runner()
        child = SimpleNamespace(pid=999, returncode=0, poll=lambda: 0)
        with (self.run / 'owner.lock').open('a') as owner, (self.run / 'training.lock').open('a') as train, \
             (self.run / 'evaluation.lock').open('a') as evaluation:
            runner.owner = owner
            with patch.object(subprocess, 'Popen', return_value=child) as launch:
                runner.child(['fake-eval'], 'evaluation', float('inf'))
                self.assertEqual(set(launch.call_args.kwargs['pass_fds']), {owner.fileno()})
                runner.eval_lock = evaluation
                runner.child(['fake-eval'], 'evaluation', float('inf'))
                self.assertEqual(set(launch.call_args.kwargs['pass_fds']), {owner.fileno(), evaluation.fileno()})
                runner.eval_lock = None
                runner.lock = train
                runner.child(['fake-train'], 'training', float('inf'))
                self.assertEqual(set(launch.call_args.kwargs['pass_fds']), {owner.fileno(), train.fileno()})

    def test_active_or_pending_intent_never_clears(self):
        runner = self.runner()
        claim = self.write_intent()
        for state in ('RUNNING', 'PENDING', 'COMPLETING'):
            with self.subTest(state=state), patch.object(subprocess, 'run', return_value=process_result(state)) as call:
                self.assertTrue(runner.has_four_gpu_intent())
                self.assertEqual(json.loads(runner.intent.read_text()), claim)
                call.assert_called_once()

    def test_scheduler_errors_preserve_intent_without_terminating_runner(self):
        runner = self.runner()
        self.write_intent()
        for error in (OSError('scheduler unavailable'), subprocess.TimeoutExpired('squeue', 30)):
            with self.subTest(error=type(error).__name__), patch.object(subprocess, 'run', side_effect=error):
                self.assertTrue(runner.has_four_gpu_intent())
                self.assertTrue(runner.intent.exists())
        with patch.object(subprocess, 'run', return_value=process_result(returncode=1)) as call:
            self.assertTrue(runner.has_four_gpu_intent())
            call.assert_called_once()  # Failed squeue is not an empty queue.

    def test_accounting_lag_preserves_intent_but_confirmed_timeout_clears_it(self):
        runner = self.runner()
        self.write_intent()
        with patch.object(subprocess, 'run', side_effect=[process_result(), process_result()]):
            self.assertTrue(runner.has_four_gpu_intent())
        with patch.object(subprocess, 'run', side_effect=[process_result(), process_result('202|TIMEOUT|\n')]):
            self.assertFalse(runner.has_four_gpu_intent())
        self.assertFalse(runner.intent.exists())

    def test_old_terminal_claim_cannot_erase_replacement_allocation_intent(self):
        runner = self.runner()
        self.write_intent()

        def scheduler(command, **kwargs):
            if command[0] == 'squeue':
                return process_result()
            self.write_intent('303')
            return process_result('202|TIMEOUT|\n')

        with patch.object(subprocess, 'run', side_effect=scheduler):
            self.assertTrue(runner.has_four_gpu_intent())
        self.assertEqual(json.loads(runner.intent.read_text())['jobid'], '303')

    def test_bridge_initializes_evaluates_zero_trains_to_100_then_evaluates_100(self):
        runner = self.runner()
        events = []

        def child(command, kind, deadline):
            if len(events) > 5:
                self.fail('Unexpected unbounded bridge loop')
            if kind == 'initialization':
                self.assertIn('--initialize-only', command)
                self.checkpoint(0)
                events.append('initialize')
            elif kind == 'evaluation':
                step = int(command[command.index('--expected-step') + 1])
                self.metric(step)
                events.append(f'evaluate_{step}')
            elif kind == 'training':
                self.assertEqual(command[command.index('--run-update-limit') + 1], '100')
                self.assertTrue((self.run / 'downstream_validation/step_00000000/metrics.json').exists())
                self.checkpoint(100)
                (self.run / 'TRAIN_COMPLETE').touch()
                events.append('train_100')
            else:
                self.fail(kind)
            return True

        with patch.object(runner, 'child', side_effect=child):
            runner.bridge_and_evaluate()
        self.assertEqual(events, ['initialize', 'evaluate_0', 'train_100', 'evaluate_100'])
        self.assertEqual(len(self.summary_commands), 2)

    def test_evaluation_lock_is_held_during_child_and_released_afterward(self):
        runner = self.runner()
        self.checkpoint(0)

        def child(command, kind, deadline):
            self.assertEqual(kind, 'evaluation')
            self.assertIsNotNone(runner.eval_lock)
            with (self.run / 'evaluation.lock').open('a') as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.metric(0)
            return True

        with patch.object(runner, 'child', side_effect=child):
            self.assertTrue(runner.evaluate_pending())
        self.assertIsNone(runner.eval_lock)
        with (self.run / 'evaluation.lock').open('a') as contender:
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_existing_evaluation_waits_without_duplicate_child_or_runner_failure(self):
        runner = self.runner()
        self.checkpoint(0)
        with (self.run / 'evaluation.lock').open('a') as active_evaluation:
            fcntl.flock(active_evaluation, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(runner, 'child') as child, \
                 patch.object(runner_module.time, 'sleep', side_effect=EndPoll):
                with self.assertRaises(EndPoll):
                    runner.evaluate_pending()
            child.assert_not_called()

    def test_bridge_yields_to_active_four_gpu_intent_without_starting_training(self):
        runner = self.runner()
        self.write_intent()
        with patch.object(subprocess, 'run', return_value=process_result('RUNNING')), \
             patch.object(runner, 'child') as child, patch.object(runner_module.time, 'sleep', side_effect=EndPoll):
            with self.assertRaises(EndPoll):
                runner.bridge_and_evaluate()
        child.assert_not_called()

    def test_resume_step_is_authoritative_when_progress_sidecar_missing_or_stale(self):
        for sidecar in ('missing', 'stale'):
            with self.subTest(sidecar=sidecar):
                (self.run / 'TRAIN_COMPLETE').unlink(missing_ok=True)
                runner = self.runner()
                self.checkpoint(75)
                self.metric(75)
                if sidecar == 'missing':
                    (self.run / 'progress.json').unlink()
                else:
                    (self.run / 'progress.json').write_text(json.dumps({'step': 0}))

                def child(command, kind, deadline):
                    self.assertEqual(kind, 'training')
                    self.assertEqual(command[command.index('--run-update-limit') + 1], '25')
                    (self.run / 'TRAIN_COMPLETE').touch()
                    return True

                with patch.object(runner, 'child', side_effect=child) as launched, \
                     patch.object(runner_module.torch, 'load', wraps=runner_module.torch.load) as load:
                    runner.bridge_and_evaluate()
                launched.assert_called_once()
                load.assert_called_once_with(self.run / 'resume.pt', map_location='cpu', weights_only=False)

    def test_four_gpu_waits_for_step_zero_evaluation_and_locks_training(self):
        runner = self.runner('train')
        events = []

        def child(command, kind, deadline):
            self.assertIn('--nproc_per_node=4', command)
            if kind == 'initialization':
                self.assertIn('--initialize-only', command)
                self.checkpoint(0)
                events.append('initialize')
            else:
                self.assertEqual(kind, 'training')
                self.assertTrue((self.run / 'downstream_validation/step_00000000/metrics.json').exists())
                # A separate open-file description cannot acquire the held lock.
                with (self.run / 'training.lock').open('a') as contender:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                events.append('train')
            return True

        def evaluation_finishes(seconds):
            self.assertEqual(events, ['initialize'])
            # The training lock must be released while waiting for evaluation.
            with (self.run / 'training.lock').open('a') as contender:
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
            events.append('evaluate_0')
            self.metric(0)

        with patch.object(runner, 'child', side_effect=child), \
             patch.object(runner_module.time, 'sleep', side_effect=evaluation_finishes):
            runner.train_four()
        self.assertEqual(events, ['initialize', 'evaluate_0', 'train'])
        self.assertFalse(runner.intent.exists())

    def test_pause_prevents_any_bridge_child(self):
        runner = self.runner()
        (self.run / 'PAUSE_REQUESTED').touch()
        with patch.object(runner, 'child') as child:
            runner.bridge_and_evaluate()
        child.assert_not_called()

    def test_old_evaluation_version_cannot_be_accepted_as_completed(self):
        runner = self.runner()
        self.checkpoint(0)
        self.metric(0, evaluation_version=2)
        with self.assertRaises(ValueError):
            runner.pending()

    def test_mismatched_evaluation_provenance_never_opens_four_gpu_training_gate(self):
        runner = self.runner('train')
        self.checkpoint(0)
        for change in ({'evaluation_version': 2}, {'config_sha256': 'wrong-run'},
                       {'cohort_sha256': 'wrong-cohort'}, {'test_donors_used': True},
                       {'backbone_kind': 'scgpt_spatial'}, {'masked_validation': {'enabled': False}}):
            with self.subTest(change=change):
                self.metric(0, **change)
                with patch.object(runner, 'child') as child:
                    with self.assertRaises(ValueError):
                        runner.train_four()
                child.assert_not_called()

    def supervisor_args(self, launch_initial=True):
        return SimpleNamespace(role='train', jobid='101', run_dir=self.run,
                               max_new_allocations=3, poll_seconds=30, quiet_console=True,
                               launch_initial=launch_initial, reuse_idle_job=[])

    def test_launch_initial_waits_pending_then_launches_correct_v3_holder_once(self):
        controller = supervisor_module.Supervisor(self.supervisor_args())
        pending = {'101': dict(jobid='101', state='PENDING', name='authorized_initial')}
        running = {'101': dict(jobid='101', state='RUNNING', name='authorized_initial')}
        process = SimpleNamespace(pid=999, returncode=None, poll=lambda: None)
        with patch.object(supervisor_module, 'queue_snapshot', side_effect=[pending, running]), \
             patch.object(subprocess, 'Popen', return_value=process) as launch, \
             patch.object(supervisor_module.time, 'sleep', side_effect=[None, EndPoll]):
            with self.assertRaises(EndPoll):
                controller.run_foreground()
        self.addCleanup(controller.child_log.close)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0][-1], 'bash scripts/hold_subcellular_v3_task.sh train')
        self.assertTrue(controller.state['jobs']['101']['launch_attempted'])
        self.assertEqual(controller.state['new_allocations'], 0)

    def test_launch_initial_on_restart_cannot_relaunch_recorded_running_task(self):
        args = self.supervisor_args()
        first = supervisor_module.Supervisor(args)
        first.state['jobs']['101']['launch_attempted'] = True
        first.update('compute_launched')
        controller = supervisor_module.Supervisor(args)
        running = {'101': dict(jobid='101', state='RUNNING', name='authorized_initial')}
        with patch.object(supervisor_module, 'queue_snapshot', return_value=running), \
             patch.object(supervisor_module, 'compute_session_present', return_value=True) as reconcile, \
             patch.object(subprocess, 'Popen') as launch, \
             patch.object(supervisor_module.time, 'sleep', side_effect=EndPoll):
            with self.assertRaises(EndPoll):
                controller.run_foreground()
        reconcile.assert_called_once_with('train', '101')
        launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
