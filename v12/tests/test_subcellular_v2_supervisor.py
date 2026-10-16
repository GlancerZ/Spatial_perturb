"""Stdlib-only scheduler QA: all Slurm commands and compute launch are mocked."""
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('v2_allocation_supervisor',
    Path(__file__).resolve().parents[1] / 'scripts/supervise_subcellular_v2_allocations.py')
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class EndPoll(BaseException):
    """End one mock monitoring iteration without being mistaken for I/O failure."""


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.root_patch = patch.object(module, 'ROOT', self.root)
        self.root_patch.start()
        (self.root / 'logs').mkdir()
        self.run = self.root / 'outputs/run'
        self.run.mkdir(parents=True)
        self.args = SimpleNamespace(role='train', jobid='101', run_dir=self.run,
                                    max_new_allocations=3, poll_seconds=30)
        self.live = {'101': {'jobid': '101', 'state': 'RUNNING', 'name': 'existing_training'}}
        self.final = {'state': 'TIMEOUT', 'terminal': True, 'slurm_exit_code': '0:15'}
        # Never permit any actual child process to escape a mock in these tests.
        self.popen_patch = patch.object(module.subprocess, 'Popen', side_effect=AssertionError('Unmocked compute launch'))
        self.popen = self.popen_patch.start()
        self.run_patch = patch.object(module.subprocess, 'run', side_effect=AssertionError('Unmocked Slurm command'))
        self.subprocess_run = self.run_patch.start()

    def tearDown(self):
        self.run_patch.stop()
        self.popen_patch.stop()
        self.root_patch.stop()
        self.temporary.cleanup()

    def write_exit(self, status):
        (self.root / 'logs/v2_train_101.exit').write_text(str(status))

    def supervisor(self):
        return module.Supervisor(self.args)

    def test_running_allocation_never_submits(self):
        controller = self.supervisor()
        with patch.object(module, 'queue_snapshot', return_value=self.live), \
             patch.object(module, 'compute_session_present', return_value=True), \
             patch.object(module, 'scheduler') as scheduler, \
             patch.object(module.time, 'sleep', side_effect=EndPoll):
            with self.assertRaises(EndPoll):
                controller.run_foreground()
        scheduler.assert_not_called()
        self.assertEqual(controller.state['new_allocations'], 0)
        self.popen.assert_not_called()

    def test_confirmed_timeout_signal_and_resume_can_renew(self):
        self.write_exit(143)
        (self.run / 'resume.pt').touch()
        controller = self.supervisor()
        controller.state['jobs']['101']['holder_exit_code'] = 1
        with patch.object(module, 'queue_snapshot', return_value={}), \
             patch.object(module, 'terminal_state', return_value=self.final), \
             patch.object(module, 'scheduler', return_value='202\n') as scheduler, \
             patch.object(module.time, 'sleep', side_effect=EndPoll):
            with self.assertRaises(EndPoll):
                controller.run_foreground()
        scheduler.assert_called_once()
        self.assertEqual(scheduler.call_args.args[0][0], 'sbatch')
        self.assertIn('--gpus-per-node=4', scheduler.call_args.args[0])
        self.assertEqual(controller.state['current_jobid'], '202')
        self.assertFalse(controller.state['submission_pending'])
        self.popen.assert_not_called()

    def test_algorithm_failure_never_renews_even_after_timeout(self):
        self.write_exit(1)
        (self.run / 'resume.pt').touch()
        controller = self.supervisor()
        with patch.object(module, 'queue_snapshot', return_value={}), \
             patch.object(module, 'terminal_state', return_value=self.final), \
             patch.object(module, 'scheduler') as scheduler:
            controller.run_foreground()
        scheduler.assert_not_called()
        self.assertEqual(controller.state['status'], 'action_required')

    def test_active_algorithm_failure_never_renews(self):
        self.write_exit(1)
        controller = self.supervisor()
        with patch.object(module, 'queue_snapshot', return_value=self.live), \
             patch.object(module, 'scheduler') as scheduler:
            controller.run_foreground()
        scheduler.assert_not_called()
        self.assertEqual(controller.state['status'], 'action_required')

    def test_squeue_error_is_not_completion(self):
        controller = self.supervisor()
        with patch.object(module, 'queue_snapshot', side_effect=RuntimeError('Transient scheduler error')), \
             patch.object(module, 'scheduler') as scheduler, \
             patch.object(module.time, 'sleep', side_effect=EndPoll):
            with self.assertRaises(EndPoll):
                controller.run_foreground()
        scheduler.assert_not_called()
        self.assertEqual(controller.state['status'], 'scheduler_or_monitor_error')
        self.assertEqual(controller.state['new_allocations'], 0)

    def test_stop_and_complete_prevent_allocation_requests(self):
        for marker in ('STOP_REQUESTED', 'PAUSE_REQUESTED', 'TRAIN_COMPLETE'):
            with self.subTest(marker=marker):
                (self.run / marker).touch()
                controller = self.supervisor()
                with patch.object(module, 'queue_snapshot') as queue, \
                     patch.object(module, 'scheduler') as scheduler:
                    controller.run_foreground()
                queue.assert_not_called()
                scheduler.assert_not_called()
                self.assertEqual(controller.state['status'], 'finished')
                (self.run / marker).unlink()

    def test_lost_sbatch_response_cannot_be_retried_or_restarted(self):
        self.write_exit(0)
        controller = self.supervisor()
        with patch.object(module, 'queue_snapshot', return_value={}), \
             patch.object(module, 'terminal_state', return_value=self.final), \
             patch.object(module, 'scheduler', side_effect=subprocess.TimeoutExpired('sbatch', 45)) as scheduler:
            controller.run_foreground()
        scheduler.assert_called_once()
        self.assertEqual(controller.state['status'], 'action_required')
        self.assertTrue(controller.state['submission_pending'])
        with self.assertRaisesRegex(RuntimeError, 'requires inspection'):
            self.supervisor()
        self.assertEqual(scheduler.call_count, 1)

    def test_launch_flag_without_tmux_needs_inspection_not_duplicate_launch(self):
        controller = self.supervisor()
        with patch.object(module, 'queue_snapshot', return_value=self.live), \
             patch.object(module, 'compute_session_present', return_value=False) as sessions, \
             patch.object(module, 'scheduler') as scheduler, \
             patch.object(module.time, 'sleep', return_value=None):
            controller.run_foreground()
        self.assertEqual(sessions.call_count, 2)
        scheduler.assert_not_called()
        self.popen.assert_not_called()
        self.assertEqual(controller.state['status'], 'action_required')

    def test_signal_code_alone_does_not_authorize_recovery(self):
        for state in ('RUNNING', 'CANCELLED', 'FAILED', 'OUT_OF_MEMORY', 'COMPLETED'):
            self.assertFalse(module.terminal_recovery_allowed(state, 143, 1, True))
        for state in ('TIMEOUT', 'PREEMPTED', 'NODE_FAIL'):
            self.assertTrue(module.terminal_recovery_allowed(state, 137, 1, True))
            self.assertFalse(module.terminal_recovery_allowed(state, 1, 1, True))
            self.assertFalse(module.terminal_recovery_allowed(state, 143, 1, False))

    def test_eval_completion_requires_all_published_checkpoints(self):
        self.args.role = 'eval'
        (self.run / 'TRAIN_COMPLETE').touch()
        checkpoints = self.run / 'checkpoints'
        checkpoints.mkdir()
        (checkpoints / 'step_00000100.pt').touch()
        controller = self.supervisor()
        self.assertIsNone(controller.stopping_reason())
        destination = self.run / 'downstream_validation/step_00000100'
        destination.mkdir(parents=True)
        (destination / 'metrics.json').write_text(json.dumps(dict(step=100, evaluation_version=2,
            config_sha256='test', masked_validation={'enabled': True})))
        self.assertEqual(controller.stopping_reason(), 'TRAIN_COMPLETE_AND_ALL_CHECKPOINTS_EVALUATED')

    def test_plain_scgpt_supervision_uses_separate_holder_and_exit_records(self):
        self.write_exit(1)  # Historical-family records must not stop a new run.
        with patch.object(module, 'TASK_FAMILY', 'subcellular_scgpt'), \
             patch.object(module, 'LOG_PREFIX', 'scgpt'), \
             patch.object(module, 'TMUX_PREFIX', 'subcell_scgpt'):
            self.assertIsNone(module.exit_status('train', '101'))
            self.assertEqual(module.launch_command('train', '101')[-1],
                             'bash scripts/hold_subcellular_scgpt_task.sh train')
            (self.root / 'logs/scgpt_train_101.exit').write_text('0')
            self.assertEqual(module.exit_status('train', '101'), 0)
            with patch.object(module, 'scheduler', return_value='V2_TASK_RUNNING\n') as scheduler:
                self.assertTrue(module.compute_session_present('train', '101'))
                self.assertIn('subcell_scgpt_train_101', scheduler.call_args.args[0][-1])

    def test_closed_stdout_does_not_prevent_durable_logging_or_one_renewal(self):
        self.write_exit(0)
        controller = self.supervisor()
        closed_stdout = io.StringIO()
        closed_stdout.close()
        with patch.object(module.sys, 'stdout', closed_stdout), \
             patch.object(module, 'queue_snapshot', return_value={}), \
             patch.object(module, 'terminal_state', return_value=dict(state='COMPLETED', terminal=True)), \
             patch.object(module, 'scheduler', return_value='202\n') as scheduler, \
             patch.object(module.time, 'sleep', side_effect=EndPoll):
            with self.assertRaises(EndPoll):
                controller.run_foreground()
        scheduler.assert_called_once()
        saved = json.loads(controller.path.read_text())
        self.assertEqual(saved['current_jobid'], '202')
        self.assertEqual(saved['new_allocations'], 1)
        self.assertFalse(saved['submission_pending'])
        self.assertFalse(controller.console_enabled)
        records = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
        self.assertIn('allocation_submitted', [record['status'] for record in records])

    def test_console_break_after_sbatch_acceptance_cannot_mark_submission_uncertain_or_repeat(self):
        self.write_exit(0)
        controller = self.supervisor()
        observed_statuses = []

        def print_with_disconnected_consumer(line, **kwargs):
            status = json.loads(line)['status']
            observed_statuses.append(status)
            if status == 'allocation_submitted':
                raise BrokenPipeError(32, 'SSH console disconnected after sbatch acceptance')

        new_pending = {'202': {'jobid': '202', 'state': 'PENDING', 'name': controller.job_name}}
        with patch('builtins.print', side_effect=print_with_disconnected_consumer), \
             patch.object(module, 'queue_snapshot', side_effect=[{}, new_pending]), \
             patch.object(module, 'terminal_state', return_value=dict(state='COMPLETED', terminal=True)), \
             patch.object(module, 'scheduler', return_value='202\n') as scheduler, \
             patch.object(module.time, 'sleep', side_effect=[None, EndPoll]):
            with self.assertRaises(EndPoll):
                controller.run_foreground()
        scheduler.assert_called_once()
        saved = json.loads(controller.path.read_text())
        self.assertEqual(saved['current_jobid'], '202')
        self.assertEqual(saved['status'], 'allocation_pending')
        self.assertFalse(saved['submission_pending'])
        self.assertEqual(saved['known_job_ids'], ['101', '202'])
        self.assertFalse(controller.console_enabled)
        self.assertIn('allocation_submitted', observed_statuses)
        self.assertNotIn('action_required', observed_statuses)
        self.assertEqual(self.supervisor().state['current_jobid'], '202')

    def test_durable_log_failure_is_not_swallowed_as_a_console_failure(self):
        controller = self.supervisor()
        controller.log_path.mkdir()
        with patch('builtins.print') as console:
            with self.assertRaises(IsADirectoryError):
                controller.update('test_durable_file_failure')
        console.assert_not_called()

    def test_quiet_console_keeps_durable_state_and_log(self):
        self.args.quiet_console = True
        controller = self.supervisor()
        with patch('builtins.print') as console:
            controller.update('test_quiet_console')
        console.assert_not_called()
        self.assertEqual(json.loads(controller.path.read_text())['status'], 'test_quiet_console')
        self.assertEqual(json.loads(controller.log_path.read_text())['status'], 'test_quiet_console')

    def test_explicit_idle_running_job_is_reused_without_new_submission(self):
        self.args.reuse_idle_job = ['303']
        controller = self.supervisor()
        available = {'303': {'jobid': '303', 'state': 'RUNNING', 'name': 'existing_authorized_hold'}}
        with patch.object(module, 'scheduler', return_value='303.batch\n303.extern\n') as scheduler:
            self.assertTrue(controller.renew(available))
        self.assertEqual(scheduler.call_args.args[0][0], 'squeue')
        scheduler.assert_called_once()
        self.assertEqual(controller.state['current_jobid'], '303')
        self.assertEqual(controller.state['new_allocations'], 0)
        self.assertFalse(controller.state['jobs']['303']['launch_attempted'])

    def test_busy_or_pending_explicit_job_is_not_adopted(self):
        self.args.reuse_idle_job = ['303']
        self.args.max_new_allocations = 0
        for state, steps in [('RUNNING', '303.0\n'), ('PENDING', '')]:
            with self.subTest(state=state):
                controller = self.supervisor()
                with patch.object(module, 'scheduler', return_value=steps) as scheduler:
                    self.assertFalse(controller.renew({'303': {'jobid': '303', 'state': state, 'name': 'hold'}}))
                self.assertNotIn('303', controller.state['known_job_ids'])
                if state == 'PENDING':
                    scheduler.assert_not_called()
                controller.path.unlink(missing_ok=True)

    def test_readout_ab_completion_stops_before_querying_or_requesting_resources(self):
        (self.run / 'AB_COMPLETE').touch()
        with patch.object(module, 'TASK_FAMILY', 'subcellular_readout_ab'):
            controller = self.supervisor()
            with patch.object(module, 'queue_snapshot') as queue:
                controller.run_foreground()
            queue.assert_not_called()
        self.assertEqual(controller.state['reason'], 'AB_COMPLETE')

    def test_readout_ab_honors_arm_pause_but_allows_owned_deadline_marker(self):
        arm = self.run / 'control'
        arm.mkdir()
        with patch.object(module, 'TASK_FAMILY', 'subcellular_readout_ab'):
            controller = self.supervisor()
            (arm / 'STOP_REQUESTED').write_text('readout_ab_runner: allocation_deadline\n')
            self.assertIsNone(controller.stopping_reason())
            (arm / 'PAUSE_REQUESTED').touch()
            self.assertEqual(controller.stopping_reason(), 'control/PAUSE_REQUESTED')
            (self.run / 'AB_PAUSE_REQUESTED').touch()
            self.assertEqual(controller.stopping_reason(), 'AB_PAUSE_REQUESTED')


if __name__ == '__main__':
    unittest.main()
