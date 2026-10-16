"""Acceptance must inspect both optimizer states and the actual launch shape."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from validate_subcellular_v6 import (
    ARCHITECTURE, config_hash, inspect_optimizer_state, lr_multiplier,
    optimizer_evidence, optimizer_state_fingerprint, resolve_topology,
    training_command, validate_rank_topology,
)
from train_subcellular_v6 import GlobalBatchSampler, logical_forward_seed, validate_resume


class OptimizerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.cfg = {'training': {'lr_muon': .002, 'lr_new': .0001,
                                'warmup_ratio': .1, 'min_lr_ratio': .1,
                                'global_batch_size': 64, 'batch_size': 16, 'epochs': 1}}
        self.manifest = {'optimizer': 'muon', 'fixture': 'two-independent-parameter-groups'}
        self.optimizer = self.state_at(2)

    def state_at(self, step):
        factor = lr_multiplier(step, 100, self.cfg['training'])
        return {
            'schema': 'subcellular_v6_muon_v1', 'steps_completed': step,
            'manifest': copy.deepcopy(self.manifest),
            'initialized_parameters': {'muon': ['hidden.weight'], 'aux_adamw': ['hidden.bias']},
            'muon': {
                'state': {0: {'momentum_buffer': torch.ones(2, 3)}},
                'param_groups': [dict(name='muon_hidden', use_muon=True, params=[0],
                                      param_names=['hidden.weight'], base_lr=.002, lr=.002 * factor)],
            },
            'aux_adamw': {
                'state': {0: {'step': torch.tensor(float(step)),
                              'exp_avg': torch.ones(3), 'exp_avg_sq': torch.full((3,), .1)}},
                'param_groups': [dict(name='adamw_aux', use_muon=False, params=[0],
                                      param_names=['hidden.bias'], base_lr=.0001, lr=.0001 * factor)],
            },
        }

    def inspect(self, state=None, step=2, **kwargs):
        return inspect_optimizer_state(self.optimizer if state is None else state,
                                        self.cfg, step, 100, self.manifest, **kwargs)

    def test_records_distinct_muon_and_adamw_state_and_scheduled_rates(self):
        result = self.inspect()
        self.assertEqual(result['steps_completed'], 2)
        self.assertEqual(result['buffers']['muon']['parameter_states'], 1)
        self.assertEqual(result['buffers']['adamw']['step_values'], [2])
        self.assertAlmostEqual(result['learning_rates']['muon'], .0004)
        self.assertAlmostEqual(result['learning_rates']['adamw'], .00002)
        self.assertNotEqual(result['buffers']['muon']['state_sha256'],
                            result['buffers']['adamw']['state_sha256'])

    def test_rejects_old_adamw_schema_missing_muon_momentum_or_aux_moments(self):
        wrong = copy.deepcopy(self.optimizer)
        wrong['schema'] = 'legacy_adamw'
        with self.assertRaisesRegex(AssertionError, 'schema'):
            self.inspect(wrong)
        for branch, buffer in (('muon', 'momentum_buffer'), ('aux_adamw', 'exp_avg'),
                               ('aux_adamw', 'exp_avg_sq')):
            wrong = copy.deepcopy(self.optimizer)
            del wrong[branch]['state'][0][buffer]
            with self.subTest(branch=branch, buffer=buffer), self.assertRaisesRegex(AssertionError, 'buffer'):
                self.inspect(wrong)

    def test_rejects_reset_adamw_counter_even_if_wrapper_counter_is_correct(self):
        wrong = self.state_at(3)
        wrong['aux_adamw']['state'][0]['step'] = torch.tensor(1.)
        with self.assertRaisesRegex(AssertionError, 'step did not persist'):
            self.inspect(wrong, step=3)

    def test_rejects_nonfinite_or_zeroed_momentum_after_two_updates(self):
        for value in (float('nan'), 0.):
            wrong = copy.deepcopy(self.optimizer)
            wrong['muon']['state'][0]['momentum_buffer'].fill_(value)
            with self.subTest(value=value), self.assertRaises(AssertionError):
                self.inspect(wrong)
        # At QA update one, a zero residual projection can legitimately leave
        # hidden Muon matrices with zero gradients. We still check exact state.
        first = self.state_at(1)
        first['muon']['state'][0]['momentum_buffer'].zero_()
        self.inspect(first, step=1, require_nonzero=False)

    def test_rejects_lr_assignment_that_accidentally_uses_adamw_rate_for_muon(self):
        wrong = copy.deepcopy(self.optimizer)
        wrong['muon']['param_groups'][0]['lr'] = wrong['aux_adamw']['param_groups'][0]['lr']
        with self.assertRaisesRegex(AssertionError, 'LR'):
            self.inspect(wrong)

    def test_state_hash_preserves_values_dtype_shape_and_scalar_counter(self):
        exact = copy.deepcopy(self.optimizer)
        self.assertEqual(optimizer_state_fingerprint(exact), optimizer_state_fingerprint(self.optimizer))
        exact['muon']['state'][0]['momentum_buffer'][0, 0] += 1
        self.assertNotEqual(optimizer_state_fingerprint(exact), optimizer_state_fingerprint(self.optimizer))
        self.assertNotEqual(optimizer_state_fingerprint(torch.tensor(1.)),
                            optimizer_state_fingerprint(torch.tensor([1.])))
        self.assertEqual(optimizer_state_fingerprint({'a': 1, 'b': 2}),
                         optimizer_state_fingerprint({'b': 2, 'a': 1}))

    def test_checkpoint_evidence_checks_full_optimizer_not_progress_only(self):
        schedule = dict(total_updates=100, selected_centers=6400,
                         optimizer_manifest=self.manifest)
        checkpoint = dict(model_version=6, architecture=ARCHITECTURE, step=2, complete=False,
                           epoch=0, next_update=2, physical_world_size=4,
                           config=self.cfg, config_sha256=config_hash(self.cfg),
                           global_batch_size=64, micro_batch_size=16, selected_centers=6400,
                           optimizer=self.optimizer, optimizer_manifest=self.manifest, lr_prior={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'resume.pt'
            torch.save(checkpoint, path)
            evidence = optimizer_evidence(path, self.cfg, 2, 4, schedule)
            self.assertEqual(evidence['optimizer']['buffers']['muon']['nonzero_primary_buffers'], 1)
            self.assertEqual(len(evidence['checkpoint_sha256']), 64)
            checkpoint['optimizer']['muon']['state'] = {}
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(AssertionError, 'populated muon'):
                optimizer_evidence(path, self.cfg, 2, 4, schedule)

    def test_two_to_four_gpu_resume_preserves_config_manifest_and_optimizer_cursor(self):
        schedule = dict(total_updates=100, selected_centers=6400,
                         optimizer_manifest=self.manifest)
        sources, architecture = {'model.py': 'fixed-source'}, {'route': 'A'}
        checkpoint = dict(model_version=6, architecture=ARCHITECTURE, arch_config=architecture,
                           step=2, complete=False, epoch=0, next_update=2,
                           physical_world_size=2, phase='frozen_backbone',
                           config=copy.deepcopy(self.cfg), config_sha256=config_hash(self.cfg),
                           source_sha256=sources, model={}, global_batch_size=64,
                           micro_batch_size=16, selected_centers=6400,
                           optimizer=self.state_at(2), optimizer_manifest=self.manifest, lr_prior={})
        four_gpu_sampler = GlobalBatchSampler(6400, 64, 16, world=4, rank=0)
        self.assertEqual(four_gpu_sampler.accumulation, 1)
        # Old world size is descriptive metadata, not a config/optimizer change.
        self.assertEqual(validate_resume(
            checkpoint, self.cfg, sources, architecture, micro_batch=16,
            updates_per_epoch=four_gpu_sampler.updates, selected_centers=6400,
            optimizer_manifest=self.manifest), (2, 0, 2))
        with tempfile.TemporaryDirectory() as directory:
            before_path = Path(directory) / 'two_gpu.pt'
            after_path = Path(directory) / 'four_gpu.pt'
            torch.save(checkpoint, before_path)
            before = optimizer_evidence(before_path, self.cfg, 2, 2, schedule)
            updated = dict(checkpoint, step=3, next_update=3, physical_world_size=4,
                           optimizer=self.state_at(3))
            torch.save(updated, after_path)
            after = optimizer_evidence(after_path, self.cfg, 3, 4, schedule)
            self.assertEqual(before['config_sha256'], after['config_sha256'])
            self.assertEqual(before['optimizer']['manifest'], after['optimizer']['manifest'])
            self.assertEqual(before['optimizer']['steps_completed'], 2)
            self.assertEqual(after['optimizer']['steps_completed'], 3)
            with self.assertRaisesRegex(AssertionError, 'boundary'):
                optimizer_evidence(before_path, self.cfg, 2, 4, schedule)


class DistributedLaunchTests(unittest.TestCase):
    def topology(self):
        return [dict(rank=rank, local_rank=rank % 2, hostname=f'node-{rank // 2}',
                     visible_devices='0,1', gpu_name='fixture GPU') for rank in range(4)]

    def test_one_node_and_cross_node_shapes_preserve_total_four(self):
        self.assertEqual(resolve_topology(1), (1, 4, 4))
        self.assertEqual(resolve_topology(1, training_gpus=4), (1, 4, 4))
        self.assertEqual(resolve_topology(2, training_gpus=4), (2, 2, 4))
        self.assertEqual(resolve_topology(2, 2, 4), (2, 2, 4))
        self.assertEqual(resolve_topology(4, 1), (4, 1, 4))

    def test_mismatched_assertion_or_fractional_node_split_fails(self):
        for args in ((2, 4, 4), (3, None, 4), (0, 4, 4), (1, 0, None), (True, 4, 4), (2, 4, 8)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                resolve_topology(*args)

    def test_launcher_owns_torchrun_for_every_topology(self):
        for nodes, gpus in ((1, 4), (2, 2)):
            command = training_command('/tmp/v6 smoke/config.yaml', nodes, gpus)
            self.assertEqual(command[0], 'bash')
            self.assertTrue(command[1].endswith('/scripts/launch_subcellular_v6_distributed.sh'))
            self.assertEqual(command[2:], ['/tmp/v6 smoke/config.yaml', str(nodes), str(gpus)])
            self.assertNotIn('torch.distributed.run', command)

    def test_rank_evidence_verifies_cross_node_execution_beyond_world_size(self):
        result = validate_rank_topology(self.topology(), 2, 2)
        self.assertEqual(result['hosts'], {'node-0': [0, 1], 'node-1': [0, 1]})
        single_node = [dict(row, hostname='one-node', local_rank=row['rank']) for row in self.topology()]
        with self.assertRaisesRegex(AssertionError, 'topology'):
            validate_rank_topology(single_node, 2, 2)
        validate_rank_topology(single_node, 1, 4)

    def test_duplicate_rank_or_local_gpu_assignment_is_rejected(self):
        for key in ('rank', 'local_rank'):
            rows = self.topology()
            rows[1][key] = 0
            with self.subTest(key=key), self.assertRaises(AssertionError):
                validate_rank_topology(rows, 2, 2)

    def test_old_two_gpu_topology_cannot_verify_a_new_four_gpu_launch(self):
        before = [dict(rank=rank, local_rank=rank, hostname='old-node',
                        visible_devices='0,1', gpu_name='fixture GPU') for rank in range(2)]
        after = [dict(rank=rank, local_rank=rank, hostname='new-node',
                       visible_devices='0,1,2,3', gpu_name='fixture GPU') for rank in range(4)]
        self.assertEqual(validate_rank_topology(before, 1, 2)['world_size'], 2)
        self.assertEqual(validate_rank_topology(after, 1, 4)['world_size'], 4)
        with self.assertRaisesRegex(AssertionError, 'one observed topology entry per rank'):
            validate_rank_topology(before, 1, 4)

    def test_two_to_four_gpu_switch_preserves_global_windows_and_logical_seeds(self):
        def global_windows(world, start_update):
            samplers = [GlobalBatchSampler(195, 64, 16, world=world, rank=rank, seed=42)
                        for rank in range(world)]
            for sampler in samplers:
                sampler.seek(0, start_update)
            per_rank = [list(sampler) for sampler in samplers]
            accumulation = samplers[0].accumulation
            windows = []
            for relative in range(samplers[0].updates - start_update):
                logical = []
                for micro in range(accumulation):
                    for rank in range(world):
                        logical.append((
                            per_rank[rank][relative * accumulation + micro],
                            logical_forward_seed(42, 0, start_update + relative, micro * world + rank),
                        ))
                windows.append(logical)
            return accumulation, windows

        two_accumulation, two_windows = global_windows(2, 0)
        four_accumulation, resumed_windows = global_windows(4, 2)
        self.assertEqual((two_accumulation, four_accumulation), (2, 1))
        self.assertEqual(two_windows[2:], resumed_windows)
        for logical in resumed_windows:
            self.assertEqual(len(logical), 4)
            self.assertEqual(sum(len(batch) for batch, _ in logical), 64)


if __name__ == '__main__':
    unittest.main()
