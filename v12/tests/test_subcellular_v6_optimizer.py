"""V6 official-Muon partition, strict restore, and resumed LR scheduling.

Execution is restricted to an authorized compute node. The tests do not
request resources, launch training processes, or substitute another optimizer
when the installed PyTorch lacks official Muon.
"""
import copy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from spatial_pretrain.training.subcellular_v6_optimizer import (
    SCHEMA, build_optimizer, optimizer_manifest, validate_optimizer_config,
)
from test_subcellular_v6_model import model_for
from train_subcellular_v6 import lr_multiplier, validate_resume, config_hash


def training_config(**overrides):
    result = dict(optimizer='muon', lr_muon=1e-4, lr_new=1e-4, weight_decay=.01,
                  muon_momentum=.95, muon_ns_steps=5, muon_adjust_lr_fn='match_rms_adamw',
                  warmup_ratio=.4, min_lr_ratio=.1)
    result.update(overrides)
    return result


def synthetic_update(model, optimizer, step, training, total=5):
    """Apply deterministic dense gradients, including every auxiliary tensor."""
    optimizer.zero_grad(set_to_none=True)
    for index, parameter in enumerate(p for p in model.parameters() if p.requires_grad):
        parameter.grad = parameter.detach().mul(.03).add((index + 1) * .001)
    optimizer.set_lr_multiplier(lr_multiplier(step, total, training))
    optimizer.step()


class V6OptimizerPartitionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(83)

    def test_every_trainable_tensor_has_one_group_and_final_readouts_use_adamw(self):
        for route in ('A', 'B'):
            with self.subTest(route=route):
                model = model_for(route)
                manifest = optimizer_manifest(model, training_config())
                muon, adamw = manifest['groups']
                self.assertEqual((muon['name'], adamw['name']), ('muon_hidden', 'adamw_aux'))
                self.assertTrue(muon['use_muon'])
                self.assertFalse(adamw['use_muon'])
                names = muon['param_names'] + adamw['param_names']
                expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                self.assertEqual(set(names), expected)
                self.assertEqual(len(names), len(set(names)))
                self.assertFalse(any(name.startswith('backbone.') for name in names))
                for name in ('slot_pool.gene_projection.weight', 'slot_pool.attention.in_proj_weight',
                             'slot_pool.attention.out_proj.weight', 'blocks.0.intra.attention.in_proj_weight',
                             'blocks.0.spatial.query.weight', 'blocks.0.spatial.out.weight',
                             'blocks.0.ffn.layers.0.weight', 'final_intra.attention.out_proj.weight'):
                    self.assertIn(name, muon['param_names'])
                for name in ('compartment_embedding.weight', 'slot_pool.queries', 'slot_pool.missing',
                             'blocks.0.spatial.lr_scale_logit', 'blocks.0.intra.norm.weight',
                             'blocks.0.spatial.query.bias', 'joint_decoder.output.1.weight',
                             'joint_decoder.output.3.weight'):
                    self.assertIn(name, adamw['param_names'])
                projections = ('residual_projection.weight',) if route == 'A' else (
                    'self_projection.weight', 'context_projection.weight')
                for name in projections:
                    self.assertIn(name, adamw['param_names'])
                    self.assertNotIn(name, muon['param_names'])
                self.assertEqual(sum(group['parameter_count'] for group in manifest['groups']),
                                 sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
                self.assertEqual(json.loads(json.dumps(manifest)), manifest)

    def test_missing_official_muon_fails_without_adamw_fallback(self):
        with mock.patch.object(torch.optim, 'Muon', None, create=True):
            with self.assertRaisesRegex(RuntimeError, 'Muon is unavailable.*no fallback'):
                build_optimizer(model_for('A'), training_config())

    def test_backbone_cannot_silently_enter_optimizer_partition(self):
        model = model_for('B')
        next(model.backbone.parameters()).requires_grad_(True)
        with self.assertRaisesRegex(ValueError, 'frozen-backbone'):
            optimizer_manifest(model, training_config())

    def test_configuration_requires_explicit_muon_and_fixed_adjustment_protocol(self):
        validate_optimizer_config(training_config())
        # Separate base LRs are supported and locked in the checkpoint manifest;
        # the accepted A/B experiment sets both to the same AdamW-tuned value.
        validate_optimizer_config(training_config(lr_muon=2e-4))
        for key, value in (('optimizer', 'adamw'), ('optimizer', None),
                           ('lr_muon', 0.), ('lr_new', float('nan')),
                           ('weight_decay', -1.), ('muon_adjust_lr_fn', None),
                           ('muon_adjust_lr_fn', 'original'), ('muon_momentum', .9),
                           ('muon_ns_steps', 3), ('muon_nesterov', False)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_optimizer_config(training_config(**{key: value}))


@unittest.skipUnless(callable(getattr(torch.optim, 'Muon', None)), 'Installed PyTorch lacks official Muon')
class V6OptimizerStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(89)
        self.model = model_for('B')
        self.training = training_config(lr_muon=2e-4)
        self.optimizer = build_optimizer(self.model, self.training)

    def populated_state(self):
        synthetic_update(self.model, self.optimizer, 1, self.training)
        synthetic_update(self.model, self.optimizer, 2, self.training)
        return copy.deepcopy(self.optimizer.state_dict())

    def test_state_records_both_official_optimizers_and_zero_step_is_valid(self):
        self.assertIsInstance(self.optimizer.muon_optimizer, torch.optim.Muon)
        self.assertIsInstance(self.optimizer.adamw_optimizer, torch.optim.AdamW)
        zero = self.optimizer.state_dict()
        self.assertEqual(zero['schema'], SCHEMA)
        self.assertEqual(zero['steps_completed'], 0)
        self.assertEqual(zero['muon']['state'], {})
        self.assertEqual(zero['aux_adamw']['state'], {})
        restored = build_optimizer(self.model, self.training)
        restored.load_state_dict(zero)
        self.assertTrue(restored.resume_state_verified)
        state = self.populated_state()
        self.assertEqual(state['steps_completed'], 2)
        for value in state['muon']['state'].values():
            self.assertEqual(set(value), {'momentum_buffer'})
            self.assertTrue(torch.isfinite(value['momentum_buffer']).all())
        for value in state['aux_adamw']['state'].values():
            self.assertEqual(set(value), {'step', 'exp_avg', 'exp_avg_sq'})
            self.assertEqual(value['step'].item(), 2)

    def test_missing_components_and_deleted_parameter_moments_are_rejected(self):
        original = self.populated_state()
        mutations = []
        for component in ('muon', 'aux_adamw'):
            missing = copy.deepcopy(original)
            del missing[component]
            mutations.append(missing)
            absent = copy.deepcopy(original)
            absent[component] = None
            mutations.append(absent)
            cleared = copy.deepcopy(original)
            cleared[component]['state'].clear()
            mutations.append(cleared)
            partial = copy.deepcopy(original)
            del partial[component]['state'][next(iter(partial[component]['state']))]
            mutations.append(partial)
            lost_moment = copy.deepcopy(original)
            first = next(iter(lost_moment[component]['state'].values()))
            del first['momentum_buffer' if component == 'muon' else 'exp_avg_sq']
            mutations.append(lost_moment)
        for index, state in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                build_optimizer(self.model, self.training).load_state_dict(state)

    def test_parameter_order_names_and_hyperparameters_are_bound_to_manifest(self):
        original = self.populated_state()
        changes = []
        changed = copy.deepcopy(original)
        changed['manifest']['groups'][0]['param_names'][0] = 'wrong.matrix.weight'
        changes.append(changed)
        changed = copy.deepcopy(original)
        changed['muon']['param_groups'][0]['param_names'].reverse()
        changes.append(changed)
        for key, value in (('base_lr', .02), ('momentum', .9), ('adjust_lr_fn', 'original'),
                           ('name', 'unknown_group'), ('use_muon', False)):
            changed = copy.deepcopy(original)
            changed['muon']['param_groups'][0][key] = value
            changes.append(changed)
        for index, state in enumerate(changes):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                build_optimizer(self.model, self.training).load_state_dict(state)
        changed_training = dict(self.training, lr_muon=3e-4)
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            build_optimizer(self.model, changed_training).load_state_dict(original)

    def test_restored_schedule_updates_actual_groups_and_preserves_base_lr_ratio(self):
        state = self.populated_state()
        restored = build_optimizer(self.model, self.training)
        old_groups = list(restored.param_groups)
        restored.load_state_dict(state)
        self.assertTrue(restored.resume_state_verified)
        self.assertEqual(restored.restored_parameter_states,
                         {'muon': len(state['muon']['state']), 'adamw': len(state['aux_adamw']['state'])})
        self.assertIs(restored.param_groups[0], restored.muon_optimizer.param_groups[0])
        self.assertIs(restored.param_groups[1], restored.adamw_optimizer.param_groups[0])
        self.assertIsNot(restored.param_groups[0], old_groups[0])
        for step in (3, 4, 5):
            multiplier = lr_multiplier(step, 5, self.training)
            restored.set_lr_multiplier(multiplier)
            rates = restored.learning_rates()
            self.assertEqual(rates['muon'], self.training['lr_muon'] * multiplier)
            self.assertEqual(rates['adamw'], self.training['lr_new'] * multiplier)
            self.assertAlmostEqual(rates['muon'] / rates['adamw'], 2.)
            self.assertEqual(restored.muon_optimizer.param_groups[0]['lr'], rates['muon'])
            self.assertEqual(restored.adamw_optimizer.param_groups[0]['lr'], rates['adamw'])

    def test_serialized_two_step_resume_matches_uninterrupted_third_update(self):
        state = self.populated_state()
        archive = io.BytesIO()
        torch.save(dict(model=self.model.state_dict(), optimizer=state), archive)
        archive.seek(0)
        checkpoint = torch.load(archive, map_location='cpu', weights_only=False)
        restored_model = model_for('B')
        restored_model.load_state_dict(checkpoint['model'], strict=True)
        restored_optimizer = build_optimizer(restored_model, self.training)
        restored_optimizer.load_state_dict(checkpoint['optimizer'])
        synthetic_update(self.model, self.optimizer, 3, self.training)
        synthetic_update(restored_model, restored_optimizer, 3, self.training)
        self.assertEqual(restored_optimizer.steps_completed, 3)
        for name, parameter in self.model.state_dict().items():
            torch.testing.assert_close(restored_model.state_dict()[name], parameter, atol=0, rtol=0)
        actual, expected = restored_optimizer.state_dict(), self.optimizer.state_dict()
        for component in ('muon', 'aux_adamw'):
            self.assertEqual(actual[component]['param_groups'], expected[component]['param_groups'])
            for identifier, values in expected[component]['state'].items():
                for name, tensor in values.items():
                    torch.testing.assert_close(actual[component]['state'][identifier][name], tensor, atol=0, rtol=0)

    def test_training_cursor_cannot_claim_a_different_optimizer_state(self):
        optimizer_state = self.populated_state()
        config = dict(training=dict(epochs=1, global_batch_size=64))
        architecture, sources = {'route': 'B'}, {'source': 'unchanged'}
        state = dict(model_version=6, architecture='subcellular_lr_routes_v6', arch_config=architecture,
                     config=config, config_sha256=config_hash(config), source_sha256=sources,
                     micro_batch_size=16, global_batch_size=64, phase='frozen_backbone',
                     step=2, epoch=0, next_update=2, complete=False, model={},
                     optimizer=optimizer_state, optimizer_manifest=self.optimizer.manifest)

        def validate(candidate):
            return validate_resume(candidate, config, sources, architecture, micro_batch=16,
                                   updates_per_epoch=10, optimizer_manifest=self.optimizer.manifest)

        self.assertEqual(validate(state), (2, 0, 2))
        changed = copy.deepcopy(state)
        changed['optimizer']['steps_completed'] = 1
        with self.assertRaisesRegex(ValueError, 'update count'):
            validate(changed)
        changed = copy.deepcopy(state)
        del changed['optimizer_manifest']
        with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
            validate(changed)


if __name__ == '__main__':
    unittest.main()
