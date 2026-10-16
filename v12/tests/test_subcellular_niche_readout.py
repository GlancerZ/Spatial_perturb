"""Controlled direct-niche readout initialization and optimizer migration tests."""
import copy
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from test_subcellular_v2_model import MockBackbone, example
from spatial_pretrain.models.subcellular_v2 import SubcellularV2
from prepare_subcellular_readout_ab import (PROJECTION_NAME, parameter_groups_by_name,
                                           remap_optimizer_state, prepare_arm)


def model(direct=None):
    kwargs = {} if direct is None else {'direct_niche_readout': direct}
    return SubcellularV2('.', freeze_layers=1, neighbors=2, encoder_chunk_size=2,
                        checkpointing=False, backbone=MockBackbone(), **kwargs).eval()


class NicheReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(98)
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def test_false_preserves_legacy_parameter_set_and_strict_loading(self):
        legacy_default = model()
        explicit_false = model(False)
        self.assertEqual(set(legacy_default.state_dict()), set(explicit_false.state_dict()))
        self.assertEqual(set(dict(legacy_default.named_parameters())), set(dict(explicit_false.named_parameters())))
        self.assertFalse(any(name.startswith('niche_to_joint.') for name in explicit_false.state_dict()))
        explicit_false.load_state_dict(legacy_default.state_dict(), strict=True)
        before = legacy_default(example())
        after = explicit_false(example())
        for key in ('gene_pred', 'intrinsic_pred', 'joint_pred', 'neighbor_pred'):
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)

    def test_zero_projection_preserves_all_initial_outputs_and_learns(self):
        control, variant = model(False), model(True)
        missing = variant.load_state_dict(control.state_dict(), strict=False)
        self.assertEqual(missing.missing_keys, [PROJECTION_NAME])
        self.assertEqual(missing.unexpected_keys, [])
        self.assertEqual(set(variant.state_dict()) - set(control.state_dict()), {PROJECTION_NAME})
        self.assertEqual(int(torch.count_nonzero(variant.niche_to_joint.weight)), 0)
        before, after = control(example()), variant(example())
        for key in before['embeddings']:
            torch.testing.assert_close(before['embeddings'][key], after['embeddings'][key], atol=0, rtol=0)
        for key in ('gene_pred', 'intrinsic_pred', 'joint_pred', 'neighbor_pred'):
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
        after['joint_pred'].square().mean().backward()
        gradient = variant.niche_to_joint.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_pure_niche_still_excludes_center_expression_with_direct_readout(self):
        variant = model(True)
        with torch.no_grad():
            variant.niche_to_joint.weight.copy_(torch.eye(variant.dim))
        batch = example()
        before = variant(batch)
        batch['input_bins'][:, 0, 1:5] += 20
        after = variant(batch)
        torch.testing.assert_close(before['embeddings']['niche'][:, 0], after['embeddings']['niche'][:, 0], atol=0, rtol=0)
        torch.testing.assert_close(before['neighbor_pred'], after['neighbor_pred'], atol=0, rtol=0)

    def test_optimizer_moments_are_preserved_by_name_and_projection_has_no_state(self):
        parent, variant = model(False), model(True)
        parent.set_backbone_trainable(full=True)
        names = parameter_groups_by_name(parent)
        parameters = dict(parent.named_parameters())
        optimizer = torch.optim.AdamW([
            {'params': [parameters[name] for name in names[0]], 'lr': 1e-5, 'base_lr': 1e-5},
            {'params': [parameters[name] for name in names[1]], 'lr': 1e-4, 'base_lr': 1e-4}])
        sum(parameter.square().sum() for parameter in parent.parameters()).backward()
        optimizer.step()
        original = optimizer.state_dict()
        remapped, added = remap_optimizer_state(original, parent, variant)
        self.assertEqual(added, [PROJECTION_NAME])
        expected = {name: original['state'].get(identifier) for group, group_names in zip(original['param_groups'], names)
                    for identifier, name in zip(group['params'], group_names)}
        for group, group_names in zip(remapped['param_groups'], parameter_groups_by_name(variant)):
            for identifier, name in zip(group['params'], group_names):
                if name == PROJECTION_NAME:
                    self.assertNotIn(identifier, remapped['state'])
                    continue
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(remapped['state'][identifier][key], expected[name][key], atol=0, rtol=0)
        for before, after in zip(original['param_groups'], remapped['param_groups']):
            self.assertEqual({k: v for k, v in before.items() if k != 'params'},
                             {k: v for k, v in after.items() if k != 'params'})
        # Verify the emitted optimizer state is accepted by the real AdamW API.
        target_names = parameter_groups_by_name(variant)
        target_parameters = dict(variant.named_parameters())
        restored = torch.optim.AdamW([{'params': [target_parameters[name] for name in group]} for group in target_names])
        restored.load_state_dict(remapped)
        self.assertNotIn(variant.niche_to_joint.weight, restored.state)

        cfg = {'model': {'direct_niche_readout': True}, 'experiment': 'arm', 'output_dir': '/tmp/arm'}
        parent_payload = dict(model=parent.state_dict(), optimizer=original, config={'model': {}},
            config_sha256='parent', step=3652, epoch=0, micro_step=3652, world_size=4,
            rng_by_rank=[{'torch': torch.tensor([rank], dtype=torch.uint8)} for rank in range(4)],
            backbone_provenance={'weights_sha256': 'fixed-official-hash'}, complete=False)
        payload, _ = prepare_arm(parent_payload, parent, variant, cfg, {'parent_step': 3652})
        self.assertEqual(payload['step'], 3652)
        self.assertEqual(payload['micro_step'], 3652)
        self.assertEqual(payload['world_size'], 4)
        self.assertEqual(payload['backbone_provenance'], parent_payload['backbone_provenance'])
        for rank in range(4):
            torch.testing.assert_close(payload['rng_by_rank'][rank]['torch'], parent_payload['rng_by_rank'][rank]['torch'])
        for name, value in parent_payload['model'].items():
            torch.testing.assert_close(payload['model'][name], value, atol=0, rtol=0)
        self.assertEqual(int(torch.count_nonzero(payload['model'][PROJECTION_NAME])), 0)


if __name__ == '__main__':
    unittest.main()
