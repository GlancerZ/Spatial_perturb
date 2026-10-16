"""V6 A/B readout, directed LR, gradient and information-boundary tests.

Run on an authorized compute node, as required by repository policy.
"""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spatial_pretrain.models.subcellular_v2 import patch_knn
from spatial_pretrain.models.subcellular_v6 import LRSpatialAttention, SubcellularV6


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_kind = 'scgpt'
        self.dim, self.nheads, self.pad_id = 16, 4, 0
        self.config = {'pad_value': -2.}
        self.vocab = {'<pad>': 0, '<cls>': 1, **{f'G{i}': i + 2 for i in range(30)}}
        self.encoder = nn.Embedding(32, 16, padding_idx=0)
        self.value_encoder = nn.Linear(1, 16)
        layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0., batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, genes, bins, padding, compartment):
        hidden = self.encoder(genes) + self.value_encoder(bins.unsqueeze(-1)) + compartment
        return self.transformer_encoder(hidden, src_key_padding_mask=padding)


def example():
    return dict(
        gene_ids=torch.tensor([[[1, 2, 3, 4, 5, 0],
                                [1, 3, 4, 5, 6, 0],
                                [1, 4, 5, 6, 7, 0]]]),
        input_bins=torch.tensor([[[-2., 3., 8., 7., 2., -2.],
                                  [-2., 4., 6., 8., 1., -2.],
                                  [-2., 7., 2., 1., 9., -2.]]]),
        compartment_ids=torch.tensor([[[2, 0, 0, 1, 1, 2]] * 3]),
        token_valid_mask=torch.tensor([[[True, True, True, True, True, False]] * 3]),
        coords=torch.tensor([[[0., 0.], [1., 3.], [5., 2.]]]),
        node_indices=torch.tensor([[10, 20, 30]]),
        whole_gene_ids=torch.tensor([[1, 2, 3, 4, 5, 8, 0]]),
        whole_input_bins=torch.tensor([[-2., 3., 8., 7., 2., 4., -2.]]),
        whole_token_valid_mask=torch.tensor([[True, True, True, True, True, True, False]]),
        query_gene_ids=torch.tensor([[2, 3, 4, 5, 0]]),
        query_compartment_ids=torch.tensor([[0, 0, 1, 1, 1]]),
        query_valid_mask=torch.tensor([[True, True, True, True, False]]),
        target_bins=torch.tensor([[0, 1, 50, 18, -2]]),
        lr_bias=torch.tensor([[[0., .9, .15], [.1, 0., .8], [.6, .2, 0.]]]),
        lr_coverage=torch.tensor([[[0., .9, .4], [.3, 0., .8], [.8, .5, 0.]]]),
    )


def model_for(route='A', **kwargs):
    return SubcellularV6(
        '.', neighbors=2, checkpointing=False, encoder_chunk_size=3,
        backbone=MockBackbone(), slots_per_compartment=2, slot_dim=8,
        route=route, **kwargs).eval()


def open_projections(model):
    with torch.no_grad():
        if model.route == 'A':
            nn.init.normal_(model.residual_projection.weight, std=.15)
        else:
            nn.init.normal_(model.self_projection.weight, std=.15)
            nn.init.normal_(model.context_projection.weight, std=.15)


class V6ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)
        torch.backends.mha.set_fastpath_enabled(False)
        self.batch = example()

    def close(self, left, right):
        torch.testing.assert_close(left, right, atol=6e-6, rtol=6e-6)

    def changed(self, left, right):
        self.assertGreater(float((left - right).abs().max()), 1e-6)

    def test_both_routes_are_exact_plain_scgpt_at_step_zero(self):
        for route in ('A', 'B'):
            with self.subTest(route=route):
                model = model_for(route)
                genes = self.batch['whole_gene_ids']
                valid = self.batch['whole_token_valid_mask'] & genes.ne(model.pad_id)
                with torch.no_grad():
                    hidden = model.backbone(
                        genes, self.batch['whole_input_bins'], ~valid,
                        torch.zeros((*genes.shape, model.backbone.dim)))
                reference = F.normalize(hidden[:, 0].float(), dim=-1, eps=1e-8)
                result = model(self.batch, return_embeddings=True)
                for name in ('base', 'intrinsic', 'joint'):
                    torch.testing.assert_close(result['embeddings'][name], reference, atol=0, rtol=0)
                self.assertEqual(result['branch_joint'].shape, (1, 32))
                self.assertEqual(result['branch_intrinsic'].shape, (1, 32))
                self.assertEqual(result['residual'].count_nonzero().item(), 0)
                self.assertEqual(model.arch_config['route'], route)
                self.assertEqual(model.arch_config['num_blocks'], 2)
                self.assertEqual(model.arch_config['lr_bias_reliability_shrinkage'], 4.)
                self.assertEqual(model.arch_config['lr_coverage_role'], 'diagnostic_only_never_multiply')
                for name in ('nucleus', 'cytoplasm', 'nucleus_local', 'cytoplasm_local'):
                    self.assertEqual(result['embeddings'][name].shape, (1, 16))

    def test_b_has_only_one_additional_effective_projection(self):
        models = [model_for(route) for route in ('A', 'B')]
        counts = [sum(p.numel() for p in model.parameters() if p.requires_grad) for model in models]
        self.assertEqual(counts[1] - counts[0], 16 * 32)
        self.assertEqual(models[0].arch_config['projection_parameters'], 16 * 32)
        self.assertEqual(models[1].arch_config['projection_parameters'], 2 * 16 * 32)
        self.assertFalse(hasattr(models[1], 'residual_projection'))
        self.assertFalse(hasattr(models[1], 'self_blocks'))

    def test_execution_settings_do_not_change_checkpoint_architecture(self):
        model = model_for('B')
        checkpointed = SubcellularV6(
            '.', neighbors=2, checkpointing=True, encoder_chunk_size=8,
            backbone=MockBackbone(), slots_per_compartment=2, slot_dim=8, route='B').eval()
        self.assertEqual(model.arch_config, checkpointed.arch_config)
        checkpointed.load_state_dict(model.state_dict(), strict=True)
        before = model(self.batch)
        after = checkpointed(self.batch)
        self.close(before['joint_logits'], after['joint_logits'])

    def test_first_step_opens_both_b_projections_and_then_attention_gradients(self):
        for route in ('A', 'B'):
            with self.subTest(route=route):
                model = model_for(route).train()
                original_backbone = {name: value.clone() for name, value in model.backbone.state_dict().items()}
                self.assertFalse(model.backbone.training)
                optimizer = torch.optim.SGD(model.parameters(), lr=.2)

                def backward():
                    optimizer.zero_grad(set_to_none=True)
                    result = model(self.batch)
                    selected = self.batch['query_valid_mask']
                    loss = F.cross_entropy(result['joint_logits'][selected], self.batch['target_bins'][selected])
                    loss.backward()

                backward()
                projections = ([model.residual_projection] if route == 'A' else
                               [model.self_projection, model.context_projection])
                for projection in projections:
                    self.assertIsNotNone(projection.weight.grad)
                    self.assertGreater(projection.weight.grad.abs().sum().item(), 0.)
                self.assertEqual(model.slot_pool.queries.grad.abs().sum().item(), 0.)
                optimizer.step()
                backward()
                for parameter in (
                        model.slot_pool.queries, model.blocks[0].intra.attention.out_proj.weight,
                        model.blocks[0].spatial.out.weight, model.blocks[0].spatial.lr_scale_logit,
                        model.final_intra.attention.out_proj.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.)
                self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
                for name, value in model.backbone.state_dict().items():
                    torch.testing.assert_close(value, original_backbone[name], atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, 'frozen'):
                    model.set_backbone_trainable(True)
                with self.assertRaisesRegex(ValueError, 'zero'):
                    model.set_backbone_trainable_layers(1)

    def test_neighbor_changes_context_and_nucleus_but_never_self(self):
        for route in ('A', 'B'):
            with self.subTest(route=route):
                model = model_for(route)
                open_projections(model)
                changed = copy.deepcopy(self.batch)
                changed['gene_ids'][0, 1, 3:5] = torch.tensor([19, 25])
                before = model(self.batch, return_embeddings=True)['embeddings']
                after = model(changed, return_embeddings=True)['embeddings']
                for name in ('base', 'intrinsic', 'nucleus_local', 'cytoplasm_local'):
                    self.close(before[name], after[name])
                for name in ('nucleus', 'cytoplasm', 'joint'):
                    self.changed(before[name], after[name])
                if route == 'B':
                    self.close(before['self'], after['self'])
                    self.changed(before['context_delta'], after['context_delta'])

    def test_final_intra_is_necessary_for_last_spatial_message_to_reach_nucleus(self):
        model = model_for('B', num_blocks=1)
        open_projections(model)
        changed = copy.deepcopy(self.batch)
        changed['gene_ids'][0, 1, 3:5] = torch.tensor([18, 27])
        before = model(self.batch, return_embeddings=True)['embeddings']
        after = model(changed, return_embeddings=True)['embeddings']
        self.changed(before['nucleus'], after['nucleus'])
        model.final_intra = nn.Identity()
        before = model(self.batch, return_embeddings=True)['embeddings']
        after = model(changed, return_embeddings=True)['embeddings']
        self.close(before['nucleus'], after['nucleus'])
        self.changed(before['cytoplasm'], after['cytoplasm'])

    def test_b_context_is_exact_zero_for_disabled_or_empty_center_neighborhood(self):
        model = model_for('B')
        open_projections(model)
        cases = []
        no_sender = copy.deepcopy(self.batch)
        no_sender['token_valid_mask'][0, 1:, 3:5] = False
        cases.append((no_sender, True))
        duplicate = copy.deepcopy(self.batch)
        duplicate['node_indices'].fill_(10)
        cases.append((duplicate, True))
        padded = copy.deepcopy(self.batch)
        padded['node_indices'][0, 1:] = -1
        padded['coords'][0, 1:] = float('nan')
        cases.append((padded, True))
        cases.append((self.batch, False))
        for batch, spatial in cases:
            with self.subTest(spatial=spatial, ids=batch['node_indices'].tolist()):
                result = model(batch, return_embeddings=True, spatial=spatial)
                self.assertEqual(result['context_delta'].count_nonzero().item(), 0)
                torch.testing.assert_close(result['embeddings']['joint'], result['embeddings']['self'], atol=0, rtol=0)
                self.assertGreater(result['self_delta'].abs().sum().item(), 0.)
        model.max_neighbor_distance = 0.
        result = model(self.batch, return_embeddings=True)
        self.assertFalse(result['spatial_neighbor_mask'].any())
        self.assertEqual(result['context_delta'].count_nonzero().item(), 0)

    def test_spatial_control_and_lr_control_ignore_the_disabled_inputs(self):
        for route in ('A', 'B'):
            model = model_for(route, spatial_enabled=False)
            open_projections(model)
            changed = copy.deepcopy(self.batch)
            changed['coords'].fill_(float('nan'))
            changed.pop('lr_bias')
            changed.pop('lr_coverage')
            before = model(self.batch, return_embeddings=True)
            after = model(changed, return_embeddings=True)
            torch.testing.assert_close(before['embeddings']['joint'], after['embeddings']['joint'], atol=0, rtol=0)
            self.close(before['embeddings']['joint'], before['embeddings']['intrinsic'])
        model = model_for('A', lr_enabled=False)
        open_projections(model)
        before = model(self.batch, return_embeddings=True)
        changed = copy.deepcopy(self.batch)
        changed['lr_bias'].fill_(float('nan'))
        after = model(changed, return_embeddings=True)
        torch.testing.assert_close(before['embeddings']['joint'], after['embeddings']['joint'], atol=0, rtol=0)

    def test_coverage_is_diagnostic_and_is_not_multiplied_twice(self):
        model = model_for('A')
        open_projections(model)
        before = model(self.batch, return_embeddings=True)
        changed = copy.deepcopy(self.batch)
        changed['lr_coverage'].zero_()
        after = model(changed, return_embeddings=True)
        torch.testing.assert_close(before['embeddings']['joint'], after['embeddings']['joint'], atol=0, rtol=0)
        self.assertGreater(before['lr_diagnostics']['coverage_mean'].item(), 0.)
        self.assertEqual(after['lr_diagnostics']['coverage_mean'].item(), 0.)

    def test_token_neighbor_permutations_padding_and_targets_do_not_leak(self):
        for route in ('A', 'B'):
            with self.subTest(route=route):
                model = model_for(route)
                open_projections(model)
                before = model(self.batch, return_embeddings=True)
                changed = copy.deepcopy(self.batch)
                token_order = torch.tensor([0, 4, 2, 5, 1, 3])
                for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask'):
                    changed[name] = changed[name][..., token_order]
                for name, fill in (('gene_ids', 29), ('input_bins', 99999.),
                                   ('compartment_ids', 0), ('token_valid_mask', False)):
                    extra = changed[name].new_full((1, 3, 3), fill)
                    changed[name] = torch.cat((changed[name], extra), -1)
                order = torch.tensor([0, 2, 1])
                for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask',
                             'coords', 'node_indices'):
                    changed[name] = changed[name][:, order]
                for name in ('lr_bias', 'lr_coverage'):
                    changed[name] = changed[name][:, order][:, :, order]
                changed['target_bins'].fill_(99999)
                changed['whole_input_bins'][0, -1] = 99999.
                whole_order = torch.tensor([0, 4, 2, 5, 1, 3, 6])
                for name in ('whole_gene_ids', 'whole_input_bins', 'whole_token_valid_mask'):
                    changed[name] = changed[name][:, whole_order]
                after = model(changed, return_embeddings=True)
                for name in before['embeddings']:
                    self.close(before['embeddings'][name], after['embeddings'][name])

    def test_duplicate_and_pad_senders_do_not_create_additional_edges(self):
        model = model_for('B')
        open_projections(model)
        before = model(self.batch, return_embeddings=True)
        changed = copy.deepcopy(self.batch)
        for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask', 'coords', 'node_indices'):
            changed[name] = torch.cat((changed[name], changed[name][:, 1:2], changed[name][:, 2:3]), dim=1)
        changed['node_indices'][0, -1] = -1
        changed['coords'][0, -1] = float('nan')
        for name in ('lr_bias', 'lr_coverage'):
            old = changed[name]
            changed[name] = torch.ones((1, 5, 5))
            changed[name][:, :3, :3] = old
        after = model(changed, return_embeddings=True)
        self.close(before['embeddings']['joint'], after['embeddings']['joint'])
        picked = after['spatial_neighbor_indices'][after['spatial_neighbor_mask']]
        self.assertFalse(picked.ge(3).any())

    def test_variable_gene_counts_exceed_slot_count_without_changing_readout_width(self):
        model = model_for('B')
        open_projections(model)
        before = model(self.batch, return_embeddings=True)
        expanded = copy.deepcopy(self.batch)
        for name, fill in (('gene_ids', 0), ('input_bins', -2.),
                           ('compartment_ids', 2), ('token_valid_mask', False)):
            expanded[name] = torch.cat((expanded[name], expanded[name].new_full((1, 3, 4), fill)), -1)
        expanded['gene_ids'][0, 0, -4:] = torch.tensor([10, 11, 12, 13])
        expanded['input_bins'][0, 0, -4:] = torch.tensor([3., 11., 18., 4.])
        expanded['compartment_ids'][0, 0, -4:] = torch.tensor([0, 0, 0, 1])
        expanded['token_valid_mask'][0, 0, -4:] = True
        after = model(expanded, return_embeddings=True)
        self.assertEqual(after['branch_joint'].shape, before['branch_joint'].shape)
        self.assertEqual(after['embeddings']['nucleus'].shape, (1, model.dim))
        self.changed(before['embeddings']['nucleus_local'], after['embeddings']['nucleus_local'])
        self.changed(before['embeddings']['self'], after['embeddings']['self'])
        torch.testing.assert_close(before['embeddings']['base'], after['embeddings']['base'], atol=0, rtol=0)

    def test_decoder_reads_exact_joint_and_no_target_payload(self):
        for route in ('A', 'B'):
            model = model_for(route)
            open_projections(model)
            captured = []
            handle = model.joint_decoder.register_forward_pre_hook(lambda module, args: captured.append(args))
            result = model(self.batch)
            handle.remove()
            query, context = captured[0]
            self.assertEqual(result['joint_logits'].shape, (1, 5, 51))
            self.close(context, result['embeddings']['joint'][:, None, :].expand_as(context))
            selected = self.batch['query_valid_mask']
            self.close(query.norm(dim=-1)[selected], torch.ones(int(selected.sum())))
            changed = copy.deepcopy(self.batch)
            changed['target_bins'].fill_(99999)
            actual = model(changed)
            torch.testing.assert_close(result['joint_logits'], actual['joint_logits'], atol=0, rtol=0)

    def test_lr_contract_rejects_bad_shape_range_and_missing_inputs(self):
        model = model_for()
        for name in ('lr_bias', 'lr_coverage'):
            for value in (torch.zeros(1, 3, 2), torch.zeros(1, 3, 3, dtype=torch.long),
                          torch.full((1, 3, 3), float('nan')), torch.full((1, 3, 3), -1.),
                          torch.full((1, 3, 3), 1.01)):
                bad = copy.deepcopy(self.batch)
                bad[name] = value
                with self.assertRaisesRegex(ValueError, name):
                    model(bad)
            bad = copy.deepcopy(self.batch)
            del bad[name]
            with self.assertRaises(KeyError):
                model(bad)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            model(self.batch, spatial=1)


class V6SpatialAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(23)
        self.attention = LRSpatialAttention(4, 1, lr_initial_strength=.5).eval()
        # Make LR's direction/weight effect analytically visible: QK and
        # distance are zero, while values and the output projection are identity.
        with torch.no_grad():
            for layer in (self.attention.query, self.attention.key):
                layer.weight.zero_()
                layer.bias.zero_()
            for layer in (self.attention.value, self.attention.out):
                layer.weight.copy_(torch.eye(4))
                layer.bias.zero_()
        self.states = torch.tensor([[[1., 0., 0., -1.],
                                     [2., 0., 1., -1.],
                                     [-1., 1., 0., 2.]]])
        self.coords = torch.tensor([[[0., 0.], [1., 0.], [-2., 0.]]])
        self.ids = torch.tensor([[10, 20, 30]])
        self.graph = patch_knn(self.coords, self.ids, torch.ones_like(self.ids, dtype=torch.bool), 2)

    def test_receiver_sender_direction_has_the_expected_weighted_value(self):
        bias = torch.zeros((1, 3, 3))
        bias[0, 0, 1] = 1.
        before = self.attention(self.states, self.states, self.graph, torch.zeros_like(bias))
        after = self.attention(self.states, self.states, self.graph, bias)
        weights = torch.softmax(torch.tensor([.5, 0.]), dim=0)
        normalized = self.attention.norm(self.states)[0]
        expected = weights[0] * normalized[1] + weights[1] * normalized[2]
        torch.testing.assert_close(after[0, 0], expected)
        torch.testing.assert_close(after[:, 1:], before[:, 1:], atol=0, rtol=0)
        reversed_direction = self.attention(self.states, self.states, self.graph, bias.transpose(-1, -2))
        torch.testing.assert_close(reversed_direction[:, 0], before[:, 0], atol=0, rtol=0)
        self.assertGreater((after[:, 0] - before[:, 0]).abs().max().item(), .01)

    def test_masked_non_neighbor_bias_has_no_effect_and_empty_message_is_zero(self):
        indices, valid, distance = self.graph
        valid = valid.clone()
        valid[0, 0] &= indices[0, 0].eq(1)
        graph = indices, valid, distance
        bias = torch.zeros((1, 3, 3))
        before = self.attention(self.states, self.states, graph, bias)
        bias[0, 0, 2] = 1.
        after = self.attention(self.states, self.states, graph, bias)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        with torch.no_grad():
            self.attention.out.bias.fill_(20.)
        empty = indices, torch.zeros_like(valid), distance
        result = self.attention(self.states, self.states, empty, bias)
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.count_nonzero().item(), 0)

    def test_strength_is_bounded_learnable_and_disabled_lr_is_neutral(self):
        torch.testing.assert_close(self.attention.lr_strength, torch.tensor([.5]))
        bias = torch.zeros((1, 3, 3))
        bias[0, 0, 1] = 1.
        output = self.attention(self.states, self.states, self.graph, bias)
        output[0, 0, 0].backward()
        self.assertGreater(self.attention.lr_scale_logit.grad.abs().sum().item(), 0.)
        for value in (-1000., 1000.):
            with torch.no_grad():
                self.attention.lr_scale_logit.fill_(value)
            self.assertTrue((self.attention.lr_strength >= 0.).all())
            self.assertTrue((self.attention.lr_strength <= 2.).all())
        self.attention.lr_enabled = False
        with_prior = self.attention(self.states, self.states, self.graph, bias)
        without_prior = self.attention(self.states, self.states, self.graph, None)
        torch.testing.assert_close(with_prior, without_prior, atol=0, rtol=0)

    def test_small_lr_bias_survives_bfloat16_distance_bias_addition(self):
        with torch.no_grad():
            self.attention.distance_bias[-1].bias.fill_(1.)
        bias = torch.zeros((1, 3, 3))
        bias[0, 0, 1] = 1e-4
        original = F.scaled_dot_product_attention
        with patch('torch.nn.functional.scaled_dot_product_attention', wraps=original) as attention:
            with torch.autocast('cpu', dtype=torch.bfloat16):
                result = self.attention(self.states, self.states, self.graph, bias)
        actual_bias = attention.call_args.kwargs['attn_mask']
        self.assertEqual(actual_bias.dtype, torch.float32)
        for tensor in attention.call_args.args:
            self.assertEqual(tensor.dtype, torch.float32)
        self.assertAlmostEqual(actual_bias[0, 0, 0, 0].item() - 1., 5e-5, delta=1e-7)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == '__main__':
    unittest.main()
