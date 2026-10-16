"""V5 frozen-anchor, residual-gradient and information-boundary checks.

These tests are intentionally CPU-small, but were not executed while authoring:
repository policy requires all project Python/test execution on a compute node.
"""
import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spatial_pretrain.models.subcellular_v5 import SubcellularV5


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
    )


def activate_branch(model):
    """Make every deliberately zero-initialized residual path observable."""
    with torch.no_grad():
        nn.init.normal_(model.residual_projection.weight, std=.1)
        for block in (*model.local_pre_blocks, *model.local_post_blocks):
            nn.init.normal_(block.attention.out_proj.weight, std=.15)
            nn.init.normal_(block.ffn[-1].weight, std=.04)
        for block in model.spatial_blocks:
            nn.init.normal_(block.attention.out.weight, std=.15)
            nn.init.normal_(block.attention.distance_bias[-1].weight, std=.15)


class V5ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(59)
        torch.backends.mha.set_fastpath_enabled(False)
        self.model = SubcellularV5(
            '.', neighbors=2, encoder_chunk_size=2, backbone=MockBackbone(),
            slots_per_compartment=2, slot_dim=8).eval()
        self.batch = example()

    def close(self, left, right):
        torch.testing.assert_close(left, right, atol=4e-6, rtol=4e-6)

    def changed(self, left, right):
        self.assertGreater(float((left - right).abs().max()), 1e-6)

    def reference_base(self, batch):
        genes = batch['whole_gene_ids']
        valid = batch['whole_token_valid_mask'] & genes.ne(self.model.pad_id)
        addition = torch.zeros((*genes.shape, self.model.backbone.dim))
        with torch.no_grad():
            hidden = self.model.backbone(
                genes, batch['whole_input_bins'], ~valid, addition)
        return F.normalize(hidden[:, 0].float(), dim=-1, eps=1e-8)

    def test_step_zero_is_exact_whole_cls_and_all_readouts_are_center_only(self):
        expected = self.reference_base(self.batch)
        output = self.model(self.batch, return_embeddings=True)
        embeddings = output['embeddings']
        self.assertEqual(set(embeddings), {
            'base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
            'nucleus_local', 'cytoplasm_local'})
        torch.testing.assert_close(embeddings['base'], expected, atol=0, rtol=0)
        torch.testing.assert_close(embeddings['joint'], embeddings['base'], atol=0, rtol=0)
        torch.testing.assert_close(embeddings['intrinsic'], embeddings['base'], atol=0, rtol=0)
        for name in ('base', 'intrinsic', 'joint'):
            self.assertEqual(embeddings[name].shape, (1, self.model.backbone.dim))
        for name in ('nucleus', 'cytoplasm', 'nucleus_local', 'cytoplasm_local'):
            self.assertEqual(embeddings[name].shape, (1, self.model.dim))
        self.assertEqual(output['branch_joint'].shape, (1, 2 * self.model.dim))
        self.assertEqual(output['branch_intrinsic'].shape, (1, 2 * self.model.dim))
        self.assertEqual(output['residual'].shape, (1, self.model.backbone.dim))
        self.assertEqual(float(output['residual'].abs().sum()), 0.)
        self.assertEqual(self.model.arch_config['model_type'], 'subcellular_whole_cls_residual_v5')
        self.assertEqual(self.model.arch_config['branch_dim'], 2 * self.model.dim)
        self.assertIn('no_post_norm', self.model.arch_config['fusion'])
        self.assertTrue(self.model.arch_config['spatial_enabled'])

    def test_base_ignores_patch_branch_coordinates_and_compartment_embedding(self):
        expected = self.model(self.batch, return_embeddings=True)['embeddings']['base']
        changed = copy.deepcopy(self.batch)
        changed['gene_ids'][..., 1:] = torch.randint(2, 30, changed['gene_ids'][..., 1:].shape)
        changed['input_bins'][..., 1:] = torch.randn_like(changed['input_bins'][..., 1:]) * 100
        changed['compartment_ids'][..., 1:] = 1 - changed['compartment_ids'][..., 1:].clamp_max(1)
        changed['coords'] = torch.randn_like(changed['coords']) * 1000
        with torch.no_grad():
            self.model.compartment_embedding.weight.normal_(mean=30., std=10.)
        actual = self.model(changed, return_embeddings=True)['embeddings']['base']
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_whole_padding_permutation_and_target_values_do_not_leak(self):
        before = self.model(self.batch)
        changed = copy.deepcopy(self.batch)
        changed['target_bins'].fill_(99999)
        changed['whole_gene_ids'][0, -1] = self.model.pad_id
        changed['whole_input_bins'][0, -1] = 1e8
        changed['whole_token_valid_mask'][0, -1] = True
        after = self.model(changed)
        torch.testing.assert_close(before['embeddings']['base'], after['embeddings']['base'], atol=0, rtol=0)
        self.close(before['joint_logits'], after['joint_logits'])

        permuted = copy.deepcopy(self.batch)
        order = torch.tensor([0, 5, 2, 6, 1, 4, 3])
        for name in ('whole_gene_ids', 'whole_input_bins', 'whole_token_valid_mask'):
            permuted[name] = permuted[name][:, order]
        self.close(before['embeddings']['base'], self.model(permuted)['embeddings']['base'])

        # Invalid non-PAD payload is masked just as strictly as explicit PAD.
        invalid = copy.deepcopy(self.batch)
        invalid['whole_gene_ids'][0, -1] = 29
        invalid['whole_input_bins'][0, -1] = -99999.
        self.close(before['embeddings']['base'], self.model(invalid)['embeddings']['base'])

    def test_whole_input_contract_rejects_bad_shape_mask_and_cls(self):
        for field in ('whole_gene_ids', 'whole_input_bins', 'whole_token_valid_mask'):
            bad = copy.deepcopy(self.batch)
            del bad[field]
            with self.assertRaisesRegex(KeyError, field):
                self.model(bad)
        bad = copy.deepcopy(self.batch)
        bad['whole_input_bins'] = bad['whole_input_bins'].unsqueeze(1)
        with self.assertRaisesRegex(ValueError, r'\[B,L\]'):
            self.model(bad)
        bad = copy.deepcopy(self.batch)
        bad['whole_gene_ids'] = bad['whole_gene_ids'].expand(2, -1)
        bad['whole_input_bins'] = bad['whole_input_bins'].expand(2, -1)
        bad['whole_token_valid_mask'] = bad['whole_token_valid_mask'].expand(2, -1)
        with self.assertRaisesRegex(ValueError, 'same batch size'):
            self.model(bad)
        bad = copy.deepcopy(self.batch)
        bad['whole_token_valid_mask'] = bad['whole_token_valid_mask'].long()
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.model(bad)
        for field, value in (('whole_gene_ids', 2), ('whole_token_valid_mask', False)):
            bad = copy.deepcopy(self.batch)
            bad[field][0, 0] = value
            with self.assertRaisesRegex(ValueError, 'valid CLS'):
                self.model(bad)

    def test_backbone_stays_frozen_eval_and_unchanged_after_training_step(self):
        before = {name: value.detach().clone() for name, value in self.model.backbone.state_dict().items()}
        self.model.train()
        self.assertFalse(self.model.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in self.model.backbone.parameters()))
        output = self.model(self.batch)
        loss = F.cross_entropy(
            output['joint_logits'][self.batch['query_valid_mask']],
            self.batch['target_bins'][self.batch['query_valid_mask']])
        loss.backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.model.backbone.parameters()))
        torch.optim.SGD(self.model.parameters(), lr=.05).step()
        for name, value in self.model.backbone.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            self.model.set_backbone_trainable(True)
        with self.assertRaisesRegex(ValueError, 'zero'):
            self.model.set_backbone_trainable_layers(1)

    def test_zero_projection_learns_first_then_opens_branch_gradients(self):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.2)

        def backward():
            optimizer.zero_grad(set_to_none=True)
            output = self.model(self.batch)
            selected = self.batch['query_valid_mask']
            loss = F.cross_entropy(
                output['joint_logits'][selected], self.batch['target_bins'][selected])
            loss.backward()
            return loss

        backward()
        projection_grad = self.model.residual_projection.weight.grad
        self.assertIsNotNone(projection_grad)
        self.assertGreater(float(projection_grad.abs().sum()), 0.)
        # The zero projection blocks the reconstruction gradient into slots on
        # the first step (the shared compartment identity also serves queries).
        for parameter in (self.model.slot_pool.queries,
                          self.model.local_pre_blocks[0].attention.out_proj.weight,
                          self.model.spatial_blocks[0].attention.out.weight,
                          self.model.local_post_blocks[0].attention.out_proj.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(float(parameter.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertGreater(float(self.model.residual_projection.weight.abs().sum()), 0.)

        backward()
        for parameter in (self.model.slot_pool.queries,
                          self.model.compartment_embedding.weight,
                          self.model.local_pre_blocks[0].attention.out_proj.weight,
                          self.model.spatial_blocks[0].attention.out.weight,
                          self.model.local_post_blocks[0].attention.out_proj.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_only_cytoplasm_crosses_cells_and_feedback_can_change_center_nucleus(self):
        activate_branch(self.model)
        changed = copy.deepcopy(self.batch)
        # Change only a neighboring cell's cytoplasmic genes.
        changed['gene_ids'][0, 1, 3:5] = torch.tensor([19, 20])
        before = self.model(self.batch, return_embeddings=True)
        after = self.model(changed, return_embeddings=True)
        for name in ('base', 'intrinsic', 'nucleus_local', 'cytoplasm_local'):
            self.close(before['embeddings'][name], after['embeddings'][name])
        self.changed(before['embeddings']['cytoplasm'], after['embeddings']['cytoplasm'])
        self.changed(before['embeddings']['nucleus'], after['embeddings']['nucleus'])
        self.changed(before['embeddings']['joint'], after['embeddings']['joint'])

        no_feedback = copy.deepcopy(self.model)
        no_feedback.local_post_blocks = nn.ModuleList()
        left = no_feedback(self.batch, return_embeddings=True)
        right = no_feedback(changed, return_embeddings=True)
        self.close(left['embeddings']['nucleus'], right['embeddings']['nucleus'])
        self.changed(left['embeddings']['cytoplasm'], right['embeddings']['cytoplasm'])

        no_spatial_left = self.model(self.batch, return_embeddings=True, spatial=False)
        no_spatial_right = self.model(changed, return_embeddings=True, spatial=False)
        self.close(no_spatial_left['embeddings']['joint'], no_spatial_right['embeddings']['joint'])
        self.close(no_spatial_left['embeddings']['joint'], no_spatial_left['embeddings']['intrinsic'])

    def test_spatial_control_default_matches_explicit_override_with_same_weights(self):
        control = SubcellularV5(
            '.', neighbors=2, encoder_chunk_size=2, backbone=MockBackbone(),
            slots_per_compartment=2, slot_dim=8, spatial_enabled=False).eval()
        control.load_state_dict(self.model.state_dict())
        self.assertFalse(control.arch_config['spatial_enabled'])
        default = control(self.batch, return_embeddings=True)
        explicit = control(self.batch, return_embeddings=True, spatial=False)
        for name in default['embeddings']:
            torch.testing.assert_close(default['embeddings'][name], explicit['embeddings'][name],
                                       atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'boolean'):
            control(self.batch, spatial=1)

    def test_decoder_reads_only_exact_final_joint_and_gene_compartment_identity(self):
        activate_branch(self.model)
        captured = []
        handle = self.model.joint_decoder.register_forward_pre_hook(
            lambda module, args: captured.append(args))
        output = self.model(self.batch)
        handle.remove()
        query, context = captured[0]
        self.assertEqual(output['joint_logits'].shape, (1, 5, 51))
        self.assertEqual(set(output) - {'compartment_visible_mask'},
                         {'embeddings', 'branch_joint', 'residual', 'joint_logits'})
        self.close(context, output['embeddings']['joint'][:, None].expand_as(context))
        selected = self.batch['query_valid_mask']
        self.close(query.norm(dim=-1)[selected], torch.ones(int(selected.sum())))

    def test_variable_compartment_gene_counts_and_extra_padding_keep_fixed_readouts(self):
        activate_branch(self.model)
        before = self.model(self.batch, return_embeddings=True)
        expanded = copy.deepcopy(self.batch)
        order = torch.tensor([0, 4, 2, 5, 1, 3])
        for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask'):
            expanded[name] = expanded[name][..., order]
        for name, fill in (('gene_ids', 0), ('input_bins', -2.),
                           ('compartment_ids', 0), ('token_valid_mask', False)):
            extra = expanded[name].new_full((1, 3, 4), fill)
            expanded[name] = torch.cat((expanded[name], extra), -1)
        after = self.model(expanded, return_embeddings=True)
        for name in before['embeddings']:
            self.close(before['embeddings'][name], after['embeddings'][name])
        self.close(before['branch_joint'], after['branch_joint'])


if __name__ == '__main__':
    unittest.main()
