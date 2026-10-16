"""CPU checks for V3 CLS information boundaries, residuals and binned losses."""
import copy
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spatial_pretrain.models.subcellular_v3 import SubcellularV3
from spatial_pretrain.models.subcellular_v2 import patch_knn
from spatial_pretrain.training.subcellular_v3 import binned_reconstruction_loss, losses


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_kind = 'scgpt'
        self.dim, self.nheads, self.pad_id = 16, 4, 0
        self.config = {'pad_value': -2.}
        self.vocab = {'<pad>': 0, '<cls>': 1, **{f'G{i}': i + 2 for i in range(18)}}
        self.encoder = nn.Embedding(20, 16, padding_idx=0)
        self.value_encoder = nn.Linear(1, 16)
        layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0., batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, genes, bins, padding, compartment):
        hidden = self.encoder(genes) + self.value_encoder(bins.unsqueeze(-1)) + compartment
        return self.transformer_encoder(hidden, src_key_padding_mask=padding)


def example():
    return dict(
        gene_ids=torch.tensor([[[1, 2, 3, 4, 5, 0], [1, 3, 4, 5, 6, 0], [1, 4, 5, 6, 7, 0]]]),
        input_bins=torch.tensor([[[-2., 3., 8., 7., 2., -2.], [-2., 4., 6., 8., 1., -2.],
                                  [-2., 7., 2., 1., 9., -2.]]]),
        compartment_ids=torch.tensor([[[2, 0, 0, 1, 1, 2]] * 3]),
        token_valid_mask=torch.tensor([[[True, True, True, True, True, False]] * 3]),
        coords=torch.tensor([[[0., 0.], [1., 3.], [5., 2.]]]),
        node_indices=torch.tensor([[10, 20, 30]]),
        query_gene_ids=torch.tensor([[2, 3, 4, 5, 0]]),
        query_compartment_ids=torch.tensor([[0, 0, 1, 1, 1]]),
        query_valid_mask=torch.tensor([[True, True, True, True, False]]),
        target_bins=torch.tensor([[0, 1, 50, 18, -2]]),
        task_mode=torch.zeros(1, dtype=torch.long),
    )


def activate_exchange(model, spatial=True):
    # Nonzero projections test the learned architecture, not just its initial identity.
    with torch.no_grad():
        for block in model.local_blocks:
            nn.init.normal_(block.attention.out_proj.weight, std=.1)
            nn.init.normal_(block.ffn[-1].weight, std=.04)
        if spatial:
            for block in model.spatial_blocks:
                nn.init.normal_(block.attention.out.weight, std=.12)
                nn.init.normal_(block.attention.distance_bias[-1].weight, std=.1)


class V3ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        torch.backends.mha.set_fastpath_enabled(False)
        self.model = SubcellularV3('.', neighbors=2, encoder_chunk_size=2, backbone=MockBackbone()).eval()
        self.batch = example()

    def close(self, left, right):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-6)

    def test_initial_identity_cls_origin_and_sequence_isolation(self):
        seen = []
        handle = self.model.backbone.register_forward_pre_hook(
            lambda module, args: seen.append(tuple(value.detach().clone() for value in args)))
        output = self.model(self.batch, return_embeddings=True)
        handle.remove()
        embeddings = output['embeddings']
        self.assertEqual(set(embeddings), {'base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm'})
        self.assertEqual(embeddings['joint'].shape, (1, 3, 32))
        torch.testing.assert_close(embeddings['base'], embeddings['joint'], atol=0, rtol=0)
        torch.testing.assert_close(embeddings['base'], embeddings['intrinsic'], atol=0, rtol=0)
        torch.testing.assert_close(embeddings['nucleus'], embeddings['base'][..., :16], atol=0, rtol=0)
        torch.testing.assert_close(embeddings['cytoplasm'], embeddings['base'][..., 16:], atol=0, rtol=0)
        genes = torch.cat([item[0] for item in seen])
        bins = torch.cat([item[1] for item in seen])
        padding = torch.cat([item[2] for item in seen])
        self.assertEqual(genes.tolist(), [[1, 2, 3], [1, 4, 5], [1, 3, 4],
                                         [1, 5, 6], [1, 4, 5], [1, 6, 7]])
        self.assertTrue(bins[:, 0].eq(-2).all())
        self.assertFalse(padding[:, 0].any())
        self.assertTrue(output['compartment_visible_mask'].all())

    def test_gene_permutation_and_padding_do_not_change_outputs_after_exchange(self):
        activate_exchange(self.model)
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        order = torch.tensor([0, 4, 1, 5, 3, 2])
        for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask'):
            self.batch[name] = self.batch[name][..., order]
        invalid = ~self.batch['token_valid_mask']
        self.batch['gene_ids'][invalid] = 17
        self.batch['input_bins'][invalid] = 999.
        self.batch['compartment_ids'][invalid] = 0
        after = self.model(self.batch, return_embeddings=True)['embeddings']
        for name in before:
            self.close(before[name], after[name])

    def test_all_zero_residual_output_projections_receive_gradients(self):
        self.model.train()
        output = self.model(self.batch)
        loss, _ = losses(output, self.batch, {'loss': {'compartment': 1., 'joint': 1.}}, 1)
        loss.backward()
        projections = [block.attention.out_proj for block in self.model.local_blocks]
        projections += [block.ffn[-1] for block in self.model.local_blocks]
        projections += [block.attention.out for block in self.model.spatial_blocks]
        for projection in projections:
            self.assertIsNotNone(projection.weight.grad)
            self.assertTrue(torch.isfinite(projection.weight.grad).all())
            self.assertGreater(float(projection.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.backbone.parameters()))

    def test_frozen_encoder_transmits_compartment_input_gradient_and_top_k_control(self):
        initial, _ = self.model.encode_compartments(self.batch)
        (initial * torch.randn_like(initial)).sum().backward()
        self.assertGreater(float(self.model.compartment_embedding.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.backbone.parameters()))
        self.model.set_backbone_trainable_layers(1)
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.encoder.parameters()))
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.transformer_encoder.layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in self.model.backbone.transformer_encoder.layers[1].parameters()))
        self.model.set_backbone_trainable(full=True)
        self.assertTrue(all(p.requires_grad for p in self.model.backbone.parameters()))
        self.model.set_backbone_trainable_layers(0)
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
        with self.assertRaises(ValueError):
            self.model.set_backbone_trainable_layers(3)

    def test_same_cell_nuclear_and_cytoplasmic_states_influence_each_other(self):
        activate_exchange(self.model, spatial=False)
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        for source_compartment, target_name, untouched_half in ((0, 'cytoplasm', slice(16, 32)),
                                                               (1, 'nucleus', slice(0, 16))):
            changed = copy.deepcopy(self.batch)
            selected = changed['compartment_ids'][0, 0].eq(source_compartment)
            changed['gene_ids'][0, 0, selected] = torch.tensor([11, 12])
            after = self.model(changed, return_embeddings=True)['embeddings']
            self.close(before['base'][:, 0, untouched_half], after['base'][:, 0, untouched_half])
            self.assertGreater(float((before[target_name][:, 0] - after[target_name][:, 0]).abs().max()), 1e-5)

    def test_spatial_block_accepts_cytoplasm_only_and_leaves_nucleus_unchanged(self):
        activate_exchange(self.model)
        initial, visible = self.model.encode_compartments(self.batch)
        graph = patch_knn(self.batch['coords'], self.batch['node_indices'], visible[..., 1], 2)
        block = self.model.spatial_blocks[0]
        captured = []
        handle = block.attention.register_forward_pre_hook(lambda module, args: captured.append(args))
        after = block(initial, graph)
        handle.remove()
        self.close(captured[0][0], initial[..., 1, :])
        self.close(captured[0][1], initial[..., 1, :])
        torch.testing.assert_close(after[..., 0, :], initial[..., 0, :], atol=0, rtol=0)
        self.assertGreater(float((after[..., 1, :] - initial[..., 1, :]).abs().max()), 1e-5)

    def test_neighbor_nucleus_can_influence_center_nucleus_only_indirectly(self):
        activate_exchange(self.model)
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        self.batch['gene_ids'][0, 1, 1:3] = torch.tensor([11, 12])
        after = self.model(self.batch, return_embeddings=True)['embeddings']
        self.close(before['base'][:, 0], after['base'][:, 0])
        self.close(before['intrinsic'][:, 0], after['intrinsic'][:, 0])
        self.assertGreater(float((before['nucleus'][:, 0] - after['nucleus'][:, 0]).abs().max()), 1e-5)

    def test_hidden_cytoplasm_never_becomes_sender_after_local_imputation(self):
        activate_exchange(self.model)
        self.batch['token_valid_mask'][0, 1, 3:5] = False
        before = self.model(self.batch, return_embeddings=True)
        self.assertFalse(before['compartment_visible_mask'][0, 1, 1])
        self.batch['gene_ids'][0, 1, 1:3] = torch.tensor([11, 12])
        after = self.model(self.batch, return_embeddings=True)
        self.close(before['embeddings']['joint'][:, 0], after['embeddings']['joint'][:, 0])
        # The same masked cell is nevertheless a receiver of observed neighbors.
        self.assertGreater(float((before['embeddings']['joint'][:, 1]
                                  - before['embeddings']['intrinsic'][:, 1]).abs().max()), 1e-5)

    def test_coords_affect_only_spatial_path_and_targets_are_never_forward_inputs(self):
        activate_exchange(self.model)
        before = self.model(self.batch)
        self.batch['target_bins'].fill_(99999)
        self.batch['target_nonzero'] = torch.zeros_like(self.batch['target_bins'])
        after = self.model(self.batch)
        self.close(before['compartment_pred'], after['compartment_pred'])
        self.close(before['joint_pred'], after['joint_pred'])
        embeddings = self.model(self.batch, return_embeddings=True)['embeddings']
        self.batch['coords'][0, 1] = torch.tensor([40., 50.])
        moved = self.model(self.batch, return_embeddings=True)['embeddings']
        self.close(embeddings['base'], moved['base'])
        self.close(embeddings['intrinsic'], moved['intrinsic'])
        self.assertGreater(float((embeddings['joint'] - moved['joint']).abs().max()), 1e-5)

    def test_empty_compartments_padding_cells_and_no_neighbors_are_finite(self):
        activate_exchange(self.model)
        self.batch['token_valid_mask'][..., 1:] = False
        self.batch['node_indices'][0, 2] = -1
        output = self.model(self.batch)
        for tensor in (*output['embeddings'].values(), output['compartment_pred'], output['joint_pred']):
            self.assertTrue(torch.isfinite(tensor).all())
        self.assertFalse(output['compartment_visible_mask'].any())
        self.assertEqual(float(output['embeddings']['joint'][0, 2].abs().sum()), 0.)
        loss, parts = losses(output, self.batch, {'loss': {'compartment': 1., 'joint': 1.}}, 3)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts['compartment'], 0.)
        self.assertGreater(parts['joint'], 0.)
        self.batch['token_valid_mask'][0, 1, 3:5] = True
        output = self.model(self.batch, return_embeddings=True)
        self.assertGreater(float((output['embeddings']['joint'][0, 0]
                                  - output['embeddings']['intrinsic'][0, 0]).abs().max()), 1e-5)

    def test_spatial_backbone_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ordinary'):
            SubcellularV3('.', backbone_kind='scgpt_spatial', backbone=MockBackbone())

    def test_decoder_query_and_each_cls_context_have_matching_unit_norm(self):
        captured = {}
        handles = [decoder.register_forward_pre_hook(
            lambda module, arguments, name=name: captured.update({name: arguments}))
            for name, decoder in (('compartment', self.model.compartment_decoder),
                                  ('joint', self.model.joint_decoder))]
        try:
            self.model(self.batch)
        finally:
            for handle in handles:
                handle.remove()
        valid = self.batch['query_valid_mask']
        for name, (query, context) in captured.items():
            self.close(query.norm(dim=-1)[valid], torch.ones(int(valid.sum())))
            for block in context.split(self.model.dim, dim=-1):
                self.close(block.norm(dim=-1)[valid], torch.ones(int(valid.sum())))


class V3LossTests(unittest.TestCase):
    def test_bin_scale_zero_padding_and_balanced_compartment_reduction(self):
        batch = dict(target_bins=torch.tensor([[0, 1, 50, -2]]),
                     query_compartment_ids=torch.tensor([[0, 0, 1, 1]]),
                     query_valid_mask=torch.tensor([[True, True, True, False]]))
        prediction = torch.zeros(1, 4, 2, requires_grad=True)
        loss = binned_reconstruction_loss(prediction, batch)
        self.assertAlmostEqual(float(loss), math.log(2) + .25, places=6)
        loss.backward()
        self.assertEqual(float(prediction.grad[0, 0, 1]), 0.)  # a zero bin has no positive-rank target
        self.assertEqual(float(prediction.grad[0, 3].abs().sum()), 0.)  # PAD is never a query
        self.assertGreater(float(prediction.grad[0, 1, 1]), 0.)
        self.assertLess(float(prediction.grad[0, 2, 1]), 0.)

    def test_compartment_loss_skips_hidden_compartment_but_joint_uses_its_queries(self):
        batch = example()
        batch['token_valid_mask'][0, 0, 3:5] = False
        output = {'compartment_pred': torch.zeros(1, 5, 2, requires_grad=True),
                  'joint_pred': torch.zeros(1, 5, 2, requires_grad=True)}
        total, parts = losses(output, batch, {'loss': {'compartment': 1., 'joint': 1.}}, 10)
        total.backward()
        self.assertEqual(float(output['compartment_pred'].grad[0, 2:].abs().sum()), 0.)
        self.assertGreater(float(output['joint_pred'].grad[0, 2:4].abs().sum()), 0.)
        self.assertAlmostEqual(parts['total'], parts['compartment'] + parts['joint'], places=6)

    def test_empty_masks_are_differentiable_zero_and_bad_valid_bins_raise(self):
        batch = dict(target_bins=torch.tensor([[-2, -2]]), query_compartment_ids=torch.tensor([[0, 1]]),
                     query_valid_mask=torch.zeros(1, 2, dtype=torch.bool))
        prediction = torch.zeros(1, 2, 2, requires_grad=True)
        loss = binned_reconstruction_loss(prediction, batch)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(prediction.grad.abs().sum()), 0.)
        batch['query_valid_mask'][0, 0] = True
        with self.assertRaisesRegex(ValueError, '\\[0,50\\]'):
            binned_reconstruction_loss(prediction, batch)

    def test_configured_compartment_and_joint_weights_are_applied(self):
        batch = example()
        output = {'compartment_pred': torch.zeros(1, 5, 2),
                  'joint_pred': torch.ones(1, 5, 2)}
        weighted, parts = losses(output, batch, {'loss': {'compartment': 2., 'joint': .25}}, 1)
        self.assertAlmostEqual(float(weighted), 2 * parts['compartment'] + .25 * parts['joint'], places=6)
        joint_only, _ = losses(output, batch, {'loss': {'compartment': 0., 'joint': 1.}}, 1)
        self.assertAlmostEqual(float(joint_only), parts['joint'], places=6)


if __name__ == '__main__':
    unittest.main()
