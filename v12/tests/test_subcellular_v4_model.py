"""Small-backbone V4 information-boundary, gradient and single-objective checks."""
import copy
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from spatial_pretrain.models.subcellular_v2 import patch_knn
from spatial_pretrain.models.subcellular_v4 import SubcellularV4
from spatial_pretrain.training.subcellular_v4 import masked_bin_cross_entropy, losses


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
    # Test reachable learned paths separately from their zero-init identity.
    with torch.no_grad():
        for block in (*model.local_pre_blocks, *model.local_post_blocks):
            nn.init.normal_(block.attention.out_proj.weight, std=.15)
            nn.init.normal_(block.ffn[-1].weight, std=.04)
        if spatial:
            for block in model.spatial_blocks:
                nn.init.normal_(block.attention.out.weight, std=.15)
                nn.init.normal_(block.attention.distance_bias[-1].weight, std=.15)


class V4ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(47)
        torch.backends.mha.set_fastpath_enabled(False)
        self.model = SubcellularV4('.', neighbors=2, encoder_chunk_size=2, backbone=MockBackbone(),
                                   slots_per_compartment=2, slot_dim=8).eval()
        self.batch = example()

    def close(self, left, right):
        torch.testing.assert_close(left, right, atol=4e-6, rtol=4e-6)

    def changed(self, left, right):
        self.assertGreater(float((left - right).abs().max()), 1e-6)

    def test_initial_residual_identity_and_independent_compartment_sequences(self):
        seen = []
        handle = self.model.backbone.register_forward_pre_hook(
            lambda module, args: seen.append(tuple(value.detach().clone() for value in args)))
        output = self.model(self.batch, return_embeddings=True)
        handle.remove()
        embeddings = output['embeddings']
        self.assertEqual(set(embeddings), {'base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
                                           'nucleus_local', 'cytoplasm_local'})
        self.assertEqual(embeddings['joint'].shape, (1, 3, 32))
        self.assertEqual(embeddings['nucleus'].shape, (1, 3, 16))
        torch.testing.assert_close(embeddings['base'], embeddings['joint'], atol=0, rtol=0)
        torch.testing.assert_close(embeddings['base'], embeddings['intrinsic'], atol=0, rtol=0)
        genes = torch.cat([item[0] for item in seen])
        self.assertEqual(genes.tolist(), [[1, 2, 3], [1, 4, 5], [1, 3, 4],
                                         [1, 5, 6], [1, 4, 5], [1, 6, 7]])
        self.assertTrue(output['compartment_visible_mask'].all())
        self.assertEqual(self.model.arch_config['num_bins'], 51)
        self.assertNotIn('checkpointing', self.model.arch_config)
        self.assertFalse(hasattr(self.model, 'compartment_decoder'))

    def test_gene_permutation_padding_and_extra_pad_tokens_are_invariant(self):
        activate_exchange(self.model)
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        order = torch.tensor([0, 4, 1, 5, 3, 2])
        for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask'):
            self.batch[name] = self.batch[name][..., order]
        invalid = ~self.batch['token_valid_mask']
        self.batch['gene_ids'][invalid] = 17
        self.batch['input_bins'][invalid] = 999.
        self.batch['compartment_ids'][invalid] = 0
        for name, pad in [('gene_ids', 0), ('input_bins', -2.),
                          ('compartment_ids', 0), ('token_valid_mask', False)]:
            extra = self.batch[name].new_full((1, 3, 3), pad)
            self.batch[name] = torch.cat((self.batch[name], extra), -1)
        after = self.model(self.batch, return_embeddings=True)['embeddings']
        for name in before:
            self.close(before[name], after[name])

    def test_variable_gene_counts_and_sequence_lengths_keep_slot_shape_and_use_added_genes(self):
        # Input counts differ between cells and compartments, and can exceed K.
        # The SAME model must aggregate all valid genes into its fixed K slots.
        short_genes = [([2], [5, 6]), ([3, 4, 8], [7]), ([4, 5], [6, 7, 9, 10])]
        long_genes = copy.deepcopy(short_genes)
        long_genes[0][0].extend([11, 12, 13, 14, 15])
        long_genes[2][1].extend([16, 17, 18, 19])

        def pack(compartments):
            batch = copy.deepcopy(self.batch)
            length = 1 + max(len(nuclear) + len(cytoplasm) for nuclear, cytoplasm in compartments)
            batch['gene_ids'] = torch.zeros((1, 3, length), dtype=torch.long)
            batch['input_bins'] = torch.full((1, 3, length), -2.)
            batch['compartment_ids'] = torch.full((1, 3, length), 2, dtype=torch.long)
            batch['token_valid_mask'] = torch.zeros((1, 3, length), dtype=torch.bool)
            batch['gene_ids'][..., 0] = 1
            batch['token_valid_mask'][..., 0] = True
            for cell, pair in enumerate(compartments):
                offset = 1
                for compartment, genes in enumerate(pair):
                    selection = slice(offset, offset + len(genes))
                    batch['gene_ids'][0, cell, selection] = torch.tensor(genes)
                    batch['input_bins'][0, cell, selection] = torch.tensor(genes, dtype=torch.float32) * 2
                    batch['compartment_ids'][0, cell, selection] = compartment
                    batch['token_valid_mask'][0, cell, selection] = True
                    offset += len(genes)
            return batch

        short, long = pack(short_genes), pack(long_genes)
        self.assertEqual(short['gene_ids'].shape[-1], 7)
        self.assertEqual(long['gene_ids'].shape[-1], 11)
        short_counts = [[len(nuclear), len(cytoplasm)] for nuclear, cytoplasm in short_genes]
        long_counts = [[len(nuclear), len(cytoplasm)] for nuclear, cytoplasm in long_genes]
        self.assertEqual(short_counts, [[1, 2], [3, 1], [2, 4]])
        self.assertEqual(long_counts, [[6, 2], [3, 1], [2, 8]])
        with torch.no_grad():
            short_slots, short_visible = self.model.encode_compartments(short)
            long_slots, long_visible = self.model.encode_compartments(long)
            for slots, visible in ((short_slots, short_visible), (long_slots, long_visible)):
                self.assertEqual(slots.shape, (1, 3, 2, self.model.slots_per_compartment, self.model.slot_dim))
                self.assertTrue(visible.all())
                self.assertTrue(torch.isfinite(slots).all())
            for cell in range(3):
                for compartment in range(2):
                    left, right = short_slots[0, cell, compartment], long_slots[0, cell, compartment]
                    if (cell, compartment) in ((0, 0), (2, 1)):
                        self.changed(left, right)
                    else:
                        self.close(left, right)
            for batch in (short, long):
                embeddings = self.model(batch, return_embeddings=True)['embeddings']
                for name, embedding in embeddings.items():
                    width = 2 * self.model.dim if name in ('base', 'intrinsic', 'joint') else self.model.dim
                    self.assertEqual(embedding.shape, (1, 3, width))
                    self.assertTrue(torch.isfinite(embedding).all())
                self.assertEqual(self.model(batch)['joint_logits'].shape, (1, 5, 51))

    def test_pool_uses_gene_states_excludes_cls_pad_and_retains_slot_identity(self):
        hidden = torch.randn(2, 4, 16)
        types = torch.tensor([0, 1])
        valid = torch.tensor([[False, True, False, False], [False, False, False, False]])
        before = self.model.slot_pool(hidden, valid, types)
        hidden[~valid] = 1000 * torch.randn_like(hidden[~valid])
        after = self.model.slot_pool(hidden, valid, types)
        self.close(before, after)
        self.changed(before[0, 0], before[0, 1])
        hidden[0, 1] = torch.randn(16) * 2
        changed = self.model.slot_pool(hidden, valid, types)
        self.changed(before[0], changed[0])
        self.close(before[1], changed[1])
        self.assertTrue(torch.isfinite(changed).all())

    def test_zero_init_projections_and_pool_receive_gradients_through_frozen_encoder(self):
        self.model.train()
        self.assertFalse(self.model.backbone.training)
        output = self.model(self.batch)
        total, parts = losses(output, self.batch, {'loss': {'type': 'masked_bin_cross_entropy'}}, 0)
        total.backward()
        self.assertEqual(set(parts), {'total', 'masked_ce'})
        projections = [block.attention.out_proj for block in
                       (*self.model.local_pre_blocks, *self.model.local_post_blocks)]
        projections += [block.ffn[-1] for block in
                        (*self.model.local_pre_blocks, *self.model.local_post_blocks)]
        projections += [block.attention.out for block in self.model.spatial_blocks]
        parameters = [projection.weight for projection in projections]
        parameters += [self.model.compartment_embedding.weight, self.model.slot_pool.queries,
                       self.model.slot_pool.gene_projection.weight,
                       self.model.slot_pool.attention.in_proj_weight]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.backbone.parameters()))

    def test_backbone_layer_control_and_chunked_checkpoint_equivalence(self):
        other = copy.deepcopy(self.model)
        other.encoder_chunk_size = 64
        other.checkpointing = False
        expected = self.model(self.batch)['joint_logits']
        actual = other(self.batch)['joint_logits']
        self.close(expected, actual)
        expected.square().sum().backward()
        actual.square().sum().backward()
        self.close(self.model.slot_pool.queries.grad, other.slot_pool.queries.grad)
        self.close(self.model.compartment_embedding.weight.grad, other.compartment_embedding.weight.grad)
        self.model.set_backbone_trainable_layers(1)
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.encoder.parameters()))
        layers = self.model.backbone.transformer_encoder.layers
        self.assertTrue(all(not p.requires_grad for p in layers[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in layers[1].parameters()))
        self.model.set_backbone_trainable(full=True)
        self.assertTrue(all(p.requires_grad for p in self.model.backbone.parameters()))
        self.model.set_backbone_trainable_layers(0)
        self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
        with self.assertRaises(ValueError):
            self.model.set_backbone_trainable_layers(3)

    def test_local_pre_exchanges_both_compartments(self):
        activate_exchange(self.model, spatial=False)
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        for source, target, untouched in ((0, 'cytoplasm_local', slice(16, 32)),
                                           (1, 'nucleus_local', slice(0, 16))):
            changed = copy.deepcopy(self.batch)
            mask = changed['compartment_ids'][0, 0].eq(source)
            changed['gene_ids'][0, 0, mask] = torch.tensor([11, 12])
            after = self.model(changed, return_embeddings=True)['embeddings']
            self.close(before['base'][:, 0, untouched], after['base'][:, 0, untouched])
            self.changed(before[target][:, 0], after[target][:, 0])

    def test_spatial_qkv_are_flattened_cytoplasm_and_nucleus_is_bypassed(self):
        activate_exchange(self.model)
        initial, visible = self.model.encode_compartments(self.batch)
        graph = patch_knn(self.batch['coords'], self.batch['node_indices'], visible[..., 1], 2)
        block = self.model.spatial_blocks[0]
        captured = []
        handle = block.attention.register_forward_pre_hook(lambda module, args: captured.append(args))
        after = block(initial, graph)
        handle.remove()
        self.close(captured[0][0], initial[..., 1, :, :].flatten(-2))
        self.close(captured[0][1], initial[..., 1, :, :].flatten(-2))
        torch.testing.assert_close(after[..., 0, :, :], initial[..., 0, :, :], atol=0, rtol=0)
        self.changed(after[..., 1, :, :], initial[..., 1, :, :])

    def test_neighbor_information_reaches_final_nucleus_only_through_spatial_and_feedback(self):
        activate_exchange(self.model)
        changed = copy.deepcopy(self.batch)
        changed['gene_ids'][0, 1, 3:5] = torch.tensor([11, 12])
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        after = self.model(changed, return_embeddings=True)['embeddings']
        for name in ('base', 'intrinsic', 'nucleus_local', 'cytoplasm_local'):
            self.close(before[name][:, 0], after[name][:, 0])
        self.changed(before['nucleus'][:, 0], after['nucleus'][:, 0])
        self.changed(before['cytoplasm'][:, 0], after['cytoplasm'][:, 0])
        no_feedback = copy.deepcopy(self.model)
        no_feedback.local_post_blocks = nn.ModuleList()
        left = no_feedback(self.batch, return_embeddings=True)['embeddings']
        right = no_feedback(changed, return_embeddings=True)['embeddings']
        self.close(left['nucleus'][:, 0], right['nucleus'][:, 0])
        self.changed(left['cytoplasm'][:, 0], right['cytoplasm'][:, 0])
        no_spatial = copy.deepcopy(self.model)
        no_spatial.spatial_blocks = nn.ModuleList()
        left = no_spatial(self.batch, return_embeddings=True)['embeddings']
        right = no_spatial(changed, return_embeddings=True)['embeddings']
        self.close(left['nucleus'][:, 0], right['nucleus'][:, 0])
        self.close(left['joint'], left['intrinsic'])

    def test_coordinates_only_affect_spatial_and_targets_are_not_model_inputs(self):
        activate_exchange(self.model)
        before = self.model(self.batch)
        self.batch['target_bins'].fill_(99999)
        self.close(before['joint_logits'], self.model(self.batch)['joint_logits'])
        embeddings = self.model(self.batch, return_embeddings=True)['embeddings']
        self.batch['coords'][0, 1] = torch.tensor([40., 50.])
        moved = self.model(self.batch, return_embeddings=True)['embeddings']
        for name in ('base', 'intrinsic', 'nucleus_local', 'cytoplasm_local'):
            self.close(embeddings[name], moved[name])
        self.changed(embeddings['joint'], moved['joint'])
        self.changed(embeddings['nucleus'], moved['nucleus'])

    def test_original_empty_cytoplasm_cannot_send_after_local_or_spatial_imputation(self):
        activate_exchange(self.model)
        self.batch['token_valid_mask'][0, 1, 3:5] = False
        before = self.model(self.batch, return_embeddings=True)
        self.assertFalse(before['compartment_visible_mask'][0, 1, 1])
        self.batch['gene_ids'][0, 1, 1:3] = torch.tensor([11, 12])
        after = self.model(self.batch, return_embeddings=True)
        self.close(before['embeddings']['joint'][:, 0], after['embeddings']['joint'][:, 0])
        self.changed(before['embeddings']['joint'][:, 1], before['embeddings']['intrinsic'][:, 1])

    def test_empty_compartments_no_neighbors_and_padding_cells_are_finite(self):
        activate_exchange(self.model)
        self.batch['token_valid_mask'][..., 1:] = False
        self.batch['node_indices'][0, 2] = -1
        output = self.model(self.batch)
        for value in (*output['embeddings'].values(), output['joint_logits']):
            self.assertTrue(torch.isfinite(value).all())
        self.assertFalse(output['compartment_visible_mask'].any())
        self.assertEqual(float(output['embeddings']['joint'][0, 2].abs().sum()), 0.)
        total, _ = losses(output, self.batch, {}, 0)
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(float(self.model.slot_pool.missing.grad.abs().sum()), 0.)
        embeddings = self.model(self.batch, return_embeddings=True)['embeddings']
        self.close(embeddings['joint'], embeddings['intrinsic'])
        self.batch['token_valid_mask'][0, 1, 3:5] = True
        embeddings = self.model(self.batch, return_embeddings=True)['embeddings']
        self.changed(embeddings['joint'][:, 0], embeddings['intrinsic'][:, 0])

    def test_graph_excludes_self_duplicate_and_invalid_senders(self):
        ids = torch.tensor([[10, 20, 20, -1]])
        coords = torch.tensor([[[0., 0.], [1., 0.], [1., 0.], [3., 0.]]])
        indices, valid, _ = patch_knn(coords, ids, torch.ones_like(ids, dtype=torch.bool), 4)
        self.assertEqual(indices[0, 0][valid[0, 0]].tolist(), [1])
        self.assertFalse(valid[0, 3].any())
        for row in (1, 2):
            self.assertEqual(indices[0, row][valid[0, row]].tolist(), [0])

    def test_decoder_reads_exact_probe_embedding_and_normalized_query(self):
        captured = []
        handle = self.model.joint_decoder.register_forward_pre_hook(lambda module, args: captured.append(args))
        output = self.model(self.batch)
        handle.remove()
        query, context = captured[0]
        self.assertEqual(output['joint_logits'].shape, (1, 5, 51))
        self.close(context, output['embeddings']['joint'][:, 0, None].expand_as(context))
        selected = self.batch['query_valid_mask']
        self.close(query.norm(dim=-1)[selected], torch.ones(int(selected.sum())))
        for half in context.split(self.model.dim, -1):
            self.close(half.norm(dim=-1)[selected], torch.ones(int(selected.sum())))

    def test_spatial_backbone_and_invalid_architecture_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'ordinary'):
            SubcellularV4('.', backbone_kind='scgpt_spatial', backbone=MockBackbone())
        for parameter in ('slot_dim', 'slots_per_compartment', 'neighbors', 'encoder_chunk_size'):
            with self.assertRaises(ValueError):
                SubcellularV4('.', backbone=MockBackbone(), **{parameter: 0})
        with self.assertRaises(ValueError):
            SubcellularV4('.', backbone=MockBackbone(), local_post_layers=-1)


class V4LossTests(unittest.TestCase):
    def test_query_compartment_and_cell_means_with_analytic_gradient(self):
        batch = dict(target_bins=torch.tensor([[0, 1, 50, -2], [2, 3, -2, -2], [-2, -2, -2, -2]]),
                     query_compartment_ids=torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [0, 0, 1, 1]]),
                     query_valid_mask=torch.tensor([[True, True, True, False], [True, True, False, False],
                                                    [False, False, False, False]]))
        logits = torch.zeros(3, 4, 51, requires_grad=True)
        total = masked_bin_cross_entropy(logits, batch)
        self.assertAlmostEqual(float(total), math.log(51), places=6)
        total.backward()
        weight = torch.tensor([[1 / 8, 1 / 8, 1 / 4, 0.], [1 / 4, 1 / 4, 0., 0.], [0., 0., 0., 0.]])
        expected = torch.ones_like(logits) / 51
        safe_target = batch['target_bins'].clamp_min(0)
        expected.scatter_add_(-1, safe_target.unsqueeze(-1), -torch.ones(3, 4, 1))
        expected *= weight[..., None]
        torch.testing.assert_close(logits.grad, expected)

    def test_unequal_query_counts_match_explicit_manual_reduction(self):
        batch = dict(target_bins=torch.tensor([[0, 1, 50], [2, -2, -2]]),
                     query_compartment_ids=torch.tensor([[0, 0, 1], [1, 0, 0]]),
                     query_valid_mask=torch.tensor([[True, True, True], [True, False, False]]))
        logits = torch.arange(306, dtype=torch.float32).reshape(2, 3, 51) / 100
        logits[0, 1, 1] += 1.
        logp = logits.log_softmax(-1)
        expected = ((-logp[0, 0, 0] - logp[0, 1, 1]) / 2 - logp[0, 2, 50]) / 4 - logp[1, 0, 2] / 2
        torch.testing.assert_close(masked_bin_cross_entropy(logits, batch), expected)

    def test_empty_query_mask_and_nan_padding_are_differentiable_zero(self):
        batch = dict(target_bins=torch.tensor([[float('nan'), -2.]]),
                     query_compartment_ids=torch.tensor([[-99, 1]]),
                     query_valid_mask=torch.zeros(1, 2, dtype=torch.bool))
        logits = torch.full((1, 2, 51), float('nan'), requires_grad=True)
        total = masked_bin_cross_entropy(logits, batch)
        self.assertEqual(float(total), 0.)
        total.backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0.)
        batch['target_bins'][0, 1] = 4.
        batch['query_valid_mask'][0, 1] = True
        logits = torch.cat((torch.full((1, 1, 51), float('nan')), torch.zeros(1, 1, 51)), 1).requires_grad_()
        total = masked_bin_cross_entropy(logits, batch)
        self.assertAlmostEqual(float(total), math.log(51), places=6)
        total.backward()
        self.assertEqual(float(logits.grad[0, 0].abs().sum()), 0.)

    def test_invalid_valid_target_compartment_and_shape_are_rejected(self):
        batch = dict(target_bins=torch.tensor([[0.]]), query_compartment_ids=torch.tensor([[0]]),
                     query_valid_mask=torch.ones(1, 1, dtype=torch.bool))
        logits = torch.zeros(1, 1, 51)
        for bad in (-2., 51., .5, float('nan'), float('inf')):
            batch['target_bins'][0, 0] = bad
            with self.assertRaisesRegex(ValueError, '\\[0,50\\]'):
                masked_bin_cross_entropy(logits, batch)
        batch['target_bins'].zero_()
        batch['query_compartment_ids'].fill_(2)
        with self.assertRaisesRegex(ValueError, 'compartments'):
            masked_bin_cross_entropy(logits, batch)
        with self.assertRaisesRegex(ValueError, 'logits'):
            masked_bin_cross_entropy(logits[..., :2], batch)

    def test_only_one_loss_is_exposed_and_legacy_mixtures_are_rejected(self):
        batch = example()
        output = dict(joint_logits=torch.zeros(1, 5, 51, requires_grad=True))
        total, parts = losses(output, batch, {'loss': {'type': 'masked_bin_cross_entropy'}}, 999)
        self.assertEqual(set(parts), {'total', 'masked_ce'})
        self.assertEqual(parts['total'], parts['masked_ce'])
        self.assertAlmostEqual(float(total), math.log(51), places=6)
        for specification in ({'compartment': 1., 'joint': 1.}, {'type': 'mse'}, {'weight': 2.}):
            with self.assertRaises(ValueError):
                losses(output, batch, {'loss': specification}, 0)


if __name__ == '__main__':
    unittest.main()
