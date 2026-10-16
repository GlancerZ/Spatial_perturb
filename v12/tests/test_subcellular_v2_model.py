"""Invariants that protect V2's information boundaries and representation path."""
import unittest

import torch
from torch import nn

from spatial_pretrain.models.subcellular_v2 import SubcellularV2, patch_knn


class MockBackbone(nn.Module):
    """Small post-norm cell encoder with the production backbone protocol."""
    def __init__(self):
        super().__init__()
        self.dim, self.nheads, self.pad_id = 16, 4, 0
        self.vocab = {'<pad>': 0, '<cls>': 1, **{f'G{i}': i + 2 for i in range(14)}}
        self.encoder = nn.Embedding(16, 16, padding_idx=0)
        self.value_encoder = nn.Linear(1, 16)
        layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, genes, values, padding_mask, compartment):
        hidden = self.encoder(genes) + self.value_encoder(values.unsqueeze(-1)) + compartment
        return self.transformer_encoder(hidden, src_key_padding_mask=padding_mask)


def example():
    return {
        'gene_ids': torch.tensor([[[1, 2, 3, 4, 5, 0], [1, 3, 4, 5, 6, 0], [1, 4, 5, 6, 7, 0]]]),
        'input_bins': torch.tensor([[[-2., 3., 8., 7., 2., -2.], [-2., 4., 6., 8., 1., -2.], [-2., 7., 2., 1., 9., -2.]]]),
        'compartment_ids': torch.tensor([[[2, 0, 0, 1, 1, 2]] * 3]),
        'token_valid_mask': torch.tensor([[[True, True, True, True, True, False]] * 3]),
        'coords': torch.tensor([[[0., 0.], [1., 3.], [5., 2.]]]),
        'node_indices': torch.tensor([[10, 20, 30]]),
        'query_gene_ids': torch.tensor([[2, 3, 2, 3]]),
        'query_compartment_ids': torch.tensor([[0, 0, 1, 1]]),
        'query_valid_mask': torch.ones(1, 4, dtype=torch.bool),
        'target_log_counts': torch.ones(1, 4),
        'target_nonzero': torch.ones(1, 4),
        'neighbor_query_mask': torch.tensor([[False, False, True, True]]),
        'task_mode': torch.zeros(1, dtype=torch.long),
    }


class SubcellularV2Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        self.model = SubcellularV2('.', freeze_layers=1, neighbors=2, encoder_chunk_size=2,
                                   checkpointing=True, backbone=MockBackbone()).eval()
        self.batch = example()

    def assert_close(self, left, right):
        torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-6)

    def test_center_cannot_leak_into_pure_niche(self):
        before = self.model(self.batch)
        self.batch['input_bins'][0, 0, 1:5] += 31
        after = self.model(self.batch)
        self.assert_close(before['embeddings']['niche'][:, 0], after['embeddings']['niche'][:, 0])
        self.assert_close(before['neighbor_pred'], after['neighbor_pred'])
        self.assertFalse(torch.allclose(before['embeddings']['joint'][:, 0], after['embeddings']['joint'][:, 0]))

    def test_targets_are_not_forward_inputs(self):
        before = self.model(self.batch)
        self.batch['target_log_counts'].fill_(1000)
        self.batch['target_nonzero'].zero_()
        after = self.model(self.batch)
        for name in ('gene_pred', 'intrinsic_pred', 'joint_pred', 'neighbor_pred'):
            self.assert_close(before[name], after[name])

    def test_unordered_genes_and_padding(self):
        before = self.model(self.batch)
        order = torch.tensor([0, 4, 1, 5, 3, 2])
        for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask'):
            self.batch[name] = self.batch[name][:, :, order]
        after = self.model(self.batch)
        for name in before['embeddings']:
            self.assert_close(before['embeddings'][name], after['embeddings'][name])
        # Invalid tokens cannot influence anything even when their IDs/values change.
        invalid = ~self.batch['token_valid_mask']
        self.batch['gene_ids'][invalid] = 12
        self.batch['input_bins'][invalid] = 48.
        changed = self.model(self.batch)
        for name in before['embeddings']:
            self.assert_close(after['embeddings'][name], changed['embeddings'][name])

    def test_cell_permutation_equivariance(self):
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        order = torch.tensor([0, 2, 1])
        for name in ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask', 'coords', 'node_indices'):
            self.batch[name] = self.batch[name][:, order]
        after = self.model(self.batch, return_embeddings=True)['embeddings']
        for name in before:
            self.assert_close(before[name][:, order], after[name])

    def test_spatial_rigid_motion_and_intrinsic_independence(self):
        before = self.model(self.batch, return_embeddings=True)['embeddings']
        # A rotation plus translation preserves all distance-only spatial outputs.
        rotation = torch.tensor([[0., -1.], [1., 0.]])
        self.batch['coords'] = self.batch['coords'] @ rotation + 12.
        after = self.model(self.batch, return_embeddings=True)['embeddings']
        for name in before:
            self.assert_close(before[name], after[name])
        self.batch['coords'][0, 1] = torch.tensor([50., 50.])
        moved = self.model(self.batch, return_embeddings=True)['embeddings']
        self.assert_close(before['intrinsic'], moved['intrinsic'])

    def test_empty_cells_and_no_neighbors_are_finite(self):
        self.batch['token_valid_mask'][:, :, 1:] = False
        result = self.model(self.batch)
        for value in result['embeddings'].values():
            self.assertTrue(torch.isfinite(value).all())
        for name in ('gene_pred', 'intrinsic_pred', 'joint_pred', 'neighbor_pred'):
            self.assertTrue(torch.isfinite(result[name]).all())
        # Whole-cell masked receiver can still receive, without becoming sender.
        self.batch['token_valid_mask'][0, 1, 1:5] = True
        graph = patch_knn(self.batch['coords'], self.batch['node_indices'],
                          (self.batch['token_valid_mask'] & self.batch['compartment_ids'].eq(1)).any(-1), 2)
        self.assertTrue(graph[1][0, 0].any())
        self.assertFalse(graph[1][0, 1].any())

    def test_no_duplicate_or_padding_senders(self):
        coords = torch.tensor([[[0., 0.], [1., 0.], [1., 0.], [2., 0.], [3., 0.]]])
        ids = torch.tensor([[5, 6, 6, -1, 7]])
        index, valid, _ = patch_knn(coords, ids, torch.ones_like(ids, dtype=torch.bool), 4)
        picked = ids[0, index[0, 0][valid[0, 0]]]
        self.assertEqual(sorted(picked.tolist()), [6, 7])
        self.assertFalse(valid[0, 3].any())
        # Duplicate center entries are both excluded from its neighbor set.
        self.assertFalse((ids[0, index[0, 1][valid[0, 1]]] == 6).any())

    def test_frozen_backbone_does_not_cut_compartment_gradient(self):
        self.model.train()
        result = self.model(self.batch)
        loss = sum(result[name].square().mean() for name in
                   ('gene_pred', 'intrinsic_pred', 'joint_pred', 'neighbor_pred'))
        loss.backward()
        gradient = self.model.compartment_embedding.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.backbone.transformer_encoder.layers[0].parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.backbone.transformer_encoder.layers[1].parameters()))
        self.model.set_backbone_trainable(full=True)
        self.assertTrue(all(parameter.requires_grad for parameter in self.model.backbone.parameters()))

    def test_pooling_ignores_sentinel(self):
        # Empty compartment readout uses its missing state, independent of CLS.
        from spatial_pretrain.models.subcellular_v2 import compartment_mean
        hidden = torch.randn(1, 3, 16)
        valid = torch.tensor([[False, True, True]])
        expected = hidden[:, 1:].mean(1)
        self.assert_close(compartment_mean(hidden, valid, torch.zeros(16)), expected)
        self.assert_close(compartment_mean(hidden, torch.zeros_like(valid), torch.ones(16)), torch.ones(1, 16))


if __name__ == '__main__':
    unittest.main()
