import unittest
from copy import deepcopy
import numpy as np
import torch

from spatial_pretrain.models.spatial_bias import RelativeCellSpatialBias
from spatial_pretrain.models.compartment import CytoplasmicMeanCellAttention, CompartmentSSGTModel
from spatial_pretrain.data.compartment_binning import positive_bins, configured_edges, evaluation_edges
from types import SimpleNamespace
from spatial_pretrain.data.compartment_binning import scgpt_cell_bins


class SpatialBiasTests(unittest.TestCase):
    def test_scgpt_bins_with_position_only_in_cross_cell_stage(self):
        torch.manual_seed(17)
        model = CompartmentSSGTModel(num_genes=5, num_value_bins=32, hidden_dim=8,
            num_layers=1, gene_heads=2, cell_heads=2, dropout=0., ffn_mult=2,
            gene_padding_idx=4, cell_attention_mode='cytoplasm_mean', cell_spatial_bias=True)
        torch.nn.init.normal_(model.blocks[0].cell_attn.spatial_bias.distance_embedding.weight)
        comp = torch.tensor([[[0, 1, 1, 2]] * 3])
        rows = []
        for i in range(3):
            n, c = scgpt_cell_bins(np.log1p([i+1]), np.log1p([1, i+2]),
                                  32, np.random.default_rng(i))
            rows.append(np.r_[n, c, 33])
        batch = dict(gene_ids=torch.tensor([[[0, 1, 2, 4]] * 3]),
            input_bins=torch.tensor(np.array([rows])), compartment_ids=comp,
            token_valid_mask=comp.ne(2), node_indices=torch.tensor([[0, 1, 2]]),
            cell_roles=torch.tensor([[1, 0, 0]]))
        captured = []
        hook = model.blocks[0].gene_attn.register_forward_hook(
            lambda module, args, output: captured.append(output.detach().clone()))
        a = model(**batch, coords=torch.tensor([[[0., 0.], [5., 0.], [100., 0.]]]))['mask_logits']
        first = captured[:]
        captured.clear()
        b = model(**batch, coords=torch.tensor([[[0., 0.], [100., 0.], [5., 0.]]]))['mask_logits']
        hook.remove()
        self.assertEqual(len(first), len(captured))
        for left, right in zip(first, captured):
            torch.testing.assert_close(left, right)
        # In one block the direct coordinate-dependent update reaches only cytoplasm.
        torch.testing.assert_close(a[comp.eq(0)], b[comp.eq(0)])
        self.assertFalse(torch.allclose(a[comp.eq(1)], b[comp.eq(1)]))
        b.square().mean().backward()
        self.assertGreater(model.blocks[0].cell_attn.spatial_bias.distance_embedding.weight.grad.abs().sum(), 0)

    def test_direction_distance_and_translation(self):
        layer = RelativeCellSpatialBias(2, 16, [10., 20.])
        coords = torch.tensor([[[0., 0.], [10., 0.], [0., 10.], [-10., 0.], [0., -10.], [0., 0.]]])
        angle, radial, active = layer.categories(coords)
        self.assertEqual(angle[0, 0, 1:5].tolist(), [0, 4, 8, 12])
        self.assertEqual(radial[0, 0].tolist(), [0, 1, 1, 1, 1, 0])
        self.assertFalse(active[0, 0, 5])
        for actual, expected in zip(layer.categories(coords + 100), (angle, radial, active)):
            torch.testing.assert_close(actual, expected)
        scaled = RelativeCellSpatialBias(2, 16, [10., 20.], coordinate_scale=2.)
        for actual, expected in zip(scaled.categories(coords / 2), (angle, radial, active)):
            torch.testing.assert_close(actual, expected)

    def test_spatial_effect_gradients_and_routing(self):
        torch.manual_seed(7)
        layer = CytoplasmicMeanCellAttention(8, 2, 0., spatial_bias=True)
        tokens = torch.randn(1, 4, 3, 8, requires_grad=True)
        comp = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]])
        nodes = torch.tensor([[0, 1, 2, 2]])
        coords = torch.tensor([[[0., 0.], [5., 0.], [0., 30.], [0., 30.]]])
        base, active = layer(tokens, comp, comp.ne(2), nodes, coords)
        base.square().sum().backward()
        self.assertGreater(layer.spatial_bias.angle_embedding.weight.grad.abs().sum().item(), 0)
        self.assertGreater(layer.spatial_bias.distance_embedding.weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            layer.spatial_bias.angle_embedding.weight[0] = 5
        changed, _ = layer(tokens, comp, comp.ne(2), nodes, coords)
        self.assertFalse(torch.allclose(base, changed))
        self.assertTrue((changed[~active] == 0).all())
        # A duplicated cell must not become an additional sender.
        altered = tokens.detach().clone(); altered[:, 3] = 1e4
        duplicate, _ = layer(altered, comp, comp.ne(2), nodes, coords)
        torch.testing.assert_close(changed[:, :3], duplicate[:, :3])
        shifted, _ = layer(tokens, comp, comp.ne(2), nodes, coords + 100)
        torch.testing.assert_close(changed, shifted)
        with self.assertRaises(ValueError):
            layer(tokens, comp, comp.ne(2), nodes)

    def test_zero_bias_matches_legacy_and_empty_cells_stay_finite(self):
        legacy = CytoplasmicMeanCellAttention(8, 2, 0.)
        spatial = CytoplasmicMeanCellAttention(8, 2, 0., spatial_bias=True)
        spatial.load_state_dict(legacy.state_dict(), strict=False)
        x = torch.randn(1, 3, 2, 8)
        comp = torch.tensor([[[0, 1], [0, 1], [0, 0]]])
        nodes = torch.tensor([[0, 1, 2]])
        coords = torch.zeros(1, 3, 2)
        a, _ = legacy(x, comp, comp.ne(2), nodes)
        b, _ = spatial(x, comp, comp.ne(2), nodes, coords)
        torch.testing.assert_close(a, b)
        b.sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in spatial.parameters()))

    def test_training_and_eval_share_bins_and_precision(self):
        cfg = SimpleNamespace(num_bins=4, compartment_bin_edges=[.693359375, 1.0986328125])
        train = configured_edges(cfg)
        evaluation = evaluation_edges(cfg)
        np.testing.assert_array_equal(train, evaluation)
        values = np.log1p(np.array([1, 2, 3], dtype=np.float32))
        np.testing.assert_array_equal(positive_bins(values, evaluation), positive_bins(values.astype(np.float16), train))
        self.assertTrue((positive_bins(values, train) < cfg.num_bins).all())

    def test_full_model_cell_permutation_and_checkpoint_backward(self):
        torch.manual_seed(42)
        model = CompartmentSSGTModel(num_genes=10, num_value_bins=4, hidden_dim=8,
            num_layers=2, gene_heads=2, cell_heads=2, dropout=0., ffn_mult=2,
            gene_padding_idx=9, cell_attention_mode='cytoplasm_mean', cell_spatial_bias=True)
        for block in model.blocks:
            torch.nn.init.normal_(block.cell_attn.spatial_bias.angle_embedding.weight)
            torch.nn.init.normal_(block.cell_attn.spatial_bias.distance_embedding.weight)
        comp = torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]])
        batch = dict(gene_ids=torch.tensor([[[0, 1, 9], [2, 3, 9], [4, 5, 9]]]),
            input_bins=torch.tensor([[[1, 2, 5], [2, 3, 5], [1, 3, 5]]]),
            compartment_ids=comp, token_valid_mask=comp.ne(2),
            node_indices=torch.tensor([[0, 1, 2]]), cell_roles=torch.tensor([[1, 0, 0]]),
            coords=torch.tensor([[[0., 0.], [10., 0.], [0., 40.]]]))
        permutation = torch.tensor([2, 0, 1])
        a = model(**batch, return_embeddings=True)['cell_embeddings']
        b = model(**{k: v[:, permutation] for k, v in batch.items()}, return_embeddings=True)['cell_embeddings']
        torch.testing.assert_close(a[:, permutation], b)
        checkpointed = deepcopy(model)
        checkpointed.gradient_checkpointing = True
        reference = model(**batch)['mask_logits']
        actual = checkpointed(**batch)['mask_logits']
        torch.testing.assert_close(reference, actual)
        reference.square().sum().backward(); actual.square().sum().backward()
        for left, right in zip(model.parameters(), checkpointed.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad)


if __name__ == '__main__':
    unittest.main()
