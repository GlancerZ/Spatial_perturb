"""V9: one-hop two-view model, loss and center LR evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from spatial_pretrain.data.subcellular_v9 import center_lr_v9
from spatial_pretrain.models.subcellular_v9 import SubcellularV9
from spatial_pretrain.training.subcellular_v9 import losses, symmetric_kl
from test_subcellular_v6_model import MockBackbone, example

CFG = {"loss": {"type": "two_view_masked_bin_ce", "consistency_weight": .1}}


def batch():
    item = example()
    item["lr_center"] = torch.tensor([[0., .9, .15]])
    return item


def model(lr_enabled=True):
    torch.manual_seed(3)
    return SubcellularV9(".", neighbors=2, checkpointing=False, encoder_chunk_size=2,
                         backbone=MockBackbone(), lr_enabled=lr_enabled)


class V9ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def test_shapes_loss_and_gradients(self):
        net = model()
        net.set_backbone_trainable_layers(1)
        net.train()
        output = net(batch())
        self.assertEqual(tuple(output["self_logits"].shape), (1, 5, 51))
        self.assertEqual(tuple(output["embeddings"]["joint"].shape), (1, 32))
        total, parts = losses(output, batch(), CFG, 1)
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(parts), {"total", "ce_self", "ce_ctx", "consistency"})
        total.backward()
        for name in ("spatial.query.weight", "spatial.value.weight", "joint_decoder.output.1.weight",
                     "compartment_embedding.weight", "spatial.lr_scale_logit"):
            grad = dict(net.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(float(grad.abs().sum()), 0., name)
        frozen = net.backbone.transformer_encoder.layers[0].linear1.weight
        self.assertIsNone(frozen.grad)

    def test_context_is_zero_without_neighbours(self):
        net = model().eval()
        item = batch()
        item["node_indices"] = torch.tensor([[10, -1, -1]])
        with torch.no_grad():
            ctx = net(item, return_embeddings=True)["embeddings"]["ctx"]
        self.assertEqual(float(ctx.abs().sum()), 0.)

    def test_self_view_ignores_neighbour_tokens(self):
        net = model().eval()
        a, b = batch(), batch()
        b["input_bins"][:, 1:, 1:5] = torch.tensor([9., 9., 9., 9.])
        with torch.no_grad():
            ea = net(a, return_embeddings=True)["embeddings"]
            eb = net(b, return_embeddings=True)["embeddings"]
        torch.testing.assert_close(ea["self"], eb["self"])
        self.assertFalse(torch.allclose(ea["ctx"], eb["ctx"]))

    def test_lr_disabled_runs_without_lr_input(self):
        net = model(lr_enabled=False).eval()
        item = batch()
        del item["lr_center"]
        with torch.no_grad():
            net(item)


class V9LossAndLRTests(unittest.TestCase):
    def test_symmetric_kl_zero_for_identical_views(self):
        logits = torch.randn(2, 4, 51)
        valid = torch.ones(2, 4, dtype=torch.bool)
        self.assertAlmostEqual(float(symmetric_kl(logits, logits.clone(), valid)), 0., places=6)
        self.assertGreater(float(symmetric_kl(logits, torch.randn(2, 4, 51), valid)), 0.)

    def test_center_lr_uses_only_known_genes_and_is_bounded(self):
        genes = np.array([10, 11, 12, 13])
        pairs = np.array([[10, 11], [12, 13]])  # ligand, receptor
        weights = np.ones(2)
        nodes = np.array([0, 1, 2])
        center = np.array([0, 5, 0, 5])
        cyto = np.array([[0, 0, 0, 0], [7, 0, 0, 0], [0, 0, 3, 0]])
        known = np.ones((3, 4), dtype=bool)
        lr = center_lr_v9(center, np.ones(4, bool), cyto, known, genes, nodes, pairs, weights)
        self.assertEqual(lr[0], 0.)
        self.assertTrue(((lr >= 0) & (lr <= 1)).all())
        self.assertGreater(lr[1], 0.)
        self.assertGreater(lr[2], 0.)
        hidden = known.copy()
        hidden[1, 0] = False  # neighbour 1's ligand masked -> no evidence
        lr_hidden = center_lr_v9(center, np.ones(4, bool), cyto, hidden, genes, nodes, pairs, weights)
        self.assertEqual(lr_hidden[1], 0.)


if __name__ == "__main__":
    unittest.main()
