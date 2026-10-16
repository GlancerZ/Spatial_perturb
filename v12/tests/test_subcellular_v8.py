"""V8 pilot: residual soft cap, nonzero-weighted CE and tie-policy dispatch."""
from pathlib import Path
import sys
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from spatial_pretrain.data.subcellular_v3 import encode_context_v3
from spatial_pretrain.data.subcellular_v8 import compartment_encoder, tie_policy
from spatial_pretrain.data.subcellular_v7 import encode_compartment_context_v7
from spatial_pretrain.models.subcellular_v7 import SubcellularV7
from spatial_pretrain.models.subcellular_v8 import SubcellularV8
from spatial_pretrain.training.subcellular_v4 import masked_bin_cross_entropy
from spatial_pretrain.training.subcellular_v8 import losses, weighted_masked_bin_cross_entropy
from test_subcellular_v6_model import MockBackbone, example


def model_for_v8(route="A", cap=.5):
    torch.manual_seed(7)
    return SubcellularV8(".", neighbors=2, checkpointing=False, encoder_chunk_size=3,
                         backbone=MockBackbone(), slots_per_compartment=2, slot_dim=8,
                         route=route, residual_cap=cap)


class V8CapTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)

    def test_step_zero_equals_base_and_projection_gets_gradient(self):
        for route in ("A", "B"):
            model = model_for_v8(route)
            output = model(example())
            embeddings = model(example(), return_embeddings=True)["embeddings"]
            torch.testing.assert_close(embeddings["joint"], embeddings["base"])
            batch = example()
            valid = batch["query_valid_mask"]
            F.cross_entropy(output["joint_logits"][valid], batch["target_bins"][valid]).backward()
            name = "residual_projection" if route == "A" else "context_projection"
            weight = getattr(model, name).weight.grad
            self.assertIsNotNone(weight)
            self.assertGreater(float(weight.abs().sum()), 0.)

    def test_cap_bounds_norm_and_is_identity_for_small_residuals(self):
        model = model_for_v8("A", cap=.5)
        base = F.normalize(torch.randn(6, 16), dim=-1)
        large = torch.randn(6, 16) * 10
        self.assertTrue(bool((model.cap(large, base).norm(dim=-1) < .5 + 1e-6).all()))
        small = torch.randn(6, 16) * 1e-4
        torch.testing.assert_close(model.cap(small, base), small, rtol=1e-4, atol=1e-8)
        self.assertIs(model_for_v8("A", cap=None).cap(large, base), large)

    def test_uncapped_v8_matches_v7_forward(self):
        torch.manual_seed(7)
        v7 = SubcellularV7(".", neighbors=2, checkpointing=False, encoder_chunk_size=3,
                           backbone=MockBackbone(), slots_per_compartment=2, slot_dim=8, route="B")
        v8 = model_for_v8("B", cap=None)
        v8.load_state_dict(v7.state_dict())
        with torch.no_grad():
            for module in (v7, v8):
                module.context_projection.weight.normal_(generator=torch.Generator().manual_seed(3))
                module.self_projection.weight.normal_(generator=torch.Generator().manual_seed(4))
        torch.testing.assert_close(v8(example())["joint_logits"], v7(example())["joint_logits"])

    def test_cap_limits_trained_residual(self):
        model = model_for_v8("A", cap=.25)
        with torch.no_grad():
            model.residual_projection.weight.normal_(std=5.)
        out = model(example(), return_embeddings=True)
        residual = out["embeddings"]["joint"] - out["embeddings"]["base"]
        ratio = residual.norm(dim=-1) / out["embeddings"]["base"].norm(dim=-1)
        self.assertTrue(bool((ratio <= .25 + 1e-5).all()))


class V8LossTests(unittest.TestCase):
    def batch(self):
        torch.manual_seed(1)
        target = torch.randint(0, 51, (4, 6))
        target[:, :3] = 0
        return dict(query_valid_mask=torch.ones(4, 6, dtype=torch.bool),
                    target_bins=target,
                    query_compartment_ids=torch.tensor([0, 0, 0, 1, 1, 1]).expand(4, 6)), torch.randn(4, 6, 51)

    def test_weight_one_equals_v4_objective(self):
        batch, logits = self.batch()
        torch.testing.assert_close(weighted_masked_bin_cross_entropy(logits, batch, 1.),
                                   masked_bin_cross_entropy(logits, batch))
        total, parts = losses({"joint_logits": logits}, batch,
                              {"loss": {"type": "masked_bin_cross_entropy"}}, 1)
        self.assertEqual(parts["total"], parts["masked_ce"])

    def test_weight_emphasizes_nonzero_and_reports_plain_ce(self):
        batch, logits = self.batch()
        cfg = {"loss": {"type": "masked_bin_cross_entropy", "nonzero_weight": 4.}}
        total, parts = losses({"joint_logits": logits}, batch, cfg, 1)
        self.assertAlmostEqual(parts["masked_ce"], float(masked_bin_cross_entropy(logits, batch)), places=5)
        self.assertNotAlmostEqual(parts["total"], parts["masked_ce"], places=3)
        with self.assertRaises(ValueError):
            losses({"joint_logits": logits}, batch,
                   {"loss": {"type": "masked_bin_cross_entropy", "nonzero_weight": .5}}, 1)


class V8TiePolicyTests(unittest.TestCase):
    def test_dispatch(self):
        self.assertIs(compartment_encoder("left"), encode_compartment_context_v7)
        self.assertIs(compartment_encoder("random"), encode_context_v3)
        self.assertEqual(tie_policy({"binning": {"tie_policy": "random"}}), "random")
        with self.assertRaises(ValueError):
            tie_policy({"binning": {"tie_policy": "right"}})


if __name__ == "__main__":
    unittest.main()
