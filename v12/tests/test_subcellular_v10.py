"""Run only on an authorized compute node: V10 information-boundary tests."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pretrain.models.subcellular_v10 import SubcellularV10, build_model
from spatial_pretrain.training.subcellular_v10 import losses


class MockBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.nheads, self.pad_id = 16, 4, 0
        self.backbone_kind = "scgpt"
        self.vocab = {"<pad>": 0, "<cls>": 1}
        self.config = {"pad_value": -2.}
        self.encoder = nn.Embedding(32, self.dim, padding_idx=0)
        self.value = nn.Linear(1, self.dim)
        self.dropout = nn.Dropout(.7)

    def forward(self, genes, bins, padding, compartment):
        token = self.encoder(genes) + self.value(bins[..., None]) + compartment
        token = self.dropout(token)
        pooled = (token * (~padding)[..., None]).sum(1) / (~padding).sum(1, keepdim=True)
        return token + pooled[:, None]


def example(batch_size=3, neighbors=2):
    genes = torch.tensor([[1, 2, 3, 0]]).repeat(batch_size, 1)
    genes[:, 1] += torch.arange(batch_size)
    bins = torch.tensor([[-2., 4., 9., -2.]]).repeat(batch_size, 1)
    bins[:, 1] += torch.arange(batch_size)
    valid = genes.ne(0)
    item = dict(nucleus_gene_ids=genes.clone(), nucleus_bins=bins.clone(), nucleus_valid=valid.clone(),
                teacher_gene_ids=genes.clone(), teacher_bins=bins + valid * 2, teacher_valid=valid.clone(),
                neighbor_gene_ids=genes[:, None].repeat(1, neighbors, 1),
                neighbor_bins=bins[:, None].repeat(1, neighbors, 1),
                neighbor_valid=valid[:, None].repeat(1, neighbors, 1),
                neighbor_present=torch.ones(batch_size, neighbors, dtype=torch.bool),
                relative_coords=torch.arange(batch_size * neighbors * 2).reshape(batch_size, neighbors, 2).float(),
                cell_identity=torch.stack((torch.zeros(batch_size, dtype=torch.long), torch.arange(batch_size)), -1))
    return item


def make_model(neighbors=2):
    torch.manual_seed(13)
    return SubcellularV10(".", neighbors=neighbors, backbone=MockBackbone(), encoder_chunk_size=2)


class V10Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_frozen_teacher_and_readouts(self):
        net = make_model().train()
        self.assertFalse(net.backbone.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in net.backbone.parameters()))
        output = net(example())
        self.assertFalse(output["teacher"].requires_grad)
        self.assertFalse(output["embeddings"]["base"].requires_grad)
        for name in ("base", "nucleus", "context", "joint"):
            self.assertEqual(output["embeddings"][name].shape, (3, 16))
        self.assertEqual(output["embeddings"]["base_augmented"].shape, (3, 48))
        torch.testing.assert_close(output["teacher"].norm(dim=-1), torch.ones(3))
        torch.testing.assert_close(output["embeddings"]["base"], output["teacher"], rtol=0, atol=0)
        torch.testing.assert_close(net(example())["teacher"], output["teacher"], rtol=0, atol=0)

    def test_context_never_reads_center_or_teacher(self):
        net = make_model().eval()
        first, second = example(), example()
        second["nucleus_bins"][:, 1:3] += 7
        second["teacher_bins"][:, 1:3] += 17
        a, b = net(first), net(second)
        torch.testing.assert_close(a["context_prediction"], b["context_prediction"], rtol=0, atol=0)
        self.assertFalse(torch.allclose(a["teacher"], b["teacher"]))
        self.assertFalse(torch.allclose(a["nucleus_prediction"], b["nucleus_prediction"]))

    def test_padding_and_absent_neighbours_do_not_change_predictions(self):
        net = make_model().eval()
        first = example()
        first["neighbor_present"][:, 1] = False
        second = deepcopy(first)
        for prefix in ("nucleus", "teacher", "neighbor"):
            second[f"{prefix}_gene_ids"][..., -1] = 9999
            second[f"{prefix}_bins"][..., -1] = float("nan")
        second["neighbor_gene_ids"][:, 1] = 9999
        second["neighbor_bins"][:, 1] = float("nan")
        second["relative_coords"][:, 1] = float("nan")
        a, b = net(first), net(second)
        for name in a["embeddings"]:
            torch.testing.assert_close(a["embeddings"][name], b["embeddings"][name], rtol=0, atol=0)

    def test_neighbor_set_order_is_irrelevant(self):
        net = make_model().eval()
        first = example()
        first["neighbor_bins"][:, 1, 1:3] += 13
        second = deepcopy(first)
        for name in ("neighbor_gene_ids", "neighbor_bins", "neighbor_valid", "neighbor_present",
                     "relative_coords"):
            second[name] = second[name].flip(1)
        a, b = net(first), net(second)
        torch.testing.assert_close(a["context_prediction"], b["context_prediction"], atol=1e-6, rtol=1e-6)

    def test_each_loss_has_finite_predictor_gradients_and_no_teacher_gradients(self):
        for kind in ("cosine", "contrastive", "relational"):
            with self.subTest(kind=kind):
                net, item = make_model(), example()
                loss, metrics = losses(net(item), item, {"loss": {"type": kind}})
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(metrics["degenerate_cosine_fallbacks"], 0)
                loss.backward()
                grads = [p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
                self.assertTrue(grads)
                self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in grads))
                self.assertGreater(sum(float(grad.abs().sum()) for grad in grads), 0.)
                self.assertTrue(all(p.grad is None for p in net.backbone.parameters()))
                self.assertGreater(float(net.nucleus_predictor.network[-1].weight.grad.abs().sum()), 0.)
                self.assertGreater(float(net.context_predictor.output.weight.grad.abs().sum()), 0.)

    def test_no_neighbors_and_batch_one_are_finite_and_trainable(self):
        for count in (0, 2):
            for kind in ("cosine", "contrastive", "relational"):
                with self.subTest(count=count, kind=kind):
                    item = example(batch_size=1, neighbors=count)
                    item["neighbor_present"][:] = False
                    net = make_model(neighbors=count)
                    output = net(item)
                    self.assertEqual(float(output["context_prediction"].abs().sum()), 0.)
                    torch.testing.assert_close(output["embeddings"]["joint"], output["nucleus_prediction"])
                    loss, metrics = losses(output, item, {"loss": {"type": kind}})
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    self.assertGreater(float(net.nucleus_predictor.network[-1].weight.grad.abs().sum()), 0.)
                    if kind != "cosine":
                        self.assertGreater(metrics["degenerate_cosine_fallbacks"], 0)

    def test_duplicate_cells_are_positive_and_distinct_slices_are_distinct(self):
        item = example()
        item["cell_identity"][:] = torch.tensor([4, 12])
        output = make_model()(item)
        for kind in ("contrastive", "relational"):
            loss, metrics = losses(output, item, {"loss": {"type": kind}})
            self.assertEqual(metrics["unique_cells"], 1)
            self.assertGreater(metrics["degenerate_cosine_fallbacks"], 0)
            self.assertTrue(torch.isfinite(loss))
        item["cell_identity"][:, 0] = torch.arange(3)
        _, metrics = losses(output, item, {"loss": {"type": "contrastive"}})
        self.assertEqual(metrics["unique_cells"], 3)
        self.assertEqual(metrics["degenerate_cosine_fallbacks"], 0)

    def test_duplicate_candidate_is_never_a_false_negative(self):
        from spatial_pretrain.training.subcellular_v10 import _nce
        query = torch.tensor([[1., 0.]])
        bank = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        ids = torch.tensor([[1, 0], [1, 0], [1, 1]])
        loss, fallback = _nce(query, bank, ids[:1], ids, .1)
        expected = torch.logsumexp(torch.tensor([10., 10., 0.]), 0) - torch.logsumexp(torch.tensor([10., 10.]), 0)
        torch.testing.assert_close(loss, expected)
        self.assertEqual(fallback, 0)

    def test_empty_nuclei_keep_context_learning_and_joint_uses_context(self):
        for kind in ("cosine", "contrastive", "relational"):
            with self.subTest(kind=kind):
                net, item = make_model(), example()
                item["nucleus_present"] = torch.zeros(3, dtype=torch.bool)
                output = net(item)
                self.assertFalse(bool(output["nucleus_available"].any()))
                self.assertEqual(float(output["nucleus_prediction"].abs().sum()), 0.)
                torch.testing.assert_close(output["embeddings"]["joint"], output["context_prediction"])
                loss, metrics = losses(output, item, {"loss": {"type": kind}})
                self.assertTrue(metrics["has_learning_signal"])
                self.assertEqual(metrics["nucleus_cells"], 0)
                self.assertEqual(metrics["paired_cells"], 0)
                self.assertEqual(metrics["valid_loss_terms"], 1)
                self.assertNotIn("nucleus_teacher", metrics)
                self.assertNotIn("view_consistency", metrics)
                self.assertAlmostEqual(float(loss.detach()), metrics["context_teacher"], places=6)
                loss.backward()
                self.assertEqual(float(net.nucleus_predictor.network[-1].weight.grad.abs().sum()), 0.)
                self.assertGreater(float(net.context_predictor.output.weight.grad.abs().sum()), 0.)

    def test_both_views_missing_return_differentiable_zero_without_learning_signal(self):
        for kind in ("cosine", "contrastive", "relational"):
            with self.subTest(kind=kind):
                net, item = make_model(), example()
                item["nucleus_present"] = torch.zeros(3, dtype=torch.bool)
                item["neighbor_present"][:] = False
                output = net(item)
                self.assertEqual(float(output["embeddings"]["joint"].abs().sum()), 0.)
                torch.testing.assert_close(output["embeddings"]["base"].norm(dim=-1), torch.ones(3))
                loss, metrics = losses(output, item, {"loss": {"type": kind}})
                self.assertTrue(loss.requires_grad)
                self.assertEqual(float(loss.detach()), 0.)
                self.assertFalse(metrics["has_learning_signal"])
                self.assertEqual(metrics["valid_loss_terms"], 0)
                self.assertEqual(metrics["degenerate_cosine_fallbacks"], 0)
                loss.backward()
                for parameter in net.parameters():
                    if parameter.requires_grad:
                        self.assertIsNotNone(parameter.grad)
                        self.assertEqual(float(parameter.grad.abs().sum()), 0.)

    def test_partial_empty_nucleus_has_no_anchor_or_pair_gradient(self):
        for kind in ("cosine", "contrastive", "relational"):
            with self.subTest(kind=kind):
                item, net = example(), make_model()
                item["nucleus_present"] = torch.tensor([False, True, True])
                output = net(item)
                output["nucleus_prediction"].retain_grad()
                loss, metrics = losses(output, item, {"loss": {"type": kind}})
                self.assertEqual(metrics["nucleus_cells"], 2)
                self.assertEqual(metrics["paired_cells"], 2)
                self.assertEqual(metrics["context_cells"], 3)
                loss.backward()
                self.assertEqual(float(output["nucleus_prediction"].grad[0].abs().sum()), 0.)
                self.assertGreater(float(output["nucleus_prediction"].grad[1:].abs().sum()), 0.)

    def test_cached_raw_cls_matches_uncached_without_calling_backbone(self):
        net, item = make_model().eval(), example()
        item["neighbor_present"][0, 1] = False
        item["nucleus_present"] = torch.tensor([True, False, True])
        features = net.encode_views(item)
        self.assertEqual(set(features), {"teacher_features", "nucleus_features", "neighbor_features"})
        self.assertTrue(all(not value.requires_grad for value in features.values()))
        self.assertEqual(float(features["neighbor_features"][0, 1].abs().sum()), 0.)
        expected = net(item)
        cached = dict(features, neighbor_present=item["neighbor_present"],
                      relative_coords=item["relative_coords"], nucleus_present=item["nucleus_present"])
        with patch.object(net.backbone, "forward", side_effect=AssertionError("Backbone must not run")):
            actual = net(cached)
        for key in expected["embeddings"]:
            torch.testing.assert_close(actual["embeddings"][key], expected["embeddings"][key], rtol=0, atol=0)
        torch.testing.assert_close(actual["nucleus_available"], expected["nucleus_available"])
        torch.testing.assert_close(actual["context_available"], expected["context_available"])

    def test_cached_features_require_complete_finite_matching_shapes_and_are_detached(self):
        net, item = make_model(), example()
        features = net.encode_views(item)
        with self.assertRaisesRegex(ValueError, "all three"):
            net(dict(item, teacher_features=features["teacher_features"]))
        cached = dict(features, neighbor_present=item["neighbor_present"],
                      relative_coords=item["relative_coords"])
        with self.assertRaisesRegex(ValueError, "Raw CLS features must match"):
            net(dict(cached, nucleus_features=features["nucleus_features"][:1]))
        invalid = features["teacher_features"].clone()
        invalid[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite floating-point"):
            net(dict(cached, teacher_features=invalid))
        for name in features:
            cached[name] = cached[name].detach().requires_grad_(True)
        output = net(cached)
        loss, _ = losses(output, item, {"loss": {"type": "cosine"}})
        loss.backward()
        self.assertTrue(all(cached[name].grad is None for name in features))

    def test_model_builder_rejects_spatial_backbone(self):
        with self.assertRaises(ValueError):
            build_model({"model": {"backbone_kind": "scgpt_spatial"}}, MockBackbone())


if __name__ == "__main__":
    unittest.main()
