"""V11 information-boundary and reduction tests; run only on a compute node."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pretrain.models.subcellular_v11 import SubcellularV11, LRNeighborEncoder, build_model
from spatial_pretrain.training.subcellular_v11 import (
    losses, loss_sums_and_counts, loss_counts_from_batch, reduce_loss_sums,
)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.nheads, self.pad_id = 16, 4, 0
        self.backbone_kind = "scgpt"
        self.vocab = {"<pad>": 0, "<cls>": 1, **{f"g{i}": i for i in range(2, 32)}}
        self.config = {"pad_value": -2.}
        self.encoder = nn.Embedding(32, 16, padding_idx=0)
        self.value_encoder = nn.Linear(1, 16)
        layer = nn.TransformerEncoderLayer(16, 4, 32, dropout=0., batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)

    def forward(self, genes, bins, padding, compartment):
        features = self.encoder(genes) + self.value_encoder(bins[..., None]) + compartment
        return self.transformer_encoder(features, src_key_padding_mask=padding)


def example(batch_size=3, neighbors=3):
    genes = torch.tensor([[1, 2, 3, 0]]).repeat(batch_size, 1)
    genes[:, 1] += torch.arange(batch_size) % 8
    bins = torch.tensor([[-2., 4., 9., -2.]]).repeat(batch_size, 1)
    bins[:, 1] += torch.arange(batch_size)
    near_genes = genes[:, None].repeat(1, neighbors, 1)
    near_bins = bins[:, None].repeat(1, neighbors, 1)
    near_bins[:, :, 1] += torch.arange(neighbors)[None] * 7
    coords = torch.zeros(batch_size, neighbors, 2)
    coords[:, :, 0] = torch.arange(neighbors) * 10 + 10
    centers = torch.stack((torch.zeros(batch_size, dtype=torch.long), torch.arange(batch_size)), -1)
    near_ids = torch.zeros(batch_size, neighbors, 2, dtype=torch.long)
    near_ids[:, :, 1] = torch.arange(neighbors)[None] + 100
    lr = torch.arange(neighbors * neighbors).reshape(1, neighbors, neighbors).float()
    lr = lr.repeat(batch_size, 1, 1) / max(neighbors * neighbors, 1)
    return dict(nucleus_gene_ids=genes, nucleus_bins=bins, nucleus_valid=genes.ne(0),
                nucleus_present=torch.ones(batch_size, dtype=torch.bool),
                neighbor_gene_ids=near_genes, neighbor_bins=near_bins, neighbor_valid=near_genes.ne(0),
                neighbor_present=torch.ones(batch_size, neighbors, dtype=torch.bool),
                relative_coords=coords, neighbor_lr_bias=lr, neighbor_lr_coverage=lr * .2,
                query_gene_ids=torch.tensor([[2, 3, 4, 0]]).repeat(batch_size, 1),
                query_bins=torch.tensor([[0, 12, 50, -99]]).repeat(batch_size, 1),
                query_valid_mask=torch.tensor([[True, True, True, False]]).repeat(batch_size, 1),
                cell_identity=centers, neighbor_cell_identity=near_ids)


def make_model(neighbors=3, **kwargs):
    torch.manual_seed(17)
    return SubcellularV11(".", neighbors=neighbors, backbone=DummyBackbone(),
                         encoder_chunk_size=2, backbone_trainable_layers=2, **kwargs)


def slice_batch(batch, sl):
    return {name: value[sl] for name, value in batch.items()}


class V11Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_readouts_shared_decoder_and_no_target_input(self):
        net, item = make_model().eval(), example()
        with patch.object(net.joint_decoder, "forward", wraps=net.joint_decoder.forward) as decoder:
            first = net(item)
            self.assertEqual(decoder.call_count, 3)
        self.assertEqual(set(first["embeddings"]), {"nucleus", "spatial", "fused", "concat"})
        for name in ("nucleus", "spatial", "fused"):
            self.assertEqual(first["embeddings"][name].shape, (3, 16))
            self.assertEqual(first["logits"][name].shape, (3, 4, 51))
        torch.testing.assert_close(first["embeddings"]["concat"], torch.cat((
            F.normalize(first["embeddings"]["nucleus"], dim=-1),
            F.normalize(first["embeddings"]["spatial"], dim=-1)), -1))
        changed = deepcopy(item)
        changed["query_bins"][:] = 37
        changed["query_gene_ids"][:, :3] += 5
        second = net(changed)
        for name in first["embeddings"]:
            torch.testing.assert_close(first["embeddings"][name], second["embeddings"][name], atol=0, rtol=0)
        changed["query_gene_ids"] = item["query_gene_ids"]
        third = net(changed)
        for name in first["logits"]:
            torch.testing.assert_close(first["logits"][name], third["logits"][name], atol=0, rtol=0)
        for key in ("query_bins", "query_gene_ids", "query_valid_mask"):
            changed.pop(key)
        with patch.object(net.joint_decoder, "forward", side_effect=AssertionError("decoder called")):
            net(changed, return_embeddings=True)

    def test_independent_input_paths_before_fusion(self):
        net, item = make_model().eval(), example()
        first = net(item)["embeddings"]
        changed = deepcopy(item)
        changed["nucleus_bins"][:, 1:3] += 30
        second = net(changed)["embeddings"]
        torch.testing.assert_close(first["spatial"], second["spatial"], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first["nucleus"], second["nucleus"]))
        self.assertFalse(torch.allclose(first["fused"], second["fused"]))
        changed = deepcopy(item)
        changed["neighbor_bins"][:, :, 1:3] += 30
        third = net(changed)["embeddings"]
        torch.testing.assert_close(first["nucleus"], third["nucleus"], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first["spatial"], third["spatial"]))

    def test_gene_order_padding_and_missing_slots_are_ignored(self):
        net, item = make_model().eval(), example()
        item["neighbor_present"][:, -1] = False
        altered = deepcopy(item)
        for prefix in ("nucleus", "neighbor"):
            for suffix in ("gene_ids", "bins", "valid"):
                altered[f"{prefix}_{suffix}"] = altered[f"{prefix}_{suffix}"][..., [0, 2, 1, 3]]
            altered[f"{prefix}_gene_ids"][..., -1] = 99999
            altered[f"{prefix}_bins"][..., -1] = float("nan")
        altered["neighbor_gene_ids"][:, -1] = 99999
        altered["neighbor_bins"][:, -1] = float("nan")
        altered["relative_coords"][:, -1] = float("nan")
        altered["neighbor_lr_bias"][:, -1, :] = float("nan")
        altered["neighbor_lr_bias"][:, :, -1] = float("nan")
        a, b = net(item), net(altered)
        for name in a["embeddings"]:
            torch.testing.assert_close(a["embeddings"][name], b["embeddings"][name], atol=1e-6, rtol=1e-5)

    def test_graph_radius_knn_self_exclusion_and_lr_direction(self):
        torch.manual_seed(3)
        module = LRNeighborEncoder(16, 4, spatial_radius=15., graph_neighbors=2)
        coords = torch.tensor([[[0., 0.], [5., 0.], [0., 5.], [100., 0.]]])
        present = torch.ones(1, 4, dtype=torch.bool)
        graph, _ = module.graph(present, coords)
        self.assertFalse(bool(graph.diagonal(dim1=-2, dim2=-1).any()))
        self.assertFalse(bool(graph[:, 3, :].any() | graph[:, :, 3].any()))
        self.assertTrue(bool(graph.sum(-1).le(2).all()))
        with torch.no_grad():
            for layer in (module.query, module.key, module.edge_position[-1]):
                layer.weight.zero_()
                layer.bias.zero_()
        features = torch.randn(1, 4, 16)
        prior = torch.zeros(1, 4, 4)
        _, baseline, _, _ = module(features, present, coords, prior)
        prior[0, 0, 1] = 1.  # Sender 1 -> receiver 0 only.
        _, directed, _, _ = module(features, present, coords, prior)
        self.assertTrue(bool((directed[0, :, 0, 1] > baseline[0, :, 0, 1]).all()))
        torch.testing.assert_close(directed[:, :, 1:, :], baseline[:, :, 1:, :], atol=0, rtol=0)
        tight = LRNeighborEncoder(16, 4, spatial_radius=15., graph_neighbors=1)
        tight_graph, _ = tight.graph(present, coords)
        self.assertTrue(bool(tight_graph.sum(-1).le(1).all()))

    def test_center_and_duplicate_id_are_excluded(self):
        net, item = make_model().eval(), example()
        item["neighbor_cell_identity"][:, 0] = item["cell_identity"]
        item["neighbor_cell_identity"][:, 2] = item["neighbor_cell_identity"][:, 1]
        output = net(item)
        self.assertEqual(float(output["readout_attention"][:, :, 0].abs().sum()), 0.)
        self.assertEqual(float(output["readout_attention"][:, :, 2].abs().sum()), 0.)
        changed = deepcopy(item)
        changed["neighbor_bins"][:, [0, 2], :] += 100
        torch.testing.assert_close(output["embeddings"]["spatial"], net(changed)["embeddings"]["spatial"], atol=0, rtol=0)

    def test_neighbor_order_equivariance_and_coverage_diagnostic_only(self):
        net, item = make_model().eval(), example()
        altered = deepcopy(item)
        names = ("neighbor_gene_ids", "neighbor_bins", "neighbor_valid", "neighbor_present",
                 "relative_coords", "neighbor_cell_identity")
        for name in names:
            altered[name] = altered[name].flip(1)
        for name in ("neighbor_lr_bias", "neighbor_lr_coverage"):
            altered[name] = altered[name].flip(1).flip(2)
        altered["neighbor_lr_coverage"][:] = 1.
        a, b = net(item), net(altered)
        for name in a["embeddings"]:
            torch.testing.assert_close(a["embeddings"][name], b["embeddings"][name], atol=1e-6, rtol=1e-5)

    def test_fused_supervision_updates_both_paths_lr_and_shared_decoder(self):
        net, item = make_model(), example()
        result = net(item)
        loss, _ = losses(result, item, {"loss": {"modes": ["fused"]}})
        loss.backward()
        for parameter in (net.fusion[0].weight, net.joint_decoder.output[1].weight,
                          net.spatial_encoder.value.weight, net.spatial_encoder.lr_scale_logit,
                          net.backbone.transformer_encoder.layers[-1].linear1.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(net.backbone.encoder.weight.grad)
        self.assertIsNone(net.backbone.transformer_encoder.layers[0].linear1.weight.grad)
        # Each branch independently reaches the same shared decoder parameters.
        for mode in ("nucleus", "spatial"):
            net.zero_grad(set_to_none=True)
            losses(net(item), item, {"loss": {"modes": [mode]}})[0].backward()
            self.assertGreater(float(net.joint_decoder.output[-1].weight.grad.abs().sum()), 0.)

    def test_suffix_train_modes_and_chunk_checkpoint_gradients(self):
        net = make_model(checkpointing=True).train()
        self.assertFalse(net.backbone.encoder.training)
        layers = net.backbone.transformer_encoder.layers
        self.assertFalse(layers[0].training)
        self.assertTrue(layers[1].training and layers[2].training)
        item = example()
        losses(net(item), item, {})[0].backward()
        self.assertGreater(float(layers[1].linear1.weight.grad.abs().sum()), 0.)
        net.eval()
        self.assertTrue(all(not layer.training for layer in layers))
        net.set_backbone_trainable_layers(0)
        net.train()
        self.assertTrue(all(not p.requires_grad for p in net.backbone.parameters()))
        self.assertTrue(all(not layer.training for layer in layers))
        net.set_backbone_trainable_layers(3)
        self.assertTrue(all(layer.training for layer in layers))
        with self.assertRaises(ValueError):
            net.set_backbone_trainable_layers(-1)

    def test_missing_view_fallback_and_preforward_counts_match(self):
        net, item = make_model(max_neighbor_distance=25.).eval(), example()
        item["nucleus_present"][0] = False
        item["neighbor_present"][1] = False
        item["nucleus_valid"][2, 1:] = False
        item["query_valid_mask"][2] = False
        result = net(item)
        torch.testing.assert_close(result["embeddings"]["fused"][0], result["embeddings"]["spatial"][0])
        torch.testing.assert_close(result["embeddings"]["fused"][1], result["embeddings"]["nucleus"][1])
        pre = loss_counts_from_batch(item, {}, mode_available=net.mode_availability(item))
        _, post = loss_sums_and_counts(result, item, {})
        for mode in pre:
            torch.testing.assert_close(pre[mode], post[mode])
        self.assertFalse(bool(result["readout_attention"][:, :, 2].any()))

    def test_empty_inputs_empty_queries_and_zero_graph_backward(self):
        for count in (0, 3):
            for empty_queries in (False, True):
                with self.subTest(neighbors=count, empty_queries=empty_queries):
                    net, item = make_model(neighbors=count), example(neighbors=count)
                    item["nucleus_present"][:] = False
                    item["neighbor_present"][:] = False
                    if empty_queries:
                        for name in ("query_gene_ids", "query_bins", "query_valid_mask"):
                            item[name] = item[name][:, :0]
                    result = net(item)
                    for value in result["embeddings"].values():
                        self.assertTrue(bool(torch.isfinite(value).all()))
                        self.assertEqual(float(value.abs().sum()), 0.)
                    loss, metrics = losses(result, item, {})
                    self.assertEqual(float(loss), 0.)
                    self.assertFalse(metrics["has_learning_signal"])
                    loss.backward()
                    for parameter in net.parameters():
                        if parameter.requires_grad:
                            self.assertIsNotNone(parameter.grad)
                            self.assertEqual(float(parameter.grad.abs().sum()), 0.)

    def test_gene_cell_mode_reduction_and_padding_targets(self):
        torch.manual_seed(5)
        item = dict(query_bins=torch.tensor([[0, 1, -99], [2, 1, 0]]),
                    query_valid_mask=torch.tensor([[True, True, False], [True, False, False]]))
        logits = {name: torch.randn(2, 3, 3, requires_grad=True) for name in ("nucleus", "spatial", "fused")}
        available = dict(nucleus=torch.tensor([True, True]), spatial=torch.tensor([True, False]),
                         fused=torch.tensor([True, True]))
        output = dict(logits=logits, mode_available=available)
        actual, _ = losses(output, item, {})
        expected = []
        for mode in logits:
            cells = []
            for row in range(2):
                if available[mode][row]:
                    mask = item["query_valid_mask"][row]
                    cells.append(F.cross_entropy(logits[mode][row, mask], item["query_bins"][row, mask]))
            expected.append(torch.stack(cells).mean())
        torch.testing.assert_close(actual, torch.stack(expected).mean())

    def test_global_counts_microbatch_gradients_equal_full_batch(self):
        torch.manual_seed(9)
        item = example(batch_size=5)
        logits = {mode: torch.randn(5, 4, 51, requires_grad=True) for mode in ("nucleus", "spatial", "fused")}
        available = dict(nucleus=torch.tensor([False, False, True, True, True]),
                         spatial=torch.tensor([True, False, False, True, True]),
                         fused=torch.tensor([True, False, True, True, True]))
        output = dict(logits=logits, mode_available=available)
        full_sums, global_counts = loss_sums_and_counts(output, item, {})
        full = reduce_loss_sums(full_sums, global_counts)
        full_grad = torch.autograd.grad(full, tuple(logits.values()), retain_graph=True)
        micros = []
        # Emulate unequal rank/microbatch shapes; DDP then averages rank gradients.
        for sl in (slice(0, 1), slice(1, 3), slice(3, 5)):
            part = dict(logits={k: v[sl] for k, v in logits.items()},
                        mode_available={k: v[sl] for k, v in available.items()})
            sums, counts = loss_sums_and_counts(part, slice_batch(item, sl), {})
            micros.append(reduce_loss_sums(sums, counts, global_counts=global_counts, world_size=2))
        distributed = sum(micros) / 2
        torch.testing.assert_close(full, distributed)
        micro_grad = torch.autograd.grad(distributed, tuple(logits.values()))
        for a, b in zip(full_grad, micro_grad):
            torch.testing.assert_close(a, b)

    def test_no_teacher_or_feature_cache_and_factory(self):
        net, item = make_model(), example()
        item["nucleus_features"] = torch.randn(3, 16)
        with self.assertRaisesRegex(ValueError, "feature caches"):
            net(item)
        with self.assertRaisesRegex(ValueError, "ordinary"):
            SubcellularV11(".", backbone=DummyBackbone(), backbone_kind="scgpt_spatial")
        model = build_model({"assets_dir": ".", "model": {"backbone_trainable_layers": 1}}, backbone=DummyBackbone())
        self.assertEqual(model.backbone_trainable_layers, 1)
        self.assertNotIn("teacher", model.arch_config)
        with self.assertRaises(ValueError):
            losses({}, {}, {"loss": {"type": "cosine"}})

    def test_masked_bin_cross_entropy_alias_matches(self):
        net, item = make_model().eval(), example()
        result = net(item)
        expected, _ = losses(result, item, {"loss": {"type": "bin_ce"}})
        actual, _ = losses(result, item, {"loss": {"type": "masked_bin_cross_entropy"}})
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
