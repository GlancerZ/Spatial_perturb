"""V12 information boundaries and joint-count likelihood; compute nodes only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pretrain.models.subcellular_v12 import (
    SubcellularV12, LRNeighborEncoderV12, MODES, READOUTS, build_model,
)
from spatial_pretrain.training.subcellular_v12 import (
    losses, loss_sums_and_counts, loss_counts_from_batch, reduce_loss_sums,
    joint_count_log_prob, joint_count_nll,
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

    def forward(self, genes, bins, padding, addition):
        features = self.encoder(genes) + self.value_encoder(bins[..., None]) + addition
        return self.transformer_encoder(features, src_key_padding_mask=padding)


def example(batch_size=3, neighbors=3):
    genes = torch.tensor([[1, 2, 3, 0]]).repeat(batch_size, 1)
    genes[:, 1] += torch.arange(batch_size) % 8
    bins = torch.tensor([[-2., 4., 9., -2.]]).repeat(batch_size, 1)
    bins[:, 1] += torch.arange(batch_size)
    counts = torch.tensor([[0., 4., 11., 0.]]).repeat(batch_size, 1).log1p()
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
    query_n = torch.tensor([[0, 1, 7, 0]]).repeat(batch_size, 1)
    query_c = torch.tensor([[2, 3, 0, 0]]).repeat(batch_size, 1)
    return dict(nucleus_gene_ids=genes, nucleus_bins=bins, nucleus_valid=genes.ne(0),
                nucleus_log_counts=counts, nucleus_present=torch.ones(batch_size, dtype=torch.bool),
                cytoplasm_gene_ids=genes.clone(), cytoplasm_bins=bins + torch.tensor([0., 3., 4., 0.]),
                cytoplasm_valid=genes.ne(0), cytoplasm_log_counts=counts + torch.tensor([0., .5, .8, 0.]),
                cytoplasm_present=torch.ones(batch_size, dtype=torch.bool),
                neighbor_gene_ids=near_genes, neighbor_bins=near_bins, neighbor_valid=near_genes.ne(0),
                neighbor_present=torch.ones(batch_size, neighbors, dtype=torch.bool),
                relative_coords=coords, neighbor_lr_bias=lr, neighbor_lr_coverage=lr * .2,
                center_lr_bias=torch.arange(neighbors).float()[None].repeat(batch_size, 1) / max(neighbors, 1),
                center_lr_coverage=torch.ones(batch_size, neighbors) * .2,
                center_statistics=torch.tensor([[16., 23., 2., 2.]]).repeat(batch_size, 1).log1p(),
                query_gene_ids=torch.tensor([[2, 3, 4, 0]]).repeat(batch_size, 1),
                query_nucleus_counts=query_n, query_cytoplasm_counts=query_c, query_total_counts=query_n + query_c,
                query_valid_mask=torch.tensor([[True, True, True, False]]).repeat(batch_size, 1),
                cell_identity=centers, neighbor_cell_identity=near_ids)


def make_model(neighbors=3, **kwargs):
    torch.manual_seed(17)
    return SubcellularV12(".", neighbors=neighbors, backbone=DummyBackbone(),
                         encoder_chunk_size=2, backbone_trainable_layers=2, **kwargs)


def sliced(item, rows):
    return {name: value[rows] for name, value in item.items()}


class V12Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_readouts_count_parameters_and_no_target_input(self):
        net, item = make_model().eval(), example()
        first = net(item)
        self.assertEqual(set(first["embeddings"]), set(READOUTS))
        self.assertEqual(first["embeddings"]["concat"].shape, (3, 48))
        for mode in MODES:
            self.assertEqual(first["embeddings"][mode].shape, (3, 16))
            self.assertEqual(set(first["count_parameters"][mode]),
                             {"mu_total", "theta_total", "p_nucleus", "kappa_allocation"})
        changed = deepcopy(item)
        for key in ("query_nucleus_counts", "query_cytoplasm_counts", "query_total_counts"):
            changed[key][:] = 123456
        second = net(changed)
        for name in first["embeddings"]:
            torch.testing.assert_close(first["embeddings"][name], second["embeddings"][name], atol=0, rtol=0)
        for mode in MODES:
            for key in first["count_parameters"][mode]:
                torch.testing.assert_close(first["count_parameters"][mode][key], second["count_parameters"][mode][key], atol=0, rtol=0)
        for key in list(changed):
            if key.startswith("query_"):
                changed.pop(key)
        self.assertNotIn("count_parameters", net(changed, return_embeddings=True))

    def test_zero_count_adapter_preserves_pretrained_cls(self):
        net, item = make_model().eval(), example()
        result = net(item)
        for prefix in ("nucleus", "cytoplasm"):
            genes, bins, valid = [item[f"{prefix}_{name}"] for name in ("gene_ids", "bins", "valid")]
            with torch.no_grad():
                original = net.backbone(genes, bins, ~valid, torch.zeros((*genes.shape, net.dim)))[:, 0]
            torch.testing.assert_close(result["embeddings"][prefix], original, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(result["embeddings"]["local"],
                                   (result["embeddings"]["nucleus"] + result["embeddings"]["cytoplasm"]) / 2)
        norm = (result["embeddings"]["fused"] - result["embeddings"]["local"]).norm(dim=-1)
        torch.testing.assert_close(norm / result["embeddings"]["local"].norm(dim=-1), torch.full((3,), .05))

    def test_independent_compartments_and_pure_spatial(self):
        net, item = make_model().eval(), example()
        first = net(item)["embeddings"]
        changed = deepcopy(item)
        changed["nucleus_bins"][:, 1:3] += 20
        changed["center_statistics"] += 4
        changed["center_lr_bias"] = 1 - changed["center_lr_bias"]
        second = net(changed)["embeddings"]
        torch.testing.assert_close(first["cytoplasm"], second["cytoplasm"], atol=0, rtol=0)
        torch.testing.assert_close(first["spatial"], second["spatial"], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first["nucleus"], second["nucleus"]))
        changed = deepcopy(item)
        changed["cytoplasm_bins"][:, 1:3] += 20
        third = net(changed)["embeddings"]
        torch.testing.assert_close(first["nucleus"], third["nucleus"], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first["cytoplasm"], third["cytoplasm"]))

    def test_gene_order_padding_and_absent_neighbor_invariance(self):
        net, item = make_model().eval(), example()
        item["neighbor_present"][:, -1] = False
        altered = deepcopy(item)
        for prefix in ("nucleus", "cytoplasm", "neighbor"):
            suffixes = ("gene_ids", "bins", "valid") + (() if prefix == "neighbor" else ("log_counts",))
            for suffix in suffixes:
                altered[f"{prefix}_{suffix}"] = altered[f"{prefix}_{suffix}"][..., [0, 2, 1, 3]]
            altered[f"{prefix}_gene_ids"][..., -1] = 99999
            altered[f"{prefix}_bins"][..., -1] = float("nan")
        altered["neighbor_gene_ids"][:, -1] = 99999
        altered["neighbor_bins"][:, -1] = float("nan")
        altered["relative_coords"][:, -1] = float("nan")
        altered["neighbor_lr_bias"][:, -1, :] = float("nan")
        altered["neighbor_lr_bias"][:, :, -1] = float("nan")
        altered["center_lr_bias"][:, -1] = float("nan")
        a, b = net(item), net(altered)
        for name in READOUTS:
            torch.testing.assert_close(a["embeddings"][name], b["embeddings"][name], atol=1e-6, rtol=1e-5)

    def test_directed_lr_and_separate_conditioned_readout(self):
        torch.manual_seed(3)
        module = LRNeighborEncoderV12(16, 4, spatial_radius=15., graph_neighbors=2)
        coords = torch.tensor([[[0., 0.], [5., 0.], [0., 5.]]])
        present = torch.ones(1, 3, dtype=torch.bool)
        with torch.no_grad():
            for layer in (module.query, module.key, module.edge_position[-1], module.center_position[-1]):
                layer.weight.zero_()
                layer.bias.zero_()
            module.readout_query.zero_()
        features, graph_lr, center_lr = torch.randn(1, 3, 16), torch.zeros(1, 3, 3), torch.zeros(1, 3)
        baseline = module(features, present, coords, graph_lr, center_lr)
        graph_lr[0, 0, 1] = 1
        directed = module(features, present, coords, graph_lr, center_lr)
        self.assertTrue(bool((directed[2][0, :, 0, 1] > baseline[2][0, :, 0, 1]).all()))
        torch.testing.assert_close(directed[2][:, :, 1:], baseline[2][:, :, 1:], atol=0, rtol=0)
        graph_lr.zero_()
        center_lr[0, 2] = 1
        conditional = module(features, present, coords, graph_lr, center_lr)
        torch.testing.assert_close(conditional[0], baseline[0], atol=0, rtol=0)
        self.assertTrue(bool((conditional[4][0, :, 2] > baseline[4][0, :, 2]).all()))
        self.assertFalse(torch.allclose(conditional[1], baseline[1]))

    def test_missing_views_and_counts_agree(self):
        net, item = make_model(max_neighbor_distance=25.).eval(), example(batch_size=5)
        item["nucleus_present"][0] = False
        item["cytoplasm_present"][1] = False
        item["neighbor_present"][2] = False
        item["nucleus_present"][3:] = False
        item["cytoplasm_present"][3:] = False
        item["neighbor_present"][4] = False
        result = net(item)
        emb = result["embeddings"]
        torch.testing.assert_close(emb["local"][0], emb["cytoplasm"][0])
        torch.testing.assert_close(emb["local"][1], emb["nucleus"][1])
        torch.testing.assert_close(emb["fused"][2], emb["local"][2])
        torch.testing.assert_close(emb["fused"][3], emb["spatial"][3])
        self.assertEqual(float(emb["fused"][4].abs().sum()), 0.)
        pre = loss_counts_from_batch(item, {}, mode_available=net.mode_availability(item))
        _, post = loss_sums_and_counts(result, item, {})
        for mode in MODES:
            torch.testing.assert_close(pre[mode], post[mode])
        self.assertFalse(bool(result["readout_attention"][:, :, 2].any()))

    def test_empty_views_and_empty_queries_have_zero_gradients(self):
        for count in (0, 3):
            for empty_queries in (False, True):
                with self.subTest(count=count, empty_queries=empty_queries):
                    net, item = make_model(neighbors=count), example(neighbors=count)
                    for prefix in ("nucleus", "cytoplasm", "neighbor"):
                        item[f"{prefix}_present"][:] = False
                    if empty_queries:
                        for key in item:
                            if key.startswith("query_"):
                                item[key] = item[key][:, :0]
                    result = net(item)
                    for value in result["embeddings"].values():
                        self.assertEqual(float(value.abs().sum()), 0.)
                    loss, metrics = losses(result, item, {})
                    self.assertEqual(float(loss), 0.)
                    self.assertFalse(metrics["has_learning_signal"])
                    loss.backward()
                    for name, parameter in net.named_parameters():
                        if parameter.requires_grad:
                            self.assertIsNotNone(parameter.grad, name)
                            self.assertEqual(float(parameter.grad.abs().sum()), 0., name)

    def test_loss_updates_count_adapter_backbone_spatial_decoder(self):
        net, item = make_model(checkpointing=True), example()
        loss, _ = losses(net(item), item, {"loss": {"type": "joint_compartment_count_nll"}})
        loss.backward()
        for parameter in (net.raw_count_adapter[-1].weight, net.joint_decoder.output.weight,
                          net.spatial_encoder.value.weight, net.spatial_encoder.lr_scale_logit,
                          net.fusion_gate.weight, net.environment_residual.weight,
                          net.backbone.transformer_encoder.layers[-1].linear1.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        self.assertIsNone(net.backbone.encoder.weight.grad)
        self.assertIsNone(net.backbone.transformer_encoder.layers[0].linear1.weight.grad)
        net.set_backbone_trainable_layers(0)
        net.zero_grad(set_to_none=True)
        losses(net(item), item, {})[0].backward()
        self.assertGreater(float(net.raw_count_adapter[-1].weight.grad.abs().sum()), 0.)

    def test_joint_probability_normalizes_and_zero_allocation(self):
        # Sum p(N,C) over a support with negligible NB tail.
        pairs = [(n, total - n) for total in range(101) for n in range(total + 1)]
        n = torch.tensor([x[0] for x in pairs])
        c = torch.tensor([x[1] for x in pairs])
        params = {"mu_total": torch.full(n.shape, 2., dtype=torch.float64),
                  "theta_total": torch.full(n.shape, 3., dtype=torch.float64),
                  "p_nucleus": torch.full(n.shape, .3, dtype=torch.float64),
                  "kappa_allocation": torch.full(n.shape, 4., dtype=torch.float64)}
        self.assertAlmostEqual(float(joint_count_log_prob(n, c, params).exp().sum()), 1., places=10)
        params = {name: torch.tensor(value, dtype=torch.float64, requires_grad=True) for name, value in (
            ("mu_total", 2.), ("theta_total", 3.), ("p_nucleus", .3), ("kappa_allocation", 4.))}
        joint_count_nll(params, torch.tensor(0), torch.tensor(0)).backward()
        self.assertEqual(float(params["p_nucleus"].grad), 0.)
        self.assertEqual(float(params["kappa_allocation"].grad), 0.)

    def test_distribution_matches_torch_and_is_stable_for_large_counts(self):
        n, c = torch.tensor([0., 2., 15.], dtype=torch.float64), torch.tensor([3., 5., 8.], dtype=torch.float64)
        params = dict(mu_total=torch.full((3,), 7., dtype=torch.float64),
                      theta_total=torch.full((3,), 4., dtype=torch.float64),
                      p_nucleus=torch.full((3,), .4, dtype=torch.float64),
                      kappa_allocation=torch.full((3,), 5., dtype=torch.float64))
        # NB parameterization uses probs=mu/(mu+theta) in PyTorch.
        nb = torch.distributions.NegativeBinomial(params["theta_total"], probs=7 / 11).log_prob(n + c)
        alpha, beta = params["p_nucleus"] * params["kappa_allocation"], (1 - params["p_nucleus"]) * params["kappa_allocation"]
        beta_fn = lambda a, b: torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
        allocation = torch.lgamma(n + c + 1) - torch.lgamma(n + 1) - torch.lgamma(c + 1) + beta_fn(n + alpha, c + beta) - beta_fn(alpha, beta)
        torch.testing.assert_close(joint_count_log_prob(n, c, params), nb + allocation)
        for p in params:
            params[p].requires_grad_(True)
        large = joint_count_log_prob(n * 1e6, c * 1e6, params)
        self.assertTrue(bool(torch.isfinite(large).all()))
        large.sum().backward()
        self.assertTrue(all(bool(torch.isfinite(value.grad).all()) for value in params.values()))

    def test_target_integrity_and_padding_are_enforced(self):
        net, item = make_model().eval(), example()
        output = net(item)
        for key, value in (("query_nucleus_counts", -1), ("query_total_counts", 999)):
            changed = deepcopy(item)
            changed[key][0, 0] = value
            with self.assertRaises(ValueError):
                losses(output, changed, {})
        changed = deepcopy(item)
        changed["query_nucleus_counts"] = changed["query_nucleus_counts"].float()
        changed["query_nucleus_counts"][0, 1] = 1.5
        with self.assertRaises(ValueError):
            losses(output, changed, {})
        changed = deepcopy(item)
        changed["query_nucleus_counts"][0, -1] = 1
        changed["query_total_counts"][0, -1] = 1
        with self.assertRaises(ValueError):
            losses(output, changed, {})

    def test_exact_gene_cell_mode_and_global_accumulation_reduction(self):
        net, item = make_model().eval(), example(batch_size=5)
        item["nucleus_present"][0] = False
        item["cytoplasm_present"][1:3] = False
        item["neighbor_present"][3:] = False
        item["query_valid_mask"][1, 2] = False
        for key in ("query_nucleus_counts", "query_cytoplasm_counts", "query_total_counts"):
            item[key][1, 2] = 0
        output = net(item)
        actual, _ = losses(output, item, {})
        manual = []
        for mode in MODES:
            errors = joint_count_nll(output["count_parameters"][mode], item["query_nucleus_counts"], item["query_total_counts"])
            cells = [errors[row][item["query_valid_mask"][row]].mean() for row in range(5)
                     if bool(output["mode_available"][mode][row])]
            manual.append(torch.stack(cells).mean())
        torch.testing.assert_close(actual, torch.stack(manual).mean())
        global_counts = loss_counts_from_batch(item, {}, mode_available=net.mode_availability(item))
        accumulated = []
        for rows in (slice(0, 1), slice(1, 3), slice(3, 5)):
            small = sliced(item, rows)
            sums, counts = loss_sums_and_counts(net(small), small, {})
            accumulated.append(reduce_loss_sums(sums, counts, global_counts=global_counts))
        torch.testing.assert_close(sum(accumulated), actual, atol=1e-6, rtol=1e-6)
        net.zero_grad(set_to_none=True)
        actual.backward()
        original_grad = net.joint_decoder.output.weight.grad.clone()
        net.zero_grad(set_to_none=True)
        for rows in (slice(0, 1), slice(1, 3), slice(3, 5)):
            small = sliced(item, rows)
            losses(net(small), small, {}, global_counts=global_counts)[0].backward()
        torch.testing.assert_close(net.joint_decoder.output.weight.grad, original_grad, atol=1e-6, rtol=1e-5)

    def test_build_and_backbone_restrictions(self):
        net = build_model({"model": {"backbone_trainable_layers": 2}}, backbone=DummyBackbone())
        self.assertEqual(net.arch_config["model_type"], "subcellular_compartment_counts_v12")
        self.assertEqual(net.backbone_trainable_layers, 2)
        self.assertFalse(hasattr(net, "fusion"))
        self.assertTrue(all("joint_decoder" in name for name, _ in net.named_parameters() if "decoder" in name))
        with self.assertRaises(ValueError):
            make_model(backbone_kind="scgpt_spatial")
        with self.assertRaises(ValueError):
            make_model(fusion_initial_strength=.5, fusion_max_strength=.25)


if __name__ == "__main__":
    unittest.main()
