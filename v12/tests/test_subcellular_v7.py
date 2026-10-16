"""Deterministic binning, backbone gradients and optimizer partition for V7."""
import copy
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from spatial_pretrain.data.subcellular_v7 import (
    deterministic_quantile_bins_v7,
    encode_compartment_context_v7,
    encode_whole_cell_v7,
)
from spatial_pretrain.models.subcellular_v7 import SubcellularV7
from spatial_pretrain.training.subcellular_v7_optimizer import build_optimizer, optimizer_manifest
from test_subcellular_v6_model import MockBackbone, example


class V7DeterministicBinningTests(unittest.TestCase):
    def test_equal_values_share_one_bin_and_match_left_digitize(self):
        nuclear = np.array([0, 1, 1, 2, 9, 9], dtype=np.float32)
        cyto = np.array([0, 1, 2, 2, 9, 20], dtype=np.float32)
        left, right = deterministic_quantile_bins_v7(nuclear, cyto)
        values = np.concatenate((nuclear, cyto))
        positive = values > 0
        expected = np.zeros(len(values), dtype=np.int64)
        edges = np.quantile(values[positive], np.linspace(0, 1, 50))
        expected[positive] = np.digitize(values[positive], edges)
        np.testing.assert_array_equal(np.r_[left, right], expected)
        for value in np.unique(values[positive]):
            self.assertEqual(len(np.unique(expected[values == value])), 1)

    def test_compartment_bins_do_not_depend_on_rng_seed(self):
        nuclear = np.array([[1, 1, 2, 2, 4, 4]], dtype=np.float32)
        cyto = np.array([[1, 1, 3, 3, 4, 4]], dtype=np.float32)
        genes = np.arange(10, 16)
        outputs = [encode_compartment_context_v7(
            nuclear, cyto, genes, node_indices=np.array([7]), seed=seed,
            context_tokens_per_compartment=6, pad_token_id=0, cls_token_id=1)
            for seed in (3, 999)]
        mappings = []
        for output in outputs:
            mappings.append({
                (int(compartment), int(gene)): int(value)
                for gene, value, compartment, valid in zip(
                    output["gene_ids"][0], output["input_bins"][0],
                    output["compartment_ids"][0], output["token_valid_mask"][0])
                if valid and gene not in (0, 1)})
        self.assertEqual(mappings[0], mappings[1])

    def test_whole_cell_matches_deterministic_plain_scgpt_formula(self):
        nuclear = np.array([0, 1, 1, 3, 5, 5, 10])
        cyto = np.array([0, 0, 1, 0, 5, 5, 10])
        genes = np.arange(10, 17)
        encoded = encode_whole_cell_v7(
            nuclear, cyto, genes, pad_token_id=0, cls_token_id=1,
            max_tokens=100, seed=123)
        total = nuclear + cyto
        expressed = np.flatnonzero(total > 0)
        edges = np.quantile(total[expressed], np.linspace(0, 1, 50))
        expected = np.digitize(total[expressed], edges)
        np.testing.assert_array_equal(encoded["whole_gene_ids"][1:], genes[expressed])
        np.testing.assert_array_equal(encoded["whole_input_bins"][1:], expected)
        for value in np.unique(total[expressed]):
            self.assertEqual(len(np.unique(expected[total[expressed] == value])), 1)


def model_for_v7(route="A"):
    model = SubcellularV7(
        ".", neighbors=2, checkpointing=False, encoder_chunk_size=3,
        backbone=MockBackbone(), slots_per_compartment=2, slot_dim=8,
        route=route)
    return model


def optimizer_config():
    return dict(
        optimizer="muon", lr_muon=1e-4, lr_new=1e-4,
        lr_backbone=5e-6, backbone_trainable_layers=2,
        weight_decay=.01, muon_adjust_lr_fn="match_rms_adamw",
        muon_momentum=.95, muon_nesterov=True, muon_ns_steps=5)


class V7BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(902)
        torch.backends.mha.set_fastpath_enabled(False)

    def test_only_suffix_trains_and_whole_cls_reaches_it(self):
        model = model_for_v7("A")
        model.set_backbone_trainable_layers(1)
        model.train()
        self.assertTrue(model.backbone.training)
        self.assertFalse(model.backbone.transformer_encoder.layers[0].training)
        self.assertTrue(model.backbone.transformer_encoder.layers[1].training)
        output = model(example())
        selected = example()["query_valid_mask"]
        loss = F.cross_entropy(output["joint_logits"][selected], example()["target_bins"][selected])
        loss.backward()
        first = list(model.backbone.transformer_encoder.layers[0].parameters())
        last = list(model.backbone.transformer_encoder.layers[1].parameters())
        self.assertTrue(all(parameter.grad is None for parameter in first))
        self.assertTrue(any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                            for parameter in last))
        self.assertTrue(all(parameter.grad is None for parameter in model.backbone.encoder.parameters()))

    def test_optimizer_has_three_disjoint_groups(self):
        model = model_for_v7("B")
        model.set_backbone_trainable_layers(2)
        training = optimizer_config()
        manifest = optimizer_manifest(model, training)
        groups = {row["name"]: row for row in manifest["groups"]}
        self.assertEqual(set(groups), {
            "muon_new_hidden", "adamw_new_aux", "adamw_backbone"})
        names = [name for row in groups.values() for name in row["param_names"]]
        expected = [name for name, parameter in model.named_parameters()
                    if parameter.requires_grad]
        self.assertEqual(set(names), set(expected))
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(groups["adamw_backbone"]["param_names"])
        self.assertTrue(all(name.startswith("backbone.")
                            for name in groups["adamw_backbone"]["param_names"]))
        self.assertFalse(any(name.startswith("backbone.")
                             for name in groups["muon_new_hidden"]["param_names"]
                             + groups["adamw_new_aux"]["param_names"]))

    @unittest.skipUnless(callable(getattr(torch.optim, "Muon", None)),
                         "installed torch lacks official Muon")
    def test_optimizer_round_trip_and_rejects_missing_backbone_state(self):
        model = model_for_v7("B")
        model.set_backbone_trainable_layers(2)
        optimizer = build_optimizer(model, optimizer_config())
        output = model(example())
        selected = example()["query_valid_mask"]
        loss = F.cross_entropy(
            output["joint_logits"][selected], example()["target_bins"][selected])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved = copy.deepcopy(optimizer.state_dict())

        restored_model = model_for_v7("B")
        restored_model.set_backbone_trainable_layers(2)
        restored = build_optimizer(restored_model, optimizer_config())
        restored.load_state_dict(saved)
        self.assertTrue(restored.resume_state_verified)
        self.assertGreater(restored.restored_parameter_states["backbone_adamw"], 0)

        damaged = copy.deepcopy(saved)
        backbone_group = next(
            group for group in damaged["adamw"]["param_groups"]
            if group["name"] == "adamw_backbone")
        identifier, parameter_name = backbone_group["params"][0], backbone_group["param_names"][0]
        damaged["adamw"]["state"].pop(identifier)
        damaged["initialized_parameters"]["adamw_backbone"].remove(parameter_name)
        rejected_model = model_for_v7("B")
        rejected_model.set_backbone_trainable_layers(2)
        rejected = build_optimizer(rejected_model, optimizer_config())
        with self.assertRaisesRegex(ValueError, "incomplete parameter state"):
            rejected.load_state_dict(damaged)


if __name__ == "__main__":
    unittest.main()
