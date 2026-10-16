from __future__ import annotations

from copy import deepcopy
import gzip
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import zipfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from spatial_pretrain.config import load_config
from spatial_pretrain.data.compartment_cache import build_compartment_cache
from spatial_pretrain.data.cache import load_slice_cache
from spatial_pretrain.data.dataset import (SpatialSubgraphDataset, prepare_slice_views, collate_subgraphs,
                                           _build_subgraph_table, _candidate_nodes_from_cache)
from spatial_pretrain.data.xenium_compartments import aggregate_batches, convert_xenium_compartments, transcript_batches
from spatial_pretrain.gene_init import load_gene_init_artifact
from spatial_pretrain.models.compartment import (
    CytoplasmicCrossAttention, StackCytoplasmicCellAttention, CompartmentBlock, CompartmentSSGTModel,
)
from spatial_pretrain.training.losses import compute_pretraining_losses
from spatial_pretrain.training.runner import _evaluate, _model_inputs, run_pretraining

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(directory):
    cfg = load_config(ROOT / "configs/xenium_compartment_smoke.yaml")
    cfg.experiment.output_dir = str(directory / "outputs")
    cfg.data.h5ad_path = str(directory / "input.h5ad")
    cfg.data.cache_dir = str(directory / "cache")
    cfg.data.top_genes = 4
    cfg.data.nuclear_tokens = cfg.data.cytoplasmic_tokens = 2
    cfg.data.num_bins = 4
    cfg.data.knn_k = cfg.data.nearest_k = cfg.data.candidate_k = 3
    cfg.data.subgraph_size = 4
    cfg.data.center_selection_strategy = "all"
    cfg.data.chunk_size = 5
    cfg.model.hidden_dim = 8
    cfg.model.num_layers = 2
    cfg.model.gene_heads = cfg.model.cell_heads = 2
    cfg.model.dropout = 0.0
    cfg.model.use_genept = False
    cfg.training.device = "cpu"
    cfg.training.amp = False
    cfg.training.parallelism = "single"
    cfg.training.max_train_steps = 4
    cfg.training.val_every = 1
    cfg.training.max_val_batches = 1
    return cfg


def write_fixture(path):
    rng = np.random.default_rng(6)
    nuclear = rng.integers(0, 8, size=(20, 9), dtype=np.int32)
    cyto = rng.integers(0, 8, size=(20, 9), dtype=np.int32)
    nuclear[0] = 0
    cyto[1] = 0
    data = ad.AnnData(sp.csr_matrix(nuclear + cyto))
    data.obs_names = [f"c{i}" for i in range(20)]
    data.var_names = [f"g{i}" for i in range(9)]
    data.obsm["spatial"] = np.column_stack([np.arange(20), np.arange(20) % 4]).astype(np.float32)
    data.layers["nuclear_counts"] = sp.csr_matrix(nuclear)
    data.layers["extranuclear_counts"] = sp.csr_matrix(cyto)
    data.write_h5ad(path)


def make_dataset(directory, cfg):
    write_fixture(Path(cfg.data.h5ad_path))
    build_compartment_cache(cfg.data.h5ad_path, cfg.data.cache_dir, cfg)
    cache = load_slice_cache(cfg.data.cache_dir)
    prepared = prepare_slice_views(cache, cfg, "fixture")
    return SpatialSubgraphDataset(cache.cache_dir, prepared.train_nodes_path, prepared.train_centers_path,
                                  prepared.train_epoch0_rows, cfg, training=True, pad_gene_id=cache.num_genes)


class CompartmentDataTests(unittest.TestCase):
    def test_batched_neighbor_tables_match_scalar_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_config(Path(tmp))
            cfg.data.candidate_k = cfg.data.nearest_k = 3
            cfg.data.subgraph_size = 5
            cache = SimpleNamespace(knn_indices=np.array([[0, 1, 2], [2, 0, 3], [3, 0, 1], [0, 1, 2]]))
            for idx, allowed in enumerate((np.ones(4, dtype=bool), np.array([True, False, False, False]),
                                           np.zeros(4, dtype=bool))):
                output = Path(tmp) / f"nodes{idx}.npy"
                _build_subgraph_table(cache, np.arange(4), cfg, allowed, output)
                expected = np.stack([_candidate_nodes_from_cache(cache, cfg, i, np.random.default_rng(0), allowed)
                                     for i in range(4)])
                np.testing.assert_array_equal(np.load(output), expected)
                self.assertFalse(output.with_suffix('.npy.partial').exists())

    def test_counts_flags_quality_and_unassigned(self):
        frame = pd.DataFrame({"cell_id": ["a"] * 6 + ["UNASSIGNED", "b"],
                              "feature_name": ["g", "g", "g", "g", "control", "g", "g", "g"],
                              "overlaps_nucleus": [True, "False", 1, 0, 1, "bad", 1, 0],
                              "qv": [20, 40, 30, 19, 40, 40, 40, 40]})
        n, c, stats = aggregate_batches([frame.iloc[:3], frame.iloc[3:]], ["a", "b"], ["g"])
        np.testing.assert_array_equal(n.toarray(), [[2], [0]])
        np.testing.assert_array_equal(c.toarray(), [[1], [1]])
        self.assertEqual(stats["retained"], 4)

    def test_csv_and_parquet_zip_conversion(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frame = pd.DataFrame({"cell_id": ["a", "a", "b"], "feature_name": ["g", "g", "h"],
                                  "overlaps_nucleus": [True, False, False], "qv": [20, 30, 40]})
            ref = ad.AnnData(sp.csr_matrix([[2, 0], [0, 1]], dtype=np.int32))
            ref.obs_names, ref.var_names = ["a", "b"], ["g", "h"]
            ref.obsm["spatial"] = np.array([[0, 0], [1, 1]], dtype=np.float32)
            ref.write_h5ad(directory / "ref.h5ad")
            pq.write_table(pa.Table.from_pandas(frame), directory / "transcripts.parquet")
            for kind in ("csv", "parquet"):
                archive = directory / f"{kind}.zip"
                with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
                    if kind == "csv":
                        z.writestr("output/transcripts.csv.gz", gzip.compress(frame.to_csv(index=False).encode()))
                    else:
                        z.write(directory / "transcripts.parquet", "output/transcripts.parquet")
                output = directory / f"{kind}.h5ad"
                stats = convert_xenium_compartments(archive, directory / "ref.h5ad", output, chunk_rows=2)
                self.assertEqual(stats["reference_mismatched_entries"], 0)
                result = ad.read_h5ad(output)
                np.testing.assert_array_equal(result.layers["nuclear_counts"].toarray(), [[1, 0], [0, 0]])
                np.testing.assert_array_equal(result.layers["extranuclear_counts"].toarray(), [[1, 0], [0, 1]])

    def test_independent_256_quotas_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_config(Path(tmp))
            cfg.data.nuclear_tokens = cfg.data.cytoplasmic_tokens = 256
            cfg.data.top_genes = 512
            ds = make_dataset(Path(tmp), cfg)
            ids, bins, valid, comp = ds._select_compartment_tokens(np.array([0, 1, 2]), np.random.default_rng(42))
            self.assertEqual(ids.shape, (3, 512))
            self.assertFalse(valid[0, :256].any())  # no artificial nuclear counterparts
            self.assertFalse(valid[1, 256:].any())
            self.assertTrue((comp[:, :256][valid[:, :256]] == 0).all())
            self.assertTrue((comp[:, 256:][valid[:, 256:]] == 1).all())
            self.assertTrue((comp[~valid] == 2).all())
            self.assertTrue(((bins[valid] > 0) & (bins[valid] < cfg.data.num_bins)).all())
            sample = ds[0]
            self.assertFalse(sample["loss_mask"][~sample["token_valid_mask"]].any())
            self.assertTrue((sample["gene_ids"][~sample["token_valid_mask"]] == 9).all())
            self.assertTrue((sample["input_bins"][~sample["token_valid_mask"]] == cfg.data.num_bins + 1).all())
            # Changing the requested bin-fitting split must reject cache reuse.
            changed = deepcopy(cfg)
            changed.data.train_fraction = 0.6
            with self.assertRaises(ValueError):
                build_compartment_cache(changed.data.h5ad_path, changed.data.cache_dir, changed)

    def test_genept_aligns_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "genes.pt"
            torch.save({"embeddings": torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),
                        "gene_names": ["b", "a", "c"]}, path)
            result = load_gene_init_artifact(path, num_genes=2, hidden_dim=2, gene_names=["c", "a"])
            torch.testing.assert_close(result, torch.tensor([[5., 6.], [3., 4.]]))


class CompartmentAttentionTests(unittest.TestCase):
    def test_independent_gene_permutations_preserve_predictions_and_pooling(self):
        torch.manual_seed(91)
        model = CompartmentSSGTModel(num_genes=10, num_value_bins=4, hidden_dim=8,
            num_layers=2, gene_heads=2, cell_heads=2, dropout=0, ffn_mult=2,
            gene_padding_idx=9, cell_attention_mode="cytoplasm_cross").double().eval()
        comp = torch.tensor([[[0, 1, 1, 2], [0, 0, 1, 2]]])
        batch = dict(gene_ids=torch.tensor([[[0, 1, 2, 9], [3, 4, 5, 9]]]),
            input_bins=torch.tensor([[[1, 4, 2, 5], [2, 3, 4, 5]]]),
            compartment_ids=comp, token_valid_mask=comp.ne(2),
            cell_roles=torch.tensor([[1, 0]]), node_indices=torch.tensor([[10, 11]]))
        permutation = torch.tensor([[[2, 3, 0, 1], [3, 1, 2, 0]]])
        shuffled = {k: v.gather(2, permutation) if v.ndim == 3 else v for k, v in batch.items()}
        expected = model(**batch)["mask_logits"].gather(2, permutation[..., None].expand(-1, -1, -1, 4))
        torch.testing.assert_close(model(**shuffled)["mask_logits"], expected, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(model(**shuffled, return_embeddings=True)["cell_embeddings"],
                                   model(**batch, return_embeddings=True)["cell_embeddings"])

    def test_cross_stage_has_no_nuclear_gradient(self):
        torch.manual_seed(92)
        layer = CytoplasmicCrossAttention(8, 2, 0).double().eval()
        tokens = torch.randn(1, 3, 4, 8, dtype=torch.double, requires_grad=True)
        comp = torch.tensor([[[0, 1, 1, 2]] * 3])
        output, _ = layer(tokens, comp, comp.ne(2), torch.tensor([[0, 1, 2]]))
        output[0, 0, 1:3].square().sum().backward()
        self.assertEqual(tokens.grad[comp.ne(1)].abs().sum().item(), 0)
        self.assertGreater(tokens.grad[0, 1:, 1:3].abs().sum().item(), 0)
        self.assertTrue((output[comp.ne(1)] == 0).all())

    def test_cross_attention_routing_and_duplicate_cells(self):
        layer = CytoplasmicCrossAttention(4, 1, 0).eval()
        with torch.no_grad():
            for p in layer.parameters():
                p.zero_()
            layer.v_proj.weight.copy_(torch.eye(4))
            layer.out_proj.weight.copy_(torch.eye(4))
        x = torch.zeros(1, 3, 4, 4)
        x[:, :, 0] = 1000  # nuclear: must never enter memory
        x[0, 0, 1:3] = 2
        x[0, 1, 1:3] = 4
        x[0, 2, 1:3] = 900  # duplicate cell: must not overweight memory
        comp = torch.tensor([[[0, 1, 1, 2]] * 3])
        valid = comp.ne(2)
        nodes = torch.tensor([[10, 11, 11]])
        output, active = layer(x, comp, valid, nodes)
        torch.testing.assert_close(output[0, 0, 1:3], torch.full((2, 4), 4.))
        torch.testing.assert_close(output[0, 1:, 1:3], torch.full((2, 2, 4), 2.))
        self.assertTrue((output[~active] == 0).all())
        output, active = layer(x, comp, valid, torch.tensor([[10, 10, 10]]))
        self.assertFalse(active.any())
        self.assertEqual(output.abs().sum().item(), 0)

    def test_nuclear_positions_bypass_cross_stage(self):
        block = CompartmentBlock(8, 2, 2, 0, 2).eval()
        x = torch.randn(1, 2, 4, 8)
        comp = torch.tensor([[[0, 0, 1, 2], [0, 1, 1, 2]]])
        valid = comp.ne(2)
        output = block.across_cells(x, comp, valid, torch.tensor([[0, 1]]))
        torch.testing.assert_close(output[comp.eq(0)], x[comp.eq(0)], rtol=0, atol=0)
        self.assertEqual(output[~valid].abs().sum().item(), 0)

    def test_checkpoint_gradient_equivalence_and_padding(self):
        torch.manual_seed(7)
        kwargs = dict(num_genes=10, num_value_bins=4, hidden_dim=8, num_layers=2,
                      gene_heads=2, cell_heads=2, dropout=0, ffn_mult=2, gene_padding_idx=9)
        plain = CompartmentSSGTModel(**kwargs)
        checked = deepcopy(plain)
        checked.gradient_checkpointing = True
        comp = torch.tensor([[[0, 1, 1, 2], [0, 0, 1, 2]]])
        batch = dict(gene_ids=torch.tensor([[[0, 1, 2, 9], [3, 4, 5, 9]]]),
                     input_bins=torch.tensor([[[1, 4, 2, 5], [2, 3, 4, 5]]]),
                     cell_roles=torch.tensor([[1, 0]]), compartment_ids=comp,
                     node_indices=torch.tensor([[0, 1]]), token_valid_mask=comp.ne(2))
        a, b = plain(**batch)["mask_logits"], checked(**batch)["mask_logits"]
        torch.testing.assert_close(a, b)
        a[comp.ne(2)].square().sum().backward()
        b[comp.ne(2)].square().sum().backward()
        for (name, p), (_, q) in zip(plain.named_parameters(), checked.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, msg=name)
        torch.testing.assert_close(plain.gene_embedding.embedding.weight.grad[9], torch.zeros(8))

    def test_empty_cross_memory_backward(self):
        layer = CytoplasmicCrossAttention(8, 2, 0).train()
        x = torch.randn(1, 2, 4, 8, requires_grad=True)
        comp = torch.zeros(1, 2, 4, dtype=torch.long)
        output, active = layer(x, comp, torch.ones_like(comp, dtype=torch.bool), torch.tensor([[0, 1]]))
        output.sum().backward()
        for p in layer.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_bfloat16_forward_backward(self):
        model = CompartmentSSGTModel(num_genes=12, num_value_bins=32, hidden_dim=32, num_layers=2,
                                    gene_heads=4, cell_heads=4, dropout=0.1, ffn_mult=2,
                                    gene_padding_idx=11, gradient_checkpointing=True).cuda().train()
        comp = torch.tensor([[[0, 0, 1, 1]] * 3], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(gene_ids=torch.ones_like(comp), input_bins=torch.ones_like(comp),
                        cell_roles=torch.tensor([[1, 0, 0]], device="cuda"), compartment_ids=comp,
                        node_indices=torch.tensor([[0, 1, 2]], device="cuda"))["mask_logits"]
            loss = out.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


class StackCellAttentionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_flash_cell_attention_matches_masked_reference(self):
        torch.manual_seed(21)
        flash = StackCytoplasmicCellAttention(16, 4, 0, 2, 4, 8, require_flash=True).cuda()
        reference = deepcopy(flash)
        reference.require_flash = False
        x = torch.randn(2, 4, 6, 16, device="cuda", requires_grad=True)
        y = x.detach().clone().requires_grad_()
        comp = torch.tensor([[[0, 0, 1, 1, 1, 2], [0, 0, 1, 1, 2, 2],
                              [0, 0, 2, 2, 2, 2], [0, 0, 1, 1, 1, 1]]] * 2, device="cuda")
        nodes = torch.tensor([[0, 1, 2, 1], [3, 4, 5, 6]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, _ = flash(x, comp, comp.ne(2), nodes)
            b, _ = reference(y, comp, comp.ne(2), nodes)
        torch.testing.assert_close(a, b, atol=0.004, rtol=0.04)
        a.float().square().mean().backward()
        b.float().square().mean().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=0.00002, rtol=0.05)
        for (name, p), (_, q) in zip(flash.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=0.0001, rtol=0.06, msg=name)

    def test_matches_standard_multihead_attention_and_gradients(self):
        torch.manual_seed(18)
        layer = StackCytoplasmicCellAttention(8, 2, 0, 2, 3, 4).double().eval()
        ref_layer = deepcopy(layer)
        reference = torch.nn.MultiheadAttention(12, 2, dropout=0, batch_first=True).double()
        with torch.no_grad():
            reference.in_proj_weight.copy_(layer.qkv.weight)
            reference.in_proj_bias.zero_()
            reference.out_proj.weight.copy_(layer.proj.weight)
            reference.out_proj.bias.copy_(layer.proj.bias)
        x = torch.randn(2, 4, 5, 8, dtype=torch.double, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        comp = torch.tensor([[[0, 0, 1, 1, 2]] * 4] * 2)
        valid = comp.ne(2)
        nodes = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        out, active = layer(x, comp, valid, nodes)
        cyto_valid = valid[:, :, 2:]
        flat = ref_layer.token_in(y[:, :, 2:]).masked_fill(~cyto_valid[..., None], 0).flatten(2)
        attended, weights = reference(flat, flat, flat, attn_mask=torch.eye(4, dtype=torch.bool),
                                      average_attn_weights=False)
        self.assertEqual(weights.shape, (2, 2, 4, 4))  # cells × cells, not RNA × RNA
        expected = ref_layer.token_out(attended.reshape(2, 4, 3, 4)).masked_fill(~cyto_valid[..., None], 0)
        torch.testing.assert_close(out[:, :, 2:], expected, atol=1e-10, rtol=1e-8)
        self.assertFalse(active[:, :, :2].any())
        out.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, y.grad, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(layer.qkv.weight.grad, reference.in_proj_weight.grad, atol=1e-10, rtol=1e-8)
        torch.testing.assert_close(layer.proj.weight.grad, reference.out_proj.weight.grad, atol=1e-10, rtol=1e-8)

    def test_routing_padding_duplicates_and_cell_order(self):
        torch.manual_seed(19)
        layer = StackCytoplasmicCellAttention(8, 2, 0, 2, 3, 4).eval()
        x = torch.randn(1, 4, 5, 8)
        comp = torch.tensor([[[0, 0, 1, 1, 2]] * 4])
        valid = comp.ne(2)
        nodes = torch.tensor([[10, 11, 12, 12]])
        out, active = layer(x, comp, valid, nodes)
        changed = x.clone()
        changed[:, :, :2] = 10000
        changed[:, :, 4] = -10000
        changed[:, 3, 2:4] *= 1000  # repeated cell cannot contribute K/V
        other, _ = layer(changed, comp, valid, nodes)
        torch.testing.assert_close(out[:, :3], other[:, :3])
        self.assertTrue((out[~active] == 0).all())
        perm = torch.tensor([2, 0, 1, 3])  # preserves first occurrence of each original cell
        reordered, _ = layer(x[:, perm], comp[:, perm], valid[:, perm], nodes[:, perm])
        torch.testing.assert_close(reordered, out[:, perm])
        # One unique cell with cytoplasm: no receiving row has eligible memory.
        empty_comp = comp.clone()
        empty_comp[:, 1:, 2:] = 2
        empty, active = layer(x, empty_comp, empty_comp.ne(2), torch.tensor([[0, 1, 2, 3]]))
        self.assertFalse(active.any())
        self.assertEqual(empty.abs().sum().item(), 0)
        empty.sum().backward()
        for name, p in layer.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_nuclear_bypass_and_checkpoint_gradients(self):
        torch.manual_seed(20)
        kwargs = dict(num_genes=10, num_value_bins=4, hidden_dim=8, num_layers=2,
                      gene_heads=2, cell_heads=2, dropout=0, ffn_mult=2, gene_padding_idx=9,
                      cell_attention_mode="cytoplasm_stack_flat", nuclear_tokens=2,
                      cytoplasmic_tokens=2, cell_token_dim=4)
        model = CompartmentSSGTModel(**kwargs)
        checked = deepcopy(model)
        checked.gradient_checkpointing = True
        comp = torch.tensor([[[0, 2, 1, 2], [0, 0, 1, 1]]])
        valid = comp.ne(2)
        batch = dict(gene_ids=torch.tensor([[[0, 9, 2, 9], [3, 4, 5, 6]]]),
                     input_bins=torch.tensor([[[1, 5, 2, 5], [2, 3, 4, 2]]]),
                     cell_roles=torch.tensor([[1, 0]]), compartment_ids=comp,
                     node_indices=torch.tensor([[0, 1]]), token_valid_mask=valid)
        x = torch.randn(1, 2, 4, 8)
        crossed = model.blocks[0].across_cells(x, comp, valid, batch["node_indices"])
        torch.testing.assert_close(crossed[comp.eq(0)], x[comp.eq(0)], atol=0, rtol=0)
        a, b = model(**batch)["mask_logits"], checked(**batch)["mask_logits"]
        torch.testing.assert_close(a, b)
        a[valid].square().mean().backward()
        b[valid].square().mean().backward()
        for (name, p), (_, q) in zip(model.named_parameters(), checked.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            torch.testing.assert_close(p.grad, q.grad, msg=name)
        invalid_batch = dict(batch, compartment_ids=comp.flip(-1))
        with self.assertRaises(ValueError):
            model(**invalid_batch)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_stack_bfloat16_backward(self):
        model = CompartmentSSGTModel(num_genes=12, num_value_bins=32, hidden_dim=32, num_layers=2,
                                    gene_heads=4, cell_heads=4, dropout=0.1, ffn_mult=2,
                                    gene_padding_idx=11, gradient_checkpointing=True,
                                    cell_attention_mode="cytoplasm_stack_flat", nuclear_tokens=2,
                                    cytoplasmic_tokens=2, cell_token_dim=4).cuda().train()
        comp = torch.tensor([[[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 1, 2]]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(gene_ids=torch.ones_like(comp), input_bins=torch.ones_like(comp),
                        cell_roles=torch.tensor([[1, 0, 0]], device="cuda"), compartment_ids=comp,
                        node_indices=torch.tensor([[0, 1, 2]], device="cuda"),
                        token_valid_mask=comp.ne(2))["mask_logits"]
            loss = out.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


class TrainerTests(unittest.TestCase):
    def test_muon_interrupted_training_resumes(self):
        self.test_interrupted_training_resumes_optimizer_rng_and_batch_cursor(optimizer_name="muon")

    def test_interrupted_training_resumes_optimizer_rng_and_batch_cursor(self, optimizer_name="adamw"):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = tiny_config(directory)
            cfg.model.cell_attention_mode = "cytoplasm_stack_flat"
            cfg.model.cell_token_dim = 4
            cfg.model.dropout = 0.2
            cfg.optimizer.name = optimizer_name
            cfg.training.checkpoint_every = 1
            cfg.training.auto_resume = True
            write_fixture(Path(cfg.data.h5ad_path))
            cfg.experiment.output_dir = str(directory / "baseline")
            run_pretraining(cfg)
            expected = torch.load(directory / "baseline/resume.pt", weights_only=False)
            cfg.experiment.output_dir = str(directory / "resumed")
            cfg.training.run_update_limit = 1
            run_pretraining(cfg)
            partial = torch.load(directory / "resumed/resume.pt", weights_only=False)
            self.assertEqual((partial["epoch"], partial["micro_step"], partial["step"]), (0, 2, 1))
            self.assertFalse(partial["complete"])
            cfg.training.run_update_limit = 0
            run_pretraining(cfg)
            actual = torch.load(directory / "resumed/resume.pt", weights_only=False)
            self.assertTrue(actual["complete"])
            self.assertEqual(actual["step"], expected["step"])
            for name in expected["model"]:
                torch.testing.assert_close(actual["model"][name], expected["model"][name], atol=0, rtol=0, msg=name)
            a_states = actual["optimizer"]
            e_states = expected["optimizer"]
            for name in (("muon", "aux_adamw") if optimizer_name == "muon" else (None,)):
                a = a_states[name] if name else a_states
                e = e_states[name] if name else e_states
                torch.testing.assert_close(a["state"], e["state"], atol=0, rtol=0)
                self.assertEqual(a["param_groups"], e["param_groups"])

    def test_training_evaluation_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = tiny_config(directory)
            ds = make_dataset(directory, cfg)
            # Runner uses the same cache but gives the view cache its actual slice name.
            import shutil
            shutil.rmtree(Path(cfg.data.cache_dir) / "prepared_views")
            run_pretraining(cfg)
            saved = torch.load(directory / "outputs/last.pt", weights_only=False)
            self.assertEqual(saved["step"], 2)
            self.assertTrue(np.isfinite(saved["val_loss"]))
            model = CompartmentSSGTModel(num_genes=10, num_value_bins=4, hidden_dim=8, num_layers=2,
                                        gene_heads=2, cell_heads=2, dropout=0, ffn_mult=4, gene_padding_idx=9)
            model.load_state_dict(saved["model"])
            batch = collate_subgraphs([ds[0]])
            model.train()
            metrics = _evaluate(model, [batch], cfg, torch.device("cpu"))
            self.assertTrue(model.training)
            self.assertTrue(np.isfinite(metrics["loss"]))

    def test_stack_training_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = tiny_config(directory)
            cfg.model.cell_attention_mode = "cytoplasm_stack_flat"
            cfg.model.cell_token_dim = 4
            write_fixture(Path(cfg.data.h5ad_path))
            run_pretraining(cfg)
            saved = torch.load(directory / "outputs/last.pt", weights_only=False)
            model = CompartmentSSGTModel(num_genes=10, num_value_bins=4, hidden_dim=8, num_layers=2,
                                        gene_heads=2, cell_heads=2, dropout=0, ffn_mult=4, gene_padding_idx=9,
                                        cell_attention_mode="cytoplasm_stack_flat", nuclear_tokens=2,
                                        cytoplasmic_tokens=2, cell_token_dim=4)
            model.load_state_dict(saved["model"], strict=True)
            self.assertEqual(saved["step"], 2)
            self.assertTrue(np.isfinite(saved["val_loss"]))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
