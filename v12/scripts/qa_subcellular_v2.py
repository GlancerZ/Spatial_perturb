"""Compute-node QA: official encoder parity, gradient paths and input boundaries."""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Tuple, Union
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
import yaml
from spatial_pretrain.models.scgpt_backbone import ScGPTBackbone, resolve_backbone_assets
from spatial_pretrain.models.subcellular_v2 import SubcellularV2
from train_subcellular_v2 import losses


def load_upstream_reference(asset_dir, backbone_kind):
    """Instantiate the declared upstream TransformerModel, with unused deps omitted.

    Both reference and candidate must read the selected family's exact source and
    weights. Ordinary-scGPT QA never imports the historical spatial eval loader.
    Full upstream class/function bodies are retained; disabled Flash/DSBN/decoder
    options avoid importing libraries unrelated to inference-encoder parity.
    """
    weights, source, repository = resolve_backbone_assets(asset_dir, backbone_kind)
    cfg = json.loads((weights / 'args.json').read_text())
    vocab = json.loads((weights / 'vocab.json').read_text())
    namespace = dict(__name__=f'upstream_{backbone_kind}_qa_reference', gc=gc, math=math,
        warnings=warnings, Dict=Dict, Mapping=Mapping, Optional=Optional, Tuple=Tuple,
        Any=Any, Union=Union, torch=torch, np=np, nn=torch.nn, Tensor=torch.Tensor,
        dist=torch.distributed, F=torch.nn.functional,
        TransformerEncoder=torch.nn.TransformerEncoder,
        TransformerEncoderLayer=torch.nn.TransformerEncoderLayer,
        Bernoulli=torch.distributions.Bernoulli, trange=range, flash_attn_available=False)
    tree = ast.parse(source.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    model_class = namespace['TransformerModel']
    kwargs = dict(ntoken=len(vocab), d_model=cfg['embsize'], nhead=cfg['nheads'],
        d_hid=cfg['d_hid'], nlayers=cfg['nlayers'], nlayers_cls=cfg.get('n_layers_cls', 3),
        n_cls=1, vocab=vocab, dropout=cfg['dropout'], pad_token=cfg['pad_token'],
        pad_value=cfg['pad_value'], do_mvc=False, do_dab=False, use_batch_labels=False,
        input_emb_style='continuous', use_fast_transformer=False, pre_norm=False)
    if 'domain_spec_batchnorm' in inspect.signature(model_class).parameters:
        kwargs['domain_spec_batchnorm'] = False
    model = model_class(**kwargs)
    raw = torch.load(weights / 'best_model.pt', map_location='cpu', weights_only=False)
    prefixes = ('encoder.', 'value_encoder.', 'transformer_encoder.')
    state = {key.replace('Wqkv.', 'in_proj_'): value for key, value in raw.items() if key.startswith(prefixes)}
    expected = {key: value for key, value in model.state_dict().items() if key.startswith(prefixes)}
    if set(state) != set(expected) or any(state[key].shape != value.shape for key, value in expected.items()):
        raise ValueError('Selected upstream reference cannot load every pretrained encoder tensor exactly')
    model.load_state_dict(state, strict=False)  # unused original decoder heads are never called
    return model.eval().requires_grad_(False), vocab, dict(backbone_kind=backbone_kind,
        source_repository=repository, source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_class='TransformerModel', reference_method='_encode',
        loaded_encoder_tensors=len(state), unused_decoder_heads_excluded=True)


def gradient_norm(parameters):
    gradients = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(gradients).sum().sqrt()) if gradients else 0.0


def synthetic_batch(vocab, device):
    ordinary = torch.tensor(sorted(value for key, value in vocab.items() if not key.startswith('<')), device=device)
    if len(ordinary) < 12:
        raise ValueError('Vocabulary is too small for a meaningful pretrained QA batch')
    cls, pad = vocab['<cls>'], vocab['<pad>']
    genes = ordinary[torch.randint(len(ordinary), (1, 4, 12), device=device)]
    genes[..., 0] = cls
    values = torch.randint(1, 51, genes.shape, device=device).float()
    values[..., 0] = -2
    comp = torch.tensor([2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], device=device).view(1, 1, 12).expand(1, 4, 12).clone()
    valid = torch.arange(12, device=device)[None, None] < torch.tensor([12, 10, 8, 1], device=device)[None, :, None]
    genes[~valid] = pad
    values[~valid] = -2
    comp[~valid] = 2
    return dict(gene_ids=genes, input_bins=values, compartment_ids=comp, token_valid_mask=valid,
                coords=torch.tensor([[[0., 0.], [8., 4.], [20., 5.], [1., 14.]]], device=device),
                node_indices=torch.tensor([[3, 11, 20, 25]], device=device),
                query_gene_ids=ordinary[:8].view(1, 8),
                query_compartment_ids=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=device),
                query_valid_mask=torch.ones(1, 8, dtype=torch.bool, device=device),
                target_log_counts=torch.tensor([[0., 1.1, 0., 2.2, .69, 0., 1.4, 0.]], device=device),
                target_nonzero=torch.tensor([[0., 1., 0., 1., 1., 0., 1., 0.]], device=device),
                neighbor_query_mask=torch.tensor([[False, False, False, False, True, True, True, True]], device=device),
                task_mode=torch.tensor([0], device=device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/v2_qa.json')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(9817)
    torch.backends.mha.set_fastpath_enabled(False)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(9817)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    backbone_kind = cfg['model'].get('backbone_kind', 'scgpt_spatial')
    reference, vocab, reference_provenance = load_upstream_reference(cfg['assets_dir'], backbone_kind)
    reference = reference.to(device).eval()
    candidate = ScGPTBackbone(cfg['assets_dir'], checkpointing=False, backbone_kind=backbone_kind).to(device).eval()
    if candidate.provenance['source_sha256'] != reference_provenance['source_sha256']:
        raise AssertionError('Candidate and reference did not load the same declared upstream source')
    genes = torch.tensor(sorted(value for key, value in vocab.items() if not key.startswith('<')), device=device)
    parity = []
    with torch.no_grad():
        for length in (1, 7, 21):
            ids = genes[torch.randint(len(genes), (3, length), device=device)]
            ids[:, 0] = vocab['<cls>']
            values = torch.randint(1, 51, ids.shape, device=device).float()
            values[:, 0] = -2
            lengths = torch.tensor([length, max(1, length - 3), 1], device=device)
            padding = torch.arange(length, device=device)[None] >= lengths[:, None]
            ids[padding], values[padding] = vocab['<pad>'], -2
            expected = reference._encode(ids, values, padding)
            actual = candidate(ids, values, padding, torch.zeros((*ids.shape, candidate.dim), device=device))
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            parity.append(dict(sequence_length=length, max_abs_error=float((actual - expected).abs().max()),
                               compared_elements=actual.numel()))
    del reference
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    model_options = dict(cfg['model'])
    candidate.checkpointing = bool(model_options.get('checkpointing', True))
    model = SubcellularV2(cfg['assets_dir'], **model_options, backbone=candidate).to(device)
    batch = synthetic_batch(vocab, device)
    model.eval()
    with torch.no_grad():
        before = model(batch)
        changed_targets = dict(batch, target_log_counts=batch['target_log_counts'] + 300,
                               target_nonzero=1 - batch['target_nonzero'])
        after_targets = model(changed_targets)
        for key in ('gene_pred', 'intrinsic_pred', 'joint_pred', 'neighbor_pred'):
            torch.testing.assert_close(before[key], after_targets[key], atol=0, rtol=0)
        changed_input = dict(batch, input_bins=batch['input_bins'].clone())
        changed_input['input_bins'][:, 0, 1:] += 13
        after_center = model(changed_input)
        torch.testing.assert_close(before['embeddings']['niche'][:, 0], after_center['embeddings']['niche'][:, 0], atol=0, rtol=0)
        torch.testing.assert_close(before['neighbor_pred'], after_center['neighbor_pred'], atol=0, rtol=0)
    del before, after_targets, after_center
    gradient_checks = []
    for full in (False, True):
        model.set_backbone_trainable(full=full)
        model.train()
        model.zero_grad(set_to_none=True)
        # FP32 checks isolate gradient correctness from mixed precision kernels.
        result = model(batch)
        total, parts = losses(result, batch, cfg, max(1, cfg['training']['spatial_ramp_steps']))
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite QA reconstruction loss')
        total.backward()
        component = gradient_norm([model.compartment_embedding.weight])
        head = gradient_norm(model.repr_decoder.parameters())
        early = gradient_norm(model.backbone.transformer_encoder.layers[0].parameters())
        tail = gradient_norm(model.backbone.transformer_encoder.layers[-1].parameters())
        all_grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not all(torch.isfinite(grad).all() for grad in all_grads):
            raise FloatingPointError('Nonfinite QA gradient')
        if not (component > 0 and head > 0 and tail > 0):
            raise AssertionError('Compartment, reconstruction head or backbone tail lost gradients')
        if full and early <= 0:
            raise AssertionError('Full fine-tuning did not reach the first pretrained layer')
        if not full and model.freeze_layers > 0 and early != 0:
            raise AssertionError('Frozen first pretrained layer unexpectedly has a gradient')
        gradient_checks.append(dict(full_backbone=full, loss=parts, compartment_gradient=component,
                                    representation_head_gradient=head, first_layer_gradient=early,
                                    last_layer_gradient=tail))
    report = dict(passed=True, device=str(device), dtype='float32', official_encoder_parity=parity,
                  backbone_kind=backbone_kind, reference_provenance=reference_provenance,
                  target_independence=True, pure_niche_center_independence=True,
                  gradient_checks=gradient_checks, backbone_provenance=candidate.provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
