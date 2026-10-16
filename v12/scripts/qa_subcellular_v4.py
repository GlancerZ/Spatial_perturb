"""Bounded real-cache GPU QA for gene-set V4; never saves training weights."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
import yaml

from spatial_pretrain.data.subcellular_v3 import V3Dataset
from spatial_pretrain.models.subcellular_v4 import SubcellularV4
from spatial_pretrain.training.subcellular_v4 import losses
from qa_subcellular_v3 import (
    TOKEN_KEYS, atomic_json, choose_examples, embedding_difference, gpu_batch,
    gradient_norm,
)


def make_model(cfg):
    model = SubcellularV4(cfg['assets_dir'], **cfg['model']).cuda()
    layers = int(cfg['training']['backbone_trainable_layers'])
    if layers != 0:
        raise ValueError('This QA verifies the approved frozen-backbone V4 phase')
    model.set_backbone_trainable_layers(layers)
    model.train()
    model.backbone.eval()
    return model


def critical_parameters(model):
    parameters = {
        'compartment_embedding.weight': model.compartment_embedding.weight,
        'slot_pool.queries': model.slot_pool.queries,
        'slot_pool.gene_projection.weight': model.slot_pool.gene_projection.weight,
        'slot_pool.attention.out_proj.weight': model.slot_pool.attention.out_proj.weight,
    }
    for name in ('local_pre_blocks', 'local_post_blocks'):
        for index, block in enumerate(getattr(model, name)):
            parameters[f'{name}.{index}.attention.out_proj.weight'] = block.attention.out_proj.weight
            parameters[f'{name}.{index}.ffn.2.weight'] = block.ffn[-1].weight
    for index, block in enumerate(model.spatial_blocks):
        parameters[f'spatial_blocks.{index}.attention.out.weight'] = block.attention.out.weight
    return parameters


def real_invariants(cfg, item):
    model = make_model(cfg).eval()
    cells = min(8, len(item['node_indices']))
    small = {key: value[:cells] if key in (*TOKEN_KEYS, 'node_indices', 'coords') else value
             for key, value in item.items()}
    batch = gpu_batch([small])
    with torch.inference_mode():
        baseline = model(batch, return_embeddings=True)['embeddings']
        for name in ('intrinsic', 'joint'):
            torch.testing.assert_close(baseline[name], baseline['base'], atol=1e-7, rtol=0.)
        changed = dict(batch)
        changed['target_bins'] = (batch['target_bins'] + 17).remainder(51)
        targets = embedding_difference(model(changed, return_embeddings=True)['embeddings'],
                                       baseline, atol=0., rtol=0.)
        changed = dict(batch)
        for key, fill in (('gene_ids', model.pad_id), ('input_bins', -2.),
                          ('compartment_ids', 2), ('token_valid_mask', False)):
            tensor = batch[key]
            changed[key] = torch.cat((tensor, tensor.new_full((*tensor.shape[:-1], 7), fill)), -1)
        padding = embedding_difference(model(changed, return_embeddings=True)['embeddings'],
                                       baseline, atol=3e-5, rtol=2e-4)
        generator = torch.Generator().manual_seed(919)
        order = torch.cat((torch.zeros(1, dtype=torch.long),
                           1 + torch.randperm(batch['gene_ids'].shape[-1] - 1, generator=generator))).cuda()
        changed = dict(batch)
        for key in TOKEN_KEYS:
            changed[key] = batch[key].index_select(-1, order)
        permutation = embedding_difference(model(changed, return_embeddings=True)['embeddings'],
                                           baseline, atol=3e-5, rtol=2e-4)
    model.zero_grad(set_to_none=True)
    base = model(batch, return_embeddings=True)['embeddings']['base'][:, 0]
    (base * torch.linspace(-1., 1., base.shape[-1], device='cuda')).sum().backward()
    encoder_input_gradient = gradient_norm(model.compartment_embedding.weight)
    if encoder_input_gradient <= 0:
        raise AssertionError('Frozen scGPT detached the trainable compartment input')
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise AssertionError('Frozen scGPT accumulated a parameter gradient')
    result = dict(status='passed', real_context_cells=cells, dtype='float32',
                  target_mutation_max_abs=targets, padding_max_abs=padding,
                  gene_permutation_max_abs=permutation,
                  encoder_only_compartment_gradient_norm=encoder_input_gradient,
                  architecture=model.arch_config,
                  backbone_provenance=model.backbone.provenance)
    del model, batch, baseline, changed, base
    gc.collect()
    torch.cuda.empty_cache()
    return result


def capacity_update(cfg, items):
    torch.manual_seed(42)
    model = make_model(cfg)
    batch = gpu_batch(items)
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    critical = critical_parameters(model)
    before = {name: p.detach().cpu().clone() for name, p in critical.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=cfg['training']['lr_new'],
                                 weight_decay=cfg['training']['weight_decay'])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(batch)
        loss, parts = losses(output, batch, cfg, step=1)
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite real-cache V4 objective')
    if output['joint_logits'].shape != (*batch['target_bins'].shape, 51):
        raise AssertionError('V4 decoder must output exactly one 51-category distribution per query')
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    gradients = {name: gradient_norm(p) for name, p in critical.items()}
    if any(value <= 0 for value in gradients.values()):
        raise AssertionError(f'No gradient through a required V4 path: {gradients}')
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise AssertionError('Frozen backbone received optimizer gradients')
    norm = float(torch.nn.utils.clip_grad_norm_(parameters.values(),
                 cfg['training']['grad_clip_norm'], error_if_nonfinite=True))
    optimizer.step()
    torch.cuda.synchronize()
    changes = {name: float((p.detach().cpu() - before[name]).abs().max()) for name, p in critical.items()}
    if any(change <= 0 for change in changes.values()):
        raise AssertionError(f'Required parameter did not update: {changes}')
    if any(not torch.isfinite(p).all() for p in parameters.values()):
        raise FloatingPointError('Optimizer produced a nonfinite parameter')
    result = dict(status='passed', batch_size=len(items),
                  context_cells=int(batch['gene_ids'].shape[1]),
                  packed_token_length=int(batch['gene_ids'].shape[-1]),
                  encoder_chunk_size=model.encoder_chunk_size, dtype='bfloat16 autocast',
                  architecture=model.arch_config, loss=parts,
                  trainable_parameters=sum(p.numel() for p in parameters.values()),
                  frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                  forward_seconds=forward_seconds, forward_backward_seconds=backward_seconds,
                  optimizer_update_seconds=time.perf_counter() - started,
                  max_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  max_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  gradient_norm=norm, required_gradient_norms=gradients,
                  required_parameter_max_changes=changes,
                  frozen_backbone_has_gradients=False)
    del model, optimizer, batch, output, loss, parameters, critical
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-sizes', default='16')
    parser.add_argument('--candidate-pool', type=int, default=64)
    args = parser.parse_args()
    sizes = sorted(set(int(value.strip()) for value in args.batch_sizes.split(',')))
    if not sizes or min(sizes) < 1 or args.candidate_pool < 1:
        parser.error('Positive batch sizes and candidate pool required')
    if args.output.exists():
        raise FileExistsError('Use a new QA report path; existing evidence is preserved')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run V4 QA inside a verified Slurm GPU allocation')
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(42)
    started = time.monotonic()
    report = dict(schema='subcellular_v4_real_qa_v1', status='running',
                  jobid=os.environ['SLURM_JOB_ID'], hostname=socket.gethostname(),
                  python=sys.executable, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(),
                  config=str(args.config.resolve()),
                  config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                  sample=None, invariants=None, capacities=[], writes_training_checkpoint=False,
                  limitation='One update per size on a bounded long-sequence sample; no claim of convergence or full-epoch memory guarantee')
    atomic_json(args.output, report)
    try:
        dataset = V3Dataset(cfg, split='train')
        items, report['sample'] = choose_examples(dataset, max(sizes), args.candidate_pool)
        report['invariants'] = real_invariants(cfg, items[0])
        atomic_json(args.output, report)
        for size in sizes:
            result = capacity_update(cfg, items[:size])
            report['capacities'].append(result)
            atomic_json(args.output, report)
            print('[v4-qa-capacity]', json.dumps(result), flush=True)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.output, report)
    print('[v4-qa-passed]', str(args.output), flush=True)


if __name__ == '__main__':
    main()
