"""Compute-node QA on real V3 cache rows and the ordinary scGPT checkpoint.

Each requested batch size gets a fresh model and one disposable optimizer step.
Only a JSON report is saved: QA weights/optimizer never enter the training run.
Capacity inputs favor long sequences among a seeded, slide-stratified sample;
the report describes this bounded sample, not a worst-case memory guarantee.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
import yaml

from spatial_pretrain.data.subcellular_v3 import V3Dataset, collate_v3
from spatial_pretrain.models.subcellular_v3 import SubcellularV3
from spatial_pretrain.training.subcellular_v3 import losses


TOKEN_KEYS = ('gene_ids', 'input_bins', 'compartment_ids', 'token_valid_mask')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def gpu_batch(items):
    return {key: value.cuda() for key, value in collate_v3(items).items()}


def make_model(cfg):
    model = SubcellularV3(cfg['assets_dir'], **cfg['model']).cuda()
    model.set_backbone_trainable_layers(int(cfg['training']['backbone_trainable_layers']))
    if any(parameter.requires_grad for parameter in model.backbone.parameters()):
        raise AssertionError('Initial V3 QA requires the actual frozen-backbone phase')
    model.train()
    model.backbone.eval()
    return model


def gradient_norm(parameter):
    if parameter.grad is None:
        return 0.
    if not torch.isfinite(parameter.grad).all():
        raise FloatingPointError('Nonfinite parameter gradient')
    return float(parameter.grad.detach().float().norm())


def critical_parameters(model):
    result = {'compartment_embedding.weight': model.compartment_embedding.weight}
    for index, block in enumerate(model.local_blocks):
        result[f'local_blocks.{index}.attention.out_proj.weight'] = block.attention.out_proj.weight
        result[f'local_blocks.{index}.ffn.2.weight'] = block.ffn[-1].weight
    for index, block in enumerate(model.spatial_blocks):
        result[f'spatial_blocks.{index}.attention.out.weight'] = block.attention.out.weight
    return result


def choose_examples(dataset, maximum_batch, candidate_pool):
    """Screen real paired-mask centers, keeping visible CLSs and observed senders."""
    rng = np.random.default_rng(19073)
    starts = [0] + dataset.ends[:-1]
    stratified = [int(rng.integers(start, end)) for start, end in zip(starts, dataset.ends) if end > start]
    count = min(len(dataset), max(candidate_pool, maximum_batch * 4, len(stratified)))
    indices = list(dict.fromkeys(stratified + rng.choice(len(dataset), count, replace=False).tolist()))
    candidates = []
    for index in indices:
        item = dataset[index]
        nodes, valid = item['node_indices'], item['token_valid_mask']
        cyto = (valid & (item['compartment_ids'] == 1)).any(-1)
        neighbors = (nodes >= 0) & (nodes != nodes[0]) & cyto
        if int(item['task_mode']) != 0 or not item['compartment_visible_mask'].any() or not neighbors.any():
            continue
        per_compartment = np.stack([(valid & (item['compartment_ids'] == comp)).sum(-1) for comp in (0, 1)], -1)
        metadata = dict(dataset_index=index, slice_index=int(item['slice_index']),
                        center_node=int(nodes[0]), max_compartment_tokens=int(per_compartment.max()),
                        total_gene_tokens=int(per_compartment.sum()), real_context_cells=int((nodes >= 0).sum()),
                        visible_center_compartments=item['compartment_visible_mask'].tolist(),
                        observed_cytoplasm_neighbors=int(neighbors.sum()))
        candidates.append((metadata['max_compartment_tokens'], metadata['total_gene_tokens'], item, metadata))
    candidates.sort(key=lambda candidate: candidate[:2], reverse=True)
    if len(candidates) < maximum_batch:
        raise ValueError(f'Only {len(candidates)} eligible real patches in {len(indices)} screened; '
                         f'increase --candidate-pool for requested batch {maximum_batch}')
    selected = candidates[:maximum_batch]
    return [value[2] for value in selected], dict(screened=len(indices), eligible=len(candidates),
        selection='Seeded slide-stratified centers; paired masks; prioritize long compartment sequences',
        selected=[value[3] for value in selected])


def embedding_difference(actual, expected, *, atol, rtol):
    differences = {}
    for name in expected:
        if not torch.isfinite(actual[name]).all():
            raise FloatingPointError(f'Nonfinite {name} embedding')
        differences[name] = float((actual[name].float() - expected[name].float()).abs().max())
        torch.testing.assert_close(actual[name], expected[name], atol=atol, rtol=rtol)
    return differences


def real_invariants(cfg, item):
    """Check exact input boundaries on a small real patch in float32."""
    model = batch = baseline = changed = output = objective = None
    try:
        torch.manual_seed(42)
        model = make_model(cfg).eval()
        # Limit repeat forward cost; capacity tests below retain full contexts.
        cells = min(8, len(item['node_indices']))
        small = {key: value[:cells] if key in (*TOKEN_KEYS, 'node_indices', 'coords') else value
                 for key, value in item.items()}
        batch = gpu_batch([small])
        with torch.inference_mode():
            baseline = model(batch, return_embeddings=True)['embeddings']
            zero = {}
            for name in ('intrinsic', 'joint'):
                zero[name] = float((baseline[name] - baseline['base']).abs().max())
                torch.testing.assert_close(baseline[name], baseline['base'], atol=1e-7, rtol=0.)
            changed = dict(batch)
            changed['target_bins'] = (batch['target_bins'] + 17).remainder(51)
            changed['target_nonzero'] = ~batch['target_nonzero']
            changed['query_valid_mask'] = ~batch['query_valid_mask']
            targets = embedding_difference(model(changed, return_embeddings=True)['embeddings'], baseline,
                                           atol=0., rtol=0.)
            changed = dict(batch)
            for key, fill in (('gene_ids', model.pad_id), ('input_bins', -2.),
                              ('compartment_ids', 2), ('token_valid_mask', False)):
                tensor = batch[key]
                padding = tensor.new_full((*tensor.shape[:-1], 7), fill)
                changed[key] = torch.cat((tensor, padding), dim=-1)
            padding = embedding_difference(model(changed, return_embeddings=True)['embeddings'], baseline,
                                           atol=1e-7, rtol=0.)
            generator = torch.Generator().manual_seed(919)
            order = torch.cat((torch.zeros(1, dtype=torch.long),
                               1 + torch.randperm(batch['gene_ids'].shape[-1] - 1, generator=generator))).cuda()
            changed = dict(batch)
            for key in TOKEN_KEYS:
                changed[key] = batch[key].index_select(-1, order)
            permutation = embedding_difference(model(changed, return_embeddings=True)['embeddings'], baseline,
                                               atol=3e-5, rtol=2e-4)
        # Decoder-free objective proves the compartment input gradient actually
        # traverses the frozen encoder, rather than merely reaching query IDs.
        model.train()
        model.backbone.eval()
        model.zero_grad(set_to_none=True)
        output = model(batch, return_embeddings=True)['embeddings']['base'][:, 0]
        weights = torch.linspace(-1., 1., output.shape[-1], device='cuda')
        objective = (output * weights).sum()
        objective.backward()
        encoder_input_gradient = gradient_norm(model.compartment_embedding.weight)
        if encoder_input_gradient <= 0:
            raise AssertionError('Frozen encoder detached the trainable compartment input')
        if any(parameter.grad is not None for parameter in model.backbone.parameters()):
            raise AssertionError('Frozen backbone unexpectedly accumulated gradients')
        return dict(status='passed', real_context_cells=cells, dtype='float32',
                    zero_initialization_max_abs=zero, target_mutation_max_abs=targets,
                    padding_max_abs=padding, gene_permutation_max_abs=permutation,
                    encoder_only_compartment_gradient_norm=encoder_input_gradient,
                    backbone_provenance=model.backbone.provenance)
    finally:
        model = batch = baseline = changed = output = objective = None
        gc.collect()
        torch.cuda.empty_cache()


def capacity_update(cfg, items):
    model = optimizer = batch = output = loss = parameters = snapshots = critical = None
    handles = []
    try:
        torch.manual_seed(42)
        model = make_model(cfg)
        batch = gpu_batch(items)
        parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        snapshots = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=cfg['training']['lr_new'],
                                     weight_decay=cfg['training']['weight_decay'])
        backbone_shapes, layer_calls = [], [0] * len(model.backbone.transformer_encoder.layers)
        handles.append(model.backbone.register_forward_pre_hook(
            lambda _module, inputs: backbone_shapes.append(list(inputs[0].shape))))
        for index, layer in enumerate(model.backbone.transformer_encoder.layers):
            handles.append(layer.register_forward_pre_hook(
                lambda _module, _inputs, index=index: layer_calls.__setitem__(index, layer_calls[index] + 1)))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch)
            loss, parts = losses(output, batch, cfg, step=1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite real-data QA loss')
        torch.cuda.synchronize()
        forward_seconds = time.perf_counter() - started
        forward_calls = len(backbone_shapes)
        forward_layer_calls = layer_calls.copy()
        loss.backward()
        torch.cuda.synchronize()
        forward_backward_seconds = time.perf_counter() - started
        critical = critical_parameters(model)
        gradients = {name: gradient_norm(parameter) for name, parameter in critical.items()}
        if any(value <= 0 for value in gradients.values()):
            raise AssertionError(f'Zero gradient in required V3 paths: {gradients}')
        if any(parameter.grad is not None for parameter in model.backbone.parameters()):
            raise AssertionError('Frozen pretrained parameter received a gradient')
        total_norm = float(torch.nn.utils.clip_grad_norm_(list(parameters.values()),
                           cfg['training']['grad_clip_norm'], error_if_nonfinite=True))
        optimizer.step()
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - started
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        changed, critical_changes, maximum_change = 0, {}, 0.
        for name, parameter in parameters.items():
            after = parameter.detach().cpu()
            if not torch.isfinite(after).all():
                raise FloatingPointError(f'Optimizer produced nonfinite parameter {name}')
            change = float((after - snapshots[name]).abs().max())
            maximum_change = max(maximum_change, change)
            changed += int(change > 0.)
            if name in critical:
                critical_changes[name] = change
        if not changed or any(value <= 0 for value in critical_changes.values()):
            raise AssertionError('Optimizer failed to update a required V3 gradient path')
        return dict(status='passed', batch_size=len(items), packed_token_length=int(batch['gene_ids'].shape[-1]),
                    context_cells=int(batch['gene_ids'].shape[1]), dtype='bfloat16 autocast',
                    encoder_chunk_size=model.encoder_chunk_size,
                    encoder_trace=dict(dim=model.dim, layers=len(layer_calls),
                        forward_calls=forward_calls, recomputation_calls=len(backbone_shapes) - forward_calls,
                        forward_input_shapes=backbone_shapes[:forward_calls],
                        forward_layer_calls=forward_layer_calls, total_layer_calls=layer_calls),
                    backbone_mode='eval, frozen weights, differentiable compartment input',
                    loss=parts, forward_seconds=forward_seconds,
                    forward_backward_seconds=forward_backward_seconds, optimizer_update_seconds=update_seconds,
                    max_allocated_gib=allocated / 2**30, max_reserved_gib=reserved / 2**30,
                    gradient_norm=total_norm, required_gradient_norms=gradients,
                    required_parameter_max_changes=critical_changes, changed_parameter_tensors=changed,
                    trainable_parameter_tensors=len(parameters), maximum_parameter_change=maximum_change,
                    parameters_finite=True, frozen_backbone_has_gradients=False)
    finally:
        for handle in handles:
            handle.remove()
        model = optimizer = batch = output = loss = parameters = snapshots = critical = None
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-sizes', default='4,8,16')
    parser.add_argument('--candidate-pool', type=int, default=96)
    parser.add_argument('--encoder-chunk-size', type=int, default=None,
                        help='QA-only model override; never changes the formal config file')
    args = parser.parse_args()
    sizes = sorted(set(int(value.strip()) for value in args.batch_sizes.split(',')))
    if not sizes or min(sizes) < 1 or args.candidate_pool < 1:
        parser.error('Batch sizes and candidate pool must be positive')
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('Run this QA inside an existing Slurm GPU allocation, never on the login node')
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    cfg = yaml.safe_load(args.config.read_text())
    if args.encoder_chunk_size is not None:
        if args.encoder_chunk_size < 1:
            parser.error('--encoder-chunk-size must be positive')
        cfg['model']['encoder_chunk_size'] = args.encoder_chunk_size
    torch.cuda.set_device(0)
    started = time.monotonic()
    dataset = V3Dataset(cfg, split='train')
    items, sample = choose_examples(dataset, max(sizes), args.candidate_pool)
    report = dict(schema='subcellular_v3_real_qa_v1', status='running',
                  jobid=os.environ['SLURM_JOB_ID'], gpu=torch.cuda.get_device_name(),
                  gpu_total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  torch_version=str(torch.__version__), config=str(args.config),
                  config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                  sample=sample, invariants=None, capacities=[],
                  limitation='One update per size on long examples in a bounded sample; not a full-epoch peak-memory guarantee',
                  writes_training_checkpoint=False)
    atomic_json(args.output, report)
    try:
        report['invariants'] = real_invariants(cfg, items[0])
        print('[v3-qa-invariants]', json.dumps(report['invariants']), flush=True)
        atomic_json(args.output, report)
        for size in sizes:
            try:
                result = capacity_update(cfg, items[:size])
            except torch.cuda.OutOfMemoryError as error:
                result = dict(status='oom', batch_size=size, error=str(error),
                              max_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                              max_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
                gc.collect()
                torch.cuda.empty_cache()
            report['capacities'].append(result)
            atomic_json(args.output, report)
            print('[v3-qa-capacity]', json.dumps(result), flush=True)
        passed = [row['batch_size'] for row in report['capacities'] if row['status'] == 'passed']
        if not passed:
            raise RuntimeError('No requested batch size completed one finite optimizer update')
        report['largest_successful_batch_size'] = max(passed)
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        atomic_json(args.output, report)
    print('[v3-qa-passed]', str(args.output), flush=True)


if __name__ == '__main__':
    main()
