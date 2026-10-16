"""Evaluate a V9 checkpoint on the fixed 5,200-cell LungPF cohort.

Same cohort, donor split and fixed-C probes as V6/V7 (cell type C=0.1,
region C=0.01; the full C grid is kept too). Each selected cell is encoded
with its K nearest neighbours in the same sample, nothing masked. Readouts:
``self`` (center CLS), ``ctx`` (one-hop neighbour view), ``joint`` = [self; ctx],
and the neighbour-shuffle controls ``ctx_neighbor_shuffle`` /
``joint_neighbor_shuffle`` (neighbour expression permuted within the slide,
positions kept). Masked validation reports both CEs and the consistency term
on fixed epoch-0 validation examples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402

from evaluate_subcellular_v2 import (  # noqa: E402
    CANONICAL_COHORT, atomic_array, atomic_json, file_sha256, pack_tokens, probe_embedding)
from evaluate_subcellular_v7 import shuffle_expression_nodes, measured_visibility  # noqa: E402
from spatial_pretrain.data.subcellular_v6 import load_lr_prior  # noqa: E402
from spatial_pretrain.data.subcellular_v8 import tie_policy  # noqa: E402
from spatial_pretrain.data.subcellular_v9 import (  # noqa: E402
    V9Dataset, center_lr_v9, collate_v9, encode_cells_v9, v9_options)
from spatial_pretrain.models.subcellular_v9 import SubcellularV9  # noqa: E402
from spatial_pretrain.training.subcellular_v9 import losses  # noqa: E402
from train_subcellular_v9 import ARCHITECTURE, MODEL_VERSION, validate_config  # noqa: E402

READOUTS = ('self', 'ctx', 'joint')
SHUFFLED = ('ctx', 'joint')
FIXED_C = {'cell_type': .1, 'region': .01}


def load_model(path, expected_step=None):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('model_version') != MODEL_VERSION or saved.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected a V9 checkpoint')
    step = int(saved['step'])
    if expected_step is not None and step != expected_step:
        raise ValueError(f'Checkpoint step mismatch: {step} != {expected_step}')
    cfg = saved['config']
    validate_config(cfg)
    options = dict(cfg['model'])
    options['checkpointing'] = False
    model = SubcellularV9(cfg['assets_dir'], **options)
    model.set_backbone_trainable_layers(int(cfg['training']['backbone_trainable_layers']))
    model.load_state_dict(saved['model'], strict=True)
    del saved
    return model.eval().requires_grad_(False).cuda(), cfg, step


def extract(model, cfg, vocab, directory, obs, selected, batch_size, prior, policy, k):
    coords = obs[['x_centroid', 'y_centroid']].to_numpy(dtype=np.float32)
    cells = min(k + 1, len(obs))
    nearest = NearestNeighbors(n_neighbors=cells).fit(coords).kneighbors(
        coords[selected], return_distance=False)
    nodes = np.stack([np.r_[c, row[row != c][:cells - 1]] for c, row in zip(selected, nearest)])
    genes = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([i for i, g in enumerate(genes) if g in vocab], dtype=np.int64)
    global_ids = np.array([vocab[g] for g in genes[keep]], dtype=np.int64)
    nuclear = np.load(directory / 'nuclear.npy', mmap_mode='r', allow_pickle=False)
    cytoplasm = np.load(directory / 'cytoplasmic.npy', mmap_mode='r', allow_pickle=False)
    slice_seed = int.from_bytes(hashlib.sha256(directory.name.encode()).digest()[:4], 'little')
    visibility = measured_visibility(nuclear, cytoplasm, keep)
    shuffled = shuffle_expression_nodes(nodes, visibility, np.random.SeedSequence([42, slice_seed, 815]))
    expressions = {'real': nodes, 'neighbor_shuffle': shuffled}
    unique = np.unique(np.concatenate(list(expressions.values())))
    encoded = encode_cells_v9(
        np.log1p(nuclear[np.ix_(unique, keep)].astype(np.float32)),
        np.log1p(cytoplasm[np.ix_(unique, keep)].astype(np.float32)), global_ids,
        policy=policy, node_indices=unique, seed=42, epoch=0, slice_seed=slice_seed,
        context_tokens_per_compartment=int(cfg['data']['context_tokens_per_compartment']),
        pad_token_id=vocab['<pad>'], cls_token_id=vocab['<cls>'])
    packed = pack_tokens(encoded)
    del encoded
    result = {name: [] for name in READOUTS}
    result.update({f'{name}_neighbor_shuffle': [] for name in SHUFFLED})
    for condition, expression in expressions.items():
        mapped = np.searchsorted(unique, expression)
        for start in range(0, len(nodes), batch_size):
            end = min(start + batch_size, len(nodes))
            index = mapped[start:end]
            width = int(packed['token_valid_mask'][index].sum(axis=-1).max())
            batch = {key: torch.as_tensor(values[index, :width], device='cuda')
                     for key, values in packed.items()}
            batch['node_indices'] = torch.as_tensor(nodes[start:end], device='cuda')
            batch['coords'] = torch.as_tensor(coords[nodes[start:end]], device='cuda')
            lr = []
            for row in expression[start:end]:
                raw_n = np.asarray(nuclear[np.ix_(row, keep)], dtype=np.int64)
                raw_c = np.asarray(cytoplasm[np.ix_(row, keep)], dtype=np.int64)
                known = np.ones_like(raw_c, dtype=bool)
                lr.append(center_lr_v9(raw_n[0] + raw_c[0], known[0], raw_c, known, global_ids,
                                       np.arange(len(row)), prior['pairs'], prior['weights'])
                          if prior['metadata'].get('enabled') else np.zeros(len(row), np.float32))
            batch['lr_center'] = torch.as_tensor(np.stack(lr), device='cuda')
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch, return_embeddings=True)['embeddings']
            names = READOUTS if condition == 'real' else SHUFFLED
            for name in names:
                value = output[name].float().cpu().numpy()
                if not np.isfinite(value).all():
                    raise FloatingPointError(f'Nonfinite {name} in {directory.name}')
                result[name if condition == 'real' else f'{name}_neighbor_shuffle'].append(value)
    return {key: np.concatenate(value) for key, value in result.items()}


def masked_validation(model, cfg, batches, batch_size):
    dataset = V9Dataset(cfg, split='val')
    dataset.set_epoch(0)
    budget = min(len(dataset), batches * batch_size)
    chosen = np.sort(np.random.default_rng(173).choice(len(dataset), budget, replace=False))
    totals, by_task, count = {}, {}, 0
    for start in range(0, len(chosen), batch_size):
        items = [dataset[int(i)] for i in chosen[start:start + batch_size]]
        batch = {k: v.cuda() for k, v in collate_v9(items).items()}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch)
            _, parts = losses(output, batch, cfg, 0)
            for task in (0, 1):
                selection = batch['task_mode'].eq(task)
                if selection.any():
                    sub = {k: v[selection] for k, v in batch.items()}
                    sub_out = {k: output[k][selection] for k in ('self_logits', 'ctx_logits')}
                    _, sub_parts = losses(sub_out, sub, cfg, 0)
                    record = by_task.setdefault(('gene_mask', 'cytoplasm_mask')[task], {})
                    for key, value in sub_parts.items():
                        record.setdefault(key, []).append((value, int(selection.sum())))
        n = len(items)
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.) + value * n
        count += n
    summary = {key: value / count for key, value in totals.items()}
    summary['by_task'] = {task: {key: sum(v * w for v, w in rows) / sum(w for _, w in rows)
                                 for key, rows in record.items()} for task, record in by_task.items()}
    summary['centers'] = int(count)
    summary['lr_strength'] = model.spatial.lr_strength.detach().float().cpu().tolist()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--expected-step', type=int)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--cells-per-donor', type=int, default=400)
    parser.add_argument('--mask-validation-batches', type=int, default=16)
    args = parser.parse_args()
    started = time.monotonic()
    torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    model, cfg, step = load_model(args.checkpoint, args.expected_step)
    policy, k = tie_policy(cfg), v9_options(cfg)['neighbors']
    vocab = model.vocab
    prior = load_lr_prior(cfg, vocab)
    prepared = ROOT / 'data/benchmarking_data/lungpf/prepared'
    protocol = json.loads((prepared / 'protocol.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames, matrices = [], {}
    for sample in protocol['samples']:
        donors = [d for d in protocol['donor_split'] if sample.startswith(d)]
        if len(donors) != 1:
            raise ValueError(f'Ambiguous donor: {sample}')
        split = protocol['donor_split'][donors[0]]
        if split == 'test':
            continue
        directory = prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        selected = np.sort(np.random.default_rng(42).choice(
            len(obs), min(args.cells_per_donor, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy()
        chosen['split'] = split
        features = extract(model, cfg, vocab, directory, obs, selected, args.batch_size, prior, policy, k)
        for key, value in features.items():
            matrices.setdefault(key, []).append(value)
        frames.append(chosen)
        print(f'[embed-v9] {sample} {len(selected)} cells {time.monotonic() - started:.0f}s', flush=True)
    obs = pd.concat(frames, ignore_index=True)
    cohort = hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest()
    if args.cells_per_donor == 400 and cohort != CANONICAL_COHORT:
        raise ValueError('Evaluation cohort differs from the fixed 5,200-cell cohort')
    obs.to_parquet(args.output_dir / 'cells.parquet', index=False)
    reconstruction = masked_validation(model, cfg, args.mask_validation_batches, args.batch_size)
    del model
    torch.cuda.empty_cache()
    results = []
    for key, blocks in matrices.items():
        features = np.concatenate(blocks)
        atomic_array(args.output_dir / f'{key}_embeddings.npy', features)
        for row in probe_embedding(obs, features, key, args.output_dir):
            row['fixed_C_result'] = next(item for item in row['grid'] if item['C'] == FIXED_C[row['task']])
            results.append(row)
    report = dict(model='SubcellularV9', model_version=MODEL_VERSION, architecture=ARCHITECTURE,
                  step=step, checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=file_sha256(args.checkpoint), tie_policy=policy, neighbors=k,
                  config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                  cells=len(obs), cohort_sha256=cohort, canonical_cohort=cohort == CANONICAL_COHORT,
                  test_donors_used=False, masked_validation=reconstruction, results=results,
                  seconds=time.monotonic() - started)
    atomic_json(args.output_dir / 'metrics.json', report)
    summary = {f"{r['embedding']}/{r['task']}": round(r['fixed_C_result']['val_macro_f1'], 4) for r in results}
    print(f'[done-v9] step={step} {json.dumps(summary)} ce={json.dumps({k: v for k, v in reconstruction.items() if k != "by_task"})}',
          flush=True)


if __name__ == '__main__':
    main()
