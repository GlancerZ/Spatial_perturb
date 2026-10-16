"""Small, read-only Visium HD transfer pilot; execute on a GPU compute node.

Two explicit input contracts:
* bins: current 8-um total-count H5ADs -> bin CLS and V12 spatial-only readout.
* compartments: real, segmentation-derived nuclear/cytoplasmic count layers,
  with spatial_um coordinates -> all six V12 readouts plus ordinary scGPT.

Neither mode invents compartment assignments or reports annotation accuracy
without independent labels. Embedding QC and PCA maps are compatibility checks,
not evidence of biological accuracy or RNA velocity. The checkpoint is frozen.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch

from evaluate_subcellular_v12 import file_sha256, atomic_json
from load_v12_portable import load_model, require_compute_execution
from spatial_pretrain.data.subcellular_v10 import _encode_counts
from spatial_pretrain.data.subcellular_v12 import encode_v12_views, collate_v12
from spatial_pretrain.data.subcellular_v6 import load_lr_prior, compute_lr_features_v6


def checked_counts(value, name):
    matrix = sparse.csr_matrix(value)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    data = matrix.data
    if (not np.isfinite(data).all() or (data < 0).any()
            or (data >= 2**53).any() or not np.array_equal(data, np.rint(data))):
        raise ValueError(f'{name} must contain raw nonnegative integer counts')
    return matrix.astype(np.int64)


def bin_coordinates(adata):
    """Use the physical capture grid, never mistake image pixels for microns.

    Keeping slide-grid axes avoids estimating image scale/rotation. Both axes
    have an exact 8-um pitch; image coordinates are retained separately.
    """
    parsed = [re.fullmatch(r's_008um_(\d+)_(\d+)(?:-\d+)?', str(x), re.I)
              for x in adata.obs_names]
    if not all(parsed):
        raise ValueError('bins mode requires documented 8-um Space Ranger barcodes')
    grid = np.asarray([(int(m[2]), int(m[1])) for m in parsed], dtype=np.int64)
    if len(np.unique(grid, axis=0)) != len(grid):
        raise ValueError('Duplicate capture coordinates: multiple slices are unsupported')
    if {'array_col', 'array_row'}.issubset(adata.obs):
        original = adata.obs[['array_col', 'array_row']].to_numpy()
        if not np.array_equal(grid, original):
            raise ValueError('Barcode/grid coordinate mismatch')
    return grid.astype(np.float64) * 8., {
        'units': 'micrometer', 'source': '8-um barcode column/row times 8',
        'axes': 'physical capture grid, not histology image axes',
        'image_pixel_coordinates_used_for_neighbors': False}


def map_genes(adata, vocab):
    symbols = np.asarray(adata.var.get('gene_name', adata.var_names), dtype=str)
    valid = np.asarray([s in vocab and not s.startswith('<') for s in symbols])
    if 'feature_type' in adata.var:
        valid &= adata.var['feature_type'].astype(str).to_numpy() == 'Gene Expression'
    columns = np.flatnonzero(valid)
    ids = np.asarray([vocab[symbols[j]] for j in columns], dtype=np.int64)
    unique, inverse = np.unique(ids, return_inverse=True)
    if len(unique) < 50:
        raise ValueError('Fewer than 50 measured genes map to the pretrained vocabulary')
    # Duplicate symbols are summed as raw counts; aliases are not guessed.
    projector = sparse.csr_matrix((np.ones(len(columns), dtype=np.int64),
                                  (columns, inverse)), shape=(adata.n_vars, len(unique)))
    return unique, projector, dict(input_features=adata.n_vars,
        mapped_features=len(columns), mapped_unique_genes=len(unique),
        duplicate_mapped_columns_summed=len(columns)-len(unique),
        symbol_mapping='exact gene_name to checkpoint vocabulary; unknown excluded')


def pack_totals(raw, gene_ids, rows, model, max_tokens, seed):
    # The reused helper sums its two arguments. Zero is only its unused additive
    # argument; it is NOT a fabricated biological compartment or model input.
    packed = [_encode_counts(raw[i], np.zeros(len(gene_ids), dtype=np.int64), gene_ids,
        pad_token_id=model.pad_id, cls_token_id=model.cls_id, max_tokens=max_tokens,
        seed=seed, slice_seed=0, node=int(row)) for i, row in enumerate(rows)]
    length = max(len(x['whole_gene_ids']) for x in packed)
    genes = np.full((len(rows), length), model.pad_id, dtype=np.int64)
    bins = np.full((len(rows), length), -2., dtype=np.float32)
    valid = np.zeros((len(rows), length), dtype=bool)
    for i, item in enumerate(packed):
        n = len(item['whole_gene_ids'])
        genes[i, :n], bins[i, :n], valid[i, :n] = (
            item['whole_gene_ids'], item['whole_input_bins'], item['whole_token_valid_mask'])
    return [torch.from_numpy(x).cuda() for x in (genes, bins, valid)]


def embedding_qc(values, available):
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite embeddings')
    active = values[np.asarray(available, dtype=bool)].astype(np.float64)
    if not len(active):
        return {'available': 0}
    norm = np.linalg.norm(active, axis=1)
    if (norm <= 1e-8).any():
        raise ValueError('A present view returned an all-zero embedding')
    unit = active / norm[:, None]
    singular = np.linalg.svd(unit - unit.mean(0), compute_uv=False)
    mass = singular**2
    mass /= max(mass.sum(), 1e-30)
    return dict(available=len(active), dimension=values.shape[1],
        norm_median=float(np.median(norm)),
        centered_variance_effective_rank=float(np.exp(-(mass*np.log(mass+1e-30)).sum())),
        mean_offdiagonal_cosine=(float(((unit.sum(0)**2).sum()-len(unit)) /
                                (len(unit)*(len(unit)-1))) if len(unit)>1 else None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--assets-dir', type=Path, required=True,
                        help='Local root with scGPT_human/, code/scgpt/, source_manifest.json')
    parser.add_argument('--lr-path', type=Path, required=True,
                        help='Local curated CSV with the original checkpoint-config SHA256')
    parser.add_argument('--standalone-compute', action='store_true',
                        help='Explicitly allow a standalone CUDA server; forbidden on HPC login nodes')
    parser.add_argument('--expected-step', type=int, default=10234)
    parser.add_argument('--input-mode', choices=['bins', 'compartments'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--centers', type=int, default=1024)
    parser.add_argument('--center-ids', type=Path,
                        help='Optional newline-delimited cell IDs; overrides random center sampling')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--min-genes', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    execution = require_compute_execution(args.standalone_compute)
    if min(args.centers, args.batch_size, args.min_genes) < 1 or args.seed < 0:
        raise ValueError('Invalid pilot size or seed')
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('This pilot requires the confirmed GPU environment')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model, baseline, cfg, step, provenance = load_model(args.checkpoint, args.expected_step,
        assets_dir=args.assets_dir, lr_path=args.lr_path)
    adata = ad.read_h5ad(args.h5ad)
    if not adata.obs_names.is_unique:
        raise ValueError('Input barcodes must be unique')
    for name in ('sample_id', 'batch_id', 'slice_id', 'dataset_id'):
        if name in adata.obs and adata.obs[name].nunique(dropna=False) != 1:
            raise ValueError(f'This pilot accepts one physical section only: {name}')
    gene_ids, projector, mapping = map_genes(adata, model.backbone.vocab)
    raw_total = checked_counts(adata.X, 'X')
    total = (raw_total @ projector).tocsr()
    total.eliminate_zeros()
    nuclear = cyto = None
    if args.input_mode == 'compartments':
        if not {'nuclear_counts', 'cytoplasmic_counts'}.issubset(adata.layers):
            raise ValueError('Full V12 requires actual nuclear_counts/cytoplasmic_counts layers')
        if not adata.uns.get('compartment_provenance') or 'spatial_um' not in adata.obsm:
            raise ValueError('Compartment origin and explicit spatial_um coordinates are required')
        original_n = checked_counts(adata.layers['nuclear_counts'], 'nuclear_counts')
        original_c = checked_counts(adata.layers['cytoplasmic_counts'], 'cytoplasmic_counts')
        difference = original_n + original_c - raw_total
        difference.eliminate_zeros()
        if difference.nnz:
            raise ValueError('Nuclear + cytoplasmic counts do not equal whole-cell counts')
        nuclear, cyto = (original_n @ projector).tocsr(), (original_c @ projector).tocsr()
        coords = np.asarray(adata.obsm['spatial_um'], dtype=np.float64)
        coord_contract = dict(units='micrometer', source='input spatial_um',
                              axes='as recorded by compartment preprocessing')
    else:
        coords, coord_contract = bin_coordinates(adata)
    if coords.shape != (adata.n_obs, 2) or not np.isfinite(coords).all():
        raise ValueError('Spatial coordinates must be finite [observations,2]')
    detected = np.asarray(total.getnnz(axis=1)).ravel()
    depth = np.asarray(total.sum(axis=1)).ravel()
    tissue = (adata.obs['in_tissue'].to_numpy() == 1 if 'in_tissue' in adata.obs
              else np.ones(adata.n_obs, dtype=bool))
    candidates = np.flatnonzero(tissue & (detected >= args.min_genes))
    if len(candidates) < 2:
        raise ValueError('Fewer than two eligible observations')
    requested_centers = None
    if args.center_ids is not None:
        ids = [line.strip() for line in args.center_ids.read_text().splitlines() if line.strip()]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError('Center ID file must contain unique, nonempty IDs')
        requested = adata.obs_names.get_indexer(ids)
        if (requested < 0).any():
            examples = [ids[i] for i in np.flatnonzero(requested < 0)[:5]]
            raise ValueError(f'Center IDs missing from input section: {examples}')
        requested_centers = len(requested)
        eligible = np.zeros(adata.n_obs, dtype=bool)
        eligible[candidates] = True
        centers = np.sort(requested[eligible[requested]])
    else:
        centers = np.sort(np.random.default_rng(args.seed).choice(
            candidates, min(args.centers, len(candidates)), replace=False))
    if len(centers) < 2:
        raise ValueError('Fewer than two eligible requested centers')
    settings = cfg['v12']
    k, radius, token_budget = (int(settings['neighbors']),
        float(settings['max_neighbor_distance']), int(settings['max_tokens']))
    _, near = cKDTree(coords[candidates]).query(coords[centers], k=min(k+1, len(candidates)))
    neighbors = np.full((len(centers), k), -1, dtype=np.int64)
    for i, (center, selected) in enumerate(zip(centers, near)):
        selected = candidates[np.atleast_1d(selected)]
        selected = selected[selected != center]
        selected = selected[np.linalg.norm(coords[selected]-coords[center], axis=1) <= radius][:k]
        if len(selected) != len(np.unique(selected)):
            raise ValueError('Duplicate neighbor identity')
        neighbors[i, :len(selected)] = selected
    if (neighbors == centers[:, None]).any():
        raise ValueError('Center identity leaked into neighbors')
    prior = load_lr_prior(cfg, model.backbone.vocab)
    audit = dict(status='INPUTS_VALIDATED_INFERENCE_PENDING', step=step,
        mode=args.input_mode, observation_unit='8um_bin' if nuclear is None else 'segmented_cell',
        input=str(args.h5ad.resolve()), input_sha256=file_sha256(args.h5ad),
        checkpoint_sha256=file_sha256(args.checkpoint), provenance=provenance,
        script_sha256=file_sha256(Path(__file__)), input_shape=list(adata.shape),
        mapping=mapping, coordinates=coord_contract, seed=args.seed,
        total_candidates=len(candidates), selected_centers=len(centers),
        center_selection='requested_cell_ids' if args.center_ids else 'seeded_random_sample',
        requested_center_ids=requested_centers,
        requested_centers_failing_input_qc=(requested_centers-len(centers) if requested_centers is not None else None),
        min_mapped_positive_genes=args.min_genes, neighbors=k, radius_um=radius,
        neighbor_pool='all QC-passing tissue observations, before center subsampling',
        neighbor_count_mean=float((neighbors>=0).sum(1).mean()),
        mapped_umi_fraction=float(total.sum()/max(float(raw_total.sum()), 1)),
        max_tokens=token_budget,
        centers_exceeding_token_budget_fraction=float((detected[centers]>token_budget-1).mean()),
        lr_prior=prior['metadata'], lr_expression='mapped total counts in actual spatial neighbors',
        biological_accuracy_evaluated=False, velocity_evaluated=False,
        full_compartment_model_evaluated=nuclear is not None,
        **execution)
    atomic_json(output/'input_audit.json', audit)
    blocks, availability = {}, {}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, len(centers), args.batch_size):
            selected = centers[start:start+args.batch_size]
            nearby = neighbors[start:start+args.batch_size]
            genes, bins, valid = pack_totals(total[selected].toarray(), gene_ids, selected,
                                            model, token_budget, args.seed)
            base = baseline(genes, bins, ~valid, 0.)[:, 0].float()
            results = {'scgpt_total_cls': base}
            present = {'scgpt_total_cls': np.ones(len(selected), dtype=bool)}
            if nuclear is not None:
                items = []
                for center, row in zip(selected, nearby):
                    nodes = np.r_[center, row[row>=0]]
                    items.append(encode_v12_views(nuclear[nodes].toarray(), cyto[nodes].toarray(),
                        gene_ids, nodes, coords[nodes], pad_token_id=model.pad_id,
                        cls_token_id=model.cls_id, neighbors=k, max_tokens=token_budget,
                        mask=False, seed=args.seed, max_neighbor_distance=radius,
                        lr_pairs=prior['pairs'], lr_weights=prior['weights']))
                batch = {key: value.cuda() for key, value in collate_v12(items).items()
                         if not key.startswith('query_')}
                inferred = model(batch, return_embeddings=True)
                results.update(inferred['embeddings'])
                present.update({key: value.cpu().numpy() for key, value in inferred['mode_available'].items()})
                present['concat'] = present['local'] | present['spatial']
            else:
                results['v12_bin_cls'] = model.encode_sequences(genes, bins, valid)
                present['v12_bin_cls'] = present['scgpt_total_cls']
                rows = np.unique(nearby[nearby>=0])
                encoded = np.zeros((len(rows), model.dim), dtype=np.float32)
                for offset in range(0, len(rows), args.batch_size):
                    part = rows[offset:offset+args.batch_size]
                    tensors = pack_totals(total[part].toarray(), gene_ids, part, model, token_budget, args.seed)
                    encoded[offset:offset+len(part)] = model.encode_sequences(*tensors).float().cpu().numpy()
                features = np.zeros((len(selected), k, model.dim), dtype=np.float32)
                relative = np.zeros((len(selected), k, 2), dtype=np.float32)
                lr = np.zeros((len(selected), k, k), dtype=np.float32)
                active = nearby>=0
                for i, (center, row) in enumerate(zip(selected, nearby)):
                    nodes = row[row>=0]
                    if len(nodes):
                        features[i, :len(nodes)] = encoded[np.searchsorted(rows, nodes)]
                        relative[i, :len(nodes)] = coords[nodes]-coords[center]
                        evidence = compute_lr_features_v6(total[nodes].toarray(), gene_ids,
                            node_indices=nodes, lr_pairs=prior['pairs'], lr_weights=prior['weights'])
                        lr[i, :len(nodes), :len(nodes)] = evidence['lr_bias']
                tensors = [torch.from_numpy(x).cuda() for x in (features, active, relative, lr)]
                results['v12_spatial_bins'] = model.spatial_encoder(*tensors)[0]
                present['v12_spatial_bins'] = active.any(1)
            for key, value in results.items():
                blocks.setdefault(key, []).append(value.float().cpu().numpy())
                availability.setdefault(key, []).append(present[key])
            atomic_json(output/'progress.json', dict(completed=min(start+args.batch_size, len(centers)),
                total=len(centers), elapsed_seconds=time.monotonic()-started))
    embeddings = {name: np.concatenate(parts) for name, parts in blocks.items()}
    availability = {name: np.concatenate(parts) for name, parts in availability.items()}
    qc = {name: embedding_qc(values, availability[name]) for name, values in embeddings.items()}
    result = adata[centers].copy()
    result.obsm['spatial_um'] = coords[centers]
    result.obs['source_row'] = centers
    result.obs['mapped_umi'] = depth[centers]
    result.obs['mapped_detected_genes'] = detected[centers]
    for name, values in embeddings.items():
        result.obsm['X_'+name] = values
        result.obs[name+'_available'] = availability[name]
    result.uns['visium_v12_inference_contract_json'] = json.dumps(audit, sort_keys=True)
    result.write_h5ad(output/'embeddings.h5ad', compression='gzip')
    np.savez_compressed(output/'neighbors.npz', center_rows=centers, neighbor_rows=neighbors)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, len(embeddings), figsize=(4*len(embeddings), 4), squeeze=False)
    for axis, (name, values) in zip(axes[0], embeddings.items()):
        use = availability[name]
        if use.sum() >= 2:
            unit = values[use]/np.maximum(np.linalg.norm(values[use], axis=1, keepdims=True), 1e-8)
            pc = PCA(n_components=1, svd_solver='full').fit_transform(unit)[:, 0]
            artist = axis.scatter(coords[centers[use], 0], coords[centers[use], 1], c=pc, s=5, cmap='coolwarm')
            fig.colorbar(artist, ax=axis, shrink=.65, label='within-readout PC1')
        axis.set(title=name, xlabel='x (um)', ylabel='y (um)', aspect='equal')
        axis.invert_yaxis()
    fig.suptitle('Visium HD transfer pilot | independent PCA colors are not cell-type labels')
    fig.tight_layout()
    fig.savefig(output/'spatial_readouts.png', dpi=180)
    fig.savefig(output/'spatial_readouts.pdf')
    plt.close(fig)
    # Completion marker only follows finite/nonzero checks and output reload.
    reloaded = ad.read_h5ad(output/'embeddings.h5ad', backed='r')
    if reloaded.n_obs != len(centers) or any('X_'+key not in reloaded.obsm for key in embeddings):
        raise ValueError('Written embedding artifact failed verification')
    reloaded.file.close()
    atomic_json(output/'COMPLETE.json', dict(status='INFERENCE_AND_ARTIFACT_QC_COMPLETE',
        biological_accuracy_evaluated=False, full_compartment_model_evaluated=nuclear is not None,
        elapsed_seconds=time.monotonic()-started, embedding_qc=qc,
        artifacts={path.name:file_sha256(path) for path in output.iterdir() if path.is_file()}))
    print(json.dumps(qc, indent=2), flush=True)


if __name__ == '__main__':
    main()
