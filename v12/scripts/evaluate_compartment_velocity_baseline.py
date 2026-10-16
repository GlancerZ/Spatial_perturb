"""Route A: train-only nuclear/cytoplasmic steady-ratio residual baseline.

Run only on an authorized compute node. A residual is a hypothesis-dependent
cytoplasmic direction proxy, not measured velocity, temporal validation, or a
calibrated fate probability. No spatial edge is interpreted as a future state.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
from evaluate_subcellular_v2 import CANONICAL_COHORT, atomic_array, atomic_json, file_sha256
from evaluate_subcellular_v12 import ids_hash, sample_inputs, valid_labels, write_frame


def source_signatures():
    return {name: file_sha256(ROOT / name) for name in (
        'scripts/evaluate_compartment_velocity_baseline.py', 'scripts/evaluate_subcellular_v12.py',
        'scripts/evaluate_subcellular_v2.py')}


def paired_exposure(nuclear, cyto, reference=None):
    """One shared exposure for both compartments; never normalize independently."""
    total = (nuclear + cyto).sum(axis=1)
    if reference is None:
        reference = float(np.median(total[total > 0])) if (total > 0).any() else 1.
    exposure = np.maximum(total, 1) / reference
    return nuclear/exposure[:, None], cyto/exposure[:, None], exposure, reference


def expression_features(nuclear, cyto):
    total = nuclear + cyto
    values = np.log1p(total / np.maximum(total.sum(axis=1, keepdims=True), 1) * 1e4)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def predict_train_types(nuclear, cyto, train, train_labels):
    """Train expression centroids; held-out annotation never enters assignment."""
    features = expression_features(nuclear, cyto)
    labels = np.asarray(train_labels).astype(str)
    train = np.asarray(train, dtype=bool)
    if len(labels) != train.sum():
        raise ValueError('Only training labels may enter assignment')
    classes = np.unique(labels)
    if not len(classes):
        raise ValueError('No training labels')
    centers = np.stack([features[train][labels == name].mean(axis=0) for name in classes])
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    similarities = features @ centers.T
    result = classes[similarities.argmax(axis=1)]
    result[train] = labels
    ordered = np.sort(similarities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2] if len(classes) > 1 else np.ones(len(result))
    return result, classes, centers, margin


def fit_ratios(nuclear, cyto, labels, classes, min_cells=30, min_nonzero=10):
    """Trim joint upper 1% per gene within a training cell type, then OLS at origin.

    No validation arrays/labels are accepted by this function. Unsupported
    genes have zero placeholders plus reliable=False, never fake velocities.
    """
    nuclear, cyto = np.asarray(nuclear, dtype=np.float64), np.asarray(cyto, dtype=np.float64)
    labels = np.asarray(labels).astype(str)
    if nuclear.shape != cyto.shape or nuclear.ndim != 2 or len(nuclear) != len(labels):
        raise ValueError('Mismatched paired fitting data')
    rho = np.zeros((len(classes), nuclear.shape[1]), dtype=np.float64)
    support = np.zeros_like(rho, dtype=np.int64)
    reliable = np.zeros_like(rho, dtype=bool)
    variance = np.zeros_like(rho)
    for index, name in enumerate(classes):
        selected = labels == name
        if selected.sum() < min_cells:
            continue
        n, c = nuclear[selected], cyto[selected]
        keep = n+c <= np.quantile(n+c, .99, axis=0)[None, :]
        positive_n = ((n > 0) & keep).sum(axis=0)
        positive_c = ((c > 0) & keep).sum(axis=0)
        support[index] = ((n+c > 0) & keep).sum(axis=0)
        weight_count = np.maximum(keep.sum(axis=0), 1)
        mean_c = (c*keep).sum(axis=0)/weight_count
        variance[index] = (((c-mean_c)**2)*keep).sum(axis=0)/weight_count
        denominator = (c*c*keep).sum(axis=0)
        estimate = (c*n*keep).sum(axis=0)/np.maximum(denominator, 1e-12)
        okay = ((positive_n >= min_nonzero) & (positive_c >= min_nonzero)
                & (variance[index] > 1e-6) & (denominator > 1e-12)
                & np.isfinite(estimate) & (estimate > 0))
        rho[index] = np.where(okay, estimate, 0.)
        reliable[index] = okay
    return dict(rho=rho, reliable=reliable, support=support, cyto_variance=variance)


def row_similarity(first, second, mask):
    first, second = np.where(mask, first, 0.), np.where(mask, second, 0.)
    denominator = np.linalg.norm(first, axis=1)*np.linalg.norm(second, axis=1)
    return np.divide((first*second).sum(axis=1), denominator,
                     out=np.full(len(first), np.nan), where=denominator > 1e-12)


def finite_mean(values):
    values = np.asarray(values)
    return float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else None


def validate_baseline_completion(report, output):
    if report.get('status') != 'complete' or report.get('cohort_sha256') != CANONICAL_COHORT or report.get('cells') != 5200:
        raise ValueError('Incomplete/noncanonical Route A output')
    if report.get('test_donors_used') is not False or report.get('validation_labels_used_for_fit') is not False:
        raise ValueError('Invalid Route A split protocol')
    if report.get('temporal_direction_validated') is not False:
        raise ValueError('Unsupported temporal validation claim')
    for name, digest in report.get('files_sha256', {}).items():
        if file_sha256(output / name) != digest:
            raise ValueError(f'Route A output integrity mismatch: {name}')
    required = {'cells.parquet', 'genes.npy', 'raw_residual.npy', 'exposure_normalized_residual.npy',
                'reliable_mask.npy', 'gene_reliability.parquet', 'rho_bootstrap.npy', 'qc_strata.parquet',
                'bootstrap_positive_probability.npy', 'bootstrap_negative_probability.npy', 'bootstrap_valid_replicates.npy'}
    if not required.issubset(report.get('files_sha256', {})):
        raise ValueError('Route A missing reproducibility outputs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/benchmarking_data/lungpf/prepared')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--bootstrap-repeats', type=int, default=20)
    parser.add_argument('--thinning-repeats', type=int, default=10)
    parser.add_argument('--min-cells', type=int, default=30)
    parser.add_argument('--min-nonzero', type=int, default=10)
    args = parser.parse_args()
    if not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run Route A inside a compute Slurm step')
    if min(args.bootstrap_repeats, args.thinning_repeats, args.min_cells, args.min_nonzero) < 1:
        parser.error('All support/repeat parameters must be positive')
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        marker = output / 'complete.json'
        if marker.is_file():
            saved = json.loads(marker.read_text())
            expected = dict(bootstrap_repeats=args.bootstrap_repeats, thinning_repeats=args.thinning_repeats,
                            min_cells=args.min_cells, min_nonzero=args.min_nonzero)
            if saved.get('settings') != expected or saved.get('source_sha256') != file_sha256(Path(__file__)):
                raise ValueError('Existing Route A result uses another configuration/source')
            if saved.get('source_signatures') != source_signatures():
                raise ValueError('Existing Route A result helper source has changed')
            if saved.get('input_prepared') != str(args.prepared.resolve()) or saved.get('protocol_sha256') != file_sha256(args.prepared / 'protocol.json'):
                raise ValueError('Existing Route A result uses another input/protocol')
            current = {sample: sample_inputs(args.prepared / sample) for sample in saved.get('input_signatures', {})}
            if not current or current != saved.get('input_signatures'):
                raise ValueError('Existing Route A result input signatures have changed')
            validate_baseline_completion(saved, output)
            print('[route-a] verified existing completed result', flush=True)
            return
        raise FileExistsError('Refusing ambiguous or partial Route A output; inspect before rerun')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    protocol_path = args.prepared / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    entries, panels, signatures = [], [], {}
    for sample in protocol['samples']:
        matches = [donor for donor in protocol['donor_split'] if sample.startswith(donor)]
        if len(matches) != 1:
            raise ValueError('Ambiguous donor')
        donor = matches[0]
        split = protocol['donor_split'][donor]
        if split == 'test':
            continue
        if split not in ('train', 'val'):
            raise ValueError('Invalid split')
        directory = args.prepared / sample
        obs = pd.read_parquet(directory / 'obs.parquet')
        if not obs.patient.eq(donor).all() or ('split' in obs and not obs['split'].eq(split).all()):
            raise ValueError('Protocol/observation split mismatch')
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400, len(obs)), replace=False))
        chosen = obs.iloc[selected].copy(); chosen['split'] = split
        panel = np.load(directory / 'genes.npy', allow_pickle=False).astype(str)
        if len(np.unique(panel)) != len(panel):
            raise ValueError('Duplicate panel genes')
        entries.append((directory, chosen, selected, panel))
        panels.append(set(panel)); signatures[sample] = sample_inputs(directory)
    genes = np.asarray(sorted(set.intersection(*panels)))
    if not len(genes):
        raise ValueError('No commonly measured genes; do not impute absent genes to zero')
    frames, n_blocks, c_blocks = [], [], []
    for directory, chosen, selected, panel in entries:
        lookup = {name: index for index, name in enumerate(panel)}
        columns = [lookup[name] for name in genes]
        matrices = []
        for filename in ('nuclear.npy', 'cytoplasmic.npy'):
            matrix = np.load(directory / filename, mmap_mode='r', allow_pickle=False)
            values = np.asarray(matrix[np.ix_(selected, columns)], dtype=np.float64)
            if not np.isfinite(values).all() or (values < 0).any() or not np.equal(values, np.round(values)).all():
                raise ValueError('Route A requires raw paired integer counts')
            matrices.append(values)
        n_blocks.append(matrices[0]); c_blocks.append(matrices[1]); frames.append(chosen)
    obs = pd.concat(frames, ignore_index=True)
    if len(obs) != 5200 or ids_hash(obs.full_cell_id) != CANONICAL_COHORT or not obs.full_cell_id.is_unique:
        raise ValueError('Route A does not match canonical 5200 cohort')
    train = obs['split'].eq('train').to_numpy()
    if obs.loc[train, 'patient'].nunique() != 10 or obs.loc[~train, 'patient'].nunique() != 3:
        raise ValueError('Expected train10/val3 donor split')
    if not valid_labels(obs.loc[train, 'final_CT']).all():
        raise ValueError('Training cells require known labels for within-type fits')
    nuclear, cyto = np.concatenate(n_blocks), np.concatenate(c_blocks)
    # Exposure reference is fit using training cells only, shared N/C factor.
    _, _, _, reference = paired_exposure(nuclear[train], cyto[train])
    n, c, exposure, _ = paired_exposure(nuclear, cyto, reference)
    assigned, classes, centers, margin = predict_train_types(nuclear, cyto, train, obs.loc[train, 'final_CT'].to_numpy())
    labels = assigned[train]
    fit = fit_ratios(n[train], c[train], labels, classes, args.min_cells, args.min_nonzero)
    index = np.array([{name: i for i, name in enumerate(classes)}[name] for name in assigned])
    rho = fit['rho'][index]
    reliable = fit['reliable'][index]
    residual = n - rho*c
    raw_residual = nuclear - rho*cyto
    residual[~reliable], raw_residual[~reliable] = np.nan, np.nan
    rng = np.random.default_rng(42)
    donors = obs.loc[train, 'patient'].astype(str).to_numpy()
    donor_names = np.unique(donors)
    positive = np.zeros(n.shape, dtype=np.int32); negative = positive.copy(); valid = positive.copy()
    boot_rho, boot_valid = [], []
    for repeat in range(args.bootstrap_repeats):
        sampled = rng.choice(donor_names, size=len(donor_names), replace=True)
        rows = np.concatenate([np.flatnonzero(donors == donor) for donor in sampled])
        boot = fit_ratios(n[train][rows], c[train][rows], labels[rows], classes, args.min_cells, args.min_nonzero)
        active = reliable & boot['reliable'][index]
        values = n - boot['rho'][index]*c
        # A small fixed zero band avoids claiming confidence for exact balance.
        positive += active & (values > .1)
        negative += active & (values < -.1)
        valid += active
        boot_rho.append(boot['rho']); boot_valid.append(boot['reliable'])
    pos_probability = np.divide(positive, valid, out=np.full(n.shape, np.nan), where=valid > 0)
    neg_probability = np.divide(negative, valid, out=np.full(n.shape, np.nan), where=valid > 0)
    stable = reliable & (valid >= np.ceil(.8*args.bootstrap_repeats)) & (np.maximum(pos_probability, neg_probability) >= .95)
    thinning_cosines, thinning_agreement = [], []
    for repeat in range(args.thinning_repeats):
        thinned_n = rng.binomial(nuclear.astype(np.int64), .5).astype(np.float64)*2
        thinned_c = rng.binomial(cyto.astype(np.int64), .5).astype(np.float64)*2
        # Hold the original exposure and fitted rho fixed to isolate count noise.
        perturbed = thinned_n/exposure[:, None] - rho*thinned_c/exposure[:, None]
        thinning_cosines.append(row_similarity(np.nan_to_num(residual), perturbed, reliable))
        considered = reliable & (np.abs(np.nan_to_num(residual)) > .1)
        agree = considered & (np.sign(perturbed) == np.sign(np.nan_to_num(residual)))
        thinning_agreement.append(np.divide(agree.sum(1), considered.sum(1),
            out=np.full(len(obs), np.nan), where=considered.sum(1) > 0))
    def replicate_means(values):
        matrix = np.asarray(values)
        count = np.isfinite(matrix).sum(0)
        return np.divide(np.nansum(matrix, axis=0), count, out=np.full(matrix.shape[1], np.nan), where=count > 0)
    obs['fit_cell_type'] = assigned
    obs['fit_cell_type_source'] = np.where(train, 'training_annotation', 'train_expression_centroid_prediction')
    obs['assignment_cosine_margin'] = margin
    obs['paired_exposure'] = exposure
    obs['nuclear_library_count'] = nuclear.sum(1)
    obs['cytoplasmic_library_count'] = cyto.sum(1)
    obs['nuclear_fraction'] = nuclear.sum(1)/np.maximum((nuclear+cyto).sum(1), 1)
    obs['reliable_genes'] = reliable.sum(1)
    obs['bootstrap_stable_sign_genes'] = stable.sum(1)
    obs['thinning_mean_cosine'] = replicate_means(thinning_cosines)
    obs['thinning_mean_sign_agreement'] = replicate_means(thinning_agreement)
    obs['normalized_residual_l2'] = np.linalg.norm(np.nan_to_num(residual), axis=1)
    write_frame(output / 'cells.parquet', obs)
    arrays = dict(genes=genes, raw_residual=raw_residual.astype(np.float32), exposure_normalized_residual=residual.astype(np.float32),
        reliable_mask=reliable, bootstrap_positive_probability=pos_probability.astype(np.float32),
        bootstrap_negative_probability=neg_probability.astype(np.float32), bootstrap_valid_replicates=valid,
        bootstrap_stable_sign_mask=stable, rho=fit['rho'], rho_bootstrap=np.stack(boot_rho),
        rho_bootstrap_valid=np.stack(boot_valid), assignment_classes=np.asarray(classes), assignment_centroids=centers,
        thinning_cosines=np.asarray(thinning_cosines), thinning_sign_agreement=np.asarray(thinning_agreement))
    for name, values in arrays.items():
        atomic_array(output / f'{name}.npy', values)
    reliability_rows = []
    boot_values, boot_available = np.stack(boot_rho), np.stack(boot_valid)
    for class_index, name in enumerate(classes):
        for gene_index, gene in enumerate(genes):
            values = boot_values[:, class_index, gene_index][boot_available[:, class_index, gene_index]]
            reliability_rows.append(dict(cell_type=name, gene=gene, rho=fit['rho'][class_index, gene_index],
                reliable=bool(fit['reliable'][class_index, gene_index]), support=int(fit['support'][class_index, gene_index]),
                cyto_variance=float(fit['cyto_variance'][class_index, gene_index]), bootstrap_valid=len(values),
                rho_q025=float(np.quantile(values, .025)) if len(values) else None,
                rho_q975=float(np.quantile(values, .975)) if len(values) else None,
                training_cells=int((labels == name).sum()), training_donors=int(len(np.unique(donors[labels == name])))))
    write_frame(output / 'gene_reliability.parquet', pd.DataFrame(reliability_rows))
    qc = []
    for (split, donor, label), group in obs.groupby(['split', 'patient', 'fit_cell_type'], sort=True):
        qc.append(dict(split=split, donor=donor, fit_cell_type=label, cells=len(group),
            **{name: finite_mean(group[name]) for name in ('nuclear_library_count', 'cytoplasmic_library_count',
                'nuclear_fraction', 'reliable_genes', 'bootstrap_stable_sign_genes', 'thinning_mean_cosine',
                'thinning_mean_sign_agreement', 'assignment_cosine_margin', 'normalized_residual_l2')}))
    write_frame(output / 'qc_strata.parquet', pd.DataFrame(qc))
    report = dict(status='complete', route='A_paired_compartment_steady_ratio', cells=len(obs),
        cohort_sha256=ids_hash(obs.full_cell_id), common_measured_genes=len(genes), test_donors_used=False,
        validation_labels_used_for_fit=False, temporal_direction_validated=False, future_state_transitions_computed=False,
        settings=dict(bootstrap_repeats=args.bootstrap_repeats, thinning_repeats=args.thinning_repeats,
                      min_cells=args.min_cells, min_nonzero=args.min_nonzero),
        source_sha256=file_sha256(Path(__file__)), source_signatures=source_signatures(),
        protocol_sha256=file_sha256(protocol_path), input_signatures=signatures,
        input_prepared=str(args.prepared.resolve()), exposure_reference_train_median=reference,
        ratio_fit='Within training cell type, trim upper 1% joint counts by gene; shared-exposure through-origin N≈rho*C',
        heldout_assignment='Nearest train expression centroid in L2 log-normalized whole-cell gene space; no validation annotation',
        residual_definition='r=N-rho*C; cytoplasmic dC/dt divided by k only under two-compartment steady-ratio assumptions',
        time_scale='k=1 is a convention, not measured absolute or shared cross-gene kinetics',
        bootstrap='20 by default; resample training DONORS with replacement; hold exposure and evaluation cells fixed',
        thinning='10 by default; independent binomial p=.5 per compartment, scale 2; hold rho and original paired exposure fixed',
        reliable_cell_gene_pairs=int(reliable.sum()), stable_sign_cell_gene_pairs=int(stable.sum()),
        limitations=['Snapshot residual does not identify transcription/export/degradation rates.',
            'Nucleus is not unspliced RNA; nuclear retention, segmentation and capture differences can dominate.',
            'Bootstrap and thinning measure robustness, not true temporal direction.',
            'Cell types delimit fits; no transition between unrelated lineages is fabricated.',
            'No state-transition probabilities until direction and lineage assumptions are validated.'],
        seconds=time.monotonic()-started)
    report['files_sha256'] = {path.name: file_sha256(path) for path in output.iterdir() if path.is_file()}
    validate_baseline_completion(report, output)
    atomic_json(output / 'complete.json', report)
    print(json.dumps({key: report[key] for key in ('status', 'cells', 'common_measured_genes',
        'reliable_cell_gene_pairs', 'stable_sign_cell_gene_pairs', 'temporal_direction_validated', 'seconds')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
