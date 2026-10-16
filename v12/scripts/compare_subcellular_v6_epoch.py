"""Compare final A/B checkpoints only after both full epochs and evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from run_subcellular_v6 import ARCHITECTURE, Runner

REQUIRED_SOURCE_FILES = (
    'scripts/train_subcellular_v6.py',
    'src/spatial_pretrain/data/subcellular_v6.py',
    'src/spatial_pretrain/data/subcellular_v5.py',
    'src/spatial_pretrain/data/subcellular_v3.py',
    'src/spatial_pretrain/models/subcellular_v6.py',
    'src/spatial_pretrain/models/subcellular_v5.py',
    'src/spatial_pretrain/models/subcellular_v4.py',
    'src/spatial_pretrain/training/subcellular_v4.py',
    'src/spatial_pretrain/training/subcellular_v6_optimizer.py',
    'src/spatial_pretrain/models/scgpt_backbone.py',
    'src/spatial_pretrain/models/subcellular_v3.py',
    'src/spatial_pretrain/models/subcellular_v2.py',
    'src/spatial_pretrain/data/subcellular_v2.py',
)


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def finite_number(value, minimum=0., maximum=math.inf):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and minimum <= value <= maximum)


def valid_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def verify_schedule(cfg, schedule, progress, fingerprint, route):
    """Bind completion to the current one-epoch schedule, not a status label."""
    training, evaluation = cfg['training'], cfg['evaluation']
    step, centers = schedule.get('total_updates'), schedule.get('selected_centers')
    batch = training.get('global_batch_size')
    if (not all(positive_integer(value) for value in (step, centers, batch))
            or training.get('epochs') != 1 or isinstance(training.get('epochs'), bool)
            or training.get('checkpoint_every') != 100 or evaluation.get('interval') != 100
            or evaluation.get('cells_per_donor') != 400
            or evaluation.get('mask_validation_batches') != 16
            or evaluation.get('test_donors') is not False
            or evaluation.get('fixed_C_cell_type') != .1 or evaluation.get('fixed_C_region') != .01
            or schedule.get('updates_per_epoch') != step
            or step != (centers + batch - 1) // batch
            or schedule.get('global_batch_size') != batch
            or schedule.get('selected_center_stride') != cfg['data']['center_stride']
            or schedule.get('input_binning') != cfg['protocol']['bins']
            or schedule.get('loss') != cfg['loss']
            or schedule.get('backbone_trainable_layers') != 0
            or schedule.get('model_version') != 6 or schedule.get('architecture') != ARCHITECTURE
            or schedule.get('arch_config', {}).get('route') != route
            or progress.get('model_version') != 6 or progress.get('architecture') != ARCHITECTURE
            or progress.get('config_sha256') != fingerprint
            or progress.get('complete') is not True or progress.get('step') != step
            or progress.get('epoch') != 1 or progress.get('next_update') != 0):
        raise ValueError(f'Invalid completed one-epoch schedule/progress for route {route}')
    sources = schedule.get('source_sha256')
    if (not isinstance(sources, dict) or set(sources) != set(REQUIRED_SOURCE_FILES)
            or not all(isinstance(name, str) and valid_digest(digest) for name, digest in sources.items())):
        raise ValueError('Schedule must contain the training/model/data source fingerprints')
    if (not isinstance(schedule.get('optimizer_manifest'), dict) or not schedule['optimizer_manifest']
            or schedule.get('lr_prior', {}).get('sha256') != cfg['lr']['sha256']):
        raise ValueError('Schedule is missing the matching optimizer or LR prior')
    backbone = schedule.get('backbone_provenance', {})
    if (backbone.get('backbone_kind') != 'scgpt'
            or not all(valid_digest(backbone.get(key)) for key in
                       ('weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'))):
        raise ValueError('Schedule is missing the verified ordinary scGPT provenance')
    return step


def verify_report_identity(report, schedule, fingerprint):
    """Require evaluator evidence to agree with this run's current schedule."""
    for key in ('source_sha256', 'arch_config', 'optimizer_manifest', 'lr_prior'):
        if report.get(key) != schedule[key]:
            raise ValueError(f'Evaluation differs from current schedule: {key}')
    provenance = report.get('backbone_provenance', {})
    if (provenance.get('backbone_kind') != 'scgpt'
            or provenance.get('config_sha256') != fingerprint
            or provenance.get('model_version') != 6 or provenance.get('architecture') != ARCHITECTURE
            or provenance.get('arch_config') != schedule['arch_config']
            or provenance.get('checkpoint_source_sha256') != schedule['source_sha256']
            or provenance.get('checkpoint_sources_verified') is not True
            or provenance.get('checkpoint_provenance_verified') is not True
            or provenance.get('frozen_backbone_weights_verified') is not True
            or provenance.get('checkpoint_backbone_provenance') != schedule['backbone_provenance']
            or provenance.get('lr_prior') != schedule['lr_prior']
            or provenance.get('optimizer_manifest') != schedule['optimizer_manifest']):
        raise ValueError('Missing or incompatible frozen-backbone/source verification')
    loaded = provenance.get('loaded_pretrained_backbone', {})
    for key in ('backbone_kind', 'weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'):
        if loaded.get(key) != schedule['backbone_provenance'][key]:
            raise ValueError(f'Ordinary scGPT provenance differs from current schedule: {key}')
    for name, key in (('src/spatial_pretrain/models/subcellular_v6.py', 'evaluation_model_source_sha256'),
                      ('src/spatial_pretrain/data/subcellular_v6.py', 'evaluation_data_source_sha256'),
                      ('src/spatial_pretrain/training/subcellular_v4.py', 'evaluation_loss_source_sha256')):
        if name not in schedule['source_sha256'] or provenance.get(key) != schedule['source_sha256'][name]:
            raise ValueError(f'Evaluation implementation differs from scheduled source: {name}')
    for key in ('cohort_labels_split_sha256', 'protocol_sha256', 'vocab_mapping_sha256'):
        if not valid_digest(report.get(key)):
            raise ValueError(f'Missing evaluation protocol identity: {key}')
    train, val = report.get('train_donors'), report.get('val_donors')
    if (not isinstance(train, list) or not isinstance(val, list) or not train or not val
            or not all(isinstance(donor, str) and donor for donor in train + val)
            or len(set(train + val)) != len(train + val)
            or len(train + val) != 13 or report.get('cells_per_donor') != 400):
        raise ValueError('Invalid fixed5200 training/validation donor split')
    masked = report['masked_validation']
    centers, total = masked.get('centers'), masked.get('total_validation_centers')
    batches, size = masked.get('effective_batches'), report.get('evaluation_batch_size')
    slices = masked.get('slices_covered')
    if (not all(positive_integer(value) for value in (centers, total, batches, size, slices))
            or not slices <= centers <= total
            or centers != min(total, max(slices, 16 * size))
            or batches != (centers + size - 1) // size
            or not valid_digest(masked.get('sample_indices_sha256'))
            or not valid_digest(masked.get('query_targets_sha256'))
            or masked.get('neighbor_shuffle_applied') is not False):
        raise ValueError('Incomplete fixed masked-validation evidence')
    losses = masked.get('loss_mean', {})
    if (not finite_number(losses.get('total')) or not finite_number(losses.get('masked_ce'))
            or abs(losses['total'] - losses['masked_ce']) > 1e-8):
        raise ValueError('Invalid single-loss masked-validation result')


def verify_stable_evaluation(initial, report):
    for key in ('cohort_sha256', 'cohort_labels_split_sha256', 'protocol_sha256',
                'vocab_mapping_sha256', 'embedding_dimensions', 'train_donors', 'val_donors',
                'context_cells', 'whole_cell_protocol', 'binning_protocol', 'graph', 'seed',
                'neighbor_shuffle_seed'):
        if report.get(key) != initial.get(key):
            raise ValueError(f'Evaluation protocol changed during the epoch: {key}')
    for key in ('centers', 'total_validation_centers', 'slices_covered',
                'sample_indices_sha256', 'query_targets_sha256', 'binning_protocol'):
        if report['masked_validation'].get(key) != initial['masked_validation'].get(key):
            raise ValueError(f'Masked-validation cohort/targets changed during the epoch: {key}')
    first, current = fixed_rows(initial), fixed_rows(report)
    initial_counts = {(row['embedding'], row['task']): (row['train_cells'], row['val_cells'])
                      for row in initial['results']}
    for row in report['results']:
        key = row['embedding'], row['task']
        if ((row['train_cells'], row['val_cells']) != initial_counts[key]
                or set(current[key]['per_donor_macro_f1']) != set(first[key]['per_donor_macro_f1'])
                or not set(current[key]['per_donor_macro_f1']).issubset(report['val_donors'])):
            raise ValueError('Probe cells or validation donors changed during the epoch')
    for task in ('cell_type', 'region'):
        baseline, candidate = first['base', task], current['base', task]
        if (abs(candidate['val_macro_f1'] - baseline['val_macro_f1']) > 1e-8
                or any(abs(candidate['per_donor_macro_f1'][donor] - value) > 1e-8
                       for donor, value in baseline['per_donor_macro_f1'].items())):
            raise ValueError('The paired frozen plain scGPT baseline changed during the epoch')


def read_run(config_path, expected_route):
    cfg = yaml.safe_load(config_path.read_text())
    if cfg['model']['route'] != expected_route or cfg['training']['epochs'] != 1:
        raise ValueError('Expected the corresponding one-epoch A/B configuration')
    run = Path(cfg['output_dir'])
    schedule = json.loads((run / 'schedule.json').read_text())
    progress = json.loads((run / 'progress.json').read_text())
    status = json.loads(run.with_name(run.name + '.controller').joinpath('status.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    step = verify_schedule(cfg, schedule, progress, fingerprint, expected_route)
    if (not (run / 'TRAIN_COMPLETE').is_file()
            or status.get('status') not in ('complete', 'training_complete')
            or status.get('step') != step or status.get('total_updates') != step):
        raise ValueError(f'{expected_route} has not completed its epoch')
    # A training-only controller deliberately does not certify evaluations. Its
    # pending list may be stale after a separate evaluation worker finishes.
    # Independently inspect every agreed point, including an off-interval end.
    required = sorted({0, step} | set(range(100, step + 1, 100)))
    validator = SimpleNamespace(run=run, cfg=cfg, fingerprint=fingerprint, checkpoint_digests={})
    initial = report = None
    for saved_step in required:
        if not Runner.evaluated(validator, saved_step):
            raise ValueError(f'Missing required evaluation for route {expected_route} at step {saved_step}')
        metrics = run / 'downstream_validation' / f'step_{saved_step:08d}' / 'metrics.json'
        report = json.loads(metrics.read_text())
        verify_report_identity(report, schedule, fingerprint)
        fixed_rows(report)
        if initial is None:
            initial = report
        verify_stable_evaluation(initial, report)
    if (initial.get('anchor_comparison', {}).get('step_zero_identity_verified') is not True
            or initial['backbone_provenance'].get('ordinary_scgpt_initial_weights_verified') is not True):
        raise ValueError('Paired ordinary scGPT baseline was not verified at initialization')
    return cfg, schedule, report, initial


def fixed_rows(report):
    rows = {}
    for row in report['results']:
        fixed = row['fixed_C_result']
        if row['task'] not in ('cell_type', 'region') or fixed['C'] != (.1 if row['task'] == 'cell_type' else .01):
            raise ValueError('Mismatched probe regularization')
        value = fixed['val_macro_f1']
        if not finite_number(value, maximum=1.):
            raise ValueError('Invalid/nonfinite probe score')
        solver = fixed.get('solver', {})
        if (solver.get('converged_gradient') is not True
                or not finite_number(solver.get('gradient_max'), maximum=1e-4)
                or not finite_number(solver.get('objective'))):
            raise ValueError('Fixed-C probe has no valid convergence evidence')
        donors = fixed.get('per_donor_macro_f1')
        if (not isinstance(donors, dict) or not donors
                or not all(finite_number(score, maximum=1.) for score in donors.values())
                or not positive_integer(row.get('train_cells')) or not positive_integer(row.get('val_cells'))
                or row['train_cells'] + row['val_cells'] > 5200):
            raise ValueError('Invalid fixed-C donor scores or labeled-cell counts')
        key = row['embedding'], row['task']
        if key in rows:
            raise ValueError('Duplicate probe result')
        rows[key] = fixed
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-a', type=Path, required=True)
    parser.add_argument('--config-b', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run project analysis inside a verified compute step')
    a_cfg, a_schedule, a, a0 = read_run(args.config_a, 'A')
    b_cfg, b_schedule, b, b0 = read_run(args.config_b, 'B')
    def common(cfg):
        cfg = json.loads(json.dumps(cfg))
        for key in ('experiment', 'output_dir'):
            cfg.pop(key)
        cfg['model'].pop('route')
        return cfg
    if common(a_cfg) != common(b_cfg):
        raise ValueError('A/B differ beyond route and output identity; do not report a controlled comparison')
    for key in ('selected_centers', 'updates_per_epoch', 'total_updates', 'global_batch_size'):
        if a_schedule[key] != b_schedule[key]:
            raise ValueError(f'Training budget differs: {key}')
    for key in ('cohort_sha256', 'cohort_labels_split_sha256', 'embedding_dimensions', 'neighbor_shuffle_seed'):
        if a[key] != b[key]:
            raise ValueError(f'Evaluation protocol differs: {key}')
    for report in (b, a0, b0):
        if not a.get('source_sha256') or report.get('source_sha256') != a['source_sha256']:
            raise ValueError('Training/model/data code differs between routes or checkpoints')
        reference = a['backbone_provenance']['loaded_pretrained_backbone']
        other = report['backbone_provenance']['loaded_pretrained_backbone']
        for key in ('backbone_kind', 'weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'):
            if not reference.get(key) or other.get(key) != reference[key]:
                raise ValueError(f'Ordinary scGPT provenance differs between routes/checkpoints: {key}')
    rows_a, rows_b, rows_a0, rows_b0 = map(fixed_rows, (a, b, a0, b0))
    if set(rows_a) != set(rows_b) or len(rows_a) != 20:
        raise ValueError('Require the same seven readouts and three shuffle controls for both tasks')
    comparisons = []
    for key in sorted(rows_a):
        first, second = rows_a[key], rows_b[key]
        reference_key = 'base', key[1]
        baselines = [rows[reference_key]['val_macro_f1'] for rows in (rows_a, rows_b, rows_a0, rows_b0)]
        if max(baselines) - min(baselines) > 1e-8:
            raise ValueError('The paired frozen plain scGPT baseline changed between routes/checkpoints')
        donors_a, donors_b = first['per_donor_macro_f1'], second['per_donor_macro_f1']
        if set(donors_a) != set(donors_b):
            raise ValueError('Validation donors differ')
        comparisons.append(dict(embedding=key[0], task=key[1], plain_scgpt=baselines[0],
            A=first['val_macro_f1'], B=second['val_macro_f1'],
            B_minus_A=second['val_macro_f1']-first['val_macro_f1'],
            A_minus_scgpt=first['val_macro_f1']-baselines[0],
            B_minus_scgpt=second['val_macro_f1']-baselines[0],
            per_donor_B_minus_A={donor: donors_b[donor]-donors_a[donor] for donor in donors_a}))
    report = dict(status='both_epochs_verified', step=a['step'], selected_centers=a_schedule['selected_centers'],
        epoch_definition=a_schedule['epoch_definition'], center_stride=a_cfg['data']['center_stride'],
        cohort_sha256=a['cohort_sha256'], trainable_parameters_A=a_schedule['trainable_parameters'],
        trainable_parameters_B=b_schedule['trainable_parameters'], comparisons=comparisons,
        masked_validation_A=a['masked_validation'], masked_validation_B=b['masked_validation'],
        limitations=['Single training seed; no statistical or causal superiority claim.',
                    'B has an extra effective projection; first comparison is not parameter-matched.',
                    'Final scheduled checkpoints only; no early-checkpoint selection.',
                    'No-LR and randomized-LR independent training ablations are still needed.'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# V6 A/B：各完成一个 epoch 后的比较', '',
             f'固定终点 step {a["step"]}；训练中心 {a_schedule["selected_centers"]}；center_stride={a_cfg["data"]["center_stride"]}。',
             f'可训练参数 A={a_schedule["trainable_parameters"]:,}，B={b_schedule["trainable_parameters"]:,}。', '',
             '| 表征 | 任务 | 普通 scGPT | A | B | B−A |', '|---|---|---:|---:|---:|---:|']
    lines += [f'| {r["embedding"]} | {r["task"]} | {r["plain_scgpt"]:.6f} | {r["A"]:.6f} | {r["B"]:.6f} | {r["B_minus_A"]:+.6f} |'
              for r in comparisons]
    lines += ['', '固定 C 的供体外验证 macro-F1；未使用测试供体。单 seed、B多一层有效投影，'
                   '本轮用于筛选方向，不能据此证明LR机制或统计显著优越。']
    (args.output_dir / 'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    print(args.output_dir, flush=True)


if __name__ == '__main__':
    main()
