"""Compute-only V6 acceptance: regressions, real QA, DDP resume, small probes.

The disposable training run preserves the complete epoch schedule, minibatch,
global batch, data, and model. It stops after two updates and resumes for one.
Probe acceptance uses eight cells per donor; the formal controller separately
evaluates its actual step-zero checkpoint on the full fixed 5,200-cell cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from train_subcellular_v6 import ARCHITECTURE, config_hash, lr_multiplier, validate_config


def write_json(path, payload):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def optimizer_state_fingerprint(value):
    """Hash state values exactly, independent of CPU/GPU tensor placement."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            digest.update(f'tensor:{item.dtype}:{tuple(item.shape)}:'.encode())
            digest.update(item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b'dict:')
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(f'{type(item).__name__}:{len(item)}:'.encode())
            for member in item:
                visit(member)
        else:
            digest.update(f'{type(item).__name__}:{item!r};'.encode())

    visit(value)
    return digest.hexdigest()


def resolve_topology(nodes, gpus_per_node=None, training_gpus=None):
    """Resolve the explicit launch shape; old --training-gpus is an assertion."""
    for name, value in (('nodes', nodes), ('gpus-per-node', gpus_per_node),
                        ('training-gpus', training_gpus)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    if gpus_per_node is None:
        total = 4 if training_gpus is None else training_gpus
        if total % nodes:
            raise ValueError('training-gpus must divide evenly across nodes')
        gpus_per_node = total // nodes
    world = nodes * gpus_per_node
    if nodes > 4 or gpus_per_node > 4 or world > 4:
        raise ValueError('The V6 launcher supports at most four GPUs in total')
    if training_gpus is not None and world != training_gpus:
        raise ValueError('training-gpus must equal nodes * gpus-per-node')
    return nodes, gpus_per_node, world


def training_command(config_path, nodes, gpus_per_node):
    """Use the allocation-aware launcher for both local and cross-node DDP."""
    return ['bash', str(ROOT / 'scripts/launch_subcellular_v6_distributed.sh'),
            str(config_path), str(nodes), str(gpus_per_node)]


def validate_rank_topology(rows, nodes, gpus_per_node):
    """Verify observed ranks really span the requested nodes and local GPUs."""
    world = nodes * gpus_per_node
    if not isinstance(rows, list) or len(rows) != world:
        raise AssertionError('Trainer did not report one observed topology entry per rank')
    hosts, ranks = {}, []
    for row in rows:
        if (not isinstance(row, dict)
                or type(row.get('rank')) is not int or type(row.get('local_rank')) is not int
                or not isinstance(row.get('hostname'), str) or not row['hostname']
                or not isinstance(row.get('visible_devices'), str)
                or not isinstance(row.get('gpu_name'), str) or not row['gpu_name']):
            raise AssertionError('Invalid observed DDP rank topology')
        ranks.append(row['rank'])
        hosts.setdefault(row['hostname'], []).append(row['local_rank'])
    if (sorted(ranks) != list(range(world)) or len(hosts) != nodes
            or any(sorted(local) != list(range(gpus_per_node)) for local in hosts.values())):
        raise AssertionError('Observed rank hosts/local GPU ranks do not match requested topology')
    return dict(world_size=world, nodes=nodes, gpus_per_node=gpus_per_node,
                hosts={name: sorted(local) for name, local in sorted(hosts.items())},
                ranks=rows)


def inspect_optimizer_state(optimizer, cfg, expected_step, total_updates, expected_manifest,
                            *, require_nonzero=True):
    """Check both Muon and auxiliary AdamW buffers and their scheduled LRs."""
    if (optimizer.get('schema') != 'subcellular_v6_muon_v1'
            or optimizer.get('steps_completed') != expected_step
            or not expected_manifest or optimizer.get('manifest') != expected_manifest):
        raise AssertionError('Resume checkpoint lacks the matching Muon schema, manifest or step counter')
    factor = lr_multiplier(expected_step, total_updates, cfg['training'])
    expected_lrs = {'muon': cfg['training']['lr_muon'] * factor,
                    'adamw': cfg['training']['lr_new'] * factor}
    summaries = {}
    initialized = optimizer.get('initialized_parameters')
    if not isinstance(initialized, dict) or set(initialized) != {'muon', 'aux_adamw'}:
        raise AssertionError('Muon checkpoint lacks initialized-parameter evidence')
    for key, label, use_muon, base_lr, group_name in (
        ('muon', 'muon', True, cfg['training']['lr_muon'], 'muon_hidden'),
        ('aux_adamw', 'adamw', False, cfg['training']['lr_new'], 'adamw_aux'),
    ):
        branch = optimizer.get(key)
        if not isinstance(branch, dict) or not branch.get('state') or not branch.get('param_groups'):
            raise AssertionError(f'Missing populated {key} optimizer state')
        parameter_ids = []
        parameter_names = []
        for group in branch['param_groups']:
            if (group.get('name') != group_name or group.get('use_muon') is not use_muon
                    or group.get('base_lr') != base_lr
                    or not math.isclose(group['lr'], expected_lrs[label], rel_tol=1e-12, abs_tol=1e-15)
                    or len(group.get('param_names', [])) != len(group['params'])):
                raise AssertionError(f'{key} group identity or complete-epoch LR is inconsistent')
            parameter_ids.extend(group['params'])
            parameter_names.extend(group['param_names'])
        if len(parameter_ids) != len(set(parameter_ids)) or not set(branch['state']).issubset(parameter_ids):
            raise AssertionError(f'{key} state does not match its parameter groups')
        saved_names = [name for identifier, name in zip(parameter_ids, parameter_names)
                       if identifier in branch['state']]
        if initialized[key] != saved_names:
            raise AssertionError(f'{key} initialized-parameter evidence differs from saved buffers')
        nonzero = 0
        steps = []
        moment_keys = ('momentum_buffer',) if use_muon else ('exp_avg', 'exp_avg_sq')
        for value in branch['state'].values():
            for moment_key in moment_keys:
                moment = value.get(moment_key)
                if not torch.is_tensor(moment) or not bool(torch.isfinite(moment).all()):
                    raise AssertionError(f'Invalid saved {key} buffer {moment_key}')
            nonzero += int(bool(value[moment_keys[0]].ne(0).any()))
            if not use_muon:
                step = value.get('step')
                if step is None:
                    raise AssertionError('Auxiliary AdamW state lacks a step counter')
                number = float(step.item() if torch.is_tensor(step) else step)
                if number != expected_step:
                    raise AssertionError(f'Auxiliary AdamW step did not persist across resume: {number}')
                steps.append(int(number))
        if require_nonzero and nonzero == 0:
            raise AssertionError(f'{key} has no nonzero saved momentum/moments')
        summaries[label] = dict(parameter_states=len(branch['state']),
                                nonzero_primary_buffers=nonzero,
                                state_sha256=optimizer_state_fingerprint(branch['state']),
                                step_values=sorted(set(steps)))
    return dict(steps_completed=expected_step, schema=optimizer['schema'],
                manifest=expected_manifest, learning_rates=expected_lrs,
                buffers=summaries, state_sha256=optimizer_state_fingerprint(optimizer))


def optimizer_evidence(path, cfg, expected_step, world, schedule):
    """Inspect saved momentum/moments, not only trainer return/progress text."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state.get('model_version') != 6 or state.get('architecture') != ARCHITECTURE
            or state.get('step') != expected_step or state.get('complete') is not False
            or state.get('epoch') != 0 or state.get('next_update') != expected_step
            or state.get('physical_world_size') != world
            or state.get('config') != cfg or state.get('config_sha256') != config_hash(cfg)
            or state.get('global_batch_size') != cfg['training']['global_batch_size']
            or state.get('micro_batch_size') != cfg['training']['batch_size']
            or state.get('selected_centers') != schedule['selected_centers']):
        raise AssertionError('Saved V6 optimizer boundary or configuration is inconsistent')
    manifest = state.get('optimizer_manifest')
    if manifest != schedule.get('optimizer_manifest'):
        raise AssertionError('Checkpoint and schedule optimizer manifests differ')
    optimizer = inspect_optimizer_state(state.get('optimizer', {}), cfg, expected_step,
                                         schedule['total_updates'], manifest)
    evidence = dict(step=expected_step, optimizer=optimizer, model_version=state['model_version'],
                    architecture=state['architecture'], config_sha256=state['config_sha256'],
                    lr_prior=state.get('lr_prior'), checkpoint_sha256=file_sha256(path))
    del state
    return evidence


def validate_evaluation(metrics, cfg, step, *, masked):
    if (metrics.get('step') != step or metrics.get('evaluation_version') != 6
            or metrics.get('model_version') != 6 or metrics.get('model') != 'SubcellularV6'
            or metrics.get('architecture') != ARCHITECTURE
            or metrics.get('diagnostic_schema_version') != 1
            or metrics.get('test_donors_used') is not False
            or metrics.get('neighbor_shuffle') is not True
            or metrics.get('cells_per_donor') != 8 or metrics.get('canonical_cohort') is not False
            or not metrics.get('results')):
        raise AssertionError('Small-cohort evaluation lacks the required V6 identity/protocol evidence')
    coverage = metrics.get('coverage', [])
    if (not coverage or metrics.get('cells') != sum(row['centers'] for row in coverage)
            or any(row['centers'] != 8 for row in coverage)):
        raise AssertionError('Small-cohort evaluation must contain eight cells per selected sample/donor')
    dim = cfg['model']['slots_per_compartment'] * cfg['model']['slot_dim']
    backbone_dim = int(metrics['arch_config']['backbone_dim'])
    dimensions = {name: backbone_dim if name in ('base', 'intrinsic', 'joint') else dim
                  for name in ('base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
                               'nucleus_local', 'cytoplasm_local')}
    dimensions.update({f'{name}_neighbor_shuffle': dimensions[name]
                       for name in ('joint', 'nucleus', 'cytoplasm')})
    expected = {(name, task) for name in dimensions for task in ('cell_type', 'region')}
    probes = [(row['embedding'], row['task']) for row in metrics['results']]
    if metrics.get('embedding_dimensions') != dimensions or set(probes) != expected or len(probes) != len(expected):
        raise AssertionError('Expected all ten readouts and twenty task/readout probes')
    for row in metrics['results']:
        for result in (row['best'], row['fixed_C_result'], *row['grid']):
            if not math.isfinite(result['val_macro_f1']) or not 0 <= result['val_macro_f1'] <= 1:
                raise AssertionError('Probe result is not a finite F1 score')
    provenance = metrics.get('backbone_provenance', {})
    if (provenance.get('frozen_backbone_weights_verified') is not True
            or provenance.get('checkpoint_sources_verified') is not True
            or provenance.get('checkpoint_provenance_verified') is not True):
        raise AssertionError('Evaluator did not verify frozen scGPT/source/checkpoint identity')
    if step == 0:
        anchor = metrics.get('anchor_comparison', {})
        if (anchor.get('step_zero_identity_verified') is not True
                or anchor.get('joint_minus_base_max_abs') != 0):
            raise AssertionError('Step-zero paired plain-scGPT identity must be exact')
    validation = metrics.get('masked_validation', {})
    if masked and (validation.get('enabled') is not True
                   or validation.get('effective_batches', 0) < 1
                   or validation.get('centers', 0) < 1
                   or set(validation.get('loss_mean', {})) != {'total', 'masked_ce'}
                   or any(not math.isfinite(value) for value in validation['loss_mean'].values())):
        raise AssertionError('Resumed evaluation must execute at least one finite masked-validation batch')
    return dict(step=step, cells=metrics['cells'], cohort_sha256=metrics['cohort_sha256'],
                checkpoint_sha256=metrics['checkpoint_sha256'], readouts=dimensions,
                probe_count=len(probes), anchor_comparison=metrics['anchor_comparison'],
                masked_validation=validation, test_donors_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus-per-node', type=int)
    parser.add_argument('--training-gpus', type=int,
                        help='Compatibility assertion for nodes * gpus-per-node; defaults to total 4')
    args = parser.parse_args()
    nodes, gpus_per_node, world = resolve_topology(args.nodes, args.gpus_per_node, args.training_gpus)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run acceptance in a verified Slurm compute step')
    if not torch.cuda.is_available() or torch.cuda.device_count() < gpus_per_node:
        raise RuntimeError('This compute node must expose the requested GPUs per node')
    cfg = yaml.safe_load(args.config.resolve().read_text())
    validate_config(cfg)
    if cfg['training']['global_batch_size'] % (cfg['training']['batch_size'] * world):
        raise ValueError('GPU topology must divide the unchanged global/minibatch sizes')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    report = dict(status='running', schema='subcellular_v6_acceptance_v1', model_version=6,
                  architecture=ARCHITECTURE, route=cfg['model']['route'],
                  jobid=os.environ['SLURM_JOB_ID'], stepid=os.environ['SLURM_STEP_ID'],
                  hostname=socket.gethostname(), python=sys.executable,
                  config=str(args.config.resolve()), source_config_sha256=config_hash(cfg),
                  torch_version=str(torch.__version__),
                  visible_gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  training_gpus=world, nodes=nodes, gpus_per_node=gpus_per_node, stages=[],
                  full_experiment_started=False, scientific_improvement_claimed=False,
                  full5200_evaluation_deferred_to_formal_controller=True)
    started = time.time()
    child_environment = os.environ.copy()
    child_environment['PYTHONPATH'] = os.pathsep.join(str(path) for path in (ROOT / 'src', ROOT / 'scripts'))
    if os.environ.get('PYTHONPATH'):
        child_environment['PYTHONPATH'] += os.pathsep + os.environ['PYTHONPATH']

    def stage(name, command):
        log_path = output / 'logs' / f'{name}.log'
        entry = dict(name=name, command=command, started_at=time.time(), log=str(log_path), status='running')
        report['stages'].append(entry)
        write_json(output / 'acceptance.json', report)
        print('[v6-acceptance-stage]', name, flush=True)
        with log_path.open('x', buffering=1) as handle:
            completed = subprocess.run(command, cwd=ROOT, env=child_environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
        entry.update(returncode=completed.returncode, elapsed_seconds=time.time() - entry['started_at'],
                     status='passed' if completed.returncode == 0 else 'failed')
        write_json(output / 'acceptance.json', report)
        if completed.returncode:
            raise RuntimeError(f'{name} exited {completed.returncode}; inspect {log_path}')

    try:
        stage('unit_v6_v5_v4_regressions', [sys.executable, '-m', 'unittest', 'discover',
              '-s', 'tests', '-p', 'test_subcellular_v[456]*.py', '-v'])
        unit_log = (output / 'logs' / 'unit_v6_v5_v4_regressions.log').read_text()
        counts = re.findall(r'^Ran (\d+) tests? in ', unit_log, re.MULTILINE)
        required_suites = ('test_subcellular_v6_data.', 'test_subcellular_v6_model.',
                           'test_subcellular_v6_validation.',
                           'test_subcellular_v5_data.', 'test_subcellular_v5_model.',
                           'test_subcellular_v4_model.')
        if not counts or int(counts[-1]) < 1 or not all(name in unit_log for name in required_suites):
            raise AssertionError('Test discovery did not run all required V6/V5/V4 suites')
        report['unit_tests_run'] = int(counts[-1])
        stage('real_cache_single_gpu_qa', [sys.executable, '-u', 'scripts/qa_subcellular_v6.py',
              '--config', str(args.config.resolve()), '--output', str(output / 'real_gpu_qa.json'),
              '--batch-size', str(cfg['training']['batch_size'])])
        qa = json.loads((output / 'real_gpu_qa.json').read_text())
        qa_flags = ('step_zero_identity_verified', 'unchanged_backbone_verified',
                    'target_field_isolation_verified', 'hidden_raw_target_isolation_verified',
                    'official_binning_same_rng_verified', 'optimizer_roundtrip_verified')
        if (qa.get('status') != 'passed' or qa.get('schema') != 'subcellular_v6_real_qa_v1'
                or qa.get('route') != cfg['model']['route']
                or any(qa.get(key) is not True for key in qa_flags)
                or qa.get('batch_size') != cfg['training']['batch_size']
                or [row['step'] for row in qa.get('stages', [])] != [1, 2]):
            raise AssertionError('Real-cache QA lacks complete successful V6 evidence')
        report['real_gpu_qa'] = dict(path=str(output / 'real_gpu_qa.json'),
                                     sha256=file_sha256(output / 'real_gpu_qa.json'),
                                     verified_flags={key: qa[key] for key in qa_flags})
        smoke = json.loads(json.dumps(cfg))
        smoke['output_dir'] = str(output / 'smoke_training')
        for key in ('log_every', 'checkpoint_every', 'resume_every'):
            smoke['training'][key] = 1
        smoke_path = output / 'smoke_config.yaml'
        smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False))
        report['smoke_config_sha256'] = config_hash(smoke)
        report['smoke_changes'] = {'output_dir': smoke['output_dir'],
                                   'training.log_every': 1, 'training.checkpoint_every': 1,
                                   'training.resume_every': 1}
        train = training_command(smoke_path, nodes, gpus_per_node)
        stage('distributed_train_two_updates', train + ['--run-update-limit', '2'])
        run = Path(smoke['output_dir'])
        first = json.loads((run / 'progress.json').read_text())
        schedule = json.loads((run / 'schedule.json').read_text())
        if first.get('step') != 2 or first.get('complete') is not False:
            raise AssertionError('Smoke must save exactly two incomplete updates')
        full_updates = math.ceil(schedule['selected_centers'] / cfg['training']['global_batch_size'])
        if (schedule.get('model_version') != 6 or schedule.get('architecture') != ARCHITECTURE
                or schedule['updates_per_epoch'] != full_updates
                or schedule['total_updates'] != full_updates * cfg['training']['epochs']
                or schedule['total_updates'] <= 3 or schedule.get('initial_step') != 0
                or schedule.get('resumed') is not False):
            raise AssertionError('Smoke changed the complete-epoch schedule')
        report['initial_schedule'] = schedule
        report['initial_rank_topology'] = validate_rank_topology(schedule.get('rank_topology'), nodes, gpus_per_node)
        report['optimizer_boundaries'] = [optimizer_evidence(run / 'resume.pt', smoke, 2, world, schedule)]

        def evaluate(step, dirname, masked):
            directory = output / dirname
            checkpoint = run / 'checkpoints' / f'step_{step:08d}.pt'
            stage(dirname, [sys.executable, '-u', 'scripts/evaluate_subcellular_v6.py',
                  '--config', str(smoke_path), '--checkpoint', str(checkpoint),
                  '--expected-step', str(step), '--output-dir', str(directory),
                  '--batch-size', str(cfg['evaluation']['batch_size']), '--cells-per-donor', '8',
                  '--mask-validation-batches', '1' if masked else '0', '--neighbor-shuffle'])
            metrics = json.loads((directory / 'metrics.json').read_text())
            result = validate_evaluation(metrics, cfg, step, masked=masked)
            result.update(metrics_path=str(directory / 'metrics.json'),
                          metrics_sha256=file_sha256(directory / 'metrics.json'))
            return result

        report['step_zero_evaluation'] = evaluate(0, 'step_zero_evaluation', False)
        stage('distributed_resume_one_update', train + ['--resume', '--run-update-limit', '1'])
        final = json.loads((run / 'progress.json').read_text())
        resumed_schedule = json.loads((run / 'schedule.json').read_text())
        if (final.get('step') != 3 or final.get('complete') is not False
                or resumed_schedule.get('resumed') is not True or resumed_schedule.get('initial_step') != 2
                or resumed_schedule['total_updates'] != schedule['total_updates']
                or resumed_schedule.get('optimizer_resume_state_verified') is not True):
            raise AssertionError('Resume must restore update two and advance once on the same schedule')
        restored_counts = resumed_schedule.get('optimizer_restored_parameter_states')
        report['resumed_rank_topology'] = validate_rank_topology(
            resumed_schedule.get('rank_topology'), nodes, gpus_per_node)
        saved_counts = {name: branch['parameter_states'] for name, branch in
                        report['optimizer_boundaries'][0]['optimizer']['buffers'].items()}
        if restored_counts != saved_counts:
            raise AssertionError('Resume did not verify every saved Muon/AdamW parameter state')
        report['optimizer_restored_parameter_states'] = restored_counts
        report['optimizer_boundaries'].append(optimizer_evidence(run / 'resume.pt', smoke, 3, world, schedule))
        records = [json.loads(line) for line in (run / 'train_metrics.jsonl').read_text().splitlines()]
        if [row['step'] for row in records] != [1, 2, 3]:
            raise AssertionError('Smoke metric history has duplicate/missing updates')
        for row in records:
            factor = lr_multiplier(row['step'], schedule['total_updates'], cfg['training'])
            expected_lrs = {'muon': cfg['training']['lr_muon'] * factor,
                            'adamw': cfg['training']['lr_new'] * factor}
            recorded_lrs = row.get('optimizer_lrs', {})
            if (row.get('model_version') != 6 or row.get('loss_finite') is not True
                    or row.get('gradients_finite') is not True
                    or row['physical_world_size'] != world
                    or row['global_batch_size'] != cfg['training']['global_batch_size']
                    or row['batch_per_card'] != cfg['training']['batch_size']
                    or row['accumulation'] * row['physical_world_size'] * row['batch_per_card'] != row['global_batch_size']
                    or row['total_updates'] != schedule['total_updates']
                    or set(row['loss']) != {'total', 'masked_ce'}
                    or any(not math.isfinite(value) for value in row['loss'].values())
                    or not math.isfinite(row['gradient_norm'])
                    or set(recorded_lrs) != set(expected_lrs)
                    or any(not math.isclose(recorded_lrs[key], value, rel_tol=1e-12, abs_tol=1e-15)
                           for key, value in expected_lrs.items())):
                raise AssertionError('Smoke batch, optimizer schedule, loss or gradient evidence is inconsistent')
        report['smoke_records'] = records
        report['resumed_evaluation'] = evaluate(3, 'smoke_evaluation', True)
        if report['step_zero_evaluation']['cohort_sha256'] != report['resumed_evaluation']['cohort_sha256']:
            raise AssertionError('Step-zero and resumed probes used different cohorts')
        report.update(status='passed', evaluated_cells=report['resumed_evaluation']['cells'],
                      evaluated_readouts=report['resumed_evaluation']['readouts'],
                      strict_step_zero_identity_verified=True, optimizer_resume_verified=True,
                      unchanged_complete_epoch_schedule_verified=True, test_donors_used=False,
                      checkpoint_sha256=report['resumed_evaluation']['checkpoint_sha256'],
                      limitation='Three disposable optimizer updates and small-cohort probes; no efficacy claim')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(elapsed_seconds=time.time() - started, finished_at=time.time())
        write_json(output / 'acceptance.json', report)
    print('[v6-acceptance-passed]', output, flush=True)


if __name__ == '__main__':
    main()
