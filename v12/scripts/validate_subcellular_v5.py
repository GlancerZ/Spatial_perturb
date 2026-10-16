"""Compute-node acceptance: unit/regression tests, real GPU QA, DDP resume, evaluation.

All training is a disposable three-update smoke in a new output directory. This
does not start the scientific experiment or demonstrate downstream improvement.
"""
from __future__ import annotations

import argparse
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


def write_json(path, payload):
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--training-gpus', type=int, choices=(1, 2, 4), default=2)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run acceptance in a verified, live Slurm compute step')
    if not torch.cuda.is_available() or torch.cuda.device_count() < args.training_gpus:
        raise RuntimeError('The selected allocation must expose the requested GPUs')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'logs').mkdir()
    cfg = yaml.safe_load(args.config.resolve().read_text())
    report = dict(status='running', schema='subcellular_v5_acceptance_v1',
                  jobid=os.environ['SLURM_JOB_ID'], stepid=os.environ['SLURM_STEP_ID'],
                  hostname=socket.gethostname(), python=sys.executable,
                  config=str(args.config.resolve()),
                  torch_version=str(torch.__version__),
                  visible_gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                  training_gpus=args.training_gpus, stages=[],
                  full_experiment_started=False, scientific_improvement_claimed=False)
    started = time.time()
    # unittest discovery can import data tests before any script adds src to
    # sys.path. Explicitly expose this checkout to every acceptance subprocess.
    child_environment = os.environ.copy()
    child_environment['PYTHONPATH'] = os.pathsep.join(
        str(path) for path in (ROOT / 'src', ROOT / 'scripts'))
    if os.environ.get('PYTHONPATH'):
        child_environment['PYTHONPATH'] += os.pathsep + os.environ['PYTHONPATH']

    def stage(name, command):
        log_path = output / 'logs' / f'{name}.log'
        entry = dict(name=name, command=command, started_at=time.time(), log=str(log_path), status='running')
        report['stages'].append(entry)
        write_json(output / 'acceptance.json', report)
        print('[acceptance-stage]', name, flush=True)
        with log_path.open('x', buffering=1) as handle:
            completed = subprocess.run(command, cwd=ROOT, env=child_environment,
                                       stdout=handle, stderr=subprocess.STDOUT)
        entry.update(returncode=completed.returncode, elapsed_seconds=time.time() - entry['started_at'],
                     status='passed' if completed.returncode == 0 else 'failed')
        write_json(output / 'acceptance.json', report)
        if completed.returncode:
            raise RuntimeError(f'{name} exited {completed.returncode}; inspect {log_path}')

    try:
        # Current V5 tests and unchanged V3/V4 regressions execute here on the
        # compute node, never in the login-node driver/controller.
        stage('unit_and_v3_regression', [sys.executable, '-m', 'unittest', 'discover',
              '-s', 'tests', '-p', 'test_subcellular_v[345]*.py', '-v'])
        unit_log = (output / 'logs' / 'unit_and_v3_regression.log').read_text()
        counts = re.findall(r'^Ran (\d+) tests? in ', unit_log, re.MULTILINE)
        if not counts or int(counts[-1]) < 130:
            raise AssertionError('Expected V3/V4 regressions plus V5 acceptance tests, at least 130 tests')
        report['unit_tests_run'] = int(counts[-1])
        stage('real_cache_single_gpu_qa', [sys.executable, '-u', 'scripts/qa_subcellular_v5.py',
              '--config', str(args.config.resolve()), '--output', str(output / 'real_gpu_qa.json'),
              '--batch-size', str(cfg['training']['batch_size'])])
        qa = json.loads((output / 'real_gpu_qa.json').read_text())
        if (qa.get('status') != 'passed' or qa.get('step_zero_identity_verified') is not True
                or qa.get('unchanged_backbone_verified') is not True
                or qa.get('batch_size') != cfg['training']['batch_size']):
            raise AssertionError('QA exited without valid successful GPU evidence')
        # Keep all data, masking, batch, optimizer and LR settings. Only output
        # destination and observation frequency differ in the disposable run.
        smoke = json.loads(json.dumps(cfg))
        smoke['output_dir'] = str(output / 'smoke_training')
        for key in ('log_every', 'checkpoint_every', 'resume_every'):
            smoke['training'][key] = 1
        smoke_path = output / 'smoke_config.yaml'
        smoke_path.write_text(yaml.safe_dump(smoke, sort_keys=False))
        train = [sys.executable, '-u']
        if args.training_gpus > 1:
            train += ['-m', 'torch.distributed.run', '--standalone',
                      f'--nproc_per_node={args.training_gpus}']
        train += ['scripts/train_subcellular_v5.py', '--config', str(smoke_path)]
        stage('distributed_train_two_updates', train + ['--run-update-limit', '2'])
        progress_path = Path(smoke['output_dir']) / 'progress.json'
        first = json.loads(progress_path.read_text())
        if first.get('step') != 2 or first.get('complete') is not False:
            raise AssertionError('Smoke training did not save exactly two incomplete optimizer updates')
        # A fresh same-token plain CLS is the actual baseline. Accept V5 only
        # if the complete fixed-cohort initial readout is exactly that baseline.
        baseline_checkpoint = Path(smoke['output_dir']) / 'checkpoints' / 'step_00000000.pt'
        stage('step_zero_paired_plain_cls_evaluation', [sys.executable, '-u', 'scripts/evaluate_subcellular_v5.py',
              '--config', str(smoke_path), '--checkpoint', str(baseline_checkpoint),
              '--expected-step', '0', '--output-dir', str(output / 'step_zero_evaluation'),
              '--batch-size', str(cfg['evaluation']['batch_size']),
              '--cells-per-donor', str(cfg['evaluation']['cells_per_donor']),
              '--mask-validation-batches', '0', '--neighbor-shuffle'])
        baseline = json.loads((output / 'step_zero_evaluation' / 'metrics.json').read_text())
        if baseline.get('anchor_comparison', {}).get('step_zero_identity_verified') is not True:
            raise AssertionError('Full-cohort V5 initialization failed the plain scGPT identity check')
        report['paired_step_zero_results'] = [row for row in baseline['results']
                                             if row['embedding'] in ('base', 'joint')]
        stage('distributed_resume_one_update', train + ['--resume', '--run-update-limit', '1'])
        final = json.loads(progress_path.read_text())
        if final.get('step') != 3 or final.get('complete') is not False:
            raise AssertionError('Resume did not continue from step 2 to 3')
        records = [json.loads(line) for line in (Path(smoke['output_dir']) / 'train_metrics.jsonl').read_text().splitlines()]
        if [row['step'] for row in records] != [1, 2, 3]:
            raise AssertionError('Smoke metric history has duplicate or missing updates')
        if any(row['physical_world_size'] != args.training_gpus for row in records):
            raise AssertionError('Training did not use the requested physical GPU world size')
        for row in records:
            if (row.get('loss_finite') is not True or row.get('gradients_finite') is not True
                    or row['global_batch_size'] != cfg['training']['global_batch_size']
                    or row['batch_per_card'] != cfg['training']['batch_size']
                    or row['accumulation'] * row['physical_world_size'] * row['batch_per_card']
                    != row['global_batch_size']
                    or set(row['loss']) != {'total', 'masked_ce'}
                    or any(not math.isfinite(value) for value in row['loss'].values())
                    or not math.isfinite(row['gradient_norm'])):
                raise AssertionError('Training smoke has inconsistent batch/loss/gradient evidence')
        report['smoke_records'] = records
        checkpoint = Path(smoke['output_dir']) / 'checkpoints' / 'step_00000003.pt'
        stage('checkpoint_fixed_cohort_evaluation', [sys.executable, '-u', 'scripts/evaluate_subcellular_v5.py',
              '--config', str(smoke_path), '--checkpoint', str(checkpoint),
              '--expected-step', '3', '--output-dir', str(output / 'smoke_evaluation'),
              '--batch-size', str(cfg['evaluation']['batch_size']),
              '--cells-per-donor', str(cfg['evaluation']['cells_per_donor']),
              '--mask-validation-batches', str(cfg['evaluation']['mask_validation_batches']),
              '--neighbor-shuffle'])
        metrics = json.loads((output / 'smoke_evaluation' / 'metrics.json').read_text())
        if (metrics.get('step') != 3 or metrics.get('evaluation_version') != 5
                or metrics.get('test_donors_used') is not False
                or metrics.get('canonical_cohort') is not True or metrics.get('cells') != 5200
                or metrics.get('neighbor_shuffle') is not True
                or not metrics.get('masked_validation', {}).get('enabled')
                or not metrics.get('results')):
            raise AssertionError('Evaluation did not produce a valid V5 masked/probe report')
        dim = cfg['model']['slots_per_compartment'] * cfg['model']['slot_dim']
        backbone_dim = int(metrics['arch_config']['backbone_dim'])
        dimensions = {name: backbone_dim if name in ('base', 'intrinsic', 'joint') else dim
                      for name in ('base', 'intrinsic', 'joint', 'nucleus', 'cytoplasm',
                                   'nucleus_local', 'cytoplasm_local')}
        dimensions.update({f'{name}_neighbor_shuffle': dimensions[name]
                           for name in ('joint', 'nucleus', 'cytoplasm')})
        expected_probes = {(name, task) for name in dimensions for task in ('cell_type', 'region')}
        probes = [(row['embedding'], row['task']) for row in metrics['results']]
        if (metrics.get('embedding_dimensions') != dimensions or set(probes) != expected_probes
                or len(probes) != len(expected_probes)):
            raise AssertionError('Evaluation did not cover all seven readouts and three shuffle controls')
        if metrics['backbone_provenance'].get('frozen_backbone_weights_verified') is not True:
            raise AssertionError('Ordinary scGPT identity was not verified after resumed training')
        report.update(status='passed', evaluated_cells=metrics['cells'],
                      evaluated_readouts=metrics['embedding_dimensions'],
                      test_donors_used=metrics['test_donors_used'],
                      checkpoint_sha256=metrics['checkpoint_sha256'])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report.update(elapsed_seconds=time.time() - started, finished_at=time.time())
        write_json(output / 'acceptance.json', report)
    print('[v5-acceptance-passed]', output, flush=True)


if __name__ == '__main__':
    main()
