"""Compute-only gate with exclusive, checkpoint-safe two-to-four GPU handoff."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
UPGRADE_SCHEMA = 'subcellular_v6_upgrade_request_v1'
HANDOFF_MARKER = 'HANDOFF_TO_4GPU_REQUESTED'


class Superseded(Exception):
    """A waiting two-GPU bootstrap must yield to the recorded upgrade."""


def atomic_json(path, payload):
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def config_fingerprint(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def controller_identity(run, cfg):
    return dict(schema='subcellular_v6_controller_v1', output_dir=str(run),
                config_sha256=config_fingerprint(cfg), interval=100, cohort='fixed5200',
                neighbor_shuffle=True, neighbor_shuffle_seed=815,
                mask_validation_batches=16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('A', 'B'), required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus-per-node', type=int, default=4)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID'):
        raise RuntimeError('Run inside a verified held Slurm compute step')
    world = args.nodes * args.gpus_per_node
    if (args.nodes < 1 or args.gpus_per_node < 1 or world not in (2, 4)
            or (args.route == 'B' and world != 4) or not math.isfinite(args.deadline)):
        raise ValueError('Require route A on 2/4 GPUs or route B on 4 GPUs and a finite deadline')
    os.chdir(ROOT)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = ROOT / f'configs/subcellular_lr_v6_{args.route.lower()}_20260920.yaml'
    # Preparation writes these exact immutable output paths. Derive the lease
    # before preparation so a late two-GPU allocation cannot initialize a run.
    run = ROOT / 'outputs' / f'subcellular_lr_v6_{args.route.lower()}_20260920'
    control = run.with_name(run.name + '.controller')
    coordination = run.with_name(run.name + '.coordination')
    coordination.mkdir(parents=True, exist_ok=True)
    request_path = coordination / 'upgrade_request.json'
    lease = None
    state = dict(status='preparing', route=args.route, jobid=os.environ['SLURM_JOB_ID'],
                 node=socket.gethostname(), deadline=args.deadline, stages=[],
                 training_gpus=world, nodes=args.nodes, gpus_per_node=args.gpus_per_node,
                 coordination=str(coordination), formal_run=str(run))

    def write_state():
        state['updated_at'] = time.time()
        atomic_json(output / 'status.json', state)

    def check_deadline(name):
        if time.time() >= args.deadline:
            raise TimeoutError('Allocation safety deadline reached before '+name)

    def read_config():
        cfg = yaml.safe_load(config.read_text())
        if (cfg['training']['optimizer'] != 'muon' or cfg['model']['route'] != args.route
                or Path(cfg['output_dir']).resolve() != run):
            raise ValueError('Preparation did not produce the approved route/Muon/output configuration')
        return cfg

    def read_request():
        if not request_path.exists():
            return None
        request = json.loads(request_path.read_text())
        if (request.get('schema') != UPGRADE_SCHEMA or request.get('route') != 'A'
                or args.route != 'A' or request.get('output_dir') != str(run)
                or request.get('target_world_size') != 4
                or not isinstance(request.get('target_jobid'), str)
                or not request['target_jobid'].isdigit()):
            raise ValueError('Invalid upgrade request; refusing an ambiguous handoff')
        return request

    def superseded():
        request = read_request()
        if world == 2 and request is not None:
            state.update(status='superseded', upgrade_request=request,
                         reason='four_gpu_upgrade_requested')
            write_state()
            return True
        return False

    def publish_request():
        request = dict(schema=UPGRADE_SCHEMA, route=args.route, output_dir=str(run),
                       target_jobid=state['jobid'], target_world_size=4,
                       requested_at=time.time())
        # Hard-link publication is atomic and never replaces an earlier target.
        temporary = coordination / f'upgrade_request.{state["jobid"]}.{os.getpid()}.partial'
        temporary.write_text(json.dumps(request, indent=2, allow_nan=False) + '\n')
        try:
            try:
                os.link(temporary, request_path)
            except FileExistsError:
                pass
        finally:
            temporary.unlink()
        request = read_request()
        if request['target_jobid'] != state['jobid']:
            raise RuntimeError('A different four-GPU allocation already owns the upgrade request')
        state['upgrade_request'] = request
        write_state()

    def check_identity(cfg):
        identity_path = control / 'identity.json'
        if (not identity_path.is_file()
                or json.loads(identity_path.read_text()) != controller_identity(run, cfg)):
            raise ValueError('Existing formal run has no matching controller identity')

    def request_handoff():
        if args.route != 'A' or world != 4 or not (run / 'resume.pt').is_file():
            return
        request = read_request()
        if request is None or request['target_jobid'] != state['jobid']:
            raise ValueError('This allocation no longer owns the upgrade request')
        # Do not create a run directory, interrupt a foreign run, or pause an
        # initializer before its first complete optimizer checkpoint exists.
        check_identity(read_config())
        marker = run / HANDOFF_MARKER
        if marker.exists():
            if json.loads(marker.read_text()) != request:
                raise ValueError('Foreign handoff marker exists; refusing to replace it')
        else:
            atomic_json(marker, request)
        state['handoff_requested'] = True

    def wait_for_lock(handle, name, *, handoff=False):
        while True:
            check_deadline(name)
            if world == 2 and superseded():
                raise Superseded
            if handoff:
                request_handoff()
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                state.update(status='waiting_for_' + name)
                write_state()
                time.sleep(min(5.0, max(0.0, args.deadline - time.time())))

    def formal_resume_state():
        """Require intact provenance; never repair or replace a partial run."""
        resume = run / 'resume.pt'
        if resume.is_file():
            cfg = read_config()
            check_identity(cfg)
            if resume.stat().st_size == 0:
                raise ValueError('Formal resume checkpoint is empty')
            for filename in ('progress.json', 'schedule.json'):
                path = run / filename
                if not path.is_file():
                    raise ValueError('Formal resume checkpoint lacks ' + filename)
                metadata = json.loads(path.read_text())
                if (metadata.get('model_version') != 6
                        or metadata.get('architecture') != 'subcellular_lr_routes_v6'):
                    raise ValueError('Formal resume metadata has an incompatible architecture')
                if filename == 'progress.json' and metadata.get('config_sha256') != config_fingerprint(cfg):
                    raise ValueError('Formal resume progress has a different configuration')
            status_path = control / 'status.json'
            if status_path.is_file() and json.loads(status_path.read_text()).get('status') == 'failed':
                raise RuntimeError('Formal controller failed; explicit diagnosis is required before resuming')
            return True
        if (run.exists() and any(run.iterdir())) or control.exists():
            raise FileExistsError('Incomplete existing formal run has no resume.pt; refusing to overwrite it')
        return False

    def settle_formal_owner():
        # A bootstrap crash can leave a controller or trainer descendant alive.
        # Their own leases are authoritative even after our outer lease is free.
        with ExitStack() as held:
            for path, name in ((control / 'owner.lock', 'controller'),
                               (run / '.trainer.lock', 'trainer')):
                if path.is_file():
                    handle = held.enter_context(path.open('r+'))
                    wait_for_lock(handle, name, handoff=(args.route == 'A' and world == 4))
            resume = formal_resume_state()
            marker = run / HANDOFF_MARKER
            if args.route == 'A' and world == 4 and marker.exists():
                request = read_request()
                if (request is None or request['target_jobid'] != state['jobid']
                        or json.loads(marker.read_text()) != request):
                    raise ValueError('Refusing to remove a handoff marker owned by another request')
                marker.unlink()
                state['handoff_marker_cleared'] = True
            return resume

    def stage(name, command, *, extra_fds=()):
        check_deadline(name)
        state['status'] = name
        entry = dict(stage=name, command=command, started_at=time.time())
        state['stages'].append(entry)
        write_state()
        with (output / f'{name}.log').open('x', buffering=1) as handle:
            try:
                result = subprocess.run(command, cwd=ROOT, check=True, stdout=handle,
                                        stderr=subprocess.STDOUT,
                                        pass_fds=(lease.fileno(), *extra_fds))
            except subprocess.CalledProcessError as error:
                entry.update(finished_at=time.time(), exit_code=error.returncode)
                write_state()
                raise
        entry['finished_at'] = time.time()
        entry['exit_code'] = result.returncode
        write_state()
    try:
        if args.route == 'A' and world == 4:
            publish_request()
        if superseded():
            return
        lease = (coordination / 'bootstrap.lock').open('a')
        wait_for_lock(lease, 'bootstrap_lease', handoff=(args.route == 'A' and world == 4))
        state['lease_acquired_at'] = time.time()
        if superseded():
            return
        # Also fence any descendants left by an earlier bootstrap before QA.
        resume = settle_formal_owner()
        state['formal_resume_available'] = resume
        # Two separate allocations may enter preparation close together. Only
        # one writer prepares the immutable shared prior and A/B configs.
        lock_path = ROOT / 'resources/lr_v6/preparation.lock'
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open('a') as lock:
            wait_for_lock(lock, 'preparation')
            if superseded():
                return
            stage('prepare', [sys.executable, 'scripts/prepare_subcellular_v6_lr.py'],
                  extra_fds=(lock.fileno(),))
        cfg = read_config()
        if superseded():
            return
        stage('lr_audit', [sys.executable, 'scripts/audit_subcellular_v6_lr.py', '--config', str(config),
                           '--output', str(output / 'lr_audit.json')])
        if json.loads((output / 'lr_audit.json').read_text()).get('status') != 'passed':
            raise RuntimeError('LR audit did not produce successful evidence')
        acceptance = output / 'acceptance'
        if superseded():
            return
        stage('acceptance', [sys.executable, '-u', 'scripts/validate_subcellular_v6.py',
              '--config', str(config), '--output-dir', str(acceptance),
              '--training-gpus', str(world), '--nodes', str(args.nodes),
              '--gpus-per-node', str(args.gpus_per_node)])
        evidence = json.loads((acceptance / 'acceptance.json').read_text())
        fingerprint = config_fingerprint(cfg)
        if (evidence.get('status') != 'passed' or evidence.get('route') != args.route
                or evidence.get('source_config_sha256') != fingerprint
                or evidence.get('training_gpus') != world
                or evidence.get('optimizer_resume_verified') is not True):
            raise RuntimeError('Acceptance lacks matching successful GPU/resume evidence')
        state.update(acceptance_verified=True, config=str(config), config_sha256=fingerprint)
        # This check closes the long QA window: when four GPUs arrived during
        # QA, the two-GPU worker releases its lease without creating a run.
        if superseded():
            return
        resume = settle_formal_owner()
        if superseded():
            return
        command = [sys.executable, '-u', 'scripts/run_subcellular_v6.py',
              '--config', str(config), '--deadline', str(args.deadline),
              '--training-gpus', str(world), '--nodes', str(args.nodes),
              '--gpus-per-node', str(args.gpus_per_node)]
        if resume:
            command.append('--resume')
        state['formal_resume'] = resume
        stage('formal_controller', command)
        formal = control / 'status.json'
        state['formal_status'] = json.loads(formal.read_text()) if formal.exists() else None
        state['status'] = 'complete' if state['formal_status'] and state['formal_status']['status'] == 'complete' else 'paused'
    except Superseded:
        # superseded() already recorded the persistent target allocation.
        return
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            write_state()
        finally:
            if lease is not None:
                # Closing this descriptor (without LOCK_UN) preserves the
                # flock while any inherited stage descriptor remains open.
                lease.close()


if __name__ == '__main__':
    main()
