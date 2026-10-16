"""Content-strict V12 loading with explicitly audited local path relocation.

This module's proof helpers use only the standard library. Model construction
is deferred to load_model; the historical model/data/training source is intact.
Only load trusted torch checkpoints: torch.load uses the original pickle format.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = 12
ARCHITECTURE = 'subcellular_compartment_counts_v12'
REPOSITORY = 'https://github.com/bowang-lab/scGPT'
KNOWN_SPATIAL_SHA256 = '1ce6a9eaa7cc199b23d404ea38203956d04699aaede42b6ea67aa0bf3a88c8be'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    # Identical serialization to the original evaluator/config_hash contract.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_sources(sources, root=ROOT):
    required = {'src/spatial_pretrain/models/subcellular_v12.py',
                'src/spatial_pretrain/data/subcellular_v12.py'}
    if not isinstance(sources, dict) or not required.issubset(sources):
        raise ValueError('Checkpoint lacks model/data source hashes')
    root = Path(root).resolve()
    for name, expected in sources.items():
        if not isinstance(name, str) or Path(name).is_absolute():
            raise ValueError(f'Invalid checkpoint source path: {name!r}')
        source = (root / name).resolve()
        if (not source.is_relative_to(root) or not source.is_file()
                or file_sha256(source) != expected):
            raise ValueError(f'Checkpoint source mismatch: {name}')


def verify_portable_provenance(archived, actual):
    """Allow exactly three path values to differ, retaining every other field.

    In particular source_manifest.sha256 identifies the entire original manifest
    bytes, including fields that the upstream loader does not project to output.
    """
    for record in (archived, actual):
        if (not isinstance(record, dict) or record.get('backbone_kind') != 'scgpt'
                or record.get('source_repository') != REPOSITORY
                or record.get('weights_sha256') == KNOWN_SPATIAL_SHA256):
            raise ValueError('V12 requires official ordinary whole-human scGPT')
        manifest = record.get('source_manifest')
        if (not isinstance(manifest, dict) or manifest.get('backbone_kind') != 'scgpt'
                or manifest.get('model_name') != 'whole-human'
                or manifest.get('repository') != REPOSITORY
                or manifest.get('weights_sha256') != record.get('weights_sha256')):
            raise ValueError('Invalid ordinary whole-human source manifest provenance')
        for key in ('source_sha256', 'weights_sha256', 'args_sha256', 'vocab_sha256'):
            if not isinstance(record.get(key), str) or not re.fullmatch('[0-9a-f]{64}', record[key]):
                raise ValueError(f'Missing/invalid scGPT provenance {key}')
        if not isinstance(manifest.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', manifest['sha256']):
            raise ValueError('Missing/invalid source manifest byte SHA256')
        if any(not isinstance(record.get(key), str) or not record[key] for key in ('source', 'weights')):
            raise ValueError('Missing scGPT source/weights paths')
        if not isinstance(manifest.get('path'), str) or not manifest['path']:
            raise ValueError('Missing source manifest path')
    relocated = deepcopy(archived)
    relocations = {}
    for key in ('source', 'weights'):
        relocations[key] = {'original': archived[key], 'runtime': actual[key]}
        relocated[key] = actual[key]
    relocations['source_manifest.path'] = {
        'original': archived['source_manifest']['path'],
        'runtime': actual['source_manifest']['path']}
    relocated['source_manifest']['path'] = actual['source_manifest']['path']
    if relocated != actual:
        raise ValueError('Checkpoint initial-backbone content provenance mismatch')
    return relocations


def runtime_config(original, expected_sha256, assets_dir, lr_path):
    """Preserve the original object and alter only two declared runtime paths."""
    if not isinstance(original, dict) or fingerprint(original) != expected_sha256:
        raise ValueError('Missing or incorrect original config_sha256')
    if original.get('model', {}).get('backbone_kind') != 'scgpt':
        raise ValueError('V12 configuration must explicitly select ordinary scGPT')
    runtime = deepcopy(original)
    runtime['assets_dir'] = str(Path(assets_dir).expanduser().resolve(strict=True))
    if not Path(runtime['assets_dir']).is_dir():
        raise ValueError('assets_dir must be a directory')
    prior = runtime.get('lr')
    if not isinstance(prior, dict) or prior.get('enabled') is not True:
        raise ValueError('This V12 release requires the checkpoint enabled curated LR prior')
    path = Path(lr_path).expanduser().resolve(strict=True)
    expected = prior.get('sha256')
    if (not isinstance(expected, str) or not re.fullmatch('[0-9a-fA-F]{64}', expected)
            or not path.is_file() or file_sha256(path) != expected.lower()):
        raise ValueError('Relocated LR prior does not match original config SHA256')
    prior['path'] = str(path)
    paths = {'assets_dir': {'original': original.get('assets_dir'), 'runtime': runtime['assets_dir']},
             'lr.path': {'original': original['lr'].get('path'), 'runtime': prior['path']}}
    return runtime, paths


def verify_asset_bytes(archived, assets_dir):
    """Check every asset before executing upstream source or loading tensors."""
    verify_portable_provenance(archived, archived)
    asset_root = Path(assets_dir)
    asset_files = {
        'source_sha256': asset_root / 'code/scgpt/model/model.py',
        'weights_sha256': asset_root / 'scGPT_human/best_model.pt',
        'args_sha256': asset_root / 'scGPT_human/args.json',
        'vocab_sha256': asset_root / 'scGPT_human/vocab.json',
    }
    for key, asset in asset_files.items():
        if file_sha256(asset) != archived[key]:
            raise ValueError(f'Relocated scGPT asset content mismatch: {key}')
    if file_sha256(asset_root / 'source_manifest.json') != archived['source_manifest']['sha256']:
        raise ValueError('Relocated source manifest byte SHA256 mismatch')


def require_compute_execution(standalone_compute=False):
    """Verify host placement without fabricating Slurm state for CUDA servers."""
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    if standalone_compute:
        cluster_env = ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_CLUSTER_NAME',
                       'SLURM_CONF', 'CC_CLUSTER', 'RSNT_ARCH')
        cluster_name = re.search(
            r'login|rorqual|killarney|vulcan|narval|beluga|cedar|graham|nibi|fir\.(?:alliancecan|computecanada)',
            hostname + ' ' + fqdn, re.I)
        cluster_files = (Path('/cvmfs/soft.computecanada.ca'),
                         Path('/etc/slurm/slurm.conf'), Path('/etc/slurm.conf'))
        if cluster_name or any(os.environ.get(key) for key in cluster_env) or any(p.exists() for p in cluster_files):
            raise RuntimeError('--standalone-compute is forbidden on HPC/Slurm hosts, including login nodes')
        return {'execution_mode': 'standalone_compute', 'slurm_job_id': None,
                'slurm_step_id': None, 'hostname': hostname}
    job, step = os.environ.get('SLURM_JOB_ID'), os.environ.get('SLURM_STEP_ID')
    if not job or not step or not re.fullmatch(r'\d+(?:_\d+)?', job) or not re.fullmatch(r'\d+', step):
        raise RuntimeError('Use a verified Slurm compute step, or --standalone-compute on a standalone CUDA server')
    try:
        record = subprocess.run(['scontrol', 'show', 'step', f'{job}.{step}', '-o'],
                                check=True, capture_output=True, text=True, timeout=20).stdout
        nodes = re.search(r'(?:^|\s)Nodes=(\S+)', record)
        if not nodes:
            raise RuntimeError('Cannot establish the actual Slurm step node list')
        hosts = subprocess.run(['scontrol', 'show', 'hostnames', nodes[1]],
                               check=True, capture_output=True, text=True, timeout=20).stdout.split()
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('Cannot verify the Slurm compute step; refusing inference') from error
    if hostname.split('.')[0] not in {host.split('.')[0] for host in hosts}:
        raise RuntimeError('Current hostname is outside the verified Slurm compute step')
    if re.search(r'login', hostname, re.I):
        raise RuntimeError('Inference on an HPC login node is forbidden')
    return {'execution_mode': 'slurm_compute_step', 'slurm_job_id': job,
            'slurm_step_id': step, 'hostname': hostname}


def load_model(path, expected_step, *, assets_dir, lr_path):
    import torch

    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('model_version') != MODEL_VERSION or saved.get('architecture') != ARCHITECTURE:
        raise ValueError(f'Expected V12 {ARCHITECTURE}')
    step = saved['step']
    if (isinstance(step, bool) or not isinstance(step, int) or step < 0
            or expected_step is None or step != expected_step):
        raise ValueError('Checkpoint step mismatch (an explicit expected step is required)')
    original_cfg = saved['config']
    cfg, config_paths = runtime_config(original_cfg, saved.get('config_sha256'), assets_dir, lr_path)
    sources = saved.get('source_sha256')
    verify_sources(sources)
    # Check content before build_model executes the two upstream class bodies.
    archived_provenance = saved.get('backbone_provenance')
    verify_asset_bytes(archived_provenance, cfg['assets_dir'])
    from spatial_pretrain.models.subcellular_v12 import build_model
    from spatial_pretrain.models.scgpt_backbone import ScGPTBackbone
    from spatial_pretrain.data.subcellular_v6 import load_lr_prior

    model = build_model(cfg)
    initial = deepcopy(model.backbone.provenance)
    backbone_paths = verify_portable_provenance(saved.get('backbone_provenance'), initial)
    if saved.get('arch_config') != model.arch_config:
        raise ValueError('Checkpoint architecture configuration mismatch')
    # Keep the historical step-zero test before any fine-tuned weights are read.
    if step == 0:
        for name, tensor in model.backbone.state_dict().items():
            archived = saved['model'].get(f'backbone.{name}')
            if archived is None or not torch.equal(tensor.detach().cpu(), archived.detach().cpu()):
                raise ValueError(f'Step-zero encoder differs from original scGPT: {name}')
    model.load_state_dict(saved['model'], strict=True)
    # Construct a new plain asset encoder; never copy the fine-tuned backbone.
    baseline = ScGPTBackbone(cfg['assets_dir'], checkpointing=False, backbone_kind='scgpt')
    if baseline.provenance != initial:
        raise ValueError('Original baseline and initialization provenance differ')
    load_lr_prior(cfg, model.backbone.vocab)  # Original parser and byte contract.
    provenance = dict(config_sha256=saved['config_sha256'],
        original_config_sha256=saved['config_sha256'], runtime_config_sha256=fingerprint(cfg),
        runtime_config_path_relocations=config_paths,
        source_sha256=sources, sources_verified=True,
        original_backbone_provenance=deepcopy(saved['backbone_provenance']),
        backbone_provenance=initial, backbone_path_relocations=backbone_paths,
        source_manifest_bytes_verified=True, baseline_separate_pretrained_encoder=True)
    del saved
    return (model.eval().requires_grad_(False).cuda(),
            baseline.eval().requires_grad_(False).cuda(), cfg, step, provenance)
