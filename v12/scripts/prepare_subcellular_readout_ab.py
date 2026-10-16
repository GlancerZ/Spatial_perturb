"""Create two isolated readout A/B continuations from one frozen parent resume.

Only experiment/output_dir/model.direct_niche_readout configuration fields change.
Existing AdamW state is remapped by parameter name; the new zero projection has
no optimizer moments until its first update. This script never starts training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
import yaml
from spatial_pretrain.models.subcellular_v2 import SubcellularV2
from train_subcellular_v2 import atomic_save


PROJECTION_NAME = 'niche_to_joint.weight'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parameter_groups_by_name(model):
    names = [name for name, _ in model.named_parameters()]
    return [[name for name in names if name.startswith('backbone.')],
            [name for name in names if not name.startswith('backbone.')]]


def remap_optimizer_state(parent_optimizer, parent_model, target_model):
    """Preserve every existing group's hyperparameters and every tensor by name."""
    old_names = parameter_groups_by_name(parent_model)
    new_names = parameter_groups_by_name(target_model)
    old_groups = parent_optimizer['param_groups']
    if len(old_groups) != 2 or any(len(group['params']) != len(names) for group, names in zip(old_groups, old_names)):
        raise ValueError('Parent optimizer does not follow the original backbone/new parameter groups')
    source_states = {}
    for group, names in zip(old_groups, old_names):
        for identifier, name in zip(group['params'], names):
            if identifier in parent_optimizer['state']:
                source_states[name] = parent_optimizer['state'][identifier]
    new_parameters = dict(target_model.named_parameters())
    missing_existing = set(sum(old_names, [])) - set(new_parameters)
    if missing_existing:
        raise ValueError(f'Fork removed existing parameters: {sorted(missing_existing)}')
    added = set(new_parameters) - set(sum(old_names, []))
    if added - {PROJECTION_NAME}:
        raise ValueError(f'Unexpected new parameters: {sorted(added)}')
    destination = {'state': {}, 'param_groups': []}
    identifier = 0
    for old_group, names in zip(old_groups, new_names):
        group = copy.deepcopy({key: value for key, value in old_group.items() if key != 'params'})
        group['params'] = []
        for name in names:
            group['params'].append(identifier)
            if name in source_states:
                values = source_states[name]
                for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                    if key in values and values[key].shape != new_parameters[name].shape:
                        raise ValueError(f'Optimizer tensor shape mismatch: {name}/{key}')
                destination['state'][identifier] = copy.deepcopy(values)
            identifier += 1
        destination['param_groups'].append(group)
    return destination, sorted(added)


def prepare_arm(parent, parent_model, target_model, cfg, lineage):
    incompatible = target_model.load_state_dict(parent['model'], strict=False)
    expected_missing = [PROJECTION_NAME] if cfg['model']['direct_niche_readout'] else []
    if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise ValueError(f'Unexpected fork model-state differences: {incompatible}')
    if expected_missing and torch.count_nonzero(target_model.niche_to_joint.weight).item():
        raise ValueError('The direct-niche projection must start exactly at zero')
    optimizer, added = remap_optimizer_state(parent['optimizer'], parent_model, target_model)
    payload = {key: copy.deepcopy(value) for key, value in parent.items() if key not in ('model', 'optimizer', 'config')}
    payload.update(model=target_model.state_dict(), optimizer=optimizer, config=cfg,
                   config_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                   lineage=lineage, complete=False)
    # The source's unchanged RNG, micro-step, logical world, source hashes and
    # optimizer step counters remain usable by both serial and four-GPU trainers.
    return payload, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-resume', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--assets-dir', type=Path, help='Optional verification-only asset location; saved config retains parent assets_dir')
    parser.add_argument('--extra-updates', type=int, default=300)
    args = parser.parse_args()
    if args.extra_updates < 1:
        parser.error('--extra-updates must be positive')
    source = args.parent_resume.resolve()
    destination = args.output_root.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('A/B output root is nonempty; never overwrite an existing experiment')
    before = source.stat()
    source_sha = file_sha256(source)
    parent = torch.load(source, map_location='cpu', weights_only=False)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Parent resume changed while reading; acquire the training lock before preparing forks')
    original_cfg = copy.deepcopy(parent['config'])
    original_sha = hashlib.sha256(json.dumps(original_cfg, sort_keys=True).encode()).hexdigest()
    if original_sha != parent['config_sha256']:
        raise ValueError('Parent checkpoint config fingerprint is invalid')
    if parent.get('complete') or parent['world_size'] != 4 or len(parent['rng_by_rank']) != 4:
        raise ValueError('Need an incomplete logical-four-rank resume with four RNG streams')
    training = original_cfg['training']
    if (training['expected_world_size'], training['batch_size'], training['grad_accum_steps']) != (4, 16, 1):
        raise ValueError('A/B requires the unchanged logical-four-rank / rank-batch16 / accumulation1 configuration')
    if parent['step'] < training['unfreeze_step'] or original_cfg['model'].get('direct_niche_readout', False):
        raise ValueError('Parent must be fully unfrozen and have the original joint readout')
    assets = args.assets_dir or Path(original_cfg['assets_dir'])
    parent_model = SubcellularV2(assets, **original_cfg['model'])
    parent_model.load_state_dict(parent['model'], strict=True)
    for key in ('backbone_kind', 'weights_sha256', 'source_sha256', 'args_sha256', 'vocab_sha256'):
        if parent['backbone_provenance'].get(key) != parent_model.backbone.provenance.get(key):
            raise ValueError(f'Parent pretrained provenance differs from verified assets: {key}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + f'.preparing.{os.getpid()}')
    staging.mkdir()
    experiment = dict(schema_version=1, parent_resume=str(source), parent_checkpoint_sha256=source_sha,
        parent_step=int(parent['step']), fork_step=int(parent['step']), target_step=int(parent['step']) + args.extra_updates,
        extra_updates=args.extra_updates, logical_world_size=4, rank_batch_size=16,
        parent_config_sha256=original_sha, arms={})
    for arm, direct in [('control', False), ('direct_niche', True)]:
        cfg = copy.deepcopy(original_cfg)
        cfg['experiment'] = str(original_cfg.get('experiment', 'subcellular_scgpt')) + '_readout_ab_' + arm
        cfg['output_dir'] = str(destination / arm)
        cfg['model']['direct_niche_readout'] = direct
        target = SubcellularV2(assets, **cfg['model'])
        lineage = dict(parent_resume=str(source), parent_checkpoint_sha256=source_sha,
            parent_step=int(parent['step']), fork_step=int(parent['step']), parent_config_sha256=original_sha, arm=arm,
            model_structure_change={'direct_niche_readout': direct,
                'added_parameter': PROJECTION_NAME if direct else None,
                'initialization': 'exact zeros' if direct else 'unchanged parent model'},
            optimizer_policy='Existing AdamW states mapped by parameter name; new parameter starts without moments',
            config_changes=['experiment', 'output_dir', 'model.direct_niche_readout'],
            parent_lineage=parent.get('lineage'))
        payload, added = prepare_arm(parent, parent_model, target, cfg, lineage)
        folder = staging / arm
        (folder / 'checkpoints').mkdir(parents=True)
        text = yaml.safe_dump(cfg, sort_keys=False)
        (folder / 'config.yaml').write_text(text)
        (folder / 'run_config.yaml').write_text(text)
        atomic_save(payload, folder / 'resume.pt')
        archive = {key: value for key, value in payload.items() if key not in ('optimizer', 'rng_by_rank')}
        initial_name = f'step_{parent["step"]:08d}.pt'
        atomic_save(archive, folder / 'checkpoints' / initial_name)
        metadata = dict(**lineage, config_sha256=payload['config_sha256'],
            target_step=experiment['target_step'], extra_updates=args.extra_updates,
            initial_checkpoint=str(destination / arm / 'checkpoints' / initial_name),
            parent_epoch=parent['epoch'], parent_micro_step=parent['micro_step'],
            added_parameters=added, backbone_provenance=copy.deepcopy(parent['backbone_provenance']))
        (folder / 'fork_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        experiment['arms'][arm] = dict(config=str(destination / arm / 'config.yaml'),
            run_dir=str(destination / arm), initial_checkpoint=metadata['initial_checkpoint'],
            config_sha256=payload['config_sha256'], direct_niche_readout=direct)
        print(f'[fork-prepared] {arm} parent_step={parent["step"]} added={added}', flush=True)
        del target, payload, archive
    (staging / 'experiment.json').write_text(json.dumps(experiment, indent=2) + '\n')
    staging.replace(destination)
    print(f'[fork-ready] {destination / "experiment.json"}', flush=True)


if __name__ == '__main__':
    main()
