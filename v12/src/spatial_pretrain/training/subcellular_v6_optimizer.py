"""Strict official-Muon/AdamW optimizer partition for the V6 A/B experiment.

Only hidden Linear/MultiheadAttention matrices use Muon. Learned slot queries,
embeddings, normalizations, biases, the prediction decoder and final CLS
residual projections use AdamW. Both groups follow the same schedule multiplier
applied to their separately recorded base learning rates.
"""
from __future__ import annotations

import copy
import math
from typing import Mapping

import torch
from torch import nn


SCHEMA = 'subcellular_v6_muon_v1'
ADAMW_PREFIXES = ('joint_decoder.', 'residual_projection.', 'self_projection.',
                  'context_projection.')
MUON_SETTINGS = dict(momentum=.95, nesterov=True, ns_steps=5,
                     ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-7,
                     adjust_lr_fn='match_rms_adamw')
ADAMW_SETTINGS = dict(betas=(.9, .999), eps=1e-8, amsgrad=False,
                      maximize=False, foreach=False, capturable=False,
                      differentiable=False, fused=False)


def validate_optimizer_config(training: Mapping):
    if training.get('optimizer') != 'muon':
        raise ValueError("V6 requires explicit training.optimizer='muon'")
    for name in ('lr_muon', 'lr_new'):
        value = training.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f'training.{name} must be a finite positive learning rate')
    decay = training.get('weight_decay')
    if (isinstance(decay, bool) or not isinstance(decay, (int, float))
            or not math.isfinite(decay) or decay < 0):
        raise ValueError('training.weight_decay must be finite and nonnegative')
    if training.get('muon_adjust_lr_fn') != 'match_rms_adamw':
        raise ValueError("V6 requires explicit muon_adjust_lr_fn='match_rms_adamw'")
    if (training.get('muon_momentum', .95) != .95
            or training.get('muon_ns_steps', 5) != 5
            or isinstance(training.get('muon_ns_steps', 5), bool)
            or training.get('muon_nesterov', True) is not True):
        raise ValueError('V6 fixes Muon momentum=.95, ns_steps=5 and nesterov=True')


def _partition(model: nn.Module):
    modules = dict(model.named_modules())
    groups = {'muon_hidden': [], 'adamw_aux': []}
    # A tensor tied to any output/embedding role must remain on AdamW even if
    # its first canonical name happens to be a hidden-layer alias.
    protected = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        owner_name, _, leaf = name.rpartition('.')
        owner = modules.get(owner_name)
        if (name.startswith(ADAMW_PREFIXES) or isinstance(owner, (nn.Embedding, nn.LayerNorm))
                or name in ('slot_pool.queries', 'slot_pool.missing')
                or leaf in ('bias', 'in_proj_bias', 'bias_k', 'bias_v')):
            protected.add(id(parameter))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('backbone.'):
            raise ValueError('V6 optimizer cannot include trainable frozen-backbone parameters')
        owner_name, _, leaf = name.rpartition('.')
        owner = modules.get(owner_name)
        hidden_matrix = (parameter.ndim == 2 and id(parameter) not in protected
                         and ((isinstance(owner, nn.Linear) and leaf == 'weight')
                              or (isinstance(owner, nn.MultiheadAttention)
                                  and leaf in ('in_proj_weight', 'q_proj_weight',
                                               'k_proj_weight', 'v_proj_weight'))))
        groups['muon_hidden' if hidden_matrix else 'adamw_aux'].append((name, parameter))
    flat = [parameter for entries in groups.values() for _, parameter in entries]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if (len({id(parameter) for parameter in flat}) != len(flat)
            or {id(parameter) for parameter in flat} != {id(parameter) for parameter in expected}):
        raise ValueError('Optimizer partition must cover every trainable tensor exactly once')
    if not all(groups.values()):
        raise ValueError('V6 requires both hidden Muon matrices and auxiliary AdamW parameters')
    return groups


def optimizer_manifest(model: nn.Module, training: Mapping) -> dict:
    """Describe the complete partition without constructing either optimizer."""
    validate_optimizer_config(training)
    groups = []
    for name, entries in _partition(model).items():
        muon = name == 'muon_hidden'
        groups.append(dict(
            name=name, use_muon=muon,
            base_lr=float(training['lr_muon' if muon else 'lr_new']),
            weight_decay=float(training['weight_decay']),
            param_names=[name for name, _ in entries],
            param_shapes=[list(parameter.shape) for _, parameter in entries],
            param_dtypes=[str(parameter.dtype) for _, parameter in entries],
            tensor_count=len(entries), parameter_count=sum(parameter.numel() for _, parameter in entries),
            settings={key: list(value) if isinstance(value, tuple) else value
                      for key, value in (MUON_SETTINGS if muon else ADAMW_SETTINGS).items()}))
    return dict(schema=SCHEMA, implementation='torch.optim.Muon+torch.optim.AdamW',
                torch_version=str(torch.__version__),
                partition='hidden_linear_attention_matrices_muon_all_other_parameters_adamw',
                schedule='shared_multiplier_times_separate_base_lrs', groups=groups)


def _assert_state_equal(expected, actual, path='optimizer'):
    """Verify loaded tensor contents as well as names, hyperparameters and LR."""
    if torch.is_tensor(expected):
        if (not torch.is_tensor(actual) or expected.dtype != actual.dtype
                or expected.shape != actual.shape
                or not torch.equal(expected.detach().cpu(), actual.detach().cpu())):
            raise ValueError(f'Restored optimizer state differs at {path}')
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise ValueError(f'Restored optimizer state keys differ at {path}')
        for key in expected:
            _assert_state_equal(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, (tuple, list)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise ValueError(f'Restored optimizer state sequence differs at {path}')
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_state_equal(left, right, f'{path}.{index}')
    elif expected != actual:
        raise ValueError(f'Restored optimizer state differs at {path}')


class V6MuonOptimizer:
    """A strict composite optimizer; no partial-state or unavailable-Muon fallback."""

    def __init__(self, model: nn.Module, training: Mapping):
        self.manifest = optimizer_manifest(model, training)
        muon_class = getattr(torch.optim, 'Muon', None)
        if not callable(muon_class):
            raise RuntimeError(f'torch.optim.Muon is unavailable in torch {torch.__version__}; no fallback is allowed')
        partition = _partition(model)
        groups = []
        for description in self.manifest['groups']:
            group = {key: copy.deepcopy(description[key])
                     for key in ('name', 'use_muon', 'base_lr', 'weight_decay', 'param_names')}
            group['lr'] = group['base_lr']
            group['params'] = [parameter for _, parameter in partition[group['name']]]
            groups.append(group)
        self.muon_optimizer = muon_class([groups[0]], **MUON_SETTINGS)
        self.adamw_optimizer = torch.optim.AdamW([groups[1]], **ADAMW_SETTINGS)
        self._refresh_groups()
        self.steps_completed = 0
        self.resume_state_verified = False
        self.restored_parameter_states = {'muon': 0, 'adamw': 0}

    def _refresh_groups(self):
        # Optimizer.load_state_dict replaces its dictionaries. Never cache copies
        # here: the resumed schedule must update the actual underlying groups.
        self.param_groups = [*self.muon_optimizer.param_groups, *self.adamw_optimizer.param_groups]

    @property
    def state(self):
        return {**self.muon_optimizer.state, **self.adamw_optimizer.state}

    def zero_grad(self, set_to_none=True):
        self.muon_optimizer.zero_grad(set_to_none=set_to_none)
        self.adamw_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.muon_optimizer.step()
        self.adamw_optimizer.step()
        self.steps_completed += 1
        return loss

    def set_lr_multiplier(self, multiplier: float):
        if (isinstance(multiplier, bool) or not isinstance(multiplier, (int, float))
                or not math.isfinite(multiplier) or multiplier < 0):
            raise ValueError('The common learning-rate multiplier must be finite and nonnegative')
        for group in self.param_groups:
            group['lr'] = group['base_lr'] * multiplier

    def learning_rates(self) -> dict:
        return {'muon': float(self.muon_optimizer.param_groups[0]['lr']),
                'adamw': float(self.adamw_optimizer.param_groups[0]['lr'])}

    def state_dict(self) -> dict:
        muon, adamw = self.muon_optimizer.state_dict(), self.adamw_optimizer.state_dict()
        initialized = {}
        for name, component in (('muon', muon), ('aux_adamw', adamw)):
            group = component['param_groups'][0]
            initialized[name] = [parameter_name for identifier, parameter_name in
                                 zip(group['params'], group['param_names']) if identifier in component['state']]
        return dict(schema=SCHEMA, manifest=copy.deepcopy(self.manifest), steps_completed=self.steps_completed,
                    initialized_parameters=initialized, muon=muon, aux_adamw=adamw)

    def _validate_component(self, name, saved, optimizer, initialized, steps):
        if not isinstance(saved, dict) or set(saved) != {'state', 'param_groups'}:
            raise ValueError(f'Missing or invalid {name} optimizer state')
        current = optimizer.state_dict()
        if not isinstance(saved['state'], dict) or len(saved['param_groups']) != 1:
            raise ValueError(f'Invalid {name} optimizer groups/state')
        actual, expected = saved['param_groups'][0], current['param_groups'][0]
        if set(actual) != set(expected):
            raise ValueError(f'{name} optimizer group keys differ')
        for key in expected:
            if key != 'lr' and actual[key] != expected[key]:
                raise ValueError(f'{name} optimizer group {key} differs from the manifest')
        if not isinstance(actual['lr'], (int, float)) or not math.isfinite(actual['lr']) or actual['lr'] < 0:
            raise ValueError(f'{name} optimizer learning rate is invalid')
        identifiers = actual['params']
        if len(identifiers) != len(set(identifiers)) or not set(saved['state']).issubset(identifiers):
            raise ValueError(f'{name} optimizer parameter identifiers differ')
        present = [parameter_name for identifier, parameter_name in zip(identifiers, actual['param_names'])
                   if identifier in saved['state']]
        if not isinstance(initialized, list) or initialized != present:
            raise ValueError(f'{name} optimizer initialized parameter states are missing or reordered')
        if steps == 0 and saved['state']:
            raise ValueError(f'{name} optimizer has moments before any completed update')
        if steps > 0 and not saved['state']:
            raise ValueError(f'{name} optimizer moments are missing after completed updates')
        by_identifier = dict(zip(identifiers, optimizer.param_groups[0]['params']))
        required = {'momentum_buffer'} if name == 'muon' else {'step', 'exp_avg', 'exp_avg_sq'}
        for identifier, values in saved['state'].items():
            if not isinstance(values, dict) or set(values) != required:
                raise ValueError(f'{name} optimizer required momentum/moment state is missing')
            for key, tensor in values.items():
                shape = torch.Size([]) if key == 'step' else by_identifier[identifier].shape
                if not torch.is_tensor(tensor) or tensor.shape != shape or not bool(torch.isfinite(tensor).all()):
                    raise ValueError(f'{name} optimizer {key} has invalid shape or nonfinite values')
                if key == 'step' and (tensor.item() < 1 or tensor.item() > steps or tensor.item() != int(tensor.item())):
                    raise ValueError('AdamW parameter step disagrees with the completed optimizer updates')

    def load_state_dict(self, saved: dict):
        required = {'schema', 'manifest', 'steps_completed', 'initialized_parameters', 'muon', 'aux_adamw'}
        if not isinstance(saved, dict) or set(saved) != required or saved.get('schema') != SCHEMA:
            raise ValueError('Missing/incompatible strict V6 Muon optimizer checkpoint')
        if saved['manifest'] != self.manifest:
            raise ValueError('Optimizer manifest differs: groups, parameter names, shapes or hyperparameters changed')
        steps = saved['steps_completed']
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('Optimizer steps_completed must be a nonnegative integer')
        initialized = saved['initialized_parameters']
        if not isinstance(initialized, dict) or set(initialized) != {'muon', 'aux_adamw'}:
            raise ValueError('Optimizer initialized-parameter manifest is missing')
        for name, optimizer in (('muon', self.muon_optimizer), ('aux_adamw', self.adamw_optimizer)):
            self._validate_component(name, saved[name], optimizer, initialized[name], steps)
        # Check both components before mutating either optimizer.
        self.muon_optimizer.load_state_dict(saved['muon'])
        self.adamw_optimizer.load_state_dict(saved['aux_adamw'])
        self._refresh_groups()
        self.steps_completed = steps
        for name, optimizer in (('muon', self.muon_optimizer), ('aux_adamw', self.adamw_optimizer)):
            _assert_state_equal(saved[name], optimizer.state_dict(), name)
        self.resume_state_verified = True
        self.restored_parameter_states = {'muon': len(saved['muon']['state']),
                                          'adamw': len(saved['aux_adamw']['state'])}


def build_optimizer(model: nn.Module, training: Mapping) -> V6MuonOptimizer:
    return V6MuonOptimizer(model, training)
