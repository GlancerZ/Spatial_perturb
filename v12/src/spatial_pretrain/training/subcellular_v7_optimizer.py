"""Muon for new V7 matrices plus separate AdamW groups for aux/backbone."""
from __future__ import annotations

import copy
import math
from typing import Mapping

import torch
from torch import nn

from .subcellular_v6_optimizer import (
    ADAMW_PREFIXES,
    ADAMW_SETTINGS,
    MUON_SETTINGS,
    _assert_state_equal,
)


SCHEMA = "subcellular_v7_muon_adamw_v1"


def validate_optimizer_config(training: Mapping):
    if training.get("optimizer") != "muon":
        raise ValueError("V7 requires training.optimizer='muon'")
    for name in ("lr_muon", "lr_new", "lr_backbone"):
        value = training.get(name)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError(f"training.{name} must be finite and positive")
    decay = training.get("weight_decay")
    if (isinstance(decay, bool) or not isinstance(decay, (int, float))
            or not math.isfinite(decay) or decay < 0):
        raise ValueError("training.weight_decay must be finite and nonnegative")
    if training.get("muon_adjust_lr_fn") != "match_rms_adamw":
        raise ValueError("V7 requires muon_adjust_lr_fn='match_rms_adamw'")
    if (training.get("muon_momentum", .95) != .95
            or training.get("muon_ns_steps", 5) != 5
            or isinstance(training.get("muon_ns_steps", 5), bool)
            or training.get("muon_nesterov", True) is not True):
        raise ValueError("V7 fixes Muon momentum=.95, ns_steps=5, nesterov=True")


def _partition(model: nn.Module):
    modules = dict(model.named_modules())
    groups = {"muon_new_hidden": [], "adamw_new_aux": [], "adamw_backbone": []}
    protected = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        owner_name, _, leaf = name.rpartition(".")
        owner = modules.get(owner_name)
        if (name.startswith(ADAMW_PREFIXES) or isinstance(owner, (nn.Embedding, nn.LayerNorm))
                or name in ("slot_pool.queries", "slot_pool.missing")
                or leaf in ("bias", "in_proj_bias", "bias_k", "bias_v")):
            protected.add(id(parameter))
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            groups["adamw_backbone"].append((name, parameter))
            continue
        owner_name, _, leaf = name.rpartition(".")
        owner = modules.get(owner_name)
        hidden_matrix = (parameter.ndim == 2 and id(parameter) not in protected
                         and ((isinstance(owner, nn.Linear) and leaf == "weight")
                              or (isinstance(owner, nn.MultiheadAttention)
                                  and leaf in ("in_proj_weight", "q_proj_weight",
                                               "k_proj_weight", "v_proj_weight"))))
        groups["muon_new_hidden" if hidden_matrix else "adamw_new_aux"].append(
            (name, parameter))
    flat = [parameter for entries in groups.values() for _, parameter in entries]
    expected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if (len(flat) != len({id(parameter) for parameter in flat})
            or {id(parameter) for parameter in flat} != {id(parameter) for parameter in expected}):
        raise ValueError("V7 optimizer must cover every trainable tensor exactly once")
    if not all(groups.values()):
        raise ValueError("V7 requires nonempty Muon, new-AdamW and backbone-AdamW groups")
    return groups


def optimizer_manifest(model: nn.Module, training: Mapping) -> dict:
    validate_optimizer_config(training)
    if getattr(model, "backbone_trainable_layers", None) != int(
            training["backbone_trainable_layers"]):
        raise ValueError("Model/config backbone trainable-layer counts differ")
    descriptions = []
    for name, entries in _partition(model).items():
        if name == "muon_new_hidden":
            base_lr, optimizer, settings = training["lr_muon"], "Muon", MUON_SETTINGS
        elif name == "adamw_new_aux":
            base_lr, optimizer, settings = training["lr_new"], "AdamW", ADAMW_SETTINGS
        else:
            base_lr, optimizer, settings = training["lr_backbone"], "AdamW", ADAMW_SETTINGS
        descriptions.append(dict(
            name=name, optimizer=optimizer, base_lr=float(base_lr),
            weight_decay=float(training["weight_decay"]),
            param_names=[entry_name for entry_name, _ in entries],
            param_shapes=[list(parameter.shape) for _, parameter in entries],
            param_dtypes=[str(parameter.dtype) for _, parameter in entries],
            tensor_count=len(entries),
            parameter_count=sum(parameter.numel() for _, parameter in entries),
            settings={key: list(value) if isinstance(value, tuple) else value
                      for key, value in settings.items()}))
    return dict(
        schema=SCHEMA,
        implementation="torch.optim.Muon+torch.optim.AdamW",
        torch_version=str(torch.__version__),
        partition="new_hidden_muon_new_aux_adamw_pretrained_backbone_adamw",
        schedule="shared_multiplier_times_three_base_lrs",
        backbone_trainable_layers=int(training["backbone_trainable_layers"]),
        groups=descriptions,
    )


class V7MuonAdamWOptimizer:
    def __init__(self, model: nn.Module, training: Mapping):
        self.manifest = optimizer_manifest(model, training)
        muon_class = getattr(torch.optim, "Muon", None)
        if not callable(muon_class):
            raise RuntimeError(
                f"torch.optim.Muon is unavailable in torch {torch.__version__}; no fallback")
        partition = _partition(model)
        descriptions = {row["name"]: row for row in self.manifest["groups"]}

        def group(name):
            description = descriptions[name]
            result = dict(
                name=name, base_lr=description["base_lr"], lr=description["base_lr"],
                weight_decay=description["weight_decay"],
                param_names=list(description["param_names"]),
                params=[parameter for _, parameter in partition[name]],
            )
            if name == "muon_new_hidden":
                result["use_muon"] = True
            return result

        self.muon_optimizer = muon_class([group("muon_new_hidden")], **MUON_SETTINGS)
        self.adamw_optimizer = torch.optim.AdamW([
            group("adamw_new_aux"), group("adamw_backbone")], **ADAMW_SETTINGS)
        self._refresh_groups()
        self.steps_completed = 0
        self.resume_state_verified = False
        self.restored_parameter_states = {
            "muon": 0, "new_adamw": 0, "backbone_adamw": 0}

    def _refresh_groups(self):
        self.param_groups = [
            *self.muon_optimizer.param_groups, *self.adamw_optimizer.param_groups]

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
            raise ValueError("LR multiplier must be finite and nonnegative")
        for group in self.param_groups:
            group["lr"] = group["base_lr"] * multiplier

    def learning_rates(self) -> dict:
        return {
            "muon": float(self.muon_optimizer.param_groups[0]["lr"]),
            "new_adamw": float(self.adamw_optimizer.param_groups[0]["lr"]),
            "backbone_adamw": float(self.adamw_optimizer.param_groups[1]["lr"]),
        }

    @staticmethod
    def _initialized(component):
        rows = {}
        for group in component["param_groups"]:
            rows[group["name"]] = [
                parameter_name for identifier, parameter_name in
                zip(group["params"], group["param_names"])
                if identifier in component["state"]]
        return rows

    def state_dict(self) -> dict:
        muon = self.muon_optimizer.state_dict()
        adamw = self.adamw_optimizer.state_dict()
        self._validate_component(
            muon, self.muon_optimizer, steps=self.steps_completed,
            group_kinds={"muon_new_hidden": "muon"})
        self._validate_component(
            adamw, self.adamw_optimizer, steps=self.steps_completed,
            group_kinds={"adamw_new_aux": "adamw", "adamw_backbone": "adamw"})
        return dict(
            schema=SCHEMA, manifest=copy.deepcopy(self.manifest),
            steps_completed=self.steps_completed,
            initialized_parameters={**self._initialized(muon), **self._initialized(adamw)},
            muon=muon, adamw=adamw)

    @staticmethod
    def _validate_component(saved, optimizer, *, steps, group_kinds):
        if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
            raise ValueError("Missing or invalid optimizer component")
        current = optimizer.state_dict()
        if len(saved["param_groups"]) != len(current["param_groups"]):
            raise ValueError("Optimizer group count differs")
        if len(optimizer.param_groups) != len(current["param_groups"]):
            raise ValueError("Live optimizer group count differs")
        seen_identifiers = set()
        for actual, expected, live in zip(
                saved["param_groups"], current["param_groups"], optimizer.param_groups):
            if set(actual) != set(expected):
                raise ValueError("Optimizer group keys differ")
            for key in expected:
                if key != "lr" and actual[key] != expected[key]:
                    raise ValueError(f"Optimizer group field differs: {key}")
            if (not isinstance(actual["lr"], (int, float))
                    or not math.isfinite(actual["lr"]) or actual["lr"] < 0):
                raise ValueError("Saved optimizer learning rate is invalid")
            name = actual.get("name")
            if name not in group_kinds:
                raise ValueError(f"Unexpected optimizer group: {name}")
            identifiers = actual.get("params")
            parameter_names = actual.get("param_names")
            parameters = live.get("params")
            if (not isinstance(identifiers, list) or not isinstance(parameter_names, list)
                    or not isinstance(parameters, list)
                    or len(identifiers) != len(parameter_names)
                    or len(identifiers) != len(parameters)
                    or len(identifiers) != len(set(identifiers))
                    or seen_identifiers.intersection(identifiers)):
                raise ValueError(f"Optimizer identifiers differ in group {name}")
            seen_identifiers.update(identifiers)
            group_state = set(saved["state"]).intersection(identifiers)
            if steps == 0 and group_state:
                raise ValueError(f"Optimizer group {name} has state before any update")
            if steps > 0 and group_state != set(identifiers):
                raise ValueError(f"Optimizer group {name} has incomplete parameter state")
            required = ({"momentum_buffer"} if group_kinds[name] == "muon"
                        else {"step", "exp_avg", "exp_avg_sq"})
            for identifier, parameter in zip(identifiers, parameters):
                if identifier not in group_state:
                    continue
                values = saved["state"][identifier]
                if not isinstance(values, dict) or set(values) != required:
                    raise ValueError(f"Optimizer state fields differ in group {name}")
                for key, value in values.items():
                    expected_shape = torch.Size([]) if key == "step" else parameter.shape
                    if (not torch.is_tensor(value) or value.shape != expected_shape
                            or not bool(torch.isfinite(value).all())):
                        raise ValueError(
                            f"Optimizer state tensor {name}.{key} has invalid shape/value")
                    if key == "step" and (
                            value.item() < 1 or value.item() > steps
                            or value.item() != int(value.item())):
                        raise ValueError("AdamW state step disagrees with completed updates")
        if not set(saved["state"]).issubset(seen_identifiers):
            raise ValueError("Optimizer state contains an unknown parameter identifier")
        if steps == 0 and saved["state"]:
            raise ValueError("Optimizer has state before any completed update")
        if steps > 0 and not saved["state"]:
            raise ValueError("Optimizer state is absent after completed updates")

    def load_state_dict(self, saved: dict):
        required = {"schema", "manifest", "steps_completed", "initialized_parameters",
                    "muon", "adamw"}
        if not isinstance(saved, dict) or set(saved) != required or saved.get("schema") != SCHEMA:
            raise ValueError("Missing or incompatible V7 optimizer checkpoint")
        if saved["manifest"] != self.manifest:
            raise ValueError("V7 optimizer manifest differs")
        steps = saved["steps_completed"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid optimizer step count")
        self._validate_component(
            saved["muon"], self.muon_optimizer, steps=steps,
            group_kinds={"muon_new_hidden": "muon"})
        self._validate_component(
            saved["adamw"], self.adamw_optimizer, steps=steps,
            group_kinds={"adamw_new_aux": "adamw", "adamw_backbone": "adamw"})
        initialized = {**self._initialized(saved["muon"]), **self._initialized(saved["adamw"])}
        if saved["initialized_parameters"] != initialized:
            raise ValueError("Initialized optimizer parameter manifest differs")
        self.muon_optimizer.load_state_dict(saved["muon"])
        self.adamw_optimizer.load_state_dict(saved["adamw"])
        self._refresh_groups()
        self.steps_completed = steps
        _assert_state_equal(saved["muon"], self.muon_optimizer.state_dict(), "muon")
        _assert_state_equal(saved["adamw"], self.adamw_optimizer.state_dict(), "adamw")
        self.resume_state_verified = True
        muon_rows = self._initialized(saved["muon"])
        adamw_rows = self._initialized(saved["adamw"])
        self.restored_parameter_states = {
            "muon": len(muon_rows["muon_new_hidden"]),
            "new_adamw": len(adamw_rows["adamw_new_aux"]),
            "backbone_adamw": len(adamw_rows["adamw_backbone"]),
        }


def build_optimizer(model: nn.Module, training: Mapping) -> V7MuonAdamWOptimizer:
    return V7MuonAdamWOptimizer(model, training)
