from __future__ import annotations

import torch


class MuonWithAuxAdam:
    """Wrapper that delegates to PyTorch's official Muon and AdamW optimizers.

    Hidden 2D block weights use ``torch.optim.Muon`` while all remaining
    parameters stay on ``torch.optim.AdamW``. The wrapper keeps a combined
    ``param_groups`` view so the existing training loop can continue to update
    learning rates, unscale gradients, and zero grads as if there were a single
    optimizer.
    """

    def __init__(self, param_groups: list[dict], *, muon_adjust_lr_fn: str | None = None):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError(
                "torch.optim.Muon is unavailable in this PyTorch build. "
                f"Installed torch version: {torch.__version__}"
            )

        muon_groups: list[dict] = []
        aux_groups: list[dict] = []
        for group in param_groups:
            params = list(group["params"])
            if not params:
                continue

            normalized = {
                "params": params,
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "lr_scale": float(group.get("lr_scale", 1.0)),
                "use_muon": bool(group["use_muon"]),
            }
            if normalized["use_muon"]:
                normalized.update(
                    {
                        "momentum": float(group.get("momentum", 0.95)),
                        "nesterov": bool(group.get("nesterov", True)),
                        "ns_steps": int(group.get("ns_steps", 5)),
                    }
                )
                muon_groups.append(normalized)
            else:
                normalized.update(
                    {
                        "betas": tuple(group.get("betas", (0.9, 0.999))),
                        "eps": float(group.get("eps", 1e-8)),
                    }
                )
                aux_groups.append(normalized)

        muon_kwargs = {}
        if muon_adjust_lr_fn:
            muon_kwargs["adjust_lr_fn"] = muon_adjust_lr_fn

        self._muon_optimizer = torch.optim.Muon(muon_groups, **muon_kwargs) if muon_groups else None
        self._aux_optimizer = torch.optim.AdamW(aux_groups) if aux_groups else None

        self.param_groups: list[dict] = []
        if self._muon_optimizer is not None:
            self.param_groups.extend(self._muon_optimizer.param_groups)
        if self._aux_optimizer is not None:
            self.param_groups.extend(self._aux_optimizer.param_groups)

    @property
    def state(self) -> dict:
        merged: dict = {}
        if self._muon_optimizer is not None:
            merged.update(self._muon_optimizer.state)
        if self._aux_optimizer is not None:
            merged.update(self._aux_optimizer.state)
        return merged

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon_optimizer is not None:
            self._muon_optimizer.step()
        if self._aux_optimizer is not None:
            self._aux_optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self._muon_optimizer is not None:
            self._muon_optimizer.zero_grad(set_to_none=set_to_none)
        if self._aux_optimizer is not None:
            self._aux_optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, dict | None]:
        return {
            "muon": None if self._muon_optimizer is None else self._muon_optimizer.state_dict(),
            "aux_adamw": None if self._aux_optimizer is None else self._aux_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict[str, dict | None]) -> None:
        if self._muon_optimizer is not None and state_dict.get("muon") is not None:
            self._muon_optimizer.load_state_dict(state_dict["muon"])
        if self._aux_optimizer is not None and state_dict.get("aux_adamw") is not None:
            self._aux_optimizer.load_state_dict(state_dict["aux_adamw"])
        # load_state_dict replaces the underlying group dictionaries. Refresh
        # the wrapper view so resumed LR scheduling updates the actual groups.
        self.param_groups = []
        for optimizer in (self._muon_optimizer, self._aux_optimizer):
            if optimizer is not None:
                self.param_groups.extend(optimizer.param_groups)
