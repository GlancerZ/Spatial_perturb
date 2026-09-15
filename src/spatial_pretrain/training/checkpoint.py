"""Atomic optimizer-boundary checkpoints for multi-allocation training."""
from dataclasses import asdict
import os
import random
import numpy as np
import torch
import torch.distributed as dist


def signature(config, world_size):
    training = asdict(config.training)
    for key in ("checkpoint_every", "auto_resume", "run_update_limit"):
        training.pop(key, None)
    # Output-only option; preserve signatures of checkpoints predating it.
    training.pop("save_initial_checkpoint", None)
    data, model = asdict(config.data), asdict(config.model)
    if not data.get('compartment_bin_edges'):
        data.pop('compartment_bin_edges', None)
    if not model.get('cell_spatial_bias'):
        for key in ('cell_spatial_bias', 'cell_angle_bins', 'cell_distance_edges', 'cell_coordinate_scale'):
            model.pop(key, None)
    return {"data": data, "model": model,
            "masking": asdict(config.masking), "loss": asdict(config.loss),
            "optimizer": asdict(config.optimizer), "training": training,
            "seed": config.experiment.seed, "world_size": world_size}


def rng_state(device):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def save_resume(path, model, optimizer, scaler, config, *, epoch, micro_step,
                global_step, best_val, world_size, rank, device, complete=False):
    local = rng_state(device)
    states = [None] * world_size if rank == 0 else None
    if dist.is_initialized():
        dist.gather_object(local, states, dst=0)
    else:
        states = [local]
    if rank == 0:
        raw_model = model.module if hasattr(model, "module") else model
        payload = {"version": 1, "model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scaler": scaler.state_dict(), "config": config, "signature": signature(config, world_size),
                   "epoch": epoch, "micro_step": micro_step, "step": global_step,
                   "best_val": best_val, "rng_by_rank": states, "complete": complete}
        temporary = path.with_suffix(".pt.partial")
        with temporary.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
        if global_step > 0 and (global_step % 100 == 0 or micro_step == 0 or complete):
            archive = path.parent / "checkpoints"
            archive.mkdir(exist_ok=True)
            destination = archive / f"step_{global_step:08d}.pt"
            try:
                # resume.pt is replaced atomically; archived inodes stay intact.
                os.link(path, destination)
            except FileExistsError:
                pass
            print(f"[archive] {destination.name}", flush=True)
        print(f"[checkpoint] {path.name} epoch={epoch} micro_step={micro_step} update={global_step} complete={complete}", flush=True)
    if dist.is_initialized():
        dist.barrier()


def load_resume(path, model, optimizer, scaler, config, world_size, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != 1 or state.get("signature") != signature(config, world_size):
        raise ValueError("Resume checkpoint does not match the data, model, training configuration or GPU count")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(state.pop("model"), strict=True)
    optimizer.load_state_dict(state.pop("optimizer"))
    scaler.load_state_dict(state.pop("scaler"))
    return state
