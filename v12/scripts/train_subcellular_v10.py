"""Train V10 on an existing GPU allocation, with allocation-independent batches.

This program never requests resources. Every optimizer update sees the same
global 64-center window, including differentiably gathered predictions on DDP.
An overnight pilot is explicitly one pass over its saved fixed subset, not a
claim to have completed an epoch over all training centers.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as differentiable_all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
import yaml

from spatial_pretrain.data.subcellular_v10 import V10Dataset, collate_v10
from spatial_pretrain.models.subcellular_v10 import build_model
from spatial_pretrain.training.subcellular_v10 import losses
from spatial_pretrain.training.subcellular_v6_optimizer import V6MuonOptimizer, validate_optimizer_config

MODEL_VERSION = 10
ARCHITECTURE = "nucleus_neighbor_teacher_v10"
PAUSE_MARKERS = ("STOP_REQUESTED", "PAUSE_REQUESTED", "HANDOFF_TO_4GPU_REQUESTED")
SOURCE_FILES = (
    "scripts/train_subcellular_v10.py",
    "src/spatial_pretrain/data/subcellular_v10.py",
    "src/spatial_pretrain/data/subcellular_v2.py",
    "src/spatial_pretrain/data/subcellular_v3.py",
    "src/spatial_pretrain/data/subcellular_v5.py",
    "src/spatial_pretrain/data/subcellular_v6.py",
    "src/spatial_pretrain/data/subcellular_v6_left.py",
    "src/spatial_pretrain/data/subcellular_v7.py",
    "src/spatial_pretrain/models/subcellular_v10.py",
    "src/spatial_pretrain/models/scgpt_backbone.py",
    "src/spatial_pretrain/training/subcellular_v10.py",
    "src/spatial_pretrain/training/subcellular_v6_optimizer.py",
)
_SIGNAL_REQUESTED = False


def _signal_handler(signum, frame):
    global _SIGNAL_REQUESTED
    del signum, frame
    _SIGNAL_REQUESTED = True


def atomic_json(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_hash():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def rank_zero_action(action, rank, world):
    """Propagate a rank-zero filesystem error instead of hanging at a barrier."""
    error = [None]
    result = None
    if rank == 0:
        try:
            result = action()
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    return result


def validate_config(cfg, world):
    training = cfg["training"]
    validate_optimizer_config(training)
    if cfg.get("binning", {}).get("tie_policy") != "left":
        raise ValueError("V10 requires explicit deterministic-left binning")
    if cfg.get("binning", {}).get("num_bins", 51) != 51:
        raise ValueError("V10 retains the 51-bin ordinary scGPT contract")
    if cfg.get("model", {}).get("backbone_kind", "scgpt") != "scgpt":
        raise ValueError("V10 cannot initialize from scGPT-spatial")
    if cfg.get("model", {}).get("predictor_dropout", 0.) != 0.:
        raise ValueError("Allocation-independent V10 DDP requires predictor_dropout=0")
    if training.get("backbone_trainable_layers", 0) != 0:
        raise ValueError("V10 freezes both teacher and student scGPT encoders")
    for key, default in (("epochs", 1), ("global_batch_size", 64), ("checkpoint_every", 100),
                         ("resume_every", 25), ("log_every", 10)):
        value = training.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training.get("epochs", 1) != 1 or training.get("global_batch_size", 64) != 64:
        raise ValueError("The V10 first comparison fixes epochs=1 and global_batch_size=64")
    if world not in (1, 2, 4) or 64 % world:
        raise ValueError("V10 supports one, two or four physical GPU ranks for global batch 64")
    if training.get("num_workers", 4) < 0:
        raise ValueError("num_workers cannot be negative")
    for key, default in (("warmup_ratio", .05), ("min_lr_ratio", .1)):
        if not 0 <= training.get(key, default) <= 1:
            raise ValueError(f"training.{key} must lie in [0,1]")
    if training.get("grad_clip_norm", 1.) <= 0 or training.get("save_margin_seconds", 180) < 0:
        raise ValueError("Invalid gradient clipping or deadline margin")
    settings = cfg.get("v10", {})
    count = settings.get("pilot_centers", 65536)
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("v10.pilot_centers must be a nonnegative integer; zero means all centers")
    if cfg.get("model", {}).get("neighbors", 16) != settings.get("neighbors", 8):
        raise ValueError("model.neighbors and v10.neighbors must agree")
    if cfg.get("loss", {}).get("type") not in ("cosine", "contrastive", "relational"):
        raise ValueError("Choose one V10 objective: cosine, contrastive or relational")


class GlobalBatchSampler(Sampler):
    """Shard deterministic complete windows, with a cursor in global updates."""
    def __init__(self, indices, global_batch=64, world=1, rank=0, seed=42):
        self.indices = np.asarray(indices, dtype=np.int64)
        if not len(self.indices) or global_batch % world or not 0 <= rank < world:
            raise ValueError("Invalid selected centers or global batch partition")
        self.global_batch, self.world, self.rank, self.seed = global_batch, world, rank, seed
        self.local_batch = global_batch // world
        self.updates = math.ceil(len(self.indices) / global_batch)
        self.epoch, self.start_update = 0, 0

    def seek(self, epoch, start_update):
        if not 0 <= start_update <= self.updates or epoch < 0:
            raise ValueError("Invalid optimizer-boundary cursor")
        self.epoch, self.start_update = int(epoch), int(start_update)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.indices), generator=generator).numpy()
        order = np.resize(order, self.updates * self.global_batch)
        for update in range(self.start_update, self.updates):
            offset = update * self.global_batch + self.rank * self.local_batch
            yield self.indices[order[offset:offset + self.local_batch]].tolist()

    def __len__(self):
        return self.updates - self.start_update


def fixed_subset(dataset, cfg):
    """Select a fixed random subset of centers with nonempty whole-cell targets.

    Inspect only central cached log-count rows, grouped by slice and bounded in
    memory; do not tokenize samples, reconstruct counts or scan entire expression
    matrices. The permutation is shared by all objectives and independent of
    physical rank count. Statistics describe every candidate actually read,
    including the unused tail of the final candidate block.
    """
    dataset_size = len(dataset)
    count = int(cfg.get("v10", {}).get("pilot_centers", 65536))
    seed = int(cfg.get("v10", {}).get("pilot_seed", cfg["data"].get("seed", 42)))
    if dataset_size < 1 or count < 0 or count > dataset_size:
        raise ValueError(f"Cannot select requested {count} eligible centers from {dataset_size} cached centers")
    permutation = np.random.default_rng(seed).permutation(dataset_size)
    ends = np.asarray(dataset.ends, dtype=np.int64)
    starts = np.concatenate((np.zeros(1, dtype=np.int64), ends[:-1]))
    selected, selected_count = [], 0
    checked, excluded, observed = 0, 0, 0
    for offset in range(0, dataset_size, 4096):
        candidates = permutation[offset:offset + 4096]
        slice_ids = np.searchsorted(ends, candidates, side="right")
        eligible = np.zeros(len(candidates), dtype=bool)
        for slice_id in np.unique(slice_ids):
            positions = np.flatnonzero(slice_ids == slice_id)
            entry = dataset.slices[int(slice_id)]
            for block_start in range(0, len(positions), 1024):
                block_positions = positions[block_start:block_start + 1024]
                local = candidates[block_positions] - starts[slice_id]
                centers = np.asarray(entry["nodes"][local, 0], dtype=np.int64)
                if (centers < 0).any():
                    raise ValueError("Cached patch center is padding; cannot define a teacher")
                nuclear = np.asarray(entry["nuclear"][centers])
                cyto = np.asarray(entry["cyto"][centers])
                if (not np.isfinite(nuclear).all() or not np.isfinite(cyto).all()
                        or (nuclear < 0).any() or (cyto < 0).any()):
                    raise ValueError("Invalid cached expression in teacher-center eligibility scan")
                eligible[block_positions] = (nuclear > 0).any(axis=1) | (cyto > 0).any(axis=1)
        checked += len(candidates)
        found = candidates[eligible]
        observed += len(found)
        excluded += int((~eligible).sum())
        take = found if count == 0 else found[:max(0, count - selected_count)]
        selected.append(take)
        selected_count += len(take)
        if count and selected_count == count:
            break
    if not selected_count or (count and selected_count < count):
        raise ValueError(
            f"Insufficient nonempty whole-cell teachers: requested={count or 'all'}, "
            f"eligible_found={selected_count}, candidate_checked={checked}, zero_teacher_excluded={excluded}")
    indices = np.sort(np.concatenate(selected)).astype(np.int64)
    stats = dict(
        policy="fixed_seed_candidate_permutation_keep_center_any_nuclear_or_cytoplasmic_logcount_positive",
        requested_centers=count, candidate_checked=checked, zero_teacher_excluded=excluded,
        eligible_candidates_observed=observed, scan_exhausted=checked == dataset_size,
        total_eligible_centers=observed if checked == dataset_size else None,
        candidate_block_size=4096, maximum_rows_per_slice_read=1024,
        statistics_scope="all candidates read, including unused tail of final candidate block")
    return indices, stats


def gather_plain(tensor, world):
    if world == 1:
        return tensor.detach()
    result = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(result, tensor.detach().contiguous())
    return torch.cat(result, dim=0)


def gather_global_output(output, batch, world):
    if world == 1:
        return output, {"cell_identity": batch["cell_identity"]}
    # Every rank computes the same loss on the complete window. Autograd gather
    # sums rank contributions in backward; DDP then averages parameter gradients,
    # yielding exactly the single-global-batch objective (no extra world factor).
    combined = {
        name: torch.cat(differentiable_all_gather(output[name].contiguous()), dim=0)
        for name in ("nucleus_prediction", "context_prediction")
    }
    combined["teacher"] = gather_plain(output["teacher"], world)
    combined["context_available"] = gather_plain(output["context_available"].to(torch.uint8), world).bool()
    combined["nucleus_available"] = gather_plain(output["nucleus_available"].to(torch.uint8), world).bool()
    identity = gather_plain(batch["cell_identity"], world)
    return combined, {"cell_identity": identity}


def lr_multiplier(step, total_updates, training):
    warmup = max(1, round(total_updates * training.get("warmup_ratio", .05)))
    if step <= warmup:
        return step / warmup
    fraction = min(1., (step - warmup) / max(1, total_updates - warmup))
    minimum = training.get("min_lr_ratio", .1)
    return minimum + (1 - minimum) * .5 * (1 + math.cos(math.pi * fraction))


def validate_resume(state, cfg, sources, model, optimizer, dataset_info, total_updates):
    if state.get("model_version") != MODEL_VERSION or state.get("architecture") != ARCHITECTURE:
        raise ValueError("Only a matching V10 checkpoint can resume")
    expected = dict(config=cfg, config_sha256=config_hash(cfg), source_sha256=sources,
                    arch_config=model.arch_config, backbone_provenance=model.backbone.provenance,
                    optimizer_manifest=optimizer.manifest, dataset_info=dataset_info,
                    global_batch_size=64, total_updates=total_updates)
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"Resume {key} differs; explicit experiment revision required")
    step = state.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= total_updates:
        raise ValueError("Resume step is not a valid optimizer boundary")
    epoch, update = divmod(step, total_updates)
    if state.get("epoch") != epoch or state.get("next_update") != update:
        raise ValueError("Resume epoch/update cursor disagrees with completed steps")
    if bool(state.get("complete")) != (step == total_updates):
        raise ValueError("Resume completion flag is inconsistent")
    if state.get("optimizer", {}).get("steps_completed") != step or "model" not in state:
        raise ValueError("Resume lacks a consistent model/optimizer pair")
    return step


def _train(cfg, args, run, rank, world):
    training = cfg["training"]
    local = int(os.environ.get("LOCAL_RANK", "0"))
    seed = int(cfg["data"].get("seed", 42))
    torch.set_num_threads(int(training.get("torch_threads", 4)))
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = V10Dataset(cfg, split="train")
    selection = rank_zero_action(lambda: fixed_subset(dataset, cfg), rank, world)
    if world > 1:
        selection_payload = [selection]
        dist.broadcast_object_list(selection_payload, src=0)
        selection = selection_payload[0]
    indices, selection_stats = selection
    sampler = GlobalBatchSampler(indices, 64, world, rank, seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_v10,
                        num_workers=int(training.get("num_workers", 4)), pin_memory=True,
                        persistent_workers=False)
    total_updates = sampler.updates
    dataset_info = dict(
        full_training_centers=len(dataset), selected_centers=len(indices),
        subset_sha256=hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest(),
        cache_manifest_sha256=config_hash(dataset.manifest),
        pilot_seed=int(cfg.get("v10", {}).get("pilot_seed", seed)),
        scope="all_eligible_training_centers" if cfg.get("v10", {}).get("pilot_centers", 65536) == 0
        else "pilot_subset_of_eligible_centers",
        teacher_eligibility=selection_stats,
        repeated_centers_at_epoch_end=total_updates * 64 - len(indices),
        center_stride=cfg["data"].get("center_stride"))
    fingerprint, sources = config_hash(cfg), source_hash()
    model = build_model(cfg).cuda()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = V6MuonOptimizer(model, training)
    step = 0
    if args.resume:
        state = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
        step = validate_resume(state, cfg, sources, model, optimizer, dataset_info, total_updates)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        expected_lr = lr_multiplier(step, total_updates, training) if step else 1.
        if any(group["lr"] != group["base_lr"] * expected_lr for group in optimizer.param_groups):
            raise ValueError("Resume learning rate does not match the fixed full-pilot schedule")
        del state
    wrapped = DDP(model, device_ids=[local], find_unused_parameters=False) if world > 1 else model
    rank_info = dict(rank=rank, local_rank=local, hostname=socket.gethostname(),
                     gpu_name=torch.cuda.get_device_name(local), jobid=os.environ.get("SLURM_JOB_ID"))
    topology = [None] * world if world > 1 else [rank_info]
    if world > 1:
        dist.all_gather_object(topology, rank_info)
    initial_step, began = step, time.monotonic()
    deadline = args.deadline if args.deadline is not None else float(os.environ.get("V10_TRAIN_DEADLINE", "inf"))
    margin = float(training.get("save_margin_seconds", 180))
    torch.cuda.reset_peak_memory_stats()

    def progress(status, extra=None):
        seconds = (time.monotonic() - began) / max(1, step - initial_step)
        payload = dict(status=status, step=step, total_updates=total_updates,
                       epoch_complete=step == total_updates,
                       full_dataset_epoch_complete=step == total_updates and len(indices) == len(dataset),
                       all_eligible_centers_epoch_complete=step == total_updates
                       and selection_stats["total_eligible_centers"] == len(indices),
                       dataset_info=dataset_info, physical_world_size=world,
                       batch_per_card=sampler.local_batch, global_batch_size=64,
                       optimizer_lrs=optimizer.learning_rates(), updated_at=time.time(),
                       config_sha256=fingerprint, architecture=ARCHITECTURE, model_version=MODEL_VERSION,
                       seconds_per_update=seconds if step > initial_step else None,
                       eta_train_hours=(total_updates - step) * seconds / 3600 if step > initial_step else None)
        if extra:
            payload.update(extra)
        return payload

    def save(*, archive=False, status="paused", reason=None):
        if optimizer.steps_completed != step:
            raise ValueError("Optimizer and trainer update counts disagree before saving")
        complete = step == total_updates
        epoch, next_update = divmod(step, total_updates)
        def write():
            common = dict(
                model=model.state_dict(), config=cfg, config_sha256=fingerprint, source_sha256=sources,
                model_version=MODEL_VERSION, architecture=ARCHITECTURE, arch_config=model.arch_config,
                backbone_provenance=model.backbone.provenance, step=step,
                dataset_info=dataset_info, total_updates=total_updates, global_batch_size=64,
                optimizer_manifest=optimizer.manifest)
            if archive:
                path = run / "checkpoints" / f"step_{step:06d}.pt"
                if path.exists():
                    # A resume checkpoint can lag an already committed archive
                    # if a previous allocation ended between atomic writes.
                    old = torch.load(path, map_location="cpu", weights_only=False)
                    if any(old.get(key) != common[key] for key in
                           ("step", "config_sha256", "source_sha256", "dataset_info")):
                        raise FileExistsError(f"Incompatible archived checkpoint exists: {path}")
                    if any(not torch.equal(value.detach().cpu(), old["model"][key])
                           for key, value in common["model"].items()):
                        raise FileExistsError(f"Refusing to replace divergent archived weights: {path}")
                    del old
                else:
                    atomic_save(common, path)
            atomic_save(dict(**common, optimizer=optimizer.state_dict(), epoch=epoch,
                             next_update=next_update, complete=complete, physical_world_size=world,
                             batch_per_card=sampler.local_batch), run / "resume.pt")
            record = progress("complete" if complete else status, {"stop_reason": reason})
            atomic_json(record, run / "progress.json")
            if complete:
                atomic_json(record, run / "TRAIN_COMPLETE.json")
            print("[checkpoint]", json.dumps(dict(step=step, complete=complete, reason=reason)), flush=True)
        rank_zero_action(write, rank, world)

    def write_schedule():
        schedule = dict(
            **dataset_info, total_updates=total_updates, updates_per_epoch=total_updates,
            epoch_definition="one pass over saved eligible-center subset; scope excludes zero-expression teachers",
            physical_world_size=world, batch_per_card=sampler.local_batch, global_batch_size=64,
            accumulation=1, contrastive_bank="all ranks, differentiable prediction all_gather",
            rank_topology=topology, loss=cfg["loss"], config_sha256=fingerprint,
            source_sha256=sources, arch_config=model.arch_config,
            backbone_provenance=model.backbone.provenance, optimizer_manifest=optimizer.manifest,
            optimizer_resume_state_verified=optimizer.resume_state_verified,
            trainable_parameters=sum(p.numel() for p in parameters),
            frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
            initial_step=step, resumed=args.resume, model_version=MODEL_VERSION, architecture=ARCHITECTURE)
        atomic_json(schedule, run / "schedule.json")
        (run / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        selected_path = run / "selected_center_indices.npy"
        if not selected_path.exists():
            with selected_path.with_suffix(".npy.partial").open("wb") as handle:
                np.save(handle, indices)
            selected_path.with_suffix(".npy.partial").replace(selected_path)
        print("[schedule]", json.dumps(schedule, allow_nan=False), flush=True)
    rank_zero_action(write_schedule, rank, world)
    if step == total_updates:
        save(status="complete", reason="already_complete")
        return
    if step == 0 and not args.resume:
        save(archive=True, status="initialized")
    if args.initialize_only:
        return
    pre_stop = torch.tensor(int(_SIGNAL_REQUESTED or time.time() >= deadline - margin), device="cuda")
    if world > 1:
        dist.all_reduce(pre_stop, op=dist.ReduceOp.MAX)
    if pre_stop.item():
        save(status="paused", reason="deadline_before_first_update")
        return
    dataset.set_epoch(0)
    sampler.seek(0, step)
    wrapped.train()
    if model.backbone.training or any(p.requires_grad for p in model.backbone.parameters()):
        raise RuntimeError("V10 fixed teacher unexpectedly entered trainable/training mode")
    for batch in loader:
        batch = {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = wrapped(batch)
        combined, global_batch = gather_global_output(output, batch, world)
        loss, parts = losses(combined, global_batch, cfg, step + 1)
        if not parts.get("has_learning_signal", False):
            raise ValueError(
                f"Global V10 batch before update {step + 1} has no available nuclear or context views; "
                "refusing an optimizer/weight-decay update without learning signal")
        finite = torch.tensor(int(bool(torch.isfinite(loss))), device="cuda")
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"Nonfinite V10 loss before update {step + 1}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, training.get("grad_clip_norm", 1.),
                                             error_if_nonfinite=True)
        optimizer.set_lr_multiplier(lr_multiplier(step + 1, total_updates, training))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        # Synchronize just once per step before timing/checkpoint decisions.
        torch.cuda.synchronize()
        peaks = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                             device="cuda", dtype=torch.float64)
        if world > 1:
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        if step <= 3 or step % int(training.get("log_every", 10)) == 0 or step == total_updates:
            # Collapse diagnostics are computed from the same global teacher bank.
            teacher = combined["teacher"].detach().float()
            nucleus = combined["nucleus_prediction"].detach().float()
            context = combined["context_prediction"].detach().float()
            available = combined["context_available"]
            nucleus_available = combined["nucleus_available"]
            similarity = teacher @ teacher.T
            off_diagonal = ~torch.eye(len(teacher), dtype=torch.bool, device=teacher.device)
            diagnostics = dict(
                teacher_offdiag_cosine=float(similarity[off_diagonal].mean()),
                teacher_feature_std=float(teacher.std(0, unbiased=False).mean()),
                nucleus_feature_std=float(nucleus[nucleus_available].std(0, unbiased=False).mean())
                if nucleus_available.any() else 0.,
                context_feature_std=float(context[available].std(0, unbiased=False).mean()) if available.any() else 0.)
            memory = peaks.tolist()
            record = progress("running", dict(
                loss=parts, gradient_norm=float(norm), diagnostics=diagnostics,
                allocated_gib=memory[0] / 2**30, reserved_gib=memory[1] / 2**30,
                memory_scope="maximum over ranks since launch", loss_finite=True, gradients_finite=True,
                jobid=os.environ.get("SLURM_JOB_ID")))
            def write_metrics():
                with (run / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record, allow_nan=False) + "\n")
                atomic_json(record, run / "progress.json")
                print("[train]", json.dumps(record, allow_nan=False), flush=True)
            rank_zero_action(write_metrics, rank, world)
        reason = None
        if _SIGNAL_REQUESTED:
            reason = "termination_signal"
        elif time.time() >= deadline - margin:
            reason = "allocation_deadline"
        elif any((run / marker).exists() for marker in PAUSE_MARKERS):
            reason = "pause_marker"
        elif args.run_update_limit and step - initial_step >= args.run_update_limit:
            reason = "run_update_limit"
        stop = torch.tensor(int(reason is not None), device="cuda")
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        should_stop = bool(stop.item())
        complete = step == total_updates
        archive = step % int(training.get("checkpoint_every", 100)) == 0 or complete or should_stop
        if archive or step % int(training.get("resume_every", 25)) == 0:
            save(archive=archive, status="paused" if should_stop else "running",
                 reason=reason or ("another_rank_requested_stop" if should_stop else None))
        if should_stop or complete:
            return
    raise RuntimeError("Training iterator ended before the declared number of global updates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Explicitly restore output_dir/resume.pt")
    parser.add_argument("--run-update-limit", type=int, default=0,
                        help="Pause after N additional updates; zero runs the full planned subset")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--deadline", type=float, default=None, help="Allocation end time as Unix seconds")
    args = parser.parse_args()
    if args.run_update_limit < 0:
        parser.error("--run-update-limit must be nonnegative")
    cfg = yaml.safe_load(args.config.resolve().read_text())
    os.chdir(ROOT)
    rank, world, local = (int(os.environ.get(key, "0" if key != "WORLD_SIZE" else "1"))
                          for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    validate_config(cfg, world)
    if not torch.cuda.is_available():
        raise RuntimeError("V10 training requires CUDA on an authorized compute allocation")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    lock = None
    run = Path(cfg["output_dir"])
    try:
        def prepare():
            nonlocal lock
            if args.resume:
                if not (run / "resume.pt").is_file():
                    raise FileNotFoundError("--resume requires an existing V10 resume.pt")
            elif run.exists() and any(run.iterdir()):
                raise FileExistsError("Output directory is nonempty; choose new output or explicitly --resume")
            if any((run / marker).exists() for marker in PAUSE_MARKERS):
                raise RuntimeError("Pause/handoff marker exists; launcher must resolve it before resume")
            run.mkdir(parents=True, exist_ok=True)
            lock = (run / ".trainer.lock").open("a")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            (run / "checkpoints").mkdir(exist_ok=True)
        rank_zero_action(prepare, rank, world)
        _train(cfg, args, run, rank, world)
    finally:
        if lock is not None:
            lock.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
