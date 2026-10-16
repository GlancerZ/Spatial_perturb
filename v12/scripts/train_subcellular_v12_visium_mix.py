"""Budget-matched V12 continuation; resource allocation is exclusively external.

Each update contains the same global 64-center window. Per-mode cell-loss sums
use counts over the complete global update, not averages of microbatch means.
Checkpoint resume restores optimizer moments and the global sample cursor.
Dropout is seeded by update/rank/microbatch: same-topology resumes reproduce it;
changing world size preserves the batch/objective but changes dropout draws.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
import json
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import yaml

from train_subcellular_v10 import (
    atomic_json, atomic_save, config_hash, lr_multiplier, rank_zero_action,
)
from train_subcellular_v12 import SOURCE_FILES as ORIGINAL_SOURCE_FILES
from evaluate_subcellular_v12 import load_model as verify_initial_checkpoint, file_sha256
from spatial_pretrain.data.subcellular_v12 import collate_v12
from spatial_pretrain.data.visium_hd_mix import V12MixedDataset
from spatial_pretrain.data.visium_mix_sampler import FixedDomainBatchSampler
from spatial_pretrain.models.subcellular_v12 import build_model
from spatial_pretrain.training.subcellular_v12 import (
    loss_counts_from_batch, loss_sums_and_counts, reduce_loss_sums,
)
from spatial_pretrain.training.subcellular_v7_optimizer import V7MuonAdamWOptimizer, validate_optimizer_config

MODEL_VERSION = 12
ARCHITECTURE = "subcellular_compartment_counts_v12"
PAUSE_MARKERS = ("STOP_REQUESTED", "PAUSE_REQUESTED")
SOURCE_FILES = tuple(dict.fromkeys((*ORIGINAL_SOURCE_FILES,
    "scripts/train_subcellular_v12_visium_mix.py",
    "scripts/evaluate_subcellular_v12.py",
    "scripts/prepare_visium_hd_v12_cache.py",
    "scripts/prepare_visium_hd_mix_compartments_20261011.py",
    "scripts/prepare_subcellular_v2.py",
    "scripts/prepare_unified_training_cache.py",
    "src/spatial_pretrain/data/visium_hd_mix.py",
    "src/spatial_pretrain/data/visium_mix_sampler.py")))
_STOP = False


def on_signal(signum, frame):
    global _STOP
    _STOP = True


def validate_config(cfg, world):
    tr, model, views = cfg["training"], cfg["model"], cfg["v12"]
    validate_optimizer_config(tr)
    if cfg["binning"] != {"num_bins": 51, "tie_policy": "left"}:
        raise ValueError("V12 requires deterministic-left 51 bins")
    if model["backbone_kind"] != "scgpt":
        raise ValueError("Only ordinary scGPT is permitted")
    if model["backbone_trainable_layers"] != tr["backbone_trainable_layers"]:
        raise ValueError("Model and optimizer trainable layers disagree")
    if tr["epochs"] != 1 or tr["global_batch_size"] != 64 or world not in (1, 2, 4):
        raise ValueError("This experiment uses one epoch, global batch 64 and 1/2/4 ranks")
    for key in ("micro_batch_size", "checkpoint_every", "resume_every", "log_every", "torch_threads"):
        if isinstance(tr[key], bool) or not isinstance(tr[key], int) or tr[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if 64 // world % tr["micro_batch_size"]:
        raise ValueError("Microbatch must divide the local update batch")
    if model["neighbors"] != views["neighbors"] or model["max_neighbor_distance"] != views["max_neighbor_distance"]:
        raise ValueError("Data/model neighbor definitions disagree")
    if cfg["loss"]["type"] != "joint_compartment_count_nll":
        raise ValueError("Only joint nuclear/cytoplasmic count likelihood is enabled")
    if tr["num_workers"] < 0 or tr["grad_clip_norm"] <= 0 or tr["save_margin_seconds"] < 0:
        raise ValueError("Invalid loader, clipping or deadline settings")
    if any(not 0 <= tr[key] <= 1 for key in ("warmup_ratio", "min_lr_ratio")):
        raise ValueError("Invalid learning-rate schedule")
    if isinstance(tr.get("additional_updates"), bool) or not isinstance(tr.get("additional_updates"), int) or tr["additional_updates"] < 1:
        raise ValueError("An explicit positive continuation update budget is required")
    if tr.get("visium_centers_per_update") not in (0,16):
        raise ValueError("This matched experiment uses 0 or 16 Visium centers per 64-center update")
    if cfg["continuation"]["source_step"] != 10234:
        raise ValueError("This experiment starts from the verified V12 step 10234")


def seed_microbatch(seed, step, rank, micro):
    value = int(np.random.SeedSequence([seed, step, rank, micro, 1100]).generate_state(1)[0])
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


def train(cfg, args, run, rank, world):
    tr, seed = cfg["training"], cfg["data"].get("seed", 42)
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.set_num_threads(tr["torch_threads"])
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = V12MixedDataset(cfg, split="train", include_visium=tr["visium_centers_per_update"] > 0)
    sampler = FixedDomainBatchSampler(dataset.xenium_indices,
        dataset.visium_indices_by_sample, updates=tr["additional_updates"],
        visium_centers=tr["visium_centers_per_update"], world=world, rank=rank, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_v12,
                        num_workers=tr["num_workers"], pin_memory=True,
                        generator=torch.Generator().manual_seed(seed), persistent_workers=False)
    total, micro_size = sampler.updates, tr["micro_batch_size"]
    info = dict(full_training_centers=len(dataset),
                xenium_centers=len(dataset.xenium_indices), visium_centers=len(dataset.visium_indices),
                sampler_contract=sampler.contract, sampler_sha256=sampler.fingerprint,
                cache_manifest_sha256=config_hash(dataset.manifest),
                center_filter_metadata=dataset.center_filter_metadata,
                target_metadata=dataset.target_metadata, lr_metadata=dataset.lr_metadata,
                scope="fixed_update_budget_not_corpus_epoch",
                center_stride=cfg["data"]["center_stride"])
    fingerprint = config_hash(cfg)
    sources = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}
    model = build_model(cfg).cuda()
    initialization = dict(checkpoint=cfg["continuation"]["checkpoint"],
        sha256=file_sha256(Path(cfg["continuation"]["checkpoint"])),
        source_step=int(cfg["continuation"]["source_step"]),
        semantics="V12 weights only; new optimizer and fixed-budget sampler; not old-run resume")
    if initialization["sha256"] != cfg["continuation"]["sha256"]:
        raise ValueError("Initialization checkpoint content mismatch")
    if not args.resume:
        previous, baseline, old_cfg, source_step, provenance = verify_initial_checkpoint(
            Path(initialization["checkpoint"]), initialization["source_step"])
        if previous.arch_config != model.arch_config:
            raise ValueError("Continuation architecture differs from the initialization")
        if previous.backbone.provenance != model.backbone.provenance:
            raise ValueError("Continuation scGPT asset contents differ from the initialization")
        for key in ("model", "v12", "data", "binning", "loss", "lr"):
            if cfg[key] != old_cfg[key]:
                raise ValueError(f"Matched continuation must preserve {key}")
        for key in ("optimizer", "lr_muon", "lr_new", "lr_backbone", "weight_decay", "grad_clip_norm",
                    "warmup_ratio", "min_lr_ratio", "backbone_trainable_layers", "muon_momentum",
                    "muon_nesterov", "muon_ns_steps", "muon_adjust_lr_fn"):
            if tr[key] != old_cfg["training"][key]:
                raise ValueError(f"Matched continuation must preserve training.{key}")
        model.load_state_dict(previous.state_dict(), strict=True)
        del previous, baseline
        torch.cuda.empty_cache()
        # The verifier returns frozen inference models. The fresh build above
        # preserves the configured trainable layers and is the trained object.
        if not any(p.requires_grad for p in model.backbone.parameters()):
            raise ValueError("Configured scGPT backbone layers are unexpectedly frozen")
    optimizer = V7MuonAdamWOptimizer(model, tr)
    parameters = [p for p in model.parameters() if p.requires_grad]
    step = 0
    if args.resume:
        state = torch.load(run / "resume.pt", map_location="cpu", weights_only=False)
        expected = dict(model_version=MODEL_VERSION, architecture=ARCHITECTURE,
                        config_sha256=fingerprint, source_sha256=sources, dataset_info=info,
                        arch_config=model.arch_config, backbone_provenance=model.backbone.provenance,
                        optimizer_manifest=optimizer.manifest, total_updates=total, global_batch_size=64,
                        initialization=initialization)
        for key, value in expected.items():
            if state.get(key) != value:
                raise ValueError(f"Resume mismatch: {key}; an explicit experiment revision is required")
        step = state["continuation_step"]
        if state["step"] != initialization["source_step"] + step:
            raise ValueError("Global checkpoint step and continuation cursor disagree")
        if not isinstance(step, int) or not 0 <= step <= total:
            raise ValueError("Invalid resume update cursor")
        if state.get("optimizer", {}).get("steps_completed") != step:
            raise ValueError("Optimizer and checkpoint steps disagree")
        if (state.get("epoch"), state.get("next_update")) != (0, step):
            raise ValueError("Resume sampler cursor disagrees with update")
        if state.get("complete") != (step == total):
            raise ValueError("Resume completion flag disagrees with cursor")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        multiplier = lr_multiplier(step, total, tr) if step else 1.
        if any(g["lr"] != g["base_lr"] * multiplier for g in optimizer.param_groups):
            raise ValueError("Restored learning rate differs from fixed epoch schedule")
        del state
    wrapped = DDP(model, device_ids=[local], find_unused_parameters=False) if world > 1 else model
    initial_step, began = step, time.monotonic()
    deadline = args.deadline if args.deadline is not None else float(os.environ.get("V12_TRAIN_DEADLINE", "inf"))
    margin = tr["save_margin_seconds"]
    topology = [None] * world
    own = dict(rank=rank, node=socket.gethostname(), gpu=torch.cuda.get_device_name(local), jobid=os.getenv("SLURM_JOB_ID"))
    if world > 1:
        dist.all_gather_object(topology, own)
    else:
        topology[0] = own
    torch.cuda.reset_peak_memory_stats()

    def progress(status, **extra):
        speed = (time.monotonic()-began) / (step-initial_step) if step > initial_step else None
        return dict(status=status, step=initialization["source_step"]+step,
                    continuation_step=step, total_updates=total, updated_at=time.time(),
                    budget_complete=step == total, epoch_complete=False, dataset_info=info,
                    physical_world_size=world, micro_batch_per_card=micro_size,
                    accumulated_batch_per_card=sampler.local_batch,
                    accumulation=sampler.local_batch // micro_size, global_batch_size=64,
                    seconds_per_update=speed, eta_train_hours=(total-step)*speed/3600 if speed else None,
                    optimizer_lrs=optimizer.learning_rates(), jobid=os.getenv("SLURM_JOB_ID"), **extra)

    def save(archive=False, status="paused", reason=None):
        if optimizer.steps_completed != step:
            raise RuntimeError("Trainer/optimizer step mismatch before saving")
        def write():
            common = dict(model=model.state_dict(), config=cfg, config_sha256=fingerprint,
                          source_sha256=sources, model_version=MODEL_VERSION, architecture=ARCHITECTURE,
                          arch_config=model.arch_config, backbone_provenance=model.backbone.provenance,
                          optimizer_manifest=optimizer.manifest, dataset_info=info,
                          step=initialization["source_step"]+step, continuation_step=step,
                          initialization=initialization, total_updates=total, global_batch_size=64)
            if archive:
                path = run / "checkpoints" / f"step_{initialization['source_step']+step:06d}.pt"
                if path.exists():
                    old = torch.load(path, map_location="cpu", weights_only=False)
                    if any(old.get(k) != common[k] for k in ("step", "config_sha256", "source_sha256", "dataset_info")):
                        raise FileExistsError(f"Incompatible archive: {path}")
                    if any(not torch.equal(v.detach().cpu(), old["model"][k]) for k,v in common["model"].items()):
                        raise FileExistsError(f"Divergent archive: {path}")
                    del old
                else:
                    atomic_save(common, path)
            epoch, cursor = 0, step
            atomic_save(dict(**common, optimizer=optimizer.state_dict(), epoch=epoch, next_update=cursor,
                             complete=step == total, physical_world_size=world), run / "resume.pt")
            record = progress("complete" if step == total else status, stop_reason=reason)
            atomic_json(record, run / "progress.json")
            if step == total:
                atomic_json(record, run / "TRAIN_COMPLETE.json")
            print("[checkpoint]", json.dumps(dict(step=step, status=record["status"], reason=reason)), flush=True)
        rank_zero_action(write, rank, world)

    def schedule():
        record = dict(config=cfg, config_sha256=fingerprint, source_sha256=sources,
                      architecture=ARCHITECTURE, model_version=MODEL_VERSION, dataset_info=info,
                      total_updates=total, checkpoint_every=tr["checkpoint_every"], rank_topology=topology,
                      global_batch_size=64, micro_batch_per_card=micro_size,
                      accumulation=sampler.local_batch // micro_size,
                      loss_reduction="mean_over_available_modes_of_global_mean_per_cell_gene_mean_joint_count_NLL",
                      dropout_resume="deterministic_update_rank_micro; topology_change_changes_draws",
                      arch_config=model.arch_config, backbone_provenance=model.backbone.provenance,
                      optimizer_manifest=optimizer.manifest, optimizer_resume_state_verified=optimizer.resume_state_verified,
                      initialization=initialization,
                      trainable_parameters=sum(p.numel() for p in parameters), initial_step=step, resumed=args.resume)
        atomic_json(record, run / "schedule.json")
        (run / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        atomic_json(sampler.contract, run / "sampler_contract.json")
        print("[schedule]", json.dumps(record, allow_nan=False), flush=True)
    rank_zero_action(schedule, rank, world)
    if step == total:
        save(status="complete", reason="already_complete")
        return
    if step == 0 and not args.resume:
        save(archive=True, status="initialized")
    if args.initialize_only:
        return
    initial_stop = torch.tensor(int(time.time() >= deadline-margin or _STOP), device="cuda")
    if world > 1:
        dist.all_reduce(initial_stop, op=dist.ReduceOp.MAX)
    if bool(initial_stop):
        save(reason="deadline_before_first_update")
        return
    dataset.set_epoch(0)
    sampler.seek(0, step)
    wrapped.train()
    for cpu_batch in loader:
        domain_ids = cpu_batch.pop("domain_id")
        domain_counts = torch.bincount(domain_ids, minlength=2).to("cuda", dtype=torch.int64)
        if world > 1:
            dist.all_reduce(domain_counts)
        if domain_counts.tolist() != [64-tr["visium_centers_per_update"], tr["visium_centers_per_update"]]:
            raise RuntimeError("Actual global domain exposure differs from the matched sampler contract")
        logging_update = step < 3 or (step+1) % tr["log_every"] == 0 or step+1 == total
        # Availability is cheap and input-only: get exact denominators for all micros.
        local_counts = loss_counts_from_batch(cpu_batch, cfg, mode_available=model.mode_availability(cpu_batch))
        modes = list(local_counts)
        counts = torch.stack([local_counts[m] for m in modes]).to("cuda", dtype=torch.float64)
        if world > 1:
            dist.all_reduce(counts)
        global_counts = dict(zip(modes, counts))
        if not bool(counts.gt(0).any()):
            raise ValueError("Global update has no available expression supervision")
        measured_counts = torch.zeros_like(counts)
        summed_losses = torch.zeros_like(counts)
        optimizer.zero_grad(set_to_none=True)
        micro_count = sampler.local_batch // micro_size
        diag_sums = {}
        domain_loss_sums = torch.zeros((2, len(modes)), device="cuda", dtype=torch.float64)
        domain_loss_counts = torch.zeros_like(domain_loss_sums)
        domain_query_counts = torch.zeros((2, 2), device="cuda", dtype=torch.int64)
        for micro in range(micro_count):
            sl = slice(micro * micro_size, (micro+1) * micro_size)
            batch = {k: v[sl].cuda(non_blocking=True) for k,v in cpu_batch.items()}
            seed_microbatch(seed, step, rank, micro)
            context = wrapped.no_sync() if world > 1 and micro < micro_count-1 else nullcontext()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = wrapped(batch)
                sums, observed = loss_sums_and_counts(output, batch, cfg)
                loss = reduce_loss_sums(sums, observed, global_counts=global_counts, world_size=world)
                # Every rank reaches this collective before backward, including absent-view ranks.
                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite):
                    raise FloatingPointError(f"Nonfinite loss at update {step+1}, micro {micro}")
                loss.backward()
            if logging_update:
                # Detached diagnostics share predictions and never add an
                # extra training loss or another model forward.
                with torch.no_grad():
                    micro_domains = domain_ids[sl].cuda(non_blocking=True)
                    for domain in (0, 1):
                        selected = micro_domains.eq(domain)
                        if not bool(selected.any()):
                            continue
                        subset_output = {
                            "count_parameters": {mode: {k: v[selected].detach() for k, v in item.items()}
                                                 for mode, item in output["count_parameters"].items()},
                            "mode_available": {mode: v[selected] for mode, v in output["mode_available"].items()},
                        }
                        subset_batch = {k: v[selected] for k, v in batch.items() if k.startswith("query_")}
                        valid_queries = subset_batch["query_valid_mask"].bool()
                        domain_query_counts[domain, 0] += valid_queries.sum()
                        domain_query_counts[domain, 1] += (
                            valid_queries & subset_batch["query_total_counts"].gt(0)).sum()
                        dsums, dcounts = loss_sums_and_counts(subset_output, subset_batch, cfg)
                        domain_loss_sums[domain] += torch.stack([dsums[m].double() for m in modes])
                        domain_loss_counts[domain] += torch.stack([dcounts[m].double() for m in modes])
            summed_losses += torch.stack([sums[m].detach().double() for m in modes])
            measured_counts += torch.stack([observed[m].double() for m in modes])
            for name, value in output["diagnostics"].items():
                diag_sums[name] = diag_sums.get(name, 0.) + value.detach().float().mean() / micro_count
            del output, loss, sums, batch
        if world > 1:
            dist.all_reduce(summed_losses)
            dist.all_reduce(measured_counts)
        if not torch.equal(counts, measured_counts):
            raise RuntimeError("Pre-forward counts disagree with actual loss eligibility")
        norm = torch.nn.utils.clip_grad_norm_(parameters, tr["grad_clip_norm"], error_if_nonfinite=True)
        optimizer.set_lr_multiplier(lr_multiplier(step+1, total, tr))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        torch.cuda.synchronize()
        if logging_update:
            if world > 1:
                dist.all_reduce(domain_loss_sums)
                dist.all_reduce(domain_loss_counts)
                dist.all_reduce(domain_query_counts)
            per_domain = {}
            for domain, name in enumerate(("xenium", "visium_hd")):
                per_domain[name] = dict(centers=int(domain_counts[domain]),
                    valid_queries=int(domain_query_counts[domain,0]),
                    positive_queries=int(domain_query_counts[domain,1]),
                    positive_query_fraction=float(domain_query_counts[domain,1]/domain_query_counts[domain,0])
                        if bool(domain_query_counts[domain,0] > 0) else None,
                    nll={m: float(domain_loss_sums[domain,i]/domain_loss_counts[domain,i])
                         if bool(domain_loss_counts[domain,i] > 0) else None for i,m in enumerate(modes)},
                    available_cells={m: int(domain_loss_counts[domain,i]) for i,m in enumerate(modes)})
            memory = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()], device="cuda", dtype=torch.float64)
            if world > 1:
                dist.all_reduce(memory, op=dist.ReduceOp.MAX)
            parts = {f"{mode}_nll": float(summed_losses[i] / counts[i].clamp_min(1)) for i,mode in enumerate(modes)}
            parts.update({f"{mode}_cells": int(counts[i]) for i,mode in enumerate(modes)})
            active = counts.gt(0)
            parts["total"] = float((summed_losses[active]/counts[active]).mean())
            diag = {}
            for name, value in diag_sums.items():
                if world > 1:
                    dist.all_reduce(value)
                diag[name] = float(value / world)
            record = progress("running", loss=parts, gradient_norm=float(norm), diagnostics=diag,
                              domains=per_domain,
                              cumulative_center_exposure=dict(xenium=step*(64-tr["visium_centers_per_update"]),
                                                              visium_hd=step*tr["visium_centers_per_update"]),
                              allocated_gib=float(memory[0])/2**30, reserved_gib=float(memory[1])/2**30,
                              memory_scope="peak_max_across_ranks_since_launch")
            def write_metrics():
                with (run / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record, allow_nan=False)+"\n")
                atomic_json(record, run / "progress.json")
                print("[train]", json.dumps(record, allow_nan=False), flush=True)
            rank_zero_action(write_metrics, rank, world)
        reason = ("signal" if _STOP else "allocation_deadline" if time.time() >= deadline-margin else
                  "pause_marker" if any((run/m).exists() for m in PAUSE_MARKERS) else
                  "run_update_limit" if args.run_update_limit and step-initial_step >= args.run_update_limit else None)
        stop = torch.tensor(int(reason is not None), device="cuda")
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
        done = step == total
        archive = step % tr["checkpoint_every"] == 0 or done or bool(stop)
        if archive or step % tr["resume_every"] == 0:
            save(archive=archive, status="paused" if bool(stop) else "running", reason=reason)
        if done or bool(stop):
            return
    raise RuntimeError("Loader ended before the planned global updates")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--run-update-limit", type=int, default=0)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if (not os.getenv("SLURM_JOB_ID") or not os.getenv("SLURMD_NODENAME") or
            socket.gethostname().split(".")[0] != os.getenv("SLURMD_NODENAME", "").split(".")[0]):
        raise RuntimeError("This trainer must run inside an authorized Slurm compute-node step")
    if args.run_update_limit < 0:
        parser.error("run-update-limit must be nonnegative")
    cfg = yaml.safe_load(args.config.resolve().read_text())
    os.chdir(ROOT)
    rank, world, local = (int(os.getenv(k, "1" if k == "WORLD_SIZE" else "0")) for k in ("RANK","WORLD_SIZE","LOCAL_RANK"))
    validate_config(cfg, world)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA on an authorized compute allocation is required")
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    run, lock = Path(cfg["output_dir"]), None
    try:
        def prepare():
            nonlocal lock
            if args.resume:
                if not (run/"resume.pt").is_file():
                    raise FileNotFoundError("Explicit resume requires resume.pt")
            elif run.exists() and any(run.iterdir()):
                raise FileExistsError("Refusing nonempty output directory; use explicit resume")
            if any((run/m).exists() for m in PAUSE_MARKERS):
                raise RuntimeError("Unresolved pause marker")
            run.mkdir(parents=True, exist_ok=True)
            lock = (run/".trainer.lock").open("a")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            (run/"checkpoints").mkdir(exist_ok=True)
        rank_zero_action(prepare, rank, world)
        train(cfg, args, run, rank, world)
    finally:
        if lock is not None:
            lock.close()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
