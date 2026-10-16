from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from time import time, sleep

import torch
import torch.distributed as dist
from torch import amp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from spatial_pretrain.config import Config
from spatial_pretrain.data.cache import build_slice_cache, load_slice_cache
from spatial_pretrain.data.compartment_cache import build_compartment_cache
from spatial_pretrain.data.dataset import MultiSliceDataset, SpatialSubgraphDataset, collate_subgraphs, prepare_slice_views_parallel
from spatial_pretrain.gene_init import (
    build_global_gene_maps as _build_global_gene_maps,
    load_gene_init_artifact,
    resolve_slice_specs as _resolve_slice_specs,
    build_global_gene_vocabulary,
)
from spatial_pretrain.models.ssgt import SSGTModel
from spatial_pretrain.models.compartment import CompartmentSSGTModel
from spatial_pretrain.training.losses import compute_pretraining_losses
from spatial_pretrain.training.muon import MuonWithAuxAdam
from spatial_pretrain.training.checkpoint import load_resume, restore_rng, save_resume

try:
    import wandb
except ImportError:
    wandb = None


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        init_kwargs = {
            "backend": backend,
            "init_method": "env://",
        }
        if backend == "nccl":
            init_kwargs["device_id"] = torch.device("cuda", local_rank)
        dist.init_process_group(**init_kwargs)
    return distributed, rank, local_rank, world_size


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, (DDP, DataParallel)) else model


def _distributed_barrier(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def _prepare_views_for_rank(
    caches: list,
    slice_specs: list[tuple[str, str, str]],
    config: Config,
    *,
    distributed: bool,
    rank: int,
) -> list:
    slice_names = [slice_name for _, _, slice_name in slice_specs]
    if not distributed:
        return prepare_slice_views_parallel(caches, slice_names, config)

    if _is_main_process(rank):
        prepared = prepare_slice_views_parallel(caches, slice_names, config)
        _distributed_barrier(True)
        return prepared

    _distributed_barrier(True)
    load_only_config = deepcopy(config)
    load_only_config.data.view_prep_workers = 0
    return prepare_slice_views_parallel(caches, slice_names, load_only_config)


def _maybe_load_gene_init(config: Config, num_genes: int, gene_names=None) -> torch.Tensor | None:
    if not config.model.use_genept or not config.model.genept_path:
        return None
    return load_gene_init_artifact(
        config.model.genept_path,
        num_genes=num_genes,
        hidden_dim=config.model.hidden_dim,
        gene_names=gene_names,
    )


def _uses_padding_tokens(config: Config) -> bool:
    return str(config.data.gene_sampling_strategy).strip().lower() in {"random_nonzero", "compartment_balanced"}


def _append_padding_gene_init(gene_init: torch.Tensor | None) -> torch.Tensor | None:
    if gene_init is None:
        return None
    pad_row = gene_init.new_zeros((1, gene_init.shape[1]))
    return torch.cat([gene_init, pad_row], dim=0)


def _evaluate(model: torch.nn.Module, loader: DataLoader, config: Config, device: torch.device) -> dict[str, float]:
    was_training = model.training
    try:
        return _evaluate_impl(model, loader, config, device)
    finally:
        model.train(was_training)


def _model_inputs(batch):
    result = {key: batch[key] for key in ("gene_ids", "input_bins", "cell_roles")}
    result["token_valid_mask"] = batch.get("token_valid_mask")
    if "compartment_ids" in batch:
        result["compartment_ids"] = batch["compartment_ids"]
        result["node_indices"] = batch["node_indices"]
        result["coords"] = batch.get("coords")
    else:
        result["lr_bias"] = batch["lr_bias"]
    return result


def _evaluate_impl(model: torch.nn.Module, loader: DataLoader, config: Config, device: torch.device) -> dict[str, float]:
    model.eval()
    amp_dtype = _resolve_amp_dtype(config, device)
    amp_enabled = amp_dtype is not None
    totals = {"loss": 0.0}
    num_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = _move_batch(batch, device)
            with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    **_model_inputs(batch),
                    return_cell_attention_weights=False,
                )
                losses = compute_pretraining_losses(outputs, batch, config)
            totals["loss"] += float(losses["loss"].item())
            num_batches += 1
            if config.training.max_val_batches and batch_idx >= config.training.max_val_batches:
                break
    if dist.is_initialized():
        reduced = torch.tensor(
            [totals["loss"], float(num_batches)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        totals = {"loss": float(reduced[0].item())}
        num_batches = int(reduced[1].item())
    if num_batches == 0:
        return {key: 0.0 for key in totals}
    return {key: value / num_batches for key, value in totals.items()}


def _weighted_mean_across_processes(total: float, weight: float, device: torch.device) -> float:
    if dist.is_initialized():
        reduced = torch.tensor([float(total), float(weight)], device=device, dtype=torch.float64)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total = float(reduced[0].item())
        weight = float(reduced[1].item())
    return total / max(float(weight), 1.0)


def _count_optimizer_updates(num_batches: int, grad_accum_steps: int, max_train_steps: int) -> int:
    micro_batches = min(num_batches, max_train_steps) if max_train_steps else num_batches
    return max(math.ceil(micro_batches / grad_accum_steps), 1)


def _resolve_warmup_updates(config: Config, total_updates: int) -> int:
    warmup_ratio = float(getattr(config.training, "warmup_ratio", -1.0))
    if warmup_ratio >= 0:
        if total_updates <= 1:
            return 0
        return min(max(int(round(total_updates * warmup_ratio)), 0), total_updates - 1)
    return max(int(config.training.warmup_updates), 0)


def _compute_learning_rate(config: Config, update_idx: int, total_updates: int) -> float:
    base_lr = float(config.optimizer.lr)
    schedule = config.training.lr_schedule.strip().lower()
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr_schedule={config.training.lr_schedule!r}")

    warmup_updates = _resolve_warmup_updates(config, total_updates)
    min_lr_ratio = float(config.training.min_lr_ratio)
    min_lr = base_lr * min_lr_ratio

    if warmup_updates > 0 and update_idx <= warmup_updates:
        return base_lr * update_idx / warmup_updates

    if total_updates <= warmup_updates + 1:
        return base_lr

    progress = (update_idx - warmup_updates - 1) / (total_updates - warmup_updates - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr * float(group.get("lr_scale", 1.0))


def _should_use_muon(name: str, param: torch.nn.Parameter) -> bool:
    if not name.startswith("blocks.") or param.ndim < 2:
        return False
    # Learned cell-attention projections are more sensitive than the legacy
    # flattened path, so keep them on AdamW even when the surrounding block uses
    # Muon for its larger hidden 2D weights.
    excluded_suffixes = (
        ".cell_attn.q_proj.weight",
        ".cell_attn.k_proj.weight",
        ".cell_attn.v_proj.weight",
        ".cell_attn.out_proj.weight",
    )
    return not name.endswith(excluded_suffixes)


def _build_optimizer(model: torch.nn.Module, config: Config) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    optimizer_name = config.optimizer.name.strip().lower()
    if optimizer_name == "adamw":
        return (
            torch.optim.AdamW(
                model.parameters(),
                lr=config.optimizer.lr,
                betas=tuple(config.optimizer.betas),
                weight_decay=config.optimizer.weight_decay,
                eps=config.optimizer.eps,
            ),
            {"muon_params": 0, "aux_params": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        )

    if optimizer_name != "muon":
        raise ValueError(f"Unsupported optimizer.name={config.optimizer.name!r}")

    muon_params: list[torch.nn.Parameter] = []
    aux_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _should_use_muon(name, param):
            muon_params.append(param)
        else:
            aux_params.append(param)

    lr_scale = config.optimizer.muon_lr / config.optimizer.lr
    param_groups: list[dict] = []
    if muon_params:
        param_groups.append(
            {
                "params": muon_params,
                "use_muon": True,
                "lr": config.optimizer.muon_lr,
                "weight_decay": config.optimizer.weight_decay,
                "momentum": config.optimizer.muon_momentum,
                "ns_steps": config.optimizer.muon_ns_steps,
                "nesterov": config.optimizer.muon_nesterov,
                "lr_scale": lr_scale,
            }
        )
    if aux_params:
        param_groups.append(
            {
                "params": aux_params,
                "use_muon": False,
                "lr": config.optimizer.lr,
                "betas": tuple(config.optimizer.betas),
                "eps": config.optimizer.eps,
                "weight_decay": config.optimizer.weight_decay,
                "lr_scale": 1.0,
            }
        )

    return (
        MuonWithAuxAdam(
            param_groups,
            muon_adjust_lr_fn=getattr(config.optimizer, "muon_adjust_lr_fn", "match_rms_adamw"),
        ),
        {
            "muon_params": sum(p.numel() for p in muon_params),
            "aux_params": sum(p.numel() for p in aux_params),
        },
    )


def _resolve_amp_dtype(config: Config, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or not config.training.amp:
        return None
    dtype_name = config.training.amp_dtype.strip().lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported amp_dtype={config.training.amp_dtype!r}")


def _resolve_data_parallel_device_ids(config: Config) -> list[int]:
    if config.training.device_ids:
        return [int(device_id) for device_id in config.training.device_ids]
    return list(range(torch.cuda.device_count()))


def _split_batch_for_devices(batch: dict[str, torch.Tensor], num_chunks: int) -> list[dict[str, torch.Tensor]]:
    if num_chunks <= 1:
        return [batch]
    first_key = next(iter(batch))
    batch_size = int(batch[first_key].shape[0])
    if batch_size < num_chunks:
        raise ValueError(
            f"Batch size {batch_size} is smaller than the requested number of devices {num_chunks}"
        )
    chunked = {
        key: list(torch.tensor_split(value, num_chunks, dim=0))
        for key, value in batch.items()
    }
    splits = []
    for chunk_idx in range(num_chunks):
        if chunked[first_key][chunk_idx].shape[0] == 0:
            continue
        splits.append({key: parts[chunk_idx] for key, parts in chunked.items()})
    return splits


def _sync_replicas_from_master(master_model: torch.nn.Module, replica_models: list[torch.nn.Module]) -> None:
    if not replica_models:
        return
    master_params = list(master_model.parameters())
    master_buffers = list(master_model.buffers())
    with torch.no_grad():
        for replica in replica_models:
            for master_param, replica_param in zip(master_params, replica.parameters()):
                replica_param.copy_(master_param.to(device=replica_param.device, non_blocking=True))
            for master_buffer, replica_buffer in zip(master_buffers, replica.buffers()):
                replica_buffer.copy_(master_buffer.to(device=replica_buffer.device, non_blocking=True))


def _accumulate_manual_parallel_grads(
    master_model: torch.nn.Module,
    replica_models: list[torch.nn.Module],
) -> None:
    if not replica_models:
        return
    master_device = next(master_model.parameters()).device
    for replica in replica_models:
        for master_param, replica_param in zip(master_model.parameters(), replica.parameters()):
            if replica_param.grad is None:
                continue
            replica_grad = replica_param.grad.to(device=master_device, non_blocking=True)
            if master_param.grad is None:
                master_param.grad = replica_grad.clone()
            else:
                master_param.grad.add_(replica_grad)


def _init_wandb(config: Config, output_dir: Path):
    if not config.wandb.enabled:
        return None
    if wandb is None:
        raise ModuleNotFoundError("wandb is enabled in config but the package is not installed")

    wandb_dir = Path(config.wandb.dir or output_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity or None,
        name=config.wandb.name or config.experiment.name,
        group=config.wandb.group or None,
        job_type=config.wandb.job_type or None,
        dir=str(wandb_dir),
        mode=config.wandb.mode,
        tags=config.wandb.tags or None,
        config=asdict(config),
    )
    if run is not None:
        run.define_metric("update")
        run.define_metric("loss/*", step_metric="update")
        run.define_metric("optimizer/*", step_metric="update")
        run.define_metric("progress/*", step_metric="update")
        run.define_metric("system/*", step_metric="update")
    return run


def _wandb_log(run, metrics: dict[str, float | int], *, step: int) -> None:
    if run is None:
        return
    payload = dict(metrics)
    payload["update"] = int(step)
    run.log(payload, step=step)


def _make_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    sampler=None,
    seed=None,
):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "num_workers": num_workers,
        "collate_fn": collate_subgraphs,
        "pin_memory": pin_memory,
        "sampler": sampler,
        "generator": torch.Generator().manual_seed(seed) if seed is not None else None,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["multiprocessing_context"] = "spawn"
    return DataLoader(dataset, **loader_kwargs)


def _preflight_full_views(config, output_dir):
    """Build large neighborhood tables before starting NCCL's watchdog timer."""
    if (int(os.environ.get("WORLD_SIZE", "1")) < 2
            or config.data.gene_sampling_strategy != "compartment_balanced"
            or len(config.data.h5ad_paths) < 2):
        return
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", os.environ.get("MASTER_PORT", "manual"))
    key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    status_path = output_dir / f"preflight_{key}.json"
    if int(os.environ.get("RANK", "0")) == 0:
        print("[preflight] preparing all sample neighborhoods before distributed initialization", flush=True)
        try:
            specs = _resolve_slice_specs(config)
            caches = [load_slice_cache(cache_dir) for _, cache_dir, _ in specs]
            prepare_slice_views_parallel(caches, [name for _, _, name in specs], config)
            status = {"ready": True, "samples": len(specs)}
        except BaseException as exc:
            status_path.write_text(json.dumps({"ready": False, "error": str(exc)}))
            raise
        temporary = status_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(status))
        temporary.replace(status_path)
    else:
        while not status_path.exists():
            sleep(2)
        status = json.loads(status_path.read_text())
        if not status.get("ready"):
            raise RuntimeError(f"Neighborhood preflight failed on rank 0: {status}")


def run_pretraining(config: Config) -> None:
    torch.manual_seed(config.experiment.seed)
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _preflight_full_views(config, output_dir)
    distributed, rank, local_rank, world_size = _init_distributed()
    parallelism = config.training.parallelism.strip().lower()
    data_parallel_device_ids: list[int] = []
    if torch.cuda.is_available():
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(config.training.device)
            if parallelism in {"data_parallel", "manual_data_parallel"}:
                if device.type != "cuda":
                    raise ValueError(f"training.parallelism={parallelism} requires a CUDA device")
                data_parallel_device_ids = _resolve_data_parallel_device_ids(config)
                if len(data_parallel_device_ids) < 2:
                    raise ValueError(
                        f"training.parallelism={parallelism} requires at least 2 visible CUDA devices"
                    )
                torch.cuda.set_device(data_parallel_device_ids[0])
                device = torch.device("cuda", data_parallel_device_ids[0])
    else:
        device = torch.device("cpu")
    grad_accum_steps = max(int(config.training.grad_accum_steps), 1)
    amp_dtype = _resolve_amp_dtype(config, device)
    amp_enabled = amp_dtype is not None
    pin_memory = device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    slice_specs = _resolve_slice_specs(config)
    compartment_mode = config.data.gene_sampling_strategy == "compartment_balanced"
    if compartment_mode != (config.model.cell_attention_mode in {"cytoplasm_cross", "cytoplasm_stack_flat", "cytoplasm_mean"}):
        raise ValueError("compartment_balanced data requires a compartment attention model")
    if compartment_mode and config.data.use_lr_prior:
        raise ValueError("Compartment training requires use_lr_prior: false")
    caches = []
    for h5ad_path, cache_dir, _slice_name in slice_specs:
        if _is_main_process(rank):
            if compartment_mode:
                build_compartment_cache(h5ad_path, cache_dir, config)
            else:
                build_slice_cache(
                    h5ad_path=h5ad_path, cache_dir=cache_dir,
                    lr_db_path=config.data.lr_db_path if config.data.use_lr_prior else "",
                    knn_k=config.data.knn_k, chunk_size=config.data.chunk_size,
                    num_bins=config.data.num_bins, max_cells=config.data.max_cells,
                )
        _distributed_barrier(distributed)
        caches.append(load_slice_cache(cache_dir))

    global_num_genes, global_gene_maps = _build_global_gene_maps(caches)
    use_padding_tokens = _uses_padding_tokens(config)
    pad_gene_id = global_num_genes if use_padding_tokens else None
    model_num_genes = global_num_genes + (1 if use_padding_tokens else 0)

    prepared_views = _prepare_views_for_rank(
        caches,
        slice_specs,
        config,
        distributed=distributed,
        rank=rank,
    )

    train_datasets: list[SpatialSubgraphDataset] = []
    val_datasets: list[SpatialSubgraphDataset] = []
    for (_h5ad_path, _cache_dir, slice_name), cache, gene_map, prepared in zip(
        slice_specs,
        caches,
        global_gene_maps,
        prepared_views,
    ):
        train_datasets.append(
            SpatialSubgraphDataset(
                cache_dir=cache.cache_dir,
                subgraph_nodes_path=prepared.train_nodes_path,
                center_indices_path=prepared.train_centers_path,
                selected_rows=prepared.train_epoch0_rows,
                config=config,
                training=True,
                global_gene_ids=gene_map.numpy(),
                pad_gene_id=pad_gene_id,
                slice_name=slice_name,
                slice_cache=cache,
            )
        )
        val_datasets.append(
            SpatialSubgraphDataset(
                cache_dir=cache.cache_dir,
                subgraph_nodes_path=prepared.val_nodes_path,
                center_indices_path=prepared.val_centers_path,
                selected_rows=prepared.val_rows,
                config=config,
                training=False,
                global_gene_ids=gene_map.numpy(),
                pad_gene_id=pad_gene_id,
                slice_name=slice_name,
                slice_cache=cache,
            )
        )

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else MultiSliceDataset(train_datasets)
    val_dataset = val_datasets[0] if len(val_datasets) == 1 else MultiSliceDataset(val_datasets)
    if len(val_datasets) > 1 and config.training.max_val_batches:
        # A capped validation prefix must sample across the cohort, not only
        # the first slices in the manifest. Keep the permutation fixed on resume.
        generator = torch.Generator().manual_seed(config.data.split_seed)
        val_dataset = Subset(val_dataset, torch.randperm(len(val_dataset), generator=generator).tolist())
    train_num_workers = max(int(config.data.num_workers), 0)
    val_num_workers = max(train_num_workers // 2, 0)
    val_sampler = (
        DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        if distributed
        else None
    )
    val_loader = _make_dataloader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
    )

    train_dataset.set_epoch(0)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    if train_sampler is not None:
        train_sampler.set_epoch(0)
    train_loader = _make_dataloader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
        seed=config.experiment.seed,
    )

    gene_names = build_global_gene_vocabulary(cache.gene_names for cache in caches) if compartment_mode else None
    gene_init = _maybe_load_gene_init(config, global_num_genes, gene_names)
    if use_padding_tokens:
        gene_init = _append_padding_gene_init(gene_init)
    model_kwargs = dict(
        num_genes=model_num_genes,
        num_value_bins=config.data.num_bins,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        gene_heads=config.model.gene_heads,
        cell_heads=config.model.cell_heads,
        dropout=config.model.dropout,
        ffn_mult=config.model.ffn_mult,
        use_center_head=False,
        gene_embedding_init=gene_init,
        cell_attention_mode=getattr(config.model, "cell_attention_mode", "identity_flat"),
        cell_token_count=config.data.top_genes,
        cell_mla_latent_dim=getattr(config.model, "cell_mla_latent_dim", 128),
        cell_mla_qk_head_dim=getattr(config.model, "cell_mla_qk_head_dim", 64),
        cell_mla_v_head_dim=getattr(config.model, "cell_mla_v_head_dim", 64),
        cell_mla_output_rank=getattr(config.model, "cell_mla_output_rank", 128),
        use_padding_tokens=use_padding_tokens,
        gene_padding_idx=pad_gene_id,
        gradient_checkpointing=config.training.gradient_checkpointing,
    )
    if compartment_mode:
        for key in ("use_center_head", "cell_token_count",
                    "cell_mla_latent_dim", "cell_mla_qk_head_dim", "cell_mla_v_head_dim",
                    "cell_mla_output_rank", "use_padding_tokens"):
            model_kwargs.pop(key)
        model_kwargs.update(nuclear_tokens=config.data.nuclear_tokens,
                            cytoplasmic_tokens=config.data.cytoplasmic_tokens,
                            cell_token_dim=config.model.cell_token_dim,
                            require_flash_attention=config.model.require_flash_attention)
        from spatial_pretrain.models.spatial_bias import spatial_kwargs
        model_kwargs.update(spatial_kwargs(config.model))
        model = CompartmentSSGTModel(**model_kwargs).to(device)
    else:
        model = SSGTModel(**model_kwargs).to(device)
    manual_parallel_replicas: list[torch.nn.Module] = []
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    elif parallelism == "data_parallel":
        model = DataParallel(
            model,
            device_ids=data_parallel_device_ids,
            output_device=data_parallel_device_ids[0],
        )
    elif parallelism == "manual_data_parallel":
        manual_parallel_replicas = [
            deepcopy(model).to(torch.device("cuda", device_id))
            for device_id in data_parallel_device_ids[1:]
        ]
        _sync_replicas_from_master(model, manual_parallel_replicas)

    optimizer, optimizer_stats = _build_optimizer(_unwrap_model(model), config)
    scaler = amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    updates_per_epoch = _count_optimizer_updates(len(train_loader), grad_accum_steps, config.training.max_train_steps)
    total_updates = updates_per_epoch * config.training.epochs
    current_lr = _compute_learning_rate(config, update_idx=1, total_updates=total_updates)
    _set_optimizer_lr(optimizer, current_lr)
    wandb_run = _init_wandb(config, output_dir) if _is_main_process(rank) else None

    if _is_main_process(rank):
        schedule_info = {
            "epochs": config.training.epochs,
            "updates_per_epoch": updates_per_epoch,
            "total_updates": total_updates,
            "warmup_updates": _resolve_warmup_updates(config, total_updates),
            "lr_schedule": config.training.lr_schedule,
            "base_lr": config.optimizer.lr,
            "first_lr": current_lr,
            "last_lr": _compute_learning_rate(config, total_updates, total_updates),
            "world_size": world_size,
            "batch_size_per_process": config.data.batch_size,
            "grad_accum_steps": grad_accum_steps,
        }
        if distributed or parallelism == "single":
            schedule_info["global_batch_size"] = config.data.batch_size * world_size * grad_accum_steps
        (output_dir / "training_schedule.json").write_text(json.dumps(schedule_info, indent=2))
        print(f"[schedule] {json.dumps(schedule_info)}", flush=True)
        if distributed:
            print(
                f"[parallel] mode=ddp world_size={world_size} local_rank={local_rank} device={device}",
                flush=True,
            )
        elif parallelism in {"data_parallel", "manual_data_parallel"}:
            print(
                f"[parallel] mode={parallelism} device_ids={data_parallel_device_ids}",
                flush=True,
            )
        else:
            print(f"[parallel] mode=single device={device}", flush=True)
        print(
            f"[optimizer] name={config.optimizer.name} "
            f"muon_params={optimizer_stats['muon_params']} aux_params={optimizer_stats['aux_params']}",
            flush=True,
        )

    best_val = float("inf")
    global_step = 0
    start_epoch, resume_micro_step, resume_rng = 0, 0, None
    resume_path = output_dir / "resume.pt"
    if config.training.auto_resume and resume_path.exists():
        resumed = load_resume(resume_path, model, optimizer, scaler, config, world_size, device)
        start_epoch, resume_micro_step = resumed["epoch"], resumed["micro_step"]
        global_step, best_val = resumed["step"], resumed["best_val"]
        resume_rng = resumed["rng_by_rank"][rank]
        print(f"[resume] epoch={start_epoch} micro_step={resume_micro_step} update={global_step}", flush=True)
        if resumed["complete"]:
            print("[done] checkpoint already completed all configured epochs", flush=True)
            if wandb_run is not None:
                wandb_run.finish()
            if distributed:
                dist.destroy_process_group()
            return
    if getattr(config.training, "save_initial_checkpoint", False) and global_step == 0 and _is_main_process(rank):
        initial_path = output_dir / "initial.pt"
        if not initial_path.exists():
            torch.save({"model": _unwrap_model(model).state_dict(), "config": config, "step": 0}, initial_path)
            print(f"[initial] {initial_path}", flush=True)
    run_initial_step = global_step
    deadline = float(os.environ.get("SPATIAL_TRAIN_DEADLINE", "inf"))

    for epoch in range(start_epoch, config.training.epochs):
        if epoch > 0:
            train_dataset.set_epoch(epoch)
            train_sampler = (
                DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
                if distributed
                else None
            )
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loader = _make_dataloader(
                train_dataset,
                batch_size=config.data.batch_size,
                shuffle=True,
                num_workers=train_num_workers,
                pin_memory=pin_memory,
                sampler=train_sampler,
                seed=config.experiment.seed + epoch,
            )
        model.train()
        start_time = time()
        optimizer.zero_grad(set_to_none=True)
        accum_batches = 0
        accum_loss_total = 0.0
        accum_weight = 0.0

        for step, batch in enumerate(train_loader, start=1):
            if epoch == start_epoch and step <= resume_micro_step:
                continue
            if resume_rng is not None:
                restore_rng(resume_rng, device)
                resume_rng = None
            if parallelism == "manual_data_parallel":
                batch_chunks = _split_batch_for_devices(batch, len(data_parallel_device_ids))
                chunk_models = [model, *manual_parallel_replicas][:len(batch_chunks)]
                chunk_devices = [torch.device("cuda", device_id) for device_id in data_parallel_device_ids[:len(batch_chunks)]]
                total_chunk_items = sum(int(chunk["gene_ids"].shape[0]) for chunk in batch_chunks)
                step_loss = 0.0
                for chunk_model, chunk, chunk_device in zip(chunk_models, batch_chunks, chunk_devices):
                    chunk = _move_batch(chunk, chunk_device)
                    chunk_weight = int(chunk["gene_ids"].shape[0]) / total_chunk_items
                    with amp.autocast(device_type=chunk_device.type, dtype=amp_dtype, enabled=amp_enabled):
                        outputs = chunk_model(
                            **_model_inputs(chunk),
                            return_cell_attention_weights=False,
                        )
                        losses = compute_pretraining_losses(outputs, chunk, config)

                    if scaler.is_enabled():
                        raise RuntimeError("manual_data_parallel currently expects bf16/no GradScaler")
                    (losses["loss"] * chunk_weight / grad_accum_steps).backward()
                    step_loss += float(losses["loss"].item()) * chunk_weight
                _accumulate_manual_parallel_grads(model, manual_parallel_replicas)
                accum_batches += 1
                accum_loss_total += step_loss * total_chunk_items
                accum_weight += float(total_chunk_items)
            else:
                batch_items = int(batch["gene_ids"].shape[0])
                batch = _move_batch(batch, device)

                with amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    outputs = model(
                        **_model_inputs(batch),
                        return_cell_attention_weights=False,
                    )
                    losses = compute_pretraining_losses(outputs, batch, config)

                # Accumulate gradients across micro-batches to emulate a larger effective batch.
                scaler.scale(losses["loss"] / grad_accum_steps).backward()
                accum_batches += 1
                accum_loss_total += float(losses["loss"].item()) * batch_items
                accum_weight += float(batch_items)

            reached_train_limit = bool(config.training.max_train_steps and step >= config.training.max_train_steps)
            should_step = accum_batches >= grad_accum_steps or reached_train_limit or step == len(train_loader)
            if not should_step:
                continue

            next_update = global_step + 1
            current_lr = _compute_learning_rate(config, update_idx=next_update, total_updates=total_updates)
            _set_optimizer_lr(optimizer, current_lr)
            scaler.unscale_(optimizer)
            if accum_batches < grad_accum_steps:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(grad_accum_steps / accum_batches)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            for replica in manual_parallel_replicas:
                replica.zero_grad(set_to_none=True)
            if manual_parallel_replicas:
                _sync_replicas_from_master(model, manual_parallel_replicas)
            global_step += 1

            mean_loss = _weighted_mean_across_processes(accum_loss_total, accum_weight, device)

            if (
                step <= grad_accum_steps
                or global_step == run_initial_step + 1
                or step % config.training.log_every == 0
                or reached_train_limit
                or step == len(train_loader)
            ) and _is_main_process(rank):
                print(
                    f"[train] epoch={epoch+1} step={step}/{len(train_loader)} "
                    f"update={global_step} accum={accum_batches} "
                    f"loss={mean_loss:.4f} opt_lr={current_lr:.6e}",
                    flush=True,
                )

            _wandb_log(
                wandb_run,
                {
                    "loss/train": mean_loss,
                    "optimizer/lr": current_lr,
                    "progress/epoch": epoch + 1,
                    "progress/micro_step": step,
                    "progress/accum_batches": accum_batches,
                },
                step=global_step,
            )

            if config.training.val_every and global_step % config.training.val_every == 0:
                val_metrics = _evaluate(model, val_loader, config, device)
                if _is_main_process(rank):
                    print(
                        f"[val] step={global_step} loss={val_metrics['loss']:.4f}",
                        flush=True,
                    )
                    _wandb_log(
                        wandb_run,
                        {"loss/val": val_metrics["loss"]},
                        step=global_step,
                    )
                    if val_metrics["loss"] < best_val:
                        best_val = val_metrics["loss"]
                        if wandb_run is not None:
                            wandb_run.summary["best_val_loss"] = best_val
                        torch.save(
                            {
                                "model": _unwrap_model(model).state_dict(),
                                "config": config,
                                "val_loss": best_val,
                                "step": global_step,
                            },
                            output_dir / "best.pt",
                        )
                    torch.save(
                        {
                            "model": _unwrap_model(model).state_dict(),
                            "config": config,
                            "val_loss": val_metrics["loss"],
                            "step": global_step,
                        },
                        output_dir / "last.pt",
                    )

            accum_batches = 0
            accum_loss_total = 0.0
            accum_weight = 0.0

            pause = time() >= deadline or bool(config.training.run_update_limit and
                    global_step - run_initial_step >= config.training.run_update_limit)
            if distributed:
                pause_tensor = torch.tensor(int(pause), device=device)
                dist.all_reduce(pause_tensor, op=dist.ReduceOp.MAX)
                pause = bool(pause_tensor.item())
            epoch_done = reached_train_limit or step == len(train_loader)
            if config.training.checkpoint_every and (
                global_step % config.training.checkpoint_every == 0 or pause or epoch_done
            ):
                save_resume(resume_path, model, optimizer, scaler, config,
                            epoch=epoch + 1 if epoch_done else epoch,
                            micro_step=0 if epoch_done else step, global_step=global_step,
                            best_val=best_val, world_size=world_size, rank=rank, device=device)
            if pause:
                if not config.training.checkpoint_every:
                    raise RuntimeError("A training pause requires checkpoint_every > 0")
                print(f"[paused] update={global_step}; resume checkpoint saved", flush=True)
                if wandb_run is not None:
                    wandb_run.finish()
                if distributed:
                    dist.destroy_process_group()
                return

            if reached_train_limit:
                break

        epoch_seconds = time() - start_time
        if device.type == "cuda":
            peak_alloc_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            peak_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024**3)
            if _is_main_process(rank):
                print(
                    f"[epoch] {epoch+1} finished in {epoch_seconds:.1f}s "
                    f"peak_alloc_gb={peak_alloc_gb:.2f} peak_reserved_gb={peak_reserved_gb:.2f}",
                    flush=True,
                )
                _wandb_log(
                    wandb_run,
                    {
                        "system/peak_alloc_gb": peak_alloc_gb,
                        "system/peak_reserved_gb": peak_reserved_gb,
                        "system/epoch_seconds": epoch_seconds,
                        "progress/epoch": epoch + 1,
                    },
                    step=max(global_step, 1),
                )
        elif _is_main_process(rank):
            print(f"[epoch] {epoch+1} finished in {epoch_seconds:.1f}s", flush=True)

    final_val = _evaluate(model, val_loader, config, device)
    if config.training.checkpoint_every:
        save_resume(resume_path, model, optimizer, scaler, config, epoch=config.training.epochs,
                    micro_step=0, global_step=global_step, best_val=min(best_val, final_val["loss"]),
                    world_size=world_size, rank=rank, device=device, complete=True)
    if _is_main_process(rank):
        torch.save(
            {
                "model": _unwrap_model(model).state_dict(),
                "config": config,
                "val_loss": final_val["loss"],
                "step": global_step,
            },
            output_dir / "last.pt",
        )
        print(
            f"[done] val_loss={final_val['loss']:.4f}",
            flush=True,
        )
        _wandb_log(
            wandb_run,
            {"loss/final_val": final_val["loss"]},
            step=max(global_step, 1),
        )
        if wandb_run is not None:
            wandb_run.finish()
    if distributed and dist.is_initialized():
        dist.destroy_process_group()
