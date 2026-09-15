from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CODEX_ROOT = Path("/path/to/legacy/workspace")


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    output_dir: str


@dataclass
class DataConfig:
    h5ad_path: str
    cache_dir: str
    lr_db_path: str
    split_seed: int
    train_fraction: float
    batch_size: int
    num_workers: int
    max_cells: int
    chunk_size: int
    top_genes: int
    num_bins: int
    knn_k: int
    sampling_strategy: str
    subgraph_size: int
    nearest_k: int
    lr_k: int
    candidate_k: int
    lr_temperature: float
    gene_sampling_strategy: str = "top"
    split_strategy: str = "random"
    split_blocks_x: int = 1
    split_blocks_y: int = 1
    restrict_subgraph_to_split: bool = False
    center_selection_strategy: str = "all"
    max_center_overlap_ratio: float = 1.0
    max_cell_reuse: int = 0
    resample_views_each_epoch: bool = False
    h5ad_paths: list[str] = field(default_factory=list)
    cache_dirs: list[str] = field(default_factory=list)
    view_prep_workers: int = 0
    nuclear_tokens: int = 256
    cytoplasmic_tokens: int = 256
    use_lr_prior: bool = True
    # Frozen positive-expression boundaries fitted on pretraining train cells.
    compartment_bin_edges: list[float] = field(default_factory=list)


@dataclass
class ModelConfig:
    hidden_dim: int
    num_layers: int
    gene_heads: int
    cell_heads: int
    dropout: float
    ffn_mult: int
    use_genept: bool
    genept_path: str
    cell_attention_mode: str = "identity_flat"
    cell_token_dim: int = 16
    require_flash_attention: bool = False
    cell_mla_latent_dim: int = 128
    cell_mla_qk_head_dim: int = 64
    cell_mla_v_head_dim: int = 64
    cell_mla_output_rank: int = 128
    cell_spatial_bias: bool = False
    cell_angle_bins: int = 16
    cell_distance_edges: list[float] = field(default_factory=lambda: [10., 20., 40., 80., 160., 320.])
    # Multiplicative conversion from input coordinate units to micrometres.
    cell_coordinate_scale: float = 1.0


@dataclass
class MaskingConfig:
    general_mask_ratio: float
    center_mask_ratio: float
    receptor_keep_ratio: float
    uniform_mask_ratio: float = -1.0


@dataclass
class LossConfig:
    mask_reduction: str = "per_token"
    mask_scope: str = "all_cells"


@dataclass
class OptimizerConfig:
    lr: float
    weight_decay: float
    betas: list[float]
    name: str = "adamw"
    eps: float = 1e-8
    muon_lr: float = 1.0e-2
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_nesterov: bool = True
    muon_adjust_lr_fn: str = "match_rms_adamw"


@dataclass
class TrainingConfig:
    epochs: int
    device: str
    amp: bool
    grad_clip_norm: float
    log_every: int
    val_every: int
    max_train_steps: int
    max_val_batches: int
    grad_accum_steps: int = 1
    amp_dtype: str = "bfloat16"
    lr_schedule: str = "constant"
    warmup_updates: int = 0
    warmup_ratio: float = -1.0
    min_lr_ratio: float = 1.0
    parallelism: str = "single"
    device_ids: list[int] = field(default_factory=list)
    gradient_checkpointing: bool = False
    checkpoint_every: int = 0
    auto_resume: bool = False
    run_update_limit: int = 0
    save_initial_checkpoint: bool = False


@dataclass
class WandbConfig:
    enabled: bool = False
    mode: str = "offline"
    project: str = "spatial_pretrain_v2"
    entity: str = ""
    name: str = ""
    group: str = ""
    job_type: str = "pretrain"
    dir: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    masking: MaskingConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    wandb: WandbConfig


def _require(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in mapping:
        raise KeyError(f"Missing config section: {key}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise TypeError(f"Config section '{key}' must be a mapping")
    return value


def _resolve_project_path(raw_path: str, *, allow_empty: bool = False) -> str:
    if not raw_path:
        return raw_path if allow_empty else str(PROJECT_ROOT)

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        if path.is_relative_to(LEGACY_CODEX_ROOT):
            relative = path.relative_to(LEGACY_CODEX_ROOT)
            return str((PROJECT_ROOT.parent / relative).resolve())
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise TypeError("Top-level config must be a mapping")

    experiment_raw = dict(_require(raw, "experiment"))
    experiment_raw["output_dir"] = _resolve_project_path(experiment_raw["output_dir"])

    data_raw = dict(_require(raw, "data"))
    data_raw["h5ad_path"] = _resolve_project_path(data_raw["h5ad_path"])
    data_raw["cache_dir"] = _resolve_project_path(data_raw["cache_dir"])
    data_raw["lr_db_path"] = _resolve_project_path(data_raw["lr_db_path"], allow_empty=True)
    if "h5ad_paths" in data_raw:
        data_raw["h5ad_paths"] = [
            _resolve_project_path(path)
            for path in data_raw["h5ad_paths"]
        ]
    if "cache_dirs" in data_raw:
        data_raw["cache_dirs"] = [
            _resolve_project_path(path)
            for path in data_raw["cache_dirs"]
        ]

    model_raw = dict(_require(raw, "model"))
    model_raw["genept_path"] = _resolve_project_path(model_raw["genept_path"], allow_empty=True)
    wandb_raw = dict(raw.get("wandb", {}))
    if "dir" in wandb_raw:
        wandb_raw["dir"] = _resolve_project_path(wandb_raw["dir"], allow_empty=True)

    return Config(
        experiment=ExperimentConfig(**experiment_raw),
        data=DataConfig(**data_raw),
        model=ModelConfig(**model_raw),
        masking=MaskingConfig(**_require(raw, "masking")),
        loss=LossConfig(**_require(raw, "loss")),
        optimizer=OptimizerConfig(**_require(raw, "optimizer")),
        training=TrainingConfig(**_require(raw, "training")),
        wandb=WandbConfig(**wandb_raw),
    )
