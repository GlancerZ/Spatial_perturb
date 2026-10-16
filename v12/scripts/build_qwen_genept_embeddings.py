from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_pretrain.config import load_config
from spatial_pretrain.gene_init import build_global_gene_vocabulary_from_cache_dirs, resolve_slice_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GenePT-style gene init embeddings with Qwen3.")
    parser.add_argument("--config", required=True, help="Training config used to define the global gene vocabulary.")
    parser.add_argument("--summaries", required=True, help="JSONL file from scripts/fetch_ncbi_gene_summaries.py.")
    parser.add_argument(
        "--output",
        default="",
        help="Output .pt path. Defaults under data/pretrain_data/genept/artifacts/ based on the config stem.",
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B", help="HF model id or local path.")
    parser.add_argument(
        "--instruction",
        default=(
            "Encode a gene summary for downstream spatial transcriptomics pretraining. "
            "Capture gene function, pathways, localization, species context, and disease relevance."
        ),
        help="Optional Qwen instruction prefix. Use an empty string to disable.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max_length.")
    parser.add_argument(
        "--output-dim",
        type=int,
        default=0,
        help="Final output dimension. Defaults to config.model.hidden_dim.",
    )
    parser.add_argument(
        "--keep-full-dim",
        action="store_true",
        help="Keep the full Qwen embedding dimension instead of truncating to config.model.hidden_dim / --output-dim.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="Torch dtype for model weights on GPU: auto, float16, bfloat16, or float32.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Transformers attention implementation, e.g. sdpa or flash_attention_2.",
    )
    return parser.parse_args()


def _default_output_path(config_path: Path, *, keep_full_dim: bool) -> Path:
    suffix = "_qwen3_genept_full_init.pt" if keep_full_dim else "_qwen3_genept_init.pt"
    return ROOT / "data" / "pretrain_data" / "genept" / "artifacts" / f"{config_path.stem}{suffix}"


def _check_transformers_version() -> None:
    try:
        version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "transformers>=4.51.0 is required to build Qwen3 gene embeddings in this workspace."
        ) from exc
    numeric_parts: list[int] = []
    for part in version.split("."):
        digits = "".join(char for char in part if char.isdigit())
        if not digits:
            break
        numeric_parts.append(int(digits))
        if len(numeric_parts) == 2:
            break
    if len(numeric_parts) < 2:
        raise SystemExit(f"Could not parse the installed transformers version: {version}")
    major, minor = numeric_parts
    if (major, minor) < (4, 51):
        raise SystemExit(
            f"transformers>={4}.{51}.0 is required for Qwen3, but the active env has {version}."
        )


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(raw)


def _resolve_torch_dtype(raw: str, device: torch.device) -> torch.dtype:
    name = raw.strip().lower()
    if name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported --torch-dtype={raw!r}")


def _load_summary_records(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[str(record["symbol"])] = record
    return records


def _format_text(text: str, instruction: str) -> str:
    text = text.strip()
    if not instruction.strip():
        return text
    return f"Instruct: {instruction.strip()}\nQuery:{text}"


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def main() -> None:
    args = parse_args()
    _check_transformers_version()
    from transformers import AutoModel, AutoTokenizer

    config_path = Path(args.config)
    summary_path = Path(args.summaries)
    output_path = Path(args.output) if args.output else _default_output_path(config_path, keep_full_dim=args.keep_full_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    slice_specs = resolve_slice_specs(config)
    cache_dirs = [cache_dir for _, cache_dir, _ in slice_specs]
    gene_names = build_global_gene_vocabulary_from_cache_dirs(cache_dirs)
    summary_records = _load_summary_records(summary_path)

    texts: list[str] = []
    missing_symbols: list[str] = []
    for gene_name in gene_names:
        record = summary_records.get(gene_name)
        if record is None:
            missing_symbols.append(gene_name)
            text = f"Gene symbol: {gene_name}."
        else:
            text = str(record.get("text", "")).strip() or f"Gene symbol: {gene_name}."
        texts.append(_format_text(text, args.instruction))

    device = _resolve_device(args.device)
    torch_dtype = _resolve_torch_dtype(args.torch_dtype, device)
    output_dim = int(args.output_dim) if args.output_dim > 0 else int(config.model.hidden_dim)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, object] = {}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if device.type == "cuda":
        model_kwargs["torch_dtype"] = torch_dtype
    model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)
    model.eval()

    batches: list[torch.Tensor] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="qwen-genept"):
            stop = min(start + args.batch_size, len(texts))
            batch_texts = texts[start:stop]
            batch_dict = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
            outputs = model(**batch_dict)
            pooled = _last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            batches.append(pooled.to(dtype=torch.float32, device="cpu"))

    embeddings = torch.cat(batches, dim=0)
    full_dim = int(embeddings.shape[1])
    if args.keep_full_dim:
        output_dim = full_dim
    if output_dim > full_dim:
        raise ValueError(f"Requested output_dim={output_dim} exceeds model dim={full_dim}")
    truncation_strategy = "none"
    if output_dim < full_dim:
        embeddings = F.normalize(embeddings[:, :output_dim], p=2, dim=1)
        truncation_strategy = "leading_dims_mrl"

    artifact = {
        "embeddings": embeddings.contiguous(),
        "gene_names": gene_names,
        "model_name": args.model_name,
        "instruction": args.instruction,
        "summary_path": str(summary_path.resolve()),
        "full_embedding_dim": full_dim,
        "output_dim": int(embeddings.shape[1]),
        "truncation_strategy": truncation_strategy,
        "num_missing_summaries": len(missing_symbols),
    }
    torch.save(artifact, output_path)

    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "config": str(config_path.resolve()),
                "summaries": str(summary_path.resolve()),
                "output": str(output_path.resolve()),
                "model_name": args.model_name,
                "instruction": args.instruction,
                "num_genes": len(gene_names),
                "num_missing_summaries": len(missing_symbols),
                "missing_preview": missing_symbols[:50],
                "full_embedding_dim": full_dim,
                "output_dim": int(embeddings.shape[1]),
                "truncation_strategy": truncation_strategy,
                "device": str(device),
                "torch_dtype": str(torch_dtype),
                "attn_implementation": args.attn_implementation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[genept-build] saved {output_path} num_genes={len(gene_names)} output_dim={embeddings.shape[1]} "
        f"missing_summaries={len(missing_symbols)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
