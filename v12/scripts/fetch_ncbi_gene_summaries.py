from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_pretrain.config import load_config
from spatial_pretrain.gene_init import build_global_gene_vocabulary_from_cache_dirs, resolve_slice_specs

DATASETS_API = "https://api.ncbi.nlm.nih.gov/datasets/v2/gene/symbol/{symbol}/taxon/{taxon}/dataset_report"
MYGENE_API = "https://mygene.info/v3/query?q=symbol:{symbol}&species={taxon}&fields=symbol,name,summary,alias,entrezgene,type_of_gene&size=1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch NCBI gene summaries for the global training vocabulary.")
    parser.add_argument("--config", required=True, help="Training config used to define the global gene vocabulary.")
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path. Defaults under data/pretrain_data/genept/artifacts/ based on the config stem.",
    )
    parser.add_argument(
        "--taxon",
        default="auto",
        help="NCBI taxon to query. Use human, mouse, or auto. Defaults to auto for mixed-panel vocabularies.",
    )
    parser.add_argument(
        "--mygene-batch-size",
        type=int,
        default=256,
        help="Batch size for MyGene POST queries. Defaults to 256.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.12, help="Delay between NCBI requests.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of genes to fetch.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing JSONL file instead of resuming.")
    return parser.parse_args()


def _default_output_path(config_path: Path) -> Path:
    return ROOT / "data" / "pretrain_data" / "genept" / "artifacts" / f"{config_path.stem}_ncbi_gene_summaries.jsonl"


def _datasets_url(symbol: str, taxon: str) -> str:
    return DATASETS_API.format(symbol=quote(symbol, safe=""), taxon=quote(taxon, safe=""))


def _fetch_payload(symbol: str, taxon: str) -> tuple[dict, str]:
    url = _datasets_url(symbol, taxon)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "spatial-pretrain-v3/0.1",
        },
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response), url


def _pick_gene_record(payload: dict, symbol: str) -> dict:
    reports = payload.get("reports") or []
    if not reports:
        return {}
    wanted = symbol.casefold()
    for report in reports:
        gene = report.get("gene") or {}
        if str(gene.get("symbol", "")).casefold() == wanted:
            return gene
    return reports[0].get("gene") or {}


def _mygene_url(symbol: str, taxon: str) -> str:
    return MYGENE_API.format(symbol=quote(symbol, safe=""), taxon=quote(taxon, safe=""))


def _fetch_mygene_record(symbol: str, taxon: str) -> tuple[dict, str]:
    url = _mygene_url(symbol, taxon)
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "spatial-pretrain-v3/0.1",
        },
    )
    with urlopen(request, timeout=60) as response:
        payload = json.load(response)
    hits = payload.get("hits") or []
    if not hits:
        return {}, url
    for hit in hits:
        if str(hit.get("symbol", "")).casefold() == symbol.casefold():
            return hit, url
    return hits[0], url


def _fetch_mygene_batch(symbols: list[str], taxon: str) -> tuple[list[dict], str]:
    payload = urlencode(
        {
            "q": ",".join(symbols),
            "scopes": "symbol",
            "species": taxon,
            "fields": "symbol,name,summary,alias,entrezgene,type_of_gene",
        }
    ).encode()
    request = Request(
        "https://mygene.info/v3/query",
        data=payload,
        headers={
            "Accept": "application/json",
            "User-Agent": "spatial-pretrain-v3/0.1",
        },
    )
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise TypeError(f"Expected MyGene batch response list, got {type(payload)}")
    return payload, "https://mygene.info/v3/query"


def _join_summary(gene: dict) -> str:
    entries = gene.get("summary") or []
    if isinstance(entries, str):
        return entries.strip()
    parts = [
        str(entry.get("description", "")).strip()
        for entry in entries
        if str(entry.get("description", "")).strip()
    ]
    return " ".join(parts)


def _build_text(symbol: str, gene: dict) -> str:
    if not gene:
        return f"Gene symbol: {symbol}."
    parts = [f"Gene symbol: {gene.get('symbol', symbol)}."]
    description = str(gene.get("description", "") or gene.get("name", "")).strip()
    if description:
        parts.append(f"Official description: {description}.")
    gene_type = str(gene.get("type", "")).strip()
    if gene_type:
        parts.append(f"Gene type: {gene_type.replace('_', ' ').lower()}.")
    raw_synonyms = gene.get("synonyms") or gene.get("alias") or []
    if isinstance(raw_synonyms, str):
        raw_synonyms = [raw_synonyms]
    synonyms = [str(value).strip() for value in raw_synonyms if str(value).strip()]
    if synonyms:
        parts.append(f"Aliases: {', '.join(synonyms[:12])}.")
    summary_text = _join_summary(gene)
    if summary_text:
        parts.append(f"NCBI summary: {summary_text}")
    return " ".join(parts)


def _load_existing(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    records: dict[str, dict] = {}
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            symbol = str(record["symbol"])
            records[symbol] = record
    return records


def _auto_taxa(symbol: str) -> list[str]:
    # Mouse gene symbols are usually title-cased (e.g. Actb), while human symbols
    # are typically uppercase. Try the more likely taxon first, then fall back.
    if any(char.islower() for char in symbol[1:]):
        return ["mouse", "human"]
    return ["human", "mouse"]


def _resolve_taxa(symbol: str, requested: str) -> list[str]:
    taxon = requested.strip().lower()
    if taxon == "auto":
        return _auto_taxa(symbol)
    return [taxon]


def _fetch_gene(symbol: str, requested_taxon: str) -> tuple[dict, str, str]:
    last_url = ""
    for taxon in _resolve_taxa(symbol, requested_taxon):
        try:
            payload, source_url = _fetch_payload(symbol, taxon)
            last_url = source_url
            gene = _pick_gene_record(payload, symbol)
            if gene:
                gene = dict(gene)
                gene["_source"] = "ncbi"
                return gene, source_url, taxon
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
        try:
            gene, source_url = _fetch_mygene_record(symbol, taxon)
            last_url = source_url
            if gene:
                gene = dict(gene)
                gene["_source"] = "mygene"
                return gene, source_url, taxon
        except (HTTPError, URLError, TimeoutError, OSError):
            pass
    return {}, last_url, _resolve_taxa(symbol, requested_taxon)[0]


def _pick_mygene_hit(symbol: str, hits: list[dict]) -> dict:
    exact_casefold = symbol.casefold()
    for hit in hits:
        if str(hit.get("symbol", "")).casefold() == exact_casefold:
            return hit
    for hit in hits:
        if str(hit.get("query", "")).casefold() == exact_casefold:
            return hit
    return hits[0] if hits else {}


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else _default_output_path(config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    slice_specs = resolve_slice_specs(config)
    cache_dirs = [cache_dir for _, cache_dir, _ in slice_specs]
    gene_names = build_global_gene_vocabulary_from_cache_dirs(cache_dirs)
    if args.limit > 0:
        gene_names = gene_names[: args.limit]

    existing = {} if args.overwrite else _load_existing(output_path)
    mode = "w" if args.overwrite else "a"
    num_written = 0
    num_reused = 0
    pending = [symbol for symbol in gene_names if symbol not in existing]
    resolved_records: dict[str, dict] = {}

    max_passes = max(len(_resolve_taxa(symbol, args.taxon)) for symbol in pending) if pending else 0
    for pass_idx in range(max_passes):
        symbols_for_pass: dict[str, list[str]] = {}
        for symbol in pending:
            taxa = _resolve_taxa(symbol, args.taxon)
            if pass_idx >= len(taxa) or symbol in resolved_records:
                continue
            symbols_for_pass.setdefault(taxa[pass_idx], []).append(symbol)
        for taxon, symbols in symbols_for_pass.items():
            for start in range(0, len(symbols), args.mygene_batch_size):
                batch_symbols = symbols[start : start + args.mygene_batch_size]
                try:
                    batch_hits, source_url = _fetch_mygene_batch(batch_symbols, taxon)
                except (HTTPError, URLError, TimeoutError, OSError):
                    batch_hits = []
                    source_url = "https://mygene.info/v3/query"
                grouped: dict[str, list[dict]] = {}
                for hit in batch_hits:
                    query = str(hit.get("query", "")).strip()
                    if query:
                        grouped.setdefault(query, []).append(hit)
                for symbol in batch_symbols:
                    if symbol in resolved_records:
                        continue
                    hits = grouped.get(symbol, [])
                    gene = _pick_mygene_hit(symbol, hits)
                    if gene and not bool(gene.get("notfound")):
                        gene = dict(gene)
                        gene["_source"] = "mygene"
                        gene["_source_url"] = source_url
                        gene["_taxon_used"] = taxon
                        resolved_records[symbol] = gene

    with output_path.open(mode, encoding="utf-8") as handle:
        for idx, symbol in enumerate(gene_names, start=1):
            if symbol in existing:
                num_reused += 1
                continue
            if symbol in resolved_records:
                gene = resolved_records[symbol]
                source_url = str(gene.get("_source_url", ""))
                taxon_used = str(gene.get("_taxon_used", _resolve_taxa(symbol, args.taxon)[0]))
            else:
                gene = {}
                source_url = ""
                taxon_used = _resolve_taxa(symbol, args.taxon)[0]
            record = {
                "symbol": symbol,
                "official_symbol": str(gene.get("symbol", symbol)),
                "gene_id": str(gene.get("gene_id", "")),
                "description": str(gene.get("description", "")).strip(),
                "gene_type": str(gene.get("type", "")).strip(),
                "summary": _join_summary(gene),
                "synonyms": gene.get("synonyms") or gene.get("alias") or [],
                "found": bool(gene),
                "source_url": source_url,
                "taxon_used": taxon_used,
                "provider": str(gene.get("_source", "")),
                "text": _build_text(symbol, gene),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            num_written += 1
            print(
                f"[genept-fetch] {idx}/{len(gene_names)} symbol={symbol} found={record['found']} taxon={taxon_used}",
                flush=True,
            )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    print(
        f"[genept-fetch] done output={output_path} written={num_written} reused={num_reused} total={len(gene_names)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
