#!/usr/bin/env python3
"""Bounded, conservative discovery and download helper for spatial datasets.

This module intentionally uses only the Python standard library.  ``collect``
reads a reviewed registry and, when explicitly enabled, inspects a bounded
number of public HTTP(S) pages.  ``fetch`` downloads only rows explicitly
present in a manifest and never extracts or executes downloaded content.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
from html.parser import HTMLParser
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import uuid


SCHEMA_VERSION = 1
CAPABILITY_NAMES = (
    "molecule_coordinates",
    "nucleus_assignment",
    "nucleus_boundaries",
    "cell_boundaries",
    "registered_images",
    "coordinate_registration",
    "cell_assignment",
)
CAPABILITY_VALUES = {"yes", "no", "unknown"}
EVIDENCE_LEVELS = {"candidate", "documented", "verified"}
SOURCE_KINDS = {"dataset", "catalog", "documentation"}
COMPARTMENT_METHODS = {
    "explicit_assignment", "registered_boundaries", "resegment", "unknown"
}
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
GEO_RE = re.compile(r"\bG(?:SE|SM|PL|DS)\d+\b", re.IGNORECASE)
DOWNLOAD_SUFFIXES = (
    ".h5ad", ".h5", ".hdf5", ".csv", ".tsv", ".txt", ".json",
    ".jsonl", ".parquet", ".zip", ".tar", ".tar.gz", ".tgz", ".gz",
    ".bz2", ".xz", ".loom", ".mtx", ".mtx.gz", ".rds", ".zarr",
)
USER_AGENT = "subcellular-data-agent/1.0 (bounded metadata collector)"


class RegistryError(ValueError):
    """Raised for invalid registry input."""


class UnsafeURLError(ValueError):
    """Raised when a URL violates the network safety policy."""


class PageTooLargeError(ValueError):
    """Raised when a metadata page exceeds its byte budget."""


class DownloadLimitError(ValueError):
    """Raised when a download would exceed a configured byte budget."""


class IntegrityError(ValueError):
    """Raised when optional manifest integrity metadata does not match."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def unique_run_dir(output: Path, prefix: str = "run") -> Path:
    output.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for _ in range(100):
        candidate = output / f"{prefix}-{stamp}-{uuid.uuid4().hex[:10]}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique run directory")


def load_registry(path: Path) -> list[dict[str, Any]]:
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read registry {path}: {exc}") from exc
    if not isinstance(envelope, dict) or envelope.get("schema_version") != SCHEMA_VERSION:
        raise RegistryError("registry must be an object with schema_version: 1")
    sources = envelope.get("sources")
    if not isinstance(sources, list):
        raise RegistryError("registry sources must be an array")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(sources):
        if not isinstance(raw, dict):
            raise RegistryError(f"sources[{index}] must be an object")
        source = dict(raw)  # retain unknown fields in catalog.json
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise RegistryError(f"sources[{index}].id must be a non-empty string")
        if source_id in seen:
            raise RegistryError(f"duplicate source id: {source_id}")
        seen.add(source_id)
        _validate_source(source, index)
        validated.append(source)
    return validated


def _validate_source(source: Mapping[str, Any], index: int) -> None:
    prefix = f"sources[{index}]"
    for field in ("species", "discovery_urls", "evidence", "assets"):
        value = source.get(field, [])
        if not isinstance(value, list):
            raise RegistryError(f"{prefix}.{field} must be an array")
    if "inventory_keys" in source and not isinstance(source["inventory_keys"], list):
        raise RegistryError(f"{prefix}.inventory_keys must be an array when provided")
    capabilities = source.get("capabilities", {})
    if not isinstance(capabilities, dict):
        raise RegistryError(f"{prefix}.capabilities must be an object")
    for name, value in capabilities.items():
        if name in CAPABILITY_NAMES and value not in CAPABILITY_VALUES:
            raise RegistryError(f"{prefix}.capabilities.{name} has invalid value {value!r}")
    level = source.get("evidence_level", "candidate")
    if level not in EVIDENCE_LEVELS:
        raise RegistryError(f"{prefix}.evidence_level has invalid value {level!r}")
    method = source.get("compartment_method", "unknown")
    if method not in COMPARTMENT_METHODS:
        raise RegistryError(f"{prefix}.compartment_method has invalid value {method!r}")
    kind = source.get("source_kind", "dataset")
    if kind not in SOURCE_KINDS:
        raise RegistryError(f"{prefix}.source_kind has invalid value {kind!r}")
    priority = source.get("priority", 3)
    if priority not in (1, 2, 3):
        raise RegistryError(f"{prefix}.priority must be 1, 2, or 3")
    for pos, evidence in enumerate(source.get("evidence", [])):
        if not isinstance(evidence, dict) or not isinstance(evidence.get("claim"), str):
            raise RegistryError(f"{prefix}.evidence[{pos}] must have a string claim")
        if evidence.get("url"):
            validate_public_url(str(evidence["url"]), resolve=False)
    for field in ("source_url",):
        if source.get(field):
            validate_public_url(str(source[field]), resolve=False)
    for value in source.get("discovery_urls", []):
        validate_public_url(str(value), resolve=False)
    for pos, asset in enumerate(source.get("assets", [])):
        if not isinstance(asset, dict) or not asset.get("url") or not asset.get("role"):
            raise RegistryError(f"{prefix}.assets[{pos}] requires url and role")
        validate_public_url(str(asset["url"]), resolve=False)
        if asset.get("sha256") and not re.fullmatch(r"[0-9a-fA-F]{64}", str(asset["sha256"])):
            raise RegistryError(f"{prefix}.assets[{pos}].sha256 must be 64 hexadecimal characters")
        if asset.get("size_bytes") is not None:
            if not isinstance(asset["size_bytes"], int) or asset["size_bytes"] < 0:
                raise RegistryError(f"{prefix}.assets[{pos}].size_bytes must be a non-negative integer")


def _is_forbidden_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return True
    return not ip.is_global


def validate_public_url(url: str, *, resolve: bool = True) -> str:
    """Validate scheme, credentials, port, and every resolved address."""
    if not isinstance(url, str) or len(url) > 8192:
        raise UnsafeURLError("URL is missing or too long")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeURLError("only http(s) URLs are allowed")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise UnsafeURLError("URL host is missing or contains credentials")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in {"localhost", "localhost.localdomain", "metadata.google.internal"}:
        raise UnsafeURLError(f"forbidden host: {hostname}")
    try:
        port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    except ValueError as exc:
        raise UnsafeURLError("invalid URL port") from exc
    try:
        literal = ipaddress.ip_address(hostname.split("%", 1)[0])
    except ValueError:
        literal = None
    if literal is not None and not literal.is_global:
        raise UnsafeURLError(f"non-public address is forbidden: {literal}")
    if resolve:
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)}
        except socket.gaierror as exc:
            raise UnsafeURLError(f"cannot resolve host {hostname}: {exc}") from exc
        if not addresses:
            raise UnsafeURLError(f"host {hostname} resolved to no addresses")
        forbidden = sorted(address for address in addresses if _is_forbidden_ip(address))
        if forbidden:
            raise UnsafeURLError(f"host resolves to non-public address: {', '.join(forbidden)}")
    return urllib.parse.urlunsplit(parsed)


class ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    max_redirections = 8

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int,
                         msg: str, headers: Any, newurl: str) -> urllib.request.Request | None:
        safe_url = validate_public_url(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, safe_url)


def public_opener() -> urllib.request.OpenerDirector:
    # Respect the site's configured proxy environment; every requested and
    # redirected destination is still validated independently.
    return urllib.request.build_opener(urllib.request.ProxyHandler(), ValidatingRedirectHandler())


def open_public_url(url: str, timeout: float) -> Any:
    safe_url = validate_public_url(url)
    request = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    return public_opener().open(request, timeout=timeout)


def read_bounded(response: Any, limit: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = None
        if declared_size is not None and declared_size > limit:
            raise PageTooLargeError(f"declared page size exceeds {limit} bytes")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(65536, limit - total + 1))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise PageTooLargeError(f"page exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        for field in ("href", "src"):
            if attrs_dict.get(field):
                self.links.append(str(attrs_dict[field]))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _json_urls(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield from URL_RE.findall(value)
    elif isinstance(value, list):
        for item in value:
            yield from _json_urls(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _json_urls(item)


def extract_metadata(body: bytes, content_type: str, base_url: str) -> tuple[list[str], list[str]]:
    text = body.decode("utf-8", errors="replace")
    links: list[str] = []
    searchable = text
    lowered = content_type.lower()
    if "html" in lowered or re.search(r"<\s*(?:html|a)\b", text[:1000], re.I):
        parser = LinkParser()
        parser.feed(text)
        links.extend(urllib.parse.urljoin(base_url, link) for link in parser.links)
        searchable = " ".join(parser.text)
    elif "json" in lowered or text.lstrip().startswith(("{", "[")):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            links.extend(_json_urls(decoded))
    links.extend(URL_RE.findall(text))
    accessions = sorted({value.upper() for value in GEO_RE.findall(searchable + " " + text)})
    return sorted(set(links)), accessions


def is_download_candidate(url: str) -> bool:
    path = urllib.parse.urlsplit(url).path.lower()
    return path.endswith(DOWNLOAD_SUFFIXES) or any(
        marker in url.lower() for marker in ("download", "attachment", "ftp.ncbi.nlm.nih.gov/geo")
    )


def discovered_link_kind(url: str) -> str:
    if is_download_candidate(url):
        return "download"
    lowered = url.lower()
    if any(marker in lowered for marker in ("dataset", "catalog", "study", "studies", "geo/query/acc", "accession")):
        return "dataset_or_catalog"
    return "other"


def evidence_claims(source: Mapping[str, Any]) -> set[str]:
    return {
        str(item.get("claim", "")).strip().lower()
        for item in source.get("evidence", [])
        if isinstance(item, dict) and item.get("url") and item.get("detail")
    }


def _claim_supports(claims: set[str], capability: str) -> bool:
    readable = capability.replace("_", " ")
    for claim in claims:
        normalized = re.sub(r"[_=:-]+", " ", claim)
        # Claims are curated evidence pointers, not inferred truths. A negative
        # or uncertain description must never become positive via a substring.
        if re.search(r"\b(no|not|without|unavailable|unknown|absent|missing|unsupported|unverified)\b", normalized):
            continue
        if re.search(r"(?<![a-z0-9_])" + re.escape(capability) + r"(?![a-z0-9_])", claim) or readable in normalized:
            return True
    return False


def classify_source(source: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Return a conservative metadata classification and supporting reasons."""
    if source.get("source_kind", "dataset") != "dataset":
        return "discovery_only", ["catalog_and_documentation_sources_are_discovery_only"]
    if source.get("training_use") == "format_validation_only":
        return "format_validation_only", ["explicitly_excluded_from_training_candidates"]
    capabilities = source.get("capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
    level = source.get("evidence_level", "candidate")
    claims = evidence_claims(source)

    def supported(name: str) -> bool:
        return capabilities.get(name, "unknown") == "yes" and _claim_supports(claims, name)

    reasons: list[str] = []
    if level not in {"documented", "verified"}:
        reasons.append("evidence_level_below_documented")
        return "candidate", reasons
    if source.get("compartment_method", "unknown") == "explicit_assignment":
        required = ("molecule_coordinates", "cell_assignment", "nucleus_assignment")
        missing = [name for name in required if not supported(name)]
        if not missing:
            return "documented_direct", ["documented_explicit_molecule_cell_nucleus_assignment"]
        reasons.extend(f"missing_supported_{name}" for name in missing)
    if source.get("compartment_method", "unknown") == "registered_boundaries":
        required = (
            "molecule_coordinates", "nucleus_boundaries", "cell_boundaries",
            "coordinate_registration",
        )
        missing = [name for name in required if not supported(name)]
        if not missing:
            return "documented_geometry", ["documented_registered_cell_and_nucleus_geometry"]
        reasons.extend(f"missing_supported_{name}" for name in missing)
    if not reasons:
        reasons.append("no_supported_direct_or_geometry_route")
    return "candidate", reasons


def safe_tsv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    else:
        text = str(value)
    # Spreadsheet programs interpret these prefixes as formulas.
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        text = "'" + text
    return text


def render_tsv(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> str:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writerow({field: field for field in fields})
    for row in rows:
        writer.writerow({field: safe_tsv_cell(row.get(field, "")) for field in fields})
    return stream.getvalue()


def load_inventory(path: Path | None) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    if path is None:
        return result
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = set(reader.fieldnames or [])
            key_field = "source_id" if "source_id" in fields else "dataset_key" if "dataset_key" in fields else None
            if key_field is None:
                raise RegistryError("inventory TSV requires a source_id or dataset_key column")
            for row in reader:
                inventory_key = (row.get(key_field) or "").strip()
                if inventory_key:
                    result.setdefault(inventory_key, []).append(dict(row))
    except OSError as exc:
        raise RegistryError(f"cannot read inventory {path}: {exc}") from exc
    return result


def _asset_candidate(source: Mapping[str, Any], asset: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": source["id"],
        "url": asset["url"],
        "role": asset["role"],
        "filename": asset.get("filename", ""),
        "evidence_level": source.get("evidence_level", "candidate"),
        "origin": "registry_asset",
        "availability": "not_checked",
        "expected_size_bytes": asset.get("size_bytes", ""),
        "expected_sha256": asset.get("sha256", ""),
        # This is a metadata-supported candidate, never a train-ready file.
        # A processed cell matrix, landing page or schema document cannot qualify.
        "training_candidate": classify_source(source)[0] in {"documented_direct", "documented_geometry"}
        and asset.get("role") in {"raw_output_bundle", "transcripts", "molecule_coordinates", "cell_boundaries", "nucleus_boundaries", "segmentation_masks"},
    }


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def cache_paths(cache_dir: Path, url: str) -> tuple[Path, Path, Path, Path]:
    key = cache_key(url)
    return (
        cache_dir / f"{key}.body",
        cache_dir / f"{key}.json",
        cache_dir / f"{key}.headers",
        cache_dir / f"{key}.meta.tsv",
    )


def read_cached_page(cache_dir: Path, url: str, limit: int) -> tuple[bytes, str, dict[str, Any]]:
    body_path, json_path, headers_path, meta_path = cache_paths(cache_dir, url)
    try:
        size = body_path.stat().st_size
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"cache body missing: {body_path.name}") from exc
    if size > limit:
        raise PageTooLargeError(f"cached page exceeds {limit} bytes")
    body = body_path.read_bytes()
    content_type = "application/octet-stream"
    provenance: dict[str, Any] = {
        "cache_key": cache_key(url), "cache_body": str(body_path),
        "cache_body_mtime": dt.datetime.fromtimestamp(
            body_path.stat().st_mtime, tz=dt.timezone.utc
        ).replace(microsecond=0).isoformat(),
        "cache_body_sha256": hashlib.sha256(body).hexdigest(),
    }
    if json_path.exists():
        try:
            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            provenance["cache_provenance"] = parsed
            if isinstance(parsed, dict) and parsed.get("content_type"):
                content_type = str(parsed["content_type"])
        except (OSError, json.JSONDecodeError) as exc:
            provenance["cache_provenance_error"] = str(exc)
    if headers_path.exists():
        try:
            headers_text = headers_path.read_text(encoding="utf-8", errors="replace")
            provenance["cache_headers_sha256"] = hashlib.sha256(headers_text.encode("utf-8")).hexdigest()
            matches = re.findall(r"^Content-Type:\s*([^\r\n]+)", headers_text, re.I | re.M)
            if matches:
                content_type = matches[-1].strip()
        except OSError as exc:
            provenance["cache_headers_error"] = str(exc)
    if meta_path.exists():
        try:
            provenance["cache_meta_sha256"] = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        except OSError as exc:
            provenance["cache_meta_error"] = str(exc)
    return body, content_type, provenance


def _load_previous(output: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    latest_path = output / "latest.json"
    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        prior_dir = output / str(latest["run_dir"])
        catalog = json.loads((prior_dir / "catalog.json").read_text(encoding="utf-8"))
        with (prior_dir / "download_candidates.tsv").open("r", encoding="utf-8", newline="") as handle:
            candidates = list(csv.DictReader(handle, delimiter="\t"))
        return catalog, candidates
    except (OSError, KeyError, TypeError, json.JSONDecodeError, csv.Error):
        return {}, []


def compute_changes(sources: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]],
                    discovered_links: Sequence[Mapping[str, Any]], prior_catalog: Mapping[str, Any],
                    prior_candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    current_hashes = {str(row["id"]): str(row.get("registry_record_sha256", "")) for row in sources}
    previous_hashes = {
        str(row["id"]): str(row.get("registry_record_sha256", ""))
        for row in prior_catalog.get("sources", []) if isinstance(row, dict) and row.get("id")
    }
    current_links = {
        (str(row.get("source_id", "")), str(row.get("url", "")))
        for row in list(candidates) + list(discovered_links)
    }
    previous_links = {
        (str(row.get("source_id", "")), str(row.get("url", "")))
        for row in list(prior_candidates) + list(prior_catalog.get("discovered_links", []))
    }
    return {
        "baseline_available": bool(prior_catalog),
        "new_source_ids": sorted(current_hashes.keys() - previous_hashes.keys()),
        "removed_source_ids": sorted(previous_hashes.keys() - current_hashes.keys()),
        "changed_source_ids": sorted(
            key for key in current_hashes.keys() & previous_hashes.keys()
            if current_hashes[key] != previous_hashes[key]
        ),
        "new_links": [{"source_id": key[0], "url": key[1]} for key in sorted(current_links - previous_links)],
        # A failed/restricted page or changed parser is not proof of deletion.
        "removed_links": [],
        "unobserved_links": [{"source_id": key[0], "url": key[1]} for key in sorted(previous_links - current_links)],
    }


def collect_once(args: argparse.Namespace) -> Path:
    started = utc_now()
    output_dir = Path(args.output)
    prior_catalog, prior_candidates = _load_previous(output_dir)
    run_dir = unique_run_dir(output_dir, "run")
    errors: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    discovered_links: list[dict[str, Any]] = []
    try:
        sources = load_registry(Path(args.registry))
        inventory = load_inventory(Path(args.inventory) if args.inventory else None)
        cache_download_status: dict[str, dict[str, str]] = {}
        if args.page_cache:
            status_path = Path(args.page_cache) / "download_status.tsv"
            if status_path.exists():
                with status_path.open(encoding="utf-8", newline="") as handle:
                    cache_download_status = {row["url"]: dict(row) for row in csv.DictReader(handle, delimiter="\t") if row.get("url")}
        request_rows: list[dict[str, str]] = []
        page_budget = args.max_pages
        for source in sources:
            source["registry_record_sha256"] = hashlib.sha256(
                json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            classification, reasons = classify_source(source)
            source["classification"] = classification
            source["classification_reasons"] = reasons
            source["file_validation_status"] = "pending_file_validation"
            source["availability"] = "not_checked"
            inventory_keys = [str(source["id"])] + [str(value) for value in source.get("inventory_keys", [])]
            source["inventory_records"] = [
                row for key in dict.fromkeys(inventory_keys) for row in inventory.get(key, [])
            ]
            # Inventory is descriptive only, including processed_h5ad rows.
            source["inventory_status"] = "registered_unvalidated" if source["inventory_records"] else "not_listed"
            for asset in source.get("assets", []):
                candidates.append(_asset_candidate(source, asset))
            accessions = {str(source.get("accession", "")).upper()} if source.get("accession") else set()
            discovery_urls = list(dict.fromkeys(
                ([source["source_url"]] if source.get("source_url") else [])
                + list(source.get("discovery_urls", []))
            ))
            request_rows.extend({"cache_key": cache_key(url), "url": url} for url in discovery_urls)
            if args.online or args.page_cache:
                for url in discovery_urls:
                    if page_budget <= 0:
                        errors.append({"source_id": source["id"], "url": url, "error": "max_pages_exhausted"})
                        continue
                    page_budget -= 1
                    record: dict[str, Any] = {"source_id": source["id"], "requested_url": url, "fetched_at": utc_now()}
                    if url in cache_download_status:
                        record["cache_download_status"] = cache_download_status[url]
                    try:
                        if args.page_cache:
                            body, content_type, provenance = read_cached_page(Path(args.page_cache), url, args.max_page_bytes)
                            final_url = url
                            links, found_accessions = extract_metadata(body, content_type, final_url)
                            accessions.update(found_accessions)
                            record.update({
                                "status": "ok", "retrieval": "page_cache", "final_url": final_url,
                                "content_type": content_type, "bytes": len(body),
                                "sha256": hashlib.sha256(body).hexdigest(), "links_found": len(links),
                                **provenance,
                            })
                        elif args.online:
                            with open_public_url(url, args.timeout) as response:
                                body = read_bounded(response, args.max_page_bytes)
                                final_url = response.geturl()
                                content_type = response.headers.get("Content-Type", "application/octet-stream")
                                links, found_accessions = extract_metadata(body, content_type, final_url)
                                accessions.update(found_accessions)
                                record.update({
                                    "status": "ok", "retrieval": "online",
                                    "http_status": getattr(response, "status", 200),
                                    "final_url": final_url, "content_type": content_type,
                                    "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                                    "date": response.headers.get("Date", ""),
                                    "links_found": len(links),
                                })
                        else:
                            raise RuntimeError("unreachable retrieval mode")
                        for link in links:
                            try:
                                safe_link = validate_public_url(link, resolve=False)
                            except UnsafeURLError:
                                continue
                            kind = discovered_link_kind(safe_link)
                            discovered_links.append({
                                "source_id": source["id"], "origin": final_url,
                                "url": safe_link, "kind": kind,
                            })
                            if is_download_candidate(safe_link):
                                candidates.append({
                                    "source_id": source["id"], "url": safe_link,
                                    "role": "auto_discovered", "filename": "",
                                    "evidence_level": "candidate", "origin": final_url,
                                    "availability": "discovered_unchecked", "training_candidate": False,
                                })
                    except FileNotFoundError as exc:
                        record.update({"status": "not_fetched", "retrieval": "page_cache", "error": str(exc)})
                        errors.append(dict(record))
                    except Exception as exc:  # an individual page must not abort the bounded run
                        record.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
                        errors.append(dict(record))
                    pages.append(record)
            source["discovered_accessions"] = sorted(value for value in accessions if value)

        candidates = _dedupe_candidates(candidates)
        discovered_links = _dedupe_discovered_links(discovered_links)
        changes = compute_changes(sources, candidates, discovered_links, prior_catalog, prior_candidates)
        if args.emit_requests:
            deduped_requests = list({row["url"]: row for row in request_rows}.values())
            atomic_write_text(Path(args.emit_requests), render_tsv(deduped_requests, ("cache_key", "url")))
        catalog = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "policy": {
                "train_ready_is_never_asserted": True,
                "local_files": "pending_file_validation",
                "automatic_discovery_evidence_level": "candidate",
            },
            "sources": sources,
            "pages": pages,
            "errors": errors,
            "discovered_links": discovered_links,
            "changes": changes,
        }
        _write_collect_outputs(run_dir, catalog, candidates, discovered_links, started, args)
        _write_latest(Path(args.output), run_dir, "collect", changes)
        return run_dir
    except BaseException as exc:
        failure = {
            "command": "collect", "started_at": started, "finished_at": utc_now(),
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        }
        atomic_write_json(run_dir / "run.json", failure)
        raise


def _dedupe_candidates(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("source_id", "")), str(row.get("url", "")), str(row.get("role", "")))
        if key not in seen:
            seen.add(key)
            result.append(dict(row))
    return result


def _dedupe_discovered_links(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get("source_id", "")), str(row.get("origin", "")), str(row.get("url", "")))
        if key not in seen:
            seen.add(key)
            result.append(dict(row))
    return result


def _write_collect_outputs(run_dir: Path, catalog: Mapping[str, Any], candidates: list[dict[str, Any]],
                           discovered_links: list[dict[str, Any]], started: str,
                           args: argparse.Namespace) -> None:
    atomic_write_json(run_dir / "catalog.json", catalog)
    catalog_fields = (
        "id", "title", "platform", "species", "tissue", "source_url", "accession",
        "license", "source_kind", "priority", "evidence_level", "compartment_method",
        "classification", "classification_reasons", "file_validation_status", "availability",
        "inventory_status", "discovered_accessions", "notes",
    )
    atomic_write_text(run_dir / "catalog.tsv", render_tsv(catalog["sources"], catalog_fields))
    candidate_fields = (
        "source_id", "url", "role", "filename", "evidence_level", "origin",
        "availability", "training_candidate", "expected_size_bytes", "expected_sha256",
    )
    atomic_write_text(run_dir / "download_candidates.tsv", render_tsv(candidates, candidate_fields))
    atomic_write_text(
        run_dir / "discovered_links.tsv",
        render_tsv(discovered_links, ("source_id", "origin", "url", "kind")),
    )
    report_lines = [
        "# Subcellular data collection report", "", f"Generated: {html.escape(str(catalog['generated_at']))}", "",
        f"Sources: {len(catalog['sources'])}", f"Pages attempted: {len(catalog['pages'])}",
        f"Download candidates: {len(candidates)}", f"Errors: {len(catalog['errors'])}", "",
        "All local files remain `pending_file_validation`; this report never labels a source train-ready.", "",
        "## Changes since previous collection", "",
        f"Baseline available: {bool(catalog['changes']['baseline_available'])}",
        f"New sources: {len(catalog['changes']['new_source_ids'])}",
        f"Changed sources: {len(catalog['changes']['changed_source_ids'])}",
        f"Removed sources: {len(catalog['changes']['removed_source_ids'])}",
        f"New links: {len(catalog['changes']['new_links'])}",
        f"Previously seen links not observed this run (not confirmed removed): {len(catalog['changes']['unobserved_links'])}", "",
        "## Sources", "",
    ]
    for source in catalog["sources"]:
        report_lines.append(
            f"- **{html.escape(str(source.get('title') or source['id']))}** "
            f"(`{html.escape(str(source['id']))}`): {html.escape(str(source['classification']))}; "
            f"availability={html.escape(str(source['availability']))}"
        )
    if catalog["errors"]:
        report_lines.extend(["", "## Errors", ""])
        for error in catalog["errors"]:
            report_lines.append(
                f"- `{html.escape(str(error.get('source_id', '')))}` "
                f"{html.escape(str(error.get('url') or error.get('requested_url', '')))}: "
                f"{html.escape(str(error.get('error', 'unknown error')))}"
            )
    atomic_write_text(run_dir / "report.md", "\n".join(report_lines) + "\n")
    run = {
        "command": "collect", "status": "completed_with_errors" if catalog["errors"] else "completed", "started_at": started,
        "finished_at": utc_now(), "run_dir": str(run_dir.resolve()),
        "online": bool(args.online), "max_pages": args.max_pages,
        "max_page_bytes": args.max_page_bytes, "timeout": args.timeout,
        "counts": {"sources": len(catalog["sources"]), "pages": len(catalog["pages"]),
                   "candidates": len(candidates), "discovered_links": len(discovered_links),
                   "errors": len(catalog["errors"])},
        "changes": catalog["changes"],
    }
    atomic_write_json(run_dir / "run.json", run)


def _write_latest(output: Path, run_dir: Path, command: str, changes: Mapping[str, Any] | None = None) -> None:
    payload = {
        "command": command, "run_dir": run_dir.name,
        "run_path": str(run_dir.resolve()), "updated_at": utc_now(),
    }
    if changes is not None:
        payload["changes"] = changes
    atomic_write_json(output / "latest.json", payload)


def safe_filename(raw: str, url: str) -> str:
    name = raw.strip() if raw else Path(urllib.parse.unquote(urllib.parse.urlsplit(url).path)).name
    if not name:
        name = "download.bin"
    if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name or "\x00" in name:
        raise ValueError(f"unsafe filename: {name!r}")
    return name


def safe_component(raw: str) -> str:
    if not raw or raw in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", raw):
        raise ValueError(f"unsafe source_id path component: {raw!r}")
    return raw


def read_manifest(path: Path) -> list[dict[str, str]]:
    required = {"source_id", "url", "role", "filename", "evidence_level"}
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"manifest missing columns: {', '.join(sorted(missing))}")
            rows = [dict(row) for row in reader]
    except OSError as exc:
        raise ValueError(f"cannot read manifest {path}: {exc}") from exc
    for row in rows:
        safe_component(row["source_id"])
        validate_public_url(row["url"], resolve=False)
        safe_filename(row.get("filename", ""), row["url"])
        if row["evidence_level"] not in EVIDENCE_LEVELS:
            raise ValueError(f"invalid evidence_level: {row['evidence_level']!r}")
        expected_sha256 = (row.get("expected_sha256") or row.get("sha256") or "").strip().lower()
        if expected_sha256 and not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError(f"invalid expected SHA-256: {expected_sha256!r}")
        expected_size = (row.get("expected_size_bytes") or row.get("size_bytes") or "").strip()
        if expected_size:
            try:
                if int(expected_size) < 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(f"invalid expected size: {expected_size!r}") from exc
    return rows


def _download_one(row: Mapping[str, str], run_dir: Path, timeout: float,
                  max_file_bytes: int, total_remaining: int) -> tuple[dict[str, Any], int]:
    source_id = safe_component(row["source_id"])
    filename = safe_filename(row.get("filename", ""), row["url"])
    destination_dir = run_dir / "files" / source_id
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = destination_dir / filename
    part_path = destination_dir / f"{filename}.part"
    record: dict[str, Any] = {
        **dict(row), "status": "failed", "availability": "not_downloaded",
        "started_at": utc_now(), "path": str(final_path.relative_to(run_dir)),
        "part_path": str(part_path.relative_to(run_dir)), "bytes": 0, "sha256": "",
    }
    expected_sha256 = (row.get("expected_sha256") or row.get("sha256") or "").strip().lower()
    expected_size_text = (row.get("expected_size_bytes") or row.get("size_bytes") or "").strip()
    expected_size = int(expected_size_text) if expected_size_text else None
    record["expected_sha256"] = expected_sha256
    record["expected_size_bytes"] = expected_size if expected_size is not None else ""
    record["integrity"] = "pending" if expected_sha256 or expected_size is not None else "not_provided"
    if final_path.exists() or part_path.exists():
        record.update({
            "error_type": "FileExistsError",
            "error": f"refusing to overwrite existing path for {source_id}/{filename}",
            "finished_at": utc_now(),
        })
        return record, 0
    digest = hashlib.sha256()
    allowed = min(max_file_bytes, total_remaining)
    if allowed <= 0:
        record.update({"error_type": "DownloadLimitError", "error": "total byte budget exhausted", "finished_at": utc_now()})
        return record, 0
    written = 0
    declared_size = None
    try:
        with open_public_url(row["url"], timeout) as response:
            declared = response.headers.get("Content-Length")
            if declared:
                try:
                    declared_size = int(declared)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_file_bytes:
                    raise DownloadLimitError("declared file size exceeds per-file budget")
                if declared_size is not None and declared_size > total_remaining:
                    raise DownloadLimitError("declared file size exceeds remaining total budget")
            with part_path.open("xb") as handle:
                while True:
                    chunk = response.read(min(65536, allowed - written + 1))
                    if not chunk:
                        break
                    writable = chunk[: max(0, allowed - written)]
                    if writable:
                        handle.write(writable)
                        digest.update(writable)
                        written += len(writable)
                    if len(chunk) > len(writable):
                        raise DownloadLimitError("response exceeds byte budget")
                handle.flush()
                os.fsync(handle.fileno())
        actual_sha256 = digest.hexdigest()
        if declared_size is not None and written != declared_size:
            raise IntegrityError(f"HTTP Content-Length mismatch: expected {declared_size}, got {written}")
        if expected_size is not None and written != expected_size:
            raise IntegrityError(f"size mismatch: expected {expected_size}, got {written}")
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise IntegrityError("SHA-256 mismatch")
        os.replace(part_path, final_path)
        record.update({
            "status": "downloaded", "availability": "downloaded_pending_validation",
            "bytes": written, "sha256": actual_sha256, "finished_at": utc_now(),
            "integrity": "matched" if expected_sha256 or expected_size is not None else "not_provided",
            "final_url": response.geturl(), "http_status": getattr(response, "status", 200),
        })
    except Exception as exc:
        record.update({
            "status": "failed", "availability": "not_downloaded", "bytes": written,
            "sha256": digest.hexdigest() if written else "", "error_type": type(exc).__name__,
            "error": str(exc), "finished_at": utc_now(),
        })
        if isinstance(exc, IntegrityError):
            record["integrity"] = "mismatch"
    return record, written


def fetch_manifest(args: argparse.Namespace) -> Path:
    started = utc_now()
    run_dir = unique_run_dir(Path(args.output), "fetch")
    records: list[dict[str, Any]] = []
    total_written = 0
    try:
        rows = read_manifest(Path(args.manifest))
        for row in rows:
            record, written = _download_one(
                row, run_dir, args.timeout, args.max_file_bytes,
                max(0, args.max_total_bytes - total_written),
            )
            total_written += written
            records.append(record)
        atomic_write_text(run_dir / "fetch_results.tsv", render_tsv(records, (
            "source_id", "url", "role", "filename", "evidence_level", "status", "availability",
            "bytes", "sha256", "expected_size_bytes", "expected_sha256", "integrity",
            "path", "part_path", "http_status", "final_url",
            "error_type", "error", "started_at", "finished_at",
        )))
        failed = sum(row["status"] == "failed" for row in records)
        aggregate_status = "failed" if records and failed == len(records) else "completed_with_failures" if failed else "completed"
        run = {
            "command": "fetch", "status": aggregate_status, "started_at": started,
            "finished_at": utc_now(), "run_dir": str(run_dir.resolve()),
            "max_total_bytes": args.max_total_bytes, "max_file_bytes": args.max_file_bytes,
            "bytes_written": total_written, "rows": len(records),
            "downloaded": sum(row["status"] == "downloaded" for row in records),
            "failed": sum(row["status"] == "failed" for row in records),
            "results": records,
        }
        atomic_write_json(run_dir / "run.json", run)
        _write_latest(Path(args.output), run_dir, "fetch")
        return run_dir
    except BaseException as exc:
        atomic_write_json(run_dir / "run.json", {
            "command": "fetch", "status": "failed", "started_at": started,
            "finished_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def watch_seconds(value: str) -> int:
    parsed = int(value)
    if parsed < 300:
        raise argparse.ArgumentTypeError("--watch-seconds must be at least 300")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect registry metadata into a unique run")
    collect.add_argument("--registry", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--online", action="store_true", help="enable bounded public HTTP(S) retrieval")
    collect.add_argument("--max-pages", type=positive_int, default=40)
    collect.add_argument("--max-page-bytes", type=positive_int, default=2 * 1024 * 1024)
    collect.add_argument("--timeout", type=positive_int, default=10)
    collect.add_argument("--watch-seconds", type=watch_seconds)
    collect.add_argument("--inventory", help="optional dataset_inventory.tsv; records remain unvalidated")
    collect.add_argument("--page-cache", help="read <sha256(url)>.body and optional provenance files")
    collect.add_argument("--emit-requests", help="atomically write cache_key,url request TSV")

    fetch = subparsers.add_parser("fetch", help="download only explicitly manifested rows")
    fetch.add_argument("--manifest", required=True)
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--max-total-bytes", required=True, type=positive_int)
    fetch.add_argument("--max-file-bytes", required=True, type=positive_int)
    fetch.add_argument("--timeout", type=positive_int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            run_dir = fetch_manifest(args)
            print(run_dir)
            return 3 if json.loads((run_dir / "run.json").read_text())["failed"] else 0
        while True:
            run_dir = collect_once(args)
            print(run_dir, flush=True)
            if args.watch_seconds is None:
                return 3 if json.loads((run_dir / "run.json").read_text())["counts"]["errors"] else 0
            time.sleep(args.watch_seconds)
    except KeyboardInterrupt:
        return 130
    except (RegistryError, UnsafeURLError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
