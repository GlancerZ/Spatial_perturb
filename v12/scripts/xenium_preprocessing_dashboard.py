"""Read-only, loopback-only status page for raw Xenium preprocessing.

This process only reads small manifest and state JSON files. The preprocessing
workers write them; this server never starts or controls a worker.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "outputs" / "xenium_raw_resolvi_20260927"
PAGE = Path(__file__).with_suffix(".html")
SAFE_SAMPLE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
STATUSES = {"pending", "running", "completed", "failed"}


def read_json(path: Path, max_bytes: int) -> Any:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"{path.name} exceeds {max_bytes} bytes")
    return json.loads(path.read_text(encoding="utf-8"))


def percentage(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return round(max(0.0, min(100.0, float(value))), 1)


def count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def display_text(value: Any, limit: int = 240) -> str:
    return str(value)[:limit] if value is not None else ""


def collect_status(data_dir: Path) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result: dict[str, Any] = {
        "checked_at": now,
        "source": str(data_dir / "manifest.json"),
        "counts": {status: 0 for status in (*sorted(STATUSES), "unknown")},
        "samples": [],
        "warnings": [],
        "error": None,
    }
    try:
        manifest = read_json(data_dir / "manifest.json", 2_000_000)
        entries = manifest.get("samples") if isinstance(manifest, dict) else manifest
        if not isinstance(entries, list):
            raise ValueError("manifest.json must contain a samples array")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        result["error"] = f"Cannot load manifest: {exc}"
        return result

    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            sample_id, name = entry, entry
        elif isinstance(entry, dict):
            sample_id = entry.get("sample_id", entry.get("id"))
            name = entry.get("name") or sample_id
        else:
            result["warnings"].append("A manifest entry is neither an object nor a string")
            continue
        if not isinstance(sample_id, str) or not SAFE_SAMPLE_ID.fullmatch(sample_id) or sample_id in (".", ".."):
            result["warnings"].append(f"Invalid sample id: {display_text(sample_id, 80)}")
            continue
        if sample_id in seen:
            result["warnings"].append(f"Duplicate sample id: {sample_id}")
            continue
        seen.add(sample_id)

        item: dict[str, Any] = {
            "sample_id": sample_id,
            "name": display_text(name, 160),
            "status": "pending",
            "progress": 0.0,
            "stage": "等待处理",
            "message": "",
            "updated_at": None,
            "cells": None,
            "genes": None,
            "processed_cells": None,
            "eligible_cells": None,
            "elapsed_seconds": None,
        }
        state_path = data_dir / "state" / f"{sample_id}.json"
        if state_path.exists():
            try:
                state = read_json(state_path, 256_000)
                if not isinstance(state, dict):
                    raise ValueError("state must be an object")
                reported_id = state.get("sample_id", sample_id)
                if reported_id != sample_id:
                    raise ValueError(f"sample_id mismatch: {display_text(reported_id, 80)}")
                status = state.get("status", "unknown")
                if status not in STATUSES:
                    raise ValueError(f"unrecognized status: {display_text(status, 80)}")
                item["status"] = status
                item["progress"] = percentage(state.get("progress"))
                if status == "completed":
                    item["progress"] = 100.0
                elif status == "pending" and item["progress"] is None:
                    item["progress"] = 0.0
                item["stage"] = display_text(state.get("stage"), 120)
                item["message"] = display_text(state.get("message"), 500)
                item["updated_at"] = display_text(state.get("updated_at"), 64) or None
                item["cells"] = count(state.get("cells"))
                item["genes"] = count(state.get("genes"))
                item["processed_cells"] = count(state.get("processed_cells"))
                item["eligible_cells"] = count(state.get("eligible_cells"))
                elapsed = state.get("elapsed_seconds")
                item["elapsed_seconds"] = round(float(elapsed), 1) if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and 0 <= elapsed < 1e12 and math.isfinite(elapsed) else None
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                item["status"] = "unknown"
                item["progress"] = None
                item["stage"] = "状态文件异常"
                item["message"] = display_text(exc, 500)
                result["warnings"].append(f"{sample_id}: {exc}")
        result["samples"].append(item)
        result["counts"][item["status"]] += 1
    result["total"] = len(result["samples"])
    return result


def make_handler(data_dir: Path):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path in ("/", "/index.html"):
                try:
                    body = PAGE.read_bytes()
                except OSError as exc:
                    self.send_error(500, str(exc))
                    return
                content_type = "text/html; charset=utf-8"
            elif path == "/api/status":
                body = json.dumps(collect_status(data_dir), ensure_ascii=False, allow_nan=False).encode("utf-8")
                content_type = "application/json; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

    return DashboardHandler


def write_snapshot(data_dir: Path, output: Path) -> None:
    """Embed current state so the page opens directly without an HTTP server."""
    payload = collect_status(data_dir)
    html = PAGE.read_text(encoding="utf-8")
    marker = "const embeddedSnapshot = null;"
    if html.count(marker) != 1:
        raise ValueError("Dashboard HTML snapshot marker is missing or ambiguous")
    javascript = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = html.replace(marker, f"const embeddedSnapshot = {javascript};")
    html = html.replace("<head>", '<head>\n<meta http-equiv="refresh" content="120">', 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(html)
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--snapshot", nargs="?", type=Path, const=DEFAULT_DATA_DIR / "dashboard_snapshot.html", help="write a self-contained HTML snapshot instead of serving")
    args = parser.parse_args()
    if args.snapshot is not None:
        write_snapshot(args.data_dir, args.snapshot)
        print(f"Snapshot: {args.snapshot}", flush=True)
        return
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.data_dir)) as server:
        print(f"Xenium status: http://127.0.0.1:{server.server_port}/", flush=True)
        print(f"Reading: {args.data_dir / 'manifest.json'}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
