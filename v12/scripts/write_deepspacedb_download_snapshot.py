from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "tools" / "download-status-dashboard" / "state"
JSON_PATH = STATE_DIR / "deepspacedb_download_snapshot.json"
JS_PATH = STATE_DIR / "deepspacedb_download_snapshot.js"


TASKS = [
    {
        "id": "deepspacedb-index",
        "name": "DeepSpaceDB Human Xenium Index",
        "expected_file": "data/pretrain_data/deepspacedb/manifests/deepspacedb_human_xenium_inventory.tsv",
        "root_dir": "data/pretrain_data/deepspacedb/manifests",
        "log_path": "",
    },
    {
        "id": "deepspacedb-10x",
        "name": "DeepSpaceDB 10x Xenium Minimal",
        "expected_file": "data/pretrain_data/deepspacedb/manifests/deepspacedb_10x_minimal_urls.tsv",
        "root_dir": "data/pretrain_data/deepspacedb/raw_10x_xenium",
        "log_path": "logs/deepspacedb_10x_xenium_20260415.log",
    },
    {
        "id": "deepspacedb-geo",
        "name": "DeepSpaceDB GEO Xenium Minimal",
        "expected_file": "data/pretrain_data/deepspacedb/manifests/deepspacedb_geo_minimal_urls.tsv",
        "root_dir": "data/pretrain_data/geo/raw_xenium_deepspacedb",
        "log_path": "logs/deepspacedb_geo_xenium_20260415.log",
    },
    {
        "id": "deepspacedb-dbkero",
        "name": "DeepSpaceDB DBKERO Xenium Minimal",
        "expected_file": "data/pretrain_data/deepspacedb/manifests/deepspacedb_dbkero_minimal_urls.tsv",
        "root_dir": "data/pretrain_data/dbkero",
        "log_path": "logs/deepspacedb_dbkero_xenium_20260415.log",
    },
]


def resolve(path_str: str) -> Path:
    return ROOT / path_str


def human_bytes(num: int) -> str:
    if not num:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    if value >= 100 or idx == 0:
        return f"{value:.0f} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def tail_lines(text: str, limit: int) -> list[str]:
    return [line.rstrip() for line in text.splitlines() if line.strip()][-limit:]


def count_manifest_entries(path: Path) -> int:
    if not path.exists():
        return 0
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not lines:
        return 0
    if lines[0].count("\t") >= 1:
        return max(0, len(lines) - 1)
    return len(lines)


def count_completed_urls(log_text: str) -> int:
    urls = {line.split("URL:", 1)[1].split()[0] for line in log_text.splitlines() if "URL:" in line}
    return len(urls)


def scan_dir(root: Path) -> tuple[int, int, int]:
    if not root.exists():
        return 0, 0, 0
    total_bytes = 0
    file_count = 0
    latest = 0
    for path in root.rglob("*"):
        if path.is_file():
            stat = path.stat()
            total_bytes += stat.st_size
            file_count += 1
            latest = max(latest, int(stat.st_mtime))
    return total_bytes, file_count, latest


def collect() -> dict:
    now = int(time.time())
    statuses = []
    active = complete = stalled = not_started = 0
    total_bytes = 0
    for task in TASKS:
        manifest_path = resolve(task["expected_file"])
        expected = count_manifest_entries(manifest_path)
        root_dir = resolve(task["root_dir"])
        size_bytes, file_count, latest_output = scan_dir(root_dir)
        log_text = ""
        log_mtime = 0
        if task["log_path"]:
            log_path = resolve(task["log_path"])
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8", errors="ignore")
                log_mtime = int(log_path.stat().st_mtime)
        completed_urls = count_completed_urls(log_text)
        latest = max(latest_output, log_mtime)
        status = "not-started"
        if expected == 0 and manifest_path.exists():
            status = "complete"
        elif expected and completed_urls >= expected and ("FINISHED --" in log_text or "download finished." in log_text.lower()):
            status = "complete"
        elif latest and now - latest <= 20 * 60:
            status = "active"
        elif file_count > 0 or log_text.strip():
            status = "stalled"
        if status == "active":
            active += 1
        elif status == "complete":
            complete += 1
        elif status == "stalled":
            stalled += 1
        else:
            not_started += 1
        total_bytes += size_bytes
        statuses.append(
            {
                "id": task["id"],
                "name": task["name"],
                "status": status,
                "expected_count": expected,
                "completed_count": completed_urls,
                "output_total_bytes": size_bytes,
                "output_total_bytes_human": human_bytes(size_bytes),
                "output_file_count": file_count,
                "latest_activity": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(latest)) if latest else "N/A",
                "recent_log_lines": tail_lines(log_text, 4),
                "root_dir": task["root_dir"],
                "manifest_path": task["expected_file"],
                "log_path": task["log_path"],
            }
        )
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "repo_root": str(ROOT),
        "summary": {
            "tasks": len(TASKS),
            "active": active,
            "complete": complete,
            "stalled": stalled,
            "not_started": not_started,
            "downloaded_bytes": total_bytes,
            "downloaded_bytes_human": human_bytes(total_bytes),
        },
        "statuses": statuses,
    }


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = collect()
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    JS_PATH.write_text("window.DEEPSPACEDB_DOWNLOAD_SNAPSHOT = " + json.dumps(payload, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"[deepspacedb-snapshot] wrote {JSON_PATH}")
    print(f"[deepspacedb-snapshot] wrote {JS_PATH}")


if __name__ == "__main__":
    main()
