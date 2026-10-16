from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = ROOT / "tools" / "download-status-dashboard"
STATE_DIR = TOOL_DIR / "state"
JSON_PATH = STATE_DIR / "processing_status_snapshot.json"
JS_PATH = STATE_DIR / "processing_status_snapshot.js"


def load_server_module():
    module_path = TOOL_DIR / "server.py"
    spec = importlib.util.spec_from_file_location("processing_dashboard_server", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    module = load_server_module()
    payload = module.collect_statuses()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    JS_PATH.write_text(
        "window.PROCESSING_STATUS_SNAPSHOT = " + json.dumps(payload, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    print(f"[processing-snapshot] wrote {JSON_PATH}")
    print(f"[processing-snapshot] wrote {JS_PATH}")


if __name__ == "__main__":
    main()
