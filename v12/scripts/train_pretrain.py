from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_pretrain.config import load_config
from spatial_pretrain.training.runner import run_pretraining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spatial pretraining")
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--run-update-limit", type=int, default=None,
                        help="Pause after this many optimizer updates, saving a resume checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.run_update_limit is not None:
        config.training.run_update_limit = args.run_update_limit
    run_pretraining(config)


if __name__ == "__main__":
    main()
