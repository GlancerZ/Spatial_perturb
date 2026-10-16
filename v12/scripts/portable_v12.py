"""Dispatch unchanged V12 programs from a verified compute environment."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
TASKS = {
    "prepare-cache": "prepare_subcellular_v2.py",
    "train": "train_subcellular_v12.py",
    "infer": "infer_v12_local.py",
    "prepare-hd": "prepare_visium_hd_mix_compartments_20261011.py",
    "prepare-hd-cache": "prepare_visium_hd_v12_cache.py",
    "train-hd-mix": "train_subcellular_v12_visium_mix.py",
    "eval-hd-mix": "evaluate_visium_hd_mix_crc.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--standalone-compute", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    from load_v12_portable import require_compute_execution
    require_compute_execution(args.standalone_compute)
    if args.standalone_compute and args.task not in ("prepare-cache", "train", "infer"):
        parser.error("The HD extension retains its original Slurm-only compute contract")
    forwarded = args.arguments
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if args.standalone_compute and args.task == "infer":
        forwarded = ["--standalone-compute", *forwarded]
    entry = ROOT / "scripts" / TASKS[args.task]
    os.chdir(ROOT)
    sys.argv = [str(entry), *forwarded]
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
