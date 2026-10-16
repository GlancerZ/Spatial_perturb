"""Run on a compute node; creates immutable per-slide training packages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pretrain.data.unified_compartments import convert_manifest, example_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Explicit, reviewed JSON source manifest")
    parser.add_argument("--output-dir", type=Path, help="New directory; existing outputs are never overwritten")
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--example", choices=["xenium", "cosmx"], help="Print a manifest template; verification fields must be filled")
    args = parser.parse_args()
    if args.example:
        print(json.dumps(example_manifest(args.example), indent=2))
        return
    if args.manifest is None or args.output_dir is None:
        parser.error("--manifest and --output-dir are required")
    stats = convert_manifest(args.manifest, args.output_dir, chunk_rows=args.chunk_rows)
    print(json.dumps({"output": str(args.output_dir.resolve()), "qc": stats}, indent=2))


if __name__ == "__main__":
    main()
