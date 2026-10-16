from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pretrain.data.xenium_compartments import convert_xenium_compartments


def main():
    parser = argparse.ArgumentParser(description="Aggregate Xenium nuclear/cytoplasmic counts on a compute node")
    parser.add_argument("--manifest", default="data/pretrain_data/manifests/xenium49_compartment_inputs.tsv")
    parser.add_argument("--sample", action="append", help="Sample names; default is all manifest samples")
    parser.add_argument("--output-dir", default="data/pretrain_data/10x_xenium/compartments/h5ad")
    parser.add_argument("--max-cells", type=int, default=0, help="Use a separate output directory for smoke subsets")
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--scratch-dir")
    args = parser.parse_args()
    rows = list(csv.DictReader((ROOT / args.manifest).open(), delimiter="\t"))
    if args.sample:
        missing = set(args.sample) - {r["sample"] for r in rows}
        if missing:
            raise ValueError(f"Samples not in manifest: {missing}")
        rows = [r for r in rows if r["sample"] in args.sample]
    for row in rows:
        output = ROOT / args.output_dir / (row["sample"] + ".h5ad")
        # Existing outputs are never silently accepted with different options.
        convert_xenium_compartments(ROOT / row["archive"], ROOT / row["existing_h5ad"], output,
                                    max_cells=args.max_cells, chunk_rows=args.chunk_rows,
                                    scratch_dir=args.scratch_dir)


if __name__ == "__main__":
    main()
