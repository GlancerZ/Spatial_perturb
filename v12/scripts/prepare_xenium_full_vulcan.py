"""Prepare every manifest sample as its transfer completes, on compute nodes."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import csv
import json
import multiprocessing as mp
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pretrain.config import load_config
from spatial_pretrain.data.xenium_compartments import convert_xenium_compartments
from spatial_pretrain.data.compartment_cache import build_compartment_cache


def prepare(row):
    sample = row["sample"]
    directory = ROOT / "data/pretrain_data/10x_xenium/compartments/h5ad"
    output = directory / (sample + ".h5ad")
    stamp = directory / (sample + ".prepared.json")
    source, reference = ROOT / row["archive"], ROOT / row["existing_h5ad"]
    signature = {"archive": str(source), "archive_size": source.stat().st_size,
                 "reference": str(reference), "reference_size": reference.stat().st_size,
                 "max_cells": 0, "qv_min": 20}
    with (ROOT / "logs" / ("prepare_" + sample + ".log")).open("a", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            if output.exists():
                if not stamp.exists() or json.loads(stamp.read_text()) != signature:
                    raise ValueError(f"Existing output has no matching preparation stamp: {output}")
            else:
                convert_xenium_compartments(source, reference, output)
                stamp.write_text(json.dumps(signature, indent=2))
            config = load_config(ROOT / "configs/xenium_compartment_full.yaml")
            cache = ROOT / "cache/xenium_compartments" / (sample + "_k255")
            build_compartment_cache(output, cache, config)
    return sample


def main():
    rows = list(csv.DictReader((ROOT / "data/pretrain_data/manifests/xenium49_compartment_vulcan_inputs.tsv").open(), delimiter="\t"))
    if len(rows) != 49 or len({r['sample'] for r in rows}) != 49:
        raise ValueError("Expected the complete 49-sample manifest")
    (ROOT / "logs").mkdir(exist_ok=True)
    pending = {r["sample"]: r for r in rows}
    active, done = {}, []
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        while pending or active:
            for future in list(active):
                if future.done():
                    sample = future.result()
                    done.append(sample)
                    del active[future]
                    print(f"[prepared] {len(done)}/49 {sample}", flush=True)
                    (ROOT / "logs/xenium49_preparation_status.json").write_text(json.dumps({"completed": done}, indent=2))
            for sample, row in list(pending.items()):
                if len(active) >= 2:
                    break
                ready = ROOT / "data/pretrain_data/10x_xenium/compartments/inputs/.ready" / (sample + ".done")
                if ready.exists():
                    active[pool.submit(prepare, row)] = sample
                    del pending[sample]
                    print(f"[preparing] {sample}", flush=True)
            if pending or active:
                time.sleep(2)
    print("[preparation-complete] 49/49; full training may start", flush=True)


if __name__ == "__main__":
    main()
