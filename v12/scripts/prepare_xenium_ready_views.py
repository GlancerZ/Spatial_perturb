"""Build spatial neighborhoods while other complete-sample caches arrive."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pretrain.config import load_config
from spatial_pretrain.data.cache import load_slice_cache
from spatial_pretrain.data.dataset import prepare_slice_views


def prepare(directory):
    cfg = load_config(ROOT / "configs/xenium_compartment_full.yaml")
    path = Path(directory)
    view = prepare_slice_views(load_slice_cache(path), cfg, path.name)
    return path.name, len(view.train_epoch0_rows), len(view.val_rows)


def main():
    cfg = load_config(ROOT / "configs/xenium_compartment_full.yaml")
    pending = list(cfg.data.cache_dirs)
    active, completed = {}, 0
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as pool:
        while pending or active:
            for future in list(active):
                if future.done():
                    name, train, val = future.result()
                    completed += 1
                    print(f"[views-ready] {completed}/49 {name} train_subgraphs={train} val_subgraphs={val}", flush=True)
                    del active[future]
            for path in list(pending):
                if len(active) >= 2:
                    break
                if (Path(path) / "metadata.json").exists():
                    active[pool.submit(prepare, path)] = path
                    pending.remove(path)
            if pending or active:
                time.sleep(2)
    print("[views-complete] 49/49", flush=True)


if __name__ == "__main__":
    main()
