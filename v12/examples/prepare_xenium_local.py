"""Build one Xenium sample's reference, true N/C counts and complete geometry.

ZIP input must contain the matching native cells, matrix and transcripts.
Execute on a compute node; never overwrite an existing sample output.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--standalone-compute", action="store_true")
    args = parser.parse_args()
    from load_v12_portable import require_compute_execution, file_sha256
    execution = require_compute_execution(args.standalone_compute)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.sample_id):
        parser.error("Use a single safe sample ID without directory separators")
    if args.archive.suffix.lower() != ".zip" or not args.archive.is_file():
        parser.error("This example requires a real native Xenium outs ZIP")
    import anndata as ad
    import numpy as np
    from standardize_spatial_h5ad import convert_xenium_zip_file
    from spatial_pretrain.data.xenium_compartments import convert_xenium_compartments
    from prepare_unified_training_cache import build_geometry
    base = args.output_dir.resolve()
    reference = base / "reference" / (args.sample_id + ".h5ad")
    counts = base / "h5ad" / (args.sample_id + ".h5ad")
    geometry = base / "geometry" / (args.sample_id + "_k255")
    if any(path.exists() for path in (reference, counts, geometry)):
        raise FileExistsError("Sample output exists; diagnose it or use a new output directory")
    convert_xenium_zip_file(args.archive, reference, dataset_id=args.sample_id,
        source="10x_xenium", sample_id=args.sample_id, batch_id=args.sample_id, overwrite=False)
    # The original converter checks every N+C entry against this native matrix.
    convert_xenium_compartments(args.archive, reference, counts)
    cells = ad.read_h5ad(counts, backed="r")
    try:
        coords = np.asarray(cells.obsm["spatial"], dtype=np.float32)
        if coords.shape != (cells.n_obs, 2) or not np.isfinite(coords).all():
            raise ValueError("Native Xenium cell centroids must be finite micrometer coordinates")
        neighbors = build_geometry(coords, np.full(len(coords), args.sample_id), k=255)
        geometry.mkdir(parents=True, exist_ok=False)
        np.save(geometry / "coords.npy", coords)
        np.save(geometry / "knn_indices.npy", neighbors)
        (geometry / "provenance.json").write_text(json.dumps(dict(
            sample_id=args.sample_id, archive=str(args.archive.resolve()),
            archive_sha256=file_sha256(args.archive), count_h5ad=str(counts),
            cells=cells.n_obs, graph="all source cells before selecting training centers",
            coordinate_unit="micrometer", execution=execution), indent=2) + "\n")
    finally:
        cells.file.close()
    print(json.dumps(dict(reference=str(reference), counts=str(counts), geometry=str(geometry))), flush=True)


if __name__ == "__main__":
    main()
