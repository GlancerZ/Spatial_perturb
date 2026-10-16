"""Apply one frozen LungPF linear probe to matching spatial-cell embeddings.

Run on a Slurm compute node. This produces predictions, not an accuracy score.
The LungPF label vocabulary is lung-specific; a non-lung sample requires its
own reference probe and biological validation before labels can be interpreted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import expit, softmax

EXPECTED_MODEL_IDS = {
    "geneformer": "Geneformer-V2-316M",
    "scgpt": "ordinary_scgpt_whole_human",
    "scfoundation": "scFoundation",
    "state": "ArcInstitute_STATE_SE_600M",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_embeddings(path: Path) -> tuple[list[str], np.ndarray, dict]:
    if path.is_dir():
        ids_path = path / "cell_ids.txt"
        matrix_path = path / "embeddings.npy"
        manifest_path = path / "provenance.json"
        ids = ids_path.read_text(encoding="utf-8").splitlines()
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    elif path.suffix == ".npz":
        manifest_path = Path(str(path) + ".manifest.json")
        with np.load(path, allow_pickle=False) as packed:
            ids = [value.decode("utf-8") if isinstance(value, bytes) else str(value)
                   for value in packed["cell_ids"]]
            matrix = packed["embedding"]
        matrix_path = path
    else:
        raise ValueError("Embedding input must be a sample directory or .npz file")
    provenance = json.loads(manifest_path.read_text())
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Embedding cell IDs are empty or duplicated")
    if matrix.ndim != 2 or matrix.shape[0] != len(ids) or matrix.shape[1] < 2:
        raise ValueError("Embedding matrix has invalid dimensions")
    if not np.issubdtype(matrix.dtype, np.floating):
        raise ValueError("Embedding matrix must be floating point")
    return ids, matrix, {
        "provenance": provenance, "matrix_sha256": sha256(matrix_path),
        "manifest_sha256": sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["geneformer", "scgpt", "scfoundation", "state"],
                        required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--input-h5ad", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .parquet path")
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args()
    if args.output.suffix != ".parquet" or args.chunk_rows < 1:
        parser.error("Output must be .parquet and chunk-rows positive")
    result_manifest = Path(str(args.output) + ".manifest.json")
    if args.output.exists() or result_manifest.exists():
        raise FileExistsError("Refusing to overwrite existing spatial predictions")
    report = json.loads(args.probe_report.read_text())
    if report.get("model") != args.model or report.get("task") != "cell_type":
        raise ValueError("Probe report model/task differs from requested cell-type prediction")
    ids, matrix, source = read_embeddings(args.embeddings)
    provenance = source["provenance"]
    if provenance.get("model") != EXPECTED_MODEL_IDS[args.model]:
        raise ValueError("Embedding provenance identifies a different model")
    declared_input = provenance.get("input_h5ad", provenance.get("input"))
    if not isinstance(declared_input, str) or Path(declared_input).resolve() != args.input_h5ad.resolve():
        raise ValueError("Embedding provenance points to a different H5AD")
    data = ad.read_h5ad(args.input_h5ad, backed="r")
    try:
        if list(map(str, data.obs_names)) != ids:
            raise ValueError("Embedding row order differs from input H5AD cell IDs")
        if "spatial" not in data.obsm:
            raise KeyError("Input H5AD has no obsm['spatial'] coordinates")
        spatial = np.asarray(data.obsm["spatial"])
        if spatial.shape != (len(ids), 2) or not np.isfinite(spatial).all():
            raise ValueError("Spatial coordinates must be finite two-dimensional X/Y")
        with np.load(args.probe, allow_pickle=False) as saved:
            coefficients = saved["coef"]
            intercept = saved["intercept"]
            classes = saved["classes"].astype(str)
            mean = saved["scaler_mean"]
            scale = saved["scaler_scale"]
        if (matrix.shape[1] != len(mean) or len(mean) != len(scale)
                or coefficients.shape[1] != matrix.shape[1]
                or not np.isfinite(coefficients).all()
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or (scale <= 0).any()):
            raise ValueError("Probe, scaler and embedding dimensions or values mismatch")
        if coefficients.shape[0] not in (len(classes), 1):
            raise ValueError("Probe class/weight dimensions mismatch")
        predicted = np.empty(len(ids), dtype=object)
        confidence = np.empty(len(ids), dtype=np.float32)
        margin = np.empty(len(ids), dtype=np.float32)
        for start in range(0, len(ids), args.chunk_rows):
            stop = min(start + args.chunk_rows, len(ids))
            features = np.asarray(matrix[start:stop], dtype=np.float64)
            if not np.isfinite(features).all():
                raise ValueError(f"Nonfinite embedding at rows {start}:{stop}")
            logits = ((features - mean) / scale) @ coefficients.T + intercept
            if len(classes) == 2 and coefficients.shape[0] == 1:
                positive = expit(logits[:, 0])
                probabilities = np.column_stack((1.0 - positive, positive))
            else:
                probabilities = softmax(logits, axis=1)
            order = np.argsort(probabilities, axis=1)
            winners = order[:, -1]
            predicted[start:stop] = classes[winners]
            confidence[start:stop] = probabilities[np.arange(stop - start), winners]
            margin[start:stop] = confidence[start:stop] - probabilities[
                np.arange(stop - start), order[:, -2]]
        table = pd.DataFrame({
            "sample": args.sample, "cell_id": ids,
            "x": spatial[:, 0], "y": spatial[:, 1],
            "predicted_cell_type": predicted,
            "max_probability": confidence, "top_two_margin": margin,
        })
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="spatial-pred-", dir=args.output.parent) as stage:
            staged = Path(stage) / "predictions.parquet"
            table.to_parquet(staged, index=False)
            if len(pd.read_parquet(staged, columns=["cell_id"])) != len(ids):
                raise ValueError("Stored prediction row count differs from embedding cells")
            manifest = {
                "model": args.model, "sample": args.sample,
                "cell_count": len(ids), "class_count": len(classes),
                "input_h5ad": str(args.input_h5ad.resolve()),
                "probe": str(args.probe.resolve()),
                "probe_report_sha256": sha256(args.probe_report),
                "probe_sha256": sha256(args.probe),
                "embedding_sha256": source["matrix_sha256"],
                "embedding_manifest_sha256": source["manifest_sha256"],
                "input_source": provenance.get("input_source",
                                               provenance.get("input_count_semantics",
                                                              provenance.get("counts_layer"))),
                "predictions_sha256": sha256(staged),
                "accuracy_available": False,
                "warning": "LungPF cell-type vocabulary is lung-specific; use only with compatible tissue/reference.",
            }
            staged_manifest = Path(stage) / "manifest.json"
            staged_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            os.link(staged, args.output)
            os.link(staged_manifest, result_manifest)
        print(f"{args.model}/{args.sample}: {len(ids)} spatial predictions", flush=True)
    finally:
        data.file.close()


if __name__ == "__main__":
    main()
