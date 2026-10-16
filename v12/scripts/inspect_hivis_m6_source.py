#!/usr/bin/env python3
"""Extract one CRC-verified Zenodo ZIP member and inspect its real HD schema."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import zlib


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8<<20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--inspect-only", action="store_true")
    a = p.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run inside a verified Slurm compute step")
    import anndata as ad
    import numpy as np
    from scipy import sparse
    source = a.source_dir/"human_liver_M6_HiVis.h5ad.deflate"
    output = a.source_dir/"human_liver_M6_HiVis.h5ad"
    if source.stat().st_size != 283222663:
        raise ValueError("Compressed member size mismatch")
    if a.inspect_only:
        if output.stat().st_size != 701258160:
            raise ValueError("Previously extracted member size mismatch")
        crc = 0
        with output.open("rb") as f:
            for chunk in iter(lambda:f.read(8<<20), b""):
                crc = zlib.crc32(chunk, crc)
        if f"{crc & 0xffffffff:08x}" != "68f29c92":
            raise ValueError("Previously extracted member CRC mismatch")
    elif output.exists():
        raise FileExistsError("Refusing to overwrite extracted member")
    temp = output.with_suffix(".partial.h5ad")
    crc = size = 0
    dec = zlib.decompressobj(-15)
    if not a.inspect_only:
        with source.open("rb") as f, temp.open("xb") as out:
            for chunk in iter(lambda:f.read(4<<20), b""):
                data = dec.decompress(chunk)
                size += len(data)
                crc = zlib.crc32(data,crc)
                out.write(data)
            tail = dec.flush()
            out.write(tail)
            size += len(tail)
            crc = zlib.crc32(tail,crc)
        if not dec.eof or dec.unused_data or size != 701258160 or f"{crc & 0xffffffff:08x}" != "68f29c92":
            raise ValueError(f"Member integrity failed: size={size} crc={crc:08x}")
        temp.replace(output)
    size = output.stat().st_size
    data = ad.read_h5ad(output,backed="r")
    def summarize_column(series):
        result={"dtype":str(series.dtype),"nulls":int(series.isna().sum()),"unique":int(series.nunique()),"head":series.head(5).astype(str).tolist()}
        if series.nunique() <= 100:
            result["value_counts"]={str(k):int(v) for k,v in series.value_counts(dropna=False).items()}
        if series.dtype.kind in "iuf":
            numeric = series.to_numpy(dtype=float, na_value=np.nan)
            finite = numeric[np.isfinite(numeric)]
            result["nonfinite_values"] = int((~np.isfinite(numeric)).sum())
            result["finite_min"] = float(finite.min()) if len(finite) else None
            result["finite_max"] = float(finite.max()) if len(finite) else None
        return result
    inspected = {name:summarize_column(data.obs[name]) for name in data.obs.columns}
    sample = data.X[:min(data.n_obs,4096),:]
    values = sample.data if sparse.issparse(sample) else np.asarray(sample).ravel()
    values = values[values != 0]
    count_audit={"sampled_rows":min(data.n_obs,4096),"nonzero_values":len(values),"dtype":str(sample.dtype)}
    if len(values):
        count_audit.update(min=float(values.min()),max=float(values.max()),finite=bool(np.isfinite(values).all()),
                           nonnegative=bool((values>=0).all()),integer_fraction=float(np.isclose(values,np.rint(values),rtol=0,atol=1e-6).mean()))
    result={"status":"EXTRACTED_AND_SCHEMA_INSPECTED","platform":"Visium HD", "sample":"human liver M6",
            "source_record":"https://zenodo.org/records/21821264", "member":"VisiumHD_updated/M6_M2/M6 only/2um/human_liver_M6_HiVis.h5ad",
            "compressed_sha256":sha256(source),"extracted_sha256":sha256(output),"size_bytes":size,"zip_member_crc32":"68f29c92",
            "shape":list(data.shape),"obs_columns":inspected,"obs_index_head":data.obs_names[:5].tolist(),
            "var_columns":list(data.var.columns),"var_index_head":data.var_names[:5].tolist(),
            "layers":list(data.layers.keys()),"raw_present":data.raw is not None,
            "obsm":{key:{"shape":list(data.obsm[key].shape),"head":np.asarray(data.obsm[key])[:3].tolist()} for key in data.obsm.keys()},
            "uns_keys":list(data.uns.keys()),"sampled_X_count_audit":count_audit,
            "slurm_job_id":os.environ["SLURM_JOB_ID"],"hostname":host,
            "note":"Schema inspection only; sampled integer values do not establish full raw-count provenance."}
    (a.source_dir/"schema_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)
    data.file.close()


if __name__ == "__main__":
    main()
