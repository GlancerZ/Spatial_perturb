"""Inspect published Bento H5AD metadata and checksums; compute node only."""
import hashlib
import json
from pathlib import Path
import anndata as ad

root = Path('data/subcellular_data_agent/raw_bento_20260922')
expected = {'seqfish.h5ad': '40b7ee8bfc4eb7ba6070b25ba3df0155',
            'u2os_merfish.h5ad': '9158398174529dd68bae2f4e2195116a'}
output = {}
for name, checksum in expected.items():
    path = root / name
    digest, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
            sha.update(block)
    if digest.hexdigest() != checksum:
        raise ValueError(f'Published MD5 mismatch: {path}')
    data = ad.read_h5ad(path)
    record = {'file': str(path.resolve()), 'bytes': path.stat().st_size,
              'md5': digest.hexdigest(), 'sha256': sha.hexdigest(),
              'shape': data.shape, 'obs_columns': list(data.obs.columns),
              'obs_examples': data.obs.head(2).to_dict(), 'var_columns': list(data.var.columns),
              'gene_names': data.var_names.tolist(), 'uns_keys': list(data.uns),
              'obsm_keys': list(data.obsm), 'layer_keys': list(data.layers)}
    points = data.uns.get('points')
    record['points_type'] = str(type(points))
    if hasattr(points, 'columns'):
        record['points_shape'] = points.shape
        record['points_columns'] = list(points.columns)
        record['points_head'] = points.head(5).to_dict(orient='records')
        record['points_dtypes'] = points.dtypes.astype(str).to_dict()
        for key in ('x', 'y', 'z'):
            if key in points:
                record[f'points_{key}_range'] = [points[key].min(), points[key].max()]
    for key in ('batch', 'cell_shape', 'nucleus_shape'):
        if key in data.obs:
            record[f'obs_{key}_missing'] = int(data.obs[key].isna().sum())
            if key == 'batch':
                record['batch_sizes'] = data.obs[key].astype(str).value_counts().to_dict()
    output[name] = record
    print(json.dumps(record, default=str), flush=True)
destination = Path('outputs/bento_public_input_inspection_20260922.json')
if destination.exists():
    raise FileExistsError(destination)
destination.write_text(json.dumps(output, default=str, indent=2))
