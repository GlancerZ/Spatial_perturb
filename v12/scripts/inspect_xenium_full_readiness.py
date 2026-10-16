import csv
import json
from pathlib import Path
import h5py

root = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((root / 'data/pretrain_data/manifests/xenium49_compartment_vulcan_inputs.tsv').open(), delimiter='\t'))
summary = []
for row in rows:
    with h5py.File(root / row['existing_h5ad'], 'r') as f:
        obs = f['obs']
        key = obs.attrs.get('_index', '_index')
        cells = len(obs[key])
    metadata = root / 'cache/xenium_compartments' / (row['sample']+'_k255') / 'metadata.json'
    summary.append(dict(sample=row['sample'], cells=cells, cache_ready=metadata.exists()))
result = dict(samples=len(summary), cells=sum(x['cells'] for x in summary),
              caches_ready=sum(x['cache_ready'] for x in summary),
              largest=sorted(summary, key=lambda x: x['cells'], reverse=True)[:5])
tables = list(Path('/tmp').glob('xenium-parquet-*/transcripts.parquet'))
if tables:
    import pyarrow.parquet as pq
    table = max(tables, key=lambda p: p.stat().st_size)
    try:
        result['largest_temporary_table'] = dict(bytes=table.stat().st_size,
                                               rows=pq.ParquetFile(table).metadata.num_rows)
    except Exception as exc:
        result['temporary_table_metadata'] = str(exc)
(root / 'logs/xenium49_readiness.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
