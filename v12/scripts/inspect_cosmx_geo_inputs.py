"""Read actual CosMx raw tables on a compute node before conversion."""
import json
from pathlib import Path
import pandas as pd

root = Path('data/subcellular_data_agent/raw_cosmx_geo_20260922/cosmx_lung_tma2_gsm9046089')
output = Path('outputs/cosmx_geo_raw_inspection_20260922.json')
if output.exists():
    raise FileExistsError(output)
result = {}
for path in sorted(root.glob('*.csv.gz')):
    preview = pd.read_csv(path, nrows=5)
    item = {'columns': list(preview.columns), 'preview': preview.fillna('').to_dict(orient='records')}
    if path.name == 'tx_file.csv.gz':
        table = pd.read_csv(path)
        item['rows'] = len(table)
        for col in ['target', 'gene', 'CellComp', 'fov']:
            if col in table:
                item[col] = {'unique': int(table[col].nunique()), 'counts': table[col].value_counts().head(12).to_dict()}
    elif path.name == 'metadata_file.csv.gz':
        table = pd.read_csv(path)
        item['rows'] = len(table)
        item['fovs'] = sorted(map(int, table['fov'].unique())) if 'fov' in table else []
    result[path.name] = item
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2, default=str))
print(json.dumps({key: {k: v for k,v in item.items() if k != 'preview'} for key,item in result.items()}, default=str), flush=True)
