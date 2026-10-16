import json
from pathlib import Path
import tarfile
import pandas as pd
root = Path('data/benchmarking_data/lungpf')
with tarfile.open(root/'HE_annotations.tar.gz') as archive:
    member = next(m for m in archive if m.name.endswith('cells_partitioned_by_annotation.csv'))
    ann = pd.read_csv(archive.extractfile(member))
print('PATHOLOGY', ann.shape, ann.columns.tolist(), flush=True)
print(ann.head(4).to_string(), flush=True)
ann.to_csv(root/'pathology.csv.gz', index=False)
meta_path = root/'cell_metadata.csv.gz'
if meta_path.exists():
    meta = pd.read_csv(meta_path)
    print('CELL_METADATA', meta.shape, meta.columns.tolist(), flush=True)
    print(meta.head(3).to_string(), flush=True)
    for col in meta.columns:
        if meta[col].dtype == 'object' and meta[col].nunique() < 100:
            print(col, meta[col].value_counts().to_dict(), flush=True)
import spatialformer
print('BASELINE_IMPORT_OK', spatialformer.__version__, flush=True)
