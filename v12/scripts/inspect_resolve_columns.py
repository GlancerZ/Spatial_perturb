"""Read only a few text records per RNA/assignment member on compute."""
import gzip
import io
import json
import tarfile
from pathlib import Path

path = Path('data/subcellular_data_agent/raw_public_20260922/resolve_mc_ad_cortex_gse325911/GSE325911_RAW.tar')
result = []
with tarfile.open(path) as archive:
    for member in archive:
        if not (member.name.endswith('_results.txt.gz') or member.name.endswith('_results_assigned.txt.gz')):
            continue
        with archive.extractfile(member) as handle, gzip.GzipFile(fileobj=handle) as raw, io.TextIOWrapper(raw) as text:
            lines = [next(text, '').strip() for _ in range(5)]
        result.append({'name': member.name, 'lines': lines})
        if len(result) >= 4:
            break
destination = Path('outputs/resolve_columns_20260922.json')
destination.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
