import gzip
import io
import json
from pathlib import Path
import tarfile

root = Path('data/subcellular_data_agent/raw_public_20260922/resolve_mc_ad_cortex_gse325911')
report = {}
with tarfile.open(root / 'GSE325911_RAW.tar') as archive:
    for suffix in ('results.txt.gz', 'results_assigned.txt.gz'):
        name = 'GSM9616841_APP23het_early_M1_roi1_' + suffix
        with archive.extractfile(name) as raw, gzip.GzipFile(fileobj=raw) as stream, io.TextIOWrapper(stream) as text:
            rows = []
            for index, line in enumerate(text):
                if 2560 <= index <= 2575:
                    rows.append({'row': index, 'values': line.strip().split('\t')})
                if index > 2575:
                    break
        report[suffix] = rows
Path('outputs/resolve_mismatch_20260922.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
