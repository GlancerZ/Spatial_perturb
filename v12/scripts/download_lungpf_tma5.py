"""Download corrected public transcript inputs; network transfer only."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

root = Path('data/benchmarking_data/lungpf')
root.mkdir(parents=True, exist_ok=True)
listing = Path('resources/spatialformer/GSE250346_filelist.txt').read_text()
names = sorted(set(re.findall(r'GSM850\d+_[A-Za-z0-9]+_transcripts\.csv\.gz', listing)))
# A second section of a donor represented in the original TMA1-4 cohort is excluded.
names = [n for n in names if 'TILD117MA2' not in n]
assert len(names) == 16
manifest = []
for name in names:
    gsm, sample, _ = name.split('_', 2)
    url = f'https://ftp.ncbi.nlm.nih.gov/geo/samples/{gsm[:-3]}nnn/{gsm}/suppl/{name}'
    manifest.append(dict(sample=sample, accession=gsm, url=url, path=str(root/name)))
(root/'download_manifest.json').write_text(json.dumps(manifest, indent=2))

def fetch(row):
    dest = Path(row['path'])
    if not dest.exists():
        tmp = dest.with_suffix(dest.suffix+'.partial')
        subprocess.run(['curl', '-sSL', '--fail', '--retry', '3', '-o', str(tmp), row['url']], check=True)
        tmp.replace(dest)
    print('[downloaded]', row['sample'], dest.stat().st_size, flush=True)

with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(fetch, manifest))
