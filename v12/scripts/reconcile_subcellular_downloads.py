"""Independently reconcile download receipts and local bytes on a compute node."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, action='append', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
records = []
for directory in args.directory:
    with (directory / 'downloads.tsv').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    for row in rows:
        path = directory / row['source_id'] / row['filename']
        result = {'path': str(path.resolve()), 'url': row['url'], 'source_id': row['source_id']}
        try:
            if row['status'] != 'downloaded_unvalidated':
                raise ValueError('Download receipt does not indicate a completed file')
            if path.stat().st_size != int(row['bytes']):
                raise ValueError('File size differs from download receipt')
            digest = hashlib.sha256()
            with path.open('rb') as raw:
                for block in iter(lambda: raw.read(8 * 1024**2), b''):
                    digest.update(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('SHA256 differs from download receipt')
            result.update(status='verified_download_bytes', bytes=path.stat().st_size, sha256=digest.hexdigest())
        except Exception as exc:
            result.update(status='failed', error=str(exc))
        records.append(result)
        print(json.dumps(result), flush=True)
summary = {'files': len(records), 'verified': sum(r['status'] == 'verified_download_bytes' for r in records),
           'bytes': sum(r.get('bytes', 0) for r in records), 'records': records,
           'validation_scope': 'Downloaded bytes only; not biological compartment readiness'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summary, indent=2))
raise SystemExit(int(summary['verified'] != summary['files']))
