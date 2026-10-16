import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

path = Path('data/subcellular_data_agent/raw_bento_20260922/u2os_merfish_40027485.tar.gz')
digest = hashlib.md5()
with path.open('rb') as raw:
    for block in iter(lambda: raw.read(8 * 1024**2), b''):
        digest.update(block)
if digest.hexdigest() != '3c329fc33176c1a3789c4e3f1ab94258':
    raise ValueError('Raw author archive MD5 mismatch')
report = {'path': str(path.resolve()), 'md5': digest.hexdigest(), 'members': [], 'csv_examples': []}
with tarfile.open(path, mode='r|gz') as archive:
    for member in archive:
        report['members'].append({'name': member.name, 'bytes': member.size})
        if member.name.endswith('barcodes.zip'):
            if not member.isfile() or member.size > 2 * 1024**3:
                raise ValueError('Invalid nested barcode archive')
            with archive.extractfile(member) as raw:
                payload = raw.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as nested:
                report['barcode_members'] = [{'name': n.filename, 'bytes': n.file_size} for n in nested.infolist()]
                for entry in nested.infolist():
                    if entry.filename.endswith('.csv') and not entry.filename.startswith('__MACOSX'):
                        with nested.open(entry) as raw, io.TextIOWrapper(raw) as text:
                            report['csv_examples'].append({'name': entry.filename,
                                'first_lines': [next(text, '').strip() for _ in range(4)]})
                        if len(report['csv_examples']) >= 2:
                            break
output = Path('outputs/bento_merfish_archive_20260922.json')
output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != 'members'}, indent=2))
