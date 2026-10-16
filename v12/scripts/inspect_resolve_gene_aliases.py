"""Audit original/assigned RNA pairs without changing either source."""
from collections import Counter
import gzip
import io
from itertools import zip_longest
import json
from pathlib import Path
import tarfile

source = Path('data/subcellular_data_agent/raw_public_20260922/resolve_mc_ad_cortex_gse325911/GSE325911_RAW.tar')
aliases, coordinate_errors, samples, rows = Counter(), [], [], 0
coordinate_error_count = 0
with tarfile.open(source) as archive:
    for member in archive.getmembers():
        if not member.name.endswith('_results_assigned.txt.gz'):
            continue
        original = member.name.replace('_results_assigned.txt.gz', '_results.txt.gz')
        with archive.extractfile(member) as a, archive.extractfile(original) as b, gzip.GzipFile(fileobj=a) as za, gzip.GzipFile(fileobj=b) as zb, io.TextIOWrapper(za) as ta, io.TextIOWrapper(zb) as tb:
            total = 0
            for i, (line_a, line_b) in enumerate(zip_longest(ta, tb)):
                if line_a is None or line_b is None:
                    raise ValueError(f'Row count mismatch: {member.name}')
                assigned, raw = line_a.rstrip('\r\n').split('\t'), line_b.rstrip('\r\n').split('\t')
                if len(assigned) != 5 or len(raw) != 4 or assigned[:3] != raw[:3]:
                    coordinate_error_count += 1
                    if len(coordinate_errors) < 20:
                        coordinate_errors.append({'sample': member.name, 'row': i, 'assigned': assigned, 'raw': raw})
                if len(assigned) >= 4 and len(raw) >= 4 and assigned[3] != raw[3]:
                    aliases[(raw[3], assigned[3])] += 1
                total += 1
            rows += total
            samples.append({'member': member.name, 'rows': total})
            print(f'[resolve-audit] samples={len(samples)} rows={rows} gene_aliases={len(aliases)}', flush=True)
report = {'samples': samples, 'rows': rows, 'coordinate_errors': coordinate_errors,
          'coordinate_error_count': coordinate_error_count,
          'gene_aliases': [{'original': a, 'assigned': b, 'rows': n} for (a, b), n in aliases.items()]}
Path('outputs/resolve_gene_aliases_20260922.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report['gene_aliases'], indent=2))
print(json.dumps(report['coordinate_errors'][:2], indent=2))
