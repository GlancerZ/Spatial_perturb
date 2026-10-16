import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
successful = []
for batch in (1, 2, 4, 8, 12, 16, 20, 24, 32):
    with (ROOT / 'logs' / f'batch_probe_{batch}.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
            '--nproc_per_node=2', 'scripts/benchmark_compartment_batch.py', '--batch', str(batch)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        text = (ROOT / 'logs' / f'batch_probe_{batch}.log').read_text()
        if 'out of memory' not in text.lower():
            raise RuntimeError(f'Batch probe {batch} failed for a reason other than capacity; inspect its log')
        break
    stats = json.loads((ROOT / 'logs' / f'batch_probe_{batch}.json').read_text())
    print('[capacity] '+json.dumps(stats), flush=True)
    if stats['peak_reserved_gib'] > 0.90 * stats['total_gib']:
        break
    successful.append(stats)
if not successful:
    raise RuntimeError('No batch size passed with the required memory margin')
selected = successful[-1]
(ROOT / 'logs/compartment_batch_selection.json').write_text(json.dumps(selected, indent=2))
print('[selected] '+json.dumps(selected), flush=True)
