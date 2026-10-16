"""Compute-node audit of unchanged canonical validation inputs, no test files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
prepared = root/'data/benchmarking_data/lungpf/prepared'
protocol = json.loads((prepared/'protocol.json').read_text())
vocab = json.loads((root/'resources/scgpt_official/scGPT_human/vocab.json').read_text())
rows = []
for directory in sorted(prepared.iterdir()):
    if not directory.is_dir() or not (directory/'obs.parquet').is_file():
        continue
    matches = [d for d in protocol['donor_split'] if directory.name.startswith(d)]
    if len(matches) != 1 or protocol['donor_split'][matches[0]] == 'test':
        continue
    obs = pd.read_parquet(directory/'obs.parquet')
    selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400, len(obs)), replace=False))
    genes = np.load(directory/'genes.npy', allow_pickle=False).astype(str)
    keep = np.array([i for i,g in enumerate(genes) if g in vocab])
    nuclear = np.load(directory/'nuclear.npy', mmap_mode='r')[np.ix_(selected,keep)]
    cyto = np.load(directory/'cytoplasmic.npy', mmap_mode='r')[np.ix_(selected,keep)]
    row = dict(sample=directory.name, cells=len(selected), matched_genes=len(keep),
               zero_teacher=int((~((nuclear>0)|(cyto>0)).any(axis=1)).sum()),
               zero_nucleus=int((~(nuclear>0).any(axis=1)).sum()), zero_cytoplasm=int((~(cyto>0).any(axis=1)).sum()))
    print(json.dumps(row), flush=True)
    rows.append(row)
report = dict(samples=rows, cells=sum(r['cells'] for r in rows), zero_teacher=sum(r['zero_teacher'] for r in rows))
(root/'logs/v10_benchmark_input_audit_20260929.json').write_text(json.dumps(report,indent=2)+'\n')
assert report['cells'] == 5200
print('AUDIT_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='samples'}))
