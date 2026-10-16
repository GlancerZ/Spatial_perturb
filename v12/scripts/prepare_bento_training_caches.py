"""Exercise the current V7 loader for every calibrated MERFISH FOV separately."""
from pathlib import Path
import json
import sys
import yaml
from prepare_unified_training_cache import main as prepare_cache, check_loader

root = Path.cwd()
source = root / 'data/subcellular_corpus_v1/bento_merfish_20260922'
output = root / 'cache/subcellular_unified_v1/bento_merfish'
output.mkdir(parents=True, exist_ok=False)
ready_paths = sorted(source.glob('batch_*/bundle/READY.json'))
if len(ready_paths) != 70:
    raise ValueError('Expected all 70 calibrated FOVs')
records, manifests = [], []
for ready_path in ready_paths:
    name = ready_path.parent.parent.name
    ready = json.loads(ready_path.read_text())
    target = output / name
    sys.argv = ['prepare_unified_training_cache.py', '--h5ad', str(ready_path.parent / ready['data']),
                '--output', str(target), '--vocab', str(root / 'resources/scgpt_official/scGPT_human/vocab.json'),
                '--config', str(root / 'configs/subcellular_lr_v7_a_20260921.yaml'),
                '--domain-column', 'slide_id', '--species', 'Homo sapiens',
                '--coordinate-unit', 'micrometer', '--exclude-empty-cells']
    prepare_cache()
    manifest = json.loads((target / 'v7_cache/manifest.json').read_text())
    checks = json.loads((target / 'LOADER_VALIDATED.json').read_text())
    view = json.loads((target / 'training_view.json').read_text())
    manifests.append(manifest)
    records.append({'batch': name, 'training_view': view, 'loader_checks': checks,
                    'cache_root': str(target / 'v7_cache')})
    print(f'[MERFISH-V7] {len(records)}/70 complete', flush=True)
base = manifests[0]
if any(m['data_config'] != base['data_config'] or m['vocab_sha256'] != base['vocab_sha256'] for m in manifests):
    raise ValueError('Inconsistent data contracts')
combined = dict(base, slice_count=len(manifests), slices=[m['slices'][0] for m in manifests],
                source_biological_sample_count=1, independent_fov_count=70,
                donor_split_reviewed=False, training_started=False)
cache_root = output / 'combined_v7_cache'
cache_root.mkdir()
(cache_root / 'manifest.json').write_text(json.dumps(combined, indent=2))
config = yaml.safe_load((root / 'configs/subcellular_lr_v7_a_20260921.yaml').read_text())
config['cache_root'], config['assets_dir'] = str(cache_root), str(root / 'resources/scgpt_official/scGPT_human')
try:
    checks = check_loader(config)
except BaseException:
    (cache_root / 'manifest.json').rename(cache_root / 'manifest.failed_validation.json')
    raise
summary = {'status': 'format_and_v7_loader_verified', 'fovs': len(records),
           'training_cells': sum(r['training_view']['kept_cells'] for r in records),
           'archive_cells': sum(r['training_view']['archival_cells'] for r in records),
           'combined_cache': str(cache_root), 'combined_loader_checks': checks,
           'model_matched_genes': sorted({m['slices'][0]['num_genes'] for m in manifests}),
           'unmatched_genes': sorted({g for m in manifests for g in m['slices'][0]['excluded_unmatched_genes']}),
           'training_started': False, 'donor_split_reviewed': False, 'records': records}
(output / 'LOADER_VALIDATED.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2), flush=True)
