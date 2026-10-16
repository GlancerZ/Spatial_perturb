"""Validate and expose all eight newly converted CosMx sources as one V7 cache.

This does not start training or alter any existing experiment configuration.
Donor-level evaluation splits remain a separate, explicitly recorded decision.
"""
from pathlib import Path
import json
import h5py
import yaml
from prepare_unified_training_cache import verify_bundle, string_column, geometry_fingerprint, check_loader

ROOT = Path.cwd()
output = ROOT / 'data/subcellular_corpus_v1/cosmx8_training_20260922'
output.mkdir(parents=True, exist_ok=False)
samples = ['Lung5_Rep1', 'Lung5_Rep2', 'Lung5_Rep3', 'Lung6', 'Lung9_Rep1', 'Lung9_Rep2', 'Lung12', 'Lung13']
records, manifests = [], []
for sample in samples:
    corpus = 'cosmx_lung12_20260922' if sample == 'Lung12' else 'cosmx_expansion_20260922'
    summary_path = ROOT / 'data/subcellular_corpus_v1' / corpus / sample / 'COMPLETE.json'
    record = json.loads(summary_path.read_text())
    verify_bundle(Path(record['data']))
    cache_name = 'cosmx_Lung12_checked' if sample == 'Lung12' else 'cosmx_' + sample
    cache = ROOT / 'cache/subcellular_unified_v1' / cache_name
    checks = json.loads((cache / 'LOADER_VALIDATED.json').read_text())
    view = json.loads((cache / 'training_view.json').read_text())
    manifest_path = cache / 'v7_cache/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    metadata = manifest['slices'][0]
    geometry = json.loads((cache / 'geometry/metadata.json').read_text())
    with h5py.File(metadata['signature']['source'], 'r') as handle:
        groups = string_column(handle['obs'][geometry['domain_column']])
    fingerprint = geometry_fingerprint(cache / 'geometry', groups, geometry['domain_column'])
    previous = metadata['signature'].get('verified_geometry')
    if previous is not None and previous != fingerprint:
        raise ValueError(f'Geometry changed after cache generation: {sample}')
    if previous is None:
        # Upgrade only this task's first cache to the final graph-bound contract.
        metadata['signature']['verified_geometry'] = fingerprint
        (Path(metadata['cache_dir']) / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    record.update(training_cells=view['kept_cells'], excluded_empty_cells=view['excluded_cells'],
                  v7_cache=str((cache / 'v7_cache').resolve()), loader_checks=checks,
                  model_matched_genes=metadata['num_genes'],
                  vocab_unmatched_genes=metadata['excluded_unmatched_genes'],
                  donor_split_reviewed=False, training_started=False)
    records.append(record)
    manifests.append(manifest)
    print(f"[cosmx8] verified {sample} {record['training_cells']} training cells", flush=True)
base = manifests[0]
if any(m['data_config'] != base['data_config'] or m['vocab_sha256'] != base['vocab_sha256'] for m in manifests):
    raise ValueError('Cannot combine caches with different vocabulary or data contracts')
combined = dict(base, slice_count=8, slices=[m['slices'][0] for m in manifests],
                donor_groups={r['sample_id']: r['donor_id'] for r in records},
                donor_split_reviewed=False, training_started=False,
                validation_scope='loader and spatial block split only; not independent-donor evaluation')
cache_root = output / 'v7_cache'
cache_root.mkdir()
(cache_root / 'manifest.json').write_text(json.dumps(combined, indent=2))
config = yaml.safe_load((ROOT / 'configs/subcellular_lr_v7_a_20260921.yaml').read_text())
config['cache_root'] = str(cache_root.resolve())
config['assets_dir'] = str((ROOT / 'resources/scgpt_official/scGPT_human').resolve())
try:
    checks = check_loader(config)
except BaseException:
    (cache_root / 'manifest.json').rename(cache_root / 'manifest.failed_validation.json')
    raise
(output / 'LOADER_VALIDATED.json').write_text(json.dumps(checks, indent=2))
summary = {'schema': 'subcellular_corpus_index_v1', 'status': 'format_and_loader_verified',
           'samples': len(records), 'source_donor_groups': len({r['donor_id'] for r in records}),
           'archival_cells': sum(r['cells'] for r in records),
           'training_cells': sum(r['training_cells'] for r in records),
           'training_transcripts': sum(r['training_transcripts'] for r in records),
           'nuclear_transcripts': sum(r['nuclear'] for r in records),
           'cytoplasmic_transcripts': sum(r['cytoplasmic'] for r in records),
           'membrane_transcripts': sum(r['membrane'] for r in records),
           'measured_genes_per_sample': 960, 'model_matched_genes_per_sample': 958,
           'all_vendor_expression_reconstructions_exact': True,
           'donor_split_reviewed': False, 'training_started': False,
           'combined_v7_cache': str(cache_root.resolve()), 'datasets': records}
(output / 'catalog.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != 'datasets'}, indent=2), flush=True)
