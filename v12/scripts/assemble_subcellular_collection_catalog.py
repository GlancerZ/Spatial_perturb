"""Publish a source-backed catalog of this collection, without mixing readiness levels."""
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import re
from prepare_unified_training_cache import verify_bundle

ROOT = Path.cwd()
corpus = ROOT / 'data/subcellular_corpus_v1'
output = corpus / 'collection_20260922'
output.mkdir(parents=True, exist_ok=False)
def read(path):
    return json.loads(Path(path).read_text())

xenium = read(ROOT / 'outputs/existing_compartment_audit_20260922/catalog.json')
cosmx = read(corpus / 'cosmx8_training_20260922/catalog.json')
records = []
for source in xenium['datasets']:
    records.append(dict(source, new_acquisition=False,
        readiness='existing_counts_format_verified_provenance_pending', loader_checked_this_collection=False))
for source in cosmx['datasets']:
    records.append(dict(source, dataset_id='cosmx_nsclc_' + source['sample_id'], platform='CosMx',
        species='Homo sapiens', new_acquisition=True, readiness='format_and_v7_loader_verified',
        loader_checked_this_collection=True, training_started=False))
merfish = read(corpus / 'bento_merfish_20260922/status.json')
if merfish['status'] != 'completed':
    raise ValueError('MERFISH processing is not complete; do not publish a premature catalog')
merfish_files = []
for ready_file in sorted((corpus / 'bento_merfish_20260922').glob('batch_*/bundle/READY.json')):
    ready = read(ready_file)
    if ready['training_eligibility'] != 'validated_nuclear_geometry_and_counts':
        raise ValueError('Unexpected MERFISH bundle readiness')
    source = ready_file.parent / ready['data']
    verify_bundle(source)
    merfish_files.append(str(source.resolve()))
merfish_cache = read(ROOT / 'cache/subcellular_unified_v1/bento_merfish/LOADER_VALIDATED.json')
records.append({'dataset_id': merfish['dataset_id'], 'platform': 'MERFISH', 'species': 'Homo sapiens',
    'new_acquisition': True, 'readiness': 'format_and_v7_loader_verified', 'cells': merfish['eligible_geometry_cells'],
    'source_cells': merfish['source_cells'], 'genes': merfish['biological_genes'], 'fovs': len(merfish_files),
    'nuclear_counts': merfish['nuclear_points'], 'cytoplasmic_counts': merfish['cytoplasmic_points'],
    'h5ad_paths': merfish_files, 'microns_per_pixel': merfish['microns_per_unit'],
    'loader_checked_this_collection': True, 'training_split_ready': False,
    'v7_cache': merfish_cache['combined_cache'], 'model_matched_genes': merfish_cache['model_matched_genes'],
    'specimen_type': 'cultured U2OS cells; not human tissue',
    'source_qc': str((corpus / 'bento_merfish_20260922/status.json').resolve())})
staging = []
for name in ('seqfish_raw', 'vizgen_raw', 'resolve_mc_checked', 'starmap_missing_points'):
    path = corpus / 'staging_20260922' / name / 'manifest.json'
    manifest = read(path)
    if manifest['training_ready'] is not False:
        raise ValueError('Unannotated data must not be advertised as training ready')
    staging.append({'name': name, 'manifest': str(path.resolve()), **manifest})
failed_seqfish = read(corpus / 'bento_seqfish_20260922/status.json')
quarantine = [
    {'dataset_id': 'cosmx_lung_tma2_gsm9046089', 'reason': 'Published tx payload is cell statistics rather than molecular RNA, and cell/FOV axes disagree',
     'training_ready': False, 'evidence': str((ROOT / 'outputs/cosmx_geo_raw_inspection_20260922.json').resolve())},
    {'dataset_id': failed_seqfish['dataset_id'], 'reason': 'Published RNA coordinates omit cell-gene groups with X counts 1-4; a coordinate-truncated derivative, not a complete molecular dataset', 'training_ready': False,
     'source_points': failed_seqfish['source_points'],
     'published_X_mismatched_entries': failed_seqfish['published_X_mismatched_entries'],
     'published_X_absolute_difference': failed_seqfish['published_X_absolute_difference'],
     'evidence': str((ROOT / 'outputs/bento_counts_diagnosis_20260922.json').resolve())},
]
download_reconciliation = read(ROOT / 'outputs/subcellular_download_reconciliation_20260922.json')
bento_files = read(ROOT / 'outputs/bento_public_input_inspection_20260922.json')
calibration = read(ROOT / 'outputs/bento_merfish_calibration_checked_20260922.json')
extra_downloads = [{'path': record['file'], 'bytes': record['bytes'], 'sha256': record['sha256'],
                   'publisher_md5_verified': record['md5']} for record in bento_files.values()]
extra_downloads.append({'path': calibration['raw_tar'], 'bytes': Path(calibration['raw_tar']).stat().st_size,
                       'sha256': calibration['raw_sha256'], 'publisher_md5_verified': '3c329fc33176c1a3789c4e3f1ab94258'})
tests = []
for name in ('unified_final_checks_20260922', 'bento_seqfish_checked_20260922',
             'bento_merfish_calibration_checked_20260922', 'resolve_starmap_checked_20260922'):
    text = (ROOT / 'logs' / (name + '.log')).read_text()
    counts = [int(n) for n in re.findall(r'^Ran (\d+) tests? in ', text, re.M)]
    tests.append({'log': name + '.log', 'test_count': sum(counts), 'successful_test_suites': len(re.findall(r'^OK$', text, re.M)),
                  'suite_count': len(counts), 'full_wrapper_exit': (ROOT / 'logs' / (name + '.exit')).read_text().strip()})
    if len(counts) != len(re.findall(r'^OK$', text, re.M)):
        raise ValueError('A recorded test suite did not pass')
catalog = {'schema': 'subcellular_collection_delivery_v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
    'execution_host': 'vulcan', 'project_root': str(ROOT.resolve()),
    'existing_xenium_summary': xenium['summary'],
    'new_cosmx_summary': {k: v for k, v in cosmx.items() if k != 'datasets'},
    'new_merfish_summary': {k: v for k, v in records[-1].items() if k != 'h5ad_paths'},
    'datasets': records, 'staging': staging, 'quarantine': quarantine,
    'download_bytes_verification': download_reconciliation,
    'bento_downloads': extra_downloads,
    'data_asset_files': download_reconciliation['files'] + len(extra_downloads),
    'data_asset_bytes': download_reconciliation['bytes'] + sum(r['bytes'] for r in extra_downloads),
    'tests': tests, 'passed_test_count': sum(t['test_count'] for t in tests),
    'training_started': False, 'donor_holdout_or_benchmark_overlap_reviewed': False,
    'warnings': ['No staging or quarantine entry is eligible for compartment pretraining',
                 'V7 loader verification is recorded separately from H5AD schema verification',
                 'Retained published gene universe must not be confused with a full original panel',
                 'Do not count raw seqFISH and its Bento derivative as independent biological samples']}
(output / 'catalog.json').write_text(json.dumps(catalog, indent=2))
with (output / 'datasets.tsv').open('w', newline='') as stream:
    columns = ['dataset_id', 'platform', 'species', 'readiness', 'cells', 'genes', 'new_acquisition', 'loader_checked_this_collection']
    writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(records)
print(json.dumps({'catalog': str(output / 'catalog.json'), 'datasets': len(records),
                  'staging_groups': len(staging), 'quarantine_groups': len(quarantine),
                  'passed_tests': catalog['passed_test_count']}, indent=2))
