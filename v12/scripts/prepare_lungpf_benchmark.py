"""Aggregate corrected public transcripts; freeze donor splits before inference."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

ROOT = Path('data/benchmarking_data/lungpf')
OUT = ROOT / 'prepared'

def prepare(item):
    sample = item['sample']
    dest = OUT / sample
    if (dest / 'qa.json').exists():
        obs = pd.read_parquet(dest/'obs.parquet')
        fresh = pd.read_parquet(OUT/'metadata.parquet').set_index('full_cell_id')
        obs['region_label'] = obs.full_cell_id.map(fresh.region_label)
        obs.to_parquet(dest/'obs.parquet', index=False)
        qa = json.loads((dest/'qa.json').read_text())
        qa['region_counts'] = obs.region_label.value_counts().to_dict()
        (dest/'qa.json').write_text(json.dumps(qa, indent=2))
        print(json.dumps(qa), flush=True)
        return qa
    dest.mkdir(parents=True, exist_ok=True)
    obs = pd.read_parquet(OUT / 'metadata.parquet')
    obs = obs.loc[obs['sample'].eq(sample)].copy().reset_index(drop=True)
    index = pd.Index(obs.full_cell_id)
    assert index.is_unique
    # Discover biological features from this corrected file, excluding control codewords.
    path = Path(item['path'])
    genes = set()
    for chunk in pd.read_csv(path, usecols=['feature_name'], chunksize=1_000_000):
        genes.update(chunk.feature_name.dropna().unique())
    genes = sorted(g for g in genes if not g.lower().startswith(
        ('negcontrol', 'negative', 'unassigned', 'deprecated', 'blank', 'poscontrol')))
    gi = pd.Index(genes)
    counts = np.zeros((2, len(obs), len(genes)), dtype=np.uint32)
    rows = accepted = 0
    for chunk in pd.read_csv(path, usecols=['cell_id','feature_name','qv','overlaps_nucleus'],
                             chunksize=1_000_000):
        rows += len(chunk)
        ci = index.get_indexer(chunk.cell_id)
        gj = gi.get_indexer(chunk.feature_name)
        keep = (ci >= 0) & (gj >= 0) & chunk.qv.ge(20).to_numpy()
        compartment = chunk.overlaps_nucleus.to_numpy()[keep]
        assert np.isin(compartment, [0, 1]).all()
        np.add.at(counts, (1-compartment.astype(int), ci[keep], gj[keep]), 1)
        accepted += int(keep.sum())
    nuclear, cyto = counts
    assert int(counts.sum(dtype=np.uint64)) == accepted
    nonempty = (nuclear.sum(1) + cyto.sum(1)) > 0
    obs = obs.loc[nonempty].reset_index(drop=True)
    nuclear, cyto = nuclear[nonempty], cyto[nonempty]
    np.save(dest/'nuclear.npy', nuclear)
    np.save(dest/'cytoplasmic.npy', cyto)
    np.save(dest/'genes.npy', np.asarray(genes))
    obs.to_parquet(dest/'obs.parquet', index=False)
    reference = obs.nCount_RNA.to_numpy()
    qa = dict(sample=sample, cells=len(obs), genes=len(genes), raw_transcripts=rows,
              accepted_transcripts=accepted, excluded_empty_cells=int((~nonempty).sum()),
              nuclear_reference_exact_fraction=float(np.mean(nuclear.sum(1)==reference)),
              nuclear_reference_correlation=float(np.corrcoef(nuclear.sum(1), reference)[0,1]),
              region_counts=obs.region_label.value_counts().to_dict(),
              split=obs['split'].iloc[0])
    (dest/'qa.json').write_text(json.dumps(qa, indent=2))
    print(json.dumps(qa), flush=True)
    return qa

def main():
    OUT.mkdir(exist_ok=True)
    items = json.loads((ROOT/'download_manifest.json').read_text())
    samples = [i['sample'] for i in items]
    meta = pd.read_csv(ROOT/'cell_metadata.csv.gz')
    meta = meta.loc[meta['sample'].isin(samples)].copy()
    assert len(samples)==16 and meta.patient.nunique()==16
    # Healthy donors explicitly represented in each split; disease donors shuffled
    # once with a fixed seed, without consulting cell/region class distributions.
    healthy = sorted(meta.loc[meta.disease_status.eq('Control'),'patient'].unique())
    assert len(healthy)==3, healthy
    disease = sorted(set(meta.patient)-set(healthy))
    rng = np.random.default_rng(42)
    rng.shuffle(disease)
    split = {p:'train' for p in disease[:9]}
    split.update({p:'val' for p in disease[9:11]})
    split.update({p:'test' for p in disease[11:]})
    split.update(dict(zip(healthy,['train','val','test'])))
    meta['split'] = meta.patient.map(split)
    ann = pd.read_csv(ROOT/'pathology.csv.gz')
    # H&E table retains old sample names. The corrected transcript files
    # explicitly provide the authoritative old-to-current mapping.
    mapping = {}
    for item in items:
        header = pd.read_csv(item['path'], nrows=1)
        mapping[header.old_sample_name.iloc[0]] = item['sample']
    ann = ann.loc[ann['sample'].isin(mapping)].copy()
    ann['sample'] = ann['sample'].map(mapping)
    ann['full_cell_id'] = ann['sample'] + '_' + ann.cell_id
    ann = ann.loc[ann['sample'].isin(samples)]
    matched = ann.merge(meta[['full_cell_id','x_centroid','y_centroid']],on='full_cell_id',suffixes=('_ann','_meta'))
    assert len(matched)==len(ann), (len(matched),len(ann))
    for axis in ['x_centroid','y_centroid']:
        assert np.allclose(matched[axis+'_ann'], matched[axis+'_meta'], atol=0.01)
    grouped = ann.groupby('full_cell_id').Annotation_Type
    unique = grouped.nunique()
    labels = grouped.first().loc[unique.eq(1)]
    meta['region_label'] = meta.full_cell_id.map(labels)
    meta.to_parquet(OUT/'metadata.parquet', index=False)
    protocol = dict(seed=42, donor_split=split, samples=samples,
        cell_type_label='final_CT', region_label='independent pathology Annotation_Type',
        conflicting_region_cells=int(unique.gt(1).sum()), qv_min=20, old_to_current_sample_names=mapping,
        sampling='All nonempty annotated cells; all cells retained for spatial context',
        classifier='StandardScaler + balanced LogisticRegression; C chosen on validation macro-F1',
        C_grid=[0.01,0.1,1,10], encoder_training=False)
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2))
    with ProcessPoolExecutor(max_workers=4) as pool:
        qa = list(pool.map(prepare, items))
    (OUT/'qa.json').write_text(json.dumps(qa, indent=2))

if __name__=='__main__':
    main()
