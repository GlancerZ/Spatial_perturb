"""Fixed, label-blind donor subsample; center-KNN embeddings and validation-only probes.

This is a rapid diagnostic, not the historical all-cell benchmark. Test donors
are never read for fitting, checkpoint selection, or reported pilot metrics.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score
from spatial_pretrain.models.compartment import CompartmentSSGTModel
from spatial_pretrain.models.spatial_bias import spatial_kwargs
from spatial_pretrain.data.compartment_binning import evaluation_edges, positive_bins
from spatial_pretrain.gene_init import build_global_gene_vocabulary_from_cache_dirs
from probe_lungpf_embeddings import fit_linear, predict


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg, state = checkpoint['config'], checkpoint['model']
    vocab = build_global_gene_vocabulary_from_cache_dirs(cfg.data.cache_dirs)
    model = CompartmentSSGTModel(num_genes=len(vocab)+1, num_value_bins=cfg.data.num_bins,
        hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers,
        gene_heads=cfg.model.gene_heads, cell_heads=cfg.model.cell_heads,
        dropout=cfg.model.dropout, ffn_mult=cfg.model.ffn_mult,
        gene_padding_idx=len(vocab), gene_embedding_init=state.get('gene_embedding.raw_gene_features'),
        cell_attention_mode=cfg.model.cell_attention_mode, nuclear_tokens=cfg.data.nuclear_tokens,
        cytoplasmic_tokens=cfg.data.cytoplasmic_tokens, cell_token_dim=cfg.model.cell_token_dim,
        require_flash_attention=True, **spatial_kwargs(cfg.model))
    model.load_state_dict(state, strict=True)
    edges = evaluation_edges(cfg.data)
    return model.eval().requires_grad_(False).cuda(), cfg, vocab, edges, checkpoint['step']


def extract(model, cfg, vocab, edges, directory, obs, selected, batch_size):
    coords = obs[['x_centroid','y_centroid']].to_numpy()
    size = min(cfg.data.subgraph_size, len(obs))
    search = NearestNeighbors(n_neighbors=size).fit(coords)
    near = search.kneighbors(coords[selected], return_distance=False)
    nodes = np.stack([np.r_[center, row[row != center][:size-1]] for center, row in zip(selected, near)])
    unique = np.unique(nodes)
    mapped_nodes = np.searchsorted(unique, nodes)
    genes = np.load(directory/'genes.npy')
    lookup = {g:i for i,g in enumerate(vocab)}
    keep = np.array([i for i,g in enumerate(genes) if g in lookup], dtype=np.int64)
    global_ids = np.array([lookup[g] for g in genes[keep]], dtype=np.int64)
    length = cfg.data.nuclear_tokens + cfg.data.cytoplasmic_tokens
    ids = np.full((len(unique),length), len(vocab), dtype=np.int64)
    bins = np.full_like(ids, cfg.data.num_bins+1)
    comp = np.full_like(ids, 2)
    valid = np.zeros_like(ids, dtype=bool)
    for compartment, (name,budget,offset) in enumerate([
        ('nuclear',cfg.data.nuclear_tokens,0),
        ('cytoplasmic',cfg.data.cytoplasmic_tokens,cfg.data.nuclear_tokens)]):
        raw = np.load(directory/f'{name}.npy', mmap_mode='r')
        for j,row in enumerate(unique):
            values = np.log1p(raw[row,keep].astype(np.float32)).astype(np.float16).astype(np.float32)
            expressed = np.flatnonzero(values > 0)
            if len(expressed)>budget:
                rng = np.random.default_rng(np.random.SeedSequence([42,int(row),compartment]))
                expressed = np.sort(rng.choice(expressed,budget,replace=False))
            slots = slice(offset,offset+len(expressed))
            ids[j,slots] = global_ids[expressed]
            bins[j,slots] = positive_bins(values[expressed], edges)
            comp[j,slots] = compartment
            valid[j,slots] = True
    result=[]
    for start in range(0,len(nodes),batch_size):
        index=mapped_nodes[start:start+batch_size]
        roles=np.zeros(index.shape,dtype=np.int64);roles[:,0]=1
        batch=dict(gene_ids=ids[index],input_bins=bins[index],compartment_ids=comp[index],
                   token_valid_mask=valid[index],cell_roles=roles,node_indices=nodes[start:start+batch_size],
                   coords=coords[nodes[start:start+batch_size]].astype(np.float32))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            embedding=model(**{k:torch.as_tensor(v,device='cuda') for k,v in batch.items()},
                            return_embeddings=True)['cell_embeddings'][:,0].float().cpu().numpy()
        result.append(embedding)
        if start == 0 or (start + batch_size) % 50 == 0:
            print(f'[embed-progress] {directory.name} {min(start+batch_size,len(nodes))}/{len(nodes)}',flush=True)
    return np.concatenate(result)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cells-per-donor',type=int,default=400)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--expected-step',type=int)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    args.output.mkdir(parents=True,exist_ok=True)
    model,cfg,vocab,edges,step=load_model(args.checkpoint)
    if args.expected_step is not None and step != args.expected_step:
        raise ValueError(f'Checkpoint step mismatch: {step} != {args.expected_step}')
    prepared=ROOT/'data/benchmarking_data/lungpf/prepared'
    protocol=json.loads((prepared/'protocol.json').read_text())
    frames=[];embeddings=[]
    for sample in protocol['samples']:
        directory=prepared/sample
        obs=pd.read_parquet(directory/'obs.parquet')
        donors=obs.patient.unique()
        if len(donors)!=1:
            raise ValueError(f'Expected one donor per sample: {sample}')
        split=protocol['donor_split'][donors[0]]
        if split=='test':
            continue
        if 'split' in obs and not obs['split'].eq(split).all():
            raise ValueError(f'Split mismatch: {sample}')
        selected=np.sort(np.random.default_rng(42).choice(len(obs),min(args.cells_per_donor,len(obs)),replace=False))
        chosen=obs.iloc[selected].copy();chosen['split']=split
        cache_dir=args.output/'sample_embeddings'
        cache_dir.mkdir(exist_ok=True)
        cache_path=cache_dir/f'{sample}.npy'
        cache_meta=cache_dir/f'{sample}.json'
        signature=dict(checkpoint=str(args.checkpoint.resolve()),step=step,context=cfg.data.subgraph_size,
            selected_ids_sha256=hashlib.sha256('\n'.join(chosen.full_cell_id.astype(str)).encode()).hexdigest(),
            extraction_version=2)
        if getattr(cfg.data, 'compartment_bin_edges', []) or getattr(cfg.model, 'cell_spatial_bias', False):
            signature.update(extraction_version=3, bin_edges=edges.tolist(), spatial=spatial_kwargs(cfg.model))
        if cache_path.exists() and cache_meta.exists() and json.loads(cache_meta.read_text())==signature:
            embedding=np.load(cache_path)
            if embedding.shape!=(len(selected),cfg.model.hidden_dim) or not np.isfinite(embedding).all():
                raise ValueError(f'Invalid cached embedding: {cache_path}')
        else:
            embedding=extract(model,cfg,vocab,edges,directory,obs,selected,args.batch_size)
            staged=cache_dir/f'{sample}.partial.npy'
            np.save(staged,embedding)
            staged.replace(cache_path)
            cache_meta.write_text(json.dumps(signature,indent=2))
        embeddings.append(embedding)
        frames.append(chosen)
        print(f'[embed] step={step} {sample} {split} centers={len(selected)}',flush=True)
    obs=pd.concat(frames,ignore_index=True);x=np.concatenate(embeddings)
    assert obs.full_cell_id.is_unique and not obs['split'].eq('test').any()
    assert np.isfinite(x).all()
    obs.to_parquet(args.output/'cells.parquet',index=False)
    np.save(args.output/'embeddings.npy',x)
    del model
    torch.cuda.empty_cache()
    results=[]
    for task,column in [('cell_type','final_CT'),('region','region_label')]:
        labels=obs[column]
        usable=labels.notna() & ~labels.astype(str).isin(['','nan','None'])
        train=usable & obs['split'].eq('train');val=usable & obs['split'].eq('val')
        classes=np.unique(labels[train].astype(str))
        if len(classes)<2 or not val.any():
            raise RuntimeError(f'Insufficient pilot labels for {task}')
        lookup={c:i for i,c in enumerate(classes)}
        scaler=StandardScaler().fit(x[train])
        tx=torch.tensor(scaler.transform(x[train]),device='cuda',dtype=torch.float32)
        vx=torch.tensor(scaler.transform(x[val]),device='cuda',dtype=torch.float32)
        y=torch.tensor([lookup[c] for c in labels[train].astype(str)],device='cuda')
        true=labels[val].astype(str).to_numpy();grid=[];best_pred=None;best_score=-1.
        for C in [.01,.1,1.,10.]:
            W,b,info=fit_linear(tx,y,C)
            pred=predict(vx,W,b,classes)
            f1=f1_score(true,pred,labels=np.unique(true),average='macro',zero_division=0)
            donor_scores={str(d):f1_score(true[obs.loc[val,'patient'].to_numpy()==d],
                pred[obs.loc[val,'patient'].to_numpy()==d],
                labels=np.unique(true[obs.loc[val,'patient'].to_numpy()==d]),average='macro',zero_division=0)
                for d in obs.loc[val,'patient'].unique()}
            grid.append(dict(C=C,val_macro_f1=float(f1),val_balanced_accuracy=float(balanced_accuracy_score(true,pred)),
                             per_donor_macro_f1=donor_scores,solver=info))
            if f1 > best_score:
                best_score=float(f1);best_pred=pred.copy()
        best=max(grid,key=lambda row:row['val_macro_f1'])
        prediction_frame=obs.loc[val,['full_cell_id','patient']].copy()
        prediction_frame['label']=true
        prediction_frame['prediction']=best_pred
        prediction_frame.to_parquet(args.output/f'{task}_val_predictions.parquet',index=False)
        results.append(dict(task=task,train_cells=int(train.sum()),val_cells=int(val.sum()),
            unseen_val_classes=sorted(set(true)-set(classes)),best=best,grid=grid))
        print(f'[probe] {task} step={step} best={best}',flush=True)
    report=dict(checkpoint=str(args.checkpoint.resolve()),step=step,seed=42,cells_per_donor=args.cells_per_donor,
        cohort_sha256=hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest(),
        context=f'Each target is center of its own {cfg.data.subgraph_size}-cell KNN context',
        pooling='Final valid cytoplasmic token mean' if cfg.model.cell_attention_mode=='cytoplasm_mean' else 'Final valid token mean',
        binning='shared_train_edges' if getattr(cfg.data, 'compartment_bin_edges', []) else 'legacy_median_approximation',
        bin_edges=edges.tolist(), spatial=spatial_kwargs(cfg.model),
        results=results,embedding_mean_std=float(x.std(axis=0).mean()),
        mean_embedding_norm=float(np.linalg.norm(x,axis=1).mean()),test_donors_used=False)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
