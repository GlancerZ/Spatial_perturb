"""Frozen encoders on a fixed common cohort; restartable per sample."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path('data/benchmarking_data/lungpf/prepared')
OUT = Path('outputs/xenium_step1500_benchmark')

def load_ours(checkpoint_path=None, expected_step=1500, write_metadata=True):
    from spatial_pretrain.models.compartment import CompartmentSSGTModel
    from spatial_pretrain.models.spatial_bias import spatial_kwargs
    from spatial_pretrain.data.compartment_binning import evaluation_edges
    from spatial_pretrain.gene_init import build_global_gene_vocabulary_from_cache_dirs
    ckpt = torch.load(checkpoint_path or OUT/'step1500_resume.pt', map_location='cpu', weights_only=False)
    assert ckpt['step']==expected_step
    cfg, state = ckpt['config'], ckpt['model']
    vocab = build_global_gene_vocabulary_from_cache_dirs(cfg.data.cache_dirs)
    init = state.get('gene_embedding.raw_gene_features')
    model = CompartmentSSGTModel(num_genes=len(vocab)+1, num_value_bins=cfg.data.num_bins,
        hidden_dim=cfg.model.hidden_dim, num_layers=cfg.model.num_layers,
        gene_heads=cfg.model.gene_heads, cell_heads=cfg.model.cell_heads,
        dropout=cfg.model.dropout, ffn_mult=cfg.model.ffn_mult,
        gene_padding_idx=len(vocab), gene_embedding_init=init,
        cell_attention_mode=cfg.model.cell_attention_mode,
        nuclear_tokens=cfg.data.nuclear_tokens, cytoplasmic_tokens=cfg.data.cytoplasmic_tokens,
        cell_token_dim=cfg.model.cell_token_dim, require_flash_attention=True, **spatial_kwargs(cfg.model))
    model.load_state_dict(state, strict=True)
    # Use a fixed pretraining-derived binning distribution; no test fitting.
    # Equal weight to each pretraining slide's learned quantile thresholds.
    edges = evaluation_edges(cfg.data)
    if write_metadata:
        np.save(OUT/'pretraining_bin_edges.npy', edges)
        (OUT/'our_gene_vocabulary.json').write_text(json.dumps(vocab))
    print('OURS_LOADED', ckpt['step'], len(vocab), sum(p.numel() for p in model.parameters()), flush=True)
    del ckpt, state
    return model.eval().requires_grad_(False).cuda(), vocab, edges

def spatial_groups(coords, size=256):
    """Label-free recursive spatial partition, every cell emitted once."""
    groups = []
    def divide(ids):
        if len(ids)<=size:
            groups.append(ids)
            return
        axis = np.ptp(coords[ids], axis=0).argmax()
        ids = ids[np.argsort(coords[ids,axis], kind='stable')]
        # Keep full 256-cell leaves, rather than splitting into many tiny groups.
        middle = max(size, (len(ids)//2//size)*size)
        divide(ids[:middle]); divide(ids[middle:])
    divide(np.arange(len(coords)))
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(coords)))
    return groups

def ours_sample(model, vocab, edges, directory, obs, batch_size):
    genes = np.load(directory/'genes.npy')
    lookup = {g:i for i,g in enumerate(vocab)}
    keep = np.array([i for i,g in enumerate(genes) if g in lookup])
    global_ids = np.array([lookup[g] for g in genes[keep]])
    # Preserve original panel order, as in pretraining; random inclusion only
    # when a compartment has more than 256 expressed genes.
    n = len(obs)
    ids = np.full((n,512), len(vocab), dtype=np.int64)
    bins = np.full((n,512), 33, dtype=np.int64)
    comp = np.full((n,512), 2, dtype=np.int64)
    valid = np.zeros((n,512), dtype=bool)
    rng = np.random.default_rng(42)
    for compartment, name in enumerate(['nuclear','cytoplasmic']):
        raw = np.load(directory/f'{name}.npy', mmap_mode='r')
        for row in range(n):
            values = np.log1p(raw[row,keep].astype(np.float32)).astype(np.float16).astype(np.float32)
            selected = np.flatnonzero(values>0)
            if len(selected)>256: selected=np.sort(rng.choice(selected,256,replace=False))
            slots = slice(256*compartment,256*compartment+len(selected))
            ids[row,slots] = global_ids[selected]
            bins[row,slots] = np.digitize(values[selected],edges)+1
            valid[row,slots] = True
            comp[row,slots] = compartment
    groups = spatial_groups(obs[['x_centroid','y_centroid']].to_numpy())
    result = np.empty((n,256), dtype=np.float32)
    start = time.monotonic()
    for i in range(0,len(groups),batch_size):
        current = groups[i:i+batch_size]
        # Tail group padded with invalid cells and unique negative node IDs.
        nodes = np.full((len(current),256), -1, dtype=np.int64)
        g = np.full((len(current),256,512),len(vocab),dtype=np.int64)
        b = np.full_like(g,33); c = np.full_like(g,2); v=np.zeros_like(g,dtype=bool)
        for j, rows in enumerate(current):
            k=len(rows); nodes[j,:k]=rows
            g[j,:k],b[j,:k],c[j,:k],v[j,:k]=ids[rows],bins[rows],comp[rows],valid[rows]
        roles=np.zeros(nodes.shape,dtype=np.int64); roles[:,0]=1
        batch=dict(gene_ids=g,input_bins=b,compartment_ids=c,token_valid_mask=v,
                   node_indices=nodes,cell_roles=roles,
                   coords=obs[['x_centroid','y_centroid']].to_numpy(dtype=np.float32)[nodes.clip(min=0)])
        batch={k:torch.from_numpy(a).cuda() for k,a in batch.items()}
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            embedding=model(**batch,return_embeddings=True)['cell_embeddings'].cpu().numpy()
        for j,rows in enumerate(current): result[rows]=embedding[j,:len(rows)]
        if i%40==0: print(directory.name,'ours groups',i,'/',len(groups),'seconds',round(time.monotonic()-start,1),flush=True)
    return result, dict(panel_genes=len(genes), recognized_genes=len(keep), context='disjoint spatial groups of up to 256 cells',
                        pooling='mean of final normalized valid tokens', empty_known_cells=int((~valid.any(1)).sum()))

def load_spatialformer():
    from spatialformer.tools.get_embeddings import manual_train_fm
    pkg=Path('resources/spatialformer/package/spatialformer')
    config=json.loads((pkg/'config/_config_train_large_pair.json').read_text())
    config['embedding_path']=str(pkg/'spatial_embeddings/xenium_5k_pandavid_gene_embeddings.pkl')
    model=manual_train_fm(config,use_flash_attn=False)
    ckpt=torch.load('resources/spatialformer/single_71.ckpt',map_location='cpu',weights_only=False)
    model.load_state_dict(ckpt['state_dict'],strict=True)
    print('SPATIALFORMER_LOADED', ckpt['global_step'],sum(p.numel() for p in model.parameters()),flush=True)
    return model.eval().requires_grad_(False).cuda(), pkg

def sf_sample(model,pkg,directory,obs,batch_size):
    from spatialformer.tools.get_embeddings import GeneTokenizer, collate_fn
    tokenizer=GeneTokenizer(str(pkg/'tokenizer/tokenv5.json'),mode='single',tissue='Lung',
        condition='Healthy' if obs.disease_status.iloc[0]=='Control' else 'Disease',assay='Xenium',
        gene_median_path=str(pkg/'data/gene_median.pkl'))
    genes=np.load(directory/'genes.npy'); tokenizer.genes=genes
    expression=np.load(directory/'nuclear.npy')+np.load(directory/'cytoplasmic.npy')
    result=np.empty((len(obs),512),dtype=np.float32)
    start=time.monotonic()
    for i in range(0,len(obs),batch_size):
        encoded=[tokenizer.encode(row) for row in expression[i:i+batch_size]]
        # Official collate infers pair mode from len(first_gene_list)==2,
        # which is ambiguous for a real single cell with two expressed genes.
        # Construct the documented single-cell tensors explicitly.
        sequences=[prefix+tokens+end for tokens,prefix,end in encoded]
        length=max(map(len,sequences))
        indices=torch.zeros((len(sequences),length),dtype=torch.long)
        mask=torch.zeros_like(indices,dtype=torch.bool)
        lengths=torch.zeros((len(sequences),2),dtype=torch.long)
        for row,seq in enumerate(sequences):
            indices[row,:len(seq)]=torch.tensor(seq)
            mask[row,:len(seq)]=True; lengths[row,0]=len(seq)
        batch=dict(indices=indices,attention_mask=mask,token_type_ids=torch.ones_like(indices),sequence_length=lengths)
        if i==0 and len(encoded[0][0])!=2:
            reference_batch=collate_fn(encoded)
            for key in batch: assert torch.equal(batch[key],reference_batch[key]),key
        with torch.inference_mode():
            # Same official encoder operations; omit unused MLM/adjacency heads.
            batch={k:v.cuda() for k,v in batch.items()}
            x=model.embeddings(batch['indices'])
            x+=model.token_type_embeddings(batch['token_type_ids'])
            x+=model.spatialembeds(batch['indices'])
            embedding=model.encoder(x,False,batch['attention_mask'])[0][-1][:,0,:]
            if i==0:
                reference=model.get_embeddings(batch,layers=[-1],pair_prediction=False,co_prediction=False)[0][:,0,:]
                torch.testing.assert_close(embedding,reference,rtol=1e-5,atol=1e-6)
                print('OFFICIAL_EMBEDDING_EQUIVALENCE_OK',directory.name,flush=True)
            result[i:i+len(encoded)]=embedding.float().cpu().numpy()
        if i%(batch_size*100)==0: print(directory.name,'spatialformer',i,'/',len(obs),'seconds',round(time.monotonic()-start,1),flush=True)
    return result,dict(panel_genes=len(genes),recognized_genes=sum(g in tokenizer.token_to_id for g in genes),
                       pooling='official final-layer CLS',condition_token=tokenizer.condition_id)

def main():
    global OUT
    parser=argparse.ArgumentParser(); parser.add_argument('--model',choices=['ours','spatialformer'],required=True)
    parser.add_argument('--batch-size',type=int,default=None); parser.add_argument('--sample')
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--step',type=int,default=1500)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--num-shards',type=int,default=1)
    args=parser.parse_args(); torch.set_num_threads(4)
    OUT=args.output_dir
    OUT.mkdir(exist_ok=True)
    if args.model=='ours': model,vocab,edges=load_ours(args.checkpoint,args.step,args.shard==0)
    else: model,pkg=load_spatialformer()
    directories=[d for d in sorted(ROOT.iterdir()) if d.is_dir() and (d/'qa.json').exists()]
    for directory in directories[args.shard::args.num_shards]:
        if args.sample and args.sample!=directory.name: continue
        dest=OUT/'embeddings'/directory.name; dest.mkdir(parents=True,exist_ok=True)
        if (dest/f'{args.model}.npy').exists(): continue
        obs=pd.read_parquet(directory/'obs.parquet')
        if args.model=='ours': embedding,qa=ours_sample(model,vocab,edges,directory,obs,args.batch_size or 4)
        else: embedding,qa=sf_sample(model,pkg,directory,obs,args.batch_size or 32)
        assert embedding.shape[0]==len(obs) and np.isfinite(embedding).all()
        qa.update(cells=len(obs),dimensions=embedding.shape[1],frozen=True)
        obs[['full_cell_id','sample','patient','split','final_CT','region_label']].to_parquet(dest/'cells.parquet',index=False)
        np.save(dest/f'{args.model}.partial.npy',embedding)
        (dest/f'{args.model}.partial.npy').replace(dest/f'{args.model}.npy')
        (dest/f'{args.model}.json').write_text(json.dumps(qa,indent=2))
        print('COMPLETE',directory.name,args.model,qa,flush=True)

if __name__=='__main__': main()
