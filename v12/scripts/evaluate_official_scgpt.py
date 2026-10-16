"""Frozen official scGPT-spatial V1 on the exact ongoing validation cohort.

Use the upstream standard-PyTorch inference path, bypassing imports for unused
legacy FlashAttention and torchtext. The official model math and collator are
retained; every inference encoder weight must be present in the checkpoint.
"""
import ast
import hashlib
import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score
from probe_lungpf_embeddings import fit_linear, predict

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT/'resources/scgpt_official'
OUT = ROOT/'outputs/scgpt_official_v1_validation'


def load_official():
    code = ASSET/'code/scgpt_spatial'
    # Avoid package __init__ eagerly importing unused training dependencies.
    for name, path in [('scgpt_spatial', code), ('scgpt_spatial.model', code/'model')]:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.logger = logging.getLogger(name)
        sys.modules[name] = module
    path = code/'model/model.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom)
        and n.module in ('flash_attn.flash_attention', 'flash_layers'))]
    module = types.ModuleType('scgpt_spatial.model.model')
    module.__package__ = 'scgpt_spatial.model'
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    preprocess = types.ModuleType('scgpt_spatial.preprocess')
    preprocess.__dict__.update(np=np, torch=torch, Union=Union)
    source = ast.parse((code/'preprocess.py').read_text())
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ('_digitize', 'binning')]
    assert len(functions) == 2
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(code/'preprocess.py'), 'exec'), preprocess.__dict__)
    sys.modules[preprocess.__name__] = preprocess
    DataCollator = importlib.import_module('scgpt_spatial.data_collator').DataCollator
    weights = ASSET/'scGPT_spatial_v1'
    cfg = json.loads((weights/'args.json').read_text())
    vocab = json.loads((weights/'vocab.json').read_text())
    for token in ['<pad>', '<cls>', '<eoc>']:
        if token not in vocab:
            vocab[token] = len(vocab)
    model = module.TransformerModel(ntoken=len(vocab), d_model=cfg['embsize'], nhead=cfg['nheads'],
        d_hid=cfg['d_hid'], nlayers=cfg['nlayers'], nlayers_cls=cfg['n_layers_cls'], n_cls=1,
        vocab=vocab, dropout=cfg['dropout'], pad_token=cfg['pad_token'], pad_value=cfg['pad_value'],
        do_mvc=True, do_dab=False, use_generative_training=False, use_batch_labels=False,
        explicit_zero_prob=False, use_fast_transformer=False, pre_norm=False,
        use_MVC_impute=True, use_moe_dec=True)
    state = torch.load(weights/'best_model.pt', map_location='cpu', weights_only=False)
    state = {k.replace('Wqkv.', 'in_proj_'): v for k,v in state.items()}
    expected = model.state_dict()
    active = ('encoder.', 'value_encoder.', 'transformer_encoder.')
    missing = [k for k,v in expected.items() if k.startswith(active) and
               (k not in state or state[k].shape != v.shape)]
    assert not missing, f'Unloaded inference weights: {missing}'
    matched = {k:v for k,v in state.items() if k in expected and v.shape == expected[k].shape}
    missing_unused = model.load_state_dict(matched, strict=False)
    print('[weights] all inference encoder weights loaded;', missing_unused, flush=True)
    stats = pd.read_csv(weights/'all_dict_mean_std.csv', index_col=0)
    collator = DataCollator(do_padding=True, pad_token_id=vocab[cfg['pad_token']], pad_value=cfg['pad_value'],
        do_mlm=False, do_binning=True, n_bins=51, max_length=1200, sampling=True, keep_first_n_tokens=1)
    return model.eval().requires_grad_(False).cuda(), vocab, cfg, stats, collator


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    OUT.mkdir(parents=True, exist_ok=True)
    model, vocab, cfg, gene_stats, collator = load_official()
    prepared = ROOT/'data/benchmarking_data/lungpf/prepared'
    protocol = json.loads((prepared/'protocol.json').read_text())
    frames, matrices, coverage = [], [], []
    for sample in protocol['samples']:
        directory = prepared/sample
        obs = pd.read_parquet(directory/'obs.parquet')
        assert len(obs.patient.unique()) == 1
        split = protocol['donor_split'][obs.patient.iloc[0]]
        if split == 'test':
            continue
        selected = np.sort(np.random.default_rng(42).choice(len(obs), min(400,len(obs)), replace=False))
        chosen = obs.iloc[selected].copy()
        chosen['split'] = split
        genes = np.load(directory/'genes.npy')
        keep = np.array([i for i,g in enumerate(genes) if g in vocab], dtype=np.int64)
        assert len(keep) > 0
        ids = np.array([vocab[g] for g in genes[keep]], dtype=np.int64)
        n = np.load(directory/'nuclear.npy', mmap_mode='r')
        c = np.load(directory/'cytoplasmic.npy', mmap_mode='r')
        # Match official per-slide normalization on unlabeled full slide counts.
        total = np.zeros(len(keep), dtype=np.float64)
        nonzero = np.zeros(len(keep), dtype=np.int64)
        for start in range(0,len(obs),4096):
            values = n[start:start+4096,keep].astype(np.float32)+c[start:start+4096,keep].astype(np.float32)
            total += values.sum(0, dtype=np.float64)
            nonzero += (values>0).sum(0)
        slide_mean = float(total.sum()/nonzero.sum())
        means = np.array([float(gene_stats.loc[g,'mean']) if g in gene_stats.index else
                         total[j]/max(nonzero[j],1)/slide_mean for j,g in enumerate(ids)])
        assert np.isfinite(means).all() and (means[nonzero>0]>0).all()
        rows = []
        for row in selected:
            values = (n[row,keep].astype(np.float32)+c[row,keep].astype(np.float32))/slide_mean
            expressed = np.flatnonzero(values>0)
            rows.append(dict(id=len(rows), genes=torch.from_numpy(np.r_[vocab['<cls>'],ids[expressed]]).long(),
                expressions=torch.from_numpy(np.r_[cfg['pad_value'],values[expressed]/means[expressed]]).float()))
        extracted = []
        for start in range(0,len(rows),32):
            batch = collator(rows[start:start+32])
            gene = batch['gene'].cuda()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                hidden = model._encode(gene, batch['expr'].cuda(), gene.eq(vocab[cfg['pad_token']]))
                value = hidden[:,0].float()
                value = value/value.norm(dim=1,keepdim=True).clamp_min(1e-12)
            extracted.append(value.cpu().numpy())
        x = np.concatenate(extracted)
        assert np.isfinite(x).all()
        np.save(OUT/f'{sample}.npy',x)
        matrices.append(x)
        frames.append(chosen)
        coverage.append(dict(sample=sample, matched_genes=len(keep), total_genes=len(genes),
                             slide_mean=slide_mean, fallback_gene_means=int(sum(g not in gene_stats.index for g in ids))))
        print(f'[embed] {sample} {split} n={len(x)} genes={len(keep)}/{len(genes)}',flush=True)
    obs = pd.concat(frames,ignore_index=True)
    x = np.concatenate(matrices)
    cohort = hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest()
    baseline = json.loads((ROOT/'outputs/cytoplasm_mean_scgpt_spatial_adamw_v1/downstream_validation/step_00000000/metrics.json').read_text())
    assert cohort == baseline['cohort_sha256'], 'Not the same validation cohort'
    obs.to_parquet(OUT/'cells.parquet',index=False)
    np.save(OUT/'embeddings.npy',x)
    del model
    torch.cuda.empty_cache()
    results=[]
    for task,column in [('cell_type','final_CT'),('region','region_label')]:
        labels=obs[column]
        usable=labels.notna() & ~labels.astype(str).isin(['','nan','None'])
        train=usable & obs['split'].eq('train'); val=usable & obs['split'].eq('val')
        classes=np.unique(labels[train].astype(str)); lookup={c:i for i,c in enumerate(classes)}
        scaler=StandardScaler().fit(x[train])
        tx=torch.tensor(scaler.transform(x[train]),device='cuda',dtype=torch.float32)
        vx=torch.tensor(scaler.transform(x[val]),device='cuda',dtype=torch.float32)
        y=torch.tensor([lookup[c] for c in labels[train].astype(str)],device='cuda')
        true=labels[val].astype(str).to_numpy();grid=[];best_score=-1.;best_pred=None
        for C in [.01,.1,1.,10.]:
            W,b,info=fit_linear(tx,y,C)
            pred=predict(vx,W,b,classes)
            f1=f1_score(true,pred,labels=np.unique(true),average='macro',zero_division=0)
            donors=obs.loc[val,'patient'].to_numpy()
            donor_scores={str(d):f1_score(true[donors==d],pred[donors==d],
                labels=np.unique(true[donors==d]),average='macro',zero_division=0) for d in np.unique(donors)}
            grid.append(dict(C=C,val_macro_f1=float(f1),val_balanced_accuracy=float(balanced_accuracy_score(true,pred)),
                             per_donor_macro_f1=donor_scores,solver=info))
            if f1>best_score: best_score=float(f1);best_pred=pred.copy()
        best=max(grid,key=lambda r:r['val_macro_f1'])
        prediction_frame=obs.loc[val,['full_cell_id','patient']].copy()
        prediction_frame['label']=true;prediction_frame['prediction']=best_pred
        prediction_frame.to_parquet(OUT/f'{task}_val_predictions.parquet',index=False)
        results.append(dict(task=task,train_cells=int(train.sum()),val_cells=int(val.sum()),best=best,grid=grid))
        print(f'[probe] {task}: {best}',flush=True)
    report=dict(model='Official scGPT-spatial V1',source_commit='f9442d777ba47cd3dcac8253833a233ff5bf7262',
        weights_md5='470d40fb77535afc2b4e7cffc0d665ef',cohort_sha256=cohort,results=results,gene_coverage=coverage,
        test_donors_used=False,pooling='CLS L2 normalized',input='nuclear+cytoplasmic raw counts',
        binning='Official slide/gene mean normalization then per-cell 51-bin collator',
        inference='Official standard PyTorch encoder, legacy Flash imports omitted; all encoder weights verified',
        context='Individual-cell official zero-shot encoder; no inference-time cell-cell attention')
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2))
    print('[done] '+str(OUT/'metrics.json'),flush=True)


if __name__=='__main__':
    main()
