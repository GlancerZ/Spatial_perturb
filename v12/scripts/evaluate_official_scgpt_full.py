"""All-cell official scGPT-spatial baseline using the historical test protocol."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from evaluate_official_scgpt import ROOT, load_official
import probe_lungpf_embeddings as probe

OUT=ROOT/'outputs/scgpt_official_v1_full'
HISTORICAL=Path('/scratch/glancerz/spatial-pretrain-v3/outputs/xenium_step1500_benchmark')


def main():
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(42)
    np.random.seed(42)
    OUT.mkdir(parents=True,exist_ok=True)
    model,vocab,cfg,gene_stats,collator=load_official()
    prepared=ROOT/'data/benchmarking_data/lungpf/prepared'
    protocol=json.loads((prepared/'protocol.json').read_text())
    total_cells=0
    for sample in sorted(protocol['samples']):
        directory=prepared/sample
        obs=pd.read_parquet(directory/'obs.parquet')
        columns=['full_cell_id','sample','patient','split','final_CT','region_label']
        historical=pd.read_parquet(HISTORICAL/'embeddings'/sample/'cells.parquet')
        pd.testing.assert_frame_equal(obs[columns].reset_index(drop=True),historical.reset_index(drop=True),check_dtype=False)
        assert len(obs.patient.unique())==1
        assert obs['split'].eq(protocol['donor_split'][obs.patient.iloc[0]]).all()
        total_cells+=len(obs)
        dest=OUT/'embeddings'/sample
        dest.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256('\n'.join(obs.full_cell_id.astype(str)).encode()).hexdigest()
        signature=dict(cohort_sha256=digest,method='official_v1_cls_51bins_full_v1',batch_size=128,seed=42)
        meta=dest/'scgpt_spatial.json'
        if meta.exists() and (dest/'scgpt_spatial.npy').exists():
            saved=json.loads(meta.read_text())
            assert saved['signature']==signature
            shape=np.load(dest/'scgpt_spatial.npy',mmap_mode='r').shape
            assert shape==(len(obs),cfg['embsize'])
            print(f'[cached] {sample} cells={len(obs)}',flush=True)
            continue
        # Each slide has its own deterministic RNG, making cache/resume invariant.
        torch.manual_seed(42);np.random.seed(42)
        genes=np.load(directory/'genes.npy')
        keep=np.array([i for i,g in enumerate(genes) if g in vocab],dtype=np.int64)
        assert len(keep)>0
        ids=np.array([vocab[g] for g in genes[keep]],dtype=np.int64)
        nuclear=np.load(directory/'nuclear.npy',mmap_mode='r')
        cyto=np.load(directory/'cytoplasmic.npy',mmap_mode='r')
        sums=np.zeros(len(keep),dtype=np.float64);nonzero=np.zeros(len(keep),dtype=np.int64)
        for start in range(0,len(obs),4096):
            raw=nuclear[start:start+4096,keep].astype(np.float32)+cyto[start:start+4096,keep].astype(np.float32)
            sums+=raw.sum(0,dtype=np.float64);nonzero+=(raw>0).sum(0)
        slide_mean=float(sums.sum()/nonzero.sum())
        means=np.array([float(gene_stats.loc[g,'mean']) if g in gene_stats.index else
                        sums[j]/max(nonzero[j],1)/slide_mean for j,g in enumerate(ids)])
        assert np.isfinite(means).all() and (means[nonzero>0]>0).all()
        staging=dest/'scgpt_spatial.partial.npy'
        embeddings=np.lib.format.open_memmap(staging,mode='w+',dtype=np.float32,shape=(len(obs),cfg['embsize']))
        began=time.monotonic()
        for start in range(0,len(obs),128):
            stop=min(start+128,len(obs));rows=[]
            for row in range(start,stop):
                values=(nuclear[row,keep].astype(np.float32)+cyto[row,keep].astype(np.float32))/slide_mean
                positive=np.flatnonzero(values>0)
                rows.append(dict(id=row,genes=torch.from_numpy(np.r_[vocab['<cls>'],ids[positive]]).long(),
                    expressions=torch.from_numpy(np.r_[cfg['pad_value'],values[positive]/means[positive]]).float()))
            batch=collator(rows);gene=batch['gene'].cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                hidden=model._encode(gene,batch['expr'].cuda(),gene.eq(vocab[cfg['pad_token']]))
                value=hidden[:,0].float()
                value=value/value.norm(dim=1,keepdim=True).clamp_min(1e-12)
            array=value.cpu().numpy()
            assert np.isfinite(array).all()
            embeddings[start:stop]=array
            if start==0 or stop%8192==0 or stop==len(obs):
                print(f'[full-embed] {sample} {stop}/{len(obs)} elapsed={time.monotonic()-began:.1f}s',flush=True)
        embeddings.flush();del embeddings
        staging.replace(dest/'scgpt_spatial.npy')
        obs[columns].to_parquet(dest/'cells.parquet',index=False)
        meta.write_text(json.dumps(dict(signature=signature,cells=len(obs),matched_genes=len(keep),
            total_genes=len(genes),slide_mean=slide_mean,dimensions=cfg['embsize']),indent=2))
    assert total_cells==455379,total_cells
    del model
    torch.cuda.empty_cache()
    probe.OUT=OUT
    probe.main(models=['scgpt_spatial'])
    results=json.loads((OUT/'metrics.json').read_text())
    reference=[r for r in json.loads((HISTORICAL/'metrics.json').read_text()) if r['model']=='spatialformer']
    for row in results:
        old=next(r for r in reference if r['task']==row['task'])
        for key in ['train_cells','val_cells','test_cells','train_classes','test_classes','unseen_test_classes',
                    'train_donors','val_donors','test_donors']:
            assert row[key]==old[key],(key,row[key],old[key])
    lines=['# scGPT-spatial 与 SpatialFormer：完整LungPF对照','',
           '455379细胞，16供体；10/3/3供体划分。细胞ID及标签与历史基线逐行核对一致。',
           '冻结编码器；同一分类器、同一C网格，验证集选C，以下为测试集成绩。','',
           '| 模型 | 任务 | Test macro-F1 | Test balanced accuracy |','|---|---|---:|---:|']
    for row in reference+results:
        lines.append(f"| {row['model']} | {row['task']} | {row['macro_f1']:.6f} | {row['balanced_accuracy']:.6f} |")
    lines+=['','区域任务保留训练未见的测试类别；编码器预训练语料重叠未完全排除。',
            'scGPT-spatial使用官方V1权重、整细胞原始表达、官方slide/gene均值归一化、51箱分箱与CLS表征。',
            '本报告未重新选择模型检查点，未使用test标签调参。历史test已在其他实验中反复查看。']
    (OUT/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
    print('[full-done] '+str(OUT/'REPORT.zh.md'),flush=True)


if __name__=='__main__':
    main()
