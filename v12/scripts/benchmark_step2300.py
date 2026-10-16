"""Repeat the fixed benchmark at step2300; reuse unchanged official baseline."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

STEP=int(os.environ.get('BENCHMARK_STEP', '2300'))
OLD=Path('outputs/xenium_step1500_benchmark')
OUT=Path(os.environ.get('BENCHMARK_OUTPUT_DIR', f'outputs/xenium_step{STEP}_benchmark'))
CHECKPOINT=Path(os.environ.get('BENCHMARK_CHECKPOINT', f'outputs/xenium_compartment_ready48/checkpoints/step_{STEP:08d}.pt'))
EXTRACTION_GPUS=int(os.environ.get('BENCHMARK_EXTRACTION_GPUS', '2'))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False,mmap=True)
    assert state['step']==STEP
    print('CHECKPOINT_VERIFIED',state['step'],flush=True)
    del state
    # These files describe the unchanged baseline, not newly fitted models.
    for p in OLD.glob('spatialformer_*'):
        if p.is_file() and 'preliminary' not in p.name: shutil.copy2(p,OUT/p.name)
    for sample in sorted((OLD/'embeddings').iterdir()):
        dest=OUT/'embeddings'/sample.name;dest.mkdir(parents=True,exist_ok=True)
        for name in ['spatialformer.npy','spatialformer.json','cells.parquet']:
            target=dest/name
            if not target.exists(): os.link(sample/name,target)
    workers=[]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1').split(',')
    assert 1 <= EXTRACTION_GPUS <= len(visible)
    for rank in range(EXTRACTION_GPUS):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible[rank])
        log=(OUT/f'extraction_gpu{rank}.log').open('w')
        cmd=[sys.executable,'-u','scripts/extract_lungpf_embeddings.py','--model','ours',
             '--checkpoint',str(CHECKPOINT),'--step',str(STEP),'--output-dir',str(OUT),
             '--shard',str(rank),'--num-shards',str(EXTRACTION_GPUS),'--batch-size','4']
        workers.append((subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    statuses=[]
    for proc,log in workers: statuses.append(proc.wait());log.close()
    assert statuses==[0]*EXTRACTION_GPUS,statuses
    assert np.array_equal(np.load(OLD/'pretraining_median_bin_edges.npy'),np.load(OUT/'pretraining_median_bin_edges.npy'))
    assert json.loads((OLD/'our_gene_vocabulary.json').read_text())==json.loads((OUT/'our_gene_vocabulary.json').read_text())
    for directory in sorted((OUT/'embeddings').iterdir()):
        assert pd.read_parquet(directory/'cells.parquet').equals(pd.read_parquet(OLD/'embeddings'/directory.name/'cells.parquet'))
    print('SAME_CELLS_SPLITS_LABELS_VOCABULARY_AND_BINS_VERIFIED',flush=True)
    import probe_lungpf_embeddings as probe
    probe.OUT=OUT
    probe.main()
    old=json.loads((OLD/'metrics.json').read_text())
    new=json.loads((OUT/'metrics.json').read_text())
    for row in new:
        assert all(x['converged_gradient'] for x in row['tuning'])
        predictions=pd.read_parquet(OUT/f'{row["model"]}_{row["task"]}_predictions.parquet')
        previous=pd.read_parquet(OLD/f'{row["model"]}_{row["task"]}_predictions.parquet')
        assert predictions[['full_cell_id','patient','label']].equals(previous[['full_cell_id','patient','label']])
    comparisons=[]
    for task in ['cell_type','region']:
        before=next(r for r in old if r['model']=='ours' and r['task']==task)
        after=next(r for r in new if r['model']=='ours' and r['task']==task)
        base=next(r for r in new if r['model']=='spatialformer' and r['task']==task)
        for metric in (['macro_f1','balanced_accuracy'] if task=='cell_type' else ['macro_f1']):
            comparisons.append(dict(task=task,metric=metric,reference1500=before[metric],**{f'step{STEP}':after[metric]},
                difference=after[metric]-before[metric],spatialformer=base[metric],test_cells=after['test_cells']))
    table=pd.DataFrame(comparisons);table.to_csv(OUT/'checkpoint_comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.5))
    ys=np.array([2,1,0])
    for key,offset,color,hatch,label in [('reference1500',.23,'#92bbd1',None,'Previous run · step1500'),
        (f'step{STEP}',0,'#31688e',None,f'Ours · step{STEP}'),('spatialformer',-.23,'white','///','SpatialFormer · single71')]:
        ax.barh(ys+offset,table[key],height=.21,color=color,edgecolor='#31688e',hatch=hatch,label=label)
        for y,v in zip(ys+offset,table[key]):ax.text(v+.01,y,f'{v:.3f}',va='center',fontsize=10)
    ax.set_yticks(ys,['Cell type\nmacro-F1','Cell type\nbalanced accuracy','Pathology region\nmacro-F1'])
    ax.set_xlim(0,1);ax.set_xlabel('Score (0–1)');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,frameon=False,fontsize=10)
    fig.suptitle(f'Frozen embedding benchmark: step1500 vs step{STEP}',x=.02,ha='left',fontsize=16,fontweight='bold')
    fig.text(.02,.89,'Identical donor split, cells, preprocessing and linear-probe search; encoders frozen.',fontsize=11)
    fig.text(.02,.04,'LungPF TMA5 · 3 held-out donors · Cell type n=64,611 · Pathology n=14,313\nRepeated test-set evaluation; results are exploratory. Official baseline reused unchanged.',fontsize=9)
    fig.subplots_adjust(left=.23,right=.96,top=.75,bottom=.22)
    fig.savefig(OUT/'checkpoint_comparison.png',dpi=180);fig.savefig(OUT/'checkpoint_comparison.pdf')
    lines=[f'**{STEP} step 冻结 embedding benchmark**','',f'| 指标 | 我们 1500 | 我们 {STEP} | 变化 | SpatialFormer |','|---|---:|---:|---:|---:|']
    labels=['细胞类型 macro-F1','细胞类型 balanced accuracy','独立病理区域 macro-F1']
    for label,r in zip(labels,comparisons):
        lines.append(f'| {label} | {r["reference1500"]:.4f} | {r[f"step{STEP}"]:.4f} | {r["difference"]:+.4f} | {r["spatialformer"]:.4f} |')
    lines+=['','![对比曲线](checkpoint_comparison.png)','',
        f'仅更换我们的编码器权重为精确的 step{STEP}。沿用相同 16 个供体的 10/3/3 划分、相同细胞和标签、相同基因词表、离散阈值、采样种子、空间分组、256 维均值 pooling。与旧版数据逐项核验一致。',
        '编码器冻结，重新训练同一种标准化、类别平衡、L2 正则的多项逻辑回归；C 网格仍为 0.01/0.1/1/10，仅按验证集 macro-F1 选择。所有拟合均满足相同训练目标梯度阈值。SpatialFormer 权重、embedding、分类器和预测均未改变，直接复用先前结果。',
        '细胞类型测试 64,611 个细胞（47 类）；区域测试 14,313 个细胞（16 类）。沿用全部测试类别，包括训练中未出现的 Emphysema。只有 3 个测试供体；我们的基因覆盖仍为 295/343，SpatialFormer 为 343/343；其预训练队列重叠仍未完全排除。',
        '这次按要求重复查看同一测试集，应视为探索性进展比较，不能当成独立的新测试，也不能据此宣称模型已收敛或确定架构优劣。原有方法与其他局限见 outputs/xenium_step1500_benchmark/REPORT.zh.md。',
        'checkpoint 保存策略已修复：后续每 100 step 及 epoch 结束归档完整模型、优化器和 RNG 状态，滚动 resume.pt 仍用于最新恢复。2000 step 原始权重此前已覆盖，本次没有使用或重建该权重。','']
    (OUT/'REPORT.zh.md').write_text('\n'.join(lines))
    (OUT/'comparison_qa.json').write_text(json.dumps(dict(step=STEP,checkpoint=str(CHECKPOINT.resolve()),output_directory=str(OUT.resolve()),identical_cells=True,identical_preprocessing=True,
        baseline_reused=True,all_classifiers_converged=True,comparisons=comparisons),indent=2))
    print(table.to_string(index=False),flush=True)

if __name__=='__main__':main()
