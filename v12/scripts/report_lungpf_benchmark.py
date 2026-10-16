"""Produce a source-backed static scientific comparison and Chinese report."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score

OUT=Path('outputs/xenium_step1500_benchmark')
DATA=Path('data/benchmarking_data/lungpf/prepared')

def main():
    metrics=json.loads((OUT/'metrics.json').read_text())
    lookup={(r['model'],r['task']):r for r in metrics}
    protocol=json.loads((DATA/'protocol.json').read_text())
    qa=json.loads((DATA/'qa.json').read_text())
    total=sum(r['cells'] for r in qa)
    regions=sum(sum(r['region_counts'].values()) for r in qa)
    donor=[]; closed_set=[]
    for task in ['cell_type','region']:
        left=pd.read_parquet(OUT/f'ours_{task}_predictions.parquet')
        right=pd.read_parquet(OUT/f'spatialformer_{task}_predictions.parquet')
        assert left[['full_cell_id','patient','label']].equals(right[['full_cell_id','patient','label']])
        for model,frame in [('ours',left),('spatialformer',right)]:
            expected=lookup[(model,task)]
            assert len(frame)==expected['test_cells'] and frame.full_cell_id.is_unique
            actual=f1_score(frame.label,frame.prediction,labels=frame.label.unique(),average='macro',zero_division=0)
            assert np.isclose(actual,expected['macro_f1'],atol=1e-12)
            assert np.isclose(balanced_accuracy_score(frame.label,frame.prediction),expected['balanced_accuracy'],atol=1e-12)
            assert all(t['converged_gradient'] for t in expected['tuning'])
            classes=np.load(OUT/f'{model}_{task}_classifier.npz')['classes']
            known=frame.loc[frame.label.isin(classes)]
            closed_set.append(dict(model=model,task=task,cells=len(known),classes=known.label.nunique(),
                macro_f1=f1_score(known.label,known.prediction,labels=known.label.unique(),average='macro',zero_division=0),
                balanced_accuracy=balanced_accuracy_score(known.label,known.prediction)))
            for patient,part in frame.groupby('patient'):
                donor.append(dict(task=task,model=model,patient=patient,cells=len(part),
                    macro_f1=f1_score(part.label,part.prediction,labels=part.label.unique(),average='macro',zero_division=0),
                    balanced_accuracy=balanced_accuracy_score(part.label,part.prediction)))
    pd.DataFrame(donor).to_csv(OUT/'per_donor_metrics.csv',index=False)
    pd.DataFrame(closed_set).to_csv(OUT/'known_label_supplement.csv',index=False)
    contract=dict(question='How do frozen step1500 and official SpatialFormer embeddings compare under identical linear probes?',
        family='comparison',variant='three groups of paired horizontal bars',renderer='Matplotlib standalone scientific PNG/PDF',
        observations=6,grain='model × held-out task metric',denominator='identical donor-held-out cells per task',
        scale=[0,1],palette='single blue root; filled ours, open hatched official baseline',
        export=['comparison.png','comparison.pdf'],qa='inspect PNG for labels, honest axes, exact values and cohort caption')
    (OUT/'chart_contract.json').write_text(json.dumps(contract,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10,5.5))
    specs=[('cell_type','macro_f1'),('cell_type','balanced_accuracy'),('region','macro_f1')]
    for model,offset,color,hatch,label in [('ours',.16,'#31688e',None,'Ours · step 1500'),
            ('spatialformer',-.16,'white','///','SpatialFormer · single71')]:
        ys=np.array([2,1,0])+offset
        values=[lookup[(model,task)][metric] for task,metric in specs]
        ax.barh(ys,values,color=color,edgecolor='#31688e',height=.27,hatch=hatch,label=label)
        for y,value in zip(ys,values): ax.text(value+.015,y,f'{value:.3f}',va='center',fontsize=11)
    ax.set_yticks([2,1,0],['Cell type\nmacro-F1','Cell type\nbalanced accuracy','Pathology region\nmacro-F1'])
    ax.set_xlim(0,1);ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('Score (0–1)')
    ax.set_axisbelow(True);ax.grid(axis='x',color='#e8e8e8',linewidth=.7)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2,frameon=False)
    ct=lookup[('ours','cell_type')]; region=lookup[('ours','region')]
    fig.suptitle('Frozen embedding benchmark · LungPF TMA5',x=.02,ha='left',fontsize=16,fontweight='bold')
    fig.text(.02,.90,f'3 held-out donors | Cell type n={ct["test_cells"]:,} | Independent pathology n={region["test_cells"]:,}',fontsize=11)
    fig.text(.02,.04,'Same donor split and balanced linear classifier; C selected on validation donors.\nGene coverage: ours 295/343; SpatialFormer 343/343. One split; results are preliminary.',fontsize=9,color='#444444')
    fig.subplots_adjust(left=.22,right=.96,top=.76,bottom=.22)
    fig.savefig(OUT/'comparison.png',dpi=180)
    fig.savefig(OUT/'comparison.pdf')
    plt.close(fig)
    lines=[f'**1500 step 冻结模型对比：LungPF TMA5**', '',
        f'评测已完成。使用 {len(protocol["samples"])} 个供体、{total:,} 个细胞；其中 {regions:,} 个细胞有无冲突的独立病理区域标签。训练/验证/测试按供体划分为 10/3/3，两个模型使用完全相同的细胞。1500 step 预训练验证 loss 为 0.8333。', '',
        '| 模型 | 细胞类型 macro-F1 | 细胞类型 balanced accuracy | 病理区域 macro-F1 |',
        '|---|---:|---:|---:|']
    for model,name in [('ours','我们的模型 · step1500'),('spatialformer','SpatialFormer · single71')]:
        c,r=lookup[(model,'cell_type')],lookup[(model,'region')]
        lines.append(f'| {name} | {c["macro_f1"]:.4f} | {c["balanced_accuracy"]:.4f} | {r["macro_f1"]:.4f} |')
    lines+=['','![模型对比](comparison.png)','', '**评测口径**','',
        '两个编码器完全冻结。我们的模型对最终层有效的核内及胞质 token 取均值，得到 256 维 embedding；SpatialFormer 使用官方推荐的 single71 checkpoint 和最终层 CLS，得到 512 维 embedding。官方 checkpoint 的 MD5 已核验，全部权重严格匹配；省略预测头后的 embedding 与官方提取接口进行了数值一致性校验。', '',
        '线性分类器统一为带类别平衡权重、L2 正则的多项逻辑回归。StandardScaler 只拟合训练供体；C ∈ {0.01, 0.1, 1, 10}，分别按验证集 macro-F1 选取。使用相同的 GPU 全批量 L-BFGS 求解器，并在独立小数据上与 scikit-learn LogisticRegression 核对预测概率。模型选择没有使用测试标签。没有微调编码器。', '',
        '细胞类型标签来自作者校正版 Seurat 的 final_CT；病理区域标签来自 H&E 人工标注的 Annotation_Type，未使用由表达聚类产生的 CNiche/TNiche。转录本按 qv≥20 筛选，去除控制探针，分别汇总 overlaps_nucleus=1/0 的核内/胞质计数。病理表旧样本名通过转录本中的 old_sample_name 映射，并逐细胞核验空间坐标。35 个区域标签冲突的细胞只从区域任务中排除。', '',
        f'细胞类型测试：{ct["test_cells"]:,} 个细胞、{ct["test_classes"]} 类；区域测试：{region["test_cells"]:,} 个细胞、{region["test_classes"]} 类。macro-F1 在测试集中实际出现的所有类别上平均，包含训练未见类别；balanced accuracy 为这些类别的平均召回率。', '',
        '**解释结果时需要保留的限制**', '',
        '这是训练 1500 step 后、一个固定供体划分上的初步比较，不能据此宣称架构优劣或复现论文数值。测试供体只有 3 个，逐供体分数另存 per_donor_metrics.csv。', '',
        '我们的模型约 4.12 亿参数，SpatialFormer 此 single71 权重约 6105 万参数；训练数据、训练时长、预训练目标和 embedding 维度不同。测试面板 343 个基因中，我们的词表覆盖 295 个，SpatialFormer 覆盖 343 个。SpatialFormer 按官方方式使用供体的 Healthy/Disease 条件 token。', '',
        '我们的细胞间上下文按空间递归划分为最多 256 个细胞的互不重叠局部组，每个细胞提取一次表示；该高效推理方式与逐细胞居中的 255 近邻上下文不同。每组首细胞使用训练时的中心角色，其余使用邻居角色。表达离散阈值固定为 48 个预训练切片阈值的逐位中位数，未在测试集拟合。', '',
        '测试使用 TMA5 的 16 个供体，排除了在旧批次已有切片的 TILD117。我们的 48 个预训练切片不包含该队列。SpatialFormer 的原始资料包含 LungPF 的其他切片；官方预训练数据完整成员关系的审计状态见 provenance.json，不能把未确认的排除关系写成已证明的零重叠。', '',
        '原始数据作者明确说明旧版本曾有基因名解码错误。本次使用校正后的转录本和标签；按 qv≥20 重建的核内计数与 Seurat nCount_RNA 并非逐细胞完全相等（相关性约 0.98–0.99），因此 Seurat 仅提供标签和坐标，两个模型的表达输入都来自同一套重建转录本。', '']
    for task,name in [('cell_type','细胞类型'),('region','病理区域')]:
        r=lookup[('ours',task)]
        lines.append(f'{name}训练未见的测试类别：{", ".join(r["unseen_test_classes"]) or "无"}；测试细胞标签训练覆盖率 {r["test_known_label_fraction"]:.2%}。')
    supplemental=[r for r in closed_set if r['task']=='region']
    lines+=['', '补充分析仅保留训练中出现过的区域标签，仍使用同一套已选好的分类器、不重新拟合：'+
        '；'.join(f'{r["model"]} macro-F1={r["macro_f1"]:.4f}（{r["cells"]:,} 个细胞、{r["classes"]} 类）' for r in supplemental)+'。主表仍保留所有测试类别。']
    lines+=['','**来源与复现文件**','',
        '- 论文：[SpatialFormer, Nature Computational Science](https://www.nature.com/articles/s43588-026-01016-7)。',
        '- 官方代码：[TerminatorJ/Spatialformer](https://github.com/TerminatorJ/Spatialformer)，固定 commit f02e7776e8a99c9abdc5fe5572a1a20369faf159。',
        '- 官方权重：[single71 checkpoint](https://figshare.com/articles/dataset/single_input_checkpoint_5k/31146238)。',
        '- 测试数据：[GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346)，输入下载清单在 data/benchmarking_data/lungpf/download_manifest.json。',
        '- 指标：metrics.csv / metrics.json；每类指标：*_per_class.csv；测试预测：*_predictions.parquet；线性分类器：*_classifier.npz。',
        '- 冻结权重：step1500_resume.pt（含优化器）和 model1500.pt（精简导出）；两者 step 均为 1500。',
        '- 脚本：scripts/prepare_lungpf_benchmark.py、extract_lungpf_embeddings.py、probe_lungpf_embeddings.py、report_lungpf_benchmark.py。',
        '- 核验：linear_solver_verification.json、data/benchmarking_data/lungpf/prepared/qa.json、各样本 embedding 旁的 JSON。','']
    (OUT/'REPORT.zh.md').write_text('\n'.join(lines))
    print('REPORT_COMPLETE',OUT/'REPORT.zh.md',flush=True)

if __name__=='__main__':main()
