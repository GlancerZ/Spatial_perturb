"""Audit the comparable checkpoint cohort and report exploratory validation trends."""
import json
from pathlib import Path

root=Path('outputs/cytoplasm_mean_full_muon_v1/downstream_validation')
steps=[0,200,400,700]
rows=[]
for step in steps:
    path=root/f'step_{step:08d}'/'metrics.json'
    if path.exists():
        row=json.loads(path.read_text())
        assert row['step']==step and not row['test_donors_used']
        assert all(r['best']['solver']['converged_gradient'] for r in row['results'])
        rows.append(row)
if not rows:
    raise SystemExit('No completed checkpoint metrics yet')
assert len({r['cohort_sha256'] for r in rows})==1,'Checkpoint cohorts differ'
assert len({r['context'] for r in rows})==1,'Checkpoint contexts differ'
assert len({r['pooling'] for r in rows})==1,'Checkpoint pooling differs'
lines=['# 全量Muon模型检查点：固定子集下游验证','',
       '模型使用完整48切片训练；下游仅为固定4000训练中心和1200验证中心的快速趋势检查。',
       '每个中心使用256细胞KNN上下文。测试供体不参与。小幅波动不证明显著改善。','',
       '| Step | Cell type macro-F1 | Region macro-F1 |','|---|---:|---:|']
for row in rows:
    scores={r['task']:r['best']['val_macro_f1'] for r in row['results']}
    lines.append(f"| {row['step']} | {scores['cell_type']:.6f} | {scores['region']:.6f} |")
findings=[]
if rows[0]['step']==0 and len(rows)>1:
    for task in ['cell_type','region']:
        series=[next(r for r in row['results'] if r['task']==task)['best'] for row in rows]
        start,end=series[0],series[-1]
        delta=end['val_macro_f1']-start['val_macro_f1']
        donor_deltas={d:end['per_donor_macro_f1'][d]-v for d,v in start['per_donor_macro_f1'].items()}
        positive=sum(value>0 for value in donor_deltas.values())
        increases=sum(b['val_macro_f1']>a['val_macro_f1'] for a,b in zip(series,series[1:]))
        findings.append(dict(task=task,endpoint_delta=delta,positive_donors=positive,
            donor_deltas=donor_deltas,increasing_transitions=increases,transitions=len(series)-1))
        lines.extend(['',f'**{task}**：step0至{rows[-1]["step"]}变化 {delta:+.6f}；'
            f'{positive}/{len(donor_deltas)}供体改善；相邻已测点{increases}/{len(series)-1}次上升。'])
        if delta<=0 or positive<2:
            lines.append('尚无跨供体一致改善的证据，应先诊断而非仅依据重建loss延长训练。')
        else:
            lines.append('有方向性改善信号；仍需补齐中间点、扩大验证样本并复验，不宣称统计显著。')
complete=len(rows)==len(steps)
lines.extend(['','检查点序列已齐全。' if complete else '检查点序列尚未齐全，不判断稳定趋势。',
    '分类器C在每个检查点的同一验证集选择；这是探索性验证，不是独立测试成绩。'])
(root/'trend.json').write_text(json.dumps(dict(complete=complete,steps=[r['step'] for r in rows],findings=findings),indent=2))
(root/'REPORT.zh.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines),flush=True)
