"""Spatial truth/prediction maps on identical held-out annotated cells."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path('outputs/xenium_step1500_benchmark')
DATA=Path('data/benchmarking_data/lungpf/prepared')
ours=pd.read_parquet(OUT/'ours_region_predictions.parquet')
sf=pd.read_parquet(OUT/'spatialformer_region_predictions.parquet')
assert ours[['full_cell_id','patient','label']].equals(sf[['full_cell_id','patient','label']])
merged=ours.rename(columns={'prediction':'ours'}).merge(sf[['full_cell_id','prediction']].rename(columns={'prediction':'spatialformer'}),on='full_cell_id',validate='one_to_one')
classes=sorted(set(merged.label)|set(merged.ours)|set(merged.spatialformer))
# Same categorical mapping in every panel and every donor; no ranking by outcome.
colors=[c for i,c in enumerate(plt.get_cmap('tab20').colors) if i not in (14,15)]+list(plt.get_cmap('tab20b').colors)
palette=dict(zip(classes,colors))
(OUT/'region_map_palette.json').write_text(json.dumps({k:matplotlib.colors.to_hex(v) for k,v in palette.items()},indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
fig,axes=plt.subplots(3,3,figsize=(15,15))
qa=[]
for row,patient in enumerate(sorted(merged.patient.unique())):
    sample=next(s for s in json.loads((DATA/'protocol.json').read_text())['samples'] if pd.read_parquet(DATA/s/'obs.parquet',columns=['patient']).patient.iloc[0]==patient)
    cells=pd.read_parquet(DATA/sample/'obs.parquet')
    pred=merged.loc[merged.patient.eq(patient)].merge(cells[['full_cell_id','x_centroid','y_centroid']],on='full_cell_id',validate='one_to_one')
    assert len(pred)==int(merged.patient.eq(patient).sum())
    assert np.isfinite(pred[['x_centroid','y_centroid']]).all().all()
    for col,(key,title) in enumerate([('label','Reference pathology'),('ours','Ours · step 1500'),('spatialformer','SpatialFormer · single71')]):
        ax=axes[row,col]
        ax.scatter(cells.x_centroid,cells.y_centroid,s=1,c='#e4e4e4',linewidths=0,rasterized=True)
        ax.scatter(pred.x_centroid,pred.y_centroid,s=3,c=[palette[v] for v in pred[key]],linewidths=0,rasterized=True)
        ax.set_aspect('equal');ax.invert_yaxis()
        ax.set_title(f'{sample} | {title}',fontsize=11)
        ax.set_xlabel('X (µm)');ax.set_ylabel('Y (µm)')
        ax.spines[['top','right']].set_visible(False)
    qa.append(dict(sample=sample,total_cells=len(cells),annotated_cells=len(pred)))
fig.suptitle('Held-out pathology regions: reference and predictions',x=.04,ha='left',fontsize=18,fontweight='bold')
fig.text(.04,.95,'Same cells, coordinates and region colors across panels; gray = cells without an unambiguous reference label.',fontsize=11)
handles=[Line2D([],[],marker='o',linestyle='',color=palette[c],markersize=6,label=c) for c in classes]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,frameon=False,fontsize=9)
fig.subplots_adjust(left=.06,right=.97,top=.91,bottom=.20,wspace=.27,hspace=.33)
fig.savefig(OUT/'region_spatial_comparison.png',dpi=200)
fig.savefig(OUT/'region_spatial_comparison.pdf')
# Export each donor at a readable independent resolution, preserving the shared palette.
for row,info in enumerate(qa):
    patient=sorted(merged.patient.unique())[row]
    sample=info['sample'];cells=pd.read_parquet(DATA/sample/'obs.parquet')
    pred=merged.loc[merged.patient.eq(patient)].merge(cells[['full_cell_id','x_centroid','y_centroid']],on='full_cell_id',validate='one_to_one')
    f,axs=plt.subplots(1,3,figsize=(15,7))
    for ax,(key,title) in zip(axs,[('label','Reference pathology'),('ours','Ours · step 1500'),('spatialformer','SpatialFormer · single71')]):
        ax.scatter(cells.x_centroid,cells.y_centroid,s=2,c='#e4e4e4',linewidths=0,rasterized=True)
        ax.scatter(pred.x_centroid,pred.y_centroid,s=5,c=[palette[v] for v in pred[key]],linewidths=0,rasterized=True)
        ax.set_aspect('equal');ax.invert_yaxis();ax.set_title(title)
        ax.set_xlabel('X (µm)');ax.set_ylabel('Y (µm)');ax.spines[['top','right']].set_visible(False)
    f.suptitle(f'{sample}: independently annotated tissue regions',x=.04,ha='left',fontsize=16,fontweight='bold')
    f.text(.04,.91,f'{len(pred):,} matched annotated cells / {len(cells):,} total cells. Gray: unannotated background.',fontsize=11)
    present=set(pred.label)|set(pred.ours)|set(pred.spatialformer)
    hs=[h for h in handles if h.get_label() in present]
    f.legend(handles=hs,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,frameon=False,fontsize=9)
    f.subplots_adjust(left=.06,right=.98,top=.82,bottom=.30,wspace=.25)
    f.savefig(OUT/f'region_spatial_{sample}.png',dpi=200)
    f.savefig(OUT/f'region_spatial_{sample}.pdf')
    plt.close(f)
(OUT/'region_maps_qa.json').write_text(json.dumps(qa,indent=2))
print(json.dumps(qa),flush=True)
