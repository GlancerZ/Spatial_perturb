#!/usr/bin/env python3
"""Show supervised and unsupervised Visium tasks separately, on audited cells."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    a = p.parse_args()
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") or host != os.environ.get("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Run inside a verified Slurm compute-node step")
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = a.results.resolve()
    qa = json.loads((root/"QA.json").read_text())
    if qa["status"] != "VISIUM_HD_CRC_BENCHMARK_QA_PASSED":
        raise ValueError("Independent result QA did not pass")
    csv = root/"evaluation/primary_metrics.csv"
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    audit = json.loads((root/"evaluation/audit.json").read_text())
    if audit["output_sha256"]["primary_metrics.csv"] != digest:
        raise ValueError("Primary metrics changed since evaluation")
    table = pd.read_csv(csv)
    models = ["v12_scgpt_total_cls","v12_local","v12_spatial","v12_fused","v12_concat","scgpt_spatial_v1_cls"]
    names = ["scGPT total CLS","V12 local","V12 spatial","V12 fused","V12 concat (1536d)","scGPT-spatial V1"]
    colors = ["#949BA5","#5B96BF","#84B6D5","#2166A0","#4A83AE","#CC8B28"]
    fig, axes = plt.subplots(2,2,figsize=(13.8,8.6))
    panels = []
    for row, (task, title) in enumerate([("expert_cell_type","Cell type: expert broad classes"),("annotated_region","Region: marker-assisted labels")]):
        subset = table.loc[table.task == task].set_index("readout").loc[models]
        if subset.ordered_test_cell_ids_sha256.nunique() != 1:
            raise ValueError("Panel models used different test cells")
        for column, (metric, label) in enumerate([("test_macro_f1","Linear-probe macro-F1"),("test_ari_k_known","KMeans ARI (known K)")]):
            ax = axes[row,column]
            values = subset[metric].to_numpy()
            ax.barh(range(len(models)),values,color=colors,height=.62)
            ax.set_yticks(range(len(models)),names)
            ax.invert_yaxis()
            ax.set(xlim=(0,1),xlabel=label,title=f"{title}\nTest cells: {int(subset.test_cells.iloc[0]):,}")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="x",alpha=.15)
            ax.set_axisbelow(True)
            for i, v in enumerate(values):
                ax.text(v+.012,i,f"{v:.3f}",va="center",fontsize=9)
            panels.append({"task":task,"metric":metric,"readouts":models,"test_cells":int(subset.test_cells.iloc[0]),"ordered_test_cell_ids_sha256":subset.ordered_test_cell_ids_sha256.iloc[0]})
    fig.suptitle("Visium HD CRC: V12 vs scGPT and scGPT-spatial",fontsize=15,y=.985)
    fig.text(.5,.018,"Frozen encoders; one spatial-block split; no uncertainty interval. CRC source overlaps scGPT-spatial pretraining.\nRegion macro-F1 uses 5 fixed classes; test contains 4. Region labels use H&E plus markers.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95),w_pad=3,h_pad=2)
    out = root/"summary"
    out.mkdir(exist_ok=False)
    fig.savefig(out/"visium_comparison.png",dpi=190)
    fig.savefig(out/"visium_comparison.pdf")
    plt.close(fig)
    contract={"status":"CHART_BUILT_FROM_QA_PASSED_METRICS","source_sha256":digest,"panels":panels,
              "axis_limits":[0,1],"ci":"none; single split","source_overlap":audit["source_overlap"],
              "slurm_job_id":os.environ["SLURM_JOB_ID"],"hostname":host}
    (out/"chart_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print(json.dumps({"status":contract["status"],"output_dir":str(out)}),flush=True)


if __name__ == "__main__":
    main()
