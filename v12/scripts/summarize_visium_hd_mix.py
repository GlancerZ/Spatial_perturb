"""Compare matched CRC scores after equal-budget Xenium/HD continuation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True)
    parser.add_argument("--mixed-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or socket.gethostname().split(".")[0] != os.getenv("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("Comparison must run on the authorized compute node")
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = {}
    audits = {}
    schedules = {}
    keys = ["cohort", "task", "readout"]
    consistency = ["train_cells", "val_cells", "test_cells", "classes", "ordered_test_cell_ids_sha256"]
    score_columns = ["test_macro_f1", "test_ari_k_known", "test_nmi_k_known"]
    for name, folder in (("original",args.original), ("control",args.control), ("mixed",args.mixed)):
        if not (folder / "COMPLETE.json").is_file():
            raise ValueError(f"Missing completed benchmark: {name}")
        complete = json.loads((folder / "COMPLETE.json").read_text())
        if complete["audit_sha256"] != sha256(folder / "audit.json"):
            raise ValueError("Benchmark audit does not match the completion receipt")
        audit = json.loads((folder / "audit.json").read_text())
        for filename, digest in audit["output_sha256"].items():
            if sha256(folder / filename) != digest:
                raise ValueError(f"Benchmark artifact changed: {filename}")
        audits[name] = audit
        frame = pd.read_csv(folder / "metrics.csv").set_index(keys).sort_index()
        if not frame.index.is_unique:
            raise ValueError("Duplicate score definitions")
        if frames:
            if not frame.index.equals(frames["original"].index):
                raise ValueError("Readout/task/cohort sets changed")
            if not frame[consistency].equals(frames["original"][consistency]):
                raise ValueError("Probe cohorts or score cells changed")
        frames[name] = frame
    for name in ("control", "mixed"):
        audit = audits[name]
        if audit.get("split_method") != "external_original_frozen_membership_no_new_split_search":
            raise ValueError("Continuation evaluation did not load the original frozen split")
        if audit["source_sha256"]["frozen_score_cells"] != audits["original"]["output_sha256"]["frozen_score_cells.csv"]:
            raise ValueError("Continuation membership source differs from the original benchmark receipt")
        run = getattr(args, name+"_run")
        train_complete = json.loads((run / "TRAIN_COMPLETE.json").read_text())
        schedule = json.loads((run / "schedule.json").read_text())
        expected_hd = 16 if name == "mixed" else 0
        if (train_complete.get("step"), train_complete.get("continuation_step"),
                train_complete.get("budget_complete")) != (10734, 500, True):
            raise ValueError("One training arm did not complete its fixed budget")
        if (schedule["total_updates"], schedule["global_batch_size"],
                schedule["config"]["training"]["visium_centers_per_update"]) != (500, 64, expected_hd):
            raise ValueError("Training arm differs from the matched protocol")
        schedules[name] = schedule
        baseline_rows = frames["original"].index.get_level_values("readout").isin([
            "scgpt_spatial_v1_cls", "v12_scgpt_total_cls"])
        if not np.allclose(frames[name].loc[baseline_rows,score_columns].to_numpy(),
                           frames["original"].loc[baseline_rows,score_columns].to_numpy(),
                           rtol=0,atol=1e-10,equal_nan=True):
            raise ValueError("Frozen scGPT baselines changed across evaluations")
    if schedules["control"]["initialization"] != schedules["mixed"]["initialization"]:
        raise ValueError("Training arms initialized from different V12 weights")
    for field in ("data", "model", "v12", "binning", "loss", "lr"):
        if schedules["control"]["config"][field] != schedules["mixed"]["config"][field]:
            raise ValueError(f"Unmatched configuration: {field}")
    control_training = {k:v for k,v in schedules["control"]["config"]["training"].items()
                        if k != "visium_centers_per_update"}
    mixed_training = {k:v for k,v in schedules["mixed"]["config"]["training"].items()
                      if k != "visium_centers_per_update"}
    if control_training != mixed_training:
        raise ValueError("Optimizer or continuation budgets differ between arms")
    merged = pd.concat({name: frame[score_columns] for name, frame in frames.items()}, axis=1)
    for metric in score_columns:
        merged[("mixed_minus_control",metric)] = frames["mixed"][metric]-frames["control"][metric]
        merged[("mixed_minus_original",metric)] = frames["mixed"][metric]-frames["original"][metric]
    output = args.output_dir.resolve()
    output.mkdir(parents=True,exist_ok=False)
    merged.to_csv(output / "matched_comparison.csv")
    primary = merged.loc[("primary_all_expert_cells", "annotated_region")]
    readouts = ["v12_local", "v12_spatial", "v12_fused", "v12_concat"]
    fig, axes = plt.subplots(1,2,figsize=(11,4.3))
    for axis, metric, title in zip(axes,score_columns[:2],["Region annotation macro-F1", "Region clustering ARI (known K)"]):
        x = np.arange(len(readouts))
        for offset, name, label, color in ((-.24,"original","V12 step 10234","#8b929b"),
                                         (0.,"control","Xenium +500","#337db0"),
                                         (.24,"mixed","25% HD +500","#df8e30")):
            values = primary.loc[readouts,(name,metric)].to_numpy()
            axis.bar(x+offset,values,width=.23,color=color,label=label)
        official = float(primary.loc["scgpt_spatial_v1_cls",("original",metric)])
        axis.axhline(official,color="#60606c",linestyle="--",label="scGPT-spatial")
        axis.set_xticks(x,[r.replace("v12_","") for r in readouts])
        axis.set(title=title,ylim=(0,1))
        axis.spines[["top","right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Visium HD CRC: same cells, blocks and continuation budget")
    fig.tight_layout()
    fig.savefig(output / "region_comparison.png",dpi=180)
    fig.savefig(output / "region_comparison.pdf")
    plt.close(fig)
    fused = primary.loc["v12_fused"]
    result = dict(status="MATCHED_CONTINUATION_COMPARISON_COMPLETE",
        added_updates_each_arm=500,global_batch=64,visium_fraction=.25,
        primary_fused_region={name:{metric:float(fused[(name,metric)]) for metric in score_columns[:2]}
                              for name in ("original","control","mixed","mixed_minus_control","mixed_minus_original")},
        limitation="Single CRC section and one continuation seed; F1 and unsupervised ARI measure different tasks; official model pretraining includes this CRC source",
        jobid=os.environ["SLURM_JOB_ID"],host=socket.gethostname())
    result["artifact_checks"] = dict(completion_audit_metrics_hash_chain=True,
        original_frozen_membership_verified=True, frozen_baselines_identical=True,
        matched_training_configuration=True)
    (output / "COMPLETE.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
