"""Compute-node validation of mixed V12 weight initialization and resume."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or socket.gethostname().split(".")[0] != os.getenv("SLURMD_NODENAME", "").split(".")[0]:
        raise RuntimeError("QA must run on the authorized compute node")
    import torch
    from evaluate_subcellular_v12 import load_model, atomic_json
    from spatial_pretrain.models.subcellular_v12 import build_model
    state = torch.load(args.run / "resume.pt", map_location="cpu", weights_only=False)
    schedule = json.loads((args.run / "schedule.json").read_text())
    completed = json.loads((args.run / "TRAIN_COMPLETE.json").read_text())
    assert state["step"] == 10236 and state["continuation_step"] == 2
    assert state["optimizer"]["steps_completed"] == 2 and state["complete"]
    assert (state["epoch"], state["next_update"]) == (0, 2)
    assert completed["budget_complete"] and not completed["epoch_complete"]
    assert schedule["initial_step"] == 1 and schedule["resumed"]
    assert schedule["optimizer_resume_state_verified"]
    for step in (10234, 10235, 10236):
        assert (args.run / "checkpoints" / f"step_{step:06d}.pt").is_file()
    original = torch.load(state["initialization"]["checkpoint"], map_location="cpu", weights_only=False)
    initialized = torch.load(args.run / "checkpoints/step_010234.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(v, initialized["model"][key]) for key, v in original["model"].items())
    fresh = build_model(state["config"])
    frozen = [key for key, p in fresh.named_parameters() if not p.requires_grad]
    assert all(torch.equal(original["model"][key], state["model"][key]) for key in frozen)
    changed = [key for key, p in fresh.named_parameters()
               if p.requires_grad and not torch.equal(original["model"][key], state["model"][key])]
    assert any(key.startswith("backbone.") for key in changed), "scGPT fine-tuning did not update any layer"
    assert any(not key.startswith("backbone.") for key in changed), "No new-module learning detected"
    del fresh, original, initialized
    strict, baseline, _, step, evidence = load_model(args.run / "checkpoints/step_010236.pt", 10236)
    assert evidence["sources_verified"] and step == 10236
    del strict, baseline
    atomic_json(args.run / "QA_VERIFIED.json", dict(status="REAL_MIXED_FORWARD_BACKWARD_RESUME_AND_STRICT_LOADING_VERIFIED",
        global_step=step, added_updates=2, frozen_parameter_count=len(frozen),
        updated_parameter_names=changed, jobid=os.environ["SLURM_JOB_ID"], host=socket.gethostname()))
    print("VISIUM_MIX_REAL_DATA_RESUME_AND_STRICT_LOAD_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
