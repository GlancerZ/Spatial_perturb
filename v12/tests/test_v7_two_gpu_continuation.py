from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/continue_subcellular_v7_two_gpu.sh"


FAKE_SLURM = r'''#!/usr/bin/env python3
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
state = Path(os.environ["FAKE_STATE_DIR"])
scenario = os.environ["FAKE_SCENARIO"]
user = os.environ.get("USER", "glancerz")

def counter(key):
    path = state / key
    value = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(value))
    return value

def arg_after(flag, default=""):
    return args[args.index(flag) + 1] if flag in args else default

if name == "squeue":
    if "--me" in args:
        sys.exit(0)
    if "-n" in args:
        if scenario == "duplicate":
            print("8999|PENDING|Resources")
        sys.exit(0)
    job = arg_after("-j")
    fmt = arg_after("-o")
    if fmt == "%T":
        if job == "9000":
            print("RUNNING")
        else:
            print("RUNNING")
        sys.exit(0)
    if "%T|%R|%L" in fmt:
        if job == "9000":
            sys.exit(0)
        calls = counter(f"queue-pipe-{job}")
        if calls == 1:
            print("RUNNING|node|03:59:00")
        sys.exit(0)
    if job:
        print("JOBID NAME ST TIME TIME_LEFT NODE CPUS NODELIST(REASON)")
        print(f"{job} codex-v7-a-2gpu PD 0:00 4:00:00 1 32 (Resources)")
    sys.exit(0)

if name == "scontrol":
    job = args[-1]
    if job == "9000":
        print(f"JobId=9000 JobName=codex-v6-a-4gpu UserId={user}(1000) Account=aip-junding "
              "JobState=RUNNING NumNodes=2 NumCPUs=32 TimeLimit=04:00:00 MinMemoryNode=115G "
              "TresPerNode=gres/gpu:2 SubmitLine=sbatch --wrap=sleep 14400")
    else:
        print(f"JobId={job} JobName=codex-v7-a-2gpu UserId={user}(1000) Account=aip-junding "
              "JobState=PENDING NumNodes=1 NumCPUs=32 TimeLimit=04:00:00 MinMemoryNode=230G "
              "TresPerNode=gres/gpu:2 SubmitLine=sbatch --wrap=sleep 14400")
    sys.exit(0)

if name == "sacct":
    job = arg_after("-j")
    if scenario == "abnormal" and job == "9000":
        print("9000|FAILED|1:0")
    else:
        print(f"{job}|COMPLETED|0:0")
    sys.exit(0)

if name == "sbatch":
    count = counter("sbatch-count")
    if scenario == "ambiguous":
        print(f"Submitted batch job {9000 + count}")
    else:
        print(9000 + count)
    sys.exit(0)

raise SystemExit(f"Unexpected fake command: {name} {args}")
'''


FAKE_START = r'''#!/usr/bin/env python3
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["V7_CONTINUATION_ROOT"])
scenario = os.environ["FAKE_SCENARIO"]
job = sys.argv[1]
run = root / "outputs/subcellular_lr_v7_a_20260921"
control = Path(f"{run}.controller/status.json")
progress_path = run / "progress.json"
progress = json.loads(progress_path.read_text())
before = int(progress["step"])
config = root / "configs/subcellular_lr_v7_a_20260921.yaml"
config_sha = json.loads(Path(f"{run}.controller/identity.json").read_text())["config_sha256"]
previous_schedule = json.loads((run / "schedule.json").read_text())

if scenario == "wrong_pause":
    step, complete, reason = before + 100, False, "stop_marker"
elif scenario == "two_alloc" and job == "9001":
    step, complete, reason = before + 400, False, "training_time_budget"
else:
    step, complete, reason = 10269, True, None

progress = {
    "step": step,
    "epoch": 1 if complete else 0,
    "next_update": 0 if complete else step,
    "complete": complete,
    "physical_world_size": 2,
    "saved_at": 1.0,
    "config_sha256": config_sha,
    "model_version": 7,
    "architecture": "subcellular_deterministic_lr_routes_v7",
}
progress_path.write_text(json.dumps(progress))
(run / "schedule.json").write_text(json.dumps({
    "total_updates": 10269,
    "model_version": 7,
    "architecture": "subcellular_deterministic_lr_routes_v7",
    "arch_config": {"route": "A"},
    "physical_world_size": 2,
    "global_batch_size": 64,
    "batch_per_card": 16,
    "accumulation": 2,
    "source_sha256": previous_schedule["source_sha256"],
}))
(run / "resume.pt").write_bytes(b"resume")
(run / "checkpoints").mkdir(exist_ok=True)
(run / f"checkpoints/step_{step:08d}.pt").write_bytes(b"archive")
if complete:
    (run / "TRAIN_COMPLETE").write_text("step=10269\n")
status = {
    "identity": {"schema": "subcellular_v7_controller_v1", "output_dir": str(run),
                 "config_sha256": config_sha},
    "status": "training_complete" if complete else "paused",
    "jobid": job,
    "training_gpus": 2,
    "execution_mode": "training_only",
    "evaluation_completion_verified": False,
    "active_child": None,
    "step": step,
    "training_step": step,
    "total_updates": 10269,
}
if reason:
    status["reason"] = reason
control.write_text(json.dumps(status))
'''


class TwoGpuContinuationStateMachineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fake_bin = self.root / "fake-bin"
        self.fake_state = self.root / "fake-state"
        self.fake_bin.mkdir()
        self.fake_state.mkdir()
        fake_slurm = self.fake_bin / "fake-slurm"
        fake_slurm.write_text(FAKE_SLURM)
        fake_slurm.chmod(0o755)
        for name in ("squeue", "scontrol", "sacct", "sbatch"):
            (self.fake_bin / name).symlink_to(fake_slurm)
        self.fake_start_python = self.root / "fake-start.py"
        self.fake_start_python.write_text(FAKE_START)
        self.fake_start_python.chmod(0o755)
        self.fake_start = self.root / "fake-start.sh"
        self.fake_start.write_text(
            f"#!/usr/bin/env bash\nexec python3 {self.fake_start_python} \"$@\"\n")
        self.fake_start.chmod(0o755)
        (self.root / "configs").mkdir()
        (self.root / "scripts").mkdir()
        self.config = self.root / "configs/subcellular_lr_v7_a_20260921.yaml"
        self.config.write_text("output_dir: fixture\n")
        self.dummy_source = self.root / "scripts/dummy_source.py"
        self.dummy_source.write_text("VALUE = 1\n")
        source_sha = hashlib.sha256(self.dummy_source.read_bytes()).hexdigest()
        self.run = self.root / "outputs/subcellular_lr_v7_a_20260921"
        self.run.mkdir(parents=True)
        (self.run / "checkpoints").mkdir()
        Path(f"{self.run}.controller").mkdir()
        Path(f"{self.run}.controller/owner.lock").write_text("")
        (self.run / ".trainer.lock").write_text("")
        (self.root / "logs").mkdir()
        (self.run / "schedule.json").write_text(json.dumps({
            "total_updates": 10269,
            "model_version": 7,
            "architecture": "subcellular_deterministic_lr_routes_v7",
            "arch_config": {"route": "A"},
            "physical_world_size": 4,
            "global_batch_size": 64,
            "batch_per_card": 16,
            "accumulation": 1,
            "source_sha256": {"scripts/dummy_source.py": source_sha},
        }))
        (self.run / "resume.pt").write_bytes(b"resume")
        (self.run / "run_config.yaml").write_bytes(self.config.read_bytes())

    def tearDown(self):
        self.temporary.cleanup()

    def write_status(self, *, complete: bool = False, reason: str = "training_time_budget"):
        step = 10269 if complete else 100
        raw_sha = hashlib.sha256(self.config.read_bytes()).hexdigest()
        sha = hashlib.sha256(b"canonical:" + self.config.read_bytes()).hexdigest()
        self.assertNotEqual(raw_sha, sha)
        identity = {
            "schema": "subcellular_v7_controller_v1", "output_dir": str(self.run),
            "config_sha256": sha,
        }
        Path(f"{self.run}.controller/identity.json").write_text(json.dumps(identity))
        progress = {
            "step": step, "epoch": 1 if complete else 0,
            "next_update": 0 if complete else step, "complete": complete,
            "physical_world_size": 4, "saved_at": 1.0, "config_sha256": sha,
            "model_version": 7, "architecture": "subcellular_deterministic_lr_routes_v7",
        }
        (self.run / "progress.json").write_text(json.dumps(progress))
        (self.run / f"checkpoints/step_{step:08d}.pt").write_bytes(b"archive")
        status = {
            "identity": identity,
            "status": "training_complete" if complete else "paused",
            "jobid": "9000", "training_gpus": 4, "execution_mode": "training_only",
            "evaluation_completion_verified": False, "active_child": None,
            "step": step, "training_step": step, "total_updates": 10269,
        }
        if complete:
            (self.run / "TRAIN_COMPLETE").write_text("step=10269\n")
        else:
            status["reason"] = reason
        Path(f"{self.run}.controller/status.json").write_text(json.dumps(status))

    def run_scheduler(self, scenario: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update({
            "PATH": f"{self.fake_bin}:{env['PATH']}",
            "FAKE_SCENARIO": scenario,
            "FAKE_STATE_DIR": str(self.fake_state),
            "V7_CONTINUATION_TEST_MODE": "1",
            "V7_CONTINUATION_ROOT": str(self.root),
            "V7_CONTINUATION_POLL_SECONDS": "0",
            "V7_CONTINUATION_START_SCRIPT": str(self.fake_start),
        })
        return subprocess.run(["bash", str(SCRIPT), "A", "9000", "--authorized"],
                              env=env, text=True, capture_output=True, timeout=20)

    def sbatch_count(self) -> int:
        path = self.fake_state / "sbatch-count"
        return int(path.read_text()) if path.exists() else 0

    def final_state(self) -> dict:
        return json.loads((Path(f"{self.run}.coordination") /
                           "two-gpu-continuation-20260921/state.json").read_text())

    def test_completed_four_gpu_epoch_submits_nothing(self):
        self.write_status(complete=True)
        result = self.run_scheduler("direct_complete")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.sbatch_count(), 0)
        self.assertEqual(self.final_state()["status"], "complete")

    def test_clean_pause_resumes_and_completes(self):
        self.write_status()
        result = self.run_scheduler("clean")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.sbatch_count(), 1)
        self.assertEqual(self.final_state()["status"], "complete")

    def test_abnormal_predecessor_is_rejected(self):
        self.write_status()
        result = self.run_scheduler("abnormal")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.sbatch_count(), 0)
        self.assertEqual(self.final_state()["status"], "predecessor_failed")

    def test_zero_exit_with_stop_marker_pause_is_rejected(self):
        self.write_status()
        result = self.run_scheduler("wrong_pause")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.sbatch_count(), 1)
        self.assertEqual(self.final_state()["status"], "unsafe_training_exit")

    def test_two_successive_allocations_are_managed(self):
        self.write_status()
        result = self.run_scheduler("two_alloc")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.sbatch_count(), 2)
        self.assertEqual(self.final_state()["status"], "complete")

    def test_existing_same_route_job_blocks_submission(self):
        self.write_status()
        result = self.run_scheduler("duplicate")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.sbatch_count(), 0)
        self.assertEqual(self.final_state()["status"], "conflict")

    def test_source_hash_mismatch_blocks_submission(self):
        self.write_status()
        self.dummy_source.write_text("VALUE = 2\n")
        result = self.run_scheduler("source_mismatch")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.sbatch_count(), 0)
        self.assertEqual(self.final_state()["status"], "unsafe_predecessor")

    def test_ambiguous_sbatch_output_consumes_quota_and_stops(self):
        self.write_status()
        result = self.run_scheduler("ambiguous")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.sbatch_count(), 1)
        state = self.final_state()
        self.assertEqual(state["status"], "submission_ambiguous")
        self.assertEqual(state["allocations_used"], 1)


if __name__ == "__main__":
    unittest.main()
