"""Evaluate frozen parent and candidate after verified bounded training; no UI promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
EXPERIMENT = "osc_expert_replay2000_v1"
TRAIN_UNIT = "pi05-osc-sft29-train"


def properties(unit):
    raw = subprocess.check_output(["systemctl", "--user", "show", unit,
        "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus"], text=True)
    return dict(line.split("=", 1) for line in raw.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-only", action="store_true",
                        help="Resume after the completed parent run without overwriting it")
    args = parser.parse_args()
    started = time.monotonic()
    while properties(TRAIN_UNIT)["ActiveState"] in ("active", "activating", "deactivating"):
        if time.monotonic() - started > 3700:
            raise TimeoutError("Bounded training did not exit")
        time.sleep(5)
    train = properties(TRAIN_UNIT)
    status = json.loads((ROOT / f"evidence/{EXPERIMENT}-status.json").read_text())
    assert train["Result"] == "success" and train["ExecMainStatus"] == "0", train
    assert status["status"] == "COMPLETE" and status["actual_updates"] == 2000
    assert status["sampling_preview_exact_match"] is True
    plan = json.loads((ROOT / "evidence/osc-sft29-execution-plan.json").read_text())
    for key in ("parent_files", "replay_manifest_sha256", "human_manifest_sha256"):
        assert status[key] == plan[key], key
    assert all(plan["source_sha256"][k] == v for k, v in status["source_sha256"].items())
    for path, digest in plan["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    evaluation = plan["evaluation"]
    assert hashlib.sha256((ROOT / evaluation["manifest"]).read_bytes()).hexdigest() == evaluation["manifest_sha256"]
    checkpoints = {
        "parent": ROOT / "checkpoints/pi05_multitask40_lora/multitask40_lora_v1/9999",
        "candidate": ROOT / f"checkpoints/pi05_recovery_replay_lora/{EXPERIMENT}/1999",
    }
    if args.candidate_only:
        parent = ROOT / "artifacts/20261029-osc-sft/parent"
        identity = json.loads((parent / "identity.json").read_text())
        rows = json.loads((parent / "summary.json").read_text())
        assert identity["jax_server"]["checkpoint"] == str(checkpoints["parent"])
        assert identity["manifest_sha256"] == evaluation["manifest_sha256"]
        assert len(rows) == 12 and len({r["id"] for r in rows}) == 12
        del checkpoints["parent"]
    for label, checkpoint in checkpoints.items():
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", 9875))
        unit = f"pi05-osc-sft29-{label}-server"
        log = ROOT / f"logs/{EXPERIMENT}-{label}-server.log"
        assert checkpoint.is_dir() and not log.exists()
        subprocess.run(["systemd-run", "--user", "--unit=" + unit,
            "--property=WorkingDirectory=" + str(ROOT), "--property=MemoryMax=32G", "--property=MemorySwapMax=0",
            "--property=RuntimeMaxSec=1800", "--property=StandardOutput=append:" + str(log),
            "--property=StandardError=append:" + str(log), "--setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "--setenv=OMP_NUM_THREADS=4", "--setenv=OPENBLAS_NUM_THREADS=4",
            str(ROOT / ".venv-train/bin/python"), "-u", "scripts/serve_sft.py",
            "--checkpoint", str(checkpoint), "--port", "9875"], check=True)
        try:
            wait_started = time.monotonic()
            while not (log.exists() and "server listening on" in log.read_text()):
                assert properties(unit)["ActiveState"] == "active", log.read_text() if log.exists() else unit
                if time.monotonic() - wait_started > 900:
                    raise TimeoutError("Policy startup deadline")
                time.sleep(5)
            env = dict(os.environ, PYTHONPATH="/home/robot/voice-arm-sim/.local/openpi/third_party/libero",
                TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
                CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            with (ROOT / f"logs/{EXPERIMENT}-{label}-evaluation.log").open("x") as output:
                subprocess.run([str(ROOT / ".venv-ppo/bin/python"), "-u", "scripts/evaluate_recovery.py",
                    "--port", "9875", "--expected-server-checkpoint", str(checkpoint),
                    "--manifest", evaluation["manifest"], "--split", "dev", "--max-steps", "300",
                    "--noise-seed", "42", "--output", f"artifacts/20261029-osc-sft/{label}"],
                    env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
        finally:
            subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    print("FROZEN_PARENT_AND_CANDIDATE_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
