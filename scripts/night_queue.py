"""Download/validate, then wait at most eight hours for an unoccupied training GPU."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from prepare_multitask import write_json
from train_multitask import free_gpu_mib

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "evidence/night-status.json"


def status(phase, **fields):
    record = dict(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    write_json(STATUS, record)
    print(json.dumps(record), flush=True)


def run(command, name, extra_env=None):
    with (ROOT / "logs" / name).open("a") as log:
        return subprocess.run(command, cwd=ROOT, env=dict(os.environ, **(extra_env or {})), stdout=log, stderr=subprocess.STDOUT).returncode


def main():
    # Avoid multiple background writers or two trainers for one experiment.
    with (ROOT / ".local/night.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stages = [
            ("DOWNLOADING", ["/usr/bin/python3", "-u", "scripts/prepare_multitask.py", "--workers", "6"], "download.log", {}),
            ("VALIDATING_DATA", [str(ROOT / ".venv-train/bin/python"), "-u", "scripts/validate_dataset.py"], "data-validation.log", {}),
            ("CPU_PREFLIGHT", [str(ROOT / ".venv-train/bin/python"), "-u", "scripts/train_multitask.py", "--dry-run"], "train-preflight.log", {"JAX_PLATFORMS": "cpu"}),
        ]
        for phase, command, log, env in stages:
            status(phase, actual_updates=0)
            code = run(command, log, env)
            if code:
                status("FAILED", stage=phase, exit_code=code, actual_updates=0)
                return code
        deadline = time.monotonic() + 8 * 3600
        while True:
            free = free_gpu_mib()
            if free >= 24576:
                status("STARTING_SFT", actual_updates=0, free_gpu_mib=free)
                code = run([str(ROOT / ".venv-train/bin/python"), "-u", "scripts/train_multitask.py"], "train-multitask40.log")
                if code != 75:
                    result = json.loads((ROOT / "evidence/multitask40_lora_v1-status.json").read_text())
                    status("SFT_COMPLETE_PPO_PENDING" if code == 0 else "SFT_FAILED", sft=result,
                           ppo_updates=0, ppo_blocker="Runtime, PyTorch lineage conversion and memory validation not yet completed")
                    return code
            if time.monotonic() >= deadline:
                status("WAIT_EXPIRED", actual_updates=0, free_gpu_mib=free)
                return 75
            status("WAITING_GPU", actual_updates=0, free_gpu_mib=free, required_free_mib=24576)
            time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
