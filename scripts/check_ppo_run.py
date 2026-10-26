"""Settle actual updates, normal exit, cgroup memory and saved optimizer state."""
import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    log = re.sub(r"\x1b\[[0-9;]*m", "", args.log.read_text(errors="replace"))
    updates = [json.loads(line.split("PPO_UPDATE ", 1)[1]) for line in log.splitlines()
               if "PPO_UPDATE {" in line]
    attempts = sum(u["optimizer_step_attempts"] for u in updates)
    finite = sum(u["finite_optimizer_steps"] for u in updates)
    assert len(updates) == args.rounds and attempts == finite == args.updates
    properties = dict(line.split("=", 1) for line in subprocess.check_output(
        ["systemctl", "--user", "show", args.unit, "-p", "ActiveState", "-p", "Result",
         "-p", "ExecMainStatus"], text=True).splitlines())
    assert properties == {"ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}
    records = [json.loads(line) for line in args.memory.read_text().splitlines()]
    peaks, oom = [], 0
    for record in records:
        if record.get("ControlGroup"):
            peaks.append(int(record.get("memory.peak", 0)))
            events = dict(line.split() for line in record.get("memory.events", "").splitlines())
            oom = max(oom, *(int(events.get(key, 0)) for key in ("oom", "oom_kill", "oom_group_kill")))
    assert peaks and oom == 0
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    assert saved["format"] == "rlinf-fsdp1-single-rank-adapter-v1"
    assert saved["base_weights_sha256"] == "4526e30c6aa11108a3331f7f4ae7931394629961df42736deedd7169ec154d5f"
    assert all(torch.isfinite(value).all() for value in saved["model"].values())
    steps = Counter()
    for optimizer in saved["optimizers"]:
        for state in optimizer["state"].values():
            if "step" in state:
                steps[int(state["step"])] += 1
            assert all(torch.isfinite(value).all() for value in state.values()
                       if isinstance(value, torch.Tensor))
    assert max(steps) == args.updates
    assert all(scheduler["last_epoch"] == args.rounds for scheduler in saved["lr_schedulers"])
    initials = [json.loads(line.split("RECOVERY_INITIAL ", 1)[1]) for line in log.splitlines()
                if "RECOVERY_INITIAL {" in line]
    output = {"status": "PASS", "unit": args.unit, "systemd": properties,
              "rounds": len(updates), "optimizer_updates": attempts, "finite_optimizer_updates": finite,
              "peak_memory_bytes": max(peaks), "memory_limit_gib": 48, "oom_events": oom,
              "checkpoint": str(args.checkpoint), "checkpoint_bytes": args.checkpoint.stat().st_size,
              "checkpoint_sha256": hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest(),
              "format": saved["format"], "base_weights_sha256": saved["base_weights_sha256"],
              "finite_saved_model_tensors": len(saved["model"]),
              "optimizer_state_step_counts": dict(sorted(steps.items())),
              "scheduler_epochs": [s["last_epoch"] for s in saved["lr_schedulers"]],
              "train_initial_ids": sorted({r["id"] for r in initials if r["split"] == "train"}),
              "heldout_initial_ids": sorted({r["id"] for r in initials if r["split"] == "heldout"}),
              "reset_counts": dict(Counter((r["split"] + "/" + r["category"]) for r in initials)),
              "note": "Zero-step optimizer states are reported; not all injected LoRA tensors are active."}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
