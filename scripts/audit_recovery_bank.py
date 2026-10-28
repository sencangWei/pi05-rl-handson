"""Read-only physical initial-state audit; never edit a bank or run policy actions."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party/RLinf")]


def inspect_state(raw):
    from build_recovery_bank import bowl_penetrations
    sim = raw.env.sim
    sim.forward()
    bowl = raw.env.get_object("akita_black_bowl_1")
    overlaps = bowl_penetrations(sim, bowl.contact_geoms)
    finger_contact = bool(raw.env.check_contact(raw.env.robots[0].gripper.contact_geoms, bowl))
    finite = bool(np.isfinite(raw.get_sim_state()).all())
    native_success = bool(raw.check_success())
    return {"overlaps": overlaps, "finite": finite, "native_success": native_success,
                "finger_contact": finger_contact,
                "passed": not overlaps and finite and not native_success and not finger_contact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--manual-manifest", type=Path,
                        help="Also inspect the exact pre-action state of each approved raw source")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        os.environ.setdefault(key, "egl")
    from baseline import configure_libero
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.libero.recovery import RecoveryEnv
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    raw = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
    env = RecoveryEnv(raw, {"manifest": str(args.manifest), "split": args.split})
    report = {"manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "guard_sha256": hashlib.sha256((ROOT / "scripts/build_recovery_bank.py").read_bytes()).hexdigest(),
                  "packages": {k: importlib.metadata.version(k) for k in ("mujoco", "robosuite", "numpy")},
                  "split": args.split,
                  "protocol": "restore every selected state, sim.forward, check bowl-other penetration >0.1 mm / bowl-table >1 mm; no controls",
                  "entries": []}
    try:
        for index, entry in enumerate(env.entries):
            env.seed(42)
            env.reset()
            env.restore_recovery(index)
            record = dict(id=entry["id"], state_sha256=entry["state_sha256"],
                          **inspect_state(raw))
            report["entries"].append(record)
            print(json.dumps({"id": record["id"], "passed": record["passed"], "overlaps": len(record["overlaps"])}), flush=True)
        if args.manual_manifest:
            manual = json.loads(args.manual_manifest.read_text())
            report["manual_manifest_sha256"] = hashlib.sha256(args.manual_manifest.read_bytes()).hexdigest()
            report["recorded_starts"] = []
            for audit in manual["audits"]:
                directory = Path(manual["root"]) / "corrections" / audit["source"]
                metadata = directory / "metadata.json"
                meta = json.loads(metadata.read_text())
                physics = directory / "physics-000000.npz"
                if (hashlib.sha256(metadata.read_bytes()).hexdigest() != audit["metadata_sha256"]
                        or hashlib.sha256(physics.read_bytes()).hexdigest() != audit["files"][physics.name]
                        or meta["split"] != args.split or meta["task_id"] != 8):
                    raise ValueError("Recording identity/split mismatch")
                env.seed(42)
                env.reset()
                with np.load(physics, allow_pickle=False) as data:
                    state = data["before"]
                    raw.set_init_state(state)
                    state_hash = hashlib.sha256(state.tobytes()).hexdigest()
                report["recorded_starts"].append(dict(source=audit["source"], entry=meta["bank_entry"]["id"],
                    raw_initial_state_sha256=state_hash, **inspect_state(raw)))
    finally:
        env.close()
    report["passed"] = all(r["passed"] for r in report["entries"] + report.get("recorded_starts", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
