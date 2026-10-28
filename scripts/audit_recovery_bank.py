"""Read-only physical initial-state audit; never edit a bank or run policy actions."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party/RLinf")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for key in ("MUJOCO_GL", "PYOPENGL_PLATFORM"):
        os.environ.setdefault(key, "egl")
    from baseline import configure_libero
    from build_recovery_bank import bowl_penetrations
    configure_libero()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.libero.recovery import RecoveryEnv
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    raw = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                            task.problem_folder / task.bddl_file), camera_heights=256, camera_widths=256)
    env = RecoveryEnv(raw, dict(manifest=str(args.manifest), split="dev"))
    report = dict(manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  guard_sha256=hashlib.sha256((ROOT / "scripts/build_recovery_bank.py").read_bytes()).hexdigest(),
                  packages={k: importlib.metadata.version(k) for k in ("mujoco", "robosuite", "numpy")},
                  protocol="restore every dev state, sim.forward, check bowl-other penetration >0.1 mm / bowl-table >1 mm; no controls",
                  entries=[])
    try:
        for index, entry in enumerate(env.entries):
            env.seed(42)
            env.reset()
            env.restore_recovery(index)
            sim = raw.env.sim
            sim.forward()
            bowl = raw.env.get_object("akita_black_bowl_1")
            overlaps = bowl_penetrations(sim, bowl.contact_geoms)
            finger_contact = bool(raw.env.check_contact(raw.env.robots[0].gripper.contact_geoms, bowl))
            finite = bool(np.isfinite(raw.get_sim_state()).all())
            native_success = bool(raw.check_success())
            record = dict(id=entry["id"], state_sha256=entry["state_sha256"],
                          overlaps=overlaps, finite=finite, native_success=native_success,
                          finger_contact=finger_contact,
                          passed=not overlaps and finite and not native_success and not finger_contact)
            report["entries"].append(record)
            print(json.dumps(dict(id=record["id"], passed=record["passed"], overlaps=len(overlaps))), flush=True)
    finally:
        env.close()
    report["passed"] = all(r["passed"] for r in report["entries"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
