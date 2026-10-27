"""Independently replay recorded actions before accepting learned recovery data."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from export_sft_for_ppo import sha256
from recovery_recording import validate_training_entry

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third_party/RLinf")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/recovery-bowl-v3/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".local/numba"))
    from baseline import configure_libero
    configure_libero()
    import pyarrow.parquet as pq
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from rlinf.envs.sim.libero.recovery import RecoveryEnv
    task = benchmark.get_benchmark_dict()["libero_goal"]().get_task(8)
    raw = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) /
                            task.problem_folder / task.bddl_file),
                            camera_heights=256, camera_widths=256)
    env = RecoveryEnv(raw, {"manifest": str(args.manifest), "split": "train"})
    entries = {entry["id"]: index for index, entry in enumerate(env.entries)}
    records, audits = [], []
    try:
        for directory in args.recordings:
            directory = directory.resolve()
            identity = directory / "identity.json"
            info = json.loads(identity.read_text())
            if info["split"] != "train" or info["manifest_sha256"] != sha256(args.manifest):
                raise ValueError("Recording source/split differs from frozen bank")
            for metadata in sorted(directory.glob("*.record.json")):
                record = json.loads(metadata.read_text())
                validate_training_entry(record["entry"])
                if record["prompt"] != task.language:
                    raise ValueError("Task prompt mismatch")
                parquet, physics = directory / record["path"], directory / record["physics"]
                if sha256(parquet) != record["sha256"] or sha256(physics) != record["physics_sha256"]:
                    raise ValueError("Recording content changed")
                table = pq.read_table(parquet)
                with np.load(physics, allow_pickle=False) as archive:
                    states, actions = archive["states"], archive["actions"]
                if len(states) != len(actions) + 1 or len(actions) != record["frames"]:
                    raise ValueError("Incomplete control trajectory")
                np.testing.assert_allclose(table["actions"].to_pylist(), np.clip(actions, -1, 1), rtol=0, atol=0)
                replay_errors, replay_success = {}, {}
                for label, commands in (("raw", actions), ("clipped", np.clip(actions, -1, 1))):
                    env.seed(42)
                    env.reset()
                    env.restore_recovery(entries[record["entry"]["id"]])
                    maximum = float(np.max(np.abs(raw.get_sim_state() - states[0])))
                    done = False
                    for index, action in enumerate(commands):
                        _, _, done, _ = env.step(action)
                        maximum = max(maximum, float(np.max(np.abs(raw.get_sim_state() - states[index + 1]))))
                    replay_errors[label], replay_success[label] = maximum, bool(done)
                passed = all(replay_success.values()) and max(replay_errors.values()) <= 1e-5
                audit = dict(record=str(metadata.relative_to(ROOT)), passed=passed,
                             final_released_success=replay_success, max_physics_abs_error=replay_errors,
                             controls=len(actions), tolerance=1e-5)
                audits.append(audit)
                print(json.dumps(audit), flush=True)
                if passed:
                    record.update(path=str(parquet.relative_to(ROOT)),
                                  physics=str(physics.relative_to(ROOT)),
                                  recording_identity=str(identity.relative_to(ROOT)),
                                  recording_identity_sha256=sha256(identity),
                                  independent_replay_verified=True)
                    records.append(record)
    finally:
        env.close()
    coverage = {category: sorted({r["entry"]["source_initial_id"] for r in records
                                  if r["entry"]["category"] == category})
                for category in ("empty_grasp", "object_offset", "placement_failure")}
    ready = all(coverage.values()) and len(coverage["placement_failure"]) >= 2
    result = dict(schema="pi05-model-recovery-replay-v1", ready=ready, root=str(ROOT),
                  records=records, audits=audits, coverage=coverage,
                  frames=sum(r["frames"] for r in records),
                  replay_manifest_sha256=sha256(ROOT / "evidence/dataset-manifest.json"),
                  bank_manifest_sha256=sha256(args.manifest), control_hz=20,
                  source="successful learned-policy rollouts; no human or scripted labels",
                  validation="fresh-process raw AND clipped action replay, full physics <=1e-5, released goal >=5 controls")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(ready=ready, episodes=len(records), coverage=coverage)), flush=True)
    return int(not ready)


if __name__ == "__main__":
    raise SystemExit(main())
