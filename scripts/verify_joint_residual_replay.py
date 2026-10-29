"""Replay encoded expert targets through live physics; never a policy evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np

from joint_action_contract import absolute_joint_controller
from joint_residual import transform_chunk
from motion_recording import MotionCapture
from record_joint_pairs import check_render_backend
from record_osc_pairs import SNAPSHOTS
from replay_mixed_correction import ROOT, SOURCES, sha, write_json
from simulation_snapshot import restore
from verify_osc_conversion import source_data
from verify_simulation_snapshot import create_env, initialize, measure


def replay(source, record):
    folder = (ROOT / record["path"]).parent
    result = json.loads((folder / "result.json").read_text())
    for name in ("physics.npz", "audit.json", "pairs.parquet"):
        if sha(folder / name) != result["files_sha256"][name]:
            raise ValueError("Replay source changed")
    with np.load(folder / "physics.npz", allow_pickle=False) as z:
        reference = {k: z[k] for k in z.files}
    meta, initial, _, start = source_data(source)
    env = create_env()
    errors, target_errors, rows = [], [], []
    try:
        initialize(env, meta, initial)
        restore(env, SNAPSHOTS / source / f"snapshot-{start}.npz")
        capture = MotionCapture(env)
        with absolute_joint_controller(env) as (controller, _):
            for i, original in enumerate(reference["actions"]):
                env.regenerate_obs_from_state(env.get_sim_state())
                fixed_anchor = reference["observation_state"][i, :7]
                chunk = np.repeat(original[None, :], 10, axis=0)
                encoded = transform_chunk(chunk, fixed_anchor)
                decoded = transform_chunk(encoded, capture.read()["joint_pos"], decode=True)[0]
                obs, _, done, _ = env.step(decoded.tolist())
                if not np.array_equal(controller.goal_qpos, decoded[:7]):
                    raise ValueError("Residual target was rescaled or clipped")
                target_errors.append(float(np.max(np.abs(decoded - original))))
                errors.append(float(np.max(np.abs(env.get_sim_state() - reference["after"][i]))))
                rows.append(measure(env, obs, done, i)[0])
                if errors[-1] > 1e-6:
                    raise ValueError(f"Physical residual replay divergence at {i}: {errors[-1]}")
                if (i + 1) % 100 == 0:
                    print(json.dumps(dict(source=source, replay_controls=i + 1)), flush=True)
        release = sum(r["native_goal"] and not r["finger_contact"] for r in rows[-20:])
        return dict(source=source, controls=len(rows), max_physics_error=max(errors),
                    max_target_error_rad=max(target_errors), released_goal_final20=release,
                    passed=release == 20, policy_inference=False, expert_replay=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    check_render_backend()
    if a.output.exists():
        raise FileExistsError(a.output)
    records = json.loads(a.manifest.read_text())["records"]
    results = []
    for source in SOURCES:
        results.append(replay(source, next(r for r in records if r["source"] == source)))
    write_json(a.output, dict(results=results, passed=all(r["passed"] for r in results)))
    print(json.dumps(results), flush=True)
    if not all(r["passed"] for r in results):
        raise SystemExit("Expert residual physics replay failed")


if __name__ == "__main__":
    main()
