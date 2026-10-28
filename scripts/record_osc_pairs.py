"""Re-execute two verified OSC expert trajectories and record actual PRE-action observations.

Review candidates only: no training, no original-image relabeling, no VLA inference.
"""
import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from replay_mixed_correction import ROOT, SOURCES, sha, write_json

CONVERSION = ROOT / "reports/osc-conversion-20261029/v1"
SNAPSHOTS = ROOT / "reports/full-state-replay-20261028/v1"
CAMERAS = (("image", "agentview_image"), ("wrist_image", "robot0_eye_in_hand_image"))
PROMPT = "pick up the black bowl next to the plate and place it on the plate"
SCHEMA = "pi05-osc-expert-pairs-review-v1"


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def observation_state(obs):
    """Match evaluate_recovery.payload's xyzw -> axis-angle convention exactly."""
    quat = np.asarray(obs["robot0_eef_quat"], dtype=float)
    if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1.):
        raise ValueError("Invalid xyzw quaternion")
    w = np.clip(quat[3], -1., 1.)
    den = np.sqrt(1. - w * w)
    rotation = np.zeros(3) if math.isclose(den, 0.) else quat[:3] * 2. * math.acos(w) / den
    state = np.r_[obs["robot0_eef_pos"], rotation, obs["robot0_gripper_qpos"]]
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError("Invalid 8D observation state")
    return state


def pack_before(obs, action, index, original_index, episode, controller, sim_time):
    """Copy all inputs before step(); reject bad labels instead of clipping them."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.):
        raise ValueError("Invalid 7D normalized OSC action")
    row = dict(state=observation_state(obs).tolist(), actions=action.tolist(),
               episode_index=episode, task_index=8, frame_index=index,
               original_control_index=original_index, original_controller=controller,
               controller="OSC_POSE", sim_time_s=float(sim_time), timestamp=index * .05,
               diagnostic_hold=controller == "hold", prompt=PROMPT)
    images, hashes = [], {}
    for field, key in CAMERAS:
        raw = np.asarray(obs[key])
        if raw.shape != (256, 256, 3) or raw.dtype != np.uint8:
            raise ValueError("Expected raw 256x256 RGB uint8 camera")
        image = np.ascontiguousarray(raw[::-1, ::-1])
        encoded = BytesIO()
        Image.fromarray(image).save(encoded, format="PNG")
        row[field] = {"bytes": encoded.getvalue(), "path": None}
        hashes[field] = digest(image)
        images.append(image)
    return row, np.concatenate(images, axis=1), hashes


def validate_dataset(path, physics_path, audit_path):
    """Independently decode the on-disk pairs and compare to live pre-step evidence."""
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pylist()
    with np.load(physics_path, allow_pickle=False) as z:
        before, after, actions, states = (z[k] for k in ("before", "after", "actions", "observation_state"))
    audit = json.loads(audit_path.read_text())
    if len(rows) != len(before) or not rows or len(audit) != len(rows):
        raise ValueError("Incomplete pairs")
    for i, row in enumerate(rows):
        if (row["frame_index"] != i or row["timestamp"] != i * .05 or
                row["sim_time_s"] != before[i, 0] or
                not np.array_equal(row["actions"], actions[i]) or
                not np.array_equal(row["state"], states[i]) or
                digest(states[i]) != audit[i]["state_sha256"] or
                digest(actions[i]) != audit[i]["action_sha256"]):
            raise ValueError(f"Pre-observation / action alignment failed at {i}")
        for field, _ in CAMERAS:
            image = np.asarray(Image.open(BytesIO(row[field]["bytes"])))
            if image.shape != (256, 256, 3) or image.dtype != np.uint8 or digest(image) != audit[i][field]:
                raise ValueError(f"Actual pre-action RGB mismatch at {i}/{field}")
    if (not np.array_equal(before[1:], after[:-1]) or
            not np.allclose(after[:, 0] - before[:, 0], .05, rtol=0., atol=1e-10) or
            not np.isfinite(np.r_[before.ravel(), after.ravel(), actions.ravel(), states.ravel()]).all()):
        raise ValueError("Physical timeline discontinuity")
    return dict(passed=True, paired_rows=len(rows), decoded_rgb_images=2 * len(rows),
                pre_observation_action_alignment=True, continuous_physics=True,
                unchanged_saved_actions=True)


def worker(output, source):
    import imageio.v2 as imageio
    import pyarrow as pa
    import pyarrow.parquet as pq
    from simulation_snapshot import identity, restore
    from verify_osc_conversion import source_data
    from verify_simulation_snapshot import create_env, initialize, measure

    protocol = json.loads((output / "protocol.json").read_text())
    for relative, expected in protocol["inputs_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen input changed: {relative}")
    original = CONVERSION / source
    result = json.loads((original / "tracking-result.json").read_text())
    if not result["passed"]:
        raise ValueError("Unverified expert source")
    expected_rows = json.loads((original / "tracking-steps.json").read_text())
    expected_states = np.load(original / "tracking-states.npz", allow_pickle=False)["after"]
    actions = np.load(original / "tracking-candidate-actions.npz", allow_pickle=False)["actions"]
    meta, initial, records, start = source_data(source)
    if len(actions) != result["controls"] or len(actions) != len(expected_rows):
        raise ValueError("Expert controls incomplete")
    out = output / source
    out.mkdir()
    rows, audits, before_states, after_states, obs_states = [], [], [], [], []
    env = create_env()
    try:
        initialize(env, meta, initial)
        restore(env, SNAPSHOTS / source / f"snapshot-{start}.npz")
        expected_identity = json.loads((original / "tracking-identity.json").read_text())
        actual_identity = identity(env)
        if actual_identity != {k: expected_identity[k] for k in actual_identity}:
            raise ValueError("Runtime/model mismatch")
        controller = env.env.robots[0].controller
        for key in ("input_min", "input_max", "output_min", "output_max"):
            if not np.array_equal(getattr(controller, key), expected_identity[key]):
                raise ValueError("Action scaling changed")
        if not np.array_equal(controller.initial_joint, expected_identity["nullspace_anchor"]):
            raise ValueError("Nullspace anchor changed")
        write_json(out / "identity.json", actual_identity)
        with imageio.get_writer(out / "review.mp4", fps=20) as video:
            for i, action in enumerate(actions):
                before = env.get_sim_state().copy()
                # This exact refresh already occurred in the verified candidate.
                obs_before = env.regenerate_obs_from_state(before)
                expected = expected_rows[i]
                row, image, hashes = pack_before(obs_before, action, i, expected["index"],
                                                SOURCES.index(source), expected["original_controller"], before[0])
                if not np.array_equal(env.get_sim_state(), before):
                    raise ValueError("Recording mutated pre-action physics")
                video.append_data(image)
                obs_after, _, done, _ = env.step(action.tolist())
                after = env.get_sim_state().copy()
                measured, final_image = measure(env, obs_after, done, expected["index"])
                if (not np.array_equal(after, expected_states[i]) or any(
                        measured[k] != expected[k] for k in
                        ("image_sha256", "native_goal", "grasp", "finger_contact", "contact_count", "positions"))):
                    raise ValueError(f"Verified physical/RGB trajectory diverged at {source}/{i}")
                rows.append(row)
                state = np.asarray(row["state"], dtype=np.float64)
                audits.append(dict(index=i, **hashes, state_sha256=digest(state),
                                   action_sha256=digest(action), post=measured))
                before_states.append(before)
                after_states.append(after)
                obs_states.append(state)
                if (i + 1) % 100 == 0:
                    print(json.dumps(dict(source=source, controls=i + 1, bitwise=True)), flush=True)
            # N pre-action frames followed by one final returned observation.
            video.append_data(final_image)
        pq.write_table(pa.Table.from_pylist(rows), out / "pairs.parquet", compression="zstd")
        np.savez_compressed(out / "physics.npz", before=before_states, after=after_states,
                            actions=actions, observation_state=obs_states)
        write_json(out / "pair-audit.json", audits)
        checks = validate_dataset(out / "pairs.parquet", out / "physics.npz", out / "pair-audit.json")
        checks.update(source=source, ready=False, human_review="pending", split="train_candidate",
                      expert_reference_actions=True, vla_inference=False,
                      original_controls=len(records) - start, diagnostic_hold_controls=20,
                      original_start=start, all_post_states_bitwise_match=True,
                      all_post_rgb_and_flags_exact=True, final_released_goal_controls=sum(
                          r["native_goal"] and not r["grasp"] and not r["finger_contact"]
                          for r in expected_rows[-20:]),
                      files={name: dict(sha256=sha(out / name), bytes=(out / name).stat().st_size)
                             for name in ("pairs.parquet", "physics.npz", "pair-audit.json", "review.mp4")})
        write_json(out / "result.json", checks)
        print(json.dumps(checks), flush=True)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=SOURCES)
    args = parser.parse_args()
    if args.source:
        worker(args.output, args.source)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT / "scripts" / n for n in
              ("record_osc_pairs.py", "verify_osc_conversion.py", "osc_action_conversion.py",
               "verify_simulation_snapshot.py", "simulation_snapshot.py", "replay_mixed_correction.py")]
    for source in SOURCES:
        inputs += list((CONVERSION / source).glob("tracking-*.json"))
        inputs += list((CONVERSION / source).glob("tracking-*.npz"))
        inputs += list((SNAPSHOTS / source).glob("snapshot-*.npz"))
    write_json(args.output / "protocol.json", dict(schema=SCHEMA, seed=42, ready=False,
               human_review="pending", training=False, sources=list(SOURCES), control_hz=20,
               state="eef xyz + axis-angle xyzw convention + two finger positions",
               images="actual pre-action RGB uint8 256x256; both axes reversed as inference",
               actions="verified expert 7D normalized OSC delta + gripper; no clipping",
               pairing="one current-state refresh -> copy RGB/state -> env.step(action)",
               video="N pre-action camera pairs + final post observation, 20fps",
               diagnostic_hold="last 20 rows retained and flagged; not proposed as training anchors",
               original_images_reused=False, input_data="newly executed trajectory, not exact old recording",
               inputs_sha256={str(p.relative_to(ROOT)): sha(p) for p in sorted(inputs)}))
    results = []
    for source in SOURCES:
        subprocess.run([sys.executable, "-u", __file__, "--output", str(args.output),
                        "--source", source], check=True)
        results.append(json.loads((args.output / source / "result.json").read_text()))
        write_json(args.output / "manifest.json", dict(schema=SCHEMA, ready=False,
                   human_review="pending", training=False, records=results))


if __name__ == "__main__":
    main()
