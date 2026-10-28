"""Bounded physical test: can recorded joint jogs be expressed by the existing OSC interface?"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from osc_action_conversion import acceptance, inverse_scale, pose_delta
from replay_mixed_correction import ROOT, SOURCES, load_source, sha, write_json
from simulation_snapshot import identity, restore
from verify_simulation_snapshot import create_env, execute, initialize, measure, windows

SNAPSHOTS = ROOT / "reports/full-state-replay-20261028/v1"
LIMITS = dict(position_p95_m=.02, rotation_p95_rad=.15, cabinet_extra_penetration_m=.001)
MODES = ("delta", "tracking")
HOLD = 20


def source_data(source):
    manifest = json.loads((ROOT / "evidence/manual-recovery28-manifest.json").read_text())
    _, meta, initial, records = load_source(source, manifest)
    return meta, initial, records, windows(records)[0][0]


def physical_pose(env, fk):
    """Kinematics on a separate MjData; never forward or teleport the acting simulator."""
    import mujoco
    mujoco.mj_copyData(fk, env.sim.model._model, env.sim.data._data)
    mujoco.mj_kinematics(env.sim.model._model, fk)
    robot = env.env.robots[0]
    site = env.sim.model.site_name2id(robot.gripper.important_sites["grip_site"])
    return fk.site_xpos[site].copy(), fk.site_xmat[site].reshape(3, 3).copy()


def cabinet_penetration(env):
    peak = 0.
    for c in env.sim.data.contact[:env.sim.data.ncon]:
        names = [env.sim.model.geom_id2name(int(i)) or "" for i in (c.geom1, c.geom2)]
        if (any(n.startswith(("robot0_", "gripper0_")) for n in names)
                and any("cabinet" in n.lower() for n in names)):
            peak = max(peak, -float(c.dist))
    return peak


def worker(output, source, mode):
    import imageio.v2 as imageio
    import mujoco
    meta, initial, records, start = source_data(source)
    out = output / source
    if mode == "reference":
        out.mkdir()
    expected = None if mode == "reference" else json.loads((out / "reference-steps.json").read_text())
    targets = None
    if mode != "reference":
        with np.load(out / "reference-poses.npz", allow_pickle=False) as z:
            targets = {k: z[k] for k in z.files}
    env = create_env()
    rows, states, before_pos, before_rot, after_pos, after_rot = [], [], [], [], [], []
    actions, demands, substep_peaks = [], [], []
    try:
        initialize(env, meta, initial)
        restore(env, SNAPSHOTS / source / f"snapshot-{start}.npz")
        fk = mujoco.MjData(env.sim.model._model)
        osc = env.env.robots[0].controller
        anchor = osc.initial_joint.copy()
        if not osc.use_delta or osc.impedance_mode != "fixed" or osc.position_limits is not None or osc.orientation_limits is not None:
            raise ValueError("Unsupported OSC contract")
        cabinet_geoms = [env.sim.model.geom_id2name(i) or "" for i in range(env.sim.model.ngeom)
                         if "cabinet" in (env.sim.model.geom_id2name(i) or "").lower()]
        if not cabinet_geoms:
            raise ValueError("Cabinet contact metric has no matching geometry")
        write_json(out / f"{mode}-identity.json", dict(**identity(env),
                   input_min=osc.input_min.tolist(), input_max=osc.input_max.tolist(),
                   output_min=osc.output_min.tolist(), output_max=osc.output_max.tolist(),
                   nullspace_anchor=anchor.tolist(), cabinet_geoms=cabinet_geoms))
        initial_pose = physical_pose(env, fk)
        previous_pose = initial_pose
        original_sim_step = env.sim.step

        def probe():
            original_sim_step()
            substep_peaks.append(cabinet_penetration(env))

        env.sim.step = probe
        with imageio.get_writer(out / f"{mode}.mp4", fps=20) as video:
            def record(index, obs, done, action, demand, before):
                nonlocal previous_pose
                row, image = measure(env, obs, done, index)
                pos, rot = physical_pose(env, fk)
                j = len(rows)
                target_pos, target_rot = (pos, rot) if targets is None else (targets["after_pos"][j], targets["after_rot"][j])
                row.update(eef_pos=pos.tolist(),
                           position_error_m=float(np.linalg.norm(pos - target_pos)),
                           rotation_error_rad=float(np.linalg.norm(pose_delta(pos, rot, pos, target_rot)[3:])),
                           cabinet_penetration_m=max(substep_peaks, default=0.),
                           physics_substeps=len(substep_peaks),
                           original_controller=records[index]["controller"] if index < len(records) else "hold",
                           action=np.asarray(action).tolist(), unclipped_demand=np.asarray(demand).tolist(),
                           clipped_axes=int(np.count_nonzero((np.asarray(demand) < osc.input_min) |
                                                            (np.asarray(demand) > osc.input_max))))
                rows.append(row)
                states.append(env.get_sim_state().copy())
                before_pos.append(before[0])
                before_rot.append(before[1])
                after_pos.append(pos)
                after_rot.append(rot)
                actions.append(np.asarray(action))
                demands.append(np.asarray(demand))
                previous_pose = (pos, rot)
                video.append_data(image)
                substep_peaks.clear()

            if mode == "reference":
                def callback(kind, index, obs, done):
                    if kind == "step":
                        record(index, obs, done, records[index]["action"], np.zeros(6), previous_pose)
                execute(env, records, start, len(records), callback)
                old_states = np.load(SNAPSHOTS / source / "reference-states.npz", allow_pickle=False)["after"][start:]
                if not np.array_equal(old_states, np.asarray(states)):
                    raise ValueError("Reference diverged from previously verified new reference")
            else:
                for j, r in enumerate(records[start:]):
                    env.regenerate_obs_from_state(env.get_sim_state())
                    before = physical_pose(env, fk)
                    if r["controller"] == "OSC_POSE":
                        action, demand = r["action"].copy(), r["action"][:6].copy()
                    else:
                        origin = (targets["before_pos"][j], targets["before_rot"][j]) if mode == "delta" else before
                        delta = pose_delta(*origin, targets["after_pos"][j], targets["after_rot"][j])
                        action, demand = inverse_scale(delta, osc.input_min, osc.input_max,
                                                       osc.output_min, osc.output_max, r["action"][-1])
                    obs, _, done, _ = env.step(action.tolist())
                    record(r["index"], obs, done, action, demand, before)
                if not np.array_equal(osc.initial_joint, anchor):
                    raise ValueError("OSC conversion secretly changed nullspace anchor")
            # Same unmodified interface for both arms of comparison; no teleport / extra control mode.
            hold_action = np.r_[np.zeros(6), records[-1]["action"][-1]]
            for i in range(HOLD):
                env.regenerate_obs_from_state(env.get_sim_state())
                before = physical_pose(env, fk)
                obs, _, done, _ = env.step(hold_action.tolist())
                record(len(records) + i, obs, done, hold_action, np.zeros(6), before)
        write_json(out / f"{mode}-steps.json", rows)
        np.savez_compressed(out / f"{mode}-states.npz", after=np.asarray(states))
        np.savez_compressed(out / f"{mode}-poses.npz", before_pos=before_pos, before_rot=before_rot,
                            after_pos=after_pos, after_rot=after_rot)
        # Mixed reference action dimensions differ; only OSC candidates are rectangular seven-vectors.
        if mode != "reference":
            np.savez_compressed(out / f"{mode}-candidate-actions.npz", actions=actions, demands=demands)
        result = acceptance(rows, expected or rows, LIMITS, HOLD)
        result.update(source=source, mode=mode, controls=len(rows), start=start,
                      converted_joint_controls=sum(r["controller"] == "JOINT_POSITION" for r in records[start:]),
                      clipped_controls=sum(r["clipped_axes"] > 0 for r in rows),
                      snapshot_sha256=sha(SNAPSHOTS / source / f"snapshot-{start}.npz"),
                      reference_bitwise_verified=mode == "reference",
                      training_dataset_generated=False)
        write_json(out / f"{mode}-result.json", result)
        print(json.dumps(result), flush=True)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--worker", choices=("reference", *MODES))
    p.add_argument("--source", choices=SOURCES)
    a = p.parse_args()
    if a.worker:
        worker(a.output, a.source, a.worker)
        return 0
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(seed=42, modes=["reference", *MODES], limits=LIMITS, hold_controls=HOLD,
                    timestep_s=.05, preserve_original_osc=True, convert_joint_only=True,
                    reference_nullspace_injection=False, next_state_injection=False,
                    tracking="Expert reference next pose minus actual pose, joint controls only; not a VLA policy",
                    delta="Reference consecutive physical pose difference, joint controls only",
                    training=False, sources={},
                    code_sha256={n: sha(ROOT / "scripts" / n) for n in
                                  ("verify_osc_conversion.py", "osc_action_conversion.py", "simulation_snapshot.py",
                                   "verify_simulation_snapshot.py", "replay_mixed_correction.py")})
    for source in SOURCES:
        _, _, records, start = source_data(source)
        protocol["sources"][source] = dict(start=start, stop=len(records),
                  snapshot_sha256=sha(SNAPSHOTS / source / f"snapshot-{start}.npz"),
                  raw_metadata_sha256=sha(ROOT / "corrections" / source / "metadata.json"))
    write_json(a.output / "protocol.json", protocol)  # Gates frozen before first physics worker.
    results = []
    for source in SOURCES:
        for mode in ("reference", *MODES):
            subprocess.run([sys.executable, "-u", __file__, "--output", str(a.output),
                            "--source", source, "--worker", mode], check=True)
            result = json.loads((a.output / source / f"{mode}-result.json").read_text())
            results.append(result)
            write_json(a.output / "summary.json", results)
            if mode == "reference" and not result["passed"]:
                raise ValueError("Reference itself failed the declared physical outcome gate")
    return 0  # A faithfully recorded negative result is a completed bounded experiment.


if __name__ == "__main__":
    raise SystemExit(main())
