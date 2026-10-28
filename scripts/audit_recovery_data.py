"""Read-only label and sampling audit of the approved manual6 training data."""
import argparse
import hashlib
import importlib.metadata
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from multitask_dataset import window_indices
from recovery_replay_dataset import RecoveryReplayDataset

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Checksum changed: {path}")


def run_lengths(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return (np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist()


def action_metrics(actions, states, anchors):
    actions, states = np.asarray(actions), np.asarray(states)
    if (actions.shape != (len(actions), 7) or states.shape != (len(actions), 8)
            or not np.isfinite(actions).all() or not np.isfinite(states).all()):
        raise ValueError("Invalid action/state contract")
    if not 0 < anchors <= len(actions):
        raise ValueError("Invalid anchor count")
    xyz = np.linalg.norm(actions[:, :3], axis=1)
    no_xyz = xyz < 1e-8
    pose_hold = np.linalg.norm(actions[:, :6], axis=1) < 1e-8
    transition = np.flatnonzero(np.diff(np.sign(actions[:, 6])) != 0) + 1
    near_transition = np.zeros(len(actions), dtype=bool)
    for frame in transition:
        near_transition[max(0, frame - 10):min(len(actions), frame + 11)] = True
    indices = np.stack([window_indices(i, len(actions)) for i in range(anchors)])
    windows = actions[indices]
    path = np.linalg.norm(windows[:, :, :3], axis=2).sum(axis=1)
    net = np.linalg.norm(windows[:, :, :3].sum(axis=1), axis=1)
    # A label diagnostic only; a policy need not average its predicted time steps.
    cancellation = (path > 1e-8) & (net < .25 * path)
    step_mm = np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1) * 1000
    holds = run_lengths(pose_hold)
    return {"frames": len(actions), "anchors": anchors,
                "xyz_norm_quantiles": np.quantile(xyz, [0, .25, .5, .75, 1]).tolist(),
                "zero_xyz_frames": int(no_xyz.sum()), "rotation_only_frames": int((no_xyz & ~pose_hold).sum()),
                "zero_pose_frames": int(pose_hold.sum()), "zero_pose_runs": holds,
                "longest_pose_hold_controls": max(holds, default=0),
                "zero_pose_near_gripper_transition": int((pose_hold & near_transition).sum()),
                "gripper_sign_transitions": transition.tolist(),
                "low_translation_nonzero_frames": int(((xyz > 1e-8) & (xyz < .15)).sum()),
                "close_command_frames": int((actions[:, 6] > 0).sum()),
                "window_label_count": int(indices.size),
                "window_zero_pose_labels": int(pose_hold[indices].sum()),
                "all_pose_hold_windows": int(np.all(pose_hold[indices], axis=1).sum()),
                "padded_label_count": int(sum(max(0, i + 10 - len(actions)) for i in range(anchors))),
                "translation_cancellation_windows": int(cancellation.sum()),
                "consecutive_eef_displacement_mm_quantiles": np.quantile(step_mm, [.1, .5, .9]).tolist()
                if len(step_mm) else []}


def assert_alignment(actions, states, raw_actions, raw_states):
    np.testing.assert_array_equal(actions, np.asarray(raw_actions, dtype=np.float32))
    np.testing.assert_array_equal(states, np.asarray(raw_states, dtype=np.float32))


def sampling_audit(dataset, status):
    import torch
    if status["batch_size"] != 1 or status["seed"] != 42:
        raise ValueError("Audit protocol expects the manual6 batch/seed contract")
    generator = torch.Generator().manual_seed(status["seed"])
    loader = torch.utils.data.DataLoader(range(len(dataset)), batch_size=1, shuffle=True,
                                        num_workers=0, drop_last=True, generator=generator)
    indices = []
    counts, clips, replay_tasks = Counter(), Counter(), Counter()
    snapshots = {}
    for value in loader:
        index = int(value.item())
        indices.append(index)
        source, frame = dataset.route(index)
        counts[source] += 1
        record, _ = (dataset.human if source == "human" else dataset.replay).locate(frame)
        if source == "human":
            clips[str(record["episode"])] += 1
        else:
            replay_tasks[str(record["task_id"])] += 1
        if len(indices) in (status["actual_updates"], status["actual_updates"] + 1):
            snapshots[str(len(indices))] = {"sources": dict(counts), "human_clips": dict(clips),
                                               "public_tasks": dict(replay_tasks)}
        if len(indices) == status["actual_updates"] + 1:
            break
    expected = status["fetched_samples_including_prefetch"]
    if dict(counts) != expected:
        raise ValueError(f"Seeded sampler reconstruction disagrees with recorded fetches: {counts} != {expected}")
    return {"protocol": "CPU index-only torch DataLoader, same seed/batch/workers/drop_last/shuffle; not a per-gradient source log",
                "index_sha256": hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
                "recorded_prefetch_counts_match": True, "reconstructed": snapshots,
                "nominal_human_mass": .25, "nominal_public_task8_mass": .75 / 40,
                "nominal_human_share_within_task8": .25 / (.25 + .75 / 40)}


def main():
    import pyarrow.parquet as pq
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    human_path = ROOT / "evidence/manual-recovery28-manifest.json"
    replay_path = ROOT / "evidence/dataset-manifest.json"
    status_path = ROOT / "evidence/manual6_replay2000_v1-status.json"
    human, replay, status = [json.loads(p.read_text()) for p in (human_path, replay_path, status_path)]
    checked(human_path, status["human_manifest_sha256"])
    checked(replay_path, status["replay_manifest_sha256"])
    for path, expected in status["source_sha256"].items():
        checked(ROOT / path, expected)
    dataset = RecoveryReplayDataset(human_path, replay_path)
    report = {"script_sha256": digest(__file__), "executable": sys.executable,
                  "packages": {k: importlib.metadata.version(k) for k in ("numpy", "pyarrow", "torch")},
                  "manifests": {str(p.relative_to(ROOT)): digest(p) for p in (human_path, replay_path, status_path)},
                  "scope": "Train only. No labels changed. Zero pose means zero 6D command, not measured physical stillness. Gripper command is not grasp verification.",
                  "raw_sources": [], "human_clips": [], "public_task8": []}
    for audit in human["audits"]:
        directory = ROOT / "corrections" / audit["source"]
        metadata_path = directory / "metadata.json"
        checked(metadata_path, audit["metadata_sha256"])
        meta = json.loads(metadata_path.read_text())
        for name, expected in audit["files"].items():
            checked(directory / name, expected)
        with np.load(directory / "physics-000000.npz", allow_pickle=False) as first:
            initial_digest = hashlib.sha256(first["before"].tobytes()).hexdigest()
        report["raw_sources"].append({"source": audit["source"], "inventory_files": len(audit["files"]),
            "entry": meta["bank_entry"]["id"], "raw_initial_state_sha256": initial_digest,
            "bank_state_sha256": meta["bank_entry"]["state_sha256"], "control_dt": meta["control_dt"],
            "pauses_excluded": meta["pauses_excluded"], "joint_frames_archived": audit["joint_frames_archived"]})
    human_arrays = []
    for record in human["records"]:
        path = ROOT / record["path"]
        checked(path, record["sha256"])
        table = pq.read_table(path)
        rows = table.to_pylist()
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        start = int(Path(record["source_frame_range"][0]).stem.split("-")[1])
        directory = ROOT / "corrections" / record["source"]
        raw_actions, raw_states = [], []
        for i, row in enumerate(rows):
            with np.load(directory / f"frame-{start + i:06d}.npz", allow_pickle=False) as raw:
                raw_actions.append(raw["action"])
                raw_states.append(raw["state"])
                for key in ("image", "wrist_image"):
                    np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(row[key]["bytes"]))), raw[key])
            with np.load(directory / f"physics-{start + i:06d}.npz", allow_pickle=False) as physics:
                np.testing.assert_array_equal(physics["action"].astype(np.float32), raw_actions[-1])
            if row["episode_index"] != record["episode"] or row["task_index"] != 8 or row["frame_index"] != i:
                raise ValueError("Frame identity changed")
        assert_alignment(actions, states, raw_actions, raw_states)
        if len(rows) != record["frames"] or (not record["terminal"] and record["sample_frames"] != len(rows) - 9):
            raise ValueError("Episode/window boundary changed")
        metrics = action_metrics(actions, states, record["sample_frames"])
        report["human_clips"].append(dict(episode=record["episode"], source=record["source"],
                                         entry=record["entry"]["id"], terminal=record["terminal"], **metrics))
        human_arrays.append(actions)
    for record in replay["records"]:
        if record["split"] != "train" or record["task_id"] != 8:
            continue
        path = Path(replay["root"]) / record["path"]
        checked(path, record["sha256"])
        table = pq.read_table(path, columns=["actions", "state"])
        if table.num_rows != record["frames"]:
            raise ValueError("Public episode length changed")
        report["public_task8"].append(dict(episode=record["episode"], **action_metrics(
            table["actions"].to_pylist(), table["state"].to_pylist(), table.num_rows)))
    report["alignment"] = {"all_raw_inventory_hashes_match": True, "state_action_image_pixels_exact": True,
                                "physics_actions_exact_after_float32": True, "frames_checked": sum(map(len, human_arrays))}
    report["sampling"] = sampling_audit(dataset, status)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "alignment": report["alignment"], "sampling": report["sampling"]}))


if __name__ == "__main__":
    main()
