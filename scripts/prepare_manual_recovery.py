"""Snapshot the six approved bank demonstrations without relabeling joint actions."""
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from export_sft_for_ppo import sha256
from prepare_recovery_sft import validate_frame

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["human_1793114449817419797", "human_1793121358375746008",
           "human_1793122696228231033", "human_1793122982775036691",
           "human_1793123928152741170", "human_1793124070194277437"]


def osc_runs(timeline):
    runs, current = [], []
    for item in timeline:
        if item["controller"] == "OSC_POSE":
            current.append(item)
        elif item["controller"] == "JOINT_POSITION":
            if current:
                runs.append((current, False))
                current = []
        else:
            raise ValueError("Unknown recorded controller")
    if current:
        runs.append((current, True))
    return runs


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    output = ROOT / "data/manual-recovery28-v1"
    output.mkdir(parents=True, exist_ok=False)
    compatible = json.loads((ROOT / "evidence/matched-recovery-task-compatibility.json").read_text())
    bank = ROOT / "data/recovery-bowl-v3/manifest.json"
    if sha256(bank) != compatible["bank_manifest_sha256"]:
        raise ValueError("Bank identity changed")
    bank_entries = {r["id"]: r for r in json.loads(bank.read_text())["entries"]}
    records, audits = [], []
    for name in SOURCES:
        directory = ROOT / "corrections" / name
        metadata = directory / "metadata.json"
        meta = json.loads(metadata.read_text())
        if (meta["status"] != "approved" or meta["split"] != "train"
                or meta["task_suite"] != "libero_goal" or meta["task_id"] != 8
                or meta["task"] != compatible["prompt"] or meta["control_dt"] != .05
                or not meta["human_actions_only"] or not meta["pauses_excluded"]
                or meta["approval_type"] != "manual_user_with_observation_gates"
                or not all(meta["observation_gates"].values())
                or meta["bank_entry"] != bank_entries[meta["bank_entry"]["id"]]):
            raise ValueError(f"Unapproved/mismatched raw demonstration: {name}")
        inventory = {}
        for key in ("files", "physics_files", "joint_files"):
            for item in meta.get(key, []):
                path = directory / item["path"]
                if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or sha256(path) != item["sha256"]:
                    raise ValueError(f"Raw recording changed: {path}")
                inventory[item["path"]] = item["sha256"]
        timeline = meta.get("control_timeline")
        if timeline is None:
            if meta.get("joint_frame_count", 0):
                raise ValueError("Mixed control without timeline")
            timeline = [dict(controller="OSC_POSE", path=f"frame-{i:06d}.npz",
                             physics=f"physics-{i:06d}.npz") for i in range(meta["frame_count"])]
        expected_osc = [f"frame-{i:06d}.npz" for i in range(meta["frame_count"])]
        if [t["path"] for t in timeline if t["controller"] == "OSC_POSE"] != expected_osc:
            raise ValueError("OSC timeline missing/reordered controls")
        joints = [t for t in timeline if t["controller"] == "JOINT_POSITION"]
        if len(joints) != meta.get("joint_frame_count", 0):
            raise ValueError("Joint timeline count changed")
        previous = None
        for item in timeline:
            physics = item.get("physics", item["path"])
            if physics not in inventory or item["path"] not in inventory:
                raise ValueError("Timeline refers to unchecked files")
            with np.load(directory / physics, allow_pickle=False) as stored:
                before, after = stored["before"], stored["after"]
                if before.shape != (79,) or after.shape != (79,) or not np.isfinite([before, after]).all():
                    raise ValueError("Invalid physical state")
                if previous is not None and not np.array_equal(previous, before):
                    raise ValueError("Discontinuous recorded physics")
                previous = after.copy()
        for run, terminal in osc_runs(timeline):
            if len(run) < 10 and not terminal:
                raise ValueError("OSC prefix shorter than the prediction horizon")
            episode = len(records)
            rows = []
            for index, item in enumerate(run):
                with np.load(directory / item["path"], allow_pickle=False) as data:
                    frame = {key: data[key] for key in data.files}
                validate_frame(frame)
                with np.load(directory / item["physics"], allow_pickle=False) as physics:
                    np.testing.assert_array_equal(physics["action"].astype(np.float32), frame["action"])
                row = dict(state=frame["state"].tolist(), actions=frame["action"].tolist(),
                           episode_index=episode, task_index=8, frame_index=index)
                for key in ("image", "wrist_image"):
                    encoded = BytesIO()
                    Image.fromarray(frame[key]).save(encoded, format="PNG")
                    row[key] = {"bytes": encoded.getvalue(), "path": None}
                rows.append(row)
            path = output / f"episode-{episode:02d}.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            records.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path),
                frames=len(rows), sample_frames=len(rows) if terminal else len(rows) - 9,
                terminal=terminal, controller="OSC_POSE", episode=episode, task_id=8,
                prompt=compatible["prompt"], split="train", entry=meta["bank_entry"],
                source=name, source_frame_range=[run[0]["path"], run[-1]["path"]]))
        audits.append(dict(source=name, metadata_sha256=sha256(metadata), files=inventory,
                           osc_frames=meta["frame_count"], joint_frames_archived=len(joints),
                           observation_gates=meta["observation_gates"], continuous_raw_physics=True))
        print(json.dumps(dict(source=name, osc=meta["frame_count"], archived_joint=len(joints))), flush=True)
    manifest = dict(schema="pi05-bank-human-replay-v1", ready=True, root=str(ROOT), records=records,
        bank_manifest_sha256=sha256(bank), replay_manifest_sha256=sha256(ROOT / "evidence/dataset-manifest.json"),
        raw_demonstrations=6, whole_osc_demonstrations=4, mixed_demonstrations=2,
        osc_frames=sum(r["frames"] for r in records), sample_frames=sum(r["sample_frames"] for r in records),
        joint_frames_archived=sum(a["joint_frames_archived"] for a in audits), audits=audits,
        temporal_policy="Actual 20Hz OSC actions unchanged. Split at joint control; omit last nine anchors of nonterminal runs. Only terminal runs pad final action. No resampling or joint-to-OSC conversion.",
        limitation="Mixed demonstrations supervise OSC prefixes/tails only, not the intervening joint recovery. Changed nullspace targets are not policy inputs. No claim that 7D policy learns joint maneuvers.",
        validation="Full raw inventory hashes, action/physics equality, continuous full recorded state, user approval gates; not bit-exact independent trajectory replay.",
        normalization="Existing multitask parent official LIBERO stats; 25% human / 75% 40-task public train replay")
    for path in (output / "manifest.json", ROOT / "evidence/manual-recovery28-manifest.json"):
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in ("raw_demonstrations", "osc_frames", "sample_frames", "joint_frames_archived")}))


if __name__ == "__main__":
    main()
