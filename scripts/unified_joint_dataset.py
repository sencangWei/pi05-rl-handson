"""Approved human joint recordings plus whole-episode legacy joint replay."""
from collections import Counter
from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from joint_dataset import ASSET
from joint_residual import transform_chunk
from replay_mixed_correction import ROOT, sha, write_json


def balanced_slots(counts):
    """Equal mass per episode; uniformly distribute repeated shorter windows."""
    if not counts or min(counts) < 1:
        raise ValueError("Need complete action windows")
    size = max(counts)
    return [(episode, i * count // size) for episode, count in enumerate(counts) for i in range(size)]


class UnifiedJointDataset:
    def __init__(self, manifest, split="train", representation="absolute"):
        import pyarrow.parquet as pq
        if split not in ("train", "holdout") or representation not in ("absolute", "residual"):
            raise ValueError("Invalid split or representation")
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["schema"] != "pi05-unified-joint-v1" or not self.manifest["ready"]:
            raise ValueError("Unverified unified dataset")
        records = self.manifest["records"]
        if (len({r["source"] for r in records}) != len(records)
                or {r["split"] for r in records} != {"train", "holdout"}):
            raise ValueError("Whole-episode split is not disjoint")
        self.episodes, self.representation = [], representation
        for record in records:
            if record["split"] != split:
                continue
            path = ROOT / record["path"]
            if sha(path) != record["sha256"]:
                raise ValueError("Joint pairs checksum changed")
            rows = pq.read_table(path).to_pylist()
            motion = record["motion_rows"]
            if len(rows) != record["paired_rows"] or motion < 10:
                raise ValueError("Incomplete episode")
            for i, row in enumerate(rows):
                if (row["frame_index"] != i or row["controller"] != "ABSOLUTE_JOINT_POSITION"
                        or row["diagnostic_hold"] != (i >= motion)
                        or np.shape(row["state"]) != (9,) or np.shape(row["actions"]) != (8,)
                        or not np.isfinite(row["state"] + row["actions"]).all()):
                    raise ValueError("Invalid joint episode contract")
            self.episodes.append(dict(record, rows=rows))
        counts = [e["motion_rows"] - 9 for e in self.episodes]
        self.slots = balanced_slots(counts) if split == "train" else [
            (e, i) for e, count in enumerate(counts) for i in range(count)]
        self.fetched_indices = []

    def __len__(self):
        return len(self.slots)

    def labels(self, episode, start):
        record = self.episodes[episode]
        if not 0 <= start <= record["motion_rows"] - 10:
            raise IndexError(start)
        rows = record["rows"]
        state = np.asarray(rows[start]["state"], np.float32)
        actions = np.asarray([r["actions"] for r in rows[start:start + 10]], np.float32)
        if self.representation == "residual":
            actions = transform_chunk(actions, state[:7])
        return state, actions

    def sample(self, episode, start):
        state, actions = self.labels(episode, start)
        row = self.episodes[episode]["rows"][start]
        images = {}
        for key in ("image", "wrist_image"):
            image = np.asarray(Image.open(BytesIO(row[key]["bytes"])))
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError("Expected actual 256 RGB camera")
            images[key] = image
        return dict(state=state, actions=actions, prompt=row["prompt"], **images)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        item = self.sample(*self.slots[index])
        self.fetched_indices.append(int(index))
        return item


def norm_statistics(dataset):
    states, actions = zip(*(dataset.labels(*slot) for slot in dataset.slots), strict=True)
    result = {}
    for key, values in (("state", np.asarray(states)), ("actions", np.concatenate(actions))):
        values = values.astype(np.float64)
        result[key] = dict(mean=values.mean(axis=0).tolist(), std=np.maximum(values.std(axis=0), .001).tolist())
    return dict(norm_stats=result)


def prepare(batch, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    sys.path.insert(0, str(ROOT / "support/manual_recovery_ui/scripts"))
    from unified_joint_recording import load_joint_frames
    batch = Path(batch).resolve()
    spec = json.loads(batch.read_text())
    if spec["status"] != "COLLECTION_REPLAY_VERIFIED" or spec["episodes"] != 3:
        raise ValueError("Expected three verified approved takes")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for source in spec["sources"]:
        directory = Path(source["source"])
        meta_path = directory / "metadata.json"
        if sha(meta_path) != source["metadata_sha256"]:
            raise ValueError("Raw metadata changed")
        meta = json.loads(meta_path.read_text())
        replay_path = Path(source["replay_evidence"])
        replay = json.loads(replay_path.read_text())
        if (meta["status"] != "approved" or meta["split"] != "train"
                or not all(meta["observation_gates"].values())
                or not all(replay[k] for k in ("physical_states_match", "both_cameras_equal",
                    "all_joint_velocity_eef_and_timestamp_telemetry_matches"))
                or replay["saved_next_state_injection"]):
            raise ValueError("Approval / independent replay missing")
        frames = load_joint_frames(directory, meta)
        rows = []
        for i, frame in enumerate(frames):
            row = dict(frame_index=i, controller="ABSOLUTE_JOINT_POSITION", diagnostic_hold=False,
                       state=frame["state"].tolist(), actions=frame["action"].tolist(), prompt=meta["task"])
            for key in ("image", "wrist_image"):
                buffer = BytesIO()
                Image.fromarray(frame[key]).save(buffer, format="PNG")
                row[key] = dict(bytes=buffer.getvalue(), path=None)
            rows.append(row)
        path = output / (directory.name + ".parquet")
        pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")
        snapshot = directory / meta["motion_segments"][0]["path"]
        if sha(snapshot) != meta["motion_segments"][0]["sha256"]:
            raise ValueError("Full initial snapshot changed")
        records.append(dict(source=directory.name, split="train", kind="human", path=str(path.relative_to(ROOT)),
            sha256=sha(path), motion_rows=len(rows), paired_rows=len(rows),
            raw_metadata=str(meta_path.relative_to(ROOT)), raw_metadata_sha256=sha(meta_path),
            snapshot=str(snapshot.relative_to(ROOT)), snapshot_sha256=sha(snapshot),
            replay=str(replay_path.relative_to(ROOT)), replay_sha256=sha(replay_path)))
    old_path = ROOT / "data/joint-pilot29-v3/manifest.json"
    old = json.loads(old_path.read_text())
    records += [dict(r, kind="pairs", provenance="re-executed joint expert, not original historical commands")
                for r in old["records"]]
    manifest = output / "manifest.json"
    write_json(manifest, dict(schema="pi05-unified-joint-v1", ready=True, records=records,
        batch_manifest=str(batch.relative_to(ROOT)), batch_sha256=sha(batch), old_manifest_sha256=sha(old_path),
        horizon=10, state_dim=9, action_dim=8, control_hz=20, whole_episode_split=True,
        sampling="equal mass per train episode, uniform repeated full-window slots; seed42 shuffle",
        raw_modified=False, diagnostic_holds_used=False, holdout_previously_used_for_development=True))
    summaries = {}
    for representation in ("absolute", "residual"):
        folder = output / representation
        folder.mkdir()
        local = folder / "manifest.json"
        local.write_bytes(manifest.read_bytes())
        dataset = UnifiedJointDataset(local, representation=representation)
        norm = folder / "assets" / ASSET / "norm_stats.json"
        norm.parent.mkdir(parents=True)
        write_json(norm, norm_statistics(dataset))
        counts = Counter(dataset.episodes[e]["source"] for e, _ in dataset.slots)
        summaries[representation] = dict(sampler_slots=len(dataset), per_source_slots=dict(counts),
            actual_full_windows=sum(e["motion_rows"] - 9 for e in dataset.episodes), norm_sha256=sha(norm))
    write_json(output / "preparation.json", dict(status="READY", train_frames=sum(r["motion_rows"] for r in records
        if r["split"] == "train"), holdout_frames=sum(r["motion_rows"] for r in records if r["split"] == "holdout"),
        normalization="train sampler-weighted states and full action windows; population std floor0.001",
        holdout_used_for_stats=False, representations=summaries))
    print(json.dumps(summaries), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.batch, args.output)
