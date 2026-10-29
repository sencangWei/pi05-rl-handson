"""Public normal joint replays with bounded replay of approved human recovery."""
from collections import Counter
import json
from pathlib import Path
import numpy as np

from joint_dataset import ASSET
from replay_mixed_correction import ROOT, sha, write_json
from unified_joint_dataset import UnifiedJointDataset, norm_statistics


def mixed_slots(episodes, size=40000):
    """80% normal, 20% recovery; uniform episodes and full windows in each group."""
    if size < 5 or size % 5:
        raise ValueError("Sampler size must be a positive multiple of five")
    slots = []
    for kind, mass in (("public", size * 4 // 5), ("human", size // 5)):
        group = [i for i, record in enumerate(episodes) if record["kind"] == kind]
        if not group:
            raise ValueError(f"Missing {kind} demonstrations")
        for ordinal, episode in enumerate(group):
            count = episodes[episode]["motion_rows"] - 9
            if count < 1:
                raise ValueError("Incomplete future10 window")
            allocation = mass // len(group) + (ordinal < mass % len(group))
            slots.extend((episode, i * count // allocation) for i in range(allocation))
    return slots


class PublicJointDataset(UnifiedJointDataset):
    def __init__(self, manifest, split="train", representation="residual"):
        super().__init__(manifest, split, representation)
        if self.manifest.get("public_joint_schema") != "public-normal-human-recovery-v1":
            raise ValueError("Expected independently gated public normal data")
        if split == "train":
            self.slots = mixed_slots(self.episodes)


def train_statistics(dataset):
    stats = norm_statistics(dataset)
    states = np.asarray([dataset.labels(*slot)[0] for slot in dataset.slots], dtype=np.float64)
    # Pi05 digitizes state in [-1,1]. Population z-scores would saturate
    # ordinary joint configurations; use train-only midpoint/half-range here.
    low, high = states.min(axis=0), states.max(axis=0)
    stats["norm_stats"]["state"] = dict(mean=((low + high) / 2).tolist(),
                                        std=np.maximum((high - low) / 2, .001).tolist())
    return stats


def prepare(replays, output, human_manifest):
    replays, output = Path(replays).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, rejected = [], []
    for episode in range(50):
        folder = replays / f"demo_{episode:03d}"
        result = json.loads((folder / "result.json").read_text())
        if not result["passed"]:
            rejected.append(dict(episode=episode, result_sha256=sha(folder / "result.json")))
            continue
        if (not result["no_next_state_injection"] or not result["training_eligible"]
                or not result["pair_validation"]["passed"]
                or result["episode"] != episode or result["split"] != ("train" if episode < 40 else "holdout")
                or result["source_sha256"] != "e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406"):
            raise ValueError("Public replay lacks pinned source, physical or pairing validation")
        for name, expected in result["files_sha256"].items():
            if sha(folder / name) != expected:
                raise ValueError(f"Public replay artifact changed: {folder / name}")
        pairs, snapshot = folder / "pairs.parquet", folder / "initial-snapshot.npz"
        records.append(dict(source=f"public_goal8_demo_{episode:03d}", episode=episode,
            split="train" if episode < 40 else "holdout", kind="public", path=str(pairs),
            sha256=sha(pairs), snapshot=str(snapshot), snapshot_sha256=sha(snapshot),
            motion_rows=result["motion_rows"], paired_rows=result["paired_rows"],
            replay=str(folder / "result.json"), replay_sha256=sha(folder / "result.json"),
            provenance="official OSC demonstration physically re-executed with KP150 absolute joint commands"))
    counts = Counter(r["split"] for r in records)
    if counts["train"] < 20 or counts["holdout"] < 5:
        raise ValueError(f"Too few physically successful public replays: {dict(counts)}")
    human_manifest = Path(human_manifest).resolve()
    human = json.loads(human_manifest.read_text())
    for record in human["records"]:
        if record["kind"] != "human":
            continue
        for key in ("raw_metadata", "snapshot", "replay"):
            if sha(ROOT / record[key]) != record[key + "_sha256"]:
                raise ValueError("Approved human recording changed")
        records.append(record)
    manifest = output / "manifest.json"
    write_json(manifest, dict(schema="pi05-unified-joint-v1", ready=True,
        public_joint_schema="public-normal-human-recovery-v1", records=records,
        public_split_rule="source demo_0..39 train; demo_40..49 holdout fixed before replay outcomes",
        rejected=rejected, human_manifest_sha256=sha(human_manifest),
        sampling="80% public normal,20% human recovery; equal episode mass within category",
        public_source=dict(repository="yifengzhu-hf/LIBERO-datasets",
            revision="f13aa24a3da8c43c7225569f28c562979fa0e35a",
            file="libero_goal/put_the_bowl_on_the_plate_demo.hdf5",
            sha256="e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406"),
        action_representation="current-state anchored future absolute joint residual + grip",
        horizon=10, control_hz=20, state_dim=9, action_dim=8,
        whole_episode_split=True, diagnostic_holds_used=False,
        holdout_used_for_stats=False, raw_human_modified=False))
    train = PublicJointDataset(manifest)
    norm = output / "assets" / ASSET / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    write_json(norm, train_statistics(train))
    summary = dict(status="READY", public_train_episodes=counts["train"],
        public_holdout_episodes=counts["holdout"], human_recovery_episodes=sum(r["kind"] == "human" for r in records),
        frames_by_split_kind={str(key): sum(r["motion_rows"] for r in records if (r["split"], r["kind"]) == key)
            for key in {(r["split"], r["kind"]) for r in records}},
        source_sampling_slots=dict(Counter(train.episodes[e]["source"] for e, _ in train.slots)),
        norm_sha256=sha(norm),
        normalization="train-only state midpoint/half-range floor .001 for pi05 tokenization; sampler-weighted action population std floor .001",
        public_replay_rejected=len(rejected), diagnostic_holds_in_training=False)
    write_json(output / "preparation.json", summary)
    print(json.dumps(summary), flush=True)
    return manifest


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--human-manifest", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.replays, args.output, args.human_manifest)
