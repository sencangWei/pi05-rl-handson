"""Approved complete human episodes mixed with the existing 40-task training replay."""
from bisect import bisect_right
from collections import Counter, OrderedDict
import json
from pathlib import Path

import numpy as np

from export_sft_for_ppo import sha256
from multitask_dataset import MultitaskDataset


class HumanClips(MultitaskDataset):
    schema = "pi05-human-replay-v1"

    def __init__(self, manifest):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest["schema"] != self.schema or not self.manifest["ready"]:
            raise ValueError("Recovery dataset is not ready")
        self.root = Path(self.manifest["root"])
        self.records = self.manifest["records"]
        if not self.records or any(r["split"] != "train" for r in self.records):
            raise ValueError("Only approved training episodes may enter the mixture")
        for record in self.records:
            path = (self.root / record["path"]).resolve()
            if not path.is_relative_to(self.root.resolve()) or sha256(path) != record["sha256"]:
                raise ValueError("Recovery parquet changed")
        self.ends = np.cumsum([r.get("sample_frames", r["frames"]) for r in self.records])
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        clip = bisect_right(self.ends, index)
        return self.records[clip], index - (int(self.ends[clip - 1]) if clip else 0)


class ModelRecoveryClips(HumanClips):
    schema = "pi05-model-recovery-replay-v1"

    def __init__(self, manifest):
        super().__init__(manifest)
        from recovery_recording import validate_training_entry
        coverage = {key: set() for key in ("empty_grasp", "object_offset", "placement_failure")}
        for record in self.records:
            validate_training_entry(record["entry"])
            if not record["independent_replay_verified"] or record["task_id"] != 8:
                raise ValueError("Model demonstration lacks independent recovery verification")
            for key, digest in (("physics", "physics_sha256"),
                                ("recording_identity", "recording_identity_sha256")):
                path = (self.root / record[key]).resolve()
                if not path.is_relative_to(self.root.resolve()) or sha256(path) != record[digest]:
                    raise ValueError("Recovery replay evidence changed")
            identity = json.loads((self.root / record["recording_identity"]).read_text())
            if identity["split"] != "train" or identity["manifest_sha256"] != self.manifest["bank_manifest_sha256"]:
                raise ValueError("Recovery teacher bank mismatch")
            coverage[record["entry"]["category"]].add(record["entry"]["source_initial_id"])
        if not all(coverage.values()) or len(coverage["placement_failure"]) < 2:
            raise ValueError("Insufficient independently replayed recovery coverage")


class BankHumanClips(HumanClips):
    """Actual OSC labels only; windows never cross a joint-control interruption."""
    schema = "pi05-bank-human-replay-v1"

    def __init__(self, manifest):
        super().__init__(manifest)
        from recovery_recording import validate_training_entry
        for record in self.records:
            validate_training_entry(record["entry"])
            expected = record["frames"] if record["terminal"] else record["frames"] - 9
            if record["sample_frames"] != expected or expected < 1:
                raise ValueError("Nonterminal window would cross a controller boundary")
            if record["controller"] != "OSC_POSE" or record["task_id"] != 8:
                raise ValueError("Only actual 7D OSC labels for goal task 8 are supported")


class RecoveryReplayDataset:
    def __init__(self, human_manifest, replay_manifest, public_slots=3):
        if isinstance(public_slots, bool) or not isinstance(public_slots, int) or public_slots < 1:
            raise ValueError("public_slots must be a positive integer")
        self.public_slots = public_slots
        schema = json.loads(Path(human_manifest).read_text())["schema"]
        model_data = schema == ModelRecoveryClips.schema
        self.source_name = "model_recovery" if model_data else "human"
        loader = {ModelRecoveryClips.schema: ModelRecoveryClips,
                  BankHumanClips.schema: BankHumanClips, HumanClips.schema: HumanClips}[schema]
        self.human = loader(human_manifest)
        if sha256(replay_manifest) != self.human.manifest["replay_manifest_sha256"]:
            raise ValueError("Public replay manifest changed")
        self.replay = MultitaskDataset(replay_manifest)
        self.fetched = Counter()
        self.fetched_indices = []

    def __len__(self):
        return (getattr(self, "public_slots", 3) + 1) * len(self.replay)

    def route(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        public_slots = getattr(self, "public_slots", 3)
        group, slot = divmod(index, public_slots + 1)
        if slot == 0:
            return getattr(self, "source_name", "human"), group % len(self.human)
        return "replay", (public_slots * group + slot - 1) % len(self.replay)

    def describe(self, index):
        source, frame = self.route(index)
        record, anchor = (self.replay if source == "replay" else self.human).locate(frame)
        return dict(index=int(index), source=source, episode=record["path"],
                    task_id=record["task_id"], anchor=int(anchor))

    def __getitem__(self, index):
        source, frame = self.route(index)
        sample = (self.replay if source == "replay" else self.human)[frame]
        self.fetched[source] += 1
        self.fetched_indices.append(int(index))
        return sample
