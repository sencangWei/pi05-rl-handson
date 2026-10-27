"""Approved complete human episodes mixed with the existing 40-task training replay."""
from bisect import bisect_right
from collections import Counter, OrderedDict
import json
from pathlib import Path

import numpy as np

from export_sft_for_ppo import sha256
from multitask_dataset import MultitaskDataset


class HumanClips(MultitaskDataset):
    def __init__(self, manifest):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest["schema"] != "pi05-human-replay-v1" or not self.manifest["ready"]:
            raise ValueError("Human dataset is not ready")
        self.root = Path(self.manifest["root"])
        self.records = self.manifest["records"]
        if not self.records or any(r["split"] != "train" for r in self.records):
            raise ValueError("Only approved training episodes may enter the mixture")
        for record in self.records:
            path = (self.root / record["path"]).resolve()
            if not path.is_relative_to(self.root.resolve()) or sha256(path) != record["sha256"]:
                raise ValueError("Human parquet changed")
        self.ends = np.cumsum([r["frames"] for r in self.records])
        self.cache = OrderedDict()

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        clip = bisect_right(self.ends, index)
        return self.records[clip], index - (int(self.ends[clip - 1]) if clip else 0)


class RecoveryReplayDataset:
    def __init__(self, human_manifest, replay_manifest):
        self.human = HumanClips(human_manifest)
        if sha256(replay_manifest) != self.human.manifest["replay_manifest_sha256"]:
            raise ValueError("Public replay manifest changed")
        self.replay = MultitaskDataset(replay_manifest)
        self.fetched = Counter()

    def __len__(self):
        return 4 * len(self.replay)

    def route(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        group, slot = divmod(index, 4)
        if slot == 0:
            return "human", group % len(self.human)
        return "replay", (3 * group + slot - 1) % len(self.replay)

    def __getitem__(self, index):
        source, frame = self.route(index)
        self.fetched[source] += 1
        return (self.human if source == "human" else self.replay)[frame]
