"""Task-balanced, episode-contained action windows from checksum-verified Parquet."""
from bisect import bisect_right
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path

import numpy as np


def window_indices(frame, length, horizon=10):
    if not 0 <= frame < length:
        raise IndexError(frame)
    return np.minimum(np.arange(frame, frame + horizon), length - 1)


class MultitaskDataset:
    def __init__(self, manifest, require_ready=True):
        self.manifest = json.loads(Path(manifest).read_text())
        if self.manifest["schema"] != "pi05-multitask-v1" or (require_ready and not self.manifest["ready"]):
            raise ValueError("Dataset download/verification is not ready")
        self.root = Path(self.manifest["root"])
        self.groups = []
        for task in range(40):
            records = [r for r in self.manifest["records"] if r["split"] == "train" and r["task_id"] == task]
            if not records:
                raise ValueError(f"Missing training task {task}")
            self.groups.append((records, np.cumsum([r["frames"] for r in records])))
        self.maximum = max(int(ends[-1]) for _, ends in self.groups)
        self.cache = OrderedDict()

    def __len__(self):
        return self.maximum * 40

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        records, ends = self.groups[index % 40]
        frame = (index // 40) * int(ends[-1]) // self.maximum
        clip = bisect_right(ends, frame)
        return records[clip], frame - (int(ends[clip - 1]) if clip else 0)

    def __getitem__(self, index):
        import pyarrow.parquet as pq
        from PIL import Image
        record, frame = self.locate(index)
        path = self.root / record["path"]
        if path not in self.cache:
            table = pq.read_table(path)
            if table.num_rows != record["frames"]:
                raise ValueError("Episode length changed")
            self.cache[path] = table
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        table = self.cache[path]
        row = table.slice(frame, 1).to_pylist()[0]
        if row["episode_index"] != record["episode"] or row["task_index"] != record["task_id"]:
            raise ValueError("Episode/task identity mismatch")
        indices = window_indices(frame, table.num_rows)
        actions = np.asarray(table["actions"].take(indices).to_pylist(), dtype=np.float32)
        state = np.asarray(row["state"], dtype=np.float32)
        if actions.shape != (10, 7) or state.shape != (8,) or not np.isfinite(actions).all() or not np.isfinite(state).all():
            raise ValueError("Invalid LIBERO state/action contract")
        images = {key: np.asarray(Image.open(BytesIO(row[key]["bytes"])).convert("RGB")) for key in ("image", "wrist_image")}
        if any(image.shape != (256, 256, 3) for image in images.values()):
            raise ValueError("Invalid image shape")
        return dict(state=state, actions=actions, prompt=record["prompt"], **images)
