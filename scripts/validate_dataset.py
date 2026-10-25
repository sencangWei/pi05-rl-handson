"""Validate every episode's identity, numeric rows, and first/last camera images."""
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from prepare_multitask import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "evidence/dataset-manifest.json"
    manifest = json.loads(path.read_text())
    if not manifest["ready"]:
        raise ValueError("Download not complete")
    frames = 0
    for record in manifest["records"]:
        table = pq.read_table(Path(manifest["root"]) / record["path"])
        length = record["frames"]
        if table.num_rows != length:
            raise ValueError("Episode length mismatch")
        for key, expected in (("episode_index", record["episode"]), ("task_index", record["task_id"])):
            if not np.all(np.asarray(table[key]) == expected):
                raise ValueError("Episode/task identity mismatch")
        for key, dim in (("state", 8), ("actions", 7)):
            array = np.asarray(table[key].to_pylist())
            if array.shape != (length, dim) or not np.isfinite(array).all():
                raise ValueError("Invalid numeric data")
        for index in (0, length - 1):
            row = table.slice(index, 1).to_pylist()[0]
            for key in ("image", "wrist_image"):
                image = Image.open(BytesIO(row[key]["bytes"]))
                image.load()
                if image.size != (256, 256) or image.mode != "RGB":
                    raise ValueError("Image contract mismatch")
        frames += length
    result = dict(passed=True, manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        episodes=len(manifest["records"]), frames=frames,
        scope="All numeric rows/identities; first and last RGB frames per camera; download separately verifies every file hash")
    write_json(ROOT / "evidence/data-validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
