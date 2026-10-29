"""Independent absolute-joint pilot: whole-episode split and train-only statistics."""
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from replay_mixed_correction import ROOT, SOURCES, sha, write_json

ASSET = "joint_contract29"


class JointDataset:
    def __init__(self, manifest, split="train"):
        import pyarrow.parquet as pq
        if split not in ("train", "holdout"):
            raise ValueError("Unknown split")
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest["schema"] != "pi05-joint-pilot-v1" or not self.manifest["ready"]:
            raise ValueError("Unverified joint dataset")
        records = self.manifest["records"]
        if (len(records) != 2 or {r["split"] for r in records} != {"train", "holdout"}
                or len({r["source"] for r in records}) != 2):
            raise ValueError("Expected disjoint whole-episode train/holdout split")
        self.record = next(r for r in records if r["split"] == split)
        path = ROOT / self.record["path"]
        if sha(path) != self.record["sha256"]:
            raise ValueError("Joint pairs checksum changed")
        self.rows = pq.read_table(path).to_pylist()
        self.motion = self.record["motion_rows"]
        if len(self.rows) != self.record["paired_rows"] or self.motion < 10:
            raise ValueError("Incomplete joint episode")
        for i, row in enumerate(self.rows):
            if (row["frame_index"] != i or row["controller"] != "ABSOLUTE_JOINT_POSITION"
                    or row["diagnostic_hold"] != (i >= self.motion)):
                raise ValueError("Invalid joint timeline/controller")
            if (np.shape(row["state"]) != (9,) or np.shape(row["actions"]) != (8,)
                    or not np.isfinite(row["state"] + row["actions"]).all()):
                raise ValueError("Expected finite 9D joint state / 8D absolute target")
        self.fetched_indices = []

    def __len__(self):
        return self.motion - 9  # Full future10 only, no terminal/diagnostic padding.

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        row = self.rows[index]
        images = {}
        for key in ("image", "wrist_image"):
            image = np.asarray(Image.open(BytesIO(row[key]["bytes"])))
            if image.shape != (256, 256, 3) or image.dtype != np.uint8:
                raise ValueError("Expected actual RGB uint8 camera")
            images[key] = image
        self.fetched_indices.append(int(index))
        return dict(state=np.asarray(row["state"], np.float32),
                    actions=np.asarray([r["actions"] for r in self.rows[index:index + 10]], np.float32),
                    prompt=row["prompt"], **images)


def prepare(pairs, output):
    """Verify persisted pre-action RGB/physics first; never use holdout in stats."""
    from record_osc_pairs import validate_dataset
    pairs, output = Path(pairs).resolve(), Path(output).resolve()
    manifest = json.loads((pairs / "manifest.json").read_text())
    if not manifest["machine_validated"] or not manifest["prior_source_human_approval"]:
        raise ValueError("Expert pair validation/source approval missing")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, source in enumerate(SOURCES):
        folder = pairs / source
        result = json.loads((folder / "result.json").read_text())
        for name, expected in result["files_sha256"].items():
            if sha(folder / name) != expected:
                raise ValueError("Joint collection artifact changed")
        validate_dataset(folder / "pairs.parquet", folder / "physics.npz", folder / "audit.json")
        records.append(dict(source=source, split="train" if i == 0 else "holdout",
                            path=str((folder / "pairs.parquet").relative_to(ROOT)),
                            sha256=sha(folder / "pairs.parquet"), motion_rows=result["motion_rows"],
                            paired_rows=result["paired_rows"]))
    path = output / "manifest.json"
    write_json(path, dict(schema="pi05-joint-pilot-v1", ready=True, records=records,
                         source_manifest_sha256=sha(pairs / "manifest.json"),
                         whole_episode_split=True, horizon=10, state_dim=9, action_dim=8,
                         diagnostic_holds_used=False, terminal_window_padding=False,
                         old_osc_data_modified=False, new_execution_human_review=False,
                         scope="Interface pilot from previously human-approved sources, not recovery acceptance"))
    train = JointDataset(path)
    stats = {}
    for key, field in (("state", "state"), ("actions", "actions")):
        values = np.asarray([r[field] for r in train.rows[:train.motion]], np.float64)
        stats[key] = dict(mean=values.mean(axis=0).tolist(),
                          std=np.maximum(values.std(axis=0), .001).tolist())
    norm = output / "assets" / ASSET / "norm_stats.json"
    norm.parent.mkdir(parents=True)
    write_json(norm, dict(norm_stats=stats))
    write_json(output / "statistics-provenance.json", dict(source=records[0]["source"],
               frames=train.motion, std="population, floor 0.001 per dimension", normalization="z-score",
               action_weighting="one vote per real motion frame, no repeated horizon weighting",
               holdout_used=False, diagnostic_hold_used=False, norm_sha256=sha(norm)))
    return path


@dataclass(frozen=True)
class JointOutputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :8]}


@dataclass(frozen=True)
class JointDataFactory:
    assets: str

    def create(self, assets_dirs, model_config):
        from openpi import transforms as t
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.shared import normalize
        from openpi.training.config import DataConfig, ModelTransformFactory
        return DataConfig(repo_id=ASSET, asset_id=ASSET,
            norm_stats=normalize.load(Path(self.assets) / ASSET), use_quantile_norm=False,
            repack_transforms=t.Group(inputs=[t.RepackTransform({"observation/state": "state",
                "observation/image": "image", "observation/wrist_image": "wrist_image",
                "actions": "actions", "prompt": "prompt"})]),
            data_transforms=t.Group(inputs=[LiberoInputs(model_config.model_type)], outputs=[JointOutputs()]),
            model_transforms=ModelTransformFactory()(model_config))


def make_config(experiment, manifest, steps=200, lr=5e-5):
    import dataclasses
    import training_config
    cfg = training_config.make_config(experiment, steps, 1, lr)
    return dataclasses.replace(cfg, name="pi05_joint_contract_lora",
        data=JointDataFactory(str(Path(manifest).parent / "assets")))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    print(prepare(a.pairs, a.output))
