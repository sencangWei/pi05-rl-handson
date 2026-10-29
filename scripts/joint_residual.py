"""Observation-anchored joint targets; gripper is absolute, never accumulated."""
from dataclasses import replace
from pathlib import Path

import numpy as np

from joint_dataset import JointDataFactory, JointDataset, make_config


def transform_chunk(actions, anchor, *, decode=False):
    actions, anchor = np.asarray(actions), np.asarray(anchor)
    if (actions.shape != (10, 8) or anchor.shape != (7,) or
            not np.isfinite(actions).all() or not np.isfinite(anchor).all()):
        raise ValueError("Expected finite (10,8) targets and one (7,) observation anchor")
    output = actions.astype(np.result_type(actions.dtype, anchor.dtype, np.float32), copy=True)
    output[:, :7] += anchor if decode else -anchor
    return output


class ResidualDataset(JointDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        item["actions"] = transform_chunk(item["actions"], item["state"][:7])
        return item


def residual_config(experiment, manifest):
    cfg = make_config(experiment, manifest)
    return replace(cfg, name="pi05_joint_residual_contract_lora",
                   data=JointDataFactory(str(Path(manifest).parent / "assets")))
