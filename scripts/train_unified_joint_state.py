"""State-conditioned rerun; preserve the original frozen comparison unchanged."""
from copy import deepcopy
from dataclasses import replace

import numpy as np

import train_unified_joint as baseline
from replay_mixed_correction import ROOT, sha, write_json
from unified_joint_dataset import UnifiedJointDataset

_original_config = baseline.config
_original_plan = baseline.make_plan


def state_input_canary(cfg, manifest):
    """Exercise the actual official training transform, including normalization."""
    from openpi.training.data_loader import transform_dataset
    data = UnifiedJointDataset(manifest)
    sample = data[0]
    factory = cfg.data.create(cfg.assets_dirs, cfg.model)
    stats = factory.norm_stats["state"]
    sample["state"] = np.asarray(stats.mean - .25 * stats.std, np.float32)
    transformed = transform_dataset([deepcopy(sample)], factory)[0]
    checks = []
    for dimension in range(9):
        altered = deepcopy(sample)
        altered["state"][dimension] += .5 * stats.std[dimension]
        after = transform_dataset([altered], factory)[0]
        changed = not np.array_equal(transformed["tokenized_prompt"], after["tokenized_prompt"])
        if not changed:
            raise ValueError(f"Measured state dimension {dimension} does not reach model tokens")
        np.testing.assert_array_equal(transformed["actions"], after["actions"])
        for camera in transformed["image"]:
            np.testing.assert_array_equal(transformed["image"][camera], after["image"][camera])
        checks.append(changed)
    return dict(discrete_state_input=cfg.model.discrete_state_input, dimensions_checked=9,
                each_state_dimension_changes_model_tokens=checks, images_and_actions_unchanged=True,
                state_shape=list(transformed["state"].shape), actions_shape=list(transformed["actions"].shape),
                prompt=sample["prompt"], token_count=int(transformed["tokenized_prompt_mask"].sum()))


def config(plan, manifest):
    cfg = _original_config(plan, manifest)
    cfg = replace(cfg, name="pi05_unified_joint_state_" + plan["representation"] + "_lora",
                  model=replace(cfg.model, discrete_state_input=True))
    evidence = state_input_canary(cfg, manifest)
    # The existing train/evaluate entrypoints both use this same configuration.
    output = ROOT / "reports/unified-joint-state-training-20261029" / plan["representation"]
    write_json(output / "state-input-canary.json", evidence)
    return cfg


def make_plan(manifest, representation, experiment):
    plan = _original_plan(manifest, representation, experiment)
    plan["schema"] = "pi05-unified-joint-state-plan-v2"
    plan["train"]["state_input"] = "normalized measured 7 joints + 2 fingers encoded as PI05 prompt tokens"
    paths = [ROOT / "scripts" / name for name in
             ("train_unified_joint_state.py", "evaluate_unified_joint_state.py")]
    paths += [ROOT / "third_party/openpi/src/openpi" / name for name in
              ("models/pi0.py", "models/tokenizer.py", "transforms.py", "training/config.py")]
    plan["inputs_sha256"].update({str(p.relative_to(ROOT)): sha(p) for p in paths})
    return plan


def install():
    baseline.config = config
    baseline.make_plan = make_plan


if __name__ == "__main__":
    install()
    baseline.main()
