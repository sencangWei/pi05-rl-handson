"""Apply one version-checked reward patch to our independent RLinf clone."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/RLinf"
REVISION = "30e0236981c6a1dc19202de64116834c12673828"
RELATIVE = "rlinf/envs/sim/libero/libero_env.py"
OLD = "        step_penalty = -1 if self.use_step_penalty else 0"
NEW = "        step_penalty = -float(self.cfg.get(\"step_penalty_coef\", 1.0)) if self.use_step_penalty else 0"


def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    if revision != REVISION:
        raise ValueError("Unexpected RLinf revision")
    original = subprocess.check_output(["git", "show", f"HEAD:{RELATIVE}"], cwd=SOURCE, text=True)
    if original.count(OLD) != 1:
        raise ValueError("Reward source contract changed")
    expected = original.replace(OLD, NEW)
    path = SOURCE / RELATIVE
    if path.read_text() not in (original, expected):
        raise ValueError("Refusing to overwrite other RLinf changes")
    ast.parse(expected)
    path.write_text(expected)
    patch = subprocess.check_output(["git", "diff", "--", RELATIVE], cwd=SOURCE)
    (ROOT / "configs/rlinf-step-cost.patch").write_bytes(patch)
    record = dict(revision=revision, file=RELATIVE, original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        patched_sha256=hashlib.sha256(expected.encode()).hexdigest(), reward="success*1 - 0.001 per control step",
        use_rel_reward=False, training_updates=0, semantics="LIBERO original success predicate; no grasp bonus; no learned completion model")
    (ROOT / "evidence/reward-integration.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
