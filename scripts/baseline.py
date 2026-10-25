"""Evaluate the existing genuine VLA service without changing its checkpoint or GUI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
APP = Path("/home/robot/voice-arm-sim")
APP_REVISION = "b0334d6424d85de642ac06248ddfe2009c36ee45"


def configure_libero():
    base = APP / ".local/openpi/third_party/libero/libero/libero"
    config = ROOT / ".local/libero"
    config.mkdir(parents=True, exist_ok=True)
    values = {"benchmark_root": base, "bddl_files": base / "bddl_files", "init_states": base / "init_files",
              "assets": base / "assets", "datasets": ROOT / "data/libero"}
    (config / "config.yaml").write_text("".join(f"{key}: {value}\n" for key, value in values.items()))
    os.environ["LIBERO_CONFIG_PATH"] = str(config)


class SeededPolicy:
    def __init__(self, policy, seed):
        self.policy, self.seed, self.index = policy, seed, 0
        self._ws = policy._ws

    def get_server_metadata(self):
        return self.policy.get_server_metadata()

    def infer(self, payload):
        payload = dict(payload, _probe_noise_seed=self.seed + self.index)
        self.index += 1
        return self.policy.infer(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 7, 8, 9])
    parser.add_argument("--initial-states", type=int, nargs="+", default=[0])
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--verify-bowl-release", action="store_true", help="Require no finger contact for five consecutive successful controls on task 8")
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=APP, text=True).strip()
    if actual != APP_REVISION:
        raise ValueError("Baseline source revision changed; review before running")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU rendering; VLA remains in existing server
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".local/numba"))
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    sys.path.insert(0, str(APP))
    from vla import run
    run.configure_libero = configure_libero
    reports = []
    for task in args.tasks:
        for initial in args.initial_states:
            options = run.parse_args(["--suite", "libero_goal", "--task-id", str(task),
                "--initial-state", str(initial), "--output", str(args.output / f"goal-{task:02d}-init-{initial:02d}"),
                "--seed", "42", "--max-steps", "300", "--replan-steps", "5", "--port", str(args.port)])
            check = None
            if args.verify_bowl_release and task == 8:
                from vla.evaluate_distillation import ReleasedBowlOnPlateCheck
                check = ReleasedBowlOnPlateCheck()
            report = run.rollout(options, success_check=check, policy_factory=lambda port: SeededPolicy(run.connect_policy(port), 42))
            # Same exact sparse/step reward as the PPO recipe. This records rewards, NOT policy updates.
            report["reward_return"] = (1.0 if report["result"] == "success" else 0.0) - .001 * report["steps"]
            report["training_updates"] = 0
            reports.append(report)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps({k: report[k] for k in ("task_id", "initial_state", "result", "steps", "reward_return")}), flush=True)
    return int(any(r["result"] == "exception" for r in reports))


if __name__ == "__main__":
    raise SystemExit(main())
