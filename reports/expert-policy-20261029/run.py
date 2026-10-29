"""Run four frozen expert-initial policy trials with isolated, bounded local services."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_expert_snapshots import verify_inputs  # noqa: E402

OUTPUT = ROOT / "artifacts/20261029-expert-policy"


def properties(unit):
    raw = subprocess.check_output(["systemctl", "--user", "show", unit,
        "-p", "ActiveState", "-p", "Result", "-p", "ExecMainStatus"], text=True)
    return dict(line.split("=", 1) for line in raw.splitlines())


def main():
    os.chdir(ROOT)
    env = dict(os.environ, PYTHONPATH="/home/robot/voice-arm-sim/.local/openpi/third_party/libero",
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    python = str(ROOT / ".venv-ppo/bin/python")
    subprocess.run([python, "scripts/evaluate_expert_snapshots.py", "--freeze", "--output", str(OUTPUT)],
                   env=env, check=True)
    protocol = json.loads((OUTPUT / "protocol.json").read_text())
    verify_inputs(protocol, weights=True)
    results = []
    for label, checkpoint in protocol["checkpoints"].items():
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("127.0.0.1", 9875))
        unit = f"pi05-expert-policy29-{label}-server"
        log = ROOT / f"logs/expert-policy29-{label}-server.log"
        assert (ROOT / checkpoint).is_dir() and not log.exists()
        subprocess.run(["systemd-run", "--user", "--unit=" + unit,
            "--property=WorkingDirectory=" + str(ROOT), "--property=MemoryMax=32G", "--property=MemorySwapMax=0",
            "--property=RuntimeMaxSec=1800", "--property=StandardOutput=append:" + str(log),
            "--property=StandardError=append:" + str(log), "--setenv=XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "--setenv=OMP_NUM_THREADS=4", "--setenv=OPENBLAS_NUM_THREADS=4",
            str(ROOT / ".venv-train/bin/python"), "-u", "scripts/serve_sft.py",
            "--checkpoint", str(ROOT / checkpoint), "--port", "9875"], check=True)
        try:
            started = time.monotonic()
            while not (log.exists() and "server listening on" in log.read_text()):
                assert properties(unit)["ActiveState"] == "active", log.read_text() if log.exists() else unit
                if time.monotonic() - started > 900:
                    raise TimeoutError("Policy startup deadline")
                time.sleep(5)
            for source in protocol["sources"]:
                with (ROOT / f"logs/expert-policy29-{label}-{source}.log").open("x") as output:
                    subprocess.run([python, "-u", "scripts/evaluate_expert_snapshots.py",
                        "--port", "9875", "--label", label, "--source", source,
                        "--output", str(OUTPUT)], env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
                results.append(json.loads((OUTPUT / label / source / "result.json").read_text()))
                (OUTPUT / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        finally:
            subprocess.run(["systemctl", "--user", "stop", unit], check=True)
    verify_inputs(protocol, weights=True)
    print("FOUR_EXPERT_TRAIN_INITIAL_DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
