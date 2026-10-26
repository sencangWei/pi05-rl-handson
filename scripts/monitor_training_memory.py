"""Record only the selected systemd training unit's cgroup, until it exits."""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("unit")
parser.add_argument("output", type=Path)
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as stream:
    while True:
        properties = dict(line.split("=", 1) for line in subprocess.check_output(
            ["systemctl", "--user", "show", args.unit, "-p", "ActiveState", "-p", "Result",
             "-p", "ExecMainStatus", "-p", "ControlGroup", "-p", "MemoryPeak"], text=True).splitlines())
        record = {"time": time.time(), **properties}
        cgroup = Path("/sys/fs/cgroup") / properties["ControlGroup"].lstrip("/")
        for field in ("memory.current", "memory.peak", "memory.events", "memory.stat", "pids.current"):
            path = cgroup / field
            if properties["ControlGroup"] and path.is_file():
                record[field] = path.read_text().strip()
        stream.write(json.dumps(record) + "\n")
        stream.flush()
        if properties["ActiveState"] not in ("active", "activating", "deactivating"):
            break
        time.sleep(10)
