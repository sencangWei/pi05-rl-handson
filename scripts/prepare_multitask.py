"""Download the immutable official 40-task LIBERO dataset and freeze episode splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "physical-intelligence/libero"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ensure_metadata(api, root):
    siblings = {r["rfilename"]: r for r in api["siblings"]}
    for name in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        path = root / name
        if not path.exists():
            with urllib.request.urlopen(f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{name}", timeout=60) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = path.read_bytes()
        git_hash = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
        if git_hash != siblings[name]["blobId"]:
            raise ValueError("Dataset metadata checksum mismatch: " + name)


def make_manifest(api, root, per_task=0):
    if api["sha"] != REVISION:
        raise ValueError("Unexpected dataset revision")
    tasks = {r["task"]: r["task_index"] for r in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    episodes = list(map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines()))
    siblings = {r["rfilename"]: r for r in api["siblings"]}
    selected = []
    for task, task_id in sorted(tasks.items(), key=lambda x: x[1]):
        group = [r for r in episodes if r["tasks"] == [task]]
        group.sort(key=lambda r: hashlib.sha256(f"42:{r['episode_index']}".encode()).hexdigest())
        if per_task:
            group = group[:per_task]
        if len(group) < 3:
            raise ValueError("Need at least three episodes per task")
        holdout = max(1, round(len(group) * .1))
        for index, r in enumerate(group):
            episode = r["episode_index"]
            relative = f"data/chunk-{episode // 1000:03d}/episode_{episode:06d}.parquet"
            item = siblings[relative]
            selected.append(dict(episode=episode, task_id=task_id, prompt=task, frames=r["length"],
                split="holdout" if index < holdout else "train", path=relative,
                bytes=item["size"], sha256=item["lfs"]["sha256"]))
    if len(tasks) != 40 or len({r["episode"] for r in selected}) != len(selected):
        raise ValueError("Expected 40 tasks and unique episodes")
    return dict(schema="pi05-multitask-v1", repo=REPO, revision=REVISION, seed=42,
        ready=False, root=str(root.resolve()), horizon=10, records=selected,
        train_episodes=sum(r["split"] == "train" for r in selected),
        holdout_episodes=sum(r["split"] == "holdout" for r in selected),
        bytes=sum(r["bytes"] for r in selected), sampling="uniform task, then approximately uniform control frame",
        holdout_scope="Excluded from this adaptation only; official parent may have seen all public episodes",
        tail="Repeat final action within episode; never cross episodes",
        normalization="Unchanged official pi05_libero parent statistics; no new fitting")


def download(record, root):
    path = root / record["path"]
    if path.exists() and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"]:
        return "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{record['path']}?download=true"
    for attempt in range(4):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as f:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    h.update(block)
                    f.write(block)
            if partial.stat().st_size != record["bytes"] or h.hexdigest() != record["sha256"]:
                raise ValueError("File checksum/size mismatch: " + record["path"])
            partial.replace(path)
            return "downloaded"
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=0, help="0 downloads all 1,693 episodes")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    root = ROOT / "data/libero"
    api = json.loads((ROOT / "evidence/hf-dataset-api.json").read_text())
    ensure_metadata(api, root)
    manifest = make_manifest(api, root, args.per_task)
    path = ROOT / "evidence/dataset-manifest.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["records"] != manifest["records"]:
            raise ValueError("Refusing to change an existing experiment split")
    write_json(path, manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}), flush=True)
    completed, errors = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # Reach all 40 tasks early, instead of downloading an entire task first.
        groups = [[r for r in manifest["records"] if r["task_id"] == task] for task in range(40)]
        ordered = [group[index] for index in range(max(map(len, groups))) for group in groups if index < len(group)]
        futures = {pool.submit(download, r, root): r for r in ordered}
        for future in as_completed(futures):
            record = futures[future]
            try:
                status = future.result()
                completed += 1
                print(f"{completed}/{len(futures)} episode={record['episode']} {status}", flush=True)
            except Exception as error:
                errors.append(dict(episode=record["episode"], error=str(error)))
                print(f"ERROR episode={record['episode']} {error}", flush=True)
            write_json(ROOT / "evidence/download-status.json", dict(completed=completed, total=len(futures), errors=errors))
    manifest["ready"] = not errors
    write_json(path, manifest)
    if errors:
        raise RuntimeError(f"{len(errors)} files failed; rerun safely reuses verified files")


if __name__ == "__main__":
    main()
