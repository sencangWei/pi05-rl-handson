"""Fetch the pinned public LIBERO bowl task with verified, resumable byte ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request


REPO = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SOURCE_PATH = "libero_goal/put_the_bowl_on_the_plate_demo.hdf5"
SIZE = 468246288
SHA256 = "e69528b0cf10dfc59b20698e12ec2affc03f3887309034d3eb74cac3ec929406"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{SOURCE_PATH}"
CHUNK_BYTES = 8 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata():
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/libero_goal?expand=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        entries = json.load(response)
    entry = next(item for item in entries if item["path"] == SOURCE_PATH)
    if entry["size"] != SIZE or entry["lfs"]["oid"] != SHA256:
        raise ValueError("Pinned upstream size/LFS hash changed")
    return entry


def fetch_part(folder, index):
    start = index * CHUNK_BYTES
    end = min(start + CHUNK_BYTES, SIZE) - 1
    expected = end - start + 1
    path = folder / f"part-{index:04d}"
    if path.exists() and path.stat().st_size == expected:
        return index, expected, "cached"
    for attempt in range(4):
        try:
            request = urllib.request.Request(
                URL + f"?byte_range={start}-{end}",
                headers={"Range": f"bytes={start}-{end}"},
            )
            temporary = path.with_suffix(".partial")
            with urllib.request.urlopen(request, timeout=120) as response:
                if (response.status != 206 or
                        response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}"):
                    raise ValueError("Server did not honor exact byte range")
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, 1024 * 1024)
            if temporary.stat().st_size != expected:
                raise ValueError("Truncated byte range")
            temporary.replace(path)
            return index, expected, "downloaded"
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed-partial", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entry = metadata()
    started = time.monotonic()
    parts = args.output.parent / "source-byte-ranges"
    parts.mkdir(exist_ok=True)
    count = (SIZE + CHUNK_BYTES - 1) // CHUNK_BYTES
    if args.seed_partial and args.seed_partial.exists():
        with args.seed_partial.open("rb") as source:
            for index in range(min(source.seek(0, 2) // CHUNK_BYTES, count)):
                source.seek(index * CHUNK_BYTES)
                path = parts / f"part-{index:04d}"
                if not path.exists():
                    path.write_bytes(source.read(CHUNK_BYTES))
    if not args.output.exists():
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_part, parts, index) for index in range(count)]
            for future in as_completed(futures):
                index, size, origin = future.result()
                completed += size
                print(json.dumps(dict(part=index, origin=origin, completed_bytes=completed,
                    size=SIZE, elapsed_s=round(time.monotonic() - started, 1))), flush=True)
        combined = args.output.with_suffix(".assembling")
        with combined.open("wb") as output:
            for index in range(count):
                with (parts / f"part-{index:04d}").open("rb") as source:
                    shutil.copyfileobj(source, output, 1024 * 1024)
        if combined.stat().st_size != SIZE or sha256(combined) != SHA256:
            raise ValueError("Complete source failed official LFS SHA256 verification")
        combined.replace(args.output)
    if args.output.stat().st_size != SIZE or sha256(args.output) != SHA256:
        raise ValueError("Existing source failed official LFS SHA256 verification")
    result = dict(schema="pi05-public-joint-source-v1", status="COMPLETE", repo=REPO,
        revision=REVISION, source_path=SOURCE_PATH, url=URL, local_path=str(args.output.resolve()),
        size=SIZE, sha256=SHA256, official_lfs_sha256_exact_match=True,
        upstream_metadata=entry, wall_seconds=round(time.monotonic() - started, 2))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "local_path", "size", "sha256")}), flush=True)


if __name__ == "__main__":
    main()
