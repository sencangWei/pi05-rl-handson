"""Fetch and verify the separate official PPO parent; does not connect the JAX adapter."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "RLinf/RLinf-Pi05-LIBERO-SFT"
REVISION = "45ccfcc4e28634f1576ebf78cab0fbe2fd82432d"


def main():
    api = json.loads((ROOT / "evidence/rlinf-model-api.json").read_text())
    if api["sha"] != REVISION or api["id"] != REPO:
        raise ValueError("Unexpected PPO parent identity")
    files = [item for item in api["siblings"] if item["rfilename"] != ".gitattributes"]
    output = ROOT / "models/RLinf-Pi05-LIBERO-SFT"
    status_path = ROOT / "evidence/rlinf-model-validation.json"
    status = dict(ready=False, repo=REPO, revision=REVISION, root=str(output),
                  lineage="Separate official PPO parent; not the new JAX LoRA checkpoint", files={})
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    os.environ["HF_HOME"] = str(ROOT / ".local/ppo-hf")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, revision=REVISION, local_dir=output,
                      allow_patterns=[item["rfilename"] for item in files], max_workers=2)
    for item in files:
        path = (output / item["rfilename"]).resolve()
        if not path.is_relative_to(output.resolve()) or path.stat().st_size != item["size"]:
            raise ValueError("PPO parent path/size mismatch")
        digest = hashlib.sha256() if "lfs" in item else hashlib.sha1()
        if "lfs" not in item:
            digest.update(f"blob {item['size']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        expected = item["lfs"]["sha256"] if "lfs" in item else item["blobId"]
        if digest.hexdigest() != expected:
            raise ValueError("PPO parent checksum mismatch: " + item["rfilename"])
        status["files"][item["rfilename"]] = dict(bytes=item["size"],
            algorithm="sha256" if "lfs" in item else "git-blob-sha1", digest=expected)
    status["ready"] = True
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
