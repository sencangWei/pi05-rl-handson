"""Restore all declared patches onto clean pinned upstream; compare runtime hashes."""
import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.artifacts / "evidence/ppo-runtime-patch-manifest.json").read_text())
    restored = {}
    with tempfile.TemporaryDirectory(prefix="pi05-runtime-restore-") as directory:
        root = Path(directory)
        archive = root / "upstream.tar"
        # Unrelated upstream tooling contains links outside the archive root.
        # Restore only the declared runtime directories; keep safe extraction.
        directories = sorted({Path(name).parts[0] for name in manifest["files"]})
        subprocess.run(["git", "-C", str(args.upstream_repo), "archive", "--format=tar",
                        "--output", str(archive), manifest["upstream"], *directories], check=True)
        tree = root / "tree"
        tree.mkdir()
        with tarfile.open(archive) as stream:
            stream.extractall(tree, filter="data")
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        for name in manifest["patches"]:
            subprocess.run(["git", "-C", str(tree), "apply",
                            str((args.artifacts / name).resolve())], check=True)
        for name, expected in manifest["files"].items():
            digest = hashlib.sha256((tree / name).read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Restored source differs: {name}")
            restored[name] = digest
    report = {"status": "PASS", "upstream": manifest["upstream"],
              "patches": manifest["patches"], "restored_sha256": restored,
              "artifact_commit": subprocess.check_output(
                  ["git", "-C", str(args.artifacts), "rev-parse", "HEAD"], text=True).strip()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "restored_files": len(restored),
                      "patches": len(manifest["patches"])}))


if __name__ == "__main__":
    main()
