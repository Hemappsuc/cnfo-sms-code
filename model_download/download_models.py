import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIZES = ("0.6B", "1.7B", "4B", "8B", "14B", "32B")


def save_manifest(path, manifest):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Download the six original dense Qwen3 models into models/")
    parser.add_argument("--models", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--model-root", type=Path, default=ROOT / "models")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    for size in dict.fromkeys(args.models):
        repo = f"Qwen/Qwen3-{size}"
        target = args.model_root / f"Qwen3-{size}"
        manifest_path = target / "download_manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous["repo_id"] != repo or previous["requested_revision"] != args.revision:
                raise ValueError(f"Different model or revision already exists: {target}")
            revision = previous["revision"]
        else:
            revision = HfApi().model_info(repo, revision=args.revision).sha
        target.mkdir(parents=True, exist_ok=True)
        manifest = {"repo_id": repo, "requested_revision": args.revision, "revision": revision, "complete": False}
        save_manifest(manifest_path, manifest)
        snapshot_download(
            repo_id=repo, revision=revision, local_dir=target,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "LICENSE"],
        )
        for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if not (target / required).is_file():
                raise ValueError(f"Missing {required} in {target}")
        if not list(target.glob("*.safetensors")):
            raise ValueError(f"Missing model weights: {target}")
        index = target / "model.safetensors.index.json"
        if index.exists():
            for shard in set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()):
                if not (target / shard).is_file():
                    raise ValueError(f"Missing model shard: {shard}")
        files = {}
        for path in sorted(target.iterdir()):
            if not path.is_file() or path.name == "download_manifest.json":
                continue
            checksum = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    checksum.update(block)
            files[path.name] = {"bytes": path.stat().st_size, "sha256": checksum.hexdigest()}
        manifest.update(complete=True, files=files)
        save_manifest(manifest_path, manifest)
        print(f"{repo}@{revision} -> {target}", flush=True)


if __name__ == "__main__":
    main()
