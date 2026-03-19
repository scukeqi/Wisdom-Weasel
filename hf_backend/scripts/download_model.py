"""Download the official Qwen3.5 4B base model from Hugging Face."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def default_model_dir(model_id: str) -> Path:
    repo_name = model_id.split("/")[-1]
    d_drive = Path("D:/models")
    if d_drive.exists():
        return d_drive / repo_name
    return Path.home() / "models" / repo_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model snapshot.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B-Base", help="Hugging Face model ID.")
    parser.add_argument("--local-dir", default=None, help="Destination directory for the model snapshot.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only fetch tokenizer/config metadata instead of weight files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir) if args.local_dir else default_model_dir(args.model_id)
    local_dir = local_dir.expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(args.model_id)

    allow_patterns = [
        "*.json",
        "*.model",
        "*.tiktoken",
        "*.txt",
        "*.py",
        "*.md",
        "tokenizer*",
        "merges.txt",
        "vocab.json",
    ]
    if not args.metadata_only:
        allow_patterns.extend(["*.safetensors", "*.safetensors.index.json"])

    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )

    manifest_path = local_dir / "download_manifest.json"
    manifest = {
        "model_id": args.model_id,
        "snapshot_path": snapshot_path,
        "local_dir": str(local_dir),
        "sha": info.sha,
        "safetensors": info.safetensors.parameters if info.safetensors else None,
        "metadata_only": args.metadata_only,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
