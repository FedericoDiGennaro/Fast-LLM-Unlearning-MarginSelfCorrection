#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download all base checkpoints listed in a scaling manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to scaling manifest JSON.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of model keys to download.",
    )
    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Optional Hugging Face snapshot allow patterns.",
    )
    parser.add_argument(
        "--ignore_patterns",
        nargs="*",
        default=["*.h5", "*.msgpack", "*.ot", "*.onnx"],
        help="Optional Hugging Face snapshot ignore patterns.",
    )
    return parser.parse_args()


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())

    local_root = Path(manifest["download"]["local_dir_root"])
    local_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.only) if args.only else None
    download_records = []

    for entry in manifest["models"]:
        key = entry["key"]
        if selected is not None and key not in selected:
            continue

        repo_id = entry["hf_repo_id"]
        local_dir = local_root / sanitize_repo_id(repo_id)
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        print(f"[download] {key}: {repo_id} -> {local_dir}")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.ignore_patterns,
            resume_download=True,
        )
        download_records.append(
            {
                "key": key,
                "repo_id": repo_id,
                "local_dir": str(local_dir),
                "snapshot_path": str(path),
            }
        )

    out_path = manifest_path.parent / f"{manifest['family'].lower().replace('.', '')}_downloads.json"
    out_path.write_text(json.dumps(download_records, indent=2))
    print(f"[done] Wrote download manifest to {out_path}")


if __name__ == "__main__":
    main()
