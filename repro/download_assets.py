#!/usr/bin/env python3
"""Download models and datasets needed for the MASC reproduction scripts."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


HF_CACHE_MODELS = {
    "llama2_base_tokenizer": "meta-llama/Llama-2-7b-hf",
    "llama2_chat_tokenizer": "meta-llama/Llama-2-7b-chat-hf",
    "muse_news_target": "muse-bench/MUSE-news_target",
    "muse_news_retrain": "muse-bench/MUSE-news_retrain",
}

LOCAL_MODEL_DIRS = {
    "tofu_full": (
        "open-unlearning/tofu_Llama-2-7b-chat-hf_full",
        ROOT / "open-unlearning/cache/hf/open-unlearning__tofu_Llama-2-7b-chat-hf_full",
    ),
    "tofu_retain90": (
        "open-unlearning/tofu_Llama-2-7b-chat-hf_retain90",
        ROOT / "open-unlearning/cache/hf/open-unlearning__tofu_Llama-2-7b-chat-hf_retain90",
    ),
}

TOFU_CONFIGS = ["full", "forget10", "retain90", "holdout10"]


def configure_cache() -> None:
    cache_root = ROOT / "cache"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "open-unlearning/cache/hf_datasets"))
    os.environ.setdefault("TMPDIR", str(cache_root / "tmp"))
    for key in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE", "TMPDIR"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def download_hf_cache_models(keys: list[str]) -> None:
    for key in keys:
        repo_id = HF_CACHE_MODELS[key]
        print(f"[model] {key}: {repo_id}")
        snapshot_download(repo_id=repo_id, cache_dir=str(ROOT / "cache/hub"), resume_download=True)


def download_local_models(keys: list[str]) -> None:
    for key in keys:
        repo_id, local_dir = LOCAL_MODEL_DIRS[key]
        print(f"[model] {key}: {repo_id} -> {local_dir}")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )


def download_tofu_datasets() -> None:
    cache_dir = os.environ["HF_DATASETS_CACHE"]
    for config in TOFU_CONFIGS:
        print(f"[dataset] locuslab/TOFU:{config}")
        load_dataset("locuslab/TOFU", config, split="train", cache_dir=cache_dir)


def materialize_muse_data() -> None:
    print("[dataset] MUSE News benchmark files -> muse_bench/data")
    subprocess.run([sys.executable, "load_data.py", "--corpus", "news"], cwd=ROOT / "muse_bench", check=True)


def download_qwen_scaling_family() -> None:
    manifest = ROOT / "exp_scaling/manifests/qwen25_tofu_scaling.json"
    print(f"[scaling] Qwen2.5 family from {manifest}")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "exp_scaling/scripts/download_family_models.py"),
            "--manifest",
            str(manifest),
        ],
        cwd=ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        choices=["all", "tofu", "news", "muse-data", "scaling"],
        default="all",
        help="Which asset group to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cache()

    if args.target in {"all", "tofu"}:
        download_hf_cache_models(["llama2_base_tokenizer", "llama2_chat_tokenizer"])
        download_local_models(["tofu_full", "tofu_retain90"])
        download_tofu_datasets()

    if args.target in {"all", "news"}:
        download_hf_cache_models(["llama2_base_tokenizer", "muse_news_target", "muse_news_retrain"])
        materialize_muse_data()

    if args.target in {"all", "muse-data"}:
        materialize_muse_data()

    if args.target in {"all", "scaling"}:
        download_qwen_scaling_family()


if __name__ == "__main__":
    main()
