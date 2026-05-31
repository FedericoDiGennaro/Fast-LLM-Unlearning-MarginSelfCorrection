#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(".")
OPEN_UNLEARNING_ROOT = ROOT / "open-unlearning"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch OpenUnlearning TOFU retain-only finetuning jobs across a scaling manifest."
    )
    parser.add_argument("--manifest", required=True, help="Path to scaling manifest JSON.")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of model keys to finetune.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=None,
        help="Override the number of accelerator processes.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=None,
        help="Override the per-device train batch size.",
    )
    parser.add_argument(
        "--effective_batch_size_target",
        type=int,
        default=None,
        help="Override the target effective batch size.",
    )
    parser.add_argument(
        "--retain_split",
        default="retain90",
        help="TOFU retain split to train on.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def sanitize_repo_id(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def compute_grad_accum(effective_batch: int, per_device_batch: int, num_processes: int) -> int:
    denom = per_device_batch * num_processes
    if effective_batch % denom != 0:
        raise ValueError(
            f"effective_batch_size_target={effective_batch} is not divisible by "
            f"per_device_train_batch_size * num_processes = {denom}"
        )
    return effective_batch // denom


def build_command(entry, cfg, local_model_dir: Path, output_dir: Path, args):
    num_processes = args.num_processes or cfg["num_processes"]
    per_device_batch = args.per_device_train_batch_size or cfg["per_device_train_batch_size"]
    effective_batch = args.effective_batch_size_target or cfg["effective_batch_size_target"]
    grad_accum = compute_grad_accum(effective_batch, per_device_batch, num_processes)

    retain_split = args.retain_split
    task_name = f"scaling_{entry['key']}_tofu_{retain_split}"

    cmd = [
        "accelerate",
        "launch",
        "--config_file",
        "configs/accelerate/default_config.yaml",
        "--num_processes",
        str(num_processes),
        "src/train.py",
        "experiment=finetune/tofu/default.yaml",
        f"task_name={task_name}",
        f"model={entry['open_unlearning_model']}",
        f"model.model_args.pretrained_model_name_or_path={local_model_dir}",
        f"model.tokenizer_args.pretrained_model_name_or_path={local_model_dir}",
        "data/datasets@data.train=TOFU_QA_retain",
        f"data.train.TOFU_QA_retain.args.hf_args.name={retain_split}",
        f"trainer.args.per_device_train_batch_size={per_device_batch}",
        f"trainer.args.gradient_accumulation_steps={grad_accum}",
        f"trainer.args.learning_rate={cfg['learning_rate']}",
        f"trainer.args.num_train_epochs={cfg['num_train_epochs']}",
        f"trainer.args.gradient_checkpointing={str(cfg['gradient_checkpointing']).lower()}",
        f"trainer.args.ddp_find_unused_parameters={str(cfg['ddp_find_unused_parameters']).lower()}",
        "trainer.args.do_eval=false",
        "trainer.args.eval_on_start=false",
        "trainer.args.eval_strategy=no",
        f"paths.output_dir={output_dir}",
    ]
    if cfg.get("disable_flash_attention", False):
        cmd.insert(10, "~model.model_args.attn_implementation")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{OPEN_UNLEARNING_ROOT / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
    env["HF_DATASETS_OFFLINE"] = env.get("HF_DATASETS_OFFLINE", "1")
    env["HF_HOME"] = str(OPEN_UNLEARNING_ROOT / "cache" / "hf_home")
    env["HF_DATASETS_CACHE"] = str(OPEN_UNLEARNING_ROOT / "cache" / "hf_datasets")
    env["TRANSFORMERS_CACHE"] = str(OPEN_UNLEARNING_ROOT / "cache" / "hf")
    env["HF_HUB_CACHE"] = str(OPEN_UNLEARNING_ROOT / "cache" / "hf")

    header = (
        f"[launch] {entry['key']} | retain_split={retain_split} | "
        f"per_device_batch={per_device_batch} | grad_accum={grad_accum} | "
        f"effective_batch={effective_batch} | epochs={cfg['num_train_epochs']}"
    )
    return cmd, env, header


def main():
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    cfg = manifest["tofu_full_finetune"]
    output_root = Path(cfg["output_root"]) / manifest["family"].lower().replace(".", "")
    log_root = Path(cfg["log_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    selected = set(args.only) if args.only else None

    for entry in manifest["models"]:
        if selected is not None and entry["key"] not in selected:
            continue

        local_model_dir = Path(manifest["download"]["local_dir_root"]) / sanitize_repo_id(entry["hf_repo_id"])
        output_dir = output_root / entry["key"] / f"tofu_{args.retain_split}"
        log_file = log_root / f"finetune_tofu_{args.retain_split}_{entry['key']}.log"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd, env, header = build_command(entry, cfg, local_model_dir, output_dir, args)

        print(header)
        print(" ".join(shlex.quote(x) for x in cmd))

        if args.dry_run:
            continue

        if not local_model_dir.exists():
            raise FileNotFoundError(
                f"Missing local model dir for {entry['key']}: {local_model_dir}. "
                "Run the download step first."
            )

        with log_file.open("w") as f:
            f.write(header + "\n")
            f.flush()
            subprocess.run(
                cmd,
                cwd=OPEN_UNLEARNING_ROOT,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )


if __name__ == "__main__":
    main()
