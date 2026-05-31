#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from transformers import TrainerCallback


REPO_ROOT = Path(__file__).resolve().parents[1] / "open-unlearning"
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data import get_data, get_collators  # noqa: E402
from model import get_model  # noqa: E402
from trainer import load_trainer  # noqa: E402
from trainer.utils import seed_everything  # noqa: E402


class EpochWindowTimer(TrainerCallback):
    def __init__(self, epochs_to_time: int):
        self.epochs_to_time = epochs_to_time
        self.start_perf = None
        self.end_perf = None
        self.start_wall = None
        self.end_wall = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.start_perf is None:
            self.start_perf = time.perf_counter()
            self.start_wall = datetime.now(timezone.utc)
            print(
                f"[timer] start epoch window at epoch={state.epoch} step={state.global_step} "
                f"utc={self.start_wall.isoformat()}",
                flush=True,
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.start_perf is None or self.end_perf is not None:
            return
        if state.epoch is not None and state.epoch >= float(self.epochs_to_time):
            self.end_perf = time.perf_counter()
            self.end_wall = datetime.now(timezone.utc)
            print(
                f"[timer] end epoch window at epoch={state.epoch} step={state.global_step} "
                f"utc={self.end_wall.isoformat()}",
                flush=True,
            )

    @property
    def duration_seconds(self):
        if self.start_perf is None or self.end_perf is None:
            return None
        return self.end_perf - self.start_perf

    def finalize_if_missing(self):
        if self.start_perf is not None and self.end_perf is None:
            self.end_perf = time.perf_counter()
            self.end_wall = datetime.now(timezone.utc)


def update_common_json(path: Path, method_name: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data[method_name] = payload
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Timing-only OpenUnlearning train wrapper.")
    parser.add_argument("--method", required=True)
    parser.add_argument(
        "--result_key",
        default=None,
        help="Key to store in common JSON (defaults to method name).",
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--forget_split", default="forget10")
    parser.add_argument("--retain_split", default="retain90")
    parser.add_argument("--holdout_split", default="holdout10")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--common_json", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trainer_arg",
        action="append",
        default=[],
        help="Override trainer.args with key=value (JSON values allowed).",
    )
    parser.add_argument(
        "--method_arg",
        action="append",
        default=[],
        help="Override trainer.method_args with key=value (JSON values allowed).",
    )
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(
            config_name="train.yaml",
            overrides=[
                f"experiment=unlearn/tofu/default.yaml",
                f"trainer={args.method}",
                f"model={args.model_name}",
                f"forget_split={args.forget_split}",
                f"retain_split={args.retain_split}",
                f"holdout_split={args.holdout_split}",
            ],
        )

    with open_dict(cfg):
        cfg.eval = None
        cfg.paths.root_dir = str(REPO_ROOT)
        cfg.paths.output_dir = args.output_dir
        cfg.task_name = Path(args.output_dir).name
        cfg.model.model_args.pretrained_model_name_or_path = args.model_path
        cfg.model.tokenizer_args.pretrained_model_name_or_path = args.model_path
        if "attn_implementation" in cfg.model.model_args:
            del cfg.model.model_args["attn_implementation"]

        targs = cfg.trainer.args
        targs.num_train_epochs = args.epochs
        targs.learning_rate = args.learning_rate
        targs.per_device_train_batch_size = args.per_device_train_batch_size
        targs.gradient_accumulation_steps = args.gradient_accumulation_steps
        targs.output_dir = args.output_dir
        targs.logging_dir = str(Path(args.output_dir) / "logs")
        targs.do_eval = False
        targs.eval_strategy = "no"
        targs.eval_on_start = False
        targs.save_strategy = "no"
        targs.report_to = "none"
        targs.remove_unused_columns = False
        targs.seed = args.seed
        if args.method_arg:
            for item in args.method_arg:
                if "=" not in item:
                    raise ValueError(f"Expected key=value for --method_arg, got: {item}")
                key, raw_value = item.split("=", 1)
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError:
                    value = raw_value
                cfg.trainer.method_args[key] = value
        if args.trainer_arg:
            for item in args.trainer_arg:
                if "=" not in item:
                    raise ValueError(f"Expected key=value for --trainer_arg, got: {item}")
                key, raw_value = item.split("=", 1)
                try:
                    value = json.loads(raw_value)
                except json.JSONDecodeError:
                    value = raw_value
                cfg.trainer.args[key] = value

    seed_everything(cfg.trainer.args.seed)

    model, tokenizer = get_model(cfg.model)
    data = get_data(
        cfg.data,
        mode="unlearn",
        tokenizer=tokenizer,
        template_args=cfg.model.template_args,
    )
    collator = get_collators(cfg.collator, tokenizer=tokenizer)
    trainer, _ = load_trainer(
        trainer_cfg=cfg.trainer,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=None,
        template_args=cfg.model.template_args,
    )

    timer = EpochWindowTimer(epochs_to_time=args.epochs)
    trainer.add_callback(timer)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print("[timer] composed config:", flush=True)
    print(OmegaConf.to_yaml(cfg), flush=True)

    trainer.train()
    timer.finalize_if_missing()

    result = {
        "method": args.method,
        "model_name": args.model_name,
        "model_path": args.model_path,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "holdout_split": args.holdout_split,
        "epochs_measured": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "start_utc": timer.start_wall.isoformat() if timer.start_wall else None,
        "end_utc": timer.end_wall.isoformat() if timer.end_wall else None,
        "duration_seconds": timer.duration_seconds,
        "estimated_10_epoch_seconds": timer.duration_seconds * (10.0 / args.epochs)
        if timer.duration_seconds is not None
        else None,
        "global_step": trainer.state.global_step,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    result_key = args.result_key or args.method
    result["result_key"] = result_key
    update_common_json(Path(args.common_json), result_key, result)

    print("[timer] result:", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
