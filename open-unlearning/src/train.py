import json
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
from omegaconf import DictConfig
from transformers import TrainerCallback

from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


def _evaluation_enabled(cfg: DictConfig) -> bool:
    trainer_args = cfg.trainer.args
    if trainer_args.get("do_eval", True) is False:
        return False
    strategy = str(
        trainer_args.get(
            "eval_strategy", trainer_args.get("evaluation_strategy", "no")
        )
    ).lower()
    return strategy != "no" or bool(trainer_args.get("eval_on_start", False))


class PreciseTrainTimingCallback(TrainerCallback):
    """Record wall-clock timing from the first optimizer step to train end."""

    def __init__(self, output_dir: str, metadata: dict | None = None):
        self.output_dir = Path(output_dir)
        self.metadata = metadata or {}
        self.step1_perf_counter = None
        self.step1_wall_time_utc = None

    def on_step_end(self, args, state, control, **kwargs):
        if self.step1_perf_counter is None and state.global_step >= 1:
            self.step1_perf_counter = time.perf_counter()
            self.step1_wall_time_utc = datetime.now(timezone.utc).isoformat()

    def on_train_end(self, args, state, control, **kwargs):
        timing_path = self.output_dir / "train_timing.json"
        record = {
            "timing_source": "PreciseTrainTimingCallback_v1",
            **self.metadata,
            "step1_wall_time_utc": self.step1_wall_time_utc,
            "train_end_wall_time_utc": datetime.now(timezone.utc).isoformat(),
            "global_step_at_train_end": state.global_step,
            "epoch_at_train_end": state.epoch,
            "step1_to_train_end_seconds": None,
        }
        if self.step1_perf_counter is not None:
            record["step1_to_train_end_seconds"] = (
                time.perf_counter() - self.step1_perf_counter
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timing_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs and _evaluation_enabled(cfg):
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        processing_class=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
    )
    timing_metadata = {
        "task_name": cfg.get("task_name"),
        "mode": mode,
        "trainer_handler": trainer_cfg.get("handler"),
        "data_split": cfg.get("data_split"),
        "forget_split": cfg.get("forget_split"),
        "retain_split": cfg.get("retain_split"),
        "seed": trainer_args.seed,
        "output_dir": str(trainer_args.output_dir),
    }
    trainer.add_callback(PreciseTrainTimingCallback(trainer_args.output_dir, timing_metadata))

    if trainer_args.do_train:
        trainer.train()
        if not cfg.get("skip_final_save", False):
            trainer.save_state()
            trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
