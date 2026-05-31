#!/usr/bin/env python3
"""
Teacher-forced local function-space diagnostic for TOFU full vs retain models.

This mirrors the MUSE diagnostic but adapts the example construction to TOFU's
question-answer format as implemented in open-unlearning. Each example is
formatted with the Llama-2 chat template, and token margins are scored only on
the supervised answer tokens (`labels != -100`).

Outputs:
  - forget_margins.csv
  - retain_margins.csv
  - summary.json
  - summary.txt
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


LLAMA2_CHAT_TEMPLATE = {
    "apply_chat_template": False,
    "user_start_tag": "[INST] ",
    "user_end_tag": " [/INST]",
    "asst_start_tag": "",
    "asst_end_tag": " ",
}


def preprocess_chat_instance(
    tokenizer,
    template_config: dict,
    prompt_msgs: str | list[str],
    response_msgs: str | list[str],
    max_length: int,
    predict_with_generate: bool = False,
) -> dict[str, torch.Tensor]:
    """Lightweight copy of open-unlearning's QA chat preprocessing."""
    assert len(prompt_msgs) == len(response_msgs) if isinstance(prompt_msgs, list) else True
    if isinstance(prompt_msgs, str):
        assert isinstance(response_msgs, str)
        prompt_msgs, response_msgs = [prompt_msgs], [response_msgs]

    if template_config["apply_chat_template"]:
        raise NotImplementedError("This diagnostic currently expects manual template formatting.")

    wrapped_prompt = ""
    system_prompt_with_special_tokens = template_config.get("system_prompt_with_special_tokens")
    if system_prompt_with_special_tokens:
        wrapped_prompt += system_prompt_with_special_tokens

    n_few_shot = len(prompt_msgs) - 1
    for i in range(n_few_shot):
        fs_prompt, fs_response = prompt_msgs[i], response_msgs[i]
        wrapped_prompt += (
            template_config["user_start_tag"]
            + fs_prompt
            + template_config["user_end_tag"]
            + template_config["asst_start_tag"]
            + fs_response
            + template_config["asst_end_tag"]
        )

    final_prompt, final_response = prompt_msgs[-1], response_msgs[-1]
    wrapped_prompt += (
        template_config["user_start_tag"]
        + final_prompt
        + template_config["user_end_tag"]
        + template_config["asst_start_tag"]
    )
    chat_ids = tokenizer(
        wrapped_prompt + final_response,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )["input_ids"]
    prompt_ids = tokenizer(
        wrapped_prompt,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )["input_ids"]

    if chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]

    len_matched = len(prompt_ids)
    if predict_with_generate:
        input_ids = prompt_ids
        labels = chat_ids
    else:
        input_ids = chat_ids
        labels = [IGNORE_INDEX] * len_matched + chat_ids[len_matched:]

    item = {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor([1] * len(input_ids)),
    }
    return item


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_fp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def model_primary_device(model: AutoModelForCausalLM, fallback: torch.device) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def unload_model(model: AutoModelForCausalLM) -> None:
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_dir: str, device: torch.device) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=get_fp_dtype(device),
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model


def truncate_text(text: str, max_chars: int = 180) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def summarize(values: list[float]) -> dict:
    t = torch.tensor(values, dtype=torch.float32)
    return {
        "mean": float(t.mean().item()),
        "median": float(t.median().item()),
        "min": float(t.min().item()),
        "max": float(t.max().item()),
    }


def load_tofu_pairs(split: str, cache_dir: str | None, max_samples: int | None) -> list[dict]:
    ds = load_dataset("locuslab/TOFU", name=split, split="train", cache_dir=cache_dir)
    rows: list[dict] = []
    for idx, item in enumerate(ds):
        rows.append(
            {
                "index": idx,
                "question": item["question"],
                "answer": item["answer"],
                "question_preview": truncate_text(item["question"]),
                "answer_preview": truncate_text(item["answer"]),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    print(f"TOFU {split}: loaded {len(rows)} samples")
    return rows


def pad_or_trim(tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
    if tensor.dtype not in (torch.int32, torch.int64):
        tensor = tensor.to(torch.long)
    if tensor.size(0) < target_length:
        return torch.cat(
            [tensor, torch.full((target_length - tensor.size(0),), pad_value, dtype=tensor.dtype)]
        )
    return tensor[:target_length]


def encode_qa_pairs(
    tokenizer,
    pairs: list[dict],
    max_length: int,
) -> tuple[list[dict], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    metadata: list[dict] = []

    for item in pairs:
        processed = preprocess_chat_instance(
            tokenizer=tokenizer,
            template_config=LLAMA2_CHAT_TEMPLATE,
            prompt_msgs=item["question"],
            response_msgs=item["answer"],
            max_length=max_length,
            predict_with_generate=False,
        )
        input_ids.append(pad_or_trim(processed["input_ids"], max_length, pad_id))
        labels.append(pad_or_trim(processed["labels"], max_length, -100))
        attention_masks.append(pad_or_trim(processed["attention_mask"], max_length, 0))
        metadata.append(
            {
                "index": item["index"],
                "question_preview": item["question_preview"],
                "answer_preview": item["answer_preview"],
                "valid_label_tokens": int((processed["labels"] != -100).sum().item()),
            }
        )

    return metadata, input_ids, labels, attention_masks


def token_margin_stats_from_batch(logits: torch.Tensor, labels: torch.Tensor) -> list[dict]:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid_mask = shift_labels != -100

    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gold_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    gold_nll = -gold_log_probs

    top2_vals, top2_idx = shift_logits.topk(k=2, dim=-1)
    gold_logits = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    top1_is_gold = top2_idx[..., 0].eq(safe_labels)
    competitor_logits = torch.where(top1_is_gold, top2_vals[..., 1], top2_vals[..., 0])
    margins = gold_logits - competitor_logits

    batch_stats: list[dict] = []
    for i in range(shift_labels.size(0)):
        valid = valid_mask[i]
        if valid.sum().item() == 0:
            batch_stats.append(
                {
                    "avg_nll": 0.0,
                    "mean_margin": 0.0,
                    "min_margin": 0.0,
                    "frac_margin_le_0": 0.0,
                    "frac_margin_le_0p5": 0.0,
                    "frac_margin_le_1p0": 0.0,
                    "valid_tokens": 0,
                }
            )
            continue

        ex_nll = gold_nll[i][valid]
        ex_margins = margins[i][valid]
        batch_stats.append(
            {
                "avg_nll": float(ex_nll.mean().item()),
                "mean_margin": float(ex_margins.mean().item()),
                "min_margin": float(ex_margins.min().item()),
                "frac_margin_le_0": float((ex_margins <= 0).float().mean().item()),
                "frac_margin_le_0p5": float((ex_margins <= 0.5).float().mean().item()),
                "frac_margin_le_1p0": float((ex_margins <= 1.0).float().mean().item()),
                "valid_tokens": int(valid.sum().item()),
            }
        )
    return batch_stats


def compute_margin_stats(
    model: AutoModelForCausalLM,
    input_ids: list[torch.Tensor],
    labels: list[torch.Tensor],
    attention_masks: list[torch.Tensor],
    batch_size: int,
    fallback_device: torch.device,
) -> list[dict]:
    stats: list[dict] = []
    model_device = model_primary_device(model, fallback_device)

    with torch.no_grad():
        for start in range(0, len(input_ids), batch_size):
            batch_ids = torch.stack(input_ids[start : start + batch_size]).to(model_device)
            batch_labels = torch.stack(labels[start : start + batch_size]).to(model_device)
            batch_mask = torch.stack(attention_masks[start : start + batch_size]).to(model_device)
            outputs = model(batch_ids, attention_mask=batch_mask)
            stats.extend(token_margin_stats_from_batch(outputs.logits, batch_labels))

    return stats


def compute_split_stats_for_model(
    model_dir: str,
    forget_ids: list[torch.Tensor],
    forget_labels: list[torch.Tensor],
    forget_masks: list[torch.Tensor],
    retain_ids: list[torch.Tensor],
    retain_labels: list[torch.Tensor],
    retain_masks: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
    label: str,
) -> tuple[list[dict], list[dict]]:
    print(f"Loading {label} from {model_dir}")
    model = load_model(model_dir, device)
    forget_stats = compute_margin_stats(
        model, forget_ids, forget_labels, forget_masks, batch_size, device
    )
    retain_stats = compute_margin_stats(
        model, retain_ids, retain_labels, retain_masks, batch_size, device
    )
    unload_model(model)
    return forget_stats, retain_stats


def make_rows(
    split: str,
    metadata: list[dict],
    full_stats: list[dict],
    retain_model_stats: list[dict],
) -> list[dict]:
    rows: list[dict] = []
    for meta, full, retain_model in zip(metadata, full_stats, retain_model_stats):
        row = {
            "split": split,
            "index": meta["index"],
            "question_preview": meta["question_preview"],
            "answer_preview": meta["answer_preview"],
            "valid_label_tokens": meta["valid_label_tokens"],
        }
        for key, value in full.items():
            row[f"full_{key}"] = value
        for key, value in retain_model.items():
            row[f"retain_model_{key}"] = value
        shared_keys = set(full.keys()) & set(retain_model.keys())
        for key in shared_keys:
            if key.endswith("tokens"):
                continue
            row[f"delta_{key}"] = retain_model[key] - full[key]
        rows.append(row)
    return rows


def write_rows_csv(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_split(rows: list[dict]) -> dict:
    mean_margin_delta = [row["delta_mean_margin"] for row in rows]
    min_margin_delta = [row["delta_min_margin"] for row in rows]
    frac0_delta = [row["delta_frac_margin_le_0"] for row in rows]
    frac05_delta = [row["delta_frac_margin_le_0p5"] for row in rows]
    frac10_delta = [row["delta_frac_margin_le_1p0"] for row in rows]

    return {
        "full": {
            "avg_nll": summarize([row["full_avg_nll"] for row in rows]),
            "mean_margin": summarize([row["full_mean_margin"] for row in rows]),
            "min_margin": summarize([row["full_min_margin"] for row in rows]),
            "frac_margin_le_0": summarize([row["full_frac_margin_le_0"] for row in rows]),
            "frac_margin_le_0p5": summarize([row["full_frac_margin_le_0p5"] for row in rows]),
            "frac_margin_le_1p0": summarize([row["full_frac_margin_le_1p0"] for row in rows]),
        },
        "retain_model": {
            "avg_nll": summarize([row["retain_model_avg_nll"] for row in rows]),
            "mean_margin": summarize([row["retain_model_mean_margin"] for row in rows]),
            "min_margin": summarize([row["retain_model_min_margin"] for row in rows]),
            "frac_margin_le_0": summarize([row["retain_model_frac_margin_le_0"] for row in rows]),
            "frac_margin_le_0p5": summarize([row["retain_model_frac_margin_le_0p5"] for row in rows]),
            "frac_margin_le_1p0": summarize([row["retain_model_frac_margin_le_1p0"] for row in rows]),
        },
        "deltas": {
            "avg_nll": summarize([row["delta_avg_nll"] for row in rows]),
            "mean_margin": summarize(mean_margin_delta),
            "min_margin": summarize(min_margin_delta),
            "frac_margin_le_0": summarize(frac0_delta),
            "frac_margin_le_0p5": summarize(frac05_delta),
            "frac_margin_le_1p0": summarize(frac10_delta),
        },
        "counts": {
            "retain_model_lowers_mean_margin": int(sum(x < 0 for x in mean_margin_delta)),
            "retain_model_raises_mean_margin": int(sum(x > 0 for x in mean_margin_delta)),
            "retain_model_lowers_min_margin": int(sum(x < 0 for x in min_margin_delta)),
            "retain_model_raises_min_margin": int(sum(x > 0 for x in min_margin_delta)),
            "retain_model_increases_frac_margin_le_0": int(sum(x > 0 for x in frac0_delta)),
            "retain_model_increases_frac_margin_le_0p5": int(sum(x > 0 for x in frac05_delta)),
        },
    }


def probe_aligned_summary(forget_rows: list[dict], retain_rows: list[dict]) -> dict:
    def row_mean(rows: list[dict], key: str) -> float:
        return float(torch.tensor([row[key] for row in rows], dtype=torch.float32).mean().item())

    full = {
        "retain_mean_margin": row_mean(retain_rows, "full_mean_margin"),
        "forget_mean_margin": row_mean(forget_rows, "full_mean_margin"),
    }
    full["retain_forget_margin_gap"] = full["retain_mean_margin"] - full["forget_mean_margin"]

    retain_model = {
        "retain_mean_margin": row_mean(retain_rows, "retain_model_mean_margin"),
        "forget_mean_margin": row_mean(forget_rows, "retain_model_mean_margin"),
    }
    retain_model["retain_forget_margin_gap"] = (
        retain_model["retain_mean_margin"] - retain_model["forget_mean_margin"]
    )

    delta = {key: retain_model[key] - full[key] for key in full.keys()}
    return {
        "full": full,
        "retain_model": retain_model,
        "delta_retain_minus_full": delta,
    }


def top_rows(rows: list[dict], key: str, k: int, reverse: bool = False) -> list[dict]:
    return sorted(rows, key=lambda row: row[key], reverse=reverse)[:k]


def format_example_rows(title: str, rows: list[dict]) -> str:
    lines = [title]
    for row in rows:
        lines.append(
            f"  idx={row['index']:>4}  "
            f"full_mean={row['full_mean_margin']:+.4f}  "
            f"retain_mean={row['retain_model_mean_margin']:+.4f}  "
            f"d_mean={row['delta_mean_margin']:+.4f}  "
            f"full_min={row['full_min_margin']:+.4f}  "
            f"retain_min={row['retain_model_min_margin']:+.4f}  "
            f"d_min={row['delta_min_margin']:+.4f}  "
            f"d_frac<=0={row['delta_frac_margin_le_0']:+.4f}  "
            f"d_frac<=0.5={row['delta_frac_margin_le_0p5']:+.4f}  "
            f"q={row['question_preview']}  "
            f"a={row['answer_preview']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TOFU full vs retain model answer-token margins on forget/retain splits."
    )
    parser.add_argument("--full_model_dir", type=str, required=True)
    parser.add_argument("--retain_model_dir", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    device = get_device(args.device)
    print(f"Using device: {device}")

    tokenizer_dir = args.tokenizer_dir or args.full_model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer: {tokenizer_dir}")

    forget_pairs = load_tofu_pairs(args.forget_split, args.dataset_cache_dir, args.max_samples)
    retain_pairs = load_tofu_pairs(args.retain_split, args.dataset_cache_dir, args.max_samples)

    forget_meta, forget_ids, forget_labels, forget_masks = encode_qa_pairs(
        tokenizer, forget_pairs, args.max_length
    )
    retain_meta, retain_ids, retain_labels, retain_masks = encode_qa_pairs(
        tokenizer, retain_pairs, args.max_length
    )
    print(f"Prepared {len(forget_ids)} {args.forget_split} examples and {len(retain_ids)} {args.retain_split} examples")

    forget_full, retain_full = compute_split_stats_for_model(
        args.full_model_dir,
        forget_ids,
        forget_labels,
        forget_masks,
        retain_ids,
        retain_labels,
        retain_masks,
        args.batch_size,
        device,
        "full model",
    )
    forget_retain_model, retain_retain_model = compute_split_stats_for_model(
        args.retain_model_dir,
        forget_ids,
        forget_labels,
        forget_masks,
        retain_ids,
        retain_labels,
        retain_masks,
        args.batch_size,
        device,
        "retain model",
    )

    forget_rows = make_rows(args.forget_split, forget_meta, forget_full, forget_retain_model)
    retain_rows = make_rows(args.retain_split, retain_meta, retain_full, retain_retain_model)

    write_rows_csv(out_dir / "forget_margins.csv", forget_rows)
    write_rows_csv(out_dir / "retain_margins.csv", retain_rows)

    summary = {
        "benchmark": "TOFU",
        "full_model_dir": args.full_model_dir,
        "retain_model_dir": args.retain_model_dir,
        "tokenizer_dir": tokenizer_dir,
        "dataset_cache_dir": args.dataset_cache_dir,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "device": str(device),
        "example_definition": "TOFU question-answer pairs formatted with the open-unlearning Llama-2 chat template; margins are scored only on supervised answer tokens (labels != -100).",
        "diagnostic_note": "This is a teacher-forced local function-space diagnostic. It complements, but does not replace, TOFU benchmark metrics.",
        "margin_note": "Best-competitor margin is computed as gold logit minus the largest non-gold logit; this is equivalent to gold log-prob minus best non-gold log-prob at each position.",
        "forget_example_count": len(forget_rows),
        "retain_example_count": len(retain_rows),
        args.forget_split: aggregate_split(forget_rows),
        args.retain_split: aggregate_split(retain_rows),
        "probe_aligned": probe_aligned_summary(forget_rows, retain_rows),
    }

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    txt_lines = [
        "TOFU Full vs Retain Margin Diagnostic",
        f"Full model: {args.full_model_dir}",
        f"Retain model: {args.retain_model_dir}",
        f"Forget split: {args.forget_split}",
        f"Retain split: {args.retain_split}",
        f"Example definition: {summary['example_definition']}",
        f"Diagnostic note: {summary['diagnostic_note']}",
        f"Margin note: {summary['margin_note']}",
        "",
        f"{args.forget_split} summary:",
        json.dumps(summary[args.forget_split], indent=2),
        "",
        f"{args.retain_split} summary:",
        json.dumps(summary[args.retain_split], indent=2),
        "",
        "Probe-aligned summary:",
        json.dumps(summary["probe_aligned"], indent=2),
        "",
        format_example_rows(
            f"{args.forget_split} examples with largest margin decrease (lowest delta_mean_margin):",
            top_rows(forget_rows, "delta_mean_margin", args.top_k),
        ),
        "",
        format_example_rows(
            f"{args.retain_split} examples with largest margin increase (highest delta_mean_margin):",
            top_rows(retain_rows, "delta_mean_margin", args.top_k, reverse=True),
        ),
        "",
        format_example_rows(
            f"{args.retain_split} examples with largest margin decrease (lowest delta_mean_margin):",
            top_rows(retain_rows, "delta_mean_margin", args.top_k),
        ),
        "",
        format_example_rows(
            f"{args.forget_split} examples with largest margin increase (highest delta_mean_margin):",
            top_rows(forget_rows, "delta_mean_margin", args.top_k, reverse=True),
        ),
    ]
    (out_dir / "summary.txt").write_text("\n".join(txt_lines) + "\n")

    print(
        json.dumps(
            {
                "forget_mean_delta_mean_margin": summary[args.forget_split]["deltas"]["mean_margin"]["mean"],
                "retain_mean_delta_mean_margin": summary[args.retain_split]["deltas"]["mean_margin"]["mean"],
                "forget_mean_delta_min_margin": summary[args.forget_split]["deltas"]["min_margin"]["mean"],
                "retain_mean_delta_min_margin": summary[args.retain_split]["deltas"]["min_margin"]["mean"],
                "forget_examples_with_lower_margin": summary[args.forget_split]["counts"]["retain_model_lowers_mean_margin"],
                "retain_examples_with_higher_margin": summary[args.retain_split]["counts"]["retain_model_raises_mean_margin"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
