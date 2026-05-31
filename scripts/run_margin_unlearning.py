#!/usr/bin/env python3
"""
One-step margin-based unlearning on MUSE raw text windows.

Motivation:
Robust retrain diagnostics suggest useful models often raise confidence on
retain text much more than on forget text. This objective tries to reproduce
that asymmetry directly in one step, without using retrain during training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]


def dataset_spec(corpus: str) -> tuple[str, str]:
    corpus = corpus.lower()
    if corpus == "news":
        return "muse-bench/MUSE-News", "MUSE-News"
    if corpus == "tofu":
        return "locuslab/TOFU", "TOFU"
    raise ValueError(f"Unsupported corpus: {corpus}")


def root_data_dir(corpus: str) -> Path:
    return REPO_ROOT / "muse_bench" / "data" / corpus.lower() / "raw"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_fp_dtype(device: torch.device, precision: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def resolve_default_tokenizer_dir(cache_dir: str, model_dir: str | None) -> str:
    if model_dir is not None and (Path(model_dir) / "tokenizer.model").exists():
        return model_dir
    cache_root = Path(cache_dir)
    llama2_tokenizer = (
        cache_root
        / "hub"
        / "models--meta-llama--Llama-2-7b-hf"
        / "snapshots"
        / "01c7f73d771dfac7d292323805ebc428287df4f9"
    )
    return str(llama2_tokenizer)


def load_texts_from_local_raw(corpus: str, split: str, max_samples: int | None) -> list[str]:
    raw_dir = root_data_dir(corpus)
    json_path = raw_dir / f"{split}.json"
    txt_path = raw_dir / f"{split}.txt"

    if json_path.exists():
        data = json.loads(json_path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {json_path}, got {type(data).__name__}")
        texts: list[str] = []
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
            else:
                raise ValueError(f"Unsupported item format in {json_path}: {type(item).__name__}")
            if max_samples is not None and len(texts) >= max_samples:
                break
        print(f"Local raw fallback {corpus} {split}: loaded {len(texts)} samples from {json_path}")
        return texts

    if txt_path.exists():
        text = txt_path.read_text()
        lines = [line for line in text.splitlines() if line.strip()]
        if max_samples is not None:
            lines = lines[:max_samples]
        print(f"Local raw fallback {corpus} {split}: loaded {len(lines)} samples from {txt_path}")
        return lines

    raise FileNotFoundError(f"Local raw fallback could not find {json_path} or {txt_path}")


def load_split_texts(cache_dir: str, corpus: str, split: str, max_samples: int | None) -> list[str]:
    dataset_name, display_name = dataset_spec(corpus)
    try:
        ds = load_dataset(dataset_name, "raw", cache_dir=cache_dir, split=split)
        texts: list[str] = []
        for item in ds:
            texts.append(item["text"])
            if max_samples is not None and len(texts) >= max_samples:
                break
        print(f"{display_name} {split}: loaded {len(texts)} samples")
        return texts
    except Exception as exc:
        print(f"{display_name} {split}: datasets load failed ({exc}); trying local raw fallback")
        return load_texts_from_local_raw(corpus, split, max_samples)


def load_jsonl_texts(
    path: str | Path,
    max_samples: int | None,
    *,
    text_key: str = "text",
    min_chars: int = 0,
) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find JSONL file: {path}")

    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if text_key not in payload:
                raise KeyError(f"Expected key `{text_key}` in {path} line {line_number}")
            text = str(payload[text_key]).strip()
            if len(text) < min_chars:
                continue
            texts.append(text)
            if max_samples is not None and len(texts) >= max_samples:
                break
    return texts


def load_wmdp_bio_texts(
    *,
    cache_dir: str | None,
    split: str,
    max_samples: int | None,
    retain_source: str,
    forget_path: str,
    retain_path: str,
    min_chars: int,
) -> list[str]:
    split = split.lower()
    if split == "forget":
        texts = load_jsonl_texts(forget_path, max_samples, min_chars=min_chars)
        print(f"WMDP-Bio forget: loaded {len(texts)} raw texts from {forget_path}")
        return texts

    if split != "retain":
        raise ValueError(f"Unsupported WMDP-Bio split: {split}")

    if retain_source == "wikitext_test":
        try:
            ds = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split="test",
                cache_dir=cache_dir,
            )
            texts: list[str] = []
            for item in ds:
                text = str(item["text"]).strip()
                if len(text) < min_chars:
                    continue
                texts.append(text)
                if max_samples is not None and len(texts) >= max_samples:
                    break
            print(f"WMDP-Bio retain: loaded {len(texts)} Wikitext-2 test texts")
            return texts
        except Exception as exc:
            raise RuntimeError(
                "Failed to load Wikitext-2 test split for WMDP-Bio retain data. "
                "If this environment does not have Wikitext cached, either pre-download it "
                "or rerun with `--wmdp_bio_retain_source bio_retain_corpus`."
            ) from exc

    if retain_source == "bio_retain_corpus":
        texts = load_jsonl_texts(retain_path, max_samples, min_chars=min_chars)
        print(f"WMDP-Bio retain: loaded {len(texts)} raw texts from {retain_path}")
        return texts

    raise ValueError(f"Unsupported WMDP-Bio retain source: {retain_source}")


IGNORE_INDEX = -100

LLAMA2_CHAT_TEMPLATE = {
    "apply_chat_template": False,
    "user_start_tag": "[INST] ",
    "user_end_tag": " [/INST]",
    "asst_start_tag": "",
    "asst_end_tag": " ",
}


def load_tofu_pairs(
    split: str,
    cache_dir: str | None,
    max_samples: int | None,
    *,
    verbose: bool = True,
) -> list[dict]:
    ds = load_dataset("locuslab/TOFU", name=split, split="train", cache_dir=cache_dir)
    rows: list[dict] = []
    for idx, item in enumerate(ds):
        rows.append(
            {
                "index": idx,
                "question": item["question"],
                "answer": item["answer"],
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    if verbose:
        print(f"TOFU {split}: loaded {len(rows)} samples")
    return rows


def preprocess_tofu_chat_instance(
    tokenizer,
    question: str,
    answer: str,
    max_length: int,
) -> dict[str, torch.Tensor]:
    wrapped_prompt = (
        LLAMA2_CHAT_TEMPLATE["user_start_tag"]
        + question
        + LLAMA2_CHAT_TEMPLATE["user_end_tag"]
        + LLAMA2_CHAT_TEMPLATE["asst_start_tag"]
    )
    chat_ids = tokenizer(
        wrapped_prompt + answer,
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
    if chat_ids and chat_ids[-1] != tokenizer.eos_token_id:
        chat_ids += [tokenizer.eos_token_id]
    labels = [IGNORE_INDEX] * len(prompt_ids) + chat_ids[len(prompt_ids):]
    return {
        "input_ids": torch.tensor(chat_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor([1] * len(chat_ids), dtype=torch.long),
    }


def pad_or_trim_value(tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
    tensor = tensor.to(torch.long)
    if tensor.size(0) < target_length:
        return torch.cat(
            [
                tensor,
                torch.full((target_length - tensor.size(0),), pad_value, dtype=torch.long),
            ]
        )
    return tensor[:target_length]


def encode_tofu_pairs(
    tokenizer,
    pairs: list[dict],
    max_len: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[dict]]:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    attention_masks: list[torch.Tensor] = []
    metadata: list[dict] = []
    for window_index, item in enumerate(pairs):
        processed = preprocess_tofu_chat_instance(
            tokenizer=tokenizer,
            question=item["question"],
            answer=item["answer"],
            max_length=max_len,
        )
        ids = pad_or_trim_value(processed["input_ids"], max_len, pad_id)
        lbls = pad_or_trim_value(processed["labels"], max_len, IGNORE_INDEX)
        attn = pad_or_trim_value(processed["attention_mask"], max_len, 0)
        input_ids.append(ids)
        labels.append(lbls)
        attention_masks.append(attn)
        metadata.append(
            {
                "source_index": int(item["index"]),
                "window_index": window_index,
                "window_start": 0,
                "window_end": int(attn.sum().item()),
            }
        )
    return input_ids, labels, attention_masks, metadata


def pad_or_trim(ids: torch.Tensor, target_length: int, pad_id: int) -> torch.Tensor:
    ids = ids.to(torch.long)
    if ids.size(0) < target_length:
        return torch.cat([ids, torch.full((target_length - ids.size(0),), pad_id, dtype=torch.long)])
    return ids[:target_length]


def chunk_texts_nonoverlap(
    tokenizer,
    texts: Iterable[str],
    max_len: int,
) -> tuple[list[torch.Tensor], list[dict]]:
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id
    content_window = max_len - 1 if bos_id is not None else max_len
    if content_window <= 0:
        raise ValueError(f"max_len={max_len} is too small")

    windows: list[torch.Tensor] = []
    metadata: list[dict] = []
    for source_index, text in enumerate(texts):
        raw_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(torch.long)
        starts = [0] if raw_ids.numel() == 0 else list(range(0, raw_ids.size(0), content_window))
        for window_index, start in enumerate(starts):
            end = min(start + content_window, raw_ids.size(0))
            chunk = raw_ids[start:end]
            if bos_id is not None:
                chunk = torch.cat([torch.tensor([bos_id], dtype=torch.long), chunk])
            windows.append(pad_or_trim(chunk, max_len, pad_id))
            metadata.append(
                {
                    "source_index": source_index,
                    "window_index": window_index,
                    "window_start": int(start),
                    "window_end": int(end),
                }
            )
    return windows, metadata


class WindowDataset(Dataset):
    def __init__(self, windows: list[torch.Tensor], metadata: list[dict], pad_id: int):
        self.windows = windows
        self.metadata = metadata
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        input_ids = self.windows[idx]
        attention_mask = (input_ids != self.pad_id).long()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        item.update(self.metadata[idx])
        return item


class LabelledWindowDataset(Dataset):
    def __init__(
        self,
        input_ids: list[torch.Tensor],
        labels: list[torch.Tensor],
        attention_masks: list[torch.Tensor],
        metadata: list[dict],
    ):
        self.input_ids = input_ids
        self.labels = labels
        self.attention_masks = attention_masks
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }
        item.update(self.metadata[idx])
        return item


def collate_batch(items: list[dict]) -> dict:
    batch = {
        "input_ids": torch.stack([item["input_ids"] for item in items]),
        "attention_mask": torch.stack([item["attention_mask"] for item in items]),
        "labels": torch.stack([item["labels"] for item in items]),
        "source_index": [item["source_index"] for item in items],
        "window_index": [item["window_index"] for item in items],
        "window_start": [item["window_start"] for item in items],
        "window_end": [item["window_end"] for item in items],
    }
    return batch


def to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def shift_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def retain_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def retain_kl_to_teacher(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Token-averaged KL(p_teacher || p_student) on labelled retain positions."""
    shift_student, shift_labels = shift_logits_and_labels(student_logits, labels)
    shift_teacher, _ = shift_logits_and_labels(teacher_logits, labels)
    valid = shift_labels != -100
    if valid.sum().item() == 0:
        return shift_student.new_tensor(0.0)
    student_log_probs = F.log_softmax(shift_student[valid].float(), dim=-1)
    teacher_log_probs = F.log_softmax(shift_teacher[valid].float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean", log_target=False)


def cross_entropy_stats(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict]:
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size(0), shift_labels.size(1))
    valid = shift_labels != -100
    valid_losses = token_loss[valid]
    if valid_losses.numel() == 0:
        zero = logits.new_tensor(0.0)
        return zero, {"avg_nll": 0.0, "valid_tokens": 0}
    return valid_losses.mean(), {
        "avg_nll": float(valid_losses.mean().detach().cpu().item()),
        "valid_tokens": int(valid.sum().detach().cpu().item()),
    }


def simnpo_forget_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    beta: float,
    delta: float,
) -> torch.Tensor:
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    token_loss = F.cross_entropy(
        shift_logits.transpose(-1, -2),
        shift_labels,
        reduction="none",
        ignore_index=-100,
    )
    valid = shift_labels != -100
    seq_loss = token_loss.sum(dim=-1)
    seq_len = valid.sum(dim=-1).clamp_min(1)
    normalized_loss = seq_loss / seq_len - delta
    return -F.logsigmoid(beta * normalized_loss).mean() * (2.0 / beta)


def mean_margin_tensor(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    teacher_topk: int = 10,
    use_teacher_competitor: bool = True,
    teacher_set_mode: str = "topk",
    teacher_logit_delta: float = 0.5,
    competitor_beta: float = 1.0,
    beta_parameterization: str = "tempered_posterior",
) -> torch.Tensor:
    margins, valid = compute_token_margins(
        logits,
        labels,
        teacher_logits=teacher_logits,
        teacher_topk=teacher_topk,
        use_teacher_competitor=use_teacher_competitor,
        teacher_set_mode=teacher_set_mode,
        teacher_logit_delta=teacher_logit_delta,
        competitor_beta=competitor_beta,
        beta_parameterization=beta_parameterization,
    )
    valid_margins = margins[valid]
    if valid_margins.numel() == 0:
        return logits.new_tensor(0.0)
    return valid_margins.mean()


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total_sq += float(torch.sum(grad * grad).item())
    return math.sqrt(total_sq)


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def snapshot_current_grads(
    params: list[torch.nn.Parameter],
    storage_dtype: torch.dtype | None = None,
    storage_device: str | torch.device | None = None,
) -> list[torch.Tensor | None]:
    snapshots: list[torch.Tensor | None] = []
    for param in params:
        if param.grad is None:
            snapshots.append(None)
        else:
            target_dtype = storage_dtype or param.grad.dtype
            target_device = storage_device or param.grad.device
            snapshots.append(param.grad.detach().to(device=target_device, dtype=target_dtype).clone())
    return snapshots


def grad_dot(
    grad_a: list[torch.Tensor | None],
    grad_b: list[torch.Tensor | None],
) -> float:
    total = 0.0
    for ga, gb in zip(grad_a, grad_b):
        if ga is None or gb is None:
            continue
        total += float(torch.sum(ga.float() * gb.float()).item())
    return total


def grad_norm_from_list(grads: list[torch.Tensor | None]) -> float:
    total_sq = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_f = grad.float()
        total_sq += float(torch.sum(grad_f * grad_f).item())
    return math.sqrt(total_sq)


def grad_cosine_from_lists(
    grad_a: list[torch.Tensor | None],
    grad_b: list[torch.Tensor | None],
    eps: float,
) -> float:
    norm_a = grad_norm_from_list(grad_a)
    norm_b = grad_norm_from_list(grad_b)
    if norm_a <= eps or norm_b <= eps:
        return 0.0
    return grad_dot(grad_a, grad_b) / (norm_a * norm_b + eps)


def projected_grad_norm(
    secondary_grads: list[torch.Tensor | None],
    retain_grads: list[torch.Tensor | None],
    proj_coeff: float,
) -> float:
    total_sq = 0.0
    for gs, gr in zip(secondary_grads, retain_grads):
        if gs is None and gr is None:
            continue
        gs_f = gs.float() if gs is not None else None
        gr_f = gr.float() if gr is not None else None
        if gs_f is None:
            proj = -proj_coeff * gr_f
        elif gr_f is None:
            proj = gs_f
        else:
            proj = gs_f - proj_coeff * gr_f
        total_sq += float(torch.sum(proj * proj).item())
    return math.sqrt(total_sq)


def projected_grad_cosine(
    retain_grads: list[torch.Tensor | None],
    secondary_grads: list[torch.Tensor | None],
    proj_coeff: float,
    eps: float,
) -> float:
    retain_norm = grad_norm_from_list(retain_grads)
    proj_norm = projected_grad_norm(secondary_grads, retain_grads, proj_coeff)
    if retain_norm <= eps or proj_norm <= eps:
        return 0.0

    dot = 0.0
    for gr, gs in zip(retain_grads, secondary_grads):
        if gr is None:
            continue
        gr_f = gr.float()
        if gs is None:
            proj = -proj_coeff * gr_f
        else:
            proj = gs.float() - proj_coeff * gr_f
        dot += float(torch.sum(gr_f * proj).item())
    return dot / (retain_norm * proj_norm + eps)


def assign_safe_projected_grad(
    params: list[torch.nn.Parameter],
    retain_grads: list[torch.Tensor | None],
    secondary_grads: list[torch.Tensor | None],
    proj_coeff: float,
    clip_scale: float,
    grad_accum_steps: int,
) -> None:
    scale = 1.0 / max(1, grad_accum_steps)
    for param, gr, gs in zip(params, retain_grads, secondary_grads):
        if gr is None and gs is None:
            param.grad = None
            continue
        gr_f = gr.float() if gr is not None else torch.zeros_like(gs.float())
        if gs is None:
            safe = torch.zeros_like(gr_f)
        else:
            gs_f = gs.float()
            safe = (gs_f - proj_coeff * gr_f) * clip_scale
        final = (gr_f + safe).to(device=param.device, dtype=param.dtype) * scale
        if param.grad is None:
            param.grad = final.clone()
        else:
            param.grad.copy_(final)


def compute_token_margins(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    teacher_topk: int = 10,
    use_teacher_competitor: bool = True,
    teacher_set_mode: str = "topk",
    teacher_logit_delta: float = 0.5,
    competitor_beta: float = 1.0,
    beta_parameterization: str = "tempered_posterior",
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    valid = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[safe_labels == -100] = 0
    gold_logits = shift_logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1).float()

    if teacher_logits is not None and use_teacher_competitor:
        # Use the frozen base model to choose plausible non-gold alternatives,
        # then score their joint competitor mass with the current model.
        if teacher_topk < 1:
            raise ValueError("`teacher_topk` must be >= 1.")
        if teacher_set_mode not in {"topk", "delta"}:
            raise ValueError(f"Unsupported teacher set mode: {teacher_set_mode}")
        if competitor_beta <= 0:
            raise ValueError("`competitor_beta` must be > 0.")
        if beta_parameterization not in {"tempered_posterior", "smoothmax_only"}:
            raise ValueError(f"Unsupported beta parameterization: {beta_parameterization}")
        shift_teacher, _ = shift_logits_and_labels(teacher_logits, labels)
        teacher_candidate_scores = shift_teacher.float().clone()
        teacher_candidate_scores.scatter_(
            -1,
            safe_labels.unsqueeze(-1),
            torch.finfo(teacher_candidate_scores.dtype).min,
        )
        top_k = min(teacher_topk, max(1, teacher_candidate_scores.size(-1) - 1))
        top_teacher_scores, teacher_topk_indices = teacher_candidate_scores.topk(k=top_k, dim=-1)
        selected_student_logits = shift_logits.float().gather(-1, teacher_topk_indices)
        if teacher_set_mode == "delta":
            # Keep the teacher-defined set local by retaining only alternatives
            # within delta logits of the teacher's best non-gold token. This
            # preserves the restricted-posterior interpretation on the filtered
            # set while avoiding broad aggregation over weak alternatives.
            threshold = top_teacher_scores[..., :1] - teacher_logit_delta
            selected_student_logits = selected_student_logits.masked_fill(
                top_teacher_scores < threshold,
                torch.finfo(selected_student_logits.dtype).min,
            )
        if competitor_beta == 1.0:
            competitor_logits = torch.logsumexp(selected_student_logits, dim=-1)
            margins = gold_logits - competitor_logits
        else:
            # Smooth-max competitor: beta=1 recovers logsumexp, beta->inf approaches
            # the hardest-rival max while keeping a differentiable aggregation.
            tempered_competitor_logits = torch.logsumexp(
                selected_student_logits * competitor_beta,
                dim=-1,
            )
            if beta_parameterization == "tempered_posterior":
                # Default: use the beta-tempered restricted posterior margin
                # beta * gold - logsumexp(beta * alternatives). This keeps the
                # rho-calibration threshold unchanged while beta changes the
                # competitor geometry through a sharper softmax over alternatives.
                margins = gold_logits * competitor_beta - tempered_competitor_logits
            else:
                competitor_logits = tempered_competitor_logits / competitor_beta
                margins = gold_logits - competitor_logits
    elif not use_teacher_competitor:
        # Self-competitor mode: choose the non-gold competitor set from the
        # current model itself. For k>1 this mirrors the teacher-topk
        # construction but uses the student's own top non-gold alternatives.
        self_candidate_scores = shift_logits.float().clone()
        self_candidate_scores.scatter_(
            -1,
            safe_labels.unsqueeze(-1),
            torch.finfo(self_candidate_scores.dtype).min,
        )
        top_k = min(teacher_topk, max(1, self_candidate_scores.size(-1) - 1))
        selected_student_logits, _ = self_candidate_scores.topk(k=top_k, dim=-1)
        if competitor_beta == 1.0:
            competitor_logits = torch.logsumexp(selected_student_logits, dim=-1)
            margins = gold_logits - competitor_logits
        else:
            tempered_competitor_logits = torch.logsumexp(
                selected_student_logits * competitor_beta,
                dim=-1,
            )
            if beta_parameterization == "tempered_posterior":
                margins = gold_logits * competitor_beta - tempered_competitor_logits
            else:
                competitor_logits = tempered_competitor_logits / competitor_beta
                margins = gold_logits - competitor_logits
    else:
        # Fallback when teacher-based competition was requested but no teacher
        # logits are available. Keep the historical self top-1 behavior.
        top2_vals, top2_idx = shift_logits.topk(k=2, dim=-1)
        top1_is_gold = top2_idx[..., 0].eq(safe_labels)
        competitor_logits = torch.where(top1_is_gold, top2_vals[..., 1], top2_vals[..., 0]).float()
        if competitor_beta != 1.0 and beta_parameterization == "tempered_posterior":
            margins = competitor_beta * (gold_logits - competitor_logits)
        else:
            margins = gold_logits - competitor_logits
    return margins, valid


def effective_margin_tau(margin_tau: float, forget_margin_rho: float | None) -> float:
    if forget_margin_rho is None:
        return margin_tau
    return math.log(forget_margin_rho / (1.0 - forget_margin_rho))


def buffered_margin_surrogate(margins: torch.Tensor, tau: float, eta: float) -> torch.Tensor:
    return torch.relu(margins - (tau - eta)) / eta


def hoeffding_ucb(mean_value: float, n: int, delta: float, eval_index: int) -> tuple[float, float]:
    if n <= 0:
        return float("inf"), float("nan")
    delta_j = 6.0 * delta / (math.pi ** 2 * (eval_index + 1) ** 2)
    delta_j = min(max(delta_j, 1e-12), 1.0 - 1e-12)
    radius = math.sqrt(math.log(1.0 / delta_j) / (2.0 * n))
    return mean_value + radius, delta_j


def should_stop_with_ucb(ucb: float, alpha: float, retain_ok: bool) -> bool:
    return bool(ucb <= alpha and retain_ok)


def two_stage_trigger_value_from_metrics(probe_metrics: dict, trigger_metric: str) -> float:
    if trigger_metric == "surrogate_mean":
        return float(probe_metrics["forget_continuation_surrogate_mean"])
    if trigger_metric == "violation_mean":
        return float(probe_metrics["forget_continuation_per_example_violation_mean"])
    raise ValueError(f"Unsupported two-stage trigger metric: {trigger_metric}")


def two_stage_confirmation_decision(
    *,
    violation_mean: float,
    n: int,
    alpha: float,
    retain_ok: bool,
    use_ucb: bool,
    delta: float,
    eval_index: int,
) -> tuple[bool, float | None, float | None]:
    if use_ucb:
        ucb, delta_j = hoeffding_ucb(violation_mean, n, delta, eval_index)
        return bool(ucb <= alpha and retain_ok), ucb, delta_j
    return bool(violation_mean <= alpha and retain_ok), None, None


def probe_batches_for_num_examples(num_examples: int, batch_size: int) -> int:
    return max(1, math.ceil(num_examples / max(1, batch_size)))


def build_continuation_suffix_mask(
    valid: torch.Tensor,
    cut_min_ratio: float,
    cut_max_ratio: float,
    min_prefix_tokens: int,
    min_suffix_tokens: int,
    source_indices: list[int] | None = None,
    window_indices: list[int] | None = None,
    deterministic: bool = False,
    deterministic_salt: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Build a suffix-only mask on shifted next-token target positions.

    `valid` already corresponds to shifted labels, so if the cut is at target
    position `c`, the active continuation starts at `valid_positions[c:]`.
    """
    suffix_mask = torch.zeros_like(valid, dtype=torch.bool)
    cut_ratios: list[float] = []
    suffix_lengths: list[int] = []
    fallback_count = 0

    for i in range(valid.size(0)):
        valid_positions = torch.nonzero(valid[i], as_tuple=False).squeeze(-1)
        valid_len = int(valid_positions.numel())
        if valid_len <= 0:
            continue

        # If the sequence is too short to support a meaningful split,
        # fall back to the full valid region rather than dropping it.
        if valid_len < (min_prefix_tokens + min_suffix_tokens):
            suffix_mask[i, valid_positions] = True
            suffix_lengths.append(valid_len)
            fallback_count += 1
            continue

        if deterministic:
            if source_indices is None or window_indices is None:
                raise ValueError("Deterministic continuation masking requires source/window indices.")
            mix = (
                (int(source_indices[i]) + 1) * 1000003
                ^ (int(window_indices[i]) + 1) * 9176
                ^ (int(deterministic_salt) + 1) * 2654435761
            ) & 0xFFFFFFFF
            unit = mix / 0xFFFFFFFF
            ratio = cut_min_ratio + unit * (cut_max_ratio - cut_min_ratio)
        else:
            ratio = random.uniform(cut_min_ratio, cut_max_ratio)
        cut = int(math.floor(ratio * valid_len))
        cut = max(min_prefix_tokens, min(cut, valid_len - min_suffix_tokens))

        suffix_positions = valid_positions[cut:]
        suffix_mask[i, suffix_positions] = True
        cut_ratios.append(cut / valid_len)
        suffix_lengths.append(int(suffix_positions.numel()))

    return suffix_mask, {
        "cut_ratio_mean": float(sum(cut_ratios) / len(cut_ratios)) if cut_ratios else 0.0,
        "suffix_tokens_mean": float(sum(suffix_lengths) / len(suffix_lengths)) if suffix_lengths else 0.0,
        "fallback_count": fallback_count,
    }


def forget_margin_penalty(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    teacher_topk: int,
    use_teacher_competitor: bool,
    teacher_set_mode: str,
    teacher_logit_delta: float,
    competitor_beta: float,
    beta_parameterization: str,
    tau: float,
    forget_margin_rho: float | None,
    buffer_eta: float,
    penalty_mode: str,
    loss_mode: str,
    sigmoid_beta: float,
    continuation_cut_min_ratio: float,
    continuation_cut_max_ratio: float,
    continuation_min_prefix_tokens: int,
    continuation_min_suffix_tokens: int,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
    margins, valid = compute_token_margins(
        logits,
        labels,
        teacher_logits=teacher_logits,
        teacher_topk=teacher_topk,
        use_teacher_competitor=use_teacher_competitor,
        teacher_set_mode=teacher_set_mode,
        teacher_logit_delta=teacher_logit_delta,
        competitor_beta=competitor_beta,
        beta_parameterization=beta_parameterization,
    )
    valid_margins = margins[valid]
    if penalty_mode == "continuation":
        suffix_mask, continuation_meta = build_continuation_suffix_mask(
            valid,
            cut_min_ratio=continuation_cut_min_ratio,
            cut_max_ratio=continuation_cut_max_ratio,
            min_prefix_tokens=continuation_min_prefix_tokens,
            min_suffix_tokens=continuation_min_suffix_tokens,
        )
        active_mask = valid & suffix_mask
    else:
        continuation_meta = {
            "cut_ratio_mean": 0.0,
            "suffix_tokens_mean": float(valid.sum().detach().cpu().item() / max(valid.size(0), 1)),
            "fallback_count": 0,
        }
        active_mask = valid

    active_margins = margins[active_mask]
    if active_margins.numel() == 0:
        zero = logits.new_tensor(0.0)
        return zero, {
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "frac_margin_le_0": 0.0,
            "frac_margin_le_0p5": 0.0,
            "frac_margin_gt_tau": 0.0,
            "continuation_mean_margin": 0.0,
            "continuation_min_margin": 0.0,
            "continuation_frac_margin_gt_tau": 0.0,
            "continuation_valid_tokens": 0,
            "hinge_margin_loss": 0.0,
            "buffered_hinge_margin_loss": 0.0,
            "centered_smooth_l1_loss": 0.0,
            "forget_margin_sigmoid_weight_mean": 0.0,
            "forget_margin_tau_effective": tau,
            "forget_margin_rho": forget_margin_rho,
            "forget_margin_buffer_eta": buffer_eta,
            "forget_margin_teacher_topk": teacher_topk if (teacher_logits is not None and use_teacher_competitor) else 0,
            "forget_margin_teacher_set_mode": teacher_set_mode if (teacher_logits is not None and use_teacher_competitor) else "self",
            "forget_margin_teacher_logit_delta": teacher_logit_delta if (teacher_logits is not None and use_teacher_competitor) else None,
            "forget_margin_competitor_beta": competitor_beta if (teacher_logits is not None and use_teacher_competitor) else 1.0,
            "cut_ratio_mean": continuation_meta["cut_ratio_mean"],
            "suffix_tokens_mean": continuation_meta["suffix_tokens_mean"],
            "fallback_count": continuation_meta["fallback_count"],
            "valid_tokens": 0,
        }, zero, zero

    hinge = torch.relu(active_margins - tau)
    # In rho mode, the teacher-anchored margin defines a restricted posterior:
    # sigmoid(margin) = P_theta(gold | gold union teacher top-k alternatives).
    # Setting tau = log(rho / (1-rho)) makes margin <= tau equivalent to
    # restricted gold posterior <= rho.
    # The buffered hinge upper-bounds the violation indicator 1[margin > tau].
    if forget_margin_rho is not None:
        loss = buffered_margin_surrogate(active_margins, tau=tau, eta=buffer_eta).mean()
        sigmoid_weight_mean = 1.0
        buffered_hinge_margin_loss = float(loss.detach().cpu().item())
        centered_smooth_l1_loss = 0.0
    elif loss_mode == "hinge":
        loss = hinge.mean()
        sigmoid_weight_mean = 1.0
        buffered_hinge_margin_loss = 0.0
        centered_smooth_l1_loss = 0.0
    elif loss_mode == "sigmoid_hinge":
        weights = torch.sigmoid((active_margins - tau) / sigmoid_beta)
        loss = (weights * hinge).mean()
        sigmoid_weight_mean = float(weights.detach().mean().cpu().item())
        buffered_hinge_margin_loss = 0.0
        centered_smooth_l1_loss = 0.0
    elif loss_mode == "smooth_l1_centered":
        smooth_l1_beta = max(tau, 1e-8)
        loss = F.smooth_l1_loss(
            active_margins,
            torch.zeros_like(active_margins),
            reduction="mean",
            beta=smooth_l1_beta,
        )
        sigmoid_weight_mean = 1.0
        buffered_hinge_margin_loss = 0.0
        centered_smooth_l1_loss = float(loss.detach().cpu().item())
    else:
        raise ValueError(f"Unsupported forget margin loss mode: {loss_mode}")

    if not collect_stats:
        return loss, {}, valid_margins.mean(), active_margins.mean()

    stats = {
        "mean_margin": float(valid_margins.mean().detach().cpu().item()),
        "min_margin": float(valid_margins.min().detach().cpu().item()),
        "frac_margin_le_0": float((valid_margins <= 0).float().mean().detach().cpu().item()),
        "frac_margin_le_0p5": float((valid_margins <= 0.5).float().mean().detach().cpu().item()),
        "frac_margin_gt_tau": float((valid_margins > tau).float().mean().detach().cpu().item()),
        "continuation_mean_margin": float(active_margins.mean().detach().cpu().item()),
        "continuation_min_margin": float(active_margins.min().detach().cpu().item()),
        "continuation_frac_margin_gt_tau": float((active_margins > tau).float().mean().detach().cpu().item()),
        "continuation_valid_tokens": int(active_mask.sum().detach().cpu().item()),
        "hinge_margin_loss": float(hinge.detach().mean().cpu().item()),
        "buffered_hinge_margin_loss": buffered_hinge_margin_loss,
        "centered_smooth_l1_loss": centered_smooth_l1_loss,
        "forget_margin_sigmoid_weight_mean": sigmoid_weight_mean,
        "forget_margin_tau_effective": tau,
        "forget_margin_rho": forget_margin_rho,
        "forget_margin_buffer_eta": buffer_eta,
        "forget_margin_teacher_topk": teacher_topk if (teacher_logits is not None and use_teacher_competitor) else 0,
        "forget_margin_teacher_set_mode": teacher_set_mode if (teacher_logits is not None and use_teacher_competitor) else "self",
        "forget_margin_teacher_logit_delta": teacher_logit_delta if (teacher_logits is not None and use_teacher_competitor) else None,
        "forget_margin_competitor_beta": competitor_beta if (teacher_logits is not None and use_teacher_competitor) else 1.0,
        "cut_ratio_mean": continuation_meta["cut_ratio_mean"],
        "suffix_tokens_mean": continuation_meta["suffix_tokens_mean"],
        "fallback_count": continuation_meta["fallback_count"],
        "valid_tokens": int(valid.sum().detach().cpu().item()),
    }
    return loss, stats, valid_margins.mean(), active_margins.mean()


def batch_margin_stats(
    logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor | None = None,
    teacher_topk: int = 10,
    use_teacher_competitor: bool = True,
    teacher_set_mode: str = "topk",
    teacher_logit_delta: float = 0.5,
    competitor_beta: float = 1.0,
    beta_parameterization: str = "tempered_posterior",
) -> dict:
    margins, valid = compute_token_margins(
        logits,
        labels,
        teacher_logits=teacher_logits,
        teacher_topk=teacher_topk,
        use_teacher_competitor=use_teacher_competitor,
        teacher_set_mode=teacher_set_mode,
        teacher_logit_delta=teacher_logit_delta,
        competitor_beta=competitor_beta,
        beta_parameterization=beta_parameterization,
    )
    valid_margins = margins[valid]
    if valid_margins.numel() == 0:
        return {
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "frac_margin_le_0": 0.0,
            "frac_margin_le_0p5": 0.0,
            "valid_tokens": 0,
        }
    return {
        "mean_margin": float(valid_margins.mean().detach().cpu().item()),
        "min_margin": float(valid_margins.min().detach().cpu().item()),
        "frac_margin_le_0": float((valid_margins <= 0).float().mean().detach().cpu().item()),
        "frac_margin_le_0p5": float((valid_margins <= 0.5).float().mean().detach().cpu().item()),
        "valid_tokens": int(valid.sum().detach().cpu().item()),
    }


@dataclass
class EvalSummary:
    avg_nll: float
    mean_margin: float
    mean_min_margin: float
    global_min_margin: float
    frac_margin_le_0: float
    frac_margin_le_0p5: float
    windows: int
    valid_tokens: int


def evaluate_window_loader(
    model,
    loader: DataLoader,
    device: torch.device,
    teacher_model=None,
    teacher_topk: int = 10,
    use_teacher_competitor: bool = True,
    teacher_set_mode: str = "topk",
    teacher_logit_delta: float = 0.5,
    competitor_beta: float = 1.0,
    beta_parameterization: str = "tempered_posterior",
) -> EvalSummary:
    avg_nlls: list[float] = []
    mean_margins: list[float] = []
    min_margins: list[float] = []
    frac0s: list[float] = []
    frac05s: list[float] = []
    global_min_margin = float("inf")
    total_valid_tokens = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_logits = None
            if teacher_model is not None:
                teacher_outputs = teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                teacher_logits = teacher_outputs.logits
            shift_logits, shift_labels = shift_logits_and_labels(outputs.logits, batch["labels"])
            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view(shift_labels.size(0), shift_labels.size(1))
            valid = shift_labels != -100

            margins, valid_margin_mask = compute_token_margins(
                outputs.logits,
                batch["labels"],
                teacher_logits=teacher_logits,
                teacher_topk=teacher_topk,
                use_teacher_competitor=use_teacher_competitor,
                teacher_set_mode=teacher_set_mode,
                teacher_logit_delta=teacher_logit_delta,
                competitor_beta=competitor_beta,
                beta_parameterization=beta_parameterization,
            )
            for i in range(shift_labels.size(0)):
                valid_i = valid[i]
                if valid_i.sum().item() == 0:
                    continue
                seq_loss = token_loss[i][valid_i].mean().item()
                seq_margins = margins[i][valid_margin_mask[i]]
                avg_nlls.append(float(seq_loss))
                mean_margins.append(float(seq_margins.mean().item()))
                min_margins.append(float(seq_margins.min().item()))
                frac0s.append(float((seq_margins <= 0).float().mean().item()))
                frac05s.append(float((seq_margins <= 0.5).float().mean().item()))
                global_min_margin = min(global_min_margin, float(seq_margins.min().item()))
                total_valid_tokens += int(valid_i.sum().item())

    if not avg_nlls:
        return EvalSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)

    return EvalSummary(
        avg_nll=float(np.mean(avg_nlls)),
        mean_margin=float(np.mean(mean_margins)),
        mean_min_margin=float(np.mean(min_margins)),
        global_min_margin=float(global_min_margin),
        frac_margin_le_0=float(np.mean(frac0s)),
        frac_margin_le_0p5=float(np.mean(frac05s)),
        windows=len(avg_nlls),
        valid_tokens=total_valid_tokens,
    )


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


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


def load_model(model_dir: str, device: torch.device, dtype: torch.dtype, gradient_checkpointing: bool) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.to(device)
    return model


def apply_lora(
    model: AutoModelForCausalLM,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: list[str],
    lora_bias: str,
) -> AutoModelForCausalLM:
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise RuntimeError(
            "LoRA requested but `peft` is not installed. "
            "Install it in the training environment (e.g. `pip install peft`)."
        ) from exc

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias=lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def save_model_checkpoint(
    model,
    tokenizer,
    save_dir: Path,
    merge_lora: bool,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_format = "standard"
    if merge_lora and hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
        saved_format = "merged_full_model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return model, saved_format


def build_lr_scheduler(
    optimizer: AdamW,
    *,
    schedule: str,
    total_optimizer_steps: int,
) -> LambdaLR | None:
    if schedule == "constant":
        return None
    if total_optimizer_steps <= 0:
        return None
    if schedule == "cosine":
        def lr_lambda(step: int) -> float:
            if total_optimizer_steps <= 1:
                return 1.0
            progress = min(max(step, 0), total_optimizer_steps) / total_optimizer_steps
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unsupported lr schedule: {schedule}")


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def eval_payload(retain_eval: EvalSummary, forget_eval: EvalSummary) -> dict:
    return {
        "retain": asdict(retain_eval),
        "forget": asdict(forget_eval),
        "retain_forget_margin_gap": retain_eval.mean_margin - forget_eval.mean_margin,
        "retain_forget_mean_min_margin_gap": retain_eval.mean_min_margin - forget_eval.mean_min_margin,
        "retain_forget_nll_gap": retain_eval.avg_nll - forget_eval.avg_nll,
        "retain_forget_frac_margin_le_0_gap": retain_eval.frac_margin_le_0 - forget_eval.frac_margin_le_0,
        "retain_forget_frac_margin_le_0p5_gap": retain_eval.frac_margin_le_0p5 - forget_eval.frac_margin_le_0p5,
    }


def eval_summary_from_dict(payload: dict) -> EvalSummary:
    return EvalSummary(
        avg_nll=float(payload["avg_nll"]),
        mean_margin=float(payload["mean_margin"]),
        mean_min_margin=float(payload["mean_min_margin"]),
        global_min_margin=float(payload["global_min_margin"]),
        frac_margin_le_0=float(payload["frac_margin_le_0"]),
        frac_margin_le_0p5=float(payload["frac_margin_le_0p5"]),
        windows=int(payload["windows"]),
        valid_tokens=int(payload["valid_tokens"]),
    )


def delta_payload(current: dict, baseline: dict) -> dict:
    out = {}
    for key, value in current.items():
        base_value = baseline.get(key)
        if isinstance(value, dict) and isinstance(base_value, dict):
            out[key] = delta_payload(value, base_value)
        elif isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
            out[key] = value - base_value
    return out


def split_train_holdout_indices(
    total_items: int,
    holdout_items: int,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    if holdout_items <= 0:
        all_indices = list(range(total_items))
        return all_indices, []
    if total_items <= holdout_items:
        raise ValueError(
            f"Cannot hold out {holdout_items} items from a dataset with only {total_items} items."
        )
    indices = list(range(total_items))
    rng.shuffle(indices)
    holdout = sorted(indices[:holdout_items])
    train = sorted(indices[holdout_items:])
    return train, holdout


def mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1) / math.sqrt(arr.size))


def quantile_or_zero(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    try:
        return float(np.quantile(arr, quantile, method="linear"))
    except TypeError:
        return float(np.quantile(arr, quantile, interpolation="linear"))


def early_stop_probe_metrics(
    model,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    device: torch.device,
    tau: float,
    forget_margin_rho: float | None,
    margin_teacher_topk: int,
    use_teacher_competitor: bool,
    teacher_set_mode: str,
    teacher_logit_delta: float,
    competitor_beta: float,
    beta_parameterization: str,
    penalty_mode: str,
    continuation_cut_min_ratio: float,
    continuation_cut_max_ratio: float,
    continuation_min_prefix_tokens: int,
    continuation_min_suffix_tokens: int,
    max_batches: int | None,
    forget_violation_alpha: float | None = None,
    forget_margin_buffer_eta: float = 0.5,
    retain_teacher_model=None,
    need_retain_mean: bool = True,
    need_forget_mean: bool = True,
    need_forget_cont_mean: bool = True,
    need_forget_cont_frac_gt_tau: bool = True,
    need_retain_probe_kl: bool = False,
    need_surrogate_mean: bool = False,
    need_gap_stats: bool = True,
    need_cont_gap_stats: bool = True,
    deterministic_continuation: bool = False,
    deterministic_salt: int = 0,
) -> dict:
    retain_window_mean_margins: list[float] = []
    forget_window_mean_margins: list[float] = []
    forget_cont_window_mean_margins: list[float] = []
    forget_cont_window_logsigmoid_sums: list[float] = []
    forget_cont_window_frac_gt_zero: list[float] = []
    forget_cont_window_frac_gt_tau: list[float] = []
    forget_cont_window_lengths: list[int] = []
    retain_probe_kls: list[float] = []
    retain_batches = 0
    forget_batches = 0
    forget_cont_token_count = 0
    forget_cont_token_violations = 0.0
    forget_cont_surrogate_sum = 0.0
    forget_cont_surrogate_token_count = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in retain_loader:
            batch = to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_logits = None
            if need_retain_probe_kl and retain_teacher_model is not None:
                teacher_outputs = retain_teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                teacher_logits = teacher_outputs.logits
                retain_probe_kls.append(
                    float(
                        retain_kl_to_teacher(
                            outputs.logits,
                            teacher_logits,
                            batch["labels"],
                        ).item()
                    )
                )
            elif retain_teacher_model is not None:
                teacher_outputs = retain_teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                teacher_logits = teacher_outputs.logits
            margins, valid = compute_token_margins(
                outputs.logits,
                batch["labels"],
                teacher_logits=teacher_logits,
                teacher_topk=margin_teacher_topk,
                use_teacher_competitor=use_teacher_competitor,
                teacher_set_mode=teacher_set_mode,
                teacher_logit_delta=teacher_logit_delta,
                competitor_beta=competitor_beta,
                beta_parameterization=beta_parameterization,
            )
            if need_retain_mean or need_gap_stats or need_cont_gap_stats:
                for i in range(valid.size(0)):
                    valid_i = valid[i]
                    if valid_i.sum().item() == 0:
                        continue
                    seq_margins = margins[i][valid_i]
                    retain_window_mean_margins.append(float(seq_margins.mean().item()))
            retain_batches += 1
            if max_batches is not None and retain_batches >= max_batches:
                break

        for batch in forget_loader:
            batch = to_device(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            teacher_logits = None
            if retain_teacher_model is not None:
                teacher_outputs = retain_teacher_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                teacher_logits = teacher_outputs.logits
            margins, valid = compute_token_margins(
                outputs.logits,
                batch["labels"],
                teacher_logits=teacher_logits,
                teacher_topk=margin_teacher_topk,
                use_teacher_competitor=use_teacher_competitor,
                teacher_set_mode=teacher_set_mode,
                teacher_logit_delta=teacher_logit_delta,
                competitor_beta=competitor_beta,
                beta_parameterization=beta_parameterization,
            )
            if penalty_mode == "continuation":
                suffix_mask, _ = build_continuation_suffix_mask(
                    valid,
                    cut_min_ratio=continuation_cut_min_ratio,
                    cut_max_ratio=continuation_cut_max_ratio,
                    min_prefix_tokens=continuation_min_prefix_tokens,
                    min_suffix_tokens=continuation_min_suffix_tokens,
                    source_indices=batch["source_index"],
                    window_indices=batch["window_index"],
                    deterministic=deterministic_continuation,
                    deterministic_salt=deterministic_salt,
                )
                active_mask = valid & suffix_mask
            else:
                active_mask = valid

            for i in range(valid.size(0)):
                valid_i = valid[i]
                if valid_i.sum().item() == 0:
                    continue
                if need_forget_mean or need_gap_stats:
                    valid_seq_margins = margins[i][valid_i]
                    forget_window_mean_margins.append(float(valid_seq_margins.mean().item()))

                active_i = active_mask[i]
                if active_i.sum().item() == 0:
                    if need_forget_cont_mean or need_cont_gap_stats:
                        forget_cont_window_mean_margins.append(0.0)
                        forget_cont_window_logsigmoid_sums.append(0.0)
                    if need_forget_cont_frac_gt_tau:
                        forget_cont_window_frac_gt_zero.append(0.0)
                    if need_forget_cont_frac_gt_tau:
                        forget_cont_window_frac_gt_tau.append(0.0)
                else:
                    active_seq_margins = margins[i][active_i]
                    forget_cont_window_lengths.append(int(active_seq_margins.numel()))
                    if need_forget_cont_mean or need_cont_gap_stats:
                        forget_cont_window_mean_margins.append(float(active_seq_margins.mean().item()))
                        forget_cont_window_logsigmoid_sums.append(
                            float(F.logsigmoid(active_seq_margins.float()).sum().item())
                        )
                    if need_forget_cont_frac_gt_tau:
                        frac_gt_zero = float((active_seq_margins > 0).float().mean().item())
                        frac_gt_tau = float((active_seq_margins > tau).float().mean().item())
                        forget_cont_window_frac_gt_zero.append(frac_gt_zero)
                        forget_cont_window_frac_gt_tau.append(frac_gt_tau)
                        forget_cont_token_count += int(active_seq_margins.numel())
                        forget_cont_token_violations += float((active_seq_margins > tau).float().sum().item())
                    if need_surrogate_mean and forget_margin_rho is not None:
                        surrogate = buffered_margin_surrogate(
                            active_seq_margins.float(),
                            tau=tau,
                            eta=forget_margin_buffer_eta,
                        )
                        forget_cont_surrogate_sum += float(surrogate.sum().item())
                        forget_cont_surrogate_token_count += int(active_seq_margins.numel())
            forget_batches += 1
            if max_batches is not None and forget_batches >= max_batches:
                break

    if was_training:
        model.train()

    retain_mean_margin, retain_mean_margin_se = (
        mean_and_standard_error(retain_window_mean_margins)
        if (need_retain_mean or need_gap_stats or need_cont_gap_stats)
        else (0.0, 0.0)
    )
    forget_mean_margin, forget_mean_margin_se = (
        mean_and_standard_error(forget_window_mean_margins)
        if (need_forget_mean or need_gap_stats)
        else (0.0, 0.0)
    )
    forget_continuation_mean_margin, forget_continuation_mean_margin_se = (
        mean_and_standard_error(forget_cont_window_mean_margins)
        if (need_forget_cont_mean or need_cont_gap_stats)
        else (0.0, 0.0)
    )
    gap = retain_mean_margin - forget_mean_margin if need_gap_stats else 0.0
    gap_se = math.sqrt(retain_mean_margin_se ** 2 + forget_mean_margin_se ** 2) if need_gap_stats else 0.0
    cont_gap = retain_mean_margin - forget_continuation_mean_margin if need_cont_gap_stats else 0.0
    cont_gap_se = math.sqrt(retain_mean_margin_se ** 2 + forget_continuation_mean_margin_se ** 2) if need_cont_gap_stats else 0.0
    forget_cont_mean_len = (
        float(np.mean(forget_cont_window_lengths))
        if forget_cont_window_lengths
        else 0.0
    )
    if forget_margin_rho is not None and forget_violation_alpha is not None and forget_cont_mean_len > 0:
        forget_cont_log_exact_continuation_bound = (
            (1.0 - forget_violation_alpha) * forget_cont_mean_len * math.log(forget_margin_rho)
        )
        forget_cont_exact_continuation_bound = (
            math.exp(forget_cont_log_exact_continuation_bound)
            if forget_cont_log_exact_continuation_bound > -700
            else 0.0
        )
    else:
        forget_cont_log_exact_continuation_bound = None
        forget_cont_exact_continuation_bound = None

    return {
        "retain_mean_margin": retain_mean_margin,
        "forget_mean_margin": forget_mean_margin,
        "forget_continuation_mean_margin": forget_continuation_mean_margin,
        "retain_mean_margin_se": retain_mean_margin_se,
        "forget_mean_margin_se": forget_mean_margin_se,
        "forget_continuation_mean_margin_se": forget_continuation_mean_margin_se,
        "retain_forget_margin_gap": gap,
        "retain_forget_margin_gap_se": gap_se,
        "retain_forget_margin_gap_lower_bound": gap - 1.96 * gap_se,
        "retain_forget_continuation_margin_gap": cont_gap,
        "retain_forget_continuation_margin_gap_se": cont_gap_se,
        "retain_forget_continuation_margin_gap_lower_bound": cont_gap - 1.96 * cont_gap_se,
        "retain_mean_margin_q05": quantile_or_zero(retain_window_mean_margins, 0.05),
        "retain_mean_margin_q10": quantile_or_zero(retain_window_mean_margins, 0.10),
        "retain_mean_margin_q90": quantile_or_zero(retain_window_mean_margins, 0.90),
        "forget_mean_margin_q10": quantile_or_zero(forget_window_mean_margins, 0.10),
        "forget_mean_margin_q90": quantile_or_zero(forget_window_mean_margins, 0.90),
        "forget_continuation_mean_margin_q10": quantile_or_zero(forget_cont_window_mean_margins, 0.10),
        "forget_continuation_mean_margin_q90": quantile_or_zero(forget_cont_window_mean_margins, 0.90),
        "forget_continuation_logsigmoid_sum_mean": (
            float(np.mean(forget_cont_window_logsigmoid_sums))
            if forget_cont_window_logsigmoid_sums
            else 0.0
        ),
        "forget_continuation_logsigmoid_sum_q50": quantile_or_zero(forget_cont_window_logsigmoid_sums, 0.50),
        "forget_continuation_logsigmoid_sum_q90": quantile_or_zero(forget_cont_window_logsigmoid_sums, 0.90),
        "forget_continuation_logsigmoid_sum_max": (
            float(np.max(forget_cont_window_logsigmoid_sums))
            if forget_cont_window_logsigmoid_sums
            else 0.0
        ),
        "forget_continuation_frac_margin_gt_0": (
            float(np.mean(forget_cont_window_frac_gt_zero))
            if need_forget_cont_frac_gt_tau and forget_cont_window_frac_gt_zero
            else 0.0
        ),
        "forget_continuation_frac_margin_gt_0_q90": quantile_or_zero(forget_cont_window_frac_gt_zero, 0.90),
        "forget_continuation_frac_margin_gt_tau": (
            float(np.mean(forget_cont_window_frac_gt_tau))
            if need_forget_cont_frac_gt_tau and forget_cont_window_frac_gt_tau
            else 0.0
        ),
        "forget_continuation_frac_margin_gt_tau_q90": quantile_or_zero(forget_cont_window_frac_gt_tau, 0.90),
        "forget_continuation_token_frac_margin_gt_tau": (
            float(forget_cont_token_violations / forget_cont_token_count)
            if need_forget_cont_frac_gt_tau and forget_cont_token_count > 0
            else 0.0
        ),
        "forget_continuation_per_example_violation_mean": (
            float(np.mean(forget_cont_window_frac_gt_tau))
            if need_forget_cont_frac_gt_tau and forget_cont_window_frac_gt_tau
            else float("nan")
        ),
        "forget_continuation_surrogate_mean": (
            float(forget_cont_surrogate_sum / forget_cont_surrogate_token_count)
            if need_surrogate_mean and forget_cont_surrogate_token_count > 0
            else 0.0
        ),
        "forget_continuation_per_example_count": len(forget_cont_window_frac_gt_tau),
        "forget_cont_mean_len": forget_cont_mean_len,
        "forget_cont_log_exact_continuation_bound": forget_cont_log_exact_continuation_bound,
        "forget_cont_exact_continuation_bound": forget_cont_exact_continuation_bound,
        "retain_probe_kl": (
            float(np.mean(retain_probe_kls))
            if need_retain_probe_kl and retain_probe_kls
            else 0.0
        ),
        "retain_probe_batches": retain_batches,
        "forget_probe_batches": forget_batches,
        "retain_probe_windows": len(retain_window_mean_margins),
        "forget_probe_windows": len(forget_window_mean_margins),
    }


@dataclass
class BlindTransitionStopState:
    initialized: bool = False
    start_value: float = 0.0
    ema: float = 0.0
    prev_ema: float = 0.0
    prev_delta: float = 0.0
    min_ema: float = float("inf")
    min_ema_step: int = 0
    min_delta: float = 0.0
    plateau_count: int = 0
    entered_transition: bool = False


def update_blind_transition_stop_state(
    state: BlindTransitionStopState,
    probe_value: float,
    optimizer_step: int,
    ema_alpha: float,
    enter_max: float,
    min_drop_from_start: float,
    steep_slope_min: float,
    flat_slope_max: float,
    plateau_patience: int,
    plateau_ema_max: float,
    rebound_tol: float,
    hard_floor: float,
) -> dict:
    probe_value = float(probe_value)
    if not state.initialized:
        state.initialized = True
        state.start_value = probe_value
        state.ema = probe_value
        state.prev_ema = probe_value
        state.prev_delta = 0.0
        state.min_ema = probe_value
        state.min_ema_step = optimizer_step
        state.min_delta = 0.0
        state.plateau_count = 0
        state.entered_transition = False
        return {
            "blind_transition_probe_raw": probe_value,
            "blind_transition_probe_ema": probe_value,
            "blind_transition_probe_delta": 0.0,
            "blind_transition_start_value": state.start_value,
            "blind_transition_drop_from_start": 0.0,
            "blind_transition_min_ema": state.min_ema,
            "blind_transition_min_ema_step": state.min_ema_step,
            "blind_transition_min_delta": state.min_delta,
            "blind_transition_entered": False,
            "blind_transition_plateau_count": 0,
            "blind_transition_should_stop": False,
            "blind_transition_reason": "warmup",
        }

    previous_ema = state.ema
    ema = ema_alpha * probe_value + (1.0 - ema_alpha) * previous_ema
    delta = ema - previous_ema
    drop_from_start = state.start_value - ema

    if ema < state.min_ema:
        state.min_ema = ema
        state.min_ema_step = optimizer_step
    state.min_delta = min(state.min_delta, delta)

    if abs(delta) <= flat_slope_max:
        state.plateau_count += 1
    else:
        state.plateau_count = 0

    if (not state.entered_transition) and ema <= enter_max and drop_from_start >= min_drop_from_start:
        state.entered_transition = True

    rebound_ready = (
        state.entered_transition
        and state.min_delta <= -steep_slope_min
        and state.min_ema_step < optimizer_step
        and delta >= 0.0
        and (ema - state.min_ema) >= rebound_tol
    )
    plateau_ready = (
        state.entered_transition
        and state.min_delta <= -steep_slope_min
        and state.plateau_count >= plateau_patience
        and ema <= plateau_ema_max
    )
    floor_ready = (
        state.entered_transition
        and ema <= hard_floor
    )

    should_stop = rebound_ready or plateau_ready or floor_ready
    if rebound_ready:
        reason = "rebound_after_steep_descent"
    elif plateau_ready:
        reason = "plateau_after_steep_descent"
    elif floor_ready:
        reason = "hard_floor_after_transition"
    else:
        reason = "tracking"

    state.prev_ema = previous_ema
    state.ema = ema
    state.prev_delta = delta

    return {
        "blind_transition_probe_raw": probe_value,
        "blind_transition_probe_ema": ema,
        "blind_transition_probe_delta": delta,
        "blind_transition_start_value": state.start_value,
        "blind_transition_drop_from_start": drop_from_start,
        "blind_transition_min_ema": state.min_ema,
        "blind_transition_min_ema_step": state.min_ema_step,
        "blind_transition_min_delta": state.min_delta,
        "blind_transition_entered": state.entered_transition,
        "blind_transition_plateau_count": state.plateau_count,
        "blind_transition_should_stop": should_stop,
        "blind_transition_reason": reason,
    }


def load_tofu_online_fq_modules():
    raise NotImplementedError(
        "The optional online TOFU forget-quality probe was omitted from this public bundle."
    )


def build_tofu_online_fq_loaders(
    tokenizer,
    model_family: str,
    split: str,
    ds_size: int,
    batch_size: int,
    max_length: int,
) -> tuple[DataLoader, DataLoader, int]:
    TextDatasetQA, custom_data_collator, _, _ = load_tofu_online_fq_modules()

    keep = min(ds_size, len(paraphrase_ds.data), len(perturb_ds.data))
    paraphrase_ds.data = paraphrase_ds.data.select(range(keep))
    perturb_ds.data = perturb_ds.data.select(range(keep))

    paraphrase_loader = DataLoader(
        paraphrase_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_data_collator,
    )
    perturb_loader = DataLoader(
        perturb_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_data_collator,
    )
    return paraphrase_loader, perturb_loader, keep


def evaluate_tofu_forget_log(model, paraphrase_loader: DataLoader, perturb_loader: DataLoader) -> dict:
    _, _, get_batch_loss, _ = load_tofu_online_fq_modules()

    eval_logs = {
        "average_perturb_loss": [],
        "avg_paraphrased_loss": [],
        "paraphrased_loss": [],
        "perturb_loss": [],
        "num_token_paraphrased": [],
        "num_token_perturb": [],
    }

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch, perturb_batch in zip(paraphrase_loader, perturb_loader):
            input_ids, labels, attention_mask = batch
            perturb_input_ids, perturb_labels, perturb_attention_mask = perturb_batch

            if len(perturb_input_ids.shape) > 2:
                bsz, seq_len = perturb_input_ids.shape[0:2]
            else:
                bsz = perturb_input_ids.shape[0]
                seq_len = 1

            batch_dict = {
                "input_ids": input_ids.to(model.device),
                "labels": labels.to(model.device),
                "attention_mask": attention_mask.to(model.device),
            }
            perturb_dict = {
                "input_ids": perturb_input_ids.view(bsz * seq_len, -1).to(model.device),
                "labels": perturb_labels.view(bsz * seq_len, -1).to(model.device),
                "attention_mask": perturb_attention_mask.view(bsz * seq_len, -1).to(model.device),
            }

            outputs = model(**batch_dict)
            perturb_outputs = model(**perturb_dict)

            gt_loss = get_batch_loss(outputs.logits, batch_dict["labels"]).float()
            perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_dict["labels"]).view(bsz, seq_len).float()
            num_token_gt = (batch_dict["labels"] != -100).sum(-1)
            num_token_perturb = (perturb_dict["labels"] != -100).view(bsz, seq_len, -1).sum(-1)

            eval_logs["average_perturb_loss"] += (perturb_loss / num_token_perturb).cpu().tolist()
            eval_logs["avg_paraphrased_loss"] += (gt_loss / num_token_gt).cpu().tolist()
            eval_logs["paraphrased_loss"] += gt_loss.cpu().tolist()
            eval_logs["perturb_loss"] += perturb_loss.cpu().tolist()
            eval_logs["num_token_paraphrased"] += num_token_gt.cpu().tolist()
            eval_logs["num_token_perturb"] += num_token_perturb.cpu().tolist()

    if was_training:
        model.train()

    return {"eval_log_forget.json": eval_logs}


def online_tofu_fq_probe_metrics(
    model,
    paraphrase_loader: DataLoader,
    perturb_loader: DataLoader,
    reference_eval_log: dict,
) -> dict:
    _, _, _, get_forget_quality = load_tofu_online_fq_modules()
    candidate_eval = evaluate_tofu_forget_log(model, paraphrase_loader, perturb_loader)
    forget_quality, truth_ratio = get_forget_quality(candidate_eval, reference_eval_log)

    unlearn_truth_ratio = np.asarray(truth_ratio["Unlearn Truth Ratio"], dtype=np.float64)
    ref_truth_ratio = np.asarray(truth_ratio["Retain Truth Ratio"], dtype=np.float64)

    return {
        "candidate_eval": candidate_eval,
        "forget_quality": float(forget_quality["Forget Quality"]),
        "ks_test_pvalue": float(forget_quality["KS Test PVal Forget"]),
        "ks_test_statistic": float(forget_quality["KS Test Forget"]),
        "unlearn_truth_ratio_mean": float(unlearn_truth_ratio.mean()),
        "unlearn_truth_ratio_median": float(np.median(unlearn_truth_ratio)),
        "unlearn_truth_ratio_p90": float(np.quantile(unlearn_truth_ratio, 0.9)),
        "unlearn_truth_ratio_p95": float(np.quantile(unlearn_truth_ratio, 0.95)),
        "reference_truth_ratio_mean": float(ref_truth_ratio.mean()),
        "reference_truth_ratio_median": float(np.median(ref_truth_ratio)),
        "reference_truth_ratio_p90": float(np.quantile(ref_truth_ratio, 0.9)),
        "reference_truth_ratio_p95": float(np.quantile(ref_truth_ratio, 0.95)),
        "num_examples": int(unlearn_truth_ratio.size),
    }


def tofu_prompt_text(question: str) -> str:
    return (
        LLAMA2_CHAT_TEMPLATE["user_start_tag"]
        + question
        + LLAMA2_CHAT_TEMPLATE["user_end_tag"]
        + LLAMA2_CHAT_TEMPLATE["asst_start_tag"]
    )


def evaluate_tofu_rouge_probe(
    model,
    tokenizer,
    examples: list[dict],
    *,
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    was_training = model.training
    original_padding_side = tokenizer.padding_side
    model.eval()
    tokenizer.padding_side = "left"

    rouge_l_recall: list[float] = []
    rouge_l_f1: list[float] = []
    generations: list[str] = []

    try:
        with torch.no_grad():
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                prompts = [tofu_prompt_text(item["question"]) for item in batch]
                answers = [item["answer"] for item in batch]
                encoded = tokenizer(
                    prompts,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=max_input_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(model.device) for key, value in encoded.items()}
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                decoded = tokenizer.batch_decode(
                    output[:, encoded["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                for generation, answer in zip(decoded, answers):
                    generation = generation.strip()
                    generations.append(generation)
                    score = scorer.score(answer, generation)["rougeL"]
                    rouge_l_recall.append(float(score.recall))
                    rouge_l_f1.append(float(score.fmeasure))
    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()

    recall_arr = np.asarray(rouge_l_recall, dtype=np.float64)
    f1_arr = np.asarray(rouge_l_f1, dtype=np.float64)
    return {
        "num_examples": int(len(examples)),
        "rougeL_recall_mean": float(recall_arr.mean()) if recall_arr.size else None,
        "rougeL_recall_std": float(recall_arr.std(ddof=1)) if recall_arr.size > 1 else 0.0,
        "rougeL_f1_mean": float(f1_arr.mean()) if f1_arr.size else None,
        "rougeL_f1_std": float(f1_arr.std(ddof=1)) if f1_arr.size > 1 else 0.0,
        "sample_generation": generations[0] if generations else "",
    }


def online_tofu_rouge_probe_metrics(
    model,
    tokenizer,
    *,
    retain_examples: list[dict],
    forget_examples: list[dict],
    batch_size: int,
    max_input_length: int,
    max_new_tokens: int,
) -> dict:
    retain = evaluate_tofu_rouge_probe(
        model,
        tokenizer,
        retain_examples,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )
    forget = evaluate_tofu_rouge_probe(
        model,
        tokenizer,
        forget_examples,
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )
    return {"retain": retain, "forget": forget}


def flatten_online_rouge_probe_row(
    metrics: dict,
    *,
    epoch: int,
    global_step: int,
    optimizer_step: int,
    retain_examples: int,
    forget_examples: int,
    max_new_tokens: int,
) -> dict:
    retain = metrics["retain"]
    forget = metrics["forget"]
    return {
        "tag": "online_rouge_probe",
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_step": optimizer_step,
        "retain_num_examples": retain_examples,
        "forget_num_examples": forget_examples,
        "max_new_tokens": max_new_tokens,
        "retain_rougeL_recall": retain["rougeL_recall_mean"],
        "retain_rougeL_recall_std": retain["rougeL_recall_std"],
        "retain_rougeL_f1": retain["rougeL_f1_mean"],
        "retain_rougeL_f1_std": retain["rougeL_f1_std"],
        "forget_rougeL_recall": forget["rougeL_recall_mean"],
        "forget_rougeL_recall_std": forget["rougeL_recall_std"],
        "forget_rougeL_f1": forget["rougeL_f1_mean"],
        "forget_rougeL_f1_std": forget["rougeL_f1_std"],
        "retain_sample_generation": retain["sample_generation"],
        "forget_sample_generation": forget["sample_generation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="One-step margin-based unlearning on raw text windows.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=str(REPO_ROOT / "cache"))
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--corpus", type=str, choices=["news", "tofu"], default="news")
    parser.add_argument("--forget_split", type=str, default="forget10")
    parser.add_argument("--retain_split", type=str, default="retain90")
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument(
        "--wmdp_bio_forget_path",
        type=str,
        default=str(REPO_ROOT / "data" / "wmdp" / "bio_remove_dataset.jsonl"),
        help="Path to the raw forget corpus JSONL for WMDP-Bio training.",
    )
    parser.add_argument(
        "--wmdp_bio_retain_source",
        type=str,
        choices=["wikitext_test", "bio_retain_corpus"],
        default="wikitext_test",
        help="Retain corpus source for WMDP-Bio.",
    )
    parser.add_argument(
        "--wmdp_bio_retain_path",
        type=str,
        default=str(
            REPO_ROOT
            / "cache"
            / "downloads"
            / "wmdp"
            / "wmdp-corpora"
            / "wmdp-corpora"
            / "bio-retain-corpus.jsonl"
        ),
        help="Local raw retain corpus JSONL for WMDP-Bio when `--wmdp_bio_retain_source=bio_retain_corpus`.",
    )
    parser.add_argument(
        "--wmdp_bio_min_chars",
        type=int,
        default=50,
        help="Minimum characters for WMDP-Bio raw texts.",
    )
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--steps_per_epoch_cap",
        type=int,
        default=None,
        help="Optional cap on training micro-steps per epoch. Useful for large raw corpora like WMDP.",
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_schedule", type=str, choices=["constant", "cosine"], default="constant")
    parser.add_argument("--lambda_retain", type=float, default=1.0)
    parser.add_argument("--retain_loss_mode", type=str, choices=["ce", "kl_target"], default="ce")
    parser.add_argument(
        "--retain_teacher_model_dir",
        type=str,
        default=None,
        help=(
            "Frozen reference model used for retain KL when enabled, and also for "
            "teacher-anchored top-k margins / probe KL when provided."
        ),
    )
    parser.add_argument("--lambda_forget_margin", type=float, default=0.5)
    parser.add_argument(
        "--lambda_forget_ce",
        type=float,
        default=1.0,
        help="Forget CE coefficient used by --objective_mode=grad_diff; the optimized term is -lambda_forget_ce * CE_forget.",
    )
    parser.add_argument(
        "--simnpo_forget_weight",
        type=float,
        default=0.125,
        help="Forget-loss coefficient used by --objective_mode=simnpo.",
    )
    parser.add_argument("--simnpo_beta", type=float, default=4.5)
    parser.add_argument("--simnpo_delta", type=float, default=0.0)
    parser.add_argument(
        "--forget_margin_warmup_steps",
        type=int,
        default=0,
        help=(
            "Optional linear warmup for the forget coefficient. When > 0, the effective "
            "forget weight ramps from 0 to lambda_forget_margin over this many training steps."
        ),
    )
    parser.add_argument(
        "--objective_mode",
        type=str,
        choices=[
            "forget_margin",
            "grad_diff",
            "simnpo",
            "asymmetry_gap_continuation",
            "asymmetry_gap_full",
            "retain_anchor_forget_cap_full",
        ],
        default="forget_margin",
    )
    parser.add_argument("--lambda_gap", type=float, default=1.0)
    parser.add_argument("--margin_gap_target", type=float, default=2.0)
    parser.add_argument("--retain_margin_reference_beta", type=float, default=0.95)
    parser.add_argument(
        "--retain_margin_reference_mode",
        type=str,
        choices=["ema", "constant_start", "fixed_scalar"],
        default="ema",
    )
    parser.add_argument("--retain_margin_reference_value", type=float, default=None)
    parser.add_argument("--margin_tau", type=float, default=0.0)
    parser.add_argument(
        "--margin_teacher_topk",
        type=int,
        default=10,
        help=(
            "Maximum number of non-gold teacher alternatives used to build the "
            "teacher-anchored logsumexp competitor. Under delta mode, this acts "
            "as a cap on the teacher-filtered set size."
        ),
    )
    parser.add_argument(
        "--margin_teacher_set_mode",
        type=str,
        choices=["topk", "delta"],
        default="topk",
        help=(
            "Teacher competitor set construction. `topk` keeps the fixed top-k "
            "non-gold teacher tokens; `delta` keeps teacher non-gold tokens "
            "within `margin_teacher_logit_delta` of the teacher's best non-gold "
            "token, capped by `margin_teacher_topk`."
        ),
    )
    parser.add_argument(
        "--margin_teacher_logit_delta",
        type=float,
        default=0.5,
        help=(
            "Teacher logit radius used when `--margin_teacher_set_mode=delta`. "
            "Only teacher non-gold alternatives within this many logits of the "
            "teacher's best non-gold token remain in the restricted competitor set."
        ),
    )
    parser.add_argument(
        "--margin_competitor_beta",
        type=float,
        default=1.0,
        help=(
            "Smooth-max temperature on the student-side competitor aggregation over "
            "the teacher-defined set. beta=1 keeps the current logsumexp; larger "
            "beta sharpens it toward a hard max."
        ),
    )
    parser.add_argument(
        "--margin_beta_parameterization",
        type=str,
        choices=["tempered_posterior", "smoothmax_only"],
        default="tempered_posterior",
        help=(
            "How beta enters the reported margin. `tempered_posterior` uses "
            "beta*gold - logsumexp(beta*alternatives), which preserves the rho "
            "threshold interpretation under a beta-tempered restricted posterior. "
            "`smoothmax_only` keeps the old gold - (1/beta)logsumexp(beta*alternatives) "
            "scaling and requires threshold retuning when beta changes."
        ),
    )
    parser.add_argument(
        "--margin_competitor_source",
        type=str,
        choices=["teacher", "self"],
        default="teacher",
        help="Choose teacher-selected competitors (`teacher`) or the current model's own best non-gold competitor (`self`).",
    )
    parser.add_argument(
        "--forget_margin_rho",
        type=float,
        default=None,
        help=(
            "Optional restricted-posterior budget rho for calibrated forget loss. "
            "When set, tau is derived as log(rho / (1-rho)) and the forget loss "
            "switches to the buffered hinge surrogate."
        ),
    )
    parser.add_argument(
        "--forget_margin_buffer_eta",
        type=float,
        default=0.5,
        help="Positive buffer eta for the rho-mode buffered hinge surrogate.",
    )
    parser.add_argument(
        "--forget_margin_loss_mode",
        type=str,
        choices=["hinge", "sigmoid_hinge", "smooth_l1_centered"],
        default="hinge",
        help=(
            "Forget margin loss. `hinge` is ReLU(margin - tau); "
            "`sigmoid_hinge` multiplies it by sigmoid((margin - tau) / beta); "
            "`smooth_l1_centered` applies SmoothL1 to the continuation margin "
            "around 0 using beta=tau."
        ),
    )
    parser.add_argument(
        "--forget_margin_sigmoid_beta",
        type=float,
        default=0.5,
        help="Softness beta for --forget_margin_loss_mode=sigmoid_hinge. Larger values make the transition around tau smoother.",
    )
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of module names to apply LoRA to.",
    )
    parser.add_argument("--lora_bias", type=str, choices=["none", "all", "lora_only"], default="none")
    parser.add_argument("--forget_penalty_mode", type=str, choices=["full", "continuation"], default="full")
    parser.add_argument("--continuation_cut_min_ratio", type=float, default=0.2)
    parser.add_argument("--continuation_cut_max_ratio", type=float, default=0.6)
    parser.add_argument("--continuation_min_prefix_tokens", type=int, default=32)
    parser.add_argument("--continuation_min_suffix_tokens", type=int, default=32)
    parser.add_argument("--early_stop_forget_cont_frac_margin_gt_tau_max", type=float, default=None)
    parser.add_argument("--early_stop_retain_mean_margin_min", type=float, default=None)
    parser.add_argument("--early_stop_retain_forget_margin_gap_min", type=float, default=None)
    parser.add_argument("--early_stop_retain_forget_continuation_margin_gap_min", type=float, default=None)
    parser.add_argument("--early_stop_use_ucb", action="store_true")
    parser.add_argument("--early_stop_delta", type=float, default=0.05)
    parser.add_argument("--early_stop_two_stage", action="store_true")
    parser.add_argument(
        "--early_stop_trigger_metric",
        type=str,
        choices=["surrogate_mean", "violation_mean"],
        default="violation_mean",
    )
    parser.add_argument("--early_stop_trigger_threshold", type=float, default=None)
    parser.add_argument("--early_stop_retain_kl_tolerance", type=float, default=0.0)
    parser.add_argument("--early_stop_disable_retain_guard", action="store_true")
    parser.add_argument("--early_stop_confirm_num_forget_examples", type=int, default=64)
    parser.add_argument("--early_stop_confirm_use_ucb", action="store_true")
    parser.add_argument("--early_stop_confirm_alpha", type=float, default=None)
    parser.add_argument("--early_stop_confirm_delta", type=float, default=0.05)
    parser.add_argument("--early_stop_eval_every_steps", type=int, default=10)
    parser.add_argument("--early_stop_eval_batches", type=int, default=1)
    parser.add_argument(
        "--early_stop_probe_surrogate_mean",
        action="store_true",
        help="Also compute the buffered hinge surrogate mean on each early-stop probe.",
    )
    parser.add_argument("--early_stop_blind_transition", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--early_stop_blind_transition_ema_alpha", type=float, default=0.5)
    parser.add_argument("--early_stop_blind_transition_enter_max", type=float, default=0.35)
    parser.add_argument("--early_stop_blind_transition_min_drop_from_start", type=float, default=0.45)
    parser.add_argument("--early_stop_blind_transition_steep_slope_min", type=float, default=0.05)
    parser.add_argument("--early_stop_blind_transition_flat_slope_max", type=float, default=0.03)
    parser.add_argument("--early_stop_blind_transition_plateau_patience", type=int, default=2)
    parser.add_argument("--early_stop_blind_transition_plateau_ema_max", type=float, default=0.2)
    parser.add_argument("--early_stop_blind_transition_rebound_tol", type=float, default=0.04)
    parser.add_argument("--early_stop_blind_transition_hard_floor", type=float, default=0.12)
    parser.add_argument("--early_stop_probe_deterministic", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--early_stop_probe_deterministic_salt", type=int, default=0)
    parser.add_argument("--online_fq_probe", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--online_fq_probe_start_step", type=int, default=0)
    parser.add_argument("--online_fq_probe_split", type=str, default="forget10_perturbed")
    parser.add_argument("--online_fq_probe_ds_size", type=int, default=300)
    parser.add_argument("--online_fq_probe_batch_size", type=int, default=4)
    parser.add_argument("--online_fq_probe_max_length", type=int, default=200)
    parser.add_argument("--online_fq_probe_model_family", type=str, default="llama2-7b-local")
    parser.add_argument("--online_fq_reference_log", type=str, default=None)
    parser.add_argument("--online_rouge_probe", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--online_rouge_probe_every_steps", type=int, default=5)
    parser.add_argument("--online_rouge_probe_start_step", type=int, default=0)
    parser.add_argument("--online_rouge_probe_forget_examples", type=int, default=50)
    parser.add_argument("--online_rouge_probe_retain_examples", type=int, default=50)
    parser.add_argument("--online_rouge_probe_batch_size", type=int, default=4)
    parser.add_argument("--online_rouge_probe_max_input_length", type=int, default=256)
    parser.add_argument("--online_rouge_probe_max_new_tokens", type=int, default=80)
    parser.add_argument("--holdout_online_val", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--holdout_online_val_windows_per_split", type=int, default=8)
    parser.add_argument("--holdout_online_val_threshold", type=float, default=0.0)
    parser.add_argument("--holdout_online_val_lower_bound", type=str, choices=["yes", "no"], default="no")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--optimization_schedule",
        type=str,
        choices=["joint", "alternate", "projected_correction"],
        default="joint",
    )
    parser.add_argument("--retain_steps_per_forget_step", type=int, default=3)
    parser.add_argument("--forget_step_lr_scale", type=float, default=0.3)
    parser.add_argument("--safe_correction_ratio", type=float, default=0.05)
    parser.add_argument("--safe_projection_eps", type=float, default=1e-12)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--start_eval_cache_json", type=str, default=None)
    parser.add_argument("--force_probe_loop", action="store_true")
    parser.add_argument("--timing_mode", action="store_true")
    parser.add_argument("--timing_skip_save", action="store_true")
    parser.add_argument("--common_timing_json", type=str, default=None)
    parser.add_argument(
        "--no_model_save",
        action="store_true",
        help="Skip early-stop, epoch, and final model checkpoint writes; JSON diagnostics are still saved.",
    )
    parser.add_argument(
        "--early_stop_continue_after_steps",
        type=int,
        default=0,
        help=(
            "Diagnostic mode: after the first early-stop hit, keep training for this many "
            "additional optimizer steps while continuing to log probe dynamics."
        ),
    )
    parser.add_argument("--skip_final_eval", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_jsonl = out_dir / "train_metrics.jsonl"
    config_json = out_dir / "config.json"
    eval_json = out_dir / "raw_window_eval.json"
    eval_history_jsonl = out_dir / "eval_history.jsonl"
    timing_json = out_dir / "timing_result.json"
    online_fq_jsonl = out_dir / "online_fq_probe.jsonl"
    online_rouge_jsonl = out_dir / "online_rouge_probe.jsonl"

    if args.forget_margin_sigmoid_beta <= 0:
        raise ValueError("`--forget_margin_sigmoid_beta` must be > 0.")
    if args.simnpo_beta <= 0:
        raise ValueError("`--simnpo_beta` must be > 0.")
    if args.early_stop_confirm_num_forget_examples < 1:
        raise ValueError("`--early_stop_confirm_num_forget_examples` must be >= 1.")
    if args.early_stop_continue_after_steps < 0:
        raise ValueError("`--early_stop_continue_after_steps` must be >= 0.")
    if not (0.0 < args.early_stop_confirm_delta < 1.0):
        raise ValueError("`--early_stop_confirm_delta` must be in (0, 1).")
    if args.early_stop_confirm_alpha is not None and not (0.0 <= args.early_stop_confirm_alpha <= 1.0):
        raise ValueError("`--early_stop_confirm_alpha` must be in [0, 1].")
    if args.early_stop_two_stage or args.early_stop_trigger_threshold is not None:
        if args.early_stop_trigger_metric == "surrogate_mean" and args.forget_margin_rho is None:
            raise ValueError(
                "`--early_stop_trigger_metric surrogate_mean` requires `--forget_margin_rho`."
            )
    if args.early_stop_two_stage:
        if args.early_stop_trigger_threshold is None:
            raise ValueError("`--early_stop_trigger_threshold` is required when `--early_stop_two_stage` is enabled.")
        if (
            args.early_stop_confirm_alpha is None
            and args.early_stop_forget_cont_frac_margin_gt_tau_max is None
        ):
            raise ValueError(
                "Two-stage early stopping needs either `--early_stop_confirm_alpha` "
                "or `--early_stop_forget_cont_frac_margin_gt_tau_max`."
            )
    if args.online_fq_probe == "yes":
        if args.corpus != "tofu":
            raise ValueError("`--online_fq_probe yes` is currently only supported for corpus=tofu.")
        if not args.online_fq_reference_log:
            raise ValueError("`--online_fq_reference_log` is required when `--online_fq_probe yes`.")
    if args.online_rouge_probe == "yes":
        if args.corpus != "tofu":
            raise ValueError("`--online_rouge_probe yes` is currently only supported for corpus=tofu.")
        if args.online_rouge_probe_every_steps < 1:
            raise ValueError("`--online_rouge_probe_every_steps` must be >= 1.")
        if args.online_rouge_probe_batch_size < 1:
            raise ValueError("`--online_rouge_probe_batch_size` must be >= 1.")
        if args.online_rouge_probe_forget_examples < 1 or args.online_rouge_probe_retain_examples < 1:
            raise ValueError("online ROUGE probe example counts must be >= 1.")
    if not (0.0 < args.early_stop_blind_transition_ema_alpha <= 1.0):
        raise ValueError("`--early_stop_blind_transition_ema_alpha` must be in (0, 1].")
    if args.early_stop_blind_transition_plateau_patience < 1:
        raise ValueError("`--early_stop_blind_transition_plateau_patience` must be >= 1.")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{args.cache_dir}/hub")
    os.environ.setdefault("HF_DATASETS_CACHE", f"{args.cache_dir}/datasets")

    set_seed(args.seed)
    device = get_device(args.device)
    dtype = get_fp_dtype(device, args.precision)
    quiet_timing = bool(args.timing_mode)

    tokenizer_dir = args.tokenizer_dir or resolve_default_tokenizer_dir(args.cache_dir, args.model_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    online_fq_reference_eval = None
    online_fq_paraphrase_loader = None
    online_fq_perturb_loader = None
    online_fq_num_examples = 0
    online_rouge_retain_examples: list[dict] = []
    online_rouge_forget_examples: list[dict] = []
    if args.online_fq_probe == "yes":
        online_fq_reference_eval = json.loads(Path(args.online_fq_reference_log).read_text())
        online_fq_paraphrase_loader, online_fq_perturb_loader, online_fq_num_examples = build_tofu_online_fq_loaders(
            tokenizer=tokenizer,
            model_family=args.online_fq_probe_model_family,
            split=args.online_fq_probe_split,
            ds_size=args.online_fq_probe_ds_size,
            batch_size=args.online_fq_probe_batch_size,
            max_length=args.online_fq_probe_max_length,
        )
        if not quiet_timing:
            print(
                "Configured online TOFU FQ probe: "
                f"split={args.online_fq_probe_split} ds_size={online_fq_num_examples} "
                f"start_step={args.online_fq_probe_start_step}"
            )

    if not quiet_timing:
        print(f"Using device: {device}")
        print(f"Using dtype: {dtype}")
        print(f"Tokenizer: {tokenizer_dir}")

    if args.corpus == "tofu":
        tofu_cache_dir = args.dataset_cache_dir or f"{args.cache_dir}/datasets"
        retain_pairs = load_tofu_pairs(
            args.retain_split,
            tofu_cache_dir,
            args.max_samples,
            verbose=not quiet_timing,
        )
        forget_pairs = load_tofu_pairs(
            args.forget_split,
            tofu_cache_dir,
            args.max_samples,
            verbose=not quiet_timing,
        )
        if args.online_rouge_probe == "yes":
            online_rouge_retain_examples = retain_pairs[
                : min(args.online_rouge_probe_retain_examples, len(retain_pairs))
            ]
            online_rouge_forget_examples = forget_pairs[
                : min(args.online_rouge_probe_forget_examples, len(forget_pairs))
            ]
            if not quiet_timing:
                print(
                    "Configured online TOFU ROUGE probe: "
                    f"retain={len(online_rouge_retain_examples)} "
                    f"forget={len(online_rouge_forget_examples)} "
                    f"every={args.online_rouge_probe_every_steps} optimizer steps"
                )
        retain_ids, retain_labels, retain_attention, retain_meta = encode_tofu_pairs(
            tokenizer, retain_pairs, args.max_len
        )
        forget_ids, forget_labels, forget_attention, forget_meta = encode_tofu_pairs(
            tokenizer, forget_pairs, args.max_len
        )
        if not quiet_timing:
            print(f"Prepared {len(retain_ids)} retain QA examples and {len(forget_ids)} forget QA examples")
        retain_dataset = LabelledWindowDataset(retain_ids, retain_labels, retain_attention, retain_meta)
        forget_dataset = LabelledWindowDataset(forget_ids, forget_labels, forget_attention, forget_meta)
    elif args.corpus == "wmdp_bio":
        wmdp_cache_dir = args.dataset_cache_dir or f"{args.cache_dir}/datasets"
        retain_texts = load_wmdp_bio_texts(
            cache_dir=wmdp_cache_dir,
            split="retain",
            max_samples=args.max_samples,
            retain_source=args.wmdp_bio_retain_source,
            forget_path=args.wmdp_bio_forget_path,
            retain_path=args.wmdp_bio_retain_path,
            min_chars=args.wmdp_bio_min_chars,
        )
        forget_texts = load_wmdp_bio_texts(
            cache_dir=wmdp_cache_dir,
            split="forget",
            max_samples=args.max_samples,
            retain_source=args.wmdp_bio_retain_source,
            forget_path=args.wmdp_bio_forget_path,
            retain_path=args.wmdp_bio_retain_path,
            min_chars=args.wmdp_bio_min_chars,
        )
        retain_windows, retain_meta = chunk_texts_nonoverlap(tokenizer, retain_texts, args.max_len)
        forget_windows, forget_meta = chunk_texts_nonoverlap(tokenizer, forget_texts, args.max_len)
        if not quiet_timing:
            print(
                "Prepared "
                f"{len(retain_windows)} retain windows and {len(forget_windows)} forget windows "
                f"for WMDP-Bio (retain source: {args.wmdp_bio_retain_source})"
            )
        retain_dataset = WindowDataset(retain_windows, retain_meta, pad_id)
        forget_dataset = WindowDataset(forget_windows, forget_meta, pad_id)
    else:
        retain_texts = load_split_texts(args.cache_dir, args.corpus, "retain1", args.max_samples)
        forget_texts = load_split_texts(args.cache_dir, args.corpus, "forget", args.max_samples)
        retain_windows, retain_meta = chunk_texts_nonoverlap(tokenizer, retain_texts, args.max_len)
        forget_windows, forget_meta = chunk_texts_nonoverlap(tokenizer, forget_texts, args.max_len)
        if not quiet_timing:
            print(f"Prepared {len(retain_windows)} retain windows and {len(forget_windows)} forget windows")
        retain_dataset = WindowDataset(retain_windows, retain_meta, pad_id)
        forget_dataset = WindowDataset(forget_windows, forget_meta, pad_id)
    holdout_online_val_enabled = args.holdout_online_val == "yes"
    holdout_lower_bound_enabled = args.holdout_online_val_lower_bound == "yes"
    holdout_info: dict[str, object] = {
        "enabled": holdout_online_val_enabled,
        "windows_per_split": 0,
        "threshold": args.holdout_online_val_threshold,
        "use_lower_bound": holdout_lower_bound_enabled,
        "retain_holdout_indices": [],
        "forget_holdout_indices": [],
    }

    if holdout_online_val_enabled:
        if args.holdout_online_val_windows_per_split < 1:
            raise ValueError("`--holdout_online_val_windows_per_split` must be >= 1.")
        split_rng = random.Random(args.seed)
        retain_train_indices, retain_holdout_indices = split_train_holdout_indices(
            total_items=len(retain_dataset),
            holdout_items=args.holdout_online_val_windows_per_split,
            rng=split_rng,
        )
        forget_train_indices, forget_holdout_indices = split_train_holdout_indices(
            total_items=len(forget_dataset),
            holdout_items=args.holdout_online_val_windows_per_split,
            rng=split_rng,
        )
        retain_train_dataset = Subset(retain_dataset, retain_train_indices)
        forget_train_dataset = Subset(forget_dataset, forget_train_indices)
        retain_probe_dataset = Subset(retain_dataset, retain_holdout_indices)
        forget_probe_dataset = Subset(forget_dataset, forget_holdout_indices)
        holdout_info = {
            "enabled": True,
            "windows_per_split": args.holdout_online_val_windows_per_split,
            "threshold": args.holdout_online_val_threshold,
            "use_lower_bound": holdout_lower_bound_enabled,
            "retain_holdout_indices": retain_holdout_indices,
            "forget_holdout_indices": forget_holdout_indices,
        }
        if not quiet_timing:
            print(
                "Using fixed held-out online validation split: "
                f"{len(retain_holdout_indices)} retain windows, {len(forget_holdout_indices)} forget windows "
                f"(sampled once with seed {args.seed})"
            )
    else:
        retain_train_dataset = retain_dataset
        forget_train_dataset = forget_dataset
        retain_probe_dataset = retain_dataset
        forget_probe_dataset = forget_dataset

    retain_loader = DataLoader(retain_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    forget_loader = DataLoader(forget_train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    retain_eval_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    forget_eval_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    retain_probe_loader = DataLoader(retain_probe_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    forget_probe_loader = DataLoader(forget_probe_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    steps_per_epoch = max(len(retain_loader), len(forget_loader))
    if args.steps_per_epoch_cap is not None:
        if args.steps_per_epoch_cap < 1:
            raise ValueError("`--steps_per_epoch_cap` must be >= 1.")
        steps_per_epoch = min(steps_per_epoch, args.steps_per_epoch_cap)
    total_optimizer_steps = math.ceil(steps_per_epoch * args.epochs / args.gradient_accumulation_steps)
    if not quiet_timing:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total optimizer steps: {total_optimizer_steps}")

    config_payload = vars(args).copy()
    config_payload["train_retain_windows"] = len(retain_train_dataset)
    config_payload["train_forget_windows"] = len(forget_train_dataset)
    config_payload["full_retain_windows"] = len(retain_dataset)
    config_payload["full_forget_windows"] = len(forget_dataset)
    config_payload["holdout_online_val_info"] = holdout_info
    save_json(config_json, config_payload)

    if args.retain_margin_reference_mode == "fixed_scalar" and args.retain_margin_reference_value is None:
        raise ValueError("`--retain_margin_reference_value` is required when `--retain_margin_reference_mode=fixed_scalar`.")
    if args.retain_steps_per_forget_step < 1:
        raise ValueError("`--retain_steps_per_forget_step` must be >= 1.")
    if args.forget_step_lr_scale < 0:
        raise ValueError("`--forget_step_lr_scale` must be >= 0.")
    if args.safe_correction_ratio < 0:
        raise ValueError("`--safe_correction_ratio` must be >= 0.")
    if args.safe_projection_eps <= 0:
        raise ValueError("`--safe_projection_eps` must be > 0.")
    if args.lambda_retain < 0:
        raise ValueError("`--lambda_retain` must be >= 0.")
    if args.margin_teacher_topk < 1:
        raise ValueError("`--margin_teacher_topk` must be >= 1.")
    if args.margin_teacher_logit_delta < 0:
        raise ValueError("`--margin_teacher_logit_delta` must be >= 0.")
    if args.forget_margin_rho is not None and not (0.0 < args.forget_margin_rho < 1.0):
        raise ValueError("`--forget_margin_rho` must satisfy 0 < rho < 1.")
    if args.forget_margin_buffer_eta <= 0:
        raise ValueError("`--forget_margin_buffer_eta` must be > 0.")
    if args.margin_competitor_beta <= 0:
        raise ValueError("`--margin_competitor_beta` must be > 0.")
    if not (0.0 < args.early_stop_delta < 1.0):
        raise ValueError("`--early_stop_delta` must satisfy 0 < delta < 1.")
    if args.early_stop_retain_kl_tolerance < 0:
        raise ValueError("`--early_stop_retain_kl_tolerance` must be >= 0.")
    if args.retain_loss_mode == "kl_target" and not args.retain_teacher_model_dir:
        raise ValueError("`--retain_teacher_model_dir` is required when `--retain_loss_mode=kl_target`.")
    if args.early_stop_use_ucb and args.early_stop_forget_cont_frac_margin_gt_tau_max is None:
        raise ValueError(
            "`--early_stop_use_ucb` requires `--early_stop_forget_cont_frac_margin_gt_tau_max` as the UCB threshold."
        )
    if args.early_stop_use_ucb and args.retain_loss_mode != "kl_target":
        raise ValueError("`--early_stop_use_ucb` requires `--retain_loss_mode=kl_target` for the retain KL guard.")

    effective_tau = effective_margin_tau(args.margin_tau, args.forget_margin_rho)
    config_payload["forget_margin_tau_effective"] = effective_tau
    save_json(config_json, config_payload)

    model = load_model(args.model_dir, device, dtype, args.gradient_checkpointing)
    retain_teacher_model = None
    if args.retain_teacher_model_dir is not None:
        if not quiet_timing:
            print(f"Loading frozen retain teacher from: {args.retain_teacher_model_dir}")
        retain_teacher_model = load_model(
            args.retain_teacher_model_dir,
            device,
            dtype,
            gradient_checkpointing=False,
        )
        retain_teacher_model.eval()
        for param in retain_teacher_model.parameters():
            param.requires_grad_(False)
    if args.use_lora:
        target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        if not target_modules:
            raise ValueError("`--lora_target_modules` produced an empty list.")
        model = apply_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=target_modules,
            lora_bias=args.lora_bias,
        )

    trainable_params = trainable_parameters(model)
    optimizer = AdamW(trainable_params, lr=args.lr)
    scheduler = build_lr_scheduler(
        optimizer,
        schedule=args.lr_schedule,
        total_optimizer_steps=total_optimizer_steps,
    )
    grad_snapshot_device = device if device.type == "cuda" else torch.device("cpu")
    grad_snapshot_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    start_eval_cache_path = Path(args.start_eval_cache_json) if args.start_eval_cache_json else None
    if start_eval_cache_path is not None and start_eval_cache_path.exists():
        if not quiet_timing:
            print(f"\nLoading cached starting eval from {start_eval_cache_path} ...")
        baseline_eval_payload = load_json(start_eval_cache_path)
        initial_retain_eval = eval_summary_from_dict(baseline_eval_payload["retain"])
        initial_forget_eval = eval_summary_from_dict(baseline_eval_payload["forget"])
    else:
        if not quiet_timing:
            print("\nEvaluating starting model on raw windowed retain/forget data...")
        initial_retain_eval = evaluate_window_loader(
            model,
            retain_eval_loader,
            device,
            teacher_model=retain_teacher_model,
            teacher_topk=args.margin_teacher_topk,
            use_teacher_competitor=(args.margin_competitor_source == "teacher"),
            teacher_set_mode=args.margin_teacher_set_mode,
            teacher_logit_delta=args.margin_teacher_logit_delta,
            competitor_beta=args.margin_competitor_beta,
            beta_parameterization=args.margin_beta_parameterization,
        )
        initial_forget_eval = evaluate_window_loader(
            model,
            forget_eval_loader,
            device,
            teacher_model=retain_teacher_model,
            teacher_topk=args.margin_teacher_topk,
            use_teacher_competitor=(args.margin_competitor_source == "teacher"),
            teacher_set_mode=args.margin_teacher_set_mode,
            teacher_logit_delta=args.margin_teacher_logit_delta,
            competitor_beta=args.margin_competitor_beta,
            beta_parameterization=args.margin_beta_parameterization,
        )
        baseline_eval_payload = eval_payload(initial_retain_eval, initial_forget_eval)
        if start_eval_cache_path is not None:
            start_eval_cache_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(start_eval_cache_path, baseline_eval_payload)
            if not quiet_timing:
                print(f"Saved starting eval cache to {start_eval_cache_path}")
    baseline_eval_payload["tag"] = "start"
    baseline_eval_payload["epoch"] = 0
    baseline_eval_payload["optimizer_step"] = 0
    append_jsonl(eval_history_jsonl, baseline_eval_payload)

    blind_transition_enabled = args.early_stop_blind_transition == "yes"
    deterministic_probe_enabled = (
        args.early_stop_probe_deterministic == "yes" or blind_transition_enabled
    )
    stage1_trigger_only_enabled = (
        (not args.early_stop_two_stage) and args.early_stop_trigger_threshold is not None
    )
    two_stage_surrogate_enabled = (
        (args.early_stop_two_stage or stage1_trigger_only_enabled)
        and args.early_stop_trigger_metric == "surrogate_mean"
    )
    probe_surrogate_mean_enabled = two_stage_surrogate_enabled or args.early_stop_probe_surrogate_mean
    two_stage_confirm_alpha = (
        args.early_stop_confirm_alpha
        if args.early_stop_confirm_alpha is not None
        else args.early_stop_forget_cont_frac_margin_gt_tau_max
    )
    two_stage_confirm_batches = probe_batches_for_num_examples(
        args.early_stop_confirm_num_forget_examples,
        args.batch_size,
    )
    stop_criteria_enabled = holdout_online_val_enabled or blind_transition_enabled or any(
        threshold is not None
        for threshold in [
            args.early_stop_forget_cont_frac_margin_gt_tau_max,
            args.early_stop_retain_mean_margin_min,
            args.early_stop_retain_forget_margin_gap_min,
            args.early_stop_retain_forget_continuation_margin_gap_min,
        ]
    ) or args.early_stop_use_ucb or args.early_stop_two_stage or stage1_trigger_only_enabled
    probe_loop_enabled = (
        stop_criteria_enabled
        or args.online_fq_probe == "yes"
        or args.online_rouge_probe == "yes"
        or args.force_probe_loop
    )
    blind_transition_state = BlindTransitionStopState()
    retain_probe_kl_initial = 0.0
    probe_eval_index = 0
    two_stage_confirm_eval_index = 0
    early_stop_armed = False

    if probe_loop_enabled:
        start_probe_metrics = early_stop_probe_metrics(
            model=model,
            retain_loader=retain_probe_loader,
            forget_loader=forget_probe_loader,
            device=device,
            tau=effective_tau,
            forget_margin_rho=args.forget_margin_rho,
            margin_teacher_topk=args.margin_teacher_topk,
            use_teacher_competitor=(args.margin_competitor_source == "teacher"),
            teacher_set_mode=args.margin_teacher_set_mode,
            teacher_logit_delta=args.margin_teacher_logit_delta,
            competitor_beta=args.margin_competitor_beta,
            beta_parameterization=args.margin_beta_parameterization,
            penalty_mode=args.forget_penalty_mode,
            continuation_cut_min_ratio=args.continuation_cut_min_ratio,
            continuation_cut_max_ratio=args.continuation_cut_max_ratio,
            continuation_min_prefix_tokens=args.continuation_min_prefix_tokens,
            continuation_min_suffix_tokens=args.continuation_min_suffix_tokens,
            max_batches=None if holdout_online_val_enabled else max(1, args.early_stop_eval_batches),
            forget_violation_alpha=args.early_stop_forget_cont_frac_margin_gt_tau_max,
            forget_margin_buffer_eta=args.forget_margin_buffer_eta,
            retain_teacher_model=retain_teacher_model,
            need_retain_mean=(
                args.force_probe_loop
                or holdout_online_val_enabled
                or args.early_stop_retain_mean_margin_min is not None
                or args.early_stop_retain_forget_margin_gap_min is not None
                or args.early_stop_retain_forget_continuation_margin_gap_min is not None
            ),
            need_forget_mean=(
                args.force_probe_loop
                or holdout_online_val_enabled
                or args.early_stop_retain_forget_margin_gap_min is not None
            ),
            need_forget_cont_mean=(
                args.force_probe_loop
                or args.early_stop_retain_forget_continuation_margin_gap_min is not None
            ),
            need_forget_cont_frac_gt_tau=(
                args.force_probe_loop
                or args.early_stop_forget_cont_frac_margin_gt_tau_max is not None
                or args.early_stop_use_ucb
                or args.early_stop_two_stage
                or stage1_trigger_only_enabled
                or blind_transition_enabled
                or args.online_fq_probe == "yes"
                or args.online_rouge_probe == "yes"
            ),
            need_retain_probe_kl=(
                (not args.early_stop_disable_retain_guard)
                and (args.early_stop_use_ucb or args.early_stop_two_stage or stage1_trigger_only_enabled)
                and retain_teacher_model is not None
            ),
            need_surrogate_mean=probe_surrogate_mean_enabled,
            need_gap_stats=(
                args.force_probe_loop
                or holdout_online_val_enabled
                or args.early_stop_retain_forget_margin_gap_min is not None
            ),
            need_cont_gap_stats=(
                args.force_probe_loop
                or args.early_stop_retain_forget_continuation_margin_gap_min is not None
            ),
            deterministic_continuation=deterministic_probe_enabled,
            deterministic_salt=args.early_stop_probe_deterministic_salt,
        )
        start_probe_metrics.update(
            {
                "forget_margin_tau_effective": effective_tau,
                "forget_margin_rho": args.forget_margin_rho,
                "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
                "margin_teacher_topk": args.margin_teacher_topk,
                "margin_teacher_set_mode": args.margin_teacher_set_mode,
                "margin_teacher_logit_delta": args.margin_teacher_logit_delta,
                "margin_competitor_beta": args.margin_competitor_beta,
                "margin_beta_parameterization": args.margin_beta_parameterization,
                "margin_competitor_source": args.margin_competitor_source,
            }
        )
        retain_probe_kl_initial = start_probe_metrics["retain_probe_kl"]
        if args.early_stop_use_ucb or args.early_stop_two_stage or stage1_trigger_only_enabled:
            start_retain_ok = (
                True
                if args.early_stop_disable_retain_guard
                else (
                    start_probe_metrics["retain_probe_kl"]
                    <= retain_probe_kl_initial + args.early_stop_retain_kl_tolerance
                )
            )
            start_probe_metrics.update(
                {
                    "early_stop_retain_probe_kl": start_probe_metrics["retain_probe_kl"],
                    "early_stop_retain_probe_kl_initial": retain_probe_kl_initial,
                    "early_stop_retain_ok": bool(start_retain_ok),
                    "early_stop_disable_retain_guard": bool(args.early_stop_disable_retain_guard),
                }
            )
        if args.early_stop_use_ucb:
            start_q_hat = start_probe_metrics["forget_continuation_per_example_violation_mean"]
            start_n = start_probe_metrics["forget_continuation_per_example_count"]
            start_ucb, start_delta_j = hoeffding_ucb(start_q_hat, start_n, args.early_stop_delta, probe_eval_index)
            start_probe_metrics.update(
                {
                    "early_stop_forget_q_hat": start_q_hat,
                    "early_stop_forget_ucb": start_ucb,
                    "early_stop_delta_j": start_delta_j,
                    "early_stop_forget_certified": bool(
                        start_ucb <= args.early_stop_forget_cont_frac_margin_gt_tau_max
                    ),
                    "early_stop_retain_probe_kl": start_probe_metrics["retain_probe_kl"],
                    "early_stop_retain_probe_kl_initial": retain_probe_kl_initial,
                    "early_stop_retain_ok": bool(start_retain_ok),
                    "early_stop_disable_retain_guard": bool(args.early_stop_disable_retain_guard),
                }
            )
        if args.early_stop_two_stage or stage1_trigger_only_enabled:
            start_trigger_value = two_stage_trigger_value_from_metrics(
                start_probe_metrics,
                args.early_stop_trigger_metric,
            )
            start_trigger_pass = bool(
                start_trigger_value <= args.early_stop_trigger_threshold
                and start_probe_metrics["early_stop_retain_ok"]
            )
            start_probe_metrics.update(
                {
                    "early_stop_two_stage": bool(args.early_stop_two_stage),
                    "early_stop_stage1_only": bool(stage1_trigger_only_enabled),
                    "early_stop_trigger_metric": args.early_stop_trigger_metric,
                    "early_stop_trigger_threshold": args.early_stop_trigger_threshold,
                    "early_stop_trigger_value": start_trigger_value,
                    "early_stop_trigger_pass": start_trigger_pass,
                    "early_stop_armed": False,
                    "early_stop_confirm_num_forget_examples": args.early_stop_confirm_num_forget_examples,
                    "early_stop_confirm_batches": two_stage_confirm_batches,
                    "early_stop_confirm_use_ucb": args.early_stop_confirm_use_ucb,
                    "early_stop_confirm_alpha": two_stage_confirm_alpha,
                    "early_stop_confirm_delta": args.early_stop_confirm_delta,
                }
            )
        blind_transition_metrics = {}
        if blind_transition_enabled:
            blind_transition_metrics = update_blind_transition_stop_state(
                state=blind_transition_state,
                probe_value=start_probe_metrics["forget_continuation_frac_margin_gt_tau"],
                optimizer_step=0,
                ema_alpha=args.early_stop_blind_transition_ema_alpha,
                enter_max=args.early_stop_blind_transition_enter_max,
                min_drop_from_start=args.early_stop_blind_transition_min_drop_from_start,
                steep_slope_min=args.early_stop_blind_transition_steep_slope_min,
                flat_slope_max=args.early_stop_blind_transition_flat_slope_max,
                plateau_patience=args.early_stop_blind_transition_plateau_patience,
                plateau_ema_max=args.early_stop_blind_transition_plateau_ema_max,
                rebound_tol=args.early_stop_blind_transition_rebound_tol,
                hard_floor=args.early_stop_blind_transition_hard_floor,
            )
        if not quiet_timing:
            append_jsonl(
                log_jsonl,
                {
                    "tag": "early_stop_probe",
                    "epoch": 0,
                    "global_step": 0,
                    "optimizer_step": 0,
                    **start_probe_metrics,
                    **blind_transition_metrics,
                },
            )

    if args.online_rouge_probe == "yes" and args.online_rouge_probe_start_step <= 0:
        online_rouge_metrics = online_tofu_rouge_probe_metrics(
            model=model,
            tokenizer=tokenizer,
            retain_examples=online_rouge_retain_examples,
            forget_examples=online_rouge_forget_examples,
            batch_size=args.online_rouge_probe_batch_size,
            max_input_length=args.online_rouge_probe_max_input_length,
            max_new_tokens=args.online_rouge_probe_max_new_tokens,
        )
        online_rouge_row = flatten_online_rouge_probe_row(
            online_rouge_metrics,
            epoch=0,
            global_step=0,
            optimizer_step=0,
            retain_examples=len(online_rouge_retain_examples),
            forget_examples=len(online_rouge_forget_examples),
            max_new_tokens=args.online_rouge_probe_max_new_tokens,
        )
        append_jsonl(online_rouge_jsonl, online_rouge_row)
        if not quiet_timing:
            print(json.dumps(online_rouge_row, indent=2))

    global_step = 0
    optimizer_step = 0
    running = {
        "retain_ce": 0.0,
        "forget_ce": 0.0,
        "forget_margin_loss_raw": 0.0,
        "forget_margin_loss_weighted": 0.0,
        "lambda_forget_margin_effective": 0.0,
        "forget_margin_warmup_frac": 0.0,
        "forget_margin_hinge_loss": 0.0,
        "forget_margin_centered_smooth_l1_loss": 0.0,
        "forget_margin_sigmoid_weight_mean": 0.0,
        "asymmetry_gap_loss_raw": 0.0,
        "asymmetry_gap_loss_weighted": 0.0,
        "joint_total_loss": 0.0,
        "retain_grad_norm": 0.0,
        "secondary_grad_norm": 0.0,
        "secondary_grad_proj_norm": 0.0,
        "safe_correction_norm": 0.0,
        "safe_correction_ratio_realized": 0.0,
        "retain_secondary_grad_cosine": 0.0,
        "retain_secondary_proj_cosine": 0.0,
        "forget_margin_loss": 0.0,
        "total_loss": 0.0,
        "retain_mean_margin": 0.0,
        "forget_mean_margin": 0.0,
        "retain_min_margin": 0.0,
        "forget_min_margin": 0.0,
        "retain_frac_margin_le_0": 0.0,
        "forget_frac_margin_le_0": 0.0,
        "retain_frac_margin_le_0p5": 0.0,
        "forget_frac_margin_le_0p5": 0.0,
        "forget_frac_margin_gt_tau": 0.0,
        "forget_continuation_mean_margin": 0.0,
        "forget_continuation_min_margin": 0.0,
        "forget_continuation_frac_margin_gt_tau": 0.0,
        "forget_continuation_valid_tokens": 0.0,
        "retain_forget_continuation_margin_gap": 0.0,
        "forget_cut_ratio_mean": 0.0,
        "forget_suffix_tokens_mean": 0.0,
        "retain_forget_margin_gap": 0.0,
        "retain_forget_min_margin_gap": 0.0,
        "retain_forget_frac_margin_le_0_gap": 0.0,
        "retain_forget_frac_margin_le_0p5_gap": 0.0,
        "retain_margin_reference": 0.0,
        "retain_valid_tokens": 0.0,
        "forget_valid_tokens": 0.0,
        "grad_norm": 0.0,
        "retain_phase_steps": 0.0,
        "forget_phase_steps": 0.0,
        "count": 0,
    }

    retain_iter = cycle_loader(retain_loader)
    forget_iter = cycle_loader(forget_loader)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    early_stop_triggered = False
    early_stop_payload: dict | None = None
    training_should_stop = False
    early_stop_continue_until_step: int | None = None
    timing_started = False
    timing_start_perf: float | None = None
    timing_start_wall: datetime | None = None
    timing_end_perf: float | None = None
    timing_end_wall: datetime | None = None
    if args.retain_margin_reference_mode == "constant_start":
        retain_margin_reference: float | None = float(initial_retain_eval.mean_margin)
    elif args.retain_margin_reference_mode == "fixed_scalar":
        retain_margin_reference = float(args.retain_margin_reference_value)
    else:
        retain_margin_reference = None
    if retain_margin_reference is not None:
        if not quiet_timing:
            print(
                f"Using retain margin reference ({args.retain_margin_reference_mode}): "
                f"{retain_margin_reference:.6f}"
            )
    for epoch in range(1, args.epochs + 1):
        if not quiet_timing:
            print(f"\nEpoch {epoch}/{args.epochs}")
        for _ in range(steps_per_epoch):
            if args.timing_mode and not timing_started:
                timing_started = True
                timing_start_perf = time.perf_counter()
                timing_start_wall = datetime.now(timezone.utc)
                if not quiet_timing:
                    print(
                        f"[timer] start training window at epoch={epoch} step={global_step} "
                        f"utc={timing_start_wall.isoformat()}",
                        flush=True,
                    )
            global_step += 1
            retain_batch = to_device(next(retain_iter), device)
            forget_batch = to_device(next(forget_iter), device)

            retain_outputs = model(
                input_ids=retain_batch["input_ids"],
                attention_mask=retain_batch["attention_mask"],
            )
            forget_outputs = model(
                input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
            )

            retain_ce = retain_outputs.logits.new_tensor(0.0)
            if args.retain_loss_mode != "kl_target" or not quiet_timing:
                retain_ce = retain_cross_entropy(retain_outputs.logits, retain_batch["labels"])
            retain_kl_target = retain_outputs.logits.new_tensor(0.0)
            teacher_retain_logits = None
            if retain_teacher_model is not None:
                with torch.no_grad():
                    teacher_retain_outputs = retain_teacher_model(
                        input_ids=retain_batch["input_ids"],
                        attention_mask=retain_batch["attention_mask"],
                    )
                teacher_retain_logits = teacher_retain_outputs.logits
                retain_kl_target = retain_kl_to_teacher(
                    retain_outputs.logits,
                    teacher_retain_logits,
                    retain_batch["labels"],
                )
            teacher_forget_logits = None
            if retain_teacher_model is not None:
                with torch.no_grad():
                    teacher_forget_outputs = retain_teacher_model(
                        input_ids=forget_batch["input_ids"],
                        attention_mask=forget_batch["attention_mask"],
                    )
                teacher_forget_logits = teacher_forget_outputs.logits
            retain_loss_raw = retain_kl_target if args.retain_loss_mode == "kl_target" else retain_ce
            retain_loss_weighted = args.lambda_retain * retain_loss_raw
            forget_ce = forget_outputs.logits.new_tensor(0.0)
            forget_ce_stats = {"valid_tokens": 0}
            if not quiet_timing or args.objective_mode in {"grad_diff", "simnpo"}:
                forget_ce, forget_ce_stats = cross_entropy_stats(forget_outputs.logits, forget_batch["labels"])
            simnpo_loss = forget_outputs.logits.new_tensor(0.0)
            if args.objective_mode == "simnpo":
                simnpo_loss = simnpo_forget_loss(
                    forget_outputs.logits,
                    forget_batch["labels"],
                    beta=args.simnpo_beta,
                    delta=args.simnpo_delta,
                )
            forget_margin_loss, forget_stats, forget_mean_margin_tensor, forget_cont_mean_margin_tensor = forget_margin_penalty(
                forget_outputs.logits,
                forget_batch["labels"],
                teacher_forget_logits,
                args.margin_teacher_topk,
                args.margin_competitor_source == "teacher",
                args.margin_teacher_set_mode,
                args.margin_teacher_logit_delta,
                args.margin_competitor_beta,
                args.margin_beta_parameterization,
                effective_tau,
                args.forget_margin_rho,
                args.forget_margin_buffer_eta,
                args.forget_penalty_mode,
                args.forget_margin_loss_mode,
                args.forget_margin_sigmoid_beta,
                args.continuation_cut_min_ratio,
                args.continuation_cut_max_ratio,
                args.continuation_min_prefix_tokens,
                args.continuation_min_suffix_tokens,
                collect_stats=not quiet_timing,
            )
            retain_stats = None
            retain_mean_margin_tensor = None
            if not (quiet_timing and args.objective_mode == "forget_margin"):
                retain_stats = batch_margin_stats(
                    retain_outputs.logits,
                    retain_batch["labels"],
                    teacher_logits=teacher_retain_logits,
                    teacher_topk=args.margin_teacher_topk,
                    use_teacher_competitor=(args.margin_competitor_source == "teacher"),
                    teacher_set_mode=args.margin_teacher_set_mode,
                    teacher_logit_delta=args.margin_teacher_logit_delta,
                    competitor_beta=args.margin_competitor_beta,
                    beta_parameterization=args.margin_beta_parameterization,
                )
                retain_mean_margin_tensor = mean_margin_tensor(
                    retain_outputs.logits,
                    retain_batch["labels"],
                    teacher_logits=teacher_retain_logits,
                    teacher_topk=args.margin_teacher_topk,
                    use_teacher_competitor=(args.margin_competitor_source == "teacher"),
                    teacher_set_mode=args.margin_teacher_set_mode,
                    teacher_logit_delta=args.margin_teacher_logit_delta,
                    competitor_beta=args.margin_competitor_beta,
                    beta_parameterization=args.margin_beta_parameterization,
                )
            if retain_teacher_model is not None:
                del teacher_retain_outputs
                del teacher_forget_outputs
            if args.forget_margin_warmup_steps > 0:
                forget_warmup_frac = min(global_step / float(args.forget_margin_warmup_steps), 1.0)
            else:
                forget_warmup_frac = 1.0
            lambda_forget_margin_effective = args.lambda_forget_margin * forget_warmup_frac
            if quiet_timing and args.objective_mode == "forget_margin":
                asymmetry_gap_loss = retain_loss_weighted.new_tensor(0.0)
            else:
                retain_forget_gap_tensor = retain_mean_margin_tensor - forget_mean_margin_tensor
                retain_forget_cont_gap_tensor = retain_mean_margin_tensor - forget_cont_mean_margin_tensor
                current_retain_margin = float(retain_mean_margin_tensor.detach().cpu().item())
                if args.retain_margin_reference_mode == "ema":
                    if retain_margin_reference is None:
                        retain_margin_reference = current_retain_margin
                    else:
                        beta = args.retain_margin_reference_beta
                        retain_margin_reference = beta * retain_margin_reference + (1.0 - beta) * current_retain_margin
                elif retain_margin_reference is None:
                    retain_margin_reference = current_retain_margin
                retain_ref_tensor = retain_mean_margin_tensor.new_tensor(retain_margin_reference)

                if args.objective_mode in {"grad_diff", "simnpo"}:
                    asymmetry_gap_loss = retain_loss_weighted.new_tensor(0.0)
                elif args.objective_mode == "asymmetry_gap_full":
                    asymmetry_gap_loss = torch.relu(args.margin_gap_target - retain_forget_gap_tensor)
                elif args.objective_mode == "retain_anchor_forget_cap_full":
                    asymmetry_gap_loss = torch.relu(
                        forget_mean_margin_tensor - (retain_ref_tensor - args.margin_gap_target)
                    )
                else:
                    asymmetry_gap_loss = torch.relu(args.margin_gap_target - retain_forget_cont_gap_tensor)

            if args.objective_mode == "forget_margin":
                # Warm-start the forget term so retain CE can stabilize early before the
                # teacher-anchored forget pressure reaches full strength.
                secondary_loss_weighted = lambda_forget_margin_effective * forget_margin_loss
                gap_loss_weighted = retain_loss_weighted.new_tensor(0.0)
            elif args.objective_mode == "grad_diff":
                secondary_loss_weighted = -args.lambda_forget_ce * forget_ce
                gap_loss_weighted = retain_loss_weighted.new_tensor(0.0)
            elif args.objective_mode == "simnpo":
                secondary_loss_weighted = args.simnpo_forget_weight * simnpo_loss
                gap_loss_weighted = retain_loss_weighted.new_tensor(0.0)
            else:
                gap_loss_weighted = args.lambda_gap * asymmetry_gap_loss
                secondary_loss_weighted = gap_loss_weighted

            joint_total_loss = retain_loss_weighted + secondary_loss_weighted
            retain_grad_norm_value = 0.0
            secondary_grad_norm_value = 0.0
            secondary_grad_proj_norm_value = 0.0
            safe_correction_norm_value = 0.0
            safe_correction_ratio_realized = 0.0
            retain_secondary_grad_cosine = 0.0
            retain_secondary_proj_cosine = 0.0

            if args.optimization_schedule == "projected_correction":
                update_phase = "projected"
                optimizer.zero_grad(set_to_none=True)
                needs_retain_graph = args.objective_mode in {"asymmetry_gap_full", "asymmetry_gap_continuation"}
                retain_loss_weighted.backward(retain_graph=needs_retain_graph)
                retain_grad = snapshot_current_grads(
                    trainable_params,
                    storage_dtype=grad_snapshot_dtype,
                    storage_device=grad_snapshot_device,
                )
                optimizer.zero_grad(set_to_none=True)
                secondary_loss_weighted.backward()
                secondary_grad = snapshot_current_grads(
                    trainable_params,
                    storage_dtype=grad_snapshot_dtype,
                    storage_device=grad_snapshot_device,
                )
                optimizer.zero_grad(set_to_none=True)

                retain_grad_norm_value = grad_norm_from_list(retain_grad)
                secondary_grad_norm_value = grad_norm_from_list(secondary_grad)
                if retain_grad_norm_value <= args.safe_projection_eps:
                    proj_coeff = 0.0
                    secondary_grad_proj_norm_value = 0.0
                    clip_scale = 0.0
                else:
                    proj_coeff = grad_dot(secondary_grad, retain_grad) / (
                        retain_grad_norm_value * retain_grad_norm_value + args.safe_projection_eps
                    )
                    secondary_grad_proj_norm_value = projected_grad_norm(
                        secondary_grad,
                        retain_grad,
                        proj_coeff,
                    )
                    if secondary_grad_proj_norm_value <= args.safe_projection_eps:
                        clip_scale = 0.0
                    else:
                        clip_scale = min(
                            1.0,
                            args.safe_correction_ratio * retain_grad_norm_value
                            / (secondary_grad_proj_norm_value + args.safe_projection_eps),
                        )

                safe_correction_norm_value = secondary_grad_proj_norm_value * clip_scale
                safe_correction_ratio_realized = (
                    safe_correction_norm_value / (retain_grad_norm_value + args.safe_projection_eps)
                    if retain_grad_norm_value > args.safe_projection_eps
                    else 0.0
                )
                retain_secondary_grad_cosine = grad_cosine_from_lists(
                    retain_grad,
                    secondary_grad,
                    args.safe_projection_eps,
                )
                retain_secondary_proj_cosine = projected_grad_cosine(
                    retain_grads=retain_grad,
                    secondary_grads=secondary_grad,
                    proj_coeff=proj_coeff,
                    eps=args.safe_projection_eps,
                )

                assign_safe_projected_grad(
                    params=trainable_params,
                    retain_grads=retain_grad,
                    secondary_grads=secondary_grad,
                    proj_coeff=proj_coeff,
                    clip_scale=clip_scale,
                    grad_accum_steps=args.gradient_accumulation_steps,
                )
                total_loss = joint_total_loss
                grad_norm_value = gradient_norm(trainable_params)
            elif args.optimization_schedule == "alternate":
                cycle_length = args.retain_steps_per_forget_step + 1
                schedule_step_index = (global_step - 1) // max(1, args.gradient_accumulation_steps)
                schedule_pos = schedule_step_index % cycle_length
                if schedule_pos < args.retain_steps_per_forget_step:
                    update_phase = "retain"
                    total_loss = retain_loss_weighted
                else:
                    update_phase = "forget"
                    total_loss = args.forget_step_lr_scale * secondary_loss_weighted
            else:
                update_phase = "joint"
                total_loss = joint_total_loss

            if args.optimization_schedule != "projected_correction":
                (total_loss / args.gradient_accumulation_steps).backward()
                grad_norm_value = gradient_norm(trainable_params)

            did_optimizer_step = False
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                did_optimizer_step = True

            metric_row = None
            if not quiet_timing:
                metric_row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "retain_ce": float(retain_ce.detach().cpu().item()),
                    "retain_kl_target": float(retain_kl_target.detach().cpu().item()),
                    "retain_loss": float(retain_loss_raw.detach().cpu().item()),
                    "retain_loss_weighted": float(retain_loss_weighted.detach().cpu().item()),
                    "retain_loss_mode": args.retain_loss_mode,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "forget_ce": float(forget_ce.detach().cpu().item()),
                    "forget_margin_loss_raw": float(forget_margin_loss.detach().cpu().item()),
                    "forget_margin_loss_weighted": float(secondary_loss_weighted.detach().cpu().item()),
                    "simnpo_loss_raw": float(simnpo_loss.detach().cpu().item()),
                    "simnpo_loss_weighted": (
                        float(secondary_loss_weighted.detach().cpu().item())
                        if args.objective_mode == "simnpo"
                        else 0.0
                    ),
                    "simnpo_beta": args.simnpo_beta,
                    "simnpo_delta": args.simnpo_delta,
                    "simnpo_forget_weight": args.simnpo_forget_weight,
                    "lambda_forget_margin_effective": float(lambda_forget_margin_effective),
                    "forget_margin_warmup_frac": float(forget_warmup_frac),
                    "forget_margin_loss_mode": args.forget_margin_loss_mode,
                    "forget_margin_loss_mode_effective": (
                        "buffered_hinge_rho" if args.forget_margin_rho is not None else args.forget_margin_loss_mode
                    ),
                    "forget_margin_sigmoid_beta": args.forget_margin_sigmoid_beta,
                    "forget_margin_hinge_loss": forget_stats["hinge_margin_loss"],
                    "forget_margin_buffered_hinge_loss": forget_stats["buffered_hinge_margin_loss"],
                    "forget_margin_centered_smooth_l1_loss": forget_stats["centered_smooth_l1_loss"],
                    "forget_margin_sigmoid_weight_mean": forget_stats["forget_margin_sigmoid_weight_mean"],
                    "forget_margin_rho": args.forget_margin_rho,
                    "forget_margin_tau_effective": forget_stats["forget_margin_tau_effective"],
                    "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
                    "margin_teacher_topk": args.margin_teacher_topk,
                    "margin_teacher_set_mode": args.margin_teacher_set_mode,
                    "margin_teacher_logit_delta": args.margin_teacher_logit_delta,
                    "margin_competitor_beta": args.margin_competitor_beta,
                    "margin_beta_parameterization": args.margin_beta_parameterization,
                    "margin_competitor_source": args.margin_competitor_source,
                    "asymmetry_gap_loss_raw": float(asymmetry_gap_loss.detach().cpu().item()),
                    "asymmetry_gap_loss_weighted": float(gap_loss_weighted.detach().cpu().item()),
                    "joint_total_loss": float(joint_total_loss.detach().cpu().item()),
                    "retain_grad_norm": retain_grad_norm_value,
                    "secondary_grad_norm": secondary_grad_norm_value,
                    "secondary_grad_proj_norm": secondary_grad_proj_norm_value,
                    "safe_correction_norm": safe_correction_norm_value,
                    "safe_correction_ratio_realized": safe_correction_ratio_realized,
                    "retain_secondary_grad_cosine": retain_secondary_grad_cosine,
                    "retain_secondary_proj_cosine": retain_secondary_proj_cosine,
                    "forget_margin_loss": float(forget_margin_loss.detach().cpu().item()),
                    "total_loss": float(total_loss.detach().cpu().item()),
                    "update_phase": update_phase,
                    "retain_mean_margin": retain_stats["mean_margin"],
                    "forget_mean_margin": forget_stats["mean_margin"],
                    "retain_min_margin": retain_stats["min_margin"],
                    "forget_min_margin": forget_stats["min_margin"],
                    "retain_frac_margin_le_0": retain_stats["frac_margin_le_0"],
                    "forget_frac_margin_le_0": forget_stats["frac_margin_le_0"],
                    "retain_frac_margin_le_0p5": retain_stats["frac_margin_le_0p5"],
                    "forget_frac_margin_le_0p5": forget_stats["frac_margin_le_0p5"],
                    "forget_frac_margin_gt_tau": forget_stats["frac_margin_gt_tau"],
                    "forget_continuation_mean_margin": forget_stats["continuation_mean_margin"],
                    "forget_continuation_min_margin": forget_stats["continuation_min_margin"],
                    "forget_continuation_frac_margin_gt_tau": forget_stats["continuation_frac_margin_gt_tau"],
                    "forget_continuation_valid_tokens": forget_stats["continuation_valid_tokens"],
                    "retain_forget_continuation_margin_gap": retain_stats["mean_margin"] - forget_stats["continuation_mean_margin"],
                    "forget_cut_ratio_mean": forget_stats["cut_ratio_mean"],
                    "forget_suffix_tokens_mean": forget_stats["suffix_tokens_mean"],
                    "retain_forget_margin_gap": retain_stats["mean_margin"] - forget_stats["mean_margin"],
                    "retain_forget_min_margin_gap": retain_stats["min_margin"] - forget_stats["min_margin"],
                    "retain_forget_frac_margin_le_0_gap": retain_stats["frac_margin_le_0"] - forget_stats["frac_margin_le_0"],
                    "retain_forget_frac_margin_le_0p5_gap": retain_stats["frac_margin_le_0p5"] - forget_stats["frac_margin_le_0p5"],
                    "retain_margin_reference": retain_margin_reference,
                    "retain_valid_tokens": retain_stats["valid_tokens"],
                    "forget_valid_tokens": forget_ce_stats["valid_tokens"],
                    "grad_norm": grad_norm_value,
                }
                append_jsonl(log_jsonl, metric_row)

            if (
                probe_loop_enabled
                and did_optimizer_step
                and optimizer_step > 0
                and optimizer_step % max(1, args.early_stop_eval_every_steps) == 0
            ):
                probe_metrics = early_stop_probe_metrics(
                    model=model,
                    retain_loader=retain_probe_loader,
                    forget_loader=forget_probe_loader,
                    device=device,
                    tau=effective_tau,
                    forget_margin_rho=args.forget_margin_rho,
                    margin_teacher_topk=args.margin_teacher_topk,
                    use_teacher_competitor=(args.margin_competitor_source == "teacher"),
                    teacher_set_mode=args.margin_teacher_set_mode,
                    teacher_logit_delta=args.margin_teacher_logit_delta,
                    competitor_beta=args.margin_competitor_beta,
                    beta_parameterization=args.margin_beta_parameterization,
                    penalty_mode=args.forget_penalty_mode,
                    continuation_cut_min_ratio=args.continuation_cut_min_ratio,
                    continuation_cut_max_ratio=args.continuation_cut_max_ratio,
                    continuation_min_prefix_tokens=args.continuation_min_prefix_tokens,
                    continuation_min_suffix_tokens=args.continuation_min_suffix_tokens,
                    max_batches=None if holdout_online_val_enabled else max(1, args.early_stop_eval_batches),
                    forget_violation_alpha=args.early_stop_forget_cont_frac_margin_gt_tau_max,
                    forget_margin_buffer_eta=args.forget_margin_buffer_eta,
                    retain_teacher_model=retain_teacher_model,
                    need_retain_mean=(
                        args.force_probe_loop
                        or holdout_online_val_enabled
                        or args.early_stop_retain_mean_margin_min is not None
                        or args.early_stop_retain_forget_margin_gap_min is not None
                        or args.early_stop_retain_forget_continuation_margin_gap_min is not None
                    ),
                    need_forget_mean=(
                        args.force_probe_loop
                        or holdout_online_val_enabled
                        or args.early_stop_retain_forget_margin_gap_min is not None
                    ),
                    need_forget_cont_mean=(
                        args.force_probe_loop
                        or args.early_stop_retain_forget_continuation_margin_gap_min is not None
                    ),
                    need_forget_cont_frac_gt_tau=(
                        args.force_probe_loop
                        or args.early_stop_forget_cont_frac_margin_gt_tau_max is not None
                        or args.early_stop_use_ucb
                        or args.early_stop_two_stage
                        or stage1_trigger_only_enabled
                        or blind_transition_enabled
                        or args.online_fq_probe == "yes"
                        or args.online_rouge_probe == "yes"
                    ),
                    need_retain_probe_kl=(
                        (not args.early_stop_disable_retain_guard)
                        and (args.early_stop_use_ucb or args.early_stop_two_stage or stage1_trigger_only_enabled)
                        and retain_teacher_model is not None
                    ),
                    need_surrogate_mean=probe_surrogate_mean_enabled,
                    need_gap_stats=(
                        args.force_probe_loop
                        or holdout_online_val_enabled
                        or args.early_stop_retain_forget_margin_gap_min is not None
                    ),
                    need_cont_gap_stats=(
                        args.force_probe_loop
                        or args.early_stop_retain_forget_continuation_margin_gap_min is not None
                    ),
                    deterministic_continuation=deterministic_probe_enabled,
                    deterministic_salt=args.early_stop_probe_deterministic_salt,
                )
                probe_metrics.update(
                    {
                        "forget_margin_tau_effective": effective_tau,
                        "forget_margin_rho": args.forget_margin_rho,
                        "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
                        "margin_teacher_topk": args.margin_teacher_topk,
                        "margin_teacher_set_mode": args.margin_teacher_set_mode,
                        "margin_teacher_logit_delta": args.margin_teacher_logit_delta,
                        "margin_competitor_beta": args.margin_competitor_beta,
                        "margin_beta_parameterization": args.margin_beta_parameterization,
                        "margin_competitor_source": args.margin_competitor_source,
                    }
                )
                probe_eval_index += 1
                if args.early_stop_use_ucb or args.early_stop_two_stage or stage1_trigger_only_enabled:
                    retain_ok = (
                        True
                        if args.early_stop_disable_retain_guard
                        else (
                            probe_metrics["retain_probe_kl"]
                            <= retain_probe_kl_initial + args.early_stop_retain_kl_tolerance
                        )
                    )
                    probe_metrics.update(
                        {
                            "early_stop_retain_probe_kl": probe_metrics["retain_probe_kl"],
                            "early_stop_retain_probe_kl_initial": retain_probe_kl_initial,
                            "early_stop_retain_ok": bool(retain_ok),
                            "early_stop_disable_retain_guard": bool(args.early_stop_disable_retain_guard),
                        }
                    )
                if args.early_stop_use_ucb:
                    # Early stopping in UCB mode stops at the first probe step where the
                    # empirical violation rate is certified small enough, while retain KL drift
                    # remains acceptable.
                    q_hat = probe_metrics["forget_continuation_per_example_violation_mean"]
                    n = probe_metrics["forget_continuation_per_example_count"]
                    ucb, delta_j = hoeffding_ucb(q_hat, n, args.early_stop_delta, probe_eval_index)
                    forget_certified = bool(ucb <= args.early_stop_forget_cont_frac_margin_gt_tau_max)
                    probe_metrics.update(
                        {
                            "early_stop_forget_q_hat": q_hat,
                            "early_stop_forget_ucb": ucb,
                            "early_stop_delta_j": delta_j,
                            "early_stop_forget_certified": forget_certified,
                            "early_stop_retain_probe_kl": probe_metrics["retain_probe_kl"],
                            "early_stop_retain_probe_kl_initial": retain_probe_kl_initial,
                            "early_stop_retain_ok": bool(retain_ok),
                            "early_stop_disable_retain_guard": bool(args.early_stop_disable_retain_guard),
                        }
                    )
                if args.early_stop_two_stage or stage1_trigger_only_enabled:
                    # The tiny probe only detects whether we are close enough to the
                    # forget-feasible region to justify a larger confirmation pass.
                    trigger_value = two_stage_trigger_value_from_metrics(
                        probe_metrics,
                        args.early_stop_trigger_metric,
                    )
                    early_stop_armed = bool(
                        trigger_value <= args.early_stop_trigger_threshold
                        and probe_metrics["early_stop_retain_ok"]
                    )
                    probe_metrics.update(
                        {
                            "early_stop_two_stage": bool(args.early_stop_two_stage),
                            "early_stop_stage1_only": bool(stage1_trigger_only_enabled),
                            "early_stop_trigger_metric": args.early_stop_trigger_metric,
                            "early_stop_trigger_threshold": args.early_stop_trigger_threshold,
                            "early_stop_trigger_value": trigger_value,
                            "early_stop_trigger_pass": bool(early_stop_armed),
                            "early_stop_armed": bool(early_stop_armed),
                            "early_stop_confirm_num_forget_examples": args.early_stop_confirm_num_forget_examples,
                            "early_stop_confirm_batches": two_stage_confirm_batches,
                            "early_stop_confirm_use_ucb": args.early_stop_confirm_use_ucb,
                            "early_stop_confirm_alpha": two_stage_confirm_alpha,
                            "early_stop_confirm_delta": args.early_stop_confirm_delta,
                        }
                    )
                blind_transition_metrics = {}
                if blind_transition_enabled:
                    blind_transition_metrics = update_blind_transition_stop_state(
                        state=blind_transition_state,
                        probe_value=probe_metrics["forget_continuation_frac_margin_gt_tau"],
                        optimizer_step=optimizer_step,
                        ema_alpha=args.early_stop_blind_transition_ema_alpha,
                        enter_max=args.early_stop_blind_transition_enter_max,
                        min_drop_from_start=args.early_stop_blind_transition_min_drop_from_start,
                        steep_slope_min=args.early_stop_blind_transition_steep_slope_min,
                        flat_slope_max=args.early_stop_blind_transition_flat_slope_max,
                        plateau_patience=args.early_stop_blind_transition_plateau_patience,
                        plateau_ema_max=args.early_stop_blind_transition_plateau_ema_max,
                        rebound_tol=args.early_stop_blind_transition_rebound_tol,
                        hard_floor=args.early_stop_blind_transition_hard_floor,
                    )
                if not quiet_timing:
                    append_jsonl(
                        log_jsonl,
                        {
                            "tag": "early_stop_probe",
                            "epoch": epoch,
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            **probe_metrics,
                            **blind_transition_metrics,
                        },
                    )

                confirm_metrics = None
                two_stage_confirm_should_stop = False
                criteria_probe_metrics = probe_metrics
                if args.early_stop_two_stage and early_stop_armed:
                    two_stage_confirm_eval_index += 1
                    confirm_metrics = early_stop_probe_metrics(
                        model=model,
                        retain_loader=retain_probe_loader,
                        forget_loader=forget_probe_loader,
                        device=device,
                        tau=effective_tau,
                        forget_margin_rho=args.forget_margin_rho,
                        margin_teacher_topk=args.margin_teacher_topk,
                        use_teacher_competitor=(args.margin_competitor_source == "teacher"),
                        teacher_set_mode=args.margin_teacher_set_mode,
                        teacher_logit_delta=args.margin_teacher_logit_delta,
                        competitor_beta=args.margin_competitor_beta,
                        beta_parameterization=args.margin_beta_parameterization,
                        penalty_mode=args.forget_penalty_mode,
                        continuation_cut_min_ratio=args.continuation_cut_min_ratio,
                        continuation_cut_max_ratio=args.continuation_cut_max_ratio,
                        continuation_min_prefix_tokens=args.continuation_min_prefix_tokens,
                        continuation_min_suffix_tokens=args.continuation_min_suffix_tokens,
                        max_batches=None if holdout_online_val_enabled else two_stage_confirm_batches,
                        forget_violation_alpha=two_stage_confirm_alpha,
                        forget_margin_buffer_eta=args.forget_margin_buffer_eta,
                        retain_teacher_model=retain_teacher_model,
                        need_retain_mean=(
                            args.force_probe_loop
                            or holdout_online_val_enabled
                            or args.early_stop_retain_mean_margin_min is not None
                            or args.early_stop_retain_forget_margin_gap_min is not None
                            or args.early_stop_retain_forget_continuation_margin_gap_min is not None
                        ),
                        need_forget_mean=(
                            args.force_probe_loop
                            or holdout_online_val_enabled
                            or args.early_stop_retain_forget_margin_gap_min is not None
                        ),
                        need_forget_cont_mean=(
                            args.force_probe_loop
                            or args.early_stop_retain_forget_continuation_margin_gap_min is not None
                        ),
                        need_forget_cont_frac_gt_tau=True,
                        need_retain_probe_kl=(
                            (not args.early_stop_disable_retain_guard)
                            and (retain_teacher_model is not None)
                        ),
                        need_surrogate_mean=probe_surrogate_mean_enabled,
                        need_gap_stats=(
                            args.force_probe_loop
                            or holdout_online_val_enabled
                            or args.early_stop_retain_forget_margin_gap_min is not None
                        ),
                        need_cont_gap_stats=(
                            args.force_probe_loop
                            or args.early_stop_retain_forget_continuation_margin_gap_min is not None
                        ),
                        deterministic_continuation=deterministic_probe_enabled,
                        deterministic_salt=args.early_stop_probe_deterministic_salt,
                    )
                    confirm_metrics.update(
                        {
                            "forget_margin_tau_effective": effective_tau,
                            "forget_margin_rho": args.forget_margin_rho,
                            "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
                            "margin_teacher_topk": args.margin_teacher_topk,
                            "margin_teacher_set_mode": args.margin_teacher_set_mode,
                            "margin_teacher_logit_delta": args.margin_teacher_logit_delta,
                            "margin_competitor_beta": args.margin_competitor_beta,
                            "margin_beta_parameterization": args.margin_beta_parameterization,
                            "margin_competitor_source": args.margin_competitor_source,
                            "early_stop_two_stage": True,
                            "early_stop_confirm_num_forget_examples": args.early_stop_confirm_num_forget_examples,
                            "early_stop_confirm_batches": two_stage_confirm_batches,
                            "early_stop_confirm_use_ucb": args.early_stop_confirm_use_ucb,
                            "early_stop_confirm_alpha": two_stage_confirm_alpha,
                            "early_stop_confirm_delta": args.early_stop_confirm_delta,
                            "early_stop_confirmation_eval_index": two_stage_confirm_eval_index,
                        }
                    )
                    confirm_retain_ok = (
                        True
                        if args.early_stop_disable_retain_guard
                        else (
                            confirm_metrics["retain_probe_kl"]
                            <= retain_probe_kl_initial + args.early_stop_retain_kl_tolerance
                        )
                    )
                    confirm_violation_mean = confirm_metrics["forget_continuation_per_example_violation_mean"]
                    two_stage_confirm_should_stop, confirm_ucb, confirm_delta_j = (
                        two_stage_confirmation_decision(
                            violation_mean=confirm_violation_mean,
                            n=confirm_metrics["forget_continuation_per_example_count"],
                            alpha=two_stage_confirm_alpha,
                            retain_ok=confirm_retain_ok,
                            use_ucb=args.early_stop_confirm_use_ucb,
                            delta=args.early_stop_confirm_delta,
                            eval_index=two_stage_confirm_eval_index,
                        )
                    )
                    confirm_metrics.update(
                        {
                            "early_stop_retain_probe_kl": confirm_metrics["retain_probe_kl"],
                            "early_stop_retain_probe_kl_initial": retain_probe_kl_initial,
                            "early_stop_retain_ok": bool(confirm_retain_ok),
                            "early_stop_disable_retain_guard": bool(args.early_stop_disable_retain_guard),
                            "early_stop_confirm_violation_mean": confirm_violation_mean,
                            "early_stop_confirm_ucb": confirm_ucb,
                            "early_stop_confirm_delta_j": confirm_delta_j,
                            "early_stop_confirm_forget_ok": bool(
                                confirm_ucb <= two_stage_confirm_alpha
                                if args.early_stop_confirm_use_ucb
                                else confirm_violation_mean <= two_stage_confirm_alpha
                            ),
                            "early_stop_confirm_should_stop": bool(two_stage_confirm_should_stop),
                        }
                    )
                    criteria_probe_metrics = confirm_metrics
                    if not quiet_timing:
                        append_jsonl(
                            log_jsonl,
                            {
                                "tag": "early_stop_confirm_probe",
                                "epoch": epoch,
                                "global_step": global_step,
                                "optimizer_step": optimizer_step,
                                **confirm_metrics,
                            },
                        )
                    if not two_stage_confirm_should_stop:
                        early_stop_armed = False

                if args.online_fq_probe == "yes" and optimizer_step >= args.online_fq_probe_start_step:
                    online_fq_metrics = online_tofu_fq_probe_metrics(
                        model=model,
                        paraphrase_loader=online_fq_paraphrase_loader,
                        perturb_loader=online_fq_perturb_loader,
                        reference_eval_log=online_fq_reference_eval,
                    )
                    online_fq_row = {
                        "tag": "online_fq_probe",
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "split": args.online_fq_probe_split,
                        "ds_size": online_fq_num_examples,
                        "reference_log": args.online_fq_reference_log,
                        "forget_quality": online_fq_metrics["forget_quality"],
                        "ks_test_pvalue": online_fq_metrics["ks_test_pvalue"],
                        "ks_test_statistic": online_fq_metrics["ks_test_statistic"],
                        "unlearn_truth_ratio_mean": online_fq_metrics["unlearn_truth_ratio_mean"],
                        "unlearn_truth_ratio_median": online_fq_metrics["unlearn_truth_ratio_median"],
                        "unlearn_truth_ratio_p90": online_fq_metrics["unlearn_truth_ratio_p90"],
                        "unlearn_truth_ratio_p95": online_fq_metrics["unlearn_truth_ratio_p95"],
                        "reference_truth_ratio_mean": online_fq_metrics["reference_truth_ratio_mean"],
                        "reference_truth_ratio_median": online_fq_metrics["reference_truth_ratio_median"],
                        "reference_truth_ratio_p90": online_fq_metrics["reference_truth_ratio_p90"],
                        "reference_truth_ratio_p95": online_fq_metrics["reference_truth_ratio_p95"],
                        "num_examples": online_fq_metrics["num_examples"],
                    }
                    append_jsonl(online_fq_jsonl, online_fq_row)
                    if not quiet_timing:
                        print(json.dumps(online_fq_row, indent=2))

                if (
                    args.online_rouge_probe == "yes"
                    and optimizer_step >= args.online_rouge_probe_start_step
                    and optimizer_step % args.online_rouge_probe_every_steps == 0
                ):
                    online_rouge_metrics = online_tofu_rouge_probe_metrics(
                        model=model,
                        tokenizer=tokenizer,
                        retain_examples=online_rouge_retain_examples,
                        forget_examples=online_rouge_forget_examples,
                        batch_size=args.online_rouge_probe_batch_size,
                        max_input_length=args.online_rouge_probe_max_input_length,
                        max_new_tokens=args.online_rouge_probe_max_new_tokens,
                    )
                    online_rouge_row = flatten_online_rouge_probe_row(
                        online_rouge_metrics,
                        epoch=epoch,
                        global_step=global_step,
                        optimizer_step=optimizer_step,
                        retain_examples=len(online_rouge_retain_examples),
                        forget_examples=len(online_rouge_forget_examples),
                        max_new_tokens=args.online_rouge_probe_max_new_tokens,
                    )
                    append_jsonl(online_rouge_jsonl, online_rouge_row)
                    if not quiet_timing:
                        print(json.dumps(online_rouge_row, indent=2))

                criteria_met = stop_criteria_enabled
                if holdout_online_val_enabled:
                    gap_metric_key = (
                        "retain_forget_margin_gap_lower_bound"
                        if holdout_lower_bound_enabled
                        else "retain_forget_margin_gap"
                    )
                    criteria_met = criteria_met and (
                        criteria_probe_metrics[gap_metric_key] > args.holdout_online_val_threshold
                    )
                if stage1_trigger_only_enabled:
                    criteria_met = criteria_met and bool(probe_metrics.get("early_stop_trigger_pass"))
                elif args.early_stop_two_stage:
                    criteria_met = criteria_met and bool(confirm_metrics is not None) and bool(two_stage_confirm_should_stop)
                elif args.early_stop_use_ucb:
                    criteria_met = criteria_met and should_stop_with_ucb(
                        probe_metrics["early_stop_forget_ucb"],
                        args.early_stop_forget_cont_frac_margin_gt_tau_max,
                        probe_metrics["early_stop_retain_ok"],
                    )
                elif args.early_stop_forget_cont_frac_margin_gt_tau_max is not None:
                    criteria_met = criteria_met and (
                        criteria_probe_metrics["forget_continuation_frac_margin_gt_tau"]
                        < args.early_stop_forget_cont_frac_margin_gt_tau_max
                    )
                if args.early_stop_retain_mean_margin_min is not None:
                    criteria_met = criteria_met and (
                        criteria_probe_metrics["retain_mean_margin"] > args.early_stop_retain_mean_margin_min
                    )
                if args.early_stop_retain_forget_margin_gap_min is not None:
                    criteria_met = criteria_met and (
                        criteria_probe_metrics["retain_forget_margin_gap"]
                        > args.early_stop_retain_forget_margin_gap_min
                    )
                if args.early_stop_retain_forget_continuation_margin_gap_min is not None:
                    criteria_met = criteria_met and (
                        criteria_probe_metrics["retain_forget_continuation_margin_gap"]
                        > args.early_stop_retain_forget_continuation_margin_gap_min
                    )
                if blind_transition_enabled:
                    criteria_met = criteria_met and blind_transition_metrics["blind_transition_should_stop"]

                if criteria_met and not early_stop_triggered:
                    if args.timing_mode and timing_started and timing_end_perf is None:
                        timing_end_perf = time.perf_counter()
                        timing_end_wall = datetime.now(timezone.utc)
                    early_stop_dir = out_dir / f"checkpoint-earlystop-step{global_step}"
                    skip_early_stop_save = (args.timing_mode and args.timing_skip_save) or args.no_model_save
                    if skip_early_stop_save:
                        saved_format = "skipped_no_model_save" if args.no_model_save else "skipped_for_timing"
                    else:
                        model, saved_format = save_model_checkpoint(
                            model,
                            tokenizer,
                            early_stop_dir,
                            merge_lora=args.use_lora,
                        )
                    early_stop_payload = {
                        "tag": "early_stop",
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "checkpoint_dir": None if skip_early_stop_save else str(early_stop_dir),
                        "saved_format": saved_format,
                        "continue_after_steps": args.early_stop_continue_after_steps,
                        "training_objective": {
                            "objective_mode": args.objective_mode,
                            "lambda_forget_margin": args.lambda_forget_margin,
                            "lambda_retain": args.lambda_retain,
                            "retain_loss_mode": args.retain_loss_mode,
                            "margin_tau": args.margin_tau,
                            "forget_margin_tau_effective": effective_tau,
                            "forget_margin_rho": args.forget_margin_rho,
                            "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
                            "margin_teacher_topk": args.margin_teacher_topk,
                            "forget_margin_loss_mode": args.forget_margin_loss_mode,
                            "forget_margin_sigmoid_beta": args.forget_margin_sigmoid_beta,
                            "forget_penalty_mode": args.forget_penalty_mode,
                        },
                        "criteria": {
                            "forget_continuation_frac_margin_gt_tau_max": args.early_stop_forget_cont_frac_margin_gt_tau_max,
                            "early_stop_use_ucb": args.early_stop_use_ucb,
                            "early_stop_delta": args.early_stop_delta,
                            "early_stop_two_stage": args.early_stop_two_stage,
                            "early_stop_trigger_metric": args.early_stop_trigger_metric,
                            "early_stop_trigger_threshold": args.early_stop_trigger_threshold,
                            "early_stop_confirm_num_forget_examples": args.early_stop_confirm_num_forget_examples,
                            "early_stop_confirm_use_ucb": args.early_stop_confirm_use_ucb,
                            "early_stop_confirm_alpha": two_stage_confirm_alpha,
                            "early_stop_confirm_delta": args.early_stop_confirm_delta,
                            "early_stop_retain_kl_tolerance": args.early_stop_retain_kl_tolerance,
                            "retain_mean_margin_min": args.early_stop_retain_mean_margin_min,
                            "retain_forget_margin_gap_min": args.early_stop_retain_forget_margin_gap_min,
                            "retain_forget_continuation_margin_gap_min": args.early_stop_retain_forget_continuation_margin_gap_min,
                            "blind_transition": args.early_stop_blind_transition,
                            "blind_transition_ema_alpha": args.early_stop_blind_transition_ema_alpha,
                            "blind_transition_enter_max": args.early_stop_blind_transition_enter_max,
                            "blind_transition_min_drop_from_start": args.early_stop_blind_transition_min_drop_from_start,
                            "blind_transition_steep_slope_min": args.early_stop_blind_transition_steep_slope_min,
                            "blind_transition_flat_slope_max": args.early_stop_blind_transition_flat_slope_max,
                            "blind_transition_plateau_patience": args.early_stop_blind_transition_plateau_patience,
                            "blind_transition_plateau_ema_max": args.early_stop_blind_transition_plateau_ema_max,
                            "blind_transition_rebound_tol": args.early_stop_blind_transition_rebound_tol,
                            "blind_transition_hard_floor": args.early_stop_blind_transition_hard_floor,
                            "probe_deterministic": deterministic_probe_enabled,
                            "probe_deterministic_salt": args.early_stop_probe_deterministic_salt,
                            "eval_every_steps": args.early_stop_eval_every_steps,
                            "eval_batches": args.early_stop_eval_batches,
                            "holdout_online_val": args.holdout_online_val,
                            "holdout_online_val_windows_per_split": args.holdout_online_val_windows_per_split,
                            "holdout_online_val_threshold": args.holdout_online_val_threshold,
                            "holdout_online_val_lower_bound": args.holdout_online_val_lower_bound,
                        },
                        "observed_metrics": {
                            **criteria_probe_metrics,
                            **blind_transition_metrics,
                            "early_stop_small_trigger_value": probe_metrics.get("early_stop_trigger_value"),
                            "early_stop_small_trigger_pass": probe_metrics.get("early_stop_trigger_pass"),
                            "early_stop_small_retain_ok": probe_metrics.get("early_stop_retain_ok"),
                        },
                        "holdout_online_val_info": holdout_info,
                        "full_model_eval_mode": "lora_active_eval_forward" if args.use_lora else "full_finetune_eval_forward",
                    }
                    save_json(out_dir / "early_stop.json", early_stop_payload)
                    if quiet_timing:
                        print(f"step {global_step} early_stop", flush=True)
                    else:
                        print(json.dumps(early_stop_payload, indent=2))
                        if skip_early_stop_save:
                            print(f"Early stop triggered at step {global_step}; checkpoint saving skipped.")
                        else:
                            print(
                                f"Early stop triggered at step {global_step}; "
                                f"saved checkpoint to {early_stop_dir}"
                            )
                    if args.timing_mode and timing_started and timing_end_perf is not None:
                        timing_payload = {
                            "method": out_dir.name,
                            "corpus": args.corpus,
                            "seed": args.seed,
                            "duration_seconds": timing_end_perf - timing_start_perf,
                            "step1_to_stop_seconds": timing_end_perf - timing_start_perf,
                            "start_utc": timing_start_wall.isoformat() if timing_start_wall else None,
                            "step1_wall_time_utc": timing_start_wall.isoformat() if timing_start_wall else None,
                            "end_utc": timing_end_wall.isoformat() if timing_end_wall else None,
                            "train_end_wall_time_utc": timing_end_wall.isoformat() if timing_end_wall else None,
                            "epochs_configured": args.epochs,
                            "stopped_early": True,
                            "stop_epoch": epoch,
                            "stop_global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "batch_size": args.batch_size,
                            "gradient_accumulation_steps": args.gradient_accumulation_steps,
                            "lr": args.lr,
                            "lambda_forget_margin": args.lambda_forget_margin,
                            "margin_tau": args.margin_tau,
                            "forget_margin_tau_effective": effective_tau,
                            "forget_margin_rho": args.forget_margin_rho,
                            "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
                            "margin_teacher_topk": args.margin_teacher_topk,
                            "forget_margin_loss_mode": args.forget_margin_loss_mode,
                            "forget_margin_sigmoid_beta": args.forget_margin_sigmoid_beta,
                            "forget_penalty_mode": args.forget_penalty_mode,
                            "use_lora": bool(args.use_lora),
                            "objective_mode": args.objective_mode,
                            "early_stop_forget_cont_frac_margin_gt_tau_max": args.early_stop_forget_cont_frac_margin_gt_tau_max,
                            "early_stop_use_ucb": args.early_stop_use_ucb,
                            "early_stop_delta": args.early_stop_delta,
                            "early_stop_two_stage": args.early_stop_two_stage,
                            "early_stop_trigger_metric": args.early_stop_trigger_metric,
                            "early_stop_trigger_threshold": args.early_stop_trigger_threshold,
                            "early_stop_confirm_num_forget_examples": args.early_stop_confirm_num_forget_examples,
                            "early_stop_confirm_use_ucb": args.early_stop_confirm_use_ucb,
                            "early_stop_confirm_alpha": two_stage_confirm_alpha,
                            "early_stop_confirm_delta": args.early_stop_confirm_delta,
                            "early_stop_retain_kl_tolerance": args.early_stop_retain_kl_tolerance,
                            "early_stop_retain_mean_margin_min": args.early_stop_retain_mean_margin_min,
                            "early_stop_retain_forget_margin_gap_min": args.early_stop_retain_forget_margin_gap_min,
                            "early_stop_retain_forget_continuation_margin_gap_min": args.early_stop_retain_forget_continuation_margin_gap_min,
                            "early_stop_blind_transition": args.early_stop_blind_transition,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        }
                        save_json(timing_json, timing_payload)
                        if args.common_timing_json:
                            update_common_json(Path(args.common_timing_json), out_dir.name, timing_payload)
                    early_stop_triggered = True
                    if args.early_stop_continue_after_steps > 0:
                        early_stop_continue_until_step = optimizer_step + args.early_stop_continue_after_steps
                        if not quiet_timing:
                            print(
                                "Continuing diagnostic run until optimizer_step "
                                f"{early_stop_continue_until_step} "
                                f"({args.early_stop_continue_after_steps} steps after first stop)."
                            )
                    else:
                        training_should_stop = True
                        break

                if (
                    early_stop_continue_until_step is not None
                    and optimizer_step >= early_stop_continue_until_step
                ):
                    training_should_stop = True
                    if not quiet_timing:
                        print(
                            "Post-stop diagnostic window complete at "
                            f"optimizer_step {optimizer_step}."
                        )
                    break

            if quiet_timing:
                if global_step % args.logging_steps == 0:
                    print(f"step {global_step}", flush=True)
            else:
                running["retain_ce"] += metric_row["retain_ce"]
                running["forget_ce"] += metric_row["forget_ce"]
                running["forget_margin_loss_raw"] += metric_row["forget_margin_loss_raw"]
                running["forget_margin_loss_weighted"] += metric_row["forget_margin_loss_weighted"]
                running["lambda_forget_margin_effective"] += metric_row["lambda_forget_margin_effective"]
                running["forget_margin_warmup_frac"] += metric_row["forget_margin_warmup_frac"]
                running["forget_margin_hinge_loss"] += metric_row["forget_margin_hinge_loss"]
                running["forget_margin_centered_smooth_l1_loss"] += metric_row["forget_margin_centered_smooth_l1_loss"]
                running["forget_margin_sigmoid_weight_mean"] += metric_row["forget_margin_sigmoid_weight_mean"]
                running["asymmetry_gap_loss_raw"] += metric_row["asymmetry_gap_loss_raw"]
                running["asymmetry_gap_loss_weighted"] += metric_row["asymmetry_gap_loss_weighted"]
                running["joint_total_loss"] += metric_row["joint_total_loss"]
                running["retain_grad_norm"] += metric_row["retain_grad_norm"]
                running["secondary_grad_norm"] += metric_row["secondary_grad_norm"]
                running["secondary_grad_proj_norm"] += metric_row["secondary_grad_proj_norm"]
                running["safe_correction_norm"] += metric_row["safe_correction_norm"]
                running["safe_correction_ratio_realized"] += metric_row["safe_correction_ratio_realized"]
                running["retain_secondary_grad_cosine"] += metric_row["retain_secondary_grad_cosine"]
                running["retain_secondary_proj_cosine"] += metric_row["retain_secondary_proj_cosine"]
                running["forget_margin_loss"] += metric_row["forget_margin_loss"]
                running["total_loss"] += metric_row["total_loss"]
                running["retain_mean_margin"] += metric_row["retain_mean_margin"]
                running["forget_mean_margin"] += metric_row["forget_mean_margin"]
                running["retain_min_margin"] += metric_row["retain_min_margin"]
                running["forget_min_margin"] += metric_row["forget_min_margin"]
                running["retain_frac_margin_le_0"] += metric_row["retain_frac_margin_le_0"]
                running["forget_frac_margin_le_0"] += metric_row["forget_frac_margin_le_0"]
                running["retain_frac_margin_le_0p5"] += metric_row["retain_frac_margin_le_0p5"]
                running["forget_frac_margin_le_0p5"] += metric_row["forget_frac_margin_le_0p5"]
                running["forget_frac_margin_gt_tau"] += metric_row["forget_frac_margin_gt_tau"]
                running["forget_continuation_mean_margin"] += metric_row["forget_continuation_mean_margin"]
                running["forget_continuation_min_margin"] += metric_row["forget_continuation_min_margin"]
                running["forget_continuation_frac_margin_gt_tau"] += metric_row["forget_continuation_frac_margin_gt_tau"]
                running["forget_continuation_valid_tokens"] += metric_row["forget_continuation_valid_tokens"]
                running["retain_forget_continuation_margin_gap"] += metric_row["retain_forget_continuation_margin_gap"]
                running["forget_cut_ratio_mean"] += metric_row["forget_cut_ratio_mean"]
                running["forget_suffix_tokens_mean"] += metric_row["forget_suffix_tokens_mean"]
                running["retain_forget_margin_gap"] += metric_row["retain_forget_margin_gap"]
                running["retain_forget_min_margin_gap"] += metric_row["retain_forget_min_margin_gap"]
                running["retain_forget_frac_margin_le_0_gap"] += metric_row["retain_forget_frac_margin_le_0_gap"]
                running["retain_forget_frac_margin_le_0p5_gap"] += metric_row["retain_forget_frac_margin_le_0p5_gap"]
                running["retain_margin_reference"] += metric_row["retain_margin_reference"]
                running["retain_valid_tokens"] += metric_row["retain_valid_tokens"]
                running["forget_valid_tokens"] += metric_row["forget_valid_tokens"]
                running["grad_norm"] += metric_row["grad_norm"]
                running["retain_phase_steps"] += 1 if update_phase == "retain" else 0
                running["forget_phase_steps"] += 1 if update_phase == "forget" else 0
                running["count"] += 1

                if global_step % args.logging_steps == 0:
                    count = max(1, running["count"])
                    log_line = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "optimizer_step": optimizer_step,
                        "retain_ce": running["retain_ce"] / count,
                        "forget_ce": running["forget_ce"] / count,
                        "forget_margin_loss_raw": running["forget_margin_loss_raw"] / count,
                        "forget_margin_loss_weighted": running["forget_margin_loss_weighted"] / count,
                        "lambda_forget_margin_effective": running["lambda_forget_margin_effective"] / count,
                        "forget_margin_warmup_frac": running["forget_margin_warmup_frac"] / count,
                        "forget_margin_loss_mode": args.forget_margin_loss_mode,
                        "forget_margin_sigmoid_beta": args.forget_margin_sigmoid_beta,
                        "forget_margin_hinge_loss": running["forget_margin_hinge_loss"] / count,
                        "forget_margin_centered_smooth_l1_loss": running["forget_margin_centered_smooth_l1_loss"] / count,
                        "forget_margin_sigmoid_weight_mean": running["forget_margin_sigmoid_weight_mean"] / count,
                        "asymmetry_gap_loss_raw": running["asymmetry_gap_loss_raw"] / count,
                        "asymmetry_gap_loss_weighted": running["asymmetry_gap_loss_weighted"] / count,
                        "joint_total_loss": running["joint_total_loss"] / count,
                        "retain_grad_norm": running["retain_grad_norm"] / count,
                        "secondary_grad_norm": running["secondary_grad_norm"] / count,
                        "secondary_grad_proj_norm": running["secondary_grad_proj_norm"] / count,
                        "safe_correction_norm": running["safe_correction_norm"] / count,
                        "safe_correction_ratio_realized": running["safe_correction_ratio_realized"] / count,
                        "retain_secondary_grad_cosine": running["retain_secondary_grad_cosine"] / count,
                        "retain_secondary_proj_cosine": running["retain_secondary_proj_cosine"] / count,
                        "forget_margin_loss": running["forget_margin_loss"] / count,
                        "total_loss": running["total_loss"] / count,
                        "optimization_schedule": args.optimization_schedule,
                        "retain_phase_steps": running["retain_phase_steps"],
                        "forget_phase_steps": running["forget_phase_steps"],
                        "retain_mean_margin": running["retain_mean_margin"] / count,
                        "forget_mean_margin": running["forget_mean_margin"] / count,
                        "retain_min_margin": running["retain_min_margin"] / count,
                        "forget_min_margin": running["forget_min_margin"] / count,
                        "retain_frac_margin_le_0": running["retain_frac_margin_le_0"] / count,
                        "forget_frac_margin_le_0": running["forget_frac_margin_le_0"] / count,
                        "retain_frac_margin_le_0p5": running["retain_frac_margin_le_0p5"] / count,
                        "forget_frac_margin_le_0p5": running["forget_frac_margin_le_0p5"] / count,
                        "forget_frac_margin_gt_tau": running["forget_frac_margin_gt_tau"] / count,
                        "forget_continuation_mean_margin": running["forget_continuation_mean_margin"] / count,
                        "forget_continuation_min_margin": running["forget_continuation_min_margin"] / count,
                        "forget_continuation_frac_margin_gt_tau": running["forget_continuation_frac_margin_gt_tau"] / count,
                        "forget_continuation_valid_tokens": running["forget_continuation_valid_tokens"] / count,
                        "retain_forget_continuation_margin_gap": running["retain_forget_continuation_margin_gap"] / count,
                        "forget_cut_ratio_mean": running["forget_cut_ratio_mean"] / count,
                        "forget_suffix_tokens_mean": running["forget_suffix_tokens_mean"] / count,
                        "retain_forget_margin_gap": running["retain_forget_margin_gap"] / count,
                        "retain_forget_min_margin_gap": running["retain_forget_min_margin_gap"] / count,
                        "retain_forget_frac_margin_le_0_gap": running["retain_forget_frac_margin_le_0_gap"] / count,
                        "retain_forget_frac_margin_le_0p5_gap": running["retain_forget_frac_margin_le_0p5_gap"] / count,
                        "retain_margin_reference": running["retain_margin_reference"] / count,
                        "retain_valid_tokens": running["retain_valid_tokens"] / count,
                        "forget_valid_tokens": running["forget_valid_tokens"] / count,
                        "grad_norm": running["grad_norm"] / count,
                    }
                    print(json.dumps(log_line, indent=2))
                for key in running:
                    running[key] = 0.0 if key != "count" else 0

        if training_should_stop:
            break

        if args.save_every_epoch and not args.no_model_save:
            ckpt_dir = out_dir / f"checkpoint-epoch{epoch}"
            _, _ = save_model_checkpoint(
                model,
                tokenizer,
                ckpt_dir,
                merge_lora=False,
            )

        epoch_retain_eval = evaluate_window_loader(
            model,
            retain_eval_loader,
            device,
            teacher_model=retain_teacher_model,
            teacher_topk=args.margin_teacher_topk,
            use_teacher_competitor=(args.margin_competitor_source == "teacher"),
            teacher_set_mode=args.margin_teacher_set_mode,
            teacher_logit_delta=args.margin_teacher_logit_delta,
            competitor_beta=args.margin_competitor_beta,
            beta_parameterization=args.margin_beta_parameterization,
        )
        epoch_forget_eval = evaluate_window_loader(
            model,
            forget_eval_loader,
            device,
            teacher_model=retain_teacher_model,
            teacher_topk=args.margin_teacher_topk,
            use_teacher_competitor=(args.margin_competitor_source == "teacher"),
            teacher_set_mode=args.margin_teacher_set_mode,
            teacher_logit_delta=args.margin_teacher_logit_delta,
            competitor_beta=args.margin_competitor_beta,
            beta_parameterization=args.margin_beta_parameterization,
        )
        epoch_eval_payload = eval_payload(epoch_retain_eval, epoch_forget_eval)
        epoch_eval_payload["tag"] = "epoch_end"
        epoch_eval_payload["epoch"] = epoch
        epoch_eval_payload["optimizer_step"] = optimizer_step
        epoch_eval_payload["delta_from_start"] = delta_payload(epoch_eval_payload, baseline_eval_payload)
        append_jsonl(eval_history_jsonl, epoch_eval_payload)
        if not quiet_timing:
            print(json.dumps(epoch_eval_payload, indent=2))
        model.train()

    if global_step % args.gradient_accumulation_steps != 0:
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1

    if args.timing_mode and timing_started and timing_end_perf is None:
        timing_end_perf = time.perf_counter()
        timing_end_wall = datetime.now(timezone.utc)
        timing_payload = {
            "method": out_dir.name,
            "corpus": args.corpus,
            "seed": args.seed,
            "duration_seconds": timing_end_perf - timing_start_perf,
            "step1_to_stop_seconds": timing_end_perf - timing_start_perf,
            "start_utc": timing_start_wall.isoformat() if timing_start_wall else None,
            "step1_wall_time_utc": timing_start_wall.isoformat() if timing_start_wall else None,
            "end_utc": timing_end_wall.isoformat() if timing_end_wall else None,
            "train_end_wall_time_utc": timing_end_wall.isoformat() if timing_end_wall else None,
            "epochs_configured": args.epochs,
            "stopped_early": False,
            "stop_epoch": epoch,
            "stop_global_step": global_step,
            "optimizer_step": optimizer_step,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "lr": args.lr,
            "lambda_forget_margin": args.lambda_forget_margin,
            "margin_tau": args.margin_tau,
            "margin_teacher_topk": args.margin_teacher_topk,
            "margin_teacher_set_mode": args.margin_teacher_set_mode,
            "margin_teacher_logit_delta": args.margin_teacher_logit_delta,
            "margin_competitor_beta": args.margin_competitor_beta,
            "margin_beta_parameterization": args.margin_beta_parameterization,
            "margin_competitor_source": args.margin_competitor_source,
            "forget_margin_tau_effective": effective_tau,
            "forget_margin_rho": args.forget_margin_rho,
            "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
            "forget_margin_loss_mode": args.forget_margin_loss_mode,
            "forget_margin_sigmoid_beta": args.forget_margin_sigmoid_beta,
            "forget_penalty_mode": args.forget_penalty_mode,
            "use_lora": bool(args.use_lora),
            "objective_mode": args.objective_mode,
            "early_stop_forget_cont_frac_margin_gt_tau_max": args.early_stop_forget_cont_frac_margin_gt_tau_max,
            "early_stop_retain_mean_margin_min": args.early_stop_retain_mean_margin_min,
            "early_stop_retain_forget_margin_gap_min": args.early_stop_retain_forget_margin_gap_min,
            "early_stop_retain_forget_continuation_margin_gap_min": args.early_stop_retain_forget_continuation_margin_gap_min,
            "early_stop_blind_transition": args.early_stop_blind_transition,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(timing_json, timing_payload)
        if args.common_timing_json:
            update_common_json(Path(args.common_timing_json), out_dir.name, timing_payload)

    if not ((args.timing_mode and args.timing_skip_save) or args.no_model_save):
        final_dir = out_dir / "final_model"
        model, _ = save_model_checkpoint(
            model,
            tokenizer,
            final_dir,
            merge_lora=args.use_lora,
        )

    if args.timing_mode and args.timing_skip_save:
        del model
        if retain_teacher_model is not None:
            del retain_teacher_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    if args.skip_final_eval:
        del model
        if retain_teacher_model is not None:
            del retain_teacher_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return

    if not quiet_timing:
        print("\nEvaluating final model on raw windowed retain/forget data...")
    retain_eval = evaluate_window_loader(
        model,
        retain_eval_loader,
        device,
        teacher_model=retain_teacher_model,
        teacher_topk=args.margin_teacher_topk,
        use_teacher_competitor=(args.margin_competitor_source == "teacher"),
        teacher_set_mode=args.margin_teacher_set_mode,
        teacher_logit_delta=args.margin_teacher_logit_delta,
        competitor_beta=args.margin_competitor_beta,
        beta_parameterization=args.margin_beta_parameterization,
    )
    forget_eval = evaluate_window_loader(
        model,
        forget_eval_loader,
        device,
        teacher_model=retain_teacher_model,
        teacher_topk=args.margin_teacher_topk,
        use_teacher_competitor=(args.margin_competitor_source == "teacher"),
        teacher_set_mode=args.margin_teacher_set_mode,
        teacher_logit_delta=args.margin_teacher_logit_delta,
        competitor_beta=args.margin_competitor_beta,
        beta_parameterization=args.margin_beta_parameterization,
    )
    final_eval_payload = eval_payload(retain_eval, forget_eval)
    final_eval_payload["windowing"] = {
        "mode": "nonoverlap",
        "max_len": args.max_len,
        "retain_windows": len(retain_dataset),
        "forget_windows": len(forget_dataset),
        "train_retain_windows": len(retain_train_dataset),
        "train_forget_windows": len(forget_train_dataset),
        "holdout_retain_windows": len(retain_probe_dataset) if holdout_online_val_enabled else 0,
        "holdout_forget_windows": len(forget_probe_dataset) if holdout_online_val_enabled else 0,
    }
    final_eval_payload["training_objective"] = {
        "objective_mode": args.objective_mode,
        "lambda_forget_margin": args.lambda_forget_margin,
        "lambda_retain": args.lambda_retain,
        "retain_loss_mode": args.retain_loss_mode,
        "retain_teacher_model_dir": args.retain_teacher_model_dir,
        "lambda_gap": args.lambda_gap,
        "margin_gap_target": args.margin_gap_target,
        "retain_margin_reference_beta": args.retain_margin_reference_beta,
        "retain_margin_reference_mode": args.retain_margin_reference_mode,
        "retain_margin_reference_value": args.retain_margin_reference_value,
        "optimization_schedule": args.optimization_schedule,
        "retain_steps_per_forget_step": args.retain_steps_per_forget_step,
        "forget_step_lr_scale": args.forget_step_lr_scale,
        "safe_correction_ratio": args.safe_correction_ratio,
        "safe_projection_eps": args.safe_projection_eps,
        "margin_tau": args.margin_tau,
        "margin_teacher_topk": args.margin_teacher_topk,
        "margin_teacher_set_mode": args.margin_teacher_set_mode,
        "margin_teacher_logit_delta": args.margin_teacher_logit_delta,
        "margin_competitor_beta": args.margin_competitor_beta,
        "margin_beta_parameterization": args.margin_beta_parameterization,
        "margin_competitor_source": args.margin_competitor_source,
        "forget_margin_tau_effective": effective_tau,
        "forget_margin_rho": args.forget_margin_rho,
        "forget_margin_buffer_eta": args.forget_margin_buffer_eta,
        "forget_margin_loss_mode": args.forget_margin_loss_mode,
        "forget_margin_sigmoid_beta": args.forget_margin_sigmoid_beta,
        "forget_penalty_mode": args.forget_penalty_mode,
        "continuation_cut_min_ratio": args.continuation_cut_min_ratio,
        "continuation_cut_max_ratio": args.continuation_cut_max_ratio,
        "continuation_min_prefix_tokens": args.continuation_min_prefix_tokens,
        "continuation_min_suffix_tokens": args.continuation_min_suffix_tokens,
        "early_stop_eval_every_steps": args.early_stop_eval_every_steps,
        "early_stop_eval_batches": args.early_stop_eval_batches,
        "early_stop_forget_cont_frac_margin_gt_tau_max": args.early_stop_forget_cont_frac_margin_gt_tau_max,
        "early_stop_use_ucb": args.early_stop_use_ucb,
        "early_stop_delta": args.early_stop_delta,
        "early_stop_two_stage": args.early_stop_two_stage,
        "early_stop_trigger_metric": args.early_stop_trigger_metric,
        "early_stop_trigger_threshold": args.early_stop_trigger_threshold,
        "early_stop_confirm_num_forget_examples": args.early_stop_confirm_num_forget_examples,
        "early_stop_confirm_use_ucb": args.early_stop_confirm_use_ucb,
        "early_stop_confirm_alpha": two_stage_confirm_alpha,
        "early_stop_confirm_delta": args.early_stop_confirm_delta,
        "early_stop_retain_kl_tolerance": args.early_stop_retain_kl_tolerance,
        "early_stop_disable_retain_guard": args.early_stop_disable_retain_guard,
        "early_stop_retain_mean_margin_min": args.early_stop_retain_mean_margin_min,
        "early_stop_retain_forget_margin_gap_min": args.early_stop_retain_forget_margin_gap_min,
        "early_stop_retain_forget_continuation_margin_gap_min": args.early_stop_retain_forget_continuation_margin_gap_min,
        "early_stop_blind_transition": args.early_stop_blind_transition,
        "early_stop_blind_transition_ema_alpha": args.early_stop_blind_transition_ema_alpha,
        "early_stop_blind_transition_enter_max": args.early_stop_blind_transition_enter_max,
        "early_stop_blind_transition_min_drop_from_start": args.early_stop_blind_transition_min_drop_from_start,
        "early_stop_blind_transition_steep_slope_min": args.early_stop_blind_transition_steep_slope_min,
        "early_stop_blind_transition_flat_slope_max": args.early_stop_blind_transition_flat_slope_max,
        "early_stop_blind_transition_plateau_patience": args.early_stop_blind_transition_plateau_patience,
        "early_stop_blind_transition_plateau_ema_max": args.early_stop_blind_transition_plateau_ema_max,
        "early_stop_blind_transition_rebound_tol": args.early_stop_blind_transition_rebound_tol,
        "early_stop_blind_transition_hard_floor": args.early_stop_blind_transition_hard_floor,
        "early_stop_probe_deterministic": args.early_stop_probe_deterministic,
        "early_stop_probe_deterministic_salt": args.early_stop_probe_deterministic_salt,
        "holdout_online_val": args.holdout_online_val,
        "holdout_online_val_windows_per_split": args.holdout_online_val_windows_per_split,
        "holdout_online_val_threshold": args.holdout_online_val_threshold,
        "holdout_online_val_lower_bound": args.holdout_online_val_lower_bound,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": args.lora_target_modules,
        "lora_bias": args.lora_bias,
    }
    final_eval_payload["holdout_online_val_info"] = holdout_info
    final_eval_payload["delta_from_start"] = delta_payload(final_eval_payload, baseline_eval_payload)
    save_json(eval_json, final_eval_payload)
    if not quiet_timing:
        print(json.dumps(final_eval_payload, indent=2))

    del model
    if retain_teacher_model is not None:
        del retain_teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
