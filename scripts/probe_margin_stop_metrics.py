#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_margin_unlearning import (
    WindowDataset,
    chunk_texts_nonoverlap,
    collate_batch,
    early_stop_probe_metrics,
    get_device,
    get_fp_dtype,
    load_split_texts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe our margin-based stop metrics on an arbitrary checkpoint.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--tokenizer_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--corpus", type=str, choices=["news"], default="news")
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--probe_batches", type=int, default=1)
    parser.add_argument("--margin_tau", type=float, default=1.0)
    parser.add_argument("--forget_penalty_mode", type=str, choices=["full", "continuation"], default="continuation")
    parser.add_argument("--continuation_cut_min_ratio", type=float, default=0.2)
    parser.add_argument("--continuation_cut_max_ratio", type=float, default=0.6)
    parser.add_argument("--continuation_min_prefix_tokens", type=int, default=32)
    parser.add_argument("--continuation_min_suffix_tokens", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--out_json", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    dtype = get_fp_dtype(device, args.precision)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    retain_texts = load_split_texts(args.cache_dir, args.corpus, "retain1", None)
    forget_texts = load_split_texts(args.cache_dir, args.corpus, "forget", None)
    retain_windows, retain_meta = chunk_texts_nonoverlap(tokenizer, retain_texts, args.max_len)
    forget_windows, forget_meta = chunk_texts_nonoverlap(tokenizer, forget_texts, args.max_len)

    retain_dataset = WindowDataset(retain_windows, retain_meta, pad_id)
    forget_dataset = WindowDataset(forget_windows, forget_meta, pad_id)
    retain_loader = DataLoader(retain_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    forget_loader = DataLoader(forget_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        local_files_only=True,
    )
    model.to(device)

    metrics = early_stop_probe_metrics(
        model=model,
        retain_loader=retain_loader,
        forget_loader=forget_loader,
        device=device,
        tau=args.margin_tau,
        penalty_mode=args.forget_penalty_mode,
        continuation_cut_min_ratio=args.continuation_cut_min_ratio,
        continuation_cut_max_ratio=args.continuation_cut_max_ratio,
        continuation_min_prefix_tokens=args.continuation_min_prefix_tokens,
        continuation_min_suffix_tokens=args.continuation_min_suffix_tokens,
        max_batches=None if args.probe_batches <= 0 else args.probe_batches,
        need_retain_mean=True,
        need_forget_mean=True,
        need_forget_cont_mean=True,
        need_forget_cont_frac_gt_tau=True,
        need_gap_stats=True,
        need_cont_gap_stats=True,
    )

    payload = {
        "model_dir": args.model_dir,
        "tokenizer_dir": args.tokenizer_dir,
        "corpus": args.corpus,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "probe_batches": args.probe_batches,
        "margin_tau": args.margin_tau,
        "forget_penalty_mode": args.forget_penalty_mode,
        "metrics": metrics,
    }

    text = json.dumps(payload, indent=2) + "\n"
    print(text, end="")
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)


if __name__ == "__main__":
    main()
