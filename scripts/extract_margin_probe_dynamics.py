#!/usr/bin/env python3
"""Extract probe-only dynamics from margin-unlearning train_metrics.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROBE_TAGS = {"early_stop_probe", "early_stop_confirm_probe"}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to train_metrics.jsonl")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument(
        "--metric",
        default="forget_continuation_per_example_violation_mean",
        help="Probe metric used for drift diagnostics.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("tag") in PROBE_TAGS:
                rows.append(payload)

    rows.sort(key=lambda row: (int(row.get("optimizer_step", -1)), row.get("tag", "")))

    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    columns = [
        "tag",
        "epoch",
        "global_step",
        "optimizer_step",
        "forget_continuation_per_example_violation_mean",
        "forget_continuation_surrogate_mean",
        "forget_continuation_frac_margin_gt_tau",
        "forget_continuation_token_frac_margin_gt_tau",
        "forget_continuation_per_example_count",
        "early_stop_trigger_value",
        "early_stop_trigger_pass",
        "early_stop_confirm_violation_mean",
        "early_stop_confirm_should_stop",
        "early_stop_confirm_alpha",
        "early_stop_confirm_num_forget_examples",
        "forget_cont_exact_continuation_bound",
        "forget_cont_log_exact_continuation_bound",
    ]
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})

    metric_rows = [
        row
        for row in rows
        if row.get("tag") == "early_stop_probe" and as_float(row.get(args.metric)) is not None
    ]
    values = [as_float(row[args.metric]) for row in metric_rows]
    deltas = [
        values[idx] - values[idx - 1]
        for idx in range(1, len(values))
        if values[idx] is not None and values[idx - 1] is not None
    ]
    first_stop_rows = [
        row
        for row in rows
        if row.get("tag") == "early_stop_confirm_probe"
        and bool(row.get("early_stop_confirm_should_stop"))
    ]
    summary = {
        "input": str(input_path),
        "num_probe_rows": len(rows),
        "num_small_probe_rows": len(metric_rows),
        "metric": args.metric,
        "first_optimizer_step": metric_rows[0].get("optimizer_step") if metric_rows else None,
        "last_optimizer_step": metric_rows[-1].get("optimizer_step") if metric_rows else None,
        "first_value": values[0] if values else None,
        "last_value": values[-1] if values else None,
        "mean_delta": sum(deltas) / len(deltas) if deltas else None,
        "nonpositive_delta_fraction": (
            sum(1 for delta in deltas if delta <= 0.0) / len(deltas) if deltas else None
        ),
        "first_confirm_stop_optimizer_step": (
            first_stop_rows[0].get("optimizer_step") if first_stop_rows else None
        ),
    }
    Path(args.summary_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
