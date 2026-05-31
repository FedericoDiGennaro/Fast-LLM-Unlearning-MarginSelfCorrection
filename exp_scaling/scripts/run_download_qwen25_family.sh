#!/usr/bin/env bash
set -euo pipefail

ROOT="."
VENV="$ROOT/venv_open_unlearning"
MANIFEST="$ROOT/exp_scaling/manifests/qwen25_tofu_scaling.json"
LOG_DIR="$ROOT/exp_scaling/logs"
LOG_FILE="$LOG_DIR/download_qwen25_family.log"

mkdir -p "$LOG_DIR"

source "$VENV/bin/activate"
cd "$ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Downloading Qwen2.5 scaling family from Hugging Face" | tee "$LOG_FILE"
python exp_scaling/scripts/download_family_models.py \
  --manifest "$MANIFEST" \
  2>&1 | tee -a "$LOG_FILE"
