#!/usr/bin/env bash
set -euo pipefail

ROOT="."
OU="$ROOT/open-unlearning"
PY="$ROOT/venv_open_unlearning/bin/python"
LOG_DIR="$ROOT/exp_scaling/logs"
EVAL_ROOT="$ROOT/exp_scaling/saves/eval/qwen25"
UNLEARN_ROOT="$ROOT/exp_scaling/saves/unlearn/qwen25"
RUN_NAME="tofu_margin_selftopk_besthp_qwen_template_seed42_vs_retain90"
UNLEARN_SUFFIX="tofu_margin_unlearning_forget10_retain90_cont_lora_lambda0p1_rho0p5_topk10_self_beta1_noretainguard_selftopk_besthp_qwen_template_seed42"

mkdir -p "$LOG_DIR"

declare -a MODEL_KEYS=(
  "qwen25_0p5b_instruct"
  "qwen25_1p5b_instruct"
  "qwen25_3b_instruct"
  "qwen25_7b_instruct"
)

declare -A OU_MODEL=(
  ["qwen25_0p5b_instruct"]="Qwen2.5-0.5B-Instruct"
  ["qwen25_1p5b_instruct"]="Qwen2.5-1.5B-Instruct"
  ["qwen25_3b_instruct"]="Qwen2.5-3B-Instruct"
  ["qwen25_7b_instruct"]="Qwen2.5-7B-Instruct"
)

cd "$OU"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$OU/cache/hf_home"
export HF_DATASETS_CACHE="$OU/cache/hf_datasets"
export TRITON_CACHE_DIR=/tmp/open_unlearning_triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/open_unlearning_torchinductor

MASTER_LOG="$LOG_DIR/eval_qwen25_family_${RUN_NAME}.log"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting corrected Qwen2.5 family TOFU evals" | tee "$MASTER_LOG"

for MODEL_KEY in "${MODEL_KEYS[@]}"; do
  MODEL_DIR="$UNLEARN_ROOT/$MODEL_KEY/$UNLEARN_SUFFIX/final_model"
  REFERENCE_LOGS="$EVAL_ROOT/$MODEL_KEY/tofu_retain90_vs_full/TOFU_EVAL.json"
  OUT_DIR="$EVAL_ROOT/$MODEL_KEY/$RUN_NAME"
  LOG="$LOG_DIR/eval_${MODEL_KEY}_${RUN_NAME}.log"

  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Missing model dir: $MODEL_DIR" | tee -a "$MASTER_LOG"
    exit 1
  fi
  if [[ ! -f "$REFERENCE_LOGS" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Missing retain reference logs: $REFERENCE_LOGS" | tee -a "$MASTER_LOG"
    exit 1
  fi

  mkdir -p "$OUT_DIR"
  : > "$LOG"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Evaluating $MODEL_KEY -> $OUT_DIR" | tee -a "$MASTER_LOG" "$LOG"

  "$PY" src/eval.py \
    experiment=eval/tofu/default.yaml \
    eval=tofu_no_privleak \
    task_name="scaling_${MODEL_KEY}_${RUN_NAME}" \
    model="${OU_MODEL[$MODEL_KEY]}" \
    forget_split=forget10 \
    holdout_split=holdout10 \
    retain_logs_path="$REFERENCE_LOGS" \
    '~model.model_args.attn_implementation' \
    model.model_args.pretrained_model_name_or_path="$MODEL_DIR" \
    model.tokenizer_args.pretrained_model_name_or_path="$MODEL_DIR" \
    paths.output_dir="$OUT_DIR" \
    eval.tofu.overwrite=true \
    2>&1 | tee -a "$LOG"

  if [[ ! -f "$OUT_DIR/TOFU_SUMMARY.json" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Missing TOFU summary after eval: $OUT_DIR" | tee -a "$MASTER_LOG"
    exit 1
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished $MODEL_KEY" | tee -a "$MASTER_LOG"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished corrected Qwen2.5 family TOFU evals" | tee -a "$MASTER_LOG"
