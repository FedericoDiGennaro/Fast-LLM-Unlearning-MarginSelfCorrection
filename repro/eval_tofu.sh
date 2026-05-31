#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
SEED=${SEED:-42}
DEFAULT_RUN_DIR="$ROOT/outputs/tofu_masc_seed${SEED}"

latest_checkpoint() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-earlystop-*' 2>/dev/null | sort -V | tail -n 1
}

MODEL_DIR=${MODEL_DIR:-$(latest_checkpoint "$DEFAULT_RUN_DIR")}
if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
  echo "Missing TOFU checkpoint. Set MODEL_DIR=... or run repro/run_masc_tofu.sh first." >&2
  exit 1
fi

TOKENIZER_DIR=${TOKENIZER_DIR:-"$MODEL_DIR"}
OUTPUT_DIR=${OUTPUT_DIR:-"$MODEL_DIR/evals"}
RETAIN_LOGS=${RETAIN_LOGS:-"$ROOT/open-unlearning/saves/eval/tofu_Llama-2-7b-chat-hf_retain90/TOFU_EVAL.json"}

mkdir -p "$OUTPUT_DIR"
export HF_HOME=${HF_HOME:-"$ROOT/cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$ROOT/cache/hub"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$ROOT/open-unlearning/cache/hf_datasets"}

cd "$ROOT/open-unlearning"
"$PYTHON" src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default \
  eval=tofu_no_privleak \
  model=Llama-2-7b-chat-hf \
  model.model_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path="$TOKENIZER_DIR" \
  '~model.model_args.attn_implementation' \
  task_name=tofu_masc_seed${SEED} \
  paths.output_dir="$OUTPUT_DIR" \
  forget_split=forget10 \
  holdout_split=holdout10 \
  retain_logs_path="$RETAIN_LOGS" \
  eval.tofu.overwrite=true
