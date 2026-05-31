#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-"$ROOT/outputs/tofu_masc_seed${SEED}"}
MODEL_DIR=${MODEL_DIR:-"$ROOT/open-unlearning/cache/hf/open-unlearning__tofu_Llama-2-7b-chat-hf_full"}
DATASET_CACHE=${DATASET_CACHE:-"$ROOT/open-unlearning/cache/hf_datasets"}
START_EVAL_CACHE=${START_EVAL_CACHE:-"$ROOT/open-unlearning/diagnostics/start_eval_cache/open-unlearning__tofu_Llama-2-7b-chat-hf_full__retain-retain90__forget-forget10__maxlen-512.json"}

first_snapshot() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -n 1
}

TOKENIZER_DIR=${TOKENIZER_DIR:-$(first_snapshot "$ROOT/cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots")}
if [[ -z "${TOKENIZER_DIR}" || ! -d "${TOKENIZER_DIR}" ]]; then
  echo "Missing Llama-2 tokenizer snapshot. Run: python repro/download_assets.py --target tofu" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$ROOT/logs" "$ROOT/cache"
export HF_HOME=${HF_HOME:-"$ROOT/cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$ROOT/cache/hub"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$DATASET_CACHE"}

"$PYTHON" "$ROOT/scripts/run_margin_unlearning.py" \
  --model_dir "$MODEL_DIR" \
  --retain_loss_mode ce \
  --lambda_retain 0.0 \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --cache_dir "$ROOT/cache" \
  --dataset_cache_dir "$DATASET_CACHE" \
  --out_dir "$OUT_DIR" \
  --corpus tofu \
  --forget_split forget10 \
  --retain_split retain90 \
  --start_eval_cache_json "$START_EVAL_CACHE" \
  --max_len 512 \
  --batch_size 2 \
  --epochs 3 \
  --lr 1e-4 \
  --objective_mode forget_margin \
  --lambda_forget_margin 0.05 \
  --forget_margin_rho 0.70 \
  --forget_margin_buffer_eta 0.25 \
  --margin_competitor_source self \
  --margin_teacher_topk 10 \
  --margin_competitor_beta 1.0 \
  --margin_beta_parameterization tempered_posterior \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --forget_penalty_mode continuation \
  --continuation_cut_min_ratio 0.2 \
  --continuation_cut_max_ratio 0.6 \
  --continuation_min_prefix_tokens 32 \
  --continuation_min_suffix_tokens 32 \
  --early_stop_two_stage \
  --early_stop_trigger_metric violation_mean \
  --early_stop_trigger_threshold 0.475 \
  --early_stop_confirm_num_forget_examples 32 \
  --early_stop_confirm_alpha 0.475 \
  --early_stop_disable_retain_guard \
  --early_stop_eval_every_steps 2 \
  --early_stop_eval_batches 2 \
  --seed "$SEED" \
  --device cuda \
  --gradient_checkpointing \
  --logging_steps 1 \
  --skip_final_eval
