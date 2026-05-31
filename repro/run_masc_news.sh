#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
SEED=${SEED:-42}
OUT_DIR=${OUT_DIR:-"$ROOT/outputs/news_masc_seed${SEED}"}
START_EVAL_CACHE=${START_EVAL_CACHE:-"$ROOT/outputs/cache/news_target_start_eval_raw_windows.json"}

first_snapshot() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -n 1
}

MODEL_DIR=${MODEL_DIR:-$(first_snapshot "$ROOT/cache/hub/models--muse-bench--MUSE-news_target/snapshots")}
if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
  echo "Missing MUSE News target model. Run: python repro/download_assets.py --target news" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$(dirname "$START_EVAL_CACHE")" "$ROOT/logs" "$ROOT/cache"
export HF_HOME=${HF_HOME:-"$ROOT/cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$ROOT/cache/hub"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$ROOT/cache/datasets"}

"$PYTHON" "$ROOT/scripts/run_margin_unlearning.py" \
  --model_dir "$MODEL_DIR" \
  --retain_teacher_model_dir "$MODEL_DIR" \
  --retain_loss_mode kl_target \
  --lambda_retain 1.0 \
  --cache_dir "$ROOT/cache" \
  --out_dir "$OUT_DIR" \
  --corpus news \
  --start_eval_cache_json "$START_EVAL_CACHE" \
  --max_len 1024 \
  --batch_size 2 \
  --epochs 3 \
  --lr 1e-4 \
  --objective_mode forget_margin \
  --lambda_forget_margin 0.5 \
  --forget_margin_rho 0.7 \
  --forget_margin_buffer_eta 0.5 \
  --margin_competitor_source teacher \
  --margin_teacher_topk 2 \
  --margin_teacher_set_mode delta \
  --margin_teacher_logit_delta 0.5 \
  --margin_competitor_beta 5.0 \
  --margin_beta_parameterization tempered_posterior \
  --forget_penalty_mode continuation \
  --continuation_cut_min_ratio 0.6 \
  --continuation_cut_max_ratio 0.6 \
  --continuation_min_prefix_tokens 32 \
  --continuation_min_suffix_tokens 128 \
  --early_stop_two_stage \
  --early_stop_trigger_metric violation_mean \
  --early_stop_trigger_threshold 0.55 \
  --early_stop_confirm_num_forget_examples 64 \
  --early_stop_confirm_alpha 0.55 \
  --early_stop_disable_retain_guard \
  --early_stop_eval_every_steps 2 \
  --early_stop_eval_batches 2 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --lora_bias none \
  --precision auto \
  --device cuda \
  --gradient_checkpointing \
  --seed "$SEED" \
  --skip_final_eval
