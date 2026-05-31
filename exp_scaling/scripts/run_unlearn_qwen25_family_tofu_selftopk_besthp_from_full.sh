#!/usr/bin/env bash
set -euo pipefail

ROOT=.
PY=$ROOT/venv_muse_paper/bin/python
DATASET_CACHE_DIR=$ROOT/open-unlearning/cache/hf_datasets
LOG_DIR=$ROOT/exp_scaling/logs
SAVE_ROOT=$ROOT/exp_scaling/saves/unlearn/qwen25

SEED=${SEED:-42}
GPU=${CUDA_VISIBLE_DEVICES:-0}
RUN_SUFFIX=selftopk_besthp_qwen_template_seed${SEED}

mkdir -p "$LOG_DIR" "$SAVE_ROOT"

declare -a MODELS=(
  "qwen25_0p5b_instruct"
  "qwen25_1p5b_instruct"
  "qwen25_3b_instruct"
  "qwen25_7b_instruct"
)

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting Qwen2.5 TOFU unlearning family run with LLAMA2 best-hp self-topk settings and Qwen chat template (seed=${SEED}, gpu=${GPU})"

for MODEL_KEY in "${MODELS[@]}"; do
  MODEL_DIR=$ROOT/exp_scaling/saves/finetune/qwen25/$MODEL_KEY/tofu_full
  TOKENIZER_DIR=$MODEL_DIR
  OUT_DIR=$SAVE_ROOT/$MODEL_KEY/tofu_margin_unlearning_forget10_retain90_cont_lora_lambda0p1_rho0p5_topk10_self_beta1_noretainguard_${RUN_SUFFIX}
  LOG=$LOG_DIR/train_${MODEL_KEY}_tofu_margin_lambda0p1_rho0p5_topk10_self_beta1_klretain1_noretainguard_${RUN_SUFFIX}.log

  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Missing tofu_full model dir for ${MODEL_KEY}: ${MODEL_DIR}" | tee -a "$LOG"
    exit 1
  fi

  if [[ -f "$OUT_DIR/early_stop.json" || -d "$OUT_DIR/final_model" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Skipping ${MODEL_KEY}; existing run artifacts found at ${OUT_DIR}" | tee -a "$LOG"
    continue
  fi

  mkdir -p "$OUT_DIR"
  : > "$LOG"
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting ${MODEL_KEY} from ${MODEL_DIR}" | tee -a "$LOG"

  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  CUDA_VISIBLE_DEVICES="$GPU" \
    "$PY" -u "$ROOT/scripts/run_margin_unlearning_qwen.py" \
      --model_dir "$MODEL_DIR" \
      --retain_teacher_model_dir "$MODEL_DIR" \
      --retain_loss_mode kl_target \
      --lambda_retain 1.0 \
      --tokenizer_dir "$TOKENIZER_DIR" \
      --cache_dir "$ROOT/cache" \
      --dataset_cache_dir "$DATASET_CACHE_DIR" \
      --out_dir "$OUT_DIR" \
      --corpus tofu \
      --template_name qwen25 \
      --forget_split forget10 \
      --retain_split retain90 \
      --max_len 512 \
      --batch_size 2 \
      --epochs 3 \
      --lr 1e-4 \
      --objective_mode forget_margin \
      --lambda_forget_margin 0.1 \
      --forget_margin_rho 0.5 \
      --forget_margin_buffer_eta 0.5 \
      --margin_competitor_source self \
      --margin_teacher_topk 10 \
      --margin_competitor_beta 1.0 \
      --margin_beta_parameterization tempered_posterior \
      --use_lora \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --lora_target_modules q_proj,k_proj,v_proj,o_proj \
      --lora_bias none \
      --forget_penalty_mode continuation \
      --continuation_cut_min_ratio 0.2 \
      --continuation_cut_max_ratio 0.6 \
      --continuation_min_prefix_tokens 32 \
      --continuation_min_suffix_tokens 32 \
      --early_stop_two_stage \
      --early_stop_trigger_metric violation_mean \
      --early_stop_trigger_threshold 0.2 \
      --early_stop_confirm_num_forget_examples 64 \
      --early_stop_confirm_alpha 0.2 \
      --early_stop_disable_retain_guard \
      --early_stop_eval_every_steps 2 \
      --early_stop_eval_batches 2 \
      --seed "$SEED" \
      --device cuda \
      --precision auto \
      --gradient_checkpointing \
      --logging_steps 10 \
      --skip_final_eval \
      2>&1 | tee -a "$LOG"

  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished ${MODEL_KEY}" | tee -a "$LOG"
done

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Finished Qwen2.5 TOFU unlearning family run"
