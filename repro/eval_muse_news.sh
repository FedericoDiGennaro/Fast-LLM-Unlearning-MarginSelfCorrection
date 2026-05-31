#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
SEED=${SEED:-42}
DEFAULT_RUN_DIR="$ROOT/outputs/news_masc_seed${SEED}"
INCLUDE_PRIVLEAK=${INCLUDE_PRIVLEAK:-0}

latest_checkpoint() {
  find "$1" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-earlystop-*' 2>/dev/null | sort -V | tail -n 1
}

first_snapshot() {
  find "$1" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | head -n 1
}

MODEL_DIR=${MODEL_DIR:-$(latest_checkpoint "$DEFAULT_RUN_DIR")}
TOKENIZER_DIR=${TOKENIZER_DIR:-$(first_snapshot "$ROOT/cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots")}
OUT_FILE=${OUT_FILE:-"$ROOT/outputs/evals/muse_news_masc_seed${SEED}.csv"}
TEMP_DIR=${TEMP_DIR:-"$ROOT/outputs/eval_temp/muse_news_masc_seed${SEED}"}
PER_EXAMPLE_DIR=${PER_EXAMPLE_DIR:-"$ROOT/outputs/per_example/muse_news_masc_seed${SEED}"}

if [[ -z "${MODEL_DIR}" || ! -d "${MODEL_DIR}" ]]; then
  echo "Missing News checkpoint. Set MODEL_DIR=... or run repro/run_masc_news.sh first." >&2
  exit 1
fi
if [[ -z "${TOKENIZER_DIR}" || ! -d "${TOKENIZER_DIR}" ]]; then
  echo "Missing Llama-2 tokenizer snapshot. Run: python repro/download_assets.py --target news" >&2
  exit 1
fi
if [[ ! -f "$ROOT/muse_bench/data/news/knowmem/forget_qa.json" ]]; then
  echo "MUSE benchmark data not found; materializing it with muse_bench/load_data.py"
  (cd "$ROOT/muse_bench" && "$PYTHON" load_data.py)
fi

mkdir -p "$(dirname "$OUT_FILE")" "$TEMP_DIR" "$PER_EXAMPLE_DIR"
export HF_HOME=${HF_HOME:-"$ROOT/cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$ROOT/cache/hub"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$ROOT/cache/datasets"}

METRICS=(verbmem_f knowmem_f knowmem_r)
if [[ "$INCLUDE_PRIVLEAK" == "1" ]]; then
  METRICS=(verbmem_f privleak knowmem_f knowmem_r)
fi

cd "$ROOT"
"$PYTHON" -m muse_bench.eval \
  --model_dirs "$MODEL_DIR" \
  --names "news_masc_seed${SEED}" \
  --corpus news \
  --tokenizer_dir "$TOKENIZER_DIR" \
  --data_root "$ROOT/muse_bench" \
  --temp_dir "$TEMP_DIR" \
  --per_example_out_dir "$PER_EXAMPLE_DIR" \
  --metrics "${METRICS[@]}" \
  --quantize_4bit "${QUANTIZE_4BIT:-1}" \
  --print_metrics_live 1 \
  --out_file "$OUT_FILE"
