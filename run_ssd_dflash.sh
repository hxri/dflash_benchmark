#!/usr/bin/env bash
# DFlash-SSD benchmark runner — 2× A6000 96 GB
#
# Runs three phases in sequence:
#   Phase 0  Analysis  — validates staleness assumption (fast, ~5 min)
#   Phase 1  Benchmark — AR vs DFlash vs DFlash-SSD across all paper datasets
#
# Usage:
#   ./run_ssd_dflash.sh                      # full run (analysis + benchmark)
#   ./run_ssd_dflash.sh --analyze-only       # Phase 0 only
#   ./run_ssd_dflash.sh --benchmark-only     # Phase 1 only (skip analysis)
#
# Key env-var overrides:
#   MODEL=Qwen/Qwen3-8B        target model  (default: Qwen/Qwen3-4B)
#   DRAFT=z-lab/Qwen3-8B-DFlash-b16  draft  (default: Qwen3-4B DFlash)
#   FAN_OUT=3                  fan-out F      (default: 2)
#   MAX_SAMPLES=32             samples/dataset (default: 64)
#   TARGET_GPU=0               GPU for target (default: 0)
#   DRAFT_GPU=1                GPU for draft  (default: 1)
#   DATASETS="gsm8k humaneval" override datasets

set -euo pipefail

PYTHON=".venv-cuda/bin/python"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT="${DRAFT:-z-lab/Qwen3-4B-DFlash-b16}"
FAN_OUT="${FAN_OUT:-2}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TARGET_GPU="${TARGET_GPU:-0}"
DRAFT_GPU="${DRAFT_GPU:-1}"
ACCEPTANCE_STRATEGY="${ACCEPTANCE_STRATEGY:-running}"
FAST_REFINE="${FAST_REFINE:-0}"   # set to 1 when H_SIM < 0.85 (e.g. math reasoning)
OUT_DIR="${OUT_DIR:-results/ssd_dflash}"
ANALYSIS_DIR="${OUT_DIR}/analysis"

# Datasets: union of DFlash paper + SSD paper
DATASETS="${DATASETS:-gsm8k math500 humaneval mbpp mt-bench alpaca ultrafeedback}"
# Phase-0 uses a subset (faster)
ANALYSIS_DATASETS="${ANALYSIS_DATASETS:-gsm8k humaneval}"
ANALYSIS_SAMPLES="${ANALYSIS_SAMPLES:-32}"

ANALYZE=true
BENCHMARK=true

for arg in "$@"; do
  case $arg in
    --analyze-only)   BENCHMARK=false ;;
    --benchmark-only) ANALYZE=false   ;;
  esac
done

# ── Preflight ────────────────────────────────────────────
if [[ ! -f "$PYTHON" ]]; then
  echo "ERROR: .venv-cuda not found. Run ./setup_ssd_dflash.sh first."
  exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
echo "=================================================================="
echo "DFlash-SSD  (DFlash + Speculative Speculative Decoding)"
echo "=================================================================="
echo "  Target model : $MODEL  (cuda:$TARGET_GPU)"
echo "  Draft model  : $DRAFT  (cuda:$DRAFT_GPU)"
echo "  Fan-out F    : $FAN_OUT"
echo "  Max samples  : $MAX_SAMPLES per dataset"
echo "  Max tokens   : $MAX_NEW_TOKENS"
echo "  Temperature  : $TEMPERATURE"
echo "  Datasets     : $DATASETS"
echo "  GPUs found   : $GPU_COUNT"
echo "  Output dir   : $OUT_DIR"
echo "=================================================================="
echo ""

if [[ "$GPU_COUNT" -lt 2 && "$TARGET_GPU" != "$DRAFT_GPU" ]]; then
  echo "WARNING: Only $GPU_COUNT GPU(s) found but --target-gpu $TARGET_GPU --draft-gpu $DRAFT_GPU."
  echo "         Falling back to single-GPU mode (DRAFT_GPU=$TARGET_GPU)."
  DRAFT_GPU="$TARGET_GPU"
fi

# ── Phase 0: Analysis ─────────────────────────────────────
if [[ "$ANALYZE" == "true" ]]; then
  echo ""
  echo "────────────────────────────────────────────────────"
  echo "Phase 0 — Staleness Analysis"
  echo "────────────────────────────────────────────────────"
  echo "This validates whether DFlash-SSD's core assumption holds:"
  echo "  cos_sim(H_t, H_{t+1}) ≈ 1  and  stale acceptance ≈ fresh acceptance"
  echo ""

  for DS in $ANALYSIS_DATASETS; do
    echo "  [hidden_state_sim] dataset=$DS ..."
    "$PYTHON" -m dflash.analysis.hidden_state_sim \
      --model "$MODEL" \
      --draft-model "$DRAFT" \
      --dataset "$DS" \
      --max-samples "$ANALYSIS_SAMPLES" \
      --max-new-tokens 512 \
      --gpu "$TARGET_GPU" \
      --out-dir "$ANALYSIS_DIR"
    echo ""

    echo "  [stale_draft_quality] dataset=$DS ..."
    "$PYTHON" -m dflash.analysis.stale_draft_quality \
      --model "$MODEL" \
      --draft-model "$DRAFT" \
      --dataset "$DS" \
      --max-samples "$ANALYSIS_SAMPLES" \
      --max-new-tokens 512 \
      --gpu "$TARGET_GPU" \
      --out-dir "$ANALYSIS_DIR"
    echo ""
  done

  echo "  [timing_profile] ..."
  "$PYTHON" -m dflash.analysis.timing_profile \
    --model "$MODEL" \
    --draft-model "$DRAFT" \
    --gpu "$TARGET_GPU" \
    --out-dir "$ANALYSIS_DIR"
  echo ""

  echo "Phase 0 complete. Results in $ANALYSIS_DIR/"
  echo ""
  echo "  What to look for:"
  echo "  ┌─ hidden_state_sim: median cos_sim > 0.90 → GREEN"
  echo "  ├─ stale_quality:    acceptance retention > 85% → GREEN"
  echo "  └─ timing_profile:   T_verify / T_draft > F=$FAN_OUT → GREEN"
  echo ""
fi

# ── Phase 1: Benchmark ────────────────────────────────────
if [[ "$BENCHMARK" == "true" ]]; then
  echo "────────────────────────────────────────────────────"
  echo "Phase 1 — DFlash-SSD Benchmark"
  echo "────────────────────────────────────────────────────"
  echo "Compares: AR  |  Standard DFlash  |  DFlash-SSD"
  echo ""

  # Build --dataset flags
  DATASET_FLAGS=""
  for DS in $DATASETS; do
    DATASET_FLAGS="$DATASET_FLAGS --dataset $DS"
  done

  FAST_REFINE_FLAG=""
  [[ "$FAST_REFINE" == "1" ]] && FAST_REFINE_FLAG="--fast-refine"

  "$PYTHON" -m dflash.benchmark_ssd_dflash \
    --model "$MODEL" \
    --draft-model "$DRAFT" \
    $DATASET_FLAGS \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --fan-out "$FAN_OUT" \
    --acceptance-strategy "$ACCEPTANCE_STRATEGY" \
    --target-gpu "$TARGET_GPU" \
    --draft-gpu "$DRAFT_GPU" \
    --out-dir "$OUT_DIR" \
    $FAST_REFINE_FLAG

  echo ""
  echo "Phase 1 complete. Results in $OUT_DIR/"
fi

echo ""
echo "=================================================================="
echo "Done. Check $OUT_DIR/summary.json for aggregate results."
echo ""
echo "Key metrics to inspect:"
echo "  speedup_ssd_vs_dflash  — net gain over standard DFlash"
echo "  ssd_cache_hit_rate     — fraction of steps where stale draft was used"
echo "  ssd_mean_h_cosine_sim  — hidden state similarity (should be > 0.90)"
echo "  ssd_draft_saved_ms     — avg draft latency hidden per step"
echo "=================================================================="
