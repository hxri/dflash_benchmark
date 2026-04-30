#!/usr/bin/env bash
# True Saguaro two-GPU async speculative decoding.
#
# GPU layout (fixed):
#   cuda:0 — Qwen3-4B target (verifier)
#   cuda:1 — Draft model (primary + backup speculator)
#
# Paths:
#   ssd      : Qwen3-0.6B AR draft on GPU 1 (SSD-only, Saguaro-style)
#   combined : DFlash diffusion draft on GPU 1 with embed/lm_head copies
#
# Async overlap:
#   While GPU 0 verifies the current draft batch, GPU 1 is already
#   pre-computing all likely next-round drafts (outcome cache).  On cache
#   hit the draft result is ready with zero extra latency.  On miss the
#   backup speculator runs a fresh pass using the newly verified
#   target hidden states (better quality than the primary's stale features).
#
# Usage:
#   ./run_saguaro_cuda.sh                          # combined path, gsm8k, 64 samples
#   PATH_MODE=ssd ./run_saguaro_cuda.sh            # SSD-only path
#   DATASET=humaneval ./run_saguaro_cuda.sh        # different dataset
#   SPECULATION_FANOUT=2 ./run_saguaro_cuda.sh     # 2 bonus candidates / length
#   MAX_SAMPLES=4 MAX_NEW_TOKENS=32 ./run_saguaro_cuda.sh  # quick smoke test

set -euo pipefail

PYTHON=".venv-cuda/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: .venv-cuda not found. Run ./setup_ssd_dflash.sh first."
    exit 1
fi

# ---- Configuration --------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-4B}"
SSD_DRAFT="${SSD_DRAFT:-Qwen/Qwen3-0.6B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
PATH_MODE="${PATH_MODE:-combined}"    # ssd | combined | dflash
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-5}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
# fanout=1 → one bonus candidate per acceptance length (balanced GPU-1 budget)
# fanout=2 → two bonus candidates (higher hit rate, more GPU-1 work)
SPECULATION_FANOUT="${SPECULATION_FANOUT:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"

# Both GPUs must be visible.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/saguaro/${PATH_MODE}_${RUN_TS}"
mkdir -p "$OUT_DIR"

# ---- Summary banner -------------------------------------------------------
echo "=================================================================="
echo "  True Saguaro Two-GPU Async Speculative Decoding"
echo "=================================================================="
echo "  Path             : $PATH_MODE"
echo "  Target (cuda:0)  : $MODEL"
if [[ "$PATH_MODE" == "ssd" ]]; then
echo "  Draft  (cuda:1)  : $SSD_DRAFT  (SSD AR, $NUM_DRAFT_TOKENS tokens/round)"
else
echo "  Draft  (cuda:1)  : $DRAFT_MODEL  (DFlash, block_size=$BLOCK_SIZE)"
fi
echo "  Dataset          : $DATASET  (max_samples=$MAX_SAMPLES)"
echo "  Max new tokens   : $MAX_NEW_TOKENS  temperature=$TEMPERATURE"
echo "  Speculation fanout: $SPECULATION_FANOUT branches/acceptance-len"
echo "  Output dir        : $OUT_DIR"
echo "=================================================================="

# ---- Build argument list --------------------------------------------------
ARGS=(
    --model "$MODEL"
    --path "$PATH_MODE"
    --ssd-draft "$SSD_DRAFT"
    --draft-model "$DRAFT_MODEL"
    --num-draft-tokens "$NUM_DRAFT_TOKENS"
    --block-size "$BLOCK_SIZE"
    --dataset "$DATASET"
    --max-samples "$MAX_SAMPLES"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --speculation-fanout "$SPECULATION_FANOUT"
    --out-dir "$OUT_DIR"
)

if [[ "${ENABLE_THINKING:-0}" == "1" ]]; then
    ARGS+=(--enable-thinking)
fi

# ---- Run ------------------------------------------------------------------
"$PYTHON" -m dflash.saguaro "${ARGS[@]}"

echo ""
echo "Results saved to: $OUT_DIR"
