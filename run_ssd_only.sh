#!/usr/bin/env bash
# SSD-only benchmark — Qwen3-4B target + Qwen3-0.6B standard LM draft.
# Backend: MLX (Apple Silicon). No DFlash draft involved.
#
# Compares two modes only:
#   1. Autoregressive baseline (Qwen3-4B, no draft)
#   2. Spec-SD: standard speculative decoding (Qwen3-4B + small LM draft)
#
# Runs the SSD paper datasets (arXiv:2603.03251): gsm8k, humaneval, alpaca, ultrafeedback
#
# Usage:
#   ./run_ssd_only.sh                            # all 4 SSD paper datasets, 128 samples
#   DATASET=gsm8k ./run_ssd_only.sh              # single dataset
#   MAX_SAMPLES=500 ./run_ssd_only.sh            # more samples
#   SSD_DRAFT=Qwen/Qwen3-1.7B ./run_ssd_only.sh # use 1.7B draft instead of 0.6B
#   NUM_DRAFT_TOKENS=8 ./run_ssd_only.sh         # more draft tokens per verify step

set -euo pipefail

TARGET_MODEL="${MODEL:-Qwen/Qwen3-4B}"
SSD_DRAFT="${SSD_DRAFT:-Qwen/Qwen3-0.6B}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-5}"

# SSD paper datasets
DATASETS="${DATASET:-gsm8k humaneval alpaca ultrafeedback}"

echo "=================================================="
echo "SSD-only Benchmark (MLX / Apple Silicon)"
echo "=================================================="
echo "Target model:     $TARGET_MODEL"
echo "SSD draft model:  $SSD_DRAFT"
echo "Num draft tokens: $NUM_DRAFT_TOKENS"
echo "Max samples:      $MAX_SAMPLES"
echo "Max new tokens:   $MAX_NEW_TOKENS"
echo "Temperature:      $TEMPERATURE"
echo "Datasets:         $DATASETS"
echo "=================================================="
echo ""

for DATASET in $DATASETS; do
    echo ""
    echo "---------- Dataset: $DATASET ----------"
    .venv/bin/python -m dflash.benchmark \
        --backend mlx \
        --model "$TARGET_MODEL" \
        --ssd-draft-model "$SSD_DRAFT" \
        --num-draft-tokens "$NUM_DRAFT_TOKENS" \
        --dataset "$DATASET" \
        --max-samples "$MAX_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE"
done

echo ""
echo "=================================================="
echo "Benchmark complete."
echo "=================================================="
