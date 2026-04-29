#!/usr/bin/env bash
# DFlash benchmark — Qwen3-4B (smallest paper model) on all 5 paper datasets
# Backend: MLX (Apple Silicon). Paper: arxiv.org/abs/2602.06036
#
# Usage:
#   ./run_benchmark.sh                  # all datasets, 128 samples each
#   DATASET=gsm8k ./run_benchmark.sh    # single dataset
#   MAX_SAMPLES=500 ./run_benchmark.sh  # more samples (slower)
#
# Models are downloaded on first run and cached in ~/.cache/huggingface/
# First run takes ~10-20 min to download; subsequent runs start immediately.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DATASETS="${DATASET:-gsm8k math500 humaneval mbpp mt-bench}"

echo "=================================================="
echo "DFlash Benchmark — Qwen3-4B (MLX / Apple Silicon)"
echo "=================================================="
echo "Target model:  $MODEL"
echo "Draft model:   $DRAFT_MODEL"
echo "Max samples:   $MAX_SAMPLES"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Temperature:   $TEMPERATURE"
echo "Datasets:      $DATASETS"
echo "=================================================="
echo ""

for DATASET in $DATASETS; do
    echo ""
    echo "---------- Dataset: $DATASET ----------"
    .venv/bin/python -m dflash.benchmark \
        --backend mlx \
        --model "$MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --dataset "$DATASET" \
        --max-samples "$MAX_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE"
done

echo ""
echo "=================================================="
echo "Benchmark complete."
echo "=================================================="
