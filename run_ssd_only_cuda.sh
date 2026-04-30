#!/usr/bin/env bash
# SSD-only benchmark — Qwen3-4B target + Qwen3-0.6B standard LM draft.
# Backend: transformers (CUDA). No DFlash draft involved.
#
# Compares two modes only:
#   1. Autoregressive baseline (Qwen3-4B, no draft)
#   2. Spec-SD: standard speculative decoding (Qwen3-4B + small LM draft)
#
# Runs the SSD paper datasets (arXiv:2603.03251): gsm8k, humaneval, alpaca, ultrafeedback
#
# Usage:
#   ./run_ssd_only_cuda.sh                              # all 4 SSD paper datasets
#   DATASET=gsm8k ./run_ssd_only_cuda.sh                # single dataset
#   MAX_SAMPLES=128 ./run_ssd_only_cuda.sh              # cap samples
#   GPU=1 ./run_ssd_only_cuda.sh                        # specific GPU
#   SSD_DRAFT=Qwen/Qwen3-1.7B ./run_ssd_only_cuda.sh   # larger draft model
#   NUM_DRAFT_TOKENS=8 ./run_ssd_only_cuda.sh           # more draft tokens

set -euo pipefail

TARGET_MODEL="${MODEL:-Qwen/Qwen3-4B}"
SSD_DRAFT="${SSD_DRAFT:-Qwen/Qwen3-0.6B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-5}"
SPECULATION_FANOUT="${SPECULATION_FANOUT:-2}"
GPU="${GPU:-0}"
PYTHON=".venv-cuda/bin/python"

# SSD paper datasets
DATASETS="${DATASET:-gsm8k humaneval alpaca ultrafeedback}"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: .venv-cuda not found. Run ./setup_cuda.sh first."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=================================================="
echo "SSD-only Benchmark (transformers / CUDA)"
echo "=================================================="
echo "Target model:     $TARGET_MODEL"
echo "SSD draft model:  $SSD_DRAFT"
echo "Num draft tokens: $NUM_DRAFT_TOKENS"
echo "Spec fan-out:     $SPECULATION_FANOUT"
echo "Max new tokens:   $MAX_NEW_TOKENS"
echo "Temperature:      $TEMPERATURE"
echo "Datasets:         $DATASETS"
echo "GPU:              $GPU"
echo "=================================================="
echo ""

for DATASET in $DATASETS; do
    echo ""
    echo "---------- Dataset: $DATASET ----------"
    "$PYTHON" -m dflash.benchmark \
        --backend transformers \
        --model "$TARGET_MODEL" \
        --ssd-draft-model "$SSD_DRAFT" \
        --num-draft-tokens "$NUM_DRAFT_TOKENS" \
        --speculation-fanout "$SPECULATION_FANOUT" \
        --dataset "$DATASET" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}
done

echo ""
echo "=================================================="
echo "Benchmark complete."
echo "=================================================="
