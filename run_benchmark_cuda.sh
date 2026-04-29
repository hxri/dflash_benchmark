#!/usr/bin/env bash
# DFlash benchmark — Qwen3-4B (smallest paper model) on all 5 paper datasets.
# Backend: transformers (CUDA). Tested on RTX A6000 96GB.
#
# Runs baseline autoregressive and DFlash back-to-back on each sample and
# reports: baseline tok/s, DFlash tok/s, speedup, and acceptance length.
# This matches the paper's Table 1 / Figure methodology exactly.
#
# Usage:
#   ./run_benchmark_cuda.sh                   # all 5 datasets, full test sets
#   DATASET=gsm8k ./run_benchmark_cuda.sh     # single dataset
#   MAX_SAMPLES=128 ./run_benchmark_cuda.sh   # cap samples (faster)
#   GPU=1 ./run_benchmark_cuda.sh             # use a specific GPU index

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
DATASETS="${DATASET:-gsm8k math500 humaneval mbpp mt-bench}"
GPU="${GPU:-0}"
PYTHON=".venv-cuda/bin/python"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: .venv-cuda not found. Run ./setup_cuda.sh first."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

# Optional: uncomment for multi-GPU tensor parallelism (not needed for 4B)
# TORCHRUN_ARGS="--nproc-per-node=2"
# PYTHON="torchrun $TORCHRUN_ARGS -m dflash.benchmark"

echo "=================================================="
echo "DFlash Benchmark — Qwen3-4B (transformers / CUDA)"
echo "=================================================="
echo "Target model:    $MODEL"
echo "Draft model:     $DRAFT_MODEL"
echo "Max new tokens:  $MAX_NEW_TOKENS"
echo "Temperature:     $TEMPERATURE"
echo "Datasets:        $DATASETS"
echo "GPU:             $GPU"
echo "=================================================="
echo ""

for DATASET in $DATASETS; do
    echo ""
    echo "---------- Dataset: $DATASET ----------"
    "$PYTHON" -m dflash.benchmark \
        --backend transformers \
        --model "$MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --dataset "$DATASET" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}
done

echo ""
echo "=================================================="
echo "Benchmark complete."
echo "=================================================="
