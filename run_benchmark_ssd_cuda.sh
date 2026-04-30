#!/usr/bin/env bash
# SSD baseline benchmark — Qwen3-4B target, Qwen3-0.6B standard-LM draft.
# Backend: transformers (CUDA). Tested on RTX A6000 / H100.
#
# Compares three decoding modes on all datasets from the SSD paper
# (Kumar, Dao & May, arXiv:2603.03251 / ICLR 2026):
#   1. Autoregressive baseline (Qwen3-4B only, block_size=1)
#   2. Spec-SD: standard speculative decoding (Qwen3-4B + Qwen3-0.6B LM draft)
#   3. DFlash: block-diffusion speculative decoding (Qwen3-4B + DFlash-b16 draft)
#
# The Spec-SD implementation in dflash/model.py uses the batch-verification
# algorithm: draft generates k tokens autoregressively, target verifies all k
# in one forward pass, longest matching prefix is accepted plus a bonus token.
#
# SSD paper datasets: gsm8k, humaneval, alpaca, ultrafeedback
# DFlash paper datasets: gsm8k, math500, humaneval, mbpp, mt-bench
# This script runs the union of both to enable cross-paper comparison.
#
# Usage:
#   ./run_benchmark_ssd_cuda.sh                      # all datasets
#   DATASET=gsm8k ./run_benchmark_ssd_cuda.sh        # single dataset
#   MAX_SAMPLES=128 ./run_benchmark_ssd_cuda.sh      # cap samples
#   GPU=1 ./run_benchmark_ssd_cuda.sh                # specific GPU
#   SSD_DRAFT=Qwen/Qwen3-1.7B ./run_benchmark_ssd_cuda.sh  # larger SD draft
#   NUM_DRAFT_TOKENS=8 ./run_benchmark_ssd_cuda.sh   # more draft tokens

set -euo pipefail

TARGET_MODEL="${MODEL:-Qwen/Qwen3-4B}"
DFLASH_DRAFT="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
SSD_DRAFT="${SSD_DRAFT:-Qwen/Qwen3-0.6B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-5}"
GPU="${GPU:-0}"
PYTHON=".venv-cuda/bin/python"

# Default: SSD paper datasets + DFlash paper datasets
DATASETS="${DATASET:-gsm8k humaneval alpaca ultrafeedback math500 mbpp mt-bench}"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: .venv-cuda not found. Run ./setup_cuda.sh first."
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPU"

echo "=================================================================="
echo "SSD Baseline Benchmark — Qwen3-4B (transformers / CUDA)"
echo "=================================================================="
echo "Target model:        $TARGET_MODEL"
echo "DFlash draft:        $DFLASH_DRAFT"
echo "Spec-SD draft (SSD): $SSD_DRAFT"
echo "Num draft tokens:    $NUM_DRAFT_TOKENS  (for Spec-SD)"
echo "Max new tokens:      $MAX_NEW_TOKENS"
echo "Temperature:         $TEMPERATURE"
echo "Datasets:            $DATASETS"
echo "GPU:                 $GPU"
echo "=================================================================="
echo ""
echo "Modes compared per dataset:"
echo "  [AR]      Autoregressive — Qwen3-4B only (block_size=1)"
echo "  [Spec-SD] Standard speculative decoding — Qwen3-4B + Qwen3-0.6B LM draft"
echo "  [DFlash]  Block-diffusion speculative decoding — Qwen3-4B + DFlash draft"
echo ""

for DATASET in $DATASETS; do
    echo ""
    echo "---------- Dataset: $DATASET ----------"
    "$PYTHON" -m dflash.benchmark \
        --backend transformers \
        --model "$TARGET_MODEL" \
        --draft-model "$DFLASH_DRAFT" \
        --ssd-draft-model "$SSD_DRAFT" \
        --num-draft-tokens "$NUM_DRAFT_TOKENS" \
        --dataset "$DATASET" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        ${MAX_SAMPLES:+--max-samples "$MAX_SAMPLES"}
done

echo ""
echo "=================================================================="
echo "Benchmark complete."
echo "=================================================================="
