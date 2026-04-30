#!/usr/bin/env bash
# SSD baseline benchmark — Qwen3-4B target, Qwen3-0.6B standard-LM draft.
# Backend: MLX (Apple Silicon).
#
# Compares three decoding modes on all datasets from the SSD paper
# (Kumar, Dao & May, arXiv:2603.03251 / ICLR 2026):
#   1. Autoregressive baseline (Qwen3-4B only)
#   2. Spec-SD: standard speculative decoding (Qwen3-4B + Qwen3-0.6B draft)
#   3. DFlash: block-diffusion speculative decoding (Qwen3-4B + DFlash-b16 draft)
#
# SSD paper datasets: gsm8k, humaneval, alpaca, ultrafeedback
# DFlash paper datasets: gsm8k, math500, humaneval, mbpp, mt-bench
# This script runs the union of both to enable cross-paper comparison.
#
# Usage:
#   ./run_benchmark_ssd.sh                    # all datasets, 128 samples each
#   DATASET=gsm8k ./run_benchmark_ssd.sh      # single dataset
#   MAX_SAMPLES=500 ./run_benchmark_ssd.sh    # more samples (slower)
#   SSD_DRAFT=Qwen/Qwen3-1.7B ./run_benchmark_ssd.sh  # larger SD draft

set -euo pipefail

TARGET_MODEL="${MODEL:-Qwen/Qwen3-4B}"
DFLASH_DRAFT="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
SSD_DRAFT="${SSD_DRAFT:-Qwen/Qwen3-0.6B}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.0}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-5}"

# Default: SSD paper datasets + DFlash paper datasets
DATASETS="${DATASET:-gsm8k humaneval alpaca ultrafeedback math500 mbpp mt-bench}"

echo "=================================================================="
echo "SSD Baseline Benchmark — Qwen3-4B (MLX / Apple Silicon)"
echo "=================================================================="
echo "Target model:        $TARGET_MODEL"
echo "DFlash draft:        $DFLASH_DRAFT"
echo "Spec-SD draft (SSD): $SSD_DRAFT"
echo "Num draft tokens:    $NUM_DRAFT_TOKENS  (for Spec-SD)"
echo "Max samples:         $MAX_SAMPLES"
echo "Max new tokens:      $MAX_NEW_TOKENS"
echo "Temperature:         $TEMPERATURE"
echo "Datasets:            $DATASETS"
echo "=================================================================="
echo ""
echo "Modes compared per dataset:"
echo "  [AR]      Autoregressive — Qwen3-4B only"
echo "  [Spec-SD] Standard speculative decoding — Qwen3-4B + Qwen3-0.6B LM draft"
echo "  [DFlash]  Block-diffusion speculative decoding — Qwen3-4B + DFlash draft"
echo ""

for DATASET in $DATASETS; do
    echo ""
    echo "---------- Dataset: $DATASET ----------"
    .venv/bin/python -m dflash.benchmark \
        --backend mlx \
        --model "$TARGET_MODEL" \
        --draft-model "$DFLASH_DRAFT" \
        --ssd-draft-model "$SSD_DRAFT" \
        --num-draft-tokens "$NUM_DRAFT_TOKENS" \
        --dataset "$DATASET" \
        --max-samples "$MAX_SAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE"
done

echo ""
echo "=================================================================="
echo "Benchmark complete."
echo "=================================================================="
