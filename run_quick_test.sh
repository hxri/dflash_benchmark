#!/usr/bin/env bash
# Quick smoke test: 5 samples on gsm8k to verify the setup works end-to-end.
set -euo pipefail

echo "Running quick smoke test (5 samples, gsm8k) ..."
.venv/bin/python -m dflash.benchmark \
    --backend mlx \
    --model Qwen/Qwen3-4B \
    --draft-model z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k \
    --max-samples 5 \
    --max-new-tokens 512 \
    --temperature 0.0
