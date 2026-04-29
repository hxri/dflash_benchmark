#!/usr/bin/env bash
# Quick smoke test: 5 samples on gsm8k to verify CUDA setup works end-to-end.
set -euo pipefail

PYTHON=".venv-cuda/bin/python"
if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: .venv-cuda not found. Run ./setup_cuda.sh first."
    exit 1
fi

echo "Running quick smoke test (5 samples, gsm8k, CUDA) ..."
"$PYTHON" -m dflash.benchmark \
    --backend transformers \
    --model Qwen/Qwen3-4B \
    --draft-model z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k \
    --max-samples 5 \
    --max-new-tokens 512 \
    --temperature 0.0
