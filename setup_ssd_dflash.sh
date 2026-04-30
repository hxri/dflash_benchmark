#!/usr/bin/env bash
# Set up CUDA environment for DFlash-SSD on 2× A6000 96GB.
# Installs flash-attn (required — not optional here).
#
# Usage:
#   ./setup_ssd_dflash.sh
#
# What this does:
#   1. Creates .venv-cuda (reuses existing if present)
#   2. Installs dflash[transformers] dependencies
#   3. Installs flash-attn (compiles from source, takes ~10 min)

set -euo pipefail

echo "========================================================"
echo "DFlash-SSD Setup  (2× A6000 / flash-attn required)"
echo "========================================================"

# Require 2 GPUs
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 0)
if [[ "$GPU_COUNT" -lt 2 ]]; then
    echo "WARNING: Found $GPU_COUNT GPU(s). DFlash-SSD uses GPU 0 (target) + GPU 1 (draft)."
    echo "         Single-GPU mode is supported but runs verify+draft sequentially."
fi
echo "GPUs found: $GPU_COUNT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# Create venv if needed
if [[ ! -f ".venv-cuda/bin/python" ]]; then
    echo "Creating .venv-cuda ..."
    uv venv .venv-cuda --python 3.12
else
    echo "Reusing existing .venv-cuda"
fi

echo "Installing dflash[transformers] ..."
uv pip install --python .venv-cuda/bin/python -e ".[transformers]"

echo ""
echo "Installing flash-attn (compiles from source, ~10 min) ..."
.venv-cuda/bin/pip install flash-attn --no-build-isolation
echo "flash-attn installed."

echo ""
echo "Installing rich extras for live monitor ..."
uv pip install --python .venv-cuda/bin/python "rich>=13.0"

echo ""
echo "========================================================"
echo "Setup complete."
echo ""
echo "Run DFlash-SSD benchmark:"
echo "  ./run_ssd_dflash.sh"
echo ""
echo "Run analysis only (Phase 0):"
echo "  ./run_ssd_dflash.sh --analyze-only"
echo "========================================================"
