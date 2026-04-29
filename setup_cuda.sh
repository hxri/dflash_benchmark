#!/usr/bin/env bash
# Set up CUDA virtual environment for DFlash on an RTX A6000 (or any CUDA GPU).
# Creates .venv-cuda with torch + transformers==4.57.1 + accelerate.
#
# Optionally installs flash-attn for better performance (recommended):
#   INSTALL_FLASH_ATTN=1 ./setup_cuda.sh
#
# Run this once on your Linux/CUDA machine, then use run_benchmark_cuda.sh.

set -euo pipefail

echo "Creating .venv-cuda with Python 3.12 ..."
uv venv .venv-cuda --python 3.12

echo "Installing dflash[transformers] ..."
uv pip install --python .venv-cuda/bin/python -e ".[transformers]"

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    echo "Installing flash-attn (this takes several minutes to compile) ..."
    .venv-cuda/bin/pip install flash-attn --no-build-isolation
    echo "flash-attn installed."
else
    echo ""
    echo "NOTE: flash-attn not installed. The benchmark will fall back to torch.sdpa."
    echo "For maximum speedup, run:  INSTALL_FLASH_ATTN=1 ./setup_cuda.sh"
fi

echo ""
echo "Setup complete. Run benchmarks with:"
echo "  ./run_benchmark_cuda.sh"
echo "  ./run_quick_test_cuda.sh"
