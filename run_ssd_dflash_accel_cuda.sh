#!/usr/bin/env bash
# Novel SSD+DFlash accelerated decode workflow (CUDA-only, flash-attn required).
# Designed for 2x A6000 class GPUs with Qwen3 target + DFlash draft.
#
# This is not the benchmark script. It runs the new monitored/eval pipeline:
#   1) Executes dataset evaluation with per-step traces + per-sample summaries.
#   2) Writes aggregate metrics and live decoding health indicators.
#   3) Runs a post-hoc monitor report that tells you what to watch.
#
# Usage:
#   ./run_ssd_dflash_accel_cuda.sh
#   DATASET=gsm8k MAX_SAMPLES=32 ./run_ssd_dflash_accel_cuda.sh
#   MODEL=Qwen/Qwen3-14B DRAFT_MODEL=z-lab/Qwen3-14B-DFlash-b16 ./run_ssd_dflash_accel_cuda.sh
#   NPROC=1 ./run_ssd_dflash_accel_cuda.sh      # single-GPU fallback

set -euo pipefail

PYTHON=".venv-cuda/bin/python"

MODEL="${MODEL:-Qwen/Qwen3-4B}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-gsm8k}"
MAX_SAMPLES="${MAX_SAMPLES:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MONITOR_EVERY="${MONITOR_EVERY:-8}"
SPECULATION_FANOUT="${SPECULATION_FANOUT:-2}"
NPROC="${NPROC:-2}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
LIVE_MONITOR="${LIVE_MONITOR:-1}"
WRITE_TEXT_OUTPUTS="${WRITE_TEXT_OUTPUTS:-0}"

if [[ ! -f "$PYTHON" ]]; then
    echo "ERROR: Missing .venv-cuda. Run ./setup_ssd_dflash.sh first."
    exit 1
fi

ATTN_IMPL="sdpa"
if "$PYTHON" -c "import flash_attn" >/dev/null 2>&1; then
    ATTN_IMPL="flash_attention_2"
else
    echo "WARNING: flash-attn is not installed in .venv-cuda. Falling back to torch.sdpa."
    echo "         Throughput may be lower than flash-attn."
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/ssd_dflash_accel/${RUN_TS}"
mkdir -p "$OUT_DIR"

THINK_FLAG=""
if [[ "$ENABLE_THINKING" == "1" ]]; then
    THINK_FLAG="--enable-thinking"
fi

MONITOR_FLAG=""
if [[ "$LIVE_MONITOR" == "1" ]]; then
    MONITOR_FLAG="--live-monitor"
fi

TEXT_FLAG=""
if [[ "$WRITE_TEXT_OUTPUTS" == "1" ]]; then
    TEXT_FLAG="--write-text-outputs"
fi

echo "============================================================"
echo "SSD+DFlash Accelerated Decode (CUDA / ${ATTN_IMPL})"
echo "============================================================"
echo "Model:           $MODEL"
echo "DFlash draft:    $DRAFT_MODEL"
echo "Dataset:         $DATASET"
echo "Max samples:     $MAX_SAMPLES"
echo "Max new tokens:  $MAX_NEW_TOKENS"
echo "Temperature:     $TEMPERATURE"
echo "Block size:      $BLOCK_SIZE"
echo "Spec fan-out:    $SPECULATION_FANOUT"
echo "NPROC:           $NPROC"
echo "Output dir:      $OUT_DIR"
echo "============================================================"

torchrun --nproc_per_node="$NPROC" -m dflash.ssd_dflash_accel \
    --mode dataset \
    --model "$MODEL" \
    --draft-model "$DRAFT_MODEL" \
    --dataset "$DATASET" \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --block-size "$BLOCK_SIZE" \
    --monitor-every "$MONITOR_EVERY" \
    --speculation-fanout "$SPECULATION_FANOUT" \
    --out-dir "$OUT_DIR" \
    $THINK_FLAG \
    $MONITOR_FLAG \
    $TEXT_FLAG

echo ""
echo "Generating monitor report ..."
"$PYTHON" -m dflash.analysis.ssd_dflash_monitor_report --run-dir "$OUT_DIR"

echo ""
echo "Done. Key outputs:"
echo "  Aggregate:   $OUT_DIR/aggregate.json"
echo "  Traces:      $OUT_DIR/trace_rank*.jsonl"
echo "  Summaries:   $OUT_DIR/samples_rank*.jsonl"
echo ""
echo "Live decode signals to watch:"
echo "  1) acceptance_ratio per step (healthy if stable, not collapsing)"
echo "  2) cumulative_decode_tps trend (healthy if monotonic/stable)"
echo "  3) hidden_drift_cosine (healthy if mostly high and steady)"
