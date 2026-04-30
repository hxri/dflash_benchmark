"""
DFlash-SSD benchmark: compares AR, DFlash, and DFlash-SSD on all paper datasets.

Outputs:
  - Rich live monitor per sample (DFlash-SSD mode)
  - Per-dataset summary table
  - JSONL log for post-hoc analysis: results/ssd_dflash/

Usage:
  python -m dflash.benchmark_ssd_dflash \
      --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 \
      --dataset gsm8k \
      --max-samples 64 \
      --fan-out 2 \
      --target-gpu 0 --draft-gpu 1
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import chain
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from loguru import logger
from rich import print
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template, DATASETS
from .model import DFlashDraftModel, dflash_generate, sample, extract_context_feature
from .monitor import LiveMonitor
from .ssd_dflash_cuda import AcceptancePredictor, dflash_ssd_generate


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def _get_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        logger.info("flash_attention_2 enabled")
        return "flash_attention_2"
    except ImportError:
        logger.warning("flash-attn not found — falling back to sdpa. "
                       "Run setup_ssd_dflash.sh to install it.")
        return "sdpa"


def _load_target(model_id: str, device: str, attn_impl: str) -> tuple:
    logger.info(f"Loading target: {model_id} → {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def _load_draft(draft_id: str, device: str, attn_impl: str) -> DFlashDraftModel:
    logger.info(f"Loading DFlash draft: {draft_id} → {device}")
    return DFlashDraftModel.from_pretrained(
        draft_id, attn_implementation=attn_impl, dtype=torch.bfloat16
    ).to(device).eval()


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def _run_ar(target, tokenizer, input_ids, args) -> SimpleNamespace:
    """Pure autoregressive baseline."""
    from .model import dflash_generate
    return dflash_generate(
        None, target=target, input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
        block_size=1, mask_token_id=0,
        return_stats=True,
    )


@torch.inference_mode()
def _run_dflash(draft, target, tokenizer, input_ids, args) -> SimpleNamespace:
    """Standard DFlash (sequential, same device)."""
    return dflash_generate(
        draft, target=target, input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
        return_stats=True,
    )


# ─────────────────────────────────────────────────────────
# Per-sample benchmark runner
# ─────────────────────────────────────────────────────────

def _run_one_sample(
    target, draft_sequential, draft_ssd,
    tokenizer, item: dict, args,
    ar_tps_ref: float, dflash_tps_ref: float,
) -> dict:
    """Run all three modes on a single sample and return a metrics dict."""
    messages = []
    results = {}

    for turn_text in item["turns"]:
        messages.append({"role": "user", "content": turn_text})
        prompt = _apply_chat_template(tokenizer, messages, enable_thinking=False)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(f"cuda:{args.target_gpu}")

        # ── AR baseline ──────────────────────────────────
        t0 = _cuda_time()
        ar_r = _run_ar(target, tokenizer, input_ids, args)
        ar_wall = _cuda_time() - t0
        ar_tps = ar_r.num_output_tokens / max(ar_wall, 1e-6)

        # ── Standard DFlash ──────────────────────────────
        t0 = _cuda_time()
        df_r = _run_dflash(draft_sequential, target, tokenizer, input_ids, args)
        df_wall = _cuda_time() - t0
        df_tps = df_r.num_output_tokens / max(df_wall, 1e-6)

        # ── DFlash-SSD ────────────────────────────────────
        predictor = AcceptancePredictor(
            block_size=draft_ssd.block_size,
            strategy=args.acceptance_strategy,
        )
        log_path = Path(args.out_dir) / f"steps_{args.dataset}.jsonl"
        monitor = LiveMonitor(
            model_name=args.model,
            draft_name=args.draft_model,
            block_size=draft_ssd.block_size,
            fan_out=args.fan_out,
            ar_tps=ar_tps_ref,
            dflash_tps=dflash_tps_ref,
        )
        with monitor:
            t0 = time.perf_counter()
            ssd_r = dflash_ssd_generate(
                draft_model=draft_ssd,
                target=target,
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                fan_out=args.fan_out,
                acceptance_predictor=predictor,
                fast_refine=args.fast_refine,
                return_stats=True,
                monitor=monitor,
                log_path=str(log_path),
            )
            ssd_wall = time.perf_counter() - t0
        ssd_tps = ssd_r.num_output_tokens / max(ssd_wall, 1e-6)

        generated_ids = ssd_r.output_ids[0, ssd_r.num_input_tokens:]
        output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": output_text})

        results = {
            "ar_tps":   ar_tps,
            "ar_tpot":  ar_r.time_per_output_token,
            "df_tps":   df_tps,
            "df_tpot":  df_r.time_per_output_token,
            "df_mean_accept": float(np.mean(df_r.acceptance_lengths)),
            "ssd_tps":  ssd_tps,
            "ssd_tpot": ssd_r.time_per_output_token,
            "ssd_mean_accept":  float(np.mean(ssd_r.acceptance_lengths)),
            "ssd_cache_hit_rate":        getattr(ssd_r, "cache_hit_rate", 0.0),
            "ssd_mean_h_cosine_sim":     getattr(ssd_r, "mean_h_cosine_sim", 0.0),
            "ssd_draft_latency_saved_ms":getattr(ssd_r, "draft_latency_saved_ms", 0.0),
            "speedup_ssd_vs_ar":    ssd_tps / max(ar_tps, 1e-6),
            "speedup_ssd_vs_dflash":ssd_tps / max(df_tps, 1e-6),
            "speedup_df_vs_ar":     df_tps  / max(ar_tps, 1e-6),
            "num_output_tokens":    ssd_r.num_output_tokens,
        }

    return results


# ─────────────────────────────────────────────────────────
# Dataset benchmark
# ─────────────────────────────────────────────────────────

def _run_dataset(target, draft_seq, draft_ssd, tokenizer, args) -> dict:
    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    all_results = []
    ar_tps_ref = dflash_tps_ref = 0.0

    # Compute reference throughputs from first sample for monitor display
    item0 = dataset[0]
    prompt0 = _apply_chat_template(
        tokenizer,
        [{"role": "user", "content": item0["turns"][0]}],
        enable_thinking=False,
    )
    ids0 = tokenizer.encode(prompt0, return_tensors="pt").to(f"cuda:{args.target_gpu}")
    r_ar = _run_ar(target, tokenizer, ids0, args)
    r_df = _run_dflash(draft_seq, target, tokenizer, ids0, args)
    ar_tps_ref  = r_ar.num_output_tokens / max(r_ar.time_per_output_token * r_ar.num_output_tokens, 1e-6)
    dflash_tps_ref = r_df.num_output_tokens / max(r_df.time_per_output_token * r_df.num_output_tokens, 1e-6)

    logger.info(f"Reference AR={ar_tps_ref:.1f} tok/s  DFlash={dflash_tps_ref:.1f} tok/s")

    for item in tqdm(dataset, desc=f"DFlash-SSD  {args.dataset}"):
        r = _run_one_sample(
            target, draft_seq, draft_ssd,
            tokenizer, item, args,
            ar_tps_ref=ar_tps_ref,
            dflash_tps_ref=dflash_tps_ref,
        )
        r["dataset"] = args.dataset
        all_results.append(r)

        # Save JSONL incrementally
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"results_{args.dataset}.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")

    return _aggregate(all_results)


def _aggregate(results: list[dict]) -> dict:
    def _mean(key):
        vals = [r[key] for r in results if key in r]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "n_samples": len(results),
        "ar_tps_mean":           _mean("ar_tps"),
        "dflash_tps_mean":       _mean("df_tps"),
        "ssd_tps_mean":          _mean("ssd_tps"),
        "speedup_ssd_vs_ar":     _mean("speedup_ssd_vs_ar"),
        "speedup_ssd_vs_dflash": _mean("speedup_ssd_vs_dflash"),
        "speedup_df_vs_ar":      _mean("speedup_df_vs_ar"),
        "ssd_cache_hit_rate":    _mean("ssd_cache_hit_rate"),
        "ssd_mean_accept":       _mean("ssd_mean_accept"),
        "df_mean_accept":        _mean("df_mean_accept"),
        "ssd_mean_h_cosine_sim": _mean("ssd_mean_h_cosine_sim"),
        "ssd_draft_saved_ms":    _mean("ssd_draft_latency_saved_ms"),
    }


# ─────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────

def _print_summary(all_agg: dict[str, dict], model: str, draft: str, fan_out: int):
    print()
    print(f"[bold]DFlash-SSD Results[/bold]  model={model}  draft={draft}  F={fan_out}")
    print()

    # Per-dataset table
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Dataset")
    tbl.add_column("AR tok/s",   justify="right")
    tbl.add_column("DFlash tok/s", justify="right")
    tbl.add_column("SSD tok/s",  justify="right")
    tbl.add_column("SSD/AR",     justify="right")
    tbl.add_column("SSD/DFlash", justify="right")
    tbl.add_column("Hit%",       justify="right")
    tbl.add_column("H_sim",      justify="right")
    tbl.add_column("Accept↗",    justify="right")

    for ds, agg in all_agg.items():
        speedup_color = "green" if agg["speedup_ssd_vs_dflash"] > 1.05 else (
            "yellow" if agg["speedup_ssd_vs_dflash"] > 0.95 else "red")
        tbl.add_row(
            ds,
            f"{agg['ar_tps_mean']:.1f}",
            f"{agg['dflash_tps_mean']:.1f}",
            f"{agg['ssd_tps_mean']:.1f}",
            f"{agg['speedup_ssd_vs_ar']:.2f}x",
            f"[{speedup_color}]{agg['speedup_ssd_vs_dflash']:.2f}x[/{speedup_color}]",
            f"{agg['ssd_cache_hit_rate']*100:.0f}%",
            f"{agg['ssd_mean_h_cosine_sim']:.3f}",
            f"{agg['ssd_mean_accept']:.1f}",
        )
    print(tbl)

    # Overall mean
    macro = {}
    for k in next(iter(all_agg.values())):
        vals = [v[k] for v in all_agg.values() if isinstance(v.get(k), float)]
        macro[k] = float(np.mean(vals)) if vals else 0.0

    print(
        f"\n[bold]OVERALL[/bold]  "
        f"AR={macro['ar_tps_mean']:.1f}  "
        f"DFlash={macro['dflash_tps_mean']:.1f}  "
        f"[bold cyan]DFlash-SSD={macro['ssd_tps_mean']:.1f}[/bold cyan]  "
        f"tok/s\n"
        f"  SSD/AR     = [bold]{macro['speedup_ssd_vs_ar']:.2f}x[/bold]\n"
        f"  SSD/DFlash = [bold]{macro['speedup_ssd_vs_dflash']:.2f}x[/bold]\n"
        f"  Cache hit  = {macro['ssd_cache_hit_rate']*100:.0f}%\n"
        f"  H cos_sim  = {macro['ssd_mean_h_cosine_sim']:.4f}\n"
        f"  Draft saved= {macro['ssd_draft_saved_ms']:.1f} ms/step"
    )


# ─────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="DFlash-SSD benchmark")
    p.add_argument("--model",       required=True,  help="Target model (Qwen/Qwen3-4B etc.)")
    p.add_argument("--draft-model", required=True,  help="DFlash draft model")
    p.add_argument("--dataset",     required=True,  choices=list(DATASETS),
                   action="append", dest="datasets",
                   help="Dataset(s) to benchmark (can repeat)")
    p.add_argument("--max-samples",   type=int,   default=64)
    p.add_argument("--max-new-tokens",type=int,   default=2048)
    p.add_argument("--temperature",   type=float, default=0.0)
    p.add_argument("--fan-out",  "-F",type=int,   default=2,
                   help="Fan-out F: number of blocks to pre-speculate (default 2)")
    p.add_argument("--acceptance-strategy", default="running",
                   choices=["running", "top_high"],
                   help="How to predict likely acceptance lengths for fan-out")
    p.add_argument("--fast-refine", action="store_true",
                   help="After parallel verify, run one extra DFlash pass with fresh H_t. "
                        "Fixes acceptance on tasks where H_SIM < 0.85 (e.g. math reasoning). "
                        "Adds ~T_draft latency but restores standard DFlash acceptance quality.")
    p.add_argument("--target-gpu", type=int, default=0,
                   help="GPU index for target model (default 0)")
    p.add_argument("--draft-gpu",  type=int, default=1,
                   help="GPU index for DFlash draft (default 1); use 0 for single-GPU")
    p.add_argument("--out-dir", default="results/ssd_dflash",
                   help="Directory for JSONL logs and summary")
    args = p.parse_args()

    attn_impl = _get_attn_impl()
    target_dev = f"cuda:{args.target_gpu}"
    draft_dev_ssd = f"cuda:{args.draft_gpu}"

    # Load target once
    target, tokenizer = _load_target(args.model, target_dev, attn_impl)

    # Sequential DFlash draft (same device as target, for fair comparison)
    draft_seq = _load_draft(args.draft_model, target_dev, attn_impl)

    # DFlash-SSD draft (separate GPU)
    if args.target_gpu == args.draft_gpu:
        logger.warning("target-gpu == draft-gpu: DFlash-SSD runs sequentially (no parallelism). "
                       "For full speedup use --target-gpu 0 --draft-gpu 1.")
    draft_ssd = _load_draft(args.draft_model, draft_dev_ssd, attn_impl)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_agg: dict[str, dict] = {}
    for ds in args.datasets:
        args.dataset = ds
        print(f"\n[bold]── Dataset: {ds} ──[/bold]")
        agg = _run_dataset(target, draft_seq, draft_ssd, tokenizer, args)
        all_agg[ds] = agg
        # Save per-dataset aggregate
        (out_dir / f"agg_{ds}.json").write_text(json.dumps(agg, indent=2))

    _print_summary(all_agg, args.model, args.draft_model, args.fan_out)

    # Save full summary
    summary = {"model": args.model, "draft_model": args.draft_model,
               "fan_out": args.fan_out, "datasets": all_agg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nAll results saved → {out_dir}/")


if __name__ == "__main__":
    main()
