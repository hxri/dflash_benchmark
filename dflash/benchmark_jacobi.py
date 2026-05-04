"""
Jacobi decoding benchmark for CUDA (A6000 / any HuggingFace-compatible GPU).

Compares AR baseline against Jacobi variants on standard datasets.
Includes a live Rich TUI during the "best" strategy run, then prints a final
summary table with all metrics side-by-side.

Usage:
    python -m dflash.benchmark_jacobi \\
        --model Qwen/Qwen3-8B \\
        --dataset gsm8k \\
        --max-samples 128 \\
        --max-new-tokens 512 \\
        --block-size 16 \\
        --max-iters 10 \\
        --temperature 0.0

    # Run only specific strategies (skip others):
    python -m dflash.benchmark_jacobi --model Qwen/Qwen3-8B \\
        --strategies ar ngram best

    # Live monitor on a single long generation:
    python -m dflash.benchmark_jacobi --model Qwen/Qwen3-8B \\
        --live-demo --max-new-tokens 2048
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template
from .jacobi import jacobi_generate, ar_generate, JacobiRunStats
from .jacobi_monitor import JacobiMonitorState, JacobiMonitor, make_monitor


console = Console()

STRATEGIES = ["repeat", "ngram", "context", "best"]
ALL_STRATEGIES = ["ar"] + STRATEGIES


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_name: str, device: torch.device):
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        console.print("[yellow]flash_attn not found — using sdpa (slightly slower)[/yellow]")
        attn_impl = "sdpa"

    console.print(f"[cyan]Loading[/cyan] {model_name} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_ar(model, input_ids, args) -> dict:
    r = ar_generate(
        model, input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[model.config.eos_token_id],
        temperature=args.temperature,
    )
    return {
        "tps": 1.0 / r.time_per_output_token if r.time_per_output_token > 0 else 0.0,
        "ttft": r.time_to_first_token,
        "num_out": r.num_output_tokens,
        "mean_iters": 1.0,
        "mean_util": 1.0 / args.block_size,
        "ngram_hit_rate": 0.0,
        "ctx_hit_rate": 0.0,
        "one_shot_rate": 1.0,
        "convergence_rate": 1.0 / args.block_size,
    }


def _run_jacobi(model, input_ids, args, strategy: str,
                monitor: Optional[JacobiMonitor] = None) -> dict:
    r = jacobi_generate(
        model, input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[model.config.eos_token_id],
        temperature=args.temperature,
        block_size=args.block_size,
        max_iters=args.max_iters,
        init_strategy=strategy,
        ngram_n=args.ngram_n,
        context_match_n=args.context_match_n,
        return_stats=True,
    )
    rs: JacobiRunStats = r.run_stats

    if monitor is not None:
        for s in rs.steps:
            monitor.update(s)

    return {
        "tps": rs.generation_tps,
        "ttft": rs.time_to_first_token_s,
        "num_out": rs.num_output_tokens,
        "mean_iters": rs.mean_iters,
        "mean_util": rs.mean_utilization,
        "ngram_hit_rate": rs.ngram_hit_rate,
        "ctx_hit_rate": rs.ctx_hit_rate,
        "one_shot_rate": rs.one_shot_rate,
        "convergence_rate": rs.convergence_rate,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agg(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if r]
    return float(np.mean(vals)) if vals else 0.0


def _print_summary(all_results: dict[str, list[dict]], ar_tps: float, args) -> None:
    t = Table(title="[bold]Jacobi Decoding Benchmark Summary[/bold]", show_lines=True)
    t.add_column("Strategy",       style="bold", min_width=10)
    t.add_column("TPS",            justify="right", min_width=8)
    t.add_column("Speedup",        justify="right", min_width=8)
    t.add_column("TTFT (ms)",      justify="right", min_width=10)
    t.add_column("Mean iters",     justify="right", min_width=10)
    t.add_column("Util %",         justify="right", min_width=8)
    t.add_column("1-shot %",       justify="right", min_width=9)
    t.add_column("NGram hit%",     justify="right", min_width=10)
    t.add_column("Ctx hit%",       justify="right", min_width=9)
    t.add_column("Conv%",          justify="right", min_width=7)

    for strategy, results in all_results.items():
        if not results:
            continue
        tps  = _agg(results, "tps")
        ttft = _agg(results, "ttft") * 1000.0
        mi   = _agg(results, "mean_iters")
        mu   = _agg(results, "mean_util") * 100.0
        os_  = _agg(results, "one_shot_rate") * 100.0
        ng   = _agg(results, "ngram_hit_rate") * 100.0
        cx   = _agg(results, "ctx_hit_rate") * 100.0
        cv   = _agg(results, "convergence_rate") * 100.0
        spd  = tps / ar_tps if ar_tps > 0 else 0.0

        if spd >= 1.5:
            spd_s = f"[bold green]{spd:.2f}×[/bold green]"
        elif spd >= 1.0:
            spd_s = f"[green]{spd:.2f}×[/green]"
        else:
            spd_s = f"[red]{spd:.2f}×[/red]"

        t.add_row(
            strategy,
            f"{tps:.1f}",
            spd_s,
            f"{ttft:.1f}",
            f"{mi:.2f}",
            f"{mu:.1f}",
            f"{os_:.1f}",
            f"{ng:.1f}",
            f"{cx:.1f}",
            f"{cv:.1f}",
        )

    console.print(t)
    console.print()
    console.print(f"[dim]Config: block_size={args.block_size}, max_iters={args.max_iters}, "
                  f"ngram_n={args.ngram_n}, context_match_n={args.context_match_n}, "
                  f"temperature={args.temperature}[/dim]")


def _print_iteration_dist(all_results: dict[str, list[dict]]) -> None:
    """Per-strategy histogram of iterations used is printed from step_stats."""
    pass  # Aggregated at step_stats level; the monitor shows this live.


# ─────────────────────────────────────────────────────────────────────────────
# Live demo (single long prompt, rich TUI)
# ─────────────────────────────────────────────────────────────────────────────

def _live_demo(model, tokenizer, args) -> None:
    prompt_text = (
        "Explain the entire history of the Roman Empire from its founding to its fall, "
        "covering politics, military, culture, religion, and economics in detail."
    )
    messages = [{"role": "user", "content": prompt_text}]
    input_text = _apply_chat_template(tokenizer, messages, enable_thinking=False)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(next(model.parameters()).device)

    console.print(f"[cyan]Live demo[/cyan]: Jacobi-best, block={args.block_size}, "
                  f"max_iters={args.max_iters}, max_new_tokens={args.max_new_tokens}")

    # AR warmup to get baseline TPS
    console.print("[dim]Warming up AR baseline …[/dim]")
    ar_r = ar_generate(model, input_ids, max_new_tokens=64,
                       stop_token_ids=[model.config.eos_token_id], temperature=0.0)
    ar_tps = 1.0 / ar_r.time_per_output_token if ar_r.time_per_output_token > 0 else 0.0
    console.print(f"[dim]AR baseline: {ar_tps:.1f} tok/s[/dim]")

    state, mon = make_monitor(
        block_size=args.block_size,
        max_iters=args.max_iters,
        init_strategy="best",
        model_name=args.model.split("/")[-1],
        ar_tps=ar_tps,
    )

    with mon:
        _run_jacobi(model, input_ids, args, "best", monitor=mon)

    console.print(f"\n[bold]Live demo complete.[/bold]  "
                  f"Mean speedup: [green]{state.speedup:.2f}×[/green]  "
                  f"1-shot: [cyan]{state.one_shot_rate * 100:.1f}%[/cyan]")


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.device}")
    model, tokenizer = _load_model(args.model, device)

    if args.live_demo:
        _live_demo(model, tokenizer, args)
        return

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    strategies_to_run = args.strategies if args.strategies else ALL_STRATEGIES
    all_results: dict[str, list[dict]] = {s: [] for s in strategies_to_run}

    # Warmup
    console.print("[dim]Warmup pass …[/dim]")
    warmup_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        ar_generate(model, warmup_ids, max_new_tokens=8,
                    stop_token_ids=None, temperature=0.0)

    ar_tps = 0.0

    for idx, instance in enumerate(tqdm(dataset, desc="Samples")):
        messages = []
        for turn in instance["turns"]:
            messages.append({"role": "user", "content": turn})
        input_text = _apply_chat_template(tokenizer, messages,
                                          enable_thinking=args.enable_thinking)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        for strategy in strategies_to_run:
            with torch.no_grad():
                if strategy == "ar":
                    r = _run_ar(model, input_ids, args)
                else:
                    r = _run_jacobi(model, input_ids, args, strategy)
            all_results[strategy].append(r)

        # Track AR TPS for speedup calculation.
        if "ar" in strategies_to_run and all_results["ar"]:
            ar_tps = _agg(all_results["ar"], "tps")

    console.print()
    _print_summary(all_results, ar_tps, args)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark Jacobi decoding variants vs AR on CUDA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HuggingFace model name or path")
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench",
                            "alpaca", "ultrafeedback"])
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16,
                   help="Number of tokens per Jacobi block (B)")
    p.add_argument("--max-iters", type=int, default=10,
                   help="Max Jacobi iterations per block before forcing acceptance")
    p.add_argument("--ngram-n", type=int, default=4,
                   help="N-gram order for the n-gram cache init strategy")
    p.add_argument("--context-match-n", type=int, default=3,
                   help="Tokens to match for context-search init strategy")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true",
                   help="Apply thinking chat template (Qwen3 thinking mode)")
    p.add_argument("--strategies", nargs="+", choices=ALL_STRATEGIES,
                   default=None, help="Strategies to evaluate (default: all)")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--live-demo", action="store_true",
                   help="Run a single long generation with live Rich monitor instead of benchmark")
    return p


if __name__ == "__main__":
    parser = _build_parser()
    main(parser.parse_args())
