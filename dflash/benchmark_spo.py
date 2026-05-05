"""
SPO (Speculative Policy Optimization) benchmark CLI.

Phases:
  train — pre-generate target completions, then train SPO draft with Rich TUI.
  eval  — run AR baseline, SPO speculative decoding, and optionally DFlash.
  all   — train then eval.

Usage:
    # Train from scratch:
    python -m dflash.benchmark_spo --model Qwen/Qwen3-8B --phase train \\
        --dataset gsm8k --max-samples 128 --train-steps 2000

    # Evaluate a trained checkpoint:
    python -m dflash.benchmark_spo --model Qwen/Qwen3-8B --phase eval \\
        --checkpoint-dir checkpoints/spo --max-new-tokens 512

    # Full pipeline:
    python -m dflash.benchmark_spo --model Qwen/Qwen3-8B --phase all
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template
from .spo import (
    SPOConfig, SPOTrainConfig, SPODraft, SPOTrainer,
    spo_generate, pregenerate_targets, load_pregenerated, load_spo_draft,
)
from .spo_monitor import (
    make_train_monitor, make_infer_monitor,
    SPOInferMonitor,
)

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(model_name: str, device: torch.device):
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        console.print("[yellow]flash_attn not found — using sdpa[/yellow]")
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
# AR baseline runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_ar(model, input_ids, args) -> dict:
    from .jacobi import ar_generate
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
        "mean_accept": 1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPO generation runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_spo(spo_draft, model, input_ids, args,
             monitor: SPOInferMonitor | None = None) -> dict:
    r = spo_generate(
        spo_draft, model, input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[model.config.eos_token_id],
        temperature=args.temperature,
        block_size=args.block_size,
        return_stats=True,
    )

    if monitor is not None:
        for s in r.step_stats:
            monitor.update(s)

    mean_accept = float(np.mean(r.acceptance_lengths)) if r.acceptance_lengths else 1.0
    tps = r.num_output_tokens / (r.time_per_output_token * r.num_output_tokens) \
        if r.time_per_output_token > 0 else 0.0

    return {
        "tps": tps,
        "ttft": r.time_to_first_token,
        "num_out": r.num_output_tokens,
        "mean_accept": mean_accept,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training phase
# ─────────────────────────────────────────────────────────────────────────────

def _phase_train(model, tokenizer, args) -> Path:
    device = next(model.parameters()).device

    # Pre-generate target completions
    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    data_path = Path(args.checkpoint_dir) / "pregenerated.jsonl"
    console.print(f"[cyan]Pre-generating[/cyan] target completions → {data_path}")
    pregenerate_targets(
        model, tokenizer, dataset,
        max_new_tokens=args.max_new_tokens,
        output_path=data_path,
        enable_thinking=args.enable_thinking,
    )

    sequences = load_pregenerated(data_path)
    console.print(f"[green]Loaded[/green] {len(sequences)} sequences "
                  f"(mean len {np.mean([len(s) for s in sequences]):.0f} tokens)")

    # Build draft and trainer
    spo_config = SPOConfig.from_target(
        model,
        hidden_dim=args.hidden_dim,
        block_size=args.block_size,
        context_window=args.context_window,
    )
    draft = SPODraft(spo_config).bind(model)
    console.print(f"[green]SPO draft[/green]: {draft.num_params / 1e6:.1f}M params")

    train_config = SPOTrainConfig(
        lr=args.lr,
        train_steps=args.train_steps,
        temperature=args.train_temperature,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=args.checkpoint_dir,
        entropy_coef=args.entropy_coef,
    )
    trainer = SPOTrainer(draft, train_config)

    # Resume if checkpoint exists
    ckpt_path = Path(args.checkpoint_dir) / "spo_checkpoint.pt"
    if ckpt_path.exists():
        console.print(f"[yellow]Resuming[/yellow] from step {trainer.global_step}")
        trainer.load_checkpoint(args.checkpoint_dir)

    # Train with monitor
    state, mon = make_train_monitor(total_steps=args.train_steps)

    with mon:
        for step_idx in range(trainer.global_step, args.train_steps):
            seq = sequences[step_idx % len(sequences)].unsqueeze(0).to(device)
            train_step = trainer.train_step(seq)
            mon.update(train_step)

            if train_step.step % train_config.checkpoint_every == 0:
                trainer.save_checkpoint(args.checkpoint_dir)

    trainer.save_checkpoint(args.checkpoint_dir)
    console.print(f"\n[bold green]Training complete.[/bold green] "
                  f"Final reward: {state.mean_reward:.2f}  "
                  f"Checkpoint: {args.checkpoint_dir}")
    return Path(args.checkpoint_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation phase
# ─────────────────────────────────────────────────────────────────────────────

def _phase_eval(model, tokenizer, args) -> None:
    device = next(model.parameters()).device

    # Load SPO draft
    console.print(f"[cyan]Loading SPO checkpoint[/cyan] from {args.checkpoint_dir}")
    spo_draft = load_spo_draft(args.checkpoint_dir, model)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    # Warmup
    console.print("[dim]Warmup …[/dim]")
    warmup_ids = tokenizer.encode("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        from .jacobi import ar_generate
        ar_generate(model, warmup_ids, max_new_tokens=8, stop_token_ids=None, temperature=0.0)

    # Run AR baseline
    console.print("[dim]Running AR baseline …[/dim]")
    ar_results = []
    for instance in tqdm(dataset[:min(32, len(dataset))], desc="AR baseline"):
        messages = [{"role": "user", "content": instance["turns"][0]}]
        input_text = _apply_chat_template(tokenizer, messages, args.enable_thinking)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            ar_results.append(_run_ar(model, input_ids, args))

    ar_tps = float(np.mean([r["tps"] for r in ar_results])) if ar_results else 0.0
    console.print(f"[dim]AR baseline: {ar_tps:.1f} tok/s[/dim]")

    # Run SPO with monitor
    infer_state, infer_mon = make_infer_monitor(
        block_size=args.block_size,
        model_name=args.model.split("/")[-1],
        ar_tps=ar_tps,
    )

    spo_results = []
    with infer_mon:
        for instance in tqdm(dataset, desc="SPO eval"):
            messages = [{"role": "user", "content": instance["turns"][0]}]
            input_text = _apply_chat_template(tokenizer, messages, args.enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
            with torch.no_grad():
                spo_results.append(_run_spo(spo_draft, model, input_ids, args, infer_mon))

    # Summary table
    _print_summary(ar_results, spo_results, ar_tps, args)


def _agg(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if r]
    return float(np.mean(vals)) if vals else 0.0


def _print_summary(ar_results, spo_results, ar_tps, args) -> None:
    t = Table(title="[bold]SPO Benchmark Summary[/bold]", show_lines=True)
    t.add_column("Strategy", style="bold", min_width=10)
    t.add_column("TPS", justify="right", min_width=8)
    t.add_column("Speedup", justify="right", min_width=8)
    t.add_column("TTFT (ms)", justify="right", min_width=10)
    t.add_column("Mean Accept", justify="right", min_width=12)
    t.add_column("Tokens/sample", justify="right", min_width=13)

    rows = [
        ("AR", ar_results),
        ("SPO", spo_results),
    ]

    for name, results in rows:
        if not results:
            continue
        tps = _agg(results, "tps")
        ttft = _agg(results, "ttft") * 1000.0
        accept = _agg(results, "mean_accept")
        n_out = _agg(results, "num_out")
        spd = tps / ar_tps if ar_tps > 0 else 0.0

        if spd >= 1.5:
            spd_s = f"[bold green]{spd:.2f}×[/bold green]"
        elif spd >= 1.0:
            spd_s = f"[green]{spd:.2f}×[/green]"
        else:
            spd_s = f"[red]{spd:.2f}×[/red]"

        t.add_row(
            name,
            f"{tps:.1f}",
            spd_s,
            f"{ttft:.1f}",
            f"{accept:.2f}",
            f"{n_out:.0f}",
        )

    console.print()
    console.print(t)
    console.print(f"\n[dim]Config: block_size={args.block_size}, "
                  f"hidden_dim={args.hidden_dim}, "
                  f"temperature={args.temperature}[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.device}")
    model, tokenizer = _load_model(args.model, device)

    if args.phase in ("train", "all"):
        _phase_train(model, tokenizer, args)

    if args.phase in ("eval", "all"):
        _phase_eval(model, tokenizer, args)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPO benchmark: train and evaluate RL-optimized speculative draft",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="HuggingFace model name or path")
    p.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    p.add_argument("--dataset", default="gsm8k",
                   choices=["gsm8k", "math500", "humaneval", "mbpp", "mt-bench",
                            "alpaca", "ultrafeedback"])
    p.add_argument("--max-samples", type=int, default=128)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--context-window", type=int, default=64)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--train-temperature", type=float, default=0.8,
                   help="Sampling temperature during REINFORCE training")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature during eval generation")
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--checkpoint-dir", default="checkpoints/spo")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    parser = _build_parser()
    main(parser.parse_args())
