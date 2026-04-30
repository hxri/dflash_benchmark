"""
Phase-0 analysis: measure T_verify vs T_draft_dflash ratio.

Determines whether the DFlash draft fits inside the target verify window,
and computes the maximum fan-out F that can be hidden for free.

Usage:
  python -m dflash.analysis.timing_profile \
      --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 \
      --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from loguru import logger
from rich import print
from rich.table import Table

from ..model import DFlashDraftModel, extract_context_feature, sample


def _load_models(model_id, draft_id, device, attn_impl):
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    target = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        draft_id, attn_implementation=attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return target, draft, tokenizer


def _cuda_time_ms(device) -> float:
    torch.cuda.synchronize(device)
    return torch.cuda.Event(enable_timing=True)


@torch.inference_mode()
def _time_one(target, draft, input_ids, seq_len: int, block_size: int, n_reps: int = 10):
    """Returns (mean_verify_ms, mean_draft_ms) at a given sequence length."""
    from transformers import DynamicCache

    device = input_ids.device
    mask_token_id = draft.mask_token_id

    # Build a fake cached state at seq_len tokens
    fake_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    target_cache = DynamicCache()
    with torch.no_grad():
        out = target(fake_ids, past_key_values=target_cache, use_cache=True,
                     output_hidden_states=True, logits_to_keep=1)
    H = extract_context_feature(out.hidden_states, draft.target_layer_ids)

    block_ids = torch.cat([fake_ids[:, -1:],
                           torch.full((1, block_size - 1), mask_token_id, device=device)], dim=1)
    noise = target.model.embed_tokens(block_ids)
    pos = torch.arange(seq_len, seq_len + block_size, device=device).unsqueeze(0)

    verify_times, draft_times = [], []

    for _ in range(n_reps + 2):  # +2 warmup
        # Time verify
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        verify_cache = DynamicCache()
        # copy existing cache
        for lk, lv in zip(target_cache.key_cache, target_cache.value_cache):
            verify_cache.key_cache.append(lk.clone())
            verify_cache.value_cache.append(lv.clone())
        start_ev.record()
        target(block_ids, position_ids=pos, past_key_values=verify_cache, use_cache=True)
        end_ev.record()
        torch.cuda.synchronize(device)
        if _ >= 2:
            verify_times.append(start_ev.elapsed_time(end_ev))

        # Time draft
        start_ev2 = torch.cuda.Event(enable_timing=True)
        end_ev2 = torch.cuda.Event(enable_timing=True)
        draft_cache_tmp = DynamicCache()
        start_ev2.record()
        target.lm_head(draft(
            target_hidden=H,
            noise_embedding=noise,
            position_ids=pos,
            past_key_values=draft_cache_tmp,
            use_cache=True,
            is_causal=False,
        )[:, 1 - block_size:, :])
        end_ev2.record()
        torch.cuda.synchronize(device)
        if _ >= 2:
            draft_times.append(start_ev2.elapsed_time(end_ev2))

    import statistics
    return statistics.mean(verify_times), statistics.mean(draft_times)


def profile(args) -> dict:
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    device = f"cuda:{args.gpu}"
    target, draft, tokenizer = _load_models(args.model, args.draft_model, device, attn_impl)

    seq_lengths = [256, 512, 1024, 2048]
    block_size = draft.block_size

    table = Table(title=f"Timing Profile  (block_size={block_size})", show_header=True)
    table.add_column("Seq len")
    table.add_column("T_verify (ms)", justify="right")
    table.add_column("T_draft  (ms)", justify="right")
    table.add_column("Ratio V/D", justify="right")
    table.add_column("Max fan-out F", justify="right")

    rows = []
    prompt = tokenizer.encode("The quick brown fox", return_tensors="pt").to(device)

    for slen in seq_lengths:
        if slen < prompt.shape[1]:
            continue
        t_v, t_d = _time_one(target, draft, prompt, slen, block_size, n_reps=args.reps)
        ratio = t_v / t_d if t_d > 0 else float("inf")
        max_f = int(ratio)
        rows.append({"seq_len": slen, "t_verify_ms": t_v, "t_draft_ms": t_d,
                     "ratio": ratio, "max_fanout": max_f})
        table.add_row(str(slen), f"{t_v:.1f}", f"{t_d:.1f}", f"{ratio:.1f}x", str(max_f))

    print(table)

    avg_ratio = sum(r["ratio"] for r in rows) / len(rows) if rows else 0
    rec_f = max(1, min(4, int(avg_ratio)))
    print(f"\nRecommended fan-out F = [bold]{rec_f}[/bold]  (avg ratio = {avg_ratio:.1f}x)")

    if avg_ratio >= 3:
        print("[green]GREAT[/green] — T_verify >> T_draft; high fan-out easily hides draft latency")
    elif avg_ratio >= 1.5:
        print("[yellow]GOOD[/yellow] — T_verify > T_draft; F=1–2 recommended")
    else:
        print("[orange1]TIGHT[/orange1] — T_verify ≈ T_draft; F=1 only; consider smaller block_size")

    results = {"model": args.model, "block_size": block_size,
               "rows": rows, "recommended_fanout": rec_f}
    out_path = Path(args.out_dir) / "timing_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="Phase-0: T_verify vs T_draft timing profile")
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--out-dir", default="results/analysis")
    profile(p.parse_args())


if __name__ == "__main__":
    main()
