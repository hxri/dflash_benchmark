"""
Phase-0 analysis: acceptance rate of DFlash with stale vs fresh hidden states.

For each DFlash step, compares:
  - fresh_draft  = DFlash(H_t,   [last_tok, MASK...])  — standard
  - stale_draft  = DFlash(H_t-1, [last_tok, MASK...])  — one step stale

Reports token overlap and predicted acceptance degradation.

Usage:
  python -m dflash.analysis.stale_draft_quality \
      --model Qwen/Qwen3-4B \
      --draft-model z-lab/Qwen3-4B-DFlash-b16 \
      --dataset gsm8k --max-samples 32 --gpu 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from rich import print
from rich.table import Table
from tqdm import tqdm

from ..benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template, DATASETS
from ..model import DFlashDraftModel, extract_context_feature, sample


def _load_models(model_id, draft_id, device, attn_impl):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    target = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        draft_id, attn_implementation=attn_impl, dtype=torch.bfloat16
    ).to(device).eval()
    return target, draft, AutoTokenizer.from_pretrained(model_id)


@torch.inference_mode()
def _run_stale_comparison(target, draft, input_ids, max_new_tokens, temperature):
    """
    Returns per-step overlap: fraction of DFlash draft tokens that are identical
    when using H_t (fresh) vs H_{t-1} (stale).
    """
    from transformers import DynamicCache

    block_size = draft.block_size
    mask_token_id = draft.mask_token_id
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens
    device = input_ids.device

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    target_cache = DynamicCache()
    draft_cache_fresh = DynamicCache()
    draft_cache_stale = DynamicCache()

    # Prefill
    out = target(
        input_ids,
        position_ids=position_ids[:, :num_input],
        past_key_values=target_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input] = input_ids
    output_ids[:, num_input] = sample(out.logits, temperature)
    H_fresh = extract_context_feature(out.hidden_states, draft.target_layer_ids)
    H_stale = H_fresh  # same at first step

    start = num_input
    overlaps: list[float] = []
    acceptance_fresh: list[int] = []
    acceptance_stale_sim: list[int] = []  # simulated: how many stale tokens match target posterior

    while start < max_length:
        bs = min(block_size, max_length - start)
        if bs <= 1:
            break

        block = output_ids[:, start: start + bs].clone()
        noise = target.model.embed_tokens(block)

        # Position IDs must span ctx_len + bs tokens (not just bs).
        # Use ctx_start = start - H.shape[1] so the formula works even when
        # H_stale is intentionally misaligned (one step behind H_fresh).
        ctx_fresh = H_fresh.shape[1]
        ctx_stale = H_stale.shape[1]
        pos_fresh = position_ids[:, start - ctx_fresh : start - ctx_fresh + ctx_fresh + bs]
        pos_stale = position_ids[:, start - ctx_stale : start - ctx_stale + ctx_stale + bs]

        # Fresh draft
        fresh_logits = target.lm_head(draft(
            target_hidden=H_fresh,
            noise_embedding=noise,
            position_ids=pos_fresh,
            past_key_values=draft_cache_fresh,
            use_cache=True,
            is_causal=False,
        )[:, 1 - bs:, :])
        draft_cache_fresh.crop(start)
        fresh_tokens = sample(fresh_logits)  # [1, bs-1]

        # Stale draft — same noise, but H from the previous step.
        # pos_stale accounts for H_stale potentially having a different length.
        stale_logits = target.lm_head(draft(
            target_hidden=H_stale,
            noise_embedding=noise,
            position_ids=pos_stale,
            past_key_values=draft_cache_stale,
            use_cache=True,
            is_causal=False,
        )[:, 1 - bs:, :])
        draft_cache_stale.crop(start)
        stale_tokens = sample(stale_logits)  # [1, bs-1]

        # Token overlap
        overlap = float((fresh_tokens == stale_tokens).float().mean())
        overlaps.append(overlap)

        # Verify with target using fresh tokens (standard DFlash)
        block_fresh = block.clone()
        block_fresh[:, 1:] = fresh_tokens
        out = target(
            block_fresh,
            position_ids=position_ids[:, start: start + bs],
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(out.logits, temperature)
        acc_fresh = int((block_fresh[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
        acceptance_fresh.append(acc_fresh + 1)

        # Simulate stale acceptance (how many stale tokens match posterior)
        acc_stale = int((stale_tokens == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
        acceptance_stale_sim.append(acc_stale + 1)

        start += acc_fresh + 1
        target_cache.crop(start)

        H_stale = H_fresh
        H_fresh = extract_context_feature(out.hidden_states, draft.target_layer_ids)[:, :acc_fresh + 1, :]

    return overlaps, acceptance_fresh, acceptance_stale_sim


def analyse(args) -> dict:
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    device = f"cuda:{args.gpu}"
    target, draft, tokenizer = _load_models(args.model, args.draft_model, device, attn_impl)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    all_overlaps, all_fresh, all_stale = [], [], []

    for item in tqdm(dataset, desc="stale vs fresh comparison"):
        prompt = _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": item["turns"][0]}],
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        ovl, fresh, stale = _run_stale_comparison(
            target, draft, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        all_overlaps.extend(ovl)
        all_fresh.extend(fresh)
        all_stale.extend(stale)

    mean_overlap = float(np.mean(all_overlaps)) if all_overlaps else 0.0
    mean_fresh_acc = float(np.mean(all_fresh)) if all_fresh else 0.0
    mean_stale_acc = float(np.mean(all_stale)) if all_stale else 0.0
    acc_retention = mean_stale_acc / mean_fresh_acc if mean_fresh_acc > 0 else 0.0

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "n_steps": len(all_overlaps),
        "mean_token_overlap": mean_overlap,
        "mean_fresh_acceptance": mean_fresh_acc,
        "mean_stale_acceptance": mean_stale_acc,
        "acceptance_retention": acc_retention,
    }

    table = Table(title="Stale H vs Fresh H — Draft Quality", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Steps measured", str(results["n_steps"]))
    table.add_row("Token overlap (stale vs fresh)", f"{mean_overlap * 100:.1f}%")
    table.add_row("Mean accept len (fresh H)",  f"{mean_fresh_acc:.2f}")
    table.add_row("Mean accept len (stale H)",  f"{mean_stale_acc:.2f}")
    table.add_row("Acceptance retention",        f"{acc_retention * 100:.1f}%")
    print(table)

    r = acc_retention
    if r >= 0.90:
        verdict = "[green]EXCELLENT[/green] — stale H loses < 10% acceptance; DFlash-SSD is strongly viable"
    elif r >= 0.80:
        verdict = "[yellow]GOOD[/yellow] — 80-90% acceptance retained; net speedup expected even with some degradation"
    elif r >= 0.70:
        verdict = "[orange1]MARGINAL[/orange1] — consider fast-refinement: one DFlash pass with fresh H on each hit"
    else:
        verdict = "[red]POOR[/red] — stale H causes too much degradation; re-evaluate approach"
    print(f"\nVerdict: {verdict}")

    out_path = Path(args.out_dir) / f"stale_quality_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="Phase-0: stale vs fresh hidden state draft quality")
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--dataset", default="gsm8k", choices=list(DATASETS))
    p.add_argument("--max-samples", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out-dir", default="results/analysis")
    analyse(p.parse_args())


if __name__ == "__main__":
    main()
