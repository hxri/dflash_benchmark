"""
Phase-0 analysis: cosine similarity between consecutive DFlash target hidden states.

Validates the core staleness assumption in DFlash-SSD:
  cos_sim(H_t, H_{t+1}) ≈ 1  ⟹  using H_t as proxy for H_{t+1} is viable.

Usage:
  python -m dflash.analysis.hidden_state_sim \
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
import torch.nn.functional as F
from loguru import logger
from rich import print
from rich.table import Table
from tqdm import tqdm

from ..benchmark import load_and_process_dataset, _limit_dataset, _apply_chat_template, DATASETS
from ..model import DFlashDraftModel, dflash_generate, extract_context_feature


def _load_models(model_id: str, draft_id: str, device: str, attn_impl: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"Loading target: {model_id}")
    target = AutoModelForCausalLM.from_pretrained(
        model_id, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()
    logger.info(f"Loading DFlash draft: {draft_id}")
    draft = DFlashDraftModel.from_pretrained(
        draft_id, attn_implementation=attn_impl, dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return target, draft, tokenizer


@torch.inference_mode()
def _collect_hidden_states(
    target, draft, input_ids, max_new_tokens: int, temperature: float
) -> list[torch.Tensor]:
    """Run one DFlash generation and return all per-step target hidden tensors."""
    from transformers import DynamicCache
    from ..model import sample

    block_size = draft.block_size
    mask_token_id = draft.mask_token_id
    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens
    device = input_ids.device

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    target_cache = DynamicCache()
    draft_cache = DynamicCache()

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
    H = extract_context_feature(out.hidden_states, draft.target_layer_ids)

    hidden_sequence: list[torch.Tensor] = [H.detach().cpu()]
    start = num_input

    while start < max_length:
        bs = min(block_size, max_length - start)
        if bs <= 1:
            break
        block = output_ids[:, start: start + bs].clone()
        noise = target.model.embed_tokens(block)
        draft_logits = target.lm_head(draft(
            target_hidden=H,
            noise_embedding=noise,
            position_ids=position_ids[:, draft_cache.get_seq_length(): start + bs],
            past_key_values=draft_cache,
            use_cache=True,
            is_causal=False,
        )[:, 1 - bs:, :])
        draft_cache.crop(start)
        block[:, 1:] = sample(draft_logits)

        out = target(
            block,
            position_ids=position_ids[:, start: start + bs],
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(out.logits, temperature)
        acc = int((block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0])
        start += acc + 1
        target_cache.crop(start)

        H = extract_context_feature(out.hidden_states, draft.target_layer_ids)[:, :acc + 1, :]
        hidden_sequence.append(H.detach().cpu())

    return hidden_sequence


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

    all_sims: list[float] = []  # per-consecutive-step cosine similarities
    per_layer_sims: dict[int, list[float]] = {}  # if we inspect layer-by-layer

    for sample_item in tqdm(dataset, desc="collecting hidden states"):
        user_text = sample_item["turns"][0]
        prompt = _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": user_text}],
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        hiddens = _collect_hidden_states(
            target, draft, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        # Compute cos_sim between consecutive H_t and H_{t+1}
        for h_prev, h_next in zip(hiddens[:-1], hiddens[1:]):
            # Use the LAST position of each hidden (most recent accepted token)
            v_prev = h_prev[0, -1, :].float()
            v_next = h_next[0, -1, :].float()
            sim = float(F.cosine_similarity(v_prev.unsqueeze(0), v_next.unsqueeze(0)))
            all_sims.append(sim)

    results = {
        "dataset": args.dataset,
        "model": args.model,
        "n_pairs": len(all_sims),
        "mean_cosine_sim": float(np.mean(all_sims)),
        "median_cosine_sim": float(np.median(all_sims)),
        "p10_cosine_sim": float(np.percentile(all_sims, 10)),
        "p25_cosine_sim": float(np.percentile(all_sims, 25)),
        "std_cosine_sim": float(np.std(all_sims)),
    }

    # Print rich table
    table = Table(title="Hidden State Cosine Similarity  H_t → H_{t+1}", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("# step pairs measured", str(results["n_pairs"]))
    table.add_row("Mean cos_sim", f"{results['mean_cosine_sim']:.4f}")
    table.add_row("Median cos_sim", f"{results['median_cosine_sim']:.4f}")
    table.add_row("P25 cos_sim", f"{results['p25_cosine_sim']:.4f}")
    table.add_row("P10 cos_sim (worst 10%)", f"{results['p10_cosine_sim']:.4f}")
    print(table)

    verdict = ""
    m = results["median_cosine_sim"]
    if m >= 0.95:
        verdict = "[green]EXCELLENT[/green] — stale H approximation should work very well (< 5% drift)"
    elif m >= 0.85:
        verdict = "[yellow]GOOD[/yellow] — stale H approximation viable, expect minor acceptance degradation"
    elif m >= 0.70:
        verdict = "[orange1]MARGINAL[/orange1] — consider fast-refinement pass with fresh H on cache hits"
    else:
        verdict = "[red]POOR[/red] — hidden states change too fast; staleness approach not recommended"
    print(f"\nVerdict: {verdict}")

    out_path = Path(args.out_dir) / f"hidden_sim_{args.dataset}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")
    return results


def main():
    p = argparse.ArgumentParser(description="Phase-0: hidden state cosine similarity analysis")
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
