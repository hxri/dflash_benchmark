from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from rich import print
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .benchmark import (
    DATASETS,
    _apply_chat_template,
    _limit_dataset,
    load_and_process_dataset,
)
from .model import DFlashDraftModel, extract_context_feature, sample


@dataclass
class StepTrace:
    request_id: str
    step_idx: int
    accepted_tokens: int
    proposed_tokens: int
    acceptance_ratio: float
    step_ms: float
    cumulative_decode_tokens: int
    cumulative_decode_tps: float
    hidden_drift_cosine: Optional[float]
    cache_seq_len: int


@dataclass
class SampleSummary:
    request_id: str
    dataset: str
    input_tokens: int
    output_tokens: int
    time_to_first_token: float
    time_per_output_token: float
    decode_tps: float
    mean_acceptance_tokens: float
    p10_acceptance_tokens: float
    mean_hidden_drift: Optional[float]
    stop_reason: str


@dataclass
class DFlashPrefetchEntry:
    block_output_ids: torch.Tensor
    source_target_hidden: torch.Tensor
    draft_hidden: torch.Tensor
    draft_logits: torch.Tensor
    next_bonus_logits: torch.Tensor


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _dist_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _dist_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _dist_is_main() -> bool:
    return _dist_rank() == 0


def _dist_init() -> None:
    if "RANK" not in os.environ:
        return
    from torch import distributed as torch_dist

    torch_dist.init_process_group(backend="nccl", init_method="env://")


def _dist_gather(obj: Any):
    from torch import distributed as torch_dist

    if not torch_dist.is_initialized():
        return [obj]
    if _dist_is_main():
        objs = [None for _ in range(_dist_size())]
        torch_dist.gather_object(obj, objs, dst=0)
        return objs
    torch_dist.gather_object(obj, dst=0)
    return None


def _safe_cosine(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> Optional[float]:
    if a is None or b is None:
        return None
    if a.numel() == 0 or b.numel() == 0:
        return None
    return float(F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item())


def _top_candidate_tokens(
    logits: torch.Tensor,
    *,
    top_k: int,
    exclude_token: Optional[int] = None,
) -> list[int]:
    if top_k <= 0:
        return []
    logits_1d = logits[0, 0]
    fetch_k = min(logits_1d.shape[0], top_k + (1 if exclude_token is not None else 0))
    tokens = torch.topk(logits_1d, k=fetch_k).indices.tolist()
    if exclude_token is not None:
        tokens = [token for token in tokens if token != exclude_token]
    return tokens[:top_k]


def _generate_dflash_prefetch_entry(
    model: DFlashDraftModel,
    target: torch.nn.Module,
    target_hidden: torch.Tensor,
    *,
    first_token: int,
    block_len: int,
    mask_token_id: int,
    start_position: int,
    temperature: float,
) -> DFlashPrefetchEntry:
    device = target.device
    vocab_size = target.lm_head.weight.shape[0]
    block_output_ids = torch.full(
        (1, block_len),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    block_output_ids[:, 0] = first_token
    noise_embedding = target.model.embed_tokens(block_output_ids)
    block_position_ids = torch.arange(
        start_position - target_hidden.shape[1],
        start_position + block_len,
        device=device,
    ).unsqueeze(0)

    draft_hidden = model(
        target_hidden=target_hidden,
        noise_embedding=noise_embedding,
        position_ids=block_position_ids,
        past_key_values=DynamicCache(),
        use_cache=True,
        is_causal=False,
    )[:, 1 - block_len :, :].detach()

    if block_len > 1:
        draft_logits = target.lm_head(draft_hidden[:, : block_len - 1, :]).detach()
        block_output_ids[:, 1:block_len] = sample(draft_logits, temperature)
    else:
        draft_logits = torch.empty((1, 0, vocab_size), dtype=target.lm_head.weight.dtype, device=device)

    next_bonus_logits = target.lm_head(draft_hidden[:, -1:, :]).detach()
    return DFlashPrefetchEntry(
        block_output_ids=block_output_ids,
        source_target_hidden=target_hidden.detach(),
        draft_hidden=draft_hidden,
        draft_logits=draft_logits,
        next_bonus_logits=next_bonus_logits,
    )


def _build_dflash_outcome_cache(
    model: DFlashDraftModel,
    target: torch.nn.Module,
    current_entry: DFlashPrefetchEntry,
    *,
    generated_before_round: int,
    max_new_tokens: int,
    block_size: int,
    mask_token_id: int,
    start_position: int,
    temperature: float,
    speculation_fanout: int,
) -> dict[tuple[int, int], DFlashPrefetchEntry]:
    if speculation_fanout <= 0:
        return {}

    block_len = current_entry.block_output_ids.shape[1]
    outcome_cache: dict[tuple[int, int], DFlashPrefetchEntry] = {}

    for accepted_tokens in range(1, block_len + 1):
        remaining_tokens = max_new_tokens - (generated_before_round + accepted_tokens)
        next_block_len = min(block_size, remaining_tokens)
        if next_block_len <= 0:
            continue

        if accepted_tokens < block_len:
            candidate_logits = current_entry.draft_logits[:, accepted_tokens - 1 : accepted_tokens, :]
            exclude_token = int(current_entry.block_output_ids[0, accepted_tokens].item())
        else:
            candidate_logits = current_entry.next_bonus_logits
            exclude_token = None

        bonus_candidates = _top_candidate_tokens(
            candidate_logits,
            top_k=speculation_fanout,
            exclude_token=exclude_token,
        )
        surrogate_hidden = current_entry.source_target_hidden

        for bonus_token in bonus_candidates:
            outcome_cache[(accepted_tokens, bonus_token)] = _generate_dflash_prefetch_entry(
                model,
                target,
                surrogate_hidden,
                first_token=bonus_token,
                block_len=next_block_len,
                mask_token_id=mask_token_id,
                start_position=start_position + accepted_tokens,
                temperature=temperature,
            )

    return outcome_cache


def _resolve_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash-attn not installed. Falling back to torch.sdpa for SSD+DFlash runtime. "
            "Install for best performance: .venv-cuda/bin/pip install flash-attn --no-build-isolation"
        )
        return "sdpa"


def _format_runtime_table(s: SampleSummary, block_size: int) -> Table:
    table = Table(title=f"Decode Monitor :: {s.request_id}", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Input tokens", str(s.input_tokens))
    table.add_row("Output tokens", str(s.output_tokens))
    table.add_row("TTFT (s)", f"{s.time_to_first_token:.4f}")
    table.add_row("Decode tok/s", f"{s.decode_tps:.2f}")
    table.add_row("Mean acceptance", f"{s.mean_acceptance_tokens:.2f} / {block_size}")
    table.add_row("P10 acceptance", f"{s.p10_acceptance_tokens:.2f}")
    if s.mean_hidden_drift is not None:
        table.add_row("Mean hidden drift cos", f"{s.mean_hidden_drift:.4f}")
    table.add_row("Stop reason", s.stop_reason)
    return table


@torch.inference_mode()
def ssd_dflash_generate_monitored(
    model: DFlashDraftModel,
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int],
    mask_token_id: Optional[int],
    request_id: str,
    monitor_every: int,
    live_monitor: bool,
    speculation_fanout: int,
    trace_fh,
):
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=target.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    target_cache = DynamicCache()

    prefill_start = _cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=target_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=block_size > 1,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)

    target_hidden = None
    prev_hidden_last = None
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        prev_hidden_last = target_hidden[:, -1, :].detach()

    time_to_first_token = _cuda_time() - prefill_start
    decode_start = _cuda_time()

    start = num_input_tokens
    acceptance_lengths: list[int] = []
    hidden_drifts: list[float] = []
    traces: list[StepTrace] = []
    stop_reason = "max_new_tokens"
    step_idx = 0
    prefetched_entry: Optional[DFlashPrefetchEntry] = None

    while start < max_length:
        generated_before_round = start - num_input_tokens
        block_len = min(block_size, max_length - start)
        if block_len <= 0:
            break

        t_step0 = _cuda_time()
        if prefetched_entry is None or prefetched_entry.block_output_ids.shape[1] != block_len:
            current_entry = _generate_dflash_prefetch_entry(
                model,
                target,
                target_hidden,
                first_token=int(output_ids[0, start].item()),
                block_len=block_len,
                mask_token_id=mask_token_id,
                start_position=start,
                temperature=temperature,
            )
        else:
            current_entry = prefetched_entry

        block_output_ids = current_entry.block_output_ids.clone()
        block_position_ids = position_ids[:, start : start + block_len]

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=block_len > 1,
        )
        posterior = sample(output.logits, temperature)

        if block_len > 1:
            acceptance_len = int(
                (block_output_ids[:, 1:block_len] == posterior[:, : block_len - 1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
        else:
            acceptance_len = 0

        accepted_tokens = acceptance_len + 1
        output_ids[:, start : start + accepted_tokens] = block_output_ids[:, :accepted_tokens]
        if start + accepted_tokens < output_ids.shape[1]:
            output_ids[:, start + accepted_tokens] = posterior[:, acceptance_len]

        start += accepted_tokens
        target_cache.crop(start)
        acceptance_lengths.append(accepted_tokens)

        hidden_drift = None
        if block_len > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :accepted_tokens, :]
            curr_hidden_last = target_hidden[:, -1, :].detach()
            hidden_drift = _safe_cosine(prev_hidden_last, curr_hidden_last)
            prev_hidden_last = curr_hidden_last
            if hidden_drift is not None:
                hidden_drifts.append(hidden_drift)

        actual_bonus = None
        if start + accepted_tokens < output_ids.shape[1]:
            actual_bonus = int(posterior[0, acceptance_len].item())

        next_round_cache = _build_dflash_outcome_cache(
            model,
            target,
            current_entry,
            generated_before_round=generated_before_round,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            mask_token_id=mask_token_id,
            start_position=start,
            temperature=temperature,
            speculation_fanout=speculation_fanout,
        )
        prefetched_entry = None if actual_bonus is None else next_round_cache.get((accepted_tokens, actual_bonus))

        step_ms = (_cuda_time() - t_step0) * 1000.0
        decode_tokens = max(1, start - num_input_tokens)
        decode_tps = decode_tokens / max((_cuda_time() - decode_start), 1e-9)

        trace = StepTrace(
            request_id=request_id,
            step_idx=step_idx,
            accepted_tokens=accepted_tokens,
            proposed_tokens=block_len,
            acceptance_ratio=accepted_tokens / max(block_len, 1),
            step_ms=step_ms,
            cumulative_decode_tokens=decode_tokens,
            cumulative_decode_tps=decode_tps,
            hidden_drift_cosine=hidden_drift,
            cache_seq_len=start,
        )
        traces.append(trace)
        if trace_fh is not None:
            trace_fh.write(json.dumps(asdict(trace)) + "\n")

        if live_monitor and (step_idx % max(1, monitor_every) == 0):
            drift_s = "n/a" if hidden_drift is None else f"{hidden_drift:.4f}"
            print(
                f"[monitor] req={request_id} step={step_idx:04d} "
                f"acc={accepted_tokens}/{block_len} "
                f"step_ms={step_ms:.2f} tps={decode_tps:.2f} drift={drift_s}"
            )

        step_idx += 1

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            stop_reason = "eos"
            break

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_idx = torch.isin(output_ids[0][num_input_tokens:], stop_ids).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_idx[0] + 1]
            stop_reason = "eos"

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start

    return SimpleNamespace(
        output_ids=output_ids,
        traces=traces,
        stop_reason=stop_reason,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        hidden_drifts=hidden_drifts,
    )


def _request_summary(
    request_id: str,
    dataset: str,
    result,
) -> SampleSummary:
    acc = result.acceptance_lengths
    drift = result.hidden_drifts
    return SampleSummary(
        request_id=request_id,
        dataset=dataset,
        input_tokens=result.num_input_tokens,
        output_tokens=result.num_output_tokens,
        time_to_first_token=float(result.time_to_first_token),
        time_per_output_token=float(result.time_per_output_token),
        decode_tps=1.0 / max(float(result.time_per_output_token), 1e-9),
        mean_acceptance_tokens=float(np.mean(acc)) if acc else 0.0,
        p10_acceptance_tokens=float(np.percentile(acc, 10)) if acc else 0.0,
        mean_hidden_drift=float(np.mean(drift)) if drift else None,
        stop_reason=result.stop_reason,
    )


def _aggregate(summaries: list[dict], block_size: int) -> dict:
    ttft = [s["time_to_first_token"] for s in summaries]
    tpot = [s["time_per_output_token"] for s in summaries]
    tps = [s["decode_tps"] for s in summaries]
    mean_acc = [s["mean_acceptance_tokens"] for s in summaries]
    p10_acc = [s["p10_acceptance_tokens"] for s in summaries]
    drift = [s["mean_hidden_drift"] for s in summaries if s["mean_hidden_drift"] is not None]
    mean_acceptance = float(np.mean(mean_acc)) if mean_acc else 0.0

    out = {
        "num_requests": len(summaries),
        "mean_ttft": float(np.mean(ttft)) if ttft else 0.0,
        "median_ttft": float(np.median(ttft)) if ttft else 0.0,
        "mean_time_per_output_token": float(np.mean(tpot)) if tpot else 0.0,
        "mean_decode_tps": float(np.mean(tps)) if tps else 0.0,
        "median_decode_tps": float(np.median(tps)) if tps else 0.0,
        "mean_acceptance_tokens": mean_acceptance,
        "avg_acceptance_length": mean_acceptance,
        "acceptance_definition": "dflash-style mean of per-request mean acceptance",
        "median_acceptance_tokens": float(np.median(mean_acc)) if mean_acc else 0.0,
        "p10_acceptance_tokens": float(np.mean(p10_acc)) if p10_acc else 0.0,
        "block_size": int(block_size),
        "acceptance_utilization": float(mean_acceptance / max(block_size, 1)) if mean_acc else 0.0,
        "mean_hidden_drift_cosine": float(np.mean(drift)) if drift else None,
        "stop_reasons": {k: int(v) for k, v in _hist([s["stop_reason"] for s in summaries]).items()},
    }
    return out


def _hist(values: list[Any]) -> dict[Any, int]:
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def _print_aggregate(agg: dict) -> None:
    table = Table(title="SSD+DFlash Aggregate", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("# requests", str(agg["num_requests"]))
    table.add_row("Mean decode tok/s", f"{agg['mean_decode_tps']:.2f}")
    table.add_row("Median decode tok/s", f"{agg['median_decode_tps']:.2f}")
    table.add_row("Mean TTFT (s)", f"{agg['mean_ttft']:.4f}")
    table.add_row("Avg acceptance length", f"{agg['avg_acceptance_length']:.2f} / {agg['block_size']}")
    table.add_row("Acceptance utilization", f"{agg['acceptance_utilization'] * 100:.1f}%")
    drift = agg.get("mean_hidden_drift_cosine")
    table.add_row("Mean hidden drift cos", "n/a" if drift is None else f"{drift:.4f}")
    print(table)


def _decode_watch_guidance(agg: dict) -> list[str]:
    notes = []
    util = agg["acceptance_utilization"]
    drift = agg.get("mean_hidden_drift_cosine")
    tps = agg["mean_decode_tps"]

    if util >= 0.70:
        notes.append("Acceptance is strong. Keep current block size and focus on maximizing batch throughput.")
    elif util >= 0.50:
        notes.append("Acceptance is moderate. Consider lowering block size by 2-4 to reduce wasted draft work.")
    else:
        notes.append("Acceptance is low. Check prompt domain mismatch and consider a closer DFlash draft for this target.")

    if drift is not None:
        if drift >= 0.95:
            notes.append("Hidden-state drift is very low. Stale-state assumptions are stable for long decode windows.")
        elif drift >= 0.85:
            notes.append("Hidden-state drift is acceptable. Monitor tails where acceptance drops late in generation.")
        else:
            notes.append("Hidden-state drift is high. Watch for step spikes and frequent single-token acceptances.")

    if tps < 30:
        notes.append("Decode tok/s is relatively low for 2x A6000. Confirm flash-attn kernels are active and GPU clocks are stable.")
    else:
        notes.append("Decode tok/s is healthy. Next gains should come from prompt packing or multi-request scheduling.")

    return notes


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_dataset(args: argparse.Namespace) -> None:
    _dist_init()
    torch.cuda.set_device(_dist_local_rank())
    device = torch.device(f"cuda:{_dist_local_rank()}")

    attn_impl = _resolve_attn_impl()

    logger.info(f"Loading target model: {args.model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    logger.info(f"Loading DFlash draft: {args.draft_model}")
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else int(draft.block_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / f"trace_rank{_dist_rank()}.jsonl"
    samples_path = out_dir / f"samples_rank{_dist_rank()}.jsonl"

    local_summaries: list[dict] = []

    with open(traces_path, "w") as trace_fh, open(samples_path, "w") as samples_fh:
        indices = range(_dist_rank(), len(dataset), _dist_size())
        for idx in tqdm(indices, disable=not _dist_is_main(), desc="SSD+DFlash eval"):
            item = dataset[idx]
            user_turn = item["turns"][0]
            messages = [{"role": "user", "content": user_turn}]
            prompt = _apply_chat_template(
                tokenizer,
                messages,
                enable_thinking=args.enable_thinking,
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            req_id = f"{args.dataset}:{idx}"

            result = ssd_dflash_generate_monitored(
                draft,
                target,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                block_size=block_size,
                mask_token_id=args.mask_token_id,
                request_id=req_id,
                monitor_every=args.monitor_every,
                live_monitor=args.live_monitor,
                speculation_fanout=args.speculation_fanout,
                trace_fh=trace_fh,
            )

            summary = _request_summary(req_id, args.dataset, result)
            summary_dict = asdict(summary)
            local_summaries.append(summary_dict)
            samples_fh.write(json.dumps(summary_dict) + "\n")

            if args.write_text_outputs:
                text_out = tokenizer.decode(
                    result.output_ids[0, result.num_input_tokens :],
                    skip_special_tokens=True,
                )
                with open(out_dir / f"output_{_dist_rank()}_{idx}.txt", "w") as fh:
                    fh.write(text_out)

            if args.live_monitor and _dist_is_main():
                print(_format_runtime_table(summary, block_size))

    gathered = _dist_gather(local_summaries)
    if not _dist_is_main():
        return

    flat = list(chain.from_iterable(gathered))
    agg = _aggregate(flat, block_size)

    metadata = {
        "timestamp": _timestamp(),
        "model": args.model,
        "draft_model": args.draft_model,
        "dataset": args.dataset,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "world_size": _dist_size(),
        "block_size": block_size,
        "speculation_fanout": args.speculation_fanout,
        "attention_implementation": attn_impl,
        "flash_attention": attn_impl == "flash_attention_2",
    }

    with open(out_dir / "aggregate.json", "w") as fh:
        json.dump({"metadata": metadata, "aggregate": agg}, fh, indent=2)

    _print_aggregate(agg)
    print("\nWhat to watch during decoding:")
    for note in _decode_watch_guidance(agg):
        print(f"  - {note}")

    print(f"\nSaved aggregate -> {out_dir / 'aggregate.json'}")
    print(f"Saved per-step traces -> {traces_path.parent / 'trace_rank*.jsonl'}")
    print(f"Saved per-sample summaries -> {samples_path.parent / 'samples_rank*.jsonl'}")


def _run_single_prompt(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    attn_impl = _resolve_attn_impl()

    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    block_size = args.block_size if args.block_size is not None else int(draft.block_size)
    text = args.prompt
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    result = ssd_dflash_generate_monitored(
        draft,
        target,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
        block_size=block_size,
        mask_token_id=args.mask_token_id,
        request_id="interactive",
        monitor_every=max(1, args.monitor_every),
        live_monitor=True,
        speculation_fanout=args.speculation_fanout,
        trace_fh=None,
    )

    summary = _request_summary("interactive", "prompt", result)
    print(_format_runtime_table(summary, block_size))

    decoded = tokenizer.decode(result.output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:\n")
    print(decoded)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Novel SSD+DFlash CUDA runtime with live monitoring, eval, and logging."
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16")
    p.add_argument("--dataset", default="gsm8k", choices=list(DATASETS))
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--mask-token-id", type=int, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--live-monitor", action="store_true")
    p.add_argument("--monitor-every", type=int, default=8)
    p.add_argument("--speculation-fanout", type=int, default=2)
    p.add_argument("--write-text-outputs", action="store_true")
    p.add_argument("--out-dir", default="results/ssd_dflash_accel")

    p.add_argument("--mode", choices=["dataset", "prompt"], default="dataset")
    p.add_argument("--prompt", default="")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "prompt":
        if not args.prompt:
            raise ValueError("--prompt is required when --mode prompt")
        _run_single_prompt(args)
        return

    if args.out_dir == "results/ssd_dflash_accel":
        args.out_dir = str(Path(args.out_dir) / _timestamp())

    _run_dataset(args)


if __name__ == "__main__":
    main()
