"""
DFlash-SSD: async pre-speculation for block-diffusion drafting.

Core idea
─────────
Standard DFlash is sequential:
    [draft(H_t)]  →  [verify → H_{t+1}]  →  [draft(H_{t+1})]  →  …
    cost/step = T_draft + T_verify

DFlash-SSD runs them concurrently:
    [verify(block_{t-1}) ─────────────────────►]  GPU 0
    [draft(H_{t-1}, stale) ──►]                   GPU 1
    cost/step ≈ max(T_verify, F × T_draft)

Staleness: the pre-speculated draft uses H_{t-1} (one step behind).
If cos_sim(H_{t-1}, H_t) ≈ 1 (empirically true for coherent text),
acceptance rate with stale H is close to fresh H.

Two-GPU layout:
    cuda:0  — target model  (Qwen3-4B or larger)
    cuda:1  — DFlash draft  (z-lab/Qwen3-4B-DFlash-b16)
    Hidden states transferred from cuda:0 → cuda:1 after each verify.
"""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import DynamicCache

from .model import DFlashDraftModel, extract_context_feature, sample
from .monitor import LiveMonitor, RunningStats, StepStats


# ─────────────────────────────────────────────────────────
# Acceptance-length predictor
# ─────────────────────────────────────────────────────────

class AcceptancePredictor:
    """
    Tracks acceptance-length history and predicts the top-F most likely
    acceptance lengths for the next step.

    Strategies:
      "top_high"  — always predict the top end of the range
                    (safe prior: DFlash usually has high acceptance)
      "running"   — exponential moving average of observed histogram
    """

    def __init__(self, block_size: int, strategy: str = "running", ema_alpha: float = 0.1):
        self.block_size = block_size
        self.strategy = strategy
        self.alpha = ema_alpha
        # counts[k] = smoothed frequency of acceptance_length == k
        self.counts = np.ones(block_size + 1, dtype=float)  # Laplace smoothing
        self.total = float(block_size + 1)

    def top_k(self, k: int) -> list[int]:
        """Return k most likely acceptance lengths (descending probability)."""
        if self.strategy == "top_high":
            return list(range(self.block_size, max(0, self.block_size - k), -1))
        # "running": sort by smoothed probability
        probs = self.counts / self.counts.sum()
        ranked = np.argsort(probs)[::-1]
        return [int(ranked[i]) for i in range(min(k, len(ranked)))]

    def update(self, acceptance_length: int):
        if self.strategy == "running":
            one_hot = np.zeros(self.block_size + 1, dtype=float)
            one_hot[min(acceptance_length, self.block_size)] = 1.0
            self.counts = (1 - self.alpha) * self.counts + self.alpha * one_hot
            self.total = self.counts.sum()
        # top_high doesn't update


# ─────────────────────────────────────────────────────────
# Cross-device helper
# ─────────────────────────────────────────────────────────

def _clone_module_to_device(module: torch.nn.Module, device) -> torch.nn.Module:
    """Deep-copy a module and move all parameters to `device`."""
    cloned = copy.deepcopy(module)
    return cloned.to(device).eval()


def _ensure_draft_bound(
    target: torch.nn.Module,
    draft_model: DFlashDraftModel,
    draft_device,
) -> None:
    """Idempotent: clone embed_tokens + lm_head to draft_device once per device.

    Calling this multiple times with the same draft_device is cheap (no-op after
    the first call).  This fixes the Bug-2 weight-clone-per-call issue.
    """
    marker = f"ssd_bound:{draft_device}"
    if getattr(draft_model, "_ssd_device_marker", None) == marker:
        return
    logger.info(f"Binding draft to {draft_device} (one-time clone of embed_tokens + lm_head)…")
    draft_model.embed_tokens = _clone_module_to_device(target.model.embed_tokens, draft_device)
    draft_model.lm_head = _clone_module_to_device(target.lm_head, draft_device)
    object.__setattr__(draft_model, "_ssd_device_marker", marker)
    logger.info("Binding complete.")


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between last-position hidden vectors."""
    v_a = a[0, -1, :].float()
    v_b = b[0, -1, :].float()
    return float(F.cosine_similarity(v_a.unsqueeze(0), v_b.unsqueeze(0)).item())


# ─────────────────────────────────────────────────────────
# DFlash-SSD generator
# ─────────────────────────────────────────────────────────

@torch.inference_mode()
def dflash_ssd_generate(
    draft_model: DFlashDraftModel,
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    fan_out: int = 2,
    acceptance_predictor: Optional[AcceptancePredictor] = None,
    fast_refine: bool = False,
    return_stats: bool = False,
    monitor: Optional[LiveMonitor] = None,
    log_path: Optional[str] = None,
) -> SimpleNamespace:
    """
    DFlash-SSD generation.

    Parameters
    ----------
    draft_model     DFlash draft (should be on cuda:1 for dual-GPU)
    target          Target LM (on cuda:0)
    input_ids       Prompt token IDs (on target device)
    max_new_tokens  Max tokens to generate
    stop_token_ids  EOS / stop tokens
    temperature     Sampling temperature (0 = greedy)
    block_size      Override draft block size
    mask_token_id   Override mask token id
    fan_out         F: number of draft blocks to pre-speculate
    acceptance_predictor  AcceptancePredictor instance (created internally if None)
    fast_refine     If True, after verify produces fresh H_t, run one additional
                    DFlash pass with fresh H_t to fix stale-H quality degradation.
                    Adds ~T_draft latency on top of T_wall but restores acceptance
                    to standard DFlash levels.  Use when H_SIM < 0.85.
    return_stats    If True, attach per-step stats to result
    monitor         LiveMonitor instance for real-time display
    log_path        If set, append per-step stats as JSONL

    Returns
    -------
    SimpleNamespace with fields:
        output_ids, num_input_tokens, num_output_tokens,
        time_per_output_token, acceptance_lengths, stats (if return_stats)
    """

    target_device = input_ids.device
    draft_device = next(draft_model.parameters()).device

    block_size = block_size if block_size is not None else draft_model.block_size
    mask_token_id = mask_token_id if mask_token_id is not None else draft_model.mask_token_id
    if acceptance_predictor is None:
        acceptance_predictor = AcceptancePredictor(block_size)

    num_input = input_ids.shape[1]
    max_length = num_input + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=target_device
    )
    position_ids = torch.arange(output_ids.shape[1], device=target_device).unsqueeze(0)

    # Bug-2 fix: clone once, not on every call.
    _ensure_draft_bound(target, draft_model, draft_device)
    draft_embed = draft_model.embed_tokens
    draft_lm_head = draft_model.lm_head

    target_cache = DynamicCache()

    log_file = open(log_path, "a") if log_path else None

    # ── Prefill ──────────────────────────────────────────
    prefill_start = time.perf_counter()
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
    H_prev = extract_context_feature(out.hidden_states, draft_model.target_layer_ids).detach()
    prefill_s = time.perf_counter() - prefill_start

    start = num_input
    acceptance_lengths: list[int] = [1]
    step_stats_list: list[StepStats] = []
    step_idx = 0

    decode_start = time.perf_counter()

    # Pre-speculate FIRST block (warmup — no parallelism on step 0).
    # H_prev from prefill is the full prompt hidden state (shape [1, num_input, …]),
    # so ctx_start = start - num_input = 0, matching dflash_generate's step-0 convention.
    H_prev_d1 = H_prev.to(draft_device, non_blocking=True)
    draft_cache_pre = DynamicCache()
    pre_block_ids = torch.cat([
        output_ids[:, start: start + 1].to(draft_device),
        torch.full((1, block_size - 1), mask_token_id, dtype=torch.long, device=draft_device),
    ], dim=1)
    pre_noise = draft_embed(pre_block_ids)
    # Bug-1 fix (warmup): position IDs must span ctx_len + block_size positions.
    # ctx_len = H_prev.shape[1] = num_input; ctx_start = start - num_input = 0.
    _ctx_len_init = H_prev_d1.shape[1]
    _ctx_start_init = start - _ctx_len_init          # = 0 for first step
    pre_pos_init = position_ids[:, _ctx_start_init : _ctx_start_init + _ctx_len_init + block_size].to(draft_device)
    pre_logits = draft_lm_head(draft_model(
        target_hidden=H_prev_d1,
        noise_embedding=pre_noise,
        position_ids=pre_pos_init,
        past_key_values=draft_cache_pre,
        use_cache=True,
        is_causal=False,
    )[:, 1 - block_size:, :])
    draft_cache_pre.crop(start)
    pre_draft_tokens = sample(pre_logits).to(target_device)  # [1, block_size-1]

    # ── Main decode loop ──────────────────────────────────
    while start < max_length:
        bs = min(block_size, max_length - start)
        if bs <= 1:
            break

        step_idx += 1
        top_f = acceptance_predictor.top_k(fan_out)

        # Build the verify block from the pre-speculated draft
        verify_block = torch.cat([
            output_ids[:, start: start + 1],
            pre_draft_tokens[:, : bs - 1],
        ], dim=1)  # [1, bs] on target_device
        block_positions = position_ids[:, start: start + bs]

        # ── Async launch ─────────────────────────────────
        # Thread A  (GPU 0): verify the current block
        # Thread B  (GPU 1): pre-speculate draft for the NEXT step using stale H

        verify_result: dict = {}
        pre_result: dict = {}

        def _run_verify():
            # torch.inference_mode() is thread-local — child threads do NOT inherit
            # it from the spawning thread, so we must declare it explicitly here.
            with torch.inference_mode():
                try:
                    torch.cuda.set_device(target_device)
                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    ev_s.record()
                    v_out = target(
                        verify_block,
                        position_ids=block_positions,
                        past_key_values=target_cache,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    ev_e.record()
                    torch.cuda.synchronize(target_device)
                    verify_result["out"] = v_out
                    verify_result["t_ms"] = ev_s.elapsed_time(ev_e)
                except Exception as exc:
                    verify_result["exc"] = exc

        def _run_pre_draft():
            # Same reason as above — declare inference_mode inside the thread.
            with torch.inference_mode():
                try:
                    torch.cuda.set_device(draft_device)
                    H_prev_len = H_prev_d1.shape[1]
                    ctx_start = start - H_prev_len

                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    blocks: dict[int, torch.Tensor] = {}
                    ev_s.record()

                    for k_f in top_f:
                        ctx_len_kf = min(k_f + 1, H_prev_len)
                        h_ctx = H_prev_d1[:, :ctx_len_kf, :]

                        kf_pos = position_ids[
                            :, ctx_start : ctx_start + ctx_len_kf + bs
                        ].to(draft_device, non_blocking=True)

                        first_d1 = output_ids[:, start: start + 1].to(
                            draft_device, non_blocking=True
                        )
                        blk = torch.cat([
                            first_d1,
                            torch.full(
                                (1, bs - 1), mask_token_id,
                                dtype=torch.long, device=draft_device,
                            ),
                        ], dim=1)
                        noise = draft_embed(blk)

                        tmp_cache = DynamicCache()
                        logits_d = draft_lm_head(draft_model(
                            target_hidden=h_ctx,
                            noise_embedding=noise,
                            position_ids=kf_pos,
                            past_key_values=tmp_cache,
                            use_cache=True,
                            is_causal=False,
                        )[:, 1 - bs:, :])
                        blocks[k_f] = sample(logits_d)

                    ev_e.record()
                    torch.cuda.synchronize(draft_device)
                    pre_result["blocks"] = blocks
                    pre_result["t_ms"] = ev_s.elapsed_time(ev_e)
                except Exception as exc:
                    pre_result["exc"] = exc

        t_wall_start = time.perf_counter()
        t_a = threading.Thread(target=_run_verify, daemon=True)
        t_b = threading.Thread(target=_run_pre_draft, daemon=True)
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()
        t_wall_ms = (time.perf_counter() - t_wall_start) * 1000

        # Re-raise any exception that occurred inside either thread.
        if "exc" in verify_result:
            raise verify_result["exc"]
        if "exc" in pre_result:
            raise pre_result["exc"]

        # ── Process verify result ─────────────────────────
        v_out = verify_result["out"]
        t_verify_ms = verify_result["t_ms"]
        t_draft_ms = pre_result["t_ms"]

        posterior = sample(v_out.logits, temperature)
        acceptance_length = int(
            (verify_block[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0]
        )
        # Accepted tokens: verify_block[:, :acceptance_length+1] + bonus
        output_ids[:, start: start + acceptance_length + 1] = verify_block[:, :acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        target_cache.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        # Update acceptance predictor
        acceptance_predictor.update(acceptance_length)

        # Fresh hidden states (on target device)
        H_fresh = extract_context_feature(
            v_out.hidden_states, draft_model.target_layer_ids
        )[:, :acceptance_length + 1, :].detach()

        # Cosine similarity between stale and fresh H
        h_sim = _cosine_sim(H_prev, H_fresh)

        # ── Select next draft block ───────────────────────
        # Cache lookup: was our acceptance_length one of the pre-speculated lengths?
        pre_blocks = pre_result["blocks"]  # {k_f: tokens_tensor}
        if acceptance_length in pre_blocks:
            # HIT: pre-speculated block matches the actual acceptance length.
            pre_draft_tokens = pre_blocks[acceptance_length].to(target_device)
            cache_hit = True
        else:
            # MISS: use nearest fan-out key as fallback.
            nearest_k = min(pre_blocks.keys(), key=lambda k: abs(k - acceptance_length))
            pre_draft_tokens = pre_blocks[nearest_k].to(target_device)
            cache_hit = False

        # ── Fast-refinement pass (optional) ──────────────
        # When H_SIM is low (< ~0.85), the stale pre-draft has poor acceptance.
        # One more DFlash forward with fresh H_fresh restores standard DFlash quality
        # at the cost of ~T_draft extra latency (but we already hid T_draft once).
        if fast_refine:
            # Mirror what dflash_generate does: [last_tok, MASK*..] noise input.
            refine_block_ids = torch.cat([
                output_ids[:, start: start + 1],
                torch.full((1, bs - 1), mask_token_id, dtype=torch.long, device=target_device),
            ], dim=1)
            refine_noise = target.model.embed_tokens(refine_block_ids)
            H_fresh_d1 = H_fresh.to(draft_device, non_blocking=True)
            ctx_len_fresh = H_fresh_d1.shape[1]
            ctx_start_fresh = start - ctx_len_fresh
            refine_pos = position_ids[
                :, ctx_start_fresh : ctx_start_fresh + ctx_len_fresh + bs
            ].to(draft_device)
            refine_cache = DynamicCache()
            refine_logits = draft_lm_head(draft_model(
                target_hidden=H_fresh_d1,
                noise_embedding=draft_embed(
                    refine_block_ids.to(draft_device)
                ),
                position_ids=refine_pos,
                past_key_values=refine_cache,
                use_cache=True,
                is_causal=False,
            )[:, 1 - bs:, :])
            pre_draft_tokens = sample(refine_logits).to(target_device)

        # Stale H for next step's pre-draft thread.
        H_prev = H_fresh
        H_prev_d1 = H_prev.to(draft_device, non_blocking=True)

        # ── Record step stats ─────────────────────────────
        ss = StepStats(
            step=step_idx,
            t_verify_ms=t_verify_ms,
            t_draft_ms=t_draft_ms,
            t_wall_ms=t_wall_ms,
            cache_hit=cache_hit,
            acceptance_len=acceptance_length + 1,
            block_size=bs,
            h_cosine_sim=h_sim,
            fan_out_lengths=top_f,
        )
        step_stats_list.append(ss)
        if monitor is not None:
            monitor.record(ss)
        if log_file is not None:
            import json
            log_file.write(json.dumps({
                "step": ss.step,
                "t_verify_ms": round(ss.t_verify_ms, 2),
                "t_draft_ms": round(ss.t_draft_ms, 2),
                "t_wall_ms": round(ss.t_wall_ms, 2),
                "cache_hit": ss.cache_hit,
                "acceptance_len": ss.acceptance_len,
                "h_cosine_sim": round(ss.h_cosine_sim, 5) if ss.h_cosine_sim else None,
                "fan_out_lengths": ss.fan_out_lengths,
            }) + "\n")

        # ── EOS check ────────────────────────────────────
        if stop_token_ids:
            generated = output_ids[0, num_input:start].tolist()
            if any(tok in generated for tok in stop_token_ids):
                break

    if log_file:
        log_file.close()

    # Trim output
    output_ids = output_ids[:, :min(start, max_length)]
    if stop_token_ids:
        stop_t = torch.tensor(stop_token_ids, device=target_device)
        idxs = torch.isin(output_ids[0, num_input:], stop_t).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            output_ids = output_ids[:, :num_input + idxs[0].item() + 1]

    total_decode_s = time.perf_counter() - decode_start
    num_output = output_ids.shape[1] - num_input

    result = SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input,
        num_output_tokens=num_output,
        time_to_first_token=prefill_s,
        time_per_output_token=total_decode_s / max(num_output, 1),
        acceptance_lengths=acceptance_lengths,
    )
    if return_stats:
        result.step_stats = step_stats_list
        rs = RunningStats(steps=step_stats_list)
        result.cache_hit_rate = rs.hit_rate
        result.mean_acceptance = rs.mean_accept
        result.mean_h_cosine_sim = rs.mean_h_sim
        result.draft_latency_saved_ms = rs.draft_latency_saved_ms
        result.throughput_tps = rs.throughput_tps()
    return result
