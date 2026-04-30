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
    # Logits at the bonus position from the previous verify step.
    # Used to predict top-F bonus candidates for the NEXT step's pre-draft.
    # Initialised to None; warmup step populates it before the main loop.
    prev_bonus_logits: Optional[torch.Tensor] = None

    decode_start = time.perf_counter()

    # Warmup: run one standard DFlash step (sequential, no parallelism).
    # This gives us the first real pre_draft_tokens AND populates prev_bonus_logits
    # so the main loop can do bonus-fan-out pre-speculation.
    H_prev_d1 = H_prev.to(draft_device, non_blocking=True)
    _warmup_cache = DynamicCache()
    _last_tok_d1 = output_ids[:, start: start + 1].to(draft_device)
    _warmup_block = torch.cat([
        _last_tok_d1,
        torch.full((1, block_size - 1), mask_token_id, dtype=torch.long, device=draft_device),
    ], dim=1)
    _ctx_len_init = H_prev_d1.shape[1]
    _ctx_start_init = start - _ctx_len_init  # = 0 for first step
    _warmup_pos = position_ids[
        :, _ctx_start_init : _ctx_start_init + _ctx_len_init + block_size
    ].to(draft_device)
    _warmup_logits = draft_lm_head(draft_model(
        target_hidden=H_prev_d1,
        noise_embedding=draft_embed(_warmup_block),
        position_ids=_warmup_pos,
        past_key_values=_warmup_cache,
        use_cache=True,
        is_causal=False,
    )[:, 1 - block_size:, :])
    _warmup_cache.crop(start)
    pre_draft_tokens = sample(_warmup_logits).to(target_device)  # [1, block_size-1]

    # Run warmup verify to get the first prev_bonus_logits before the main loop.
    # (Target is called sequentially here; parallelism starts from step 2.)
    _warmup_verify_block = torch.cat([
        output_ids[:, start: start + 1],
        pre_draft_tokens[:, : block_size - 1],
    ], dim=1)
    _warmup_out = target(
        _warmup_verify_block,
        position_ids=position_ids[:, start: start + block_size],
        past_key_values=target_cache,
        use_cache=True,
        output_hidden_states=True,
    )
    _warmup_posterior = sample(_warmup_out.logits, temperature)
    _warmup_acc = int(
        (_warmup_verify_block[:, 1:] == _warmup_posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0]
    )
    output_ids[:, start: start + _warmup_acc + 1] = _warmup_verify_block[:, :_warmup_acc + 1]
    output_ids[:, start + _warmup_acc + 1] = _warmup_posterior[:, _warmup_acc]
    start += _warmup_acc + 1
    target_cache.crop(start)
    acceptance_lengths.append(_warmup_acc + 1)
    H_prev = extract_context_feature(
        _warmup_out.hidden_states, draft_model.target_layer_ids
    )[:, :_warmup_acc + 1, :].detach()
    # prev_bonus_logits: the target's distribution at the bonus position from this warmup verify.
    # Used in step 1 to predict the top-F bonus candidates for the NEXT block.
    prev_bonus_logits = _warmup_out.logits[:, _warmup_acc, :].detach()
    H_prev_d1 = H_prev.to(draft_device, non_blocking=True)

    # Compute the first real pre_draft using the warmup bonus as last_tok.
    _ctx_len = H_prev_d1.shape[1]
    _ctx_start = start - _ctx_len
    _bonus_tok = output_ids[:, start: start + 1].to(draft_device)
    _first_block = torch.cat([
        _bonus_tok,
        torch.full((1, block_size - 1), mask_token_id, dtype=torch.long, device=draft_device),
    ], dim=1)
    _first_cache = DynamicCache()
    _first_pos = position_ids[:, _ctx_start : _ctx_start + _ctx_len + block_size].to(draft_device)
    _first_logits = draft_lm_head(draft_model(
        target_hidden=H_prev_d1,
        noise_embedding=draft_embed(_first_block),
        position_ids=_first_pos,
        past_key_values=_first_cache,
        use_cache=True,
        is_causal=False,
    )[:, 1 - block_size:, :])
    _first_cache.crop(start)
    pre_draft_tokens = sample(_first_logits).to(target_device)

    # ── Main decode loop ──────────────────────────────────
    while start < max_length:
        bs = min(block_size, max_length - start)
        if bs <= 1:
            break

        step_idx += 1

        # Bonus-fan-out: predict the top-F most likely bonus tokens for the
        # NEXT step using the target's logits at the previous bonus position.
        # Pre-speculate one draft per candidate so we have a correct first token.
        bonus_candidates = torch.topk(
            prev_bonus_logits[0], k=fan_out, dim=-1
        ).indices.tolist()  # list[int], length = fan_out

        # Build the verify block from the pre-speculated draft
        verify_block = torch.cat([
            output_ids[:, start: start + 1],
            pre_draft_tokens[:, : bs - 1],
        ], dim=1)  # [1, bs] on target_device
        block_positions = position_ids[:, start: start + bs]

        # ── Async launch ─────────────────────────────────
        # Thread A  (GPU 0): verify the current block
        # Thread B  (GPU 1): for each predicted bonus candidate, pre-speculate
        #                    DFlash(H_prev, [candidate, MASK…]) — CORRECT first token!

        verify_result: dict = {}
        pre_result: dict = {}

        def _run_verify():
            # torch.inference_mode() is thread-local — must be declared inside thread.
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
            with torch.inference_mode():
                try:
                    torch.cuda.set_device(draft_device)
                    H_prev_len = H_prev_d1.shape[1]
                    ctx_start = start - H_prev_len

                    ev_s = torch.cuda.Event(enable_timing=True)
                    ev_e = torch.cuda.Event(enable_timing=True)
                    # Cache keyed by BONUS TOKEN (int), not acceptance length.
                    blocks: dict[int, torch.Tensor] = {}
                    ev_s.record()

                    for bonus_cand in bonus_candidates:
                        # Use the predicted bonus as position 0 — this is the
                        # key fix: correct first token, stale H.
                        cand_tok = torch.tensor(
                            [[bonus_cand]], dtype=torch.long, device=draft_device
                        )
                        blk = torch.cat([
                            cand_tok,
                            torch.full(
                                (1, bs - 1), mask_token_id,
                                dtype=torch.long, device=draft_device,
                            ),
                        ], dim=1)
                        noise = draft_embed(blk)

                        ctx_len_this = H_prev_len
                        kf_pos = position_ids[
                            :, ctx_start : ctx_start + ctx_len_this + bs
                        ].to(draft_device, non_blocking=True)

                        tmp_cache = DynamicCache()
                        logits_d = draft_lm_head(draft_model(
                            target_hidden=H_prev_d1,
                            noise_embedding=noise,
                            position_ids=kf_pos,
                            past_key_values=tmp_cache,
                            use_cache=True,
                            is_causal=False,
                        )[:, 1 - bs:, :])
                        blocks[bonus_cand] = sample(logits_d)

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

        # Fresh hidden states
        H_fresh = extract_context_feature(
            v_out.hidden_states, draft_model.target_layer_ids
        )[:, :acceptance_length + 1, :].detach()

        h_sim = _cosine_sim(H_prev, H_fresh)

        # ── Bonus-fan-out cache lookup ────────────────────
        # After `start += acceptance_length + 1`, output_ids[start] is the bonus
        # token — the new last_tok that will become position 0 of the NEXT block.
        # output_ids[start - 1] is the last *accepted draft* token, NOT the bonus.
        actual_bonus = int(output_ids[0, start].item())   # bonus = new last_tok
        pre_blocks = pre_result["blocks"]  # {bonus_candidate: draft_tokens}

        if actual_bonus in pre_blocks:
            # HIT: pre-draft was generated with correct first token (actual_bonus).
            pre_draft_tokens = pre_blocks[actual_bonus].to(target_device)
            cache_hit = True
        else:
            # MISS: run one DFlash pass with stale H + correct first token.
            # H_prev.shape[1] = acceptance_length + 1, so
            # ctx_start = start - H_prev.shape[1] = old_start.
            miss_tok_d1 = output_ids[:, start: start + 1].to(draft_device)
            miss_blk = torch.cat([
                miss_tok_d1,
                torch.full((1, bs - 1), mask_token_id, dtype=torch.long, device=draft_device),
            ], dim=1)
            _ctx_len_miss = H_prev_d1.shape[1]
            _ctx_start_miss = start - _ctx_len_miss   # = old_start
            miss_pos = position_ids[
                :, _ctx_start_miss : _ctx_start_miss + _ctx_len_miss + bs
            ].to(draft_device)
            miss_cache = DynamicCache()
            miss_logits = draft_lm_head(draft_model(
                target_hidden=H_prev_d1,
                noise_embedding=draft_embed(miss_blk),
                position_ids=miss_pos,
                past_key_values=miss_cache,
                use_cache=True,
                is_causal=False,
            )[:, 1 - bs:, :])
            pre_draft_tokens = sample(miss_logits).to(target_device)
            cache_hit = False

        # ── Optional fast-refine with fresh H ────────────
        if fast_refine:
            refine_tok_d1 = output_ids[:, start: start + 1].to(draft_device)
            refine_blk = torch.cat([
                refine_tok_d1,
                torch.full((1, bs - 1), mask_token_id, dtype=torch.long, device=draft_device),
            ], dim=1)
            H_fresh_d1 = H_fresh.to(draft_device, non_blocking=True)
            _ctx_len_rf = H_fresh_d1.shape[1]
            _ctx_start_rf = start - _ctx_len_rf   # = old_start
            refine_pos = position_ids[
                :, _ctx_start_rf : _ctx_start_rf + _ctx_len_rf + bs
            ].to(draft_device)
            refine_cache = DynamicCache()
            refine_logits = draft_lm_head(draft_model(
                target_hidden=H_fresh_d1,
                noise_embedding=draft_embed(refine_blk),
                position_ids=refine_pos,
                past_key_values=refine_cache,
                use_cache=True,
                is_causal=False,
            )[:, 1 - bs:, :])
            pre_draft_tokens = sample(refine_logits).to(target_device)

        # Update prev_bonus_logits for next step's bonus prediction.
        prev_bonus_logits = v_out.logits[:, acceptance_length, :].detach()

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
            fan_out_lengths=bonus_candidates,
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
