"""True Saguaro two-GPU asynchronous speculative decoding.

GPU layout
----------
  device_target  (cuda:0) — Qwen3-4B target model (verifier).
  device_draft   (cuda:1) — Draft model (primary + backup speculator).
                            SSD path     : Qwen3-0.6B autoregressive LM.
                            Combined path: DFlash model with embed_tokens /
                                           lm_head copies on GPU 1.

Async overlap (true Saguaro)
-----------------------------
  Round N:
    a. Submit outcome-prefetch job to draft worker thread (GPU 1, non-blocking).
    b. Run target verify on GPU 0 (main thread, concurrent with a).
    c. Collect draft worker result — typically already finished by now.
    d. Look up (accepted_len, bonus_token) in outcome cache.
         HIT  — prefetched entry is ready; zero extra draft latency.
         MISS — backup speculator: fresh draft run using verified target_hidden.
    e. Loop (goto a with new current_entry).

Cache-miss policy (paper §3.2)
--------------------------------
  Primary    : outcome cache keyed by (accepted_len, bonus_token),
               speculation_fanout branches per acceptance length.
  Backup     : same draft model, single fresh pass.  For the combined path
               the backup uses the freshly verified target_hidden (better
               quality than the primary's stale features).
  Emergency  : if token budget is exhausted at backup, emit one AR token
               from the target directly.
"""
from __future__ import annotations

import argparse
import copy
import json
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

import numpy as np
import torch
from loguru import logger
from rich import print
from rich.table import Table
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .benchmark import DATASETS, _apply_chat_template, _limit_dataset, load_and_process_dataset
from .model import (
    DFlashDraftModel,
    SDPrefetchEntry,
    _build_sd_outcome_cache,
    _generate_sd_prefetch_entry,
    extract_context_feature,
    sample,
)
from .ssd_dflash_accel import (
    DFlashPrefetchEntry,
    SampleSummary,
    _aggregate,
    _build_dflash_outcome_cache,
    _generate_dflash_prefetch_entry,
    _hist,
    _print_aggregate,
    _request_summary,
    _safe_cosine,
)


# ---------------------------------------------------------------------------
# GPU-1 proxy for DFlash helpers (combined path)
# ---------------------------------------------------------------------------

class _TargetProxy:
    """Minimal stand-in for the target model used by DFlash prefetch helpers.

    DFlash prefetch helpers need:
      target.device              — device for new tensors
      target.lm_head.weight      — (vocab_size, hidden_size); used for shape
      target.model.embed_tokens  — token → embedding lookup
      target.lm_head(hidden)     — project draft hidden → logits

    This proxy holds GPU-1 copies of embed_tokens and lm_head so DFlash
    compute stays entirely on device_draft.
    """

    def __init__(
        self,
        embed_tokens: torch.nn.Module,
        lm_head: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.model = SimpleNamespace(embed_tokens=embed_tokens)
        self.lm_head = lm_head
        self.device = device


# ---------------------------------------------------------------------------
# Async prefetch worker
# ---------------------------------------------------------------------------

@dataclass
class _PrefetchJob:
    fn: Callable
    args: tuple
    kwargs: dict


class AsyncPrefetchWorker:
    """Background thread that runs outcome-cache prefetch on device_draft.

    Usage
    -----
      worker.submit(fn, *args, **kwargs)   # non-blocking, returns immediately
      result = worker.collect()            # blocks until job is done

    At most one outstanding job at a time.  Submitting while a prior job is
    not yet collected is a programming error and will raise.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._stream = torch.cuda.Stream(device=device)
        self._job_event = threading.Event()
        self._result_event = threading.Event()
        self._job: Optional[_PrefetchJob] = None
        self._result: Any = None
        self._error: Optional[BaseException] = None
        self._stop = False
        self._worker_thread = threading.Thread(
            target=self._loop, daemon=True, name="saguaro-draft-worker"
        )
        self._worker_thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn: Callable, *args: Any, **kwargs: Any) -> None:
        """Queue a single prefetch job.  Non-blocking."""
        if not self._result_event.is_set() and self._job is not None:
            raise RuntimeError("AsyncPrefetchWorker.submit called with outstanding uncollected job")
        self._result_event.clear()
        self._result = None
        self._error = None
        self._job = _PrefetchJob(fn=fn, args=args, kwargs=kwargs)
        self._job_event.set()

    def collect(self) -> Any:
        """Block until the outstanding job is done, then return its result."""
        self._result_event.wait()
        if self._error is not None:
            raise self._error
        with torch.cuda.device(self.device):
            consumer_stream = torch.cuda.current_stream(self.device)
            consumer_stream.wait_stream(self._stream)
            self._record_stream_recursive(self._result, consumer_stream)
        return self._result

    def stop(self) -> None:
        self._stop = True
        self._job_event.set()
        self._worker_thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            self._job_event.wait()
            self._job_event.clear()
            if self._stop:
                return
            job = self._job
            if job is None:
                continue
            try:
                with torch.inference_mode(), torch.cuda.stream(self._stream):
                    result = job.fn(*job.args, **job.kwargs)
                    self._stream.synchronize()
                self._result = result
            except Exception as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._result_event.set()

    @staticmethod
    def _record_stream_recursive(obj: Any, stream: torch.cuda.Stream) -> None:
        if obj is None:
            return
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda:
                obj.record_stream(stream)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                AsyncPrefetchWorker._record_stream_recursive(item, stream)
            return
        if isinstance(obj, dict):
            for value in obj.values():
                AsyncPrefetchWorker._record_stream_recursive(value, stream)
            return
        if isinstance(obj, DFlashPrefetchEntry):
            AsyncPrefetchWorker._record_stream_recursive(obj.block_output_ids, stream)
            AsyncPrefetchWorker._record_stream_recursive(obj.source_target_hidden, stream)
            AsyncPrefetchWorker._record_stream_recursive(obj.draft_hidden, stream)
            AsyncPrefetchWorker._record_stream_recursive(obj.draft_logits, stream)
            AsyncPrefetchWorker._record_stream_recursive(obj.next_bonus_logits, stream)
            return
        if isinstance(obj, SDPrefetchEntry):
            AsyncPrefetchWorker._record_stream_recursive(obj.step_logits, stream)
            return


# ---------------------------------------------------------------------------
# Shared trace dataclass with cache-hit field
# ---------------------------------------------------------------------------

@dataclass
class SaguaroStepTrace:
    request_id: str
    step_idx: int
    accepted_tokens: int
    proposed_tokens: int
    acceptance_ratio: float
    step_ms: float
    cumulative_decode_tokens: int
    cumulative_decode_tps: float
    cache_hit: bool
    hidden_drift_cosine: Optional[float]


def _clone_dflash_entry(entry: DFlashPrefetchEntry) -> DFlashPrefetchEntry:
    return DFlashPrefetchEntry(
        block_output_ids=entry.block_output_ids.clone(),
        source_target_hidden=entry.source_target_hidden.clone(),
        draft_hidden=entry.draft_hidden.clone(),
        draft_logits=entry.draft_logits.clone(),
        next_bonus_logits=entry.next_bonus_logits.clone(),
    )


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _validate_token_ids(token_ids: torch.Tensor, *, vocab_size: int, context: str) -> None:
    if token_ids.dtype != torch.long:
        raise ValueError(f"{context}: token_ids must be torch.long, got {token_ids.dtype}")
    bad = (token_ids < 0) | (token_ids >= vocab_size)
    if not bool(bad.any().item()):
        return
    bad_vals = token_ids[bad]
    unique_bad = torch.unique(bad_vals).tolist()
    sample_bad = unique_bad[:8]
    tok_min = int(token_ids.min().item())
    tok_max = int(token_ids.max().item())
    raise ValueError(
        f"{context}: found out-of-range token ids; "
        f"min={tok_min} max={tok_max} vocab_size={vocab_size} "
        f"examples={sample_bad}"
    )


# ---------------------------------------------------------------------------
# SSD path: Qwen3-4B (GPU 0) + Qwen3-0.6B (GPU 1), true async
# ---------------------------------------------------------------------------

@torch.inference_mode()
def saguaro_ssd_generate(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    num_draft_tokens: int,
    speculation_fanout: int,
    request_id: str,
    trace_fh,
) -> SimpleNamespace:
    """True async Saguaro SSD generate.

    target is on device_target (cuda:0).
    draft  is on device_draft  (cuda:1).
    input_ids must already be on device_target.
    """
    device_target = input_ids.device
    device_draft = next(draft.parameters()).device

    # Move prompt to GPU 1 once (stays stable throughout).
    input_ids_g1 = input_ids.to(device_draft)

    # ---- Prefill on GPU 0 -----------------------------------------------
    target_cache = DynamicCache()
    t0 = _cuda_time()
    t_out = target(input_ids, past_key_values=target_cache, use_cache=True, logits_to_keep=1)
    first_tok = int(sample(t_out.logits[:, -1:], temperature)[0, 0])
    time_to_first_token = _cuda_time() - t0
    decode_start = _cuda_time()

    output: list[int] = [first_tok]
    acceptance_lengths: list[int] = [1]
    target_cache_pos = input_ids.shape[1]
    step_idx = 0
    traces: list[SaguaroStepTrace] = []

    # ---- Prime the pipeline: first draft entry synchronously on GPU 1 ----
    initial_prefix_g1 = torch.cat(
        [input_ids_g1, torch.tensor([[first_tok]], dtype=torch.long, device=device_draft)],
        dim=1,
    )
    current_entry: SDPrefetchEntry = _generate_sd_prefetch_entry(
        draft, initial_prefix_g1, num_draft_tokens=num_draft_tokens, temperature=temperature
    )

    worker = AsyncPrefetchWorker(device=device_draft)
    stop_reason = "max_new_tokens"

    try:
        while len(output) < max_new_tokens:
            n = min(num_draft_tokens, max_new_tokens - len(output))
            # Guard: entry may have too few ids if budget was trimmed at generation.
            if len(current_entry.draft_ids) < n:
                committed_prefix_g1 = torch.cat(
                    [input_ids_g1, torch.tensor([output], dtype=torch.long, device=device_draft)],
                    dim=1,
                )
                current_entry = _generate_sd_prefetch_entry(
                    draft, committed_prefix_g1, num_draft_tokens=n, temperature=temperature
                )

            committed_snapshot = list(output)
            draft_ids = current_entry.draft_ids[:n]

            # (a) Submit prefetch job on GPU 1 (non-blocking).
            worker.submit(
                _build_sd_outcome_cache,
                draft,
                input_ids_g1,
                committed_snapshot,
                current_entry,
                max_new_tokens=max_new_tokens,
                num_draft_tokens=num_draft_tokens,
                temperature=temperature,
                speculation_fanout=speculation_fanout,
            )

            # (b) Verify on GPU 0 (concurrent with (a)).
            t_step = _cuda_time()
            curr_tok_g0 = torch.tensor([[output[-1]]], dtype=torch.long, device=device_target)
            draft_ids_g0 = torch.tensor([draft_ids], dtype=torch.long, device=device_target)
            verify_input = torch.cat([curr_tok_g0, draft_ids_g0], dim=1)
            old_pos = target_cache_pos
            t_out = target(verify_input, past_key_values=target_cache, use_cache=True)
            target_cache_pos += n + 1
            t_preds = sample(t_out.logits, temperature)[0].tolist()

            # (c) Collect outcome cache (typically already done on GPU 1).
            outcome_cache = worker.collect()

            # (d) Compute acceptance.
            accepted = 0
            for i in range(n):
                if draft_ids[i] == t_preds[i]:
                    accepted += 1
                else:
                    break
            bonus = t_preds[accepted]

            new_toks = (draft_ids[:accepted] + [bonus])[: max_new_tokens - len(output)]
            output.extend(new_toks)
            acceptance_lengths.append(len(new_toks))

            # KV rollback: keep prompt + committed (including accepted new) + curr_tok.
            target_cache.crop(old_pos + accepted + 1)
            target_cache_pos = old_pos + accepted + 1

            # (e) Cache lookup.
            next_entry = outcome_cache.get((accepted, bonus))
            cache_hit = next_entry is not None
            if not cache_hit:
                # Backup speculator: fresh draft run on GPU 1.
                backup_prefix_g1 = torch.cat(
                    [input_ids_g1, torch.tensor([list(output)], dtype=torch.long, device=device_draft)],
                    dim=1,
                )
                backup_n = min(num_draft_tokens, max_new_tokens - len(output))
                if backup_n > 0:
                    next_entry = _generate_sd_prefetch_entry(
                        draft, backup_prefix_g1, num_draft_tokens=backup_n, temperature=temperature
                    )

            step_ms = (_cuda_time() - t_step) * 1000.0
            decode_tokens = max(1, len(output) - 1)  # exclude first_tok counted in prefill
            decode_tps = decode_tokens / max(_cuda_time() - decode_start, 1e-9)

            trace = SaguaroStepTrace(
                request_id=request_id,
                step_idx=step_idx,
                accepted_tokens=accepted + 1,
                proposed_tokens=n,
                acceptance_ratio=(accepted + 1) / max(n, 1),
                step_ms=step_ms,
                cumulative_decode_tokens=decode_tokens,
                cumulative_decode_tps=decode_tps,
                cache_hit=cache_hit,
                hidden_drift_cosine=None,
            )
            traces.append(trace)
            if trace_fh is not None:
                trace_fh.write(json.dumps(asdict(trace)) + "\n")

            step_idx += 1
            current_entry = next_entry

            if stop_token_ids and any(t in output for t in stop_token_ids):
                stop_reason = "eos"
                break

            if current_entry is None:
                # Emergency: budget exhausted for backup, run one AR token from target.
                em_input = torch.tensor([[output[-1]]], dtype=torch.long, device=device_target)
                em_out = target(em_input, past_key_values=target_cache, use_cache=True, logits_to_keep=1)
                em_tok = int(sample(em_out.logits[:, -1:], temperature)[0, 0])
                output.append(em_tok)
                target_cache_pos += 1
                # Re-prime pipeline.
                re_prefix_g1 = torch.cat(
                    [input_ids_g1, torch.tensor([output], dtype=torch.long, device=device_draft)],
                    dim=1,
                )
                re_n = min(num_draft_tokens, max_new_tokens - len(output))
                if re_n <= 0:
                    break
                current_entry = _generate_sd_prefetch_entry(
                    draft, re_prefix_g1, num_draft_tokens=re_n, temperature=temperature
                )
    finally:
        worker.stop()

    # Truncate to budget.
    output = output[:max_new_tokens]

    output_ids = torch.cat(
        [input_ids, torch.tensor([output], dtype=torch.long, device=device_target)], dim=1
    )
    if stop_token_ids:
        stop_ids_t = torch.tensor(stop_token_ids, device=device_target)
        stop_idx = torch.isin(output_ids[0, input_ids.shape[1]:], stop_ids_t).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, : input_ids.shape[1] + stop_idx[0] + 1]
            stop_reason = "eos"

    num_output_tokens = output_ids.shape[1] - input_ids.shape[1]
    total_decode = _cuda_time() - decode_start
    cache_hits = sum(1 for t in traces if t.cache_hit)

    return SimpleNamespace(
        output_ids=output_ids,
        traces=traces,
        stop_reason=stop_reason,
        num_input_tokens=input_ids.shape[1],
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        hidden_drifts=[],
        cache_hits=cache_hits,
        cache_total=len(traces),
    )


# ---------------------------------------------------------------------------
# Combined path: Qwen3-4B (GPU 0) + DFlash (GPU 1, w/ embed/lm_head copies)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def saguaro_combined_generate(
    draft: DFlashDraftModel,
    target: torch.nn.Module,
    target_proxy_g1: _TargetProxy,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: int,
    mask_token_id: int,
    speculation_fanout: int,
    request_id: str,
    trace_fh,
) -> SimpleNamespace:
    """True async Saguaro combined generate (DFlash draft + target verifier).

    target        is on device_target (cuda:0).
    draft         is on device_draft  (cuda:1).
    target_proxy_g1 wraps embed_tokens + lm_head copies on device_draft.
    input_ids must already be on device_target.
    """
    device_target = input_ids.device
    device_draft = next(draft.parameters()).device
    vocab_size = int(target.lm_head.weight.shape[0])

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=device_target,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device_target).unsqueeze(0)
    target_cache = DynamicCache()

    # ---- Prefill on GPU 0 -----------------------------------------------
    t0 = _cuda_time()
    prefill_out = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=target_cache,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )
    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(prefill_out.logits, temperature)
    target_hidden_g0 = extract_context_feature(prefill_out.hidden_states, draft.target_layer_ids)
    time_to_first_token = _cuda_time() - t0
    decode_start = _cuda_time()

    start = num_input_tokens
    acceptance_lengths: list[int] = []
    hidden_drifts: list[float] = []
    traces: list[SaguaroStepTrace] = []
    step_idx = 0
    stop_reason = "max_new_tokens"
    prev_hidden_last = target_hidden_g0[:, -1, :].detach()

    # ---- Prime pipeline: first draft entry synchronously on GPU 1 ---------
    first_block_len = min(block_size, max_length - start)
    target_hidden_g1 = target_hidden_g0.to(device_draft)
    current_entry: DFlashPrefetchEntry = _generate_dflash_prefetch_entry(
        draft,
        target_proxy_g1,
        target_hidden_g1,
        first_token=int(output_ids[0, start].item()),
        block_len=first_block_len,
        mask_token_id=mask_token_id,
        start_position=start,
        temperature=temperature,
    )

    worker = AsyncPrefetchWorker(device=device_draft)

    try:
        while start < max_length:
            generated_before_round = start - num_input_tokens
            block_len = min(block_size, max_length - start)
            if block_len <= 0:
                break

            if current_entry.block_output_ids.shape[1] != block_len:
                # Block size mismatch (last block): regenerate synchronously.
                target_hidden_g1 = target_hidden_g0.to(device_draft)
                current_entry = _generate_dflash_prefetch_entry(
                    draft,
                    target_proxy_g1,
                    target_hidden_g1,
                    first_token=int(output_ids[0, start].item()),
                    block_len=block_len,
                    mask_token_id=mask_token_id,
                    start_position=start,
                    temperature=temperature,
                )

            # Stage verifier input on GPU 0 before launching prefetch kernels on GPU 1.
            block_ids_g0 = current_entry.block_output_ids.to(device_target)
            _validate_token_ids(
                block_ids_g0,
                vocab_size=vocab_size,
                context=f"verify_input req={request_id} step={step_idx}",
            )

            # (a) Submit outcome-prefetch job on GPU 1 (non-blocking).
            # Clone so worker and verifier never read the same backing storage concurrently.
            worker_input_entry = _clone_dflash_entry(current_entry)
            worker.submit(
                _build_dflash_outcome_cache,
                draft,
                target_proxy_g1,
                worker_input_entry,
                generated_before_round=generated_before_round,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                mask_token_id=mask_token_id,
                start_position=start,
                temperature=temperature,
                speculation_fanout=speculation_fanout,
            )

            # (b) Verify on GPU 0 (concurrent with (a)).
            t_step = _cuda_time()
            block_position_ids = position_ids[:, start : start + block_len]

            verify_out = target(
                block_ids_g0,
                position_ids=block_position_ids,
                past_key_values=target_cache,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = sample(verify_out.logits, temperature)

            # (c) Collect outcome cache.
            outcome_cache = worker.collect()

            # (d) Compute acceptance.
            if block_len > 1:
                acceptance_len = int(
                    (block_ids_g0[:, 1:block_len] == posterior[:, :block_len - 1])
                    .cumprod(dim=1)
                    .sum(dim=1)[0]
                    .item()
                )
            else:
                acceptance_len = 0

            accepted_tokens = acceptance_len + 1
            bonus_token = int(posterior[0, acceptance_len].item())

            output_ids[:, start : start + accepted_tokens] = block_ids_g0[:, :accepted_tokens]
            if start + accepted_tokens < output_ids.shape[1]:
                output_ids[:, start + accepted_tokens] = posterior[:, acceptance_len]

            start += accepted_tokens
            target_cache.crop(start)
            acceptance_lengths.append(accepted_tokens)

            # Extract fresh target_hidden on GPU 0 for backup speculator.
            target_hidden_g0 = extract_context_feature(
                verify_out.hidden_states, draft.target_layer_ids
            )[:, :accepted_tokens, :]
            curr_last = target_hidden_g0[:, -1, :].detach()
            hidden_drift = _safe_cosine(prev_hidden_last, curr_last)
            prev_hidden_last = curr_last
            if hidden_drift is not None:
                hidden_drifts.append(hidden_drift)

            # (e) Cache lookup.
            next_entry = outcome_cache.get((accepted_tokens, bonus_token))
            cache_hit = next_entry is not None
            if next_entry is not None:
                # Clone onto main-thread ownership to avoid allocator hazards with worker-produced tensors.
                next_entry = _clone_dflash_entry(next_entry)
            if not cache_hit:
                # Backup speculator: fresh DFlash run using verified target_hidden (GPU 1).
                backup_start = start
                backup_block_len = min(block_size, max_length - backup_start)
                if backup_block_len > 0:
                    target_hidden_g1 = target_hidden_g0.to(device_draft)
                    first_token_backup = int(output_ids[0, backup_start].item())
                    if first_token_backup == mask_token_id:
                        # bonus token hasn't been written yet — use it
                        first_token_backup = bonus_token
                    next_entry = _generate_dflash_prefetch_entry(
                        draft,
                        target_proxy_g1,
                        target_hidden_g1,
                        first_token=first_token_backup,
                        block_len=backup_block_len,
                        mask_token_id=mask_token_id,
                        start_position=backup_start,
                        temperature=temperature,
                    )

            step_ms = (_cuda_time() - t_step) * 1000.0
            decode_tokens = max(1, start - num_input_tokens)
            decode_tps = decode_tokens / max(_cuda_time() - decode_start, 1e-9)

            trace = SaguaroStepTrace(
                request_id=request_id,
                step_idx=step_idx,
                accepted_tokens=accepted_tokens,
                proposed_tokens=block_len,
                acceptance_ratio=accepted_tokens / max(block_len, 1),
                step_ms=step_ms,
                cumulative_decode_tokens=decode_tokens,
                cumulative_decode_tps=decode_tps,
                cache_hit=cache_hit,
                hidden_drift_cosine=hidden_drift,
            )
            traces.append(trace)
            if trace_fh is not None:
                trace_fh.write(json.dumps(asdict(trace)) + "\n")

            step_idx += 1
            current_entry = next_entry

            if stop_token_ids is not None and any(
                tok in output_ids[:, num_input_tokens:] for tok in stop_token_ids
            ):
                stop_reason = "eos"
                break

            if current_entry is None:
                # Emergency: AR token from target.
                em_tok_g0 = output_ids[:, start - 1 : start]
                em_out = target(
                    em_tok_g0,
                    position_ids=position_ids[:, start - 1 : start],
                    past_key_values=target_cache,
                    use_cache=True,
                    logits_to_keep=1,
                    output_hidden_states=True,
                )
                em_token = int(sample(em_out.logits[:, -1:], temperature)[0, 0])
                output_ids[:, start] = em_token
                target_hidden_g0 = extract_context_feature(
                    em_out.hidden_states, draft.target_layer_ids
                )
                start += 1
                # Re-prime with fresh entry from backup.
                re_block_len = min(block_size, max_length - start)
                if re_block_len <= 0:
                    break
                target_hidden_g1 = target_hidden_g0.to(device_draft)
                current_entry = _generate_dflash_prefetch_entry(
                    draft,
                    target_proxy_g1,
                    target_hidden_g1,
                    first_token=em_token,
                    block_len=re_block_len,
                    mask_token_id=mask_token_id,
                    start_position=start,
                    temperature=temperature,
                )
    finally:
        worker.stop()

    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_ids_t = torch.tensor(stop_token_ids, device=device_target)
        stop_idx = torch.isin(output_ids[0, num_input_tokens:], stop_ids_t).nonzero(as_tuple=True)[0]
        if stop_idx.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_idx[0] + 1]
            stop_reason = "eos"

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode = _cuda_time() - decode_start
    cache_hits = sum(1 for t in traces if t.cache_hit)

    return SimpleNamespace(
        output_ids=output_ids,
        traces=traces,
        stop_reason=stop_reason,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
        hidden_drifts=hidden_drifts,
        cache_hits=cache_hits,
        cache_total=len(traces),
    )


# ---------------------------------------------------------------------------
# Aggregate / reporting helpers
# ---------------------------------------------------------------------------

def _saguaro_aggregate(summaries: list[dict], block_size: int, path: str) -> dict:
    base = _aggregate(summaries, block_size)
    total_hits = sum(s.get("cache_hits", 0) for s in summaries)
    total_rounds = sum(s.get("cache_total", 0) for s in summaries)
    base["cache_hit_rate"] = total_hits / max(total_rounds, 1)
    base["path"] = path
    return base


def _print_saguaro_aggregate(agg: dict) -> None:
    _print_aggregate(agg)
    rate = agg.get("cache_hit_rate", 0.0)
    print(f"  Cache hit rate : {rate * 100:.1f}%  ({agg['path']} path)")


def _saguaro_request_summary(request_id: str, dataset: str, result) -> dict:
    base = _request_summary(request_id, dataset, result)
    d = asdict(base)
    d["cache_hits"] = result.cache_hits
    d["cache_total"] = result.cache_total
    return d


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_attn_impl() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        logger.warning(
            "flash-attn not installed — using torch.sdpa. "
            "Install for best performance: pip install flash-attn --no-build-isolation"
        )
        return "sdpa"


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

def _run_dataset(args: argparse.Namespace) -> None:
    device_target = torch.device("cuda:0")
    device_draft = torch.device("cuda:1")

    attn_impl = _resolve_attn_impl()

    logger.info(f"Loading target model on cuda:0: {args.model}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device_target).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    target_proxy_g1: Optional[_TargetProxy] = None

    if args.path in ("combined", "dflash"):
        logger.info(f"Loading DFlash draft on cuda:1: {args.draft_model}")
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_model,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        ).to(device_draft).eval()

        logger.info("Copying embed_tokens + lm_head to cuda:1 for DFlash proxy...")
        embed_g1 = copy.deepcopy(target.model.embed_tokens).to(device_draft).eval()
        lm_head_g1 = copy.deepcopy(target.lm_head).to(device_draft).eval()
        target_proxy_g1 = _TargetProxy(embed_g1, lm_head_g1, device_draft)

        block_size = args.block_size if args.block_size is not None else int(draft_model.block_size)
        mask_token_id = args.mask_token_id if args.mask_token_id is not None else int(draft_model.mask_token_id)
    else:
        logger.info(f"Loading SSD draft on cuda:1: {args.ssd_draft}")
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.ssd_draft,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        ).to(device_draft).eval()
        block_size = args.num_draft_tokens
        mask_token_id = None  # unused in SSD path

    dataset = load_and_process_dataset(args.dataset)
    dataset = _limit_dataset(dataset, args.max_samples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_path = out_dir / "trace.jsonl"
    samples_path = out_dir / "samples.jsonl"

    local_summaries: list[dict] = []

    with open(traces_path, "w") as trace_fh, open(samples_path, "w") as samples_fh:
        for idx in tqdm(range(len(dataset)), desc=f"Saguaro ({args.path}) eval"):
            item = dataset[idx]
            user_turn = item["turns"][0]
            messages = [{"role": "user", "content": user_turn}]
            prompt = _apply_chat_template(
                tokenizer,
                messages,
                enable_thinking=args.enable_thinking,
            )
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device_target)
            req_id = f"{args.dataset}:{idx}"

            if args.path in ("combined", "dflash"):
                result = saguaro_combined_generate(
                    draft_model,
                    target,
                    target_proxy_g1,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    block_size=block_size,
                    mask_token_id=mask_token_id,
                    speculation_fanout=args.speculation_fanout,
                    request_id=req_id,
                    trace_fh=trace_fh,
                )
            else:
                result = saguaro_ssd_generate(
                    draft_model,
                    target,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    num_draft_tokens=args.num_draft_tokens,
                    speculation_fanout=args.speculation_fanout,
                    request_id=req_id,
                    trace_fh=trace_fh,
                )

            summary_dict = _saguaro_request_summary(req_id, args.dataset, result)
            local_summaries.append(summary_dict)
            samples_fh.write(json.dumps(summary_dict) + "\n")

    agg = _saguaro_aggregate(local_summaries, block_size, args.path)

    metadata = {
        "timestamp": _timestamp(),
        "model": args.model,
        "ssd_draft": getattr(args, "ssd_draft", None),
        "draft_model": getattr(args, "draft_model", None),
        "path": args.path,
        "dataset": args.dataset,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "block_size": block_size,
        "num_draft_tokens": args.num_draft_tokens,
        "speculation_fanout": args.speculation_fanout,
        "attention_implementation": attn_impl,
        "gpu_layout": "cuda:0=target, cuda:1=draft",
    }

    with open(out_dir / "aggregate.json", "w") as fh:
        json.dump({"metadata": metadata, "aggregate": agg}, fh, indent=2)

    _print_saguaro_aggregate(agg)
    print(f"\nSaved -> {out_dir / 'aggregate.json'}")
    print(f"Traces -> {traces_path}")
    print(f"Samples -> {samples_path}")


# ---------------------------------------------------------------------------
# Prompt mode
# ---------------------------------------------------------------------------

def _run_prompt(args: argparse.Namespace) -> None:
    device_target = torch.device("cuda:0")
    device_draft = torch.device("cuda:1")
    attn_impl = _resolve_attn_impl()

    target = AutoModelForCausalLM.from_pretrained(
        args.model,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    ).to(device_target).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.path in ("combined", "dflash"):
        draft_model = DFlashDraftModel.from_pretrained(
            args.draft_model,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        ).to(device_draft).eval()
        embed_g1 = copy.deepcopy(target.model.embed_tokens).to(device_draft).eval()
        lm_head_g1 = copy.deepcopy(target.lm_head).to(device_draft).eval()
        target_proxy_g1 = _TargetProxy(embed_g1, lm_head_g1, device_draft)
        block_size = args.block_size or int(draft_model.block_size)
        mask_token_id = args.mask_token_id or int(draft_model.mask_token_id)
    else:
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.ssd_draft,
            attn_implementation=attn_impl,
            dtype=torch.bfloat16,
        ).to(device_draft).eval()
        target_proxy_g1 = None
        block_size = args.num_draft_tokens
        mask_token_id = None

    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device_target)

    if args.path in ("combined", "dflash"):
        result = saguaro_combined_generate(
            draft_model, target, target_proxy_g1, input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
            block_size=block_size,
            mask_token_id=mask_token_id,
            speculation_fanout=args.speculation_fanout,
            request_id="interactive",
            trace_fh=None,
        )
    else:
        result = saguaro_ssd_generate(
            draft_model, target, input_ids,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=args.temperature,
            num_draft_tokens=args.num_draft_tokens,
            speculation_fanout=args.speculation_fanout,
            request_id="interactive",
            trace_fh=None,
        )

    n_hits = result.cache_hits
    n_total = result.cache_total
    hit_rate = n_hits / max(n_total, 1) * 100
    tps = 1.0 / max(result.time_per_output_token, 1e-9)
    mean_acc = float(np.mean(result.acceptance_lengths)) if result.acceptance_lengths else 0.0

    table = Table(title=f"Saguaro ({args.path}) — Interactive", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Output tokens", str(result.num_output_tokens))
    table.add_row("TTFT (s)", f"{result.time_to_first_token:.4f}")
    table.add_row("Decode tok/s", f"{tps:.2f}")
    table.add_row("Mean acceptance", f"{mean_acc:.2f} / {block_size}")
    table.add_row("Cache hit rate", f"{hit_rate:.1f}%  ({n_hits}/{n_total})")
    table.add_row("Stop reason", result.stop_reason)
    print(table)

    decoded = tokenizer.decode(result.output_ids[0], skip_special_tokens=True)
    print("\nGenerated text:\n")
    print(decoded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "True Saguaro two-GPU async speculative decoding. "
            "GPU 0 = target verifier, GPU 1 = primary+backup speculator."
        )
    )
    p.add_argument("--model", default="Qwen/Qwen3-4B", help="Target model HF id.")
    p.add_argument(
        "--path",
        choices=["ssd", "combined", "dflash"],
        default="combined",
        help=(
            "ssd: SSD-only (Qwen3-0.6B AR draft). "
            "combined/dflash: DFlash diffusion draft."
        ),
    )
    # SSD draft
    p.add_argument("--ssd-draft", default="Qwen/Qwen3-0.6B", help="SSD AR draft model.")
    p.add_argument("--num-draft-tokens", type=int, default=5, help="SSD draft tokens per round.")
    # DFlash draft
    p.add_argument("--draft-model", default="z-lab/Qwen3-4B-DFlash-b16", help="DFlash draft model.")
    p.add_argument("--block-size", type=int, default=None, help="DFlash block size (default from model).")
    p.add_argument("--mask-token-id", type=int, default=None, help="DFlash mask token (default from model).")
    # Saguaro params
    p.add_argument(
        "--speculation-fanout",
        type=int,
        default=1,
        help=(
            "Outcome branches to prefetch per acceptance length. "
            "fanout=1 → one bonus candidate per length; "
            "fanout=2 → two candidates. Higher = better hit rate but more GPU-1 work."
        ),
    )
    # Dataset / generation
    p.add_argument("--dataset", default="gsm8k", choices=list(DATASETS))
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--enable-thinking", action="store_true")
    # Output
    p.add_argument("--out-dir", default="results/saguaro")
    # Mode
    p.add_argument("--mode", choices=["dataset", "prompt"], default="dataset")
    p.add_argument("--prompt", default="")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.mode == "prompt":
        if not args.prompt:
            raise ValueError("--prompt is required with --mode prompt")
        _run_prompt(args)
        return

    if args.out_dir == "results/saguaro":
        args.out_dir = str(Path(args.out_dir) / _timestamp())

    _run_dataset(args)


if __name__ == "__main__":
    main()
