"""
Jacobi decoding with smart initialization for CUDA / HuggingFace models.

Core insight (temperature = 0):
  A Jacobi fixed-point block [t_1,...,t_{B-1}] satisfies
    t_i = argmax model(context, t_1, ..., t_{i-1})  for all i.
  Such a block is identical to what AR would produce and can be accepted
  wholesale.  Finding it costs ≤ max_iters forward passes, each processing
  B tokens in one batched call — far cheaper than B sequential AR steps.

Throughput gain:
  Even in the worst case (no convergence, max_iters iterations, acceptance=1)
  we still benefit because each forward pass runs B tokens in parallel, which
  is more efficient than B single-token AR passes on GPU.

Initialization matters for convergence speed:
  If the initial block guess is already the fixed point (n-gram/context hit),
  we converge in 1 pass instead of ≥2 — approximately doubling effective TPS.

Init strategies (tried in order until one fires):
  NGram    — rolling lookup table keyed on the last n output tokens.
  Context  — backward search: find last occurrence of the tail n-gram in
             the full generated context, copy what followed.
  Repeat   — copy the last token B-1 times (always-available fallback).
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from transformers import DynamicCache


# ─────────────────────────────────────────────────────────────────────────────
# Utilities (reused from model.py to keep this module self-contained on CUDA)
# ─────────────────────────────────────────────────────────────────────────────

def _sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab = logits.shape
    logits = logits.view(-1, vocab) / temperature
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).view(bsz, seq_len)


def _cuda_sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


# ─────────────────────────────────────────────────────────────────────────────
# Initialization strategies
# ─────────────────────────────────────────────────────────────────────────────

class NGramCache:
    """
    Rolling lookup table: last-n-tokens → likely next (block_size) tokens.

    Keyed on the last `n` tokens of the full generated sequence (including the
    just-committed last token).  Updated after every accepted block.  Multiple
    n-gram lengths are tried (n down to 2) so a cold cache still gets partial
    hits quickly.
    """

    def __init__(self, n: int = 4, max_per_key: int = 8, block_size: int = 15):
        self.n = n
        self.block_size = block_size
        self._table: dict = collections.defaultdict(
            lambda: collections.deque(maxlen=max_per_key)
        )
        self.hits = 0
        self.misses = 0

    def update(self, context: list[int], new_tokens: list[int]) -> None:
        """Record that `new_tokens` followed the last n tokens of `context`."""
        if not new_tokens:
            return
        # Store the direct continuation and one sliding-window shift to
        # populate the table faster during generation.
        for shift in range(min(len(new_tokens), 2)):
            key_ctx = context[shift:] + new_tokens[:shift]
            if len(key_ctx) < self.n:
                continue
            key = tuple(key_ctx[-self.n:])
            cont = new_tokens[shift: shift + self.block_size]
            if not cont:
                continue
            while len(cont) < self.block_size:
                cont = cont + [cont[-1]]
            self._table[key].appendleft(cont)

    def lookup(self, context: list[int]) -> Optional[list[int]]:
        """Return a block-sized continuation guess, or None on total miss."""
        # Try decreasing n so shorter histories can still hit.
        for n in range(self.n, 1, -1):
            key = tuple(context[-n:])
            if key in self._table and self._table[key]:
                self.hits += 1
                return list(self._table[key][0])
        self.misses += 1
        return None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0


class ContextSearchCache:
    """
    Backward n-gram search in the full generation history.

    Finds the most recent occurrence of the tail `n_match` tokens in the
    accumulated output and returns what followed as the block guess.  Works
    well for repetitive text (code, structured output) and costs O(L) memory
    per step — acceptable since L < max_new_tokens.
    """

    def __init__(self, n_match: int = 3, block_size: int = 15):
        self.n_match = n_match
        self.block_size = block_size
        self.hits = 0
        self.misses = 0

    def lookup(self, context: list[int]) -> Optional[list[int]]:
        tail = context[-self.n_match:]
        # Skip the trivially matching last occurrence (the tail itself).
        for i in range(len(context) - self.n_match - 1, self.n_match - 1, -1):
            if context[i: i + self.n_match] == tail:
                candidate = context[i + self.n_match: i + self.n_match + self.block_size]
                if candidate:
                    while len(candidate) < self.block_size:
                        candidate.append(candidate[-1])
                    self.hits += 1
                    return candidate
        self.misses += 1
        return None

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0


def _init_block(
    output: list[int],
    ngram: NGramCache,
    ctx: ContextSearchCache,
    guess_len: int,
    strategy: str,
) -> tuple[list[int], str, bool, bool]:
    """
    Return (block_guess, strategy_name, ngram_hit, ctx_hit).
    block_guess has exactly guess_len tokens.
    """
    last = output[-1] if output else 0

    ng_hit = ctx_hit = False

    if strategy in ("ngram", "best"):
        result = ngram.lookup(output)
        if result is not None:
            g = result[:guess_len]
            while len(g) < guess_len:
                g.append(g[-1] if g else last)
            return g, "ngram", True, False

    if strategy in ("context", "best"):
        result = ctx.lookup(output)
        if result is not None:
            g = result[:guess_len]
            while len(g) < guess_len:
                g.append(g[-1] if g else last)
            return g, "context", False, True

    return [last] * guess_len, "repeat", ng_hit, ctx_hit


# ─────────────────────────────────────────────────────────────────────────────
# Per-step statistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JacobiStepStats:
    step: int
    n_iters: int          # Jacobi iterations used (1 = converged on init)
    n_accepted: int       # tokens accepted this step
    block_size: int       # block size attempted (≤ global block_size)
    t_ms: float           # wall time for this block (ms)
    init_strategy: str    # "ngram" | "context" | "repeat"
    ngram_hit: bool
    ctx_hit: bool

    @property
    def utilization(self) -> float:
        return self.n_accepted / self.block_size if self.block_size > 0 else 0.0

    @property
    def tps(self) -> float:
        return (self.n_accepted / self.t_ms * 1000.0) if self.t_ms > 0 else 0.0


@dataclass
class JacobiRunStats:
    steps: list[JacobiStepStats] = field(default_factory=list)
    num_input_tokens: int = 0
    num_output_tokens: int = 0
    time_to_first_token_s: float = 0.0
    total_decode_time_s: float = 0.0
    ngram_hit_rate: float = 0.0
    ctx_hit_rate: float = 0.0

    @property
    def mean_iters(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.n_iters for s in self.steps) / len(self.steps)

    @property
    def mean_utilization(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.utilization for s in self.steps) / len(self.steps)

    @property
    def generation_tps(self) -> float:
        if self.total_decode_time_s <= 0:
            return 0.0
        return self.num_output_tokens / self.total_decode_time_s

    @property
    def time_per_output_token(self) -> float:
        if self.num_output_tokens <= 0:
            return float("inf")
        return self.total_decode_time_s / self.num_output_tokens

    @property
    def convergence_rate(self) -> float:
        """Fraction of blocks that fully converged (accepted == block_size)."""
        if not self.steps:
            return 0.0
        full = sum(1 for s in self.steps if s.n_accepted == s.block_size)
        return full / len(self.steps)

    @property
    def one_shot_rate(self) -> float:
        """Fraction of blocks that converged on the very first iteration."""
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.n_iters == 1) / len(self.steps)


# ─────────────────────────────────────────────────────────────────────────────
# Core generation function
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def jacobi_generate(
    target: nn.Module,
    input_ids: torch.LongTensor,        # (1, seq_len)
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: int = 16,
    max_iters: int = 10,
    init_strategy: str = "best",        # "repeat" | "ngram" | "context" | "best"
    ngram_n: int = 4,
    context_match_n: int = 3,
    return_stats: bool = False,
) -> SimpleNamespace:
    """
    Jacobi decoding: iteratively refine a block of B guesses until fixed-point,
    then commit all accepted tokens and move on.

    Returns a SimpleNamespace with the same fields as dflash_generate/sd_generate
    for drop-in benchmark compatibility, plus Jacobi-specific fields when
    return_stats=True.
    """
    num_input = input_ids.shape[1]
    device = input_ids.device

    ngram_cache = NGramCache(n=ngram_n, block_size=block_size - 1)
    ctx_cache = ContextSearchCache(n_match=context_match_n, block_size=block_size - 1)

    target_cache = DynamicCache()

    # ── Prefill ───────────────────────────────────────────────────────────────
    prefill_start = _cuda_sync_time()
    out = target(input_ids, past_key_values=target_cache, use_cache=True, logits_to_keep=1)
    first_tok = int(_sample(out.logits[:, -1:], temperature)[0, 0])
    ttft = _cuda_sync_time() - prefill_start

    # output tracks every token generated (NOT including the prompt).
    # output[-1] is always the last committed token ("last_tok").
    # The KV cache after prefill holds num_input positions.
    # last_tok is NOT yet in the cache; it will become the first element of the
    # next Jacobi block.
    output: list[int] = [first_tok]

    step_stats: list[JacobiStepStats] = []
    decode_start = _cuda_sync_time()

    while len(output) < max_new_tokens:
        last_tok = output[-1]
        B = min(block_size, max_new_tokens - len(output) + 1)
        guess_len = B - 1  # we guess B-1 tokens; last_tok is the anchor (position 0)

        # cache_pos_before: how many KVs are in the cache right now.
        # After prefill: num_input.
        # After each block: num_input + len(output) - 1, because last_tok is
        # always the un-cached "bonus" token from the previous round.
        cache_pos_before = num_input + len(output) - 1

        block_guess, init_name, ng_hit, cx_hit = _init_block(
            output, ngram_cache, ctx_cache, guess_len, init_strategy
        )

        block_t0 = _cuda_sync_time()
        final_preds: list[int] = []
        final_j: int = 0
        n_iters = 0

        if guess_len == 0:
            # Edge: last token of the budget — just run one AR step.
            tok_t = torch.tensor([[last_tok]], dtype=torch.long, device=device)
            out = target(tok_t, past_key_values=target_cache, use_cache=True)
            final_preds = [int(_sample(out.logits[:, -1:], temperature)[0, 0])]
            final_j = 0
            n_iters = 1
        else:
            for it in range(max_iters):
                n_iters = it + 1

                block_input = torch.tensor(
                    [[last_tok] + block_guess], dtype=torch.long, device=device
                )
                out = target(block_input, past_key_values=target_cache, use_cache=True)
                preds = _sample(out.logits, temperature)[0].tolist()  # length B

                # Find j = first mismatch between block_guess and preds[:guess_len].
                # preds[k] is the model's prediction for position cache_pos_before+k+1
                # given the block up to position k.  If block_guess[:k] == preds[:k],
                # then preds[k] is the exactly-correct AR token at that position.
                j = 0
                while j < guess_len and block_guess[j] == preds[j]:
                    j += 1
                # preds[:j+1] are all provably correct (j matches + 1 correction).

                final_j = j
                final_preds = preds

                converged = (j == guess_len)
                last_iter = (it == max_iters - 1)

                if converged or last_iter:
                    # Commit: trim the excess KVs we don't need.
                    trim = B - j - 1
                    if trim > 0:
                        target_cache.crop(cache_pos_before + j + 1)
                    break
                else:
                    # Retry: roll the cache back and try with a better guess.
                    target_cache.crop(cache_pos_before)
                    block_guess = list(preds[:guess_len])

        block_t_ms = (_cuda_sync_time() - block_t0) * 1000.0

        new_tokens = final_preds[: final_j + 1]

        if return_stats:
            step_stats.append(JacobiStepStats(
                step=len(step_stats),
                n_iters=n_iters,
                n_accepted=len(new_tokens),
                block_size=B,
                t_ms=block_t_ms,
                init_strategy=init_name,
                ngram_hit=ng_hit,
                ctx_hit=cx_hit,
            ))

        # Update n-gram cache before extending output so the key includes last_tok.
        ngram_cache.update(output, new_tokens)
        output.extend(new_tokens)

        if stop_token_ids:
            tail = output[-len(new_tokens):]
            if any(t in tail for t in stop_token_ids):
                break

    # ── Assemble output ───────────────────────────────────────────────────────
    out_len = min(len(output), max_new_tokens)
    output_ids = torch.cat([
        input_ids,
        torch.tensor([output[:out_len]], dtype=torch.long, device=device),
    ], dim=1)

    # Truncate at first stop token.
    if stop_token_ids:
        stop_t = torch.tensor(stop_token_ids, device=device)
        stop_pos = torch.isin(output_ids[0, num_input:], stop_t).nonzero(as_tuple=True)[0]
        if stop_pos.numel() > 0:
            output_ids = output_ids[:, : num_input + stop_pos[0] + 1]
            out_len = output_ids.shape[1] - num_input

    total_decode = _cuda_sync_time() - decode_start

    if not return_stats:
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input,
            num_output_tokens=out_len,
            time_to_first_token=ttft,
            time_per_output_token=total_decode / max(out_len, 1),
            acceptance_lengths=[s.n_accepted for s in step_stats] if step_stats else [1] * out_len,
        )

    run_stats = JacobiRunStats(
        steps=step_stats,
        num_input_tokens=num_input,
        num_output_tokens=out_len,
        time_to_first_token_s=ttft,
        total_decode_time_s=total_decode,
        ngram_hit_rate=ngram_cache.hit_rate,
        ctx_hit_rate=ctx_cache.hit_rate,
    )

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input,
        num_output_tokens=out_len,
        time_to_first_token=ttft,
        time_per_output_token=total_decode / max(out_len, 1),
        acceptance_lengths=[s.n_accepted for s in step_stats],
        run_stats=run_stats,
    )


@torch.inference_mode()
def ar_generate(
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
) -> SimpleNamespace:
    """Pure autoregressive baseline using the same model + cache pattern."""
    num_input = input_ids.shape[1]
    device = input_ids.device
    cache = DynamicCache()

    t0 = _cuda_sync_time()
    out = target(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
    first_tok = int(_sample(out.logits[:, -1:], temperature)[0, 0])
    ttft = _cuda_sync_time() - t0

    output = [first_tok]
    decode_start = _cuda_sync_time()

    while len(output) < max_new_tokens:
        tok_t = torch.tensor([[output[-1]]], dtype=torch.long, device=device)
        out = target(tok_t, past_key_values=cache, use_cache=True, logits_to_keep=1)
        next_tok = int(_sample(out.logits[:, -1:], temperature)[0, 0])
        output.append(next_tok)
        if stop_token_ids and next_tok in stop_token_ids:
            break

    out_len = min(len(output), max_new_tokens)
    total = _cuda_sync_time() - decode_start

    output_ids = torch.cat([
        input_ids,
        torch.tensor([output[:out_len]], dtype=torch.long, device=device),
    ], dim=1)

    if stop_token_ids:
        stop_t = torch.tensor(stop_token_ids, device=device)
        stop_pos = torch.isin(output_ids[0, num_input:], stop_t).nonzero(as_tuple=True)[0]
        if stop_pos.numel() > 0:
            output_ids = output_ids[:, : num_input + stop_pos[0] + 1]
            out_len = output_ids.shape[1] - num_input

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input,
        num_output_tokens=out_len,
        time_to_first_token=ttft,
        time_per_output_token=total / max(out_len, 1),
        acceptance_lengths=[1] * out_len,
    )
