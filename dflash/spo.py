"""
Speculative Policy Optimization (SPO): RL-trained draft model for speculative decoding.

Instead of cross-entropy distillation, the draft is trained with REINFORCE where
reward = acceptance_length.  This teaches the policy to concentrate its limited
capacity on early block positions (where errors are catastrophic) rather than
spreading it uniformly.

Architecture:
  Target embed_tokens (frozen, shared)
       -> proj_in (embed_dim -> hidden_dim)
       -> 2-layer causal transformer (hidden_dim, cheap)
       -> proj_out (hidden_dim -> embed_dim)
       -> target lm_head (frozen, shared)
       -> B logits

No hidden-state dependency on the target, so the draft can run in parallel or
be pipelined — unlike DFlash/EAGLE.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SPOConfig:
    embed_dim: int = 4096
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    context_window: int = 64
    block_size: int = 16
    vocab_size: int = 151936
    dropout: float = 0.0

    @classmethod
    def from_target(cls, target: nn.Module, **overrides) -> "SPOConfig":
        cfg = target.config
        return cls(
            embed_dim=cfg.hidden_size,
            vocab_size=cfg.vocab_size,
            **overrides,
        )


@dataclass
class SPOTrainConfig:
    lr: float = 3e-4
    train_steps: int = 2000
    temperature: float = 0.8
    baseline_ema: float = 0.99
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_every: int = 200
    checkpoint_dir: str = "checkpoints/spo"
    log_every: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cuda_sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _sample_greedy(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _acceptance_length(draft: torch.Tensor, truth: torch.Tensor) -> int:
    """Number of leading matches between draft and truth (1-D tensors)."""
    matches = (draft == truth).long()
    return int(matches.cumprod(0).sum().item())


# ─────────────────────────────────────────────────────────────────────────────
# SPO Draft Model
# ─────────────────────────────────────────────────────────────────────────────

class SPODraft(nn.Module):
    """Tiny causal transformer that proposes B tokens for speculative verification."""

    def __init__(self, config: SPOConfig):
        super().__init__()
        self.config = config

        self.proj_in = nn.Linear(config.embed_dim, config.hidden_dim, bias=False)
        max_positions = config.context_window + config.block_size
        self.pos_emb = nn.Embedding(max_positions, config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.out_norm = nn.LayerNorm(config.hidden_dim)
        self.proj_out = nn.Linear(config.hidden_dim, config.embed_dim, bias=False)

        # Populated by bind()
        self._embed_fn: Optional[nn.Embedding] = None
        self._lm_head_fn: Optional[nn.Linear] = None
        self._device: torch.device = torch.device("cpu")

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def bind(self, target: nn.Module) -> "SPODraft":
        """Share target's embedding table and lm_head (frozen)."""
        self._embed_fn = target.model.embed_tokens
        self._lm_head_fn = target.lm_head
        self._device = next(target.parameters()).device
        self.to(device=self._device, dtype=torch.bfloat16)
        return self

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            token_ids: (1, K+B) — last K context tokens + B block guess tokens.
        Returns:
            logits: (1, B, vocab_size) — predictions for the B block positions.
        """
        B = self.config.block_size
        seq_len = token_ids.shape[1]

        with torch.no_grad():
            x = self._embed_fn(token_ids)  # (1, K+B, embed_dim)

        x = self.proj_in(x)  # (1, K+B, hidden_dim)

        positions = torch.arange(seq_len, device=x.device)
        if seq_len > self.pos_emb.num_embeddings:
            positions = positions.clamp(max=self.pos_emb.num_embeddings - 1)
        x = x + self.pos_emb(positions).unsqueeze(0)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.out_norm(x)

        block_hidden = x[:, -B:, :]  # (1, B, hidden_dim)
        block_embed = self.proj_out(block_hidden)  # (1, B, embed_dim)

        with torch.no_grad():
            logits = self._lm_head_fn(block_embed.to(self._lm_head_fn.weight.dtype))

        return logits.float()  # (1, B, vocab_size)  — float32 for stable log_softmax

    @torch.inference_mode()
    def propose(
        self,
        output_tokens: list[int],
        n_tokens: int,
        temperature: float = 0.0,
    ) -> list[int]:
        """Generate n_tokens draft tokens from the current context."""
        K = min(len(output_tokens), self.config.context_window)
        ctx = output_tokens[-K:]
        last_tok = ctx[-1] if ctx else 0

        guess = [last_tok] * self.config.block_size
        input_ids = torch.tensor([ctx + guess], dtype=torch.long, device=self._device)

        logits = self.forward(input_ids)[:, :n_tokens]  # (1, n, V)

        if temperature < 1e-5:
            return logits.argmax(dim=-1)[0].tolist()
        probs = torch.softmax(logits / temperature, dim=-1)
        tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1)
        return tokens.view(n_tokens).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# SPO Trainer  (REINFORCE with EMA baseline)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SPOTrainStep:
    step: int
    reward: float
    loss: float
    entropy: float
    baseline: float
    grad_norm: float
    lr: float
    t_ms: float


class SPOTrainer:
    """REINFORCE trainer using pre-generated target completions as ground truth."""

    def __init__(
        self,
        spo_draft: SPODraft,
        train_config: SPOTrainConfig,
    ):
        self.draft = spo_draft
        self.config = train_config
        self.optimizer = torch.optim.AdamW(
            spo_draft.parameters(), lr=train_config.lr, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=train_config.train_steps, eta_min=train_config.lr * 0.1
        )
        self.baseline = 0.0
        self.global_step = 0

    def train_step(self, full_sequence: torch.LongTensor) -> SPOTrainStep:
        """
        One REINFORCE step on a pre-generated sequence.

        Args:
            full_sequence: (1, L) — concatenation of prompt + target completion.
        """
        t0 = time.perf_counter()
        cfg = self.draft.config
        K = cfg.context_window
        B = cfg.block_size
        L = full_sequence.shape[1]

        if L < K + B + 1:
            K = max(1, L - B - 1)

        # Random position: the block starts at p, context is p-K..p-1
        p = random.randint(K, L - B)
        context_ids = full_sequence[:, p - K: p]       # (1, K)
        ground_truth = full_sequence[:, p: p + B]       # (1, B)

        # ── Forward pass 1: logits conditioned on repeat-last-token guess ─────
        last_tok = full_sequence[0, p - 1].item()
        guess = torch.full((1, B), last_tok, dtype=torch.long, device=full_sequence.device)
        input_ids = torch.cat([context_ids, guess], dim=1)  # (1, K+B)

        self.draft.train()
        logits = self.draft.forward(input_ids)  # (1, B, V)

        # Sample draft tokens
        probs = torch.softmax(logits / self.config.temperature, dim=-1)
        draft_tokens = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(1, B)

        # ── Forward pass 2: accurate log-probs conditioned on sampled tokens ──
        input_for_lp = torch.cat([context_ids, draft_tokens], dim=1)
        logits2 = self.draft.forward(input_for_lp)  # (1, B, V)
        log_probs = F.log_softmax(logits2 / self.config.temperature, dim=-1)
        action_lp = log_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)  # (1, B)

        # ── Reward: acceptance length vs ground truth ─────────────────────────
        reward = float(_acceptance_length(draft_tokens[0], ground_truth[0]))

        # ── REINFORCE with EMA baseline ───────────────────────────────────────
        self.baseline = (
            self.config.baseline_ema * self.baseline
            + (1 - self.config.baseline_ema) * reward
        )
        advantage = reward - self.baseline

        # Per-position weighting: positions beyond acceptance contribute 0
        # marginal reward, so mask them out for lower variance.
        acc_int = int(reward)
        weight_mask = torch.zeros(B, device=full_sequence.device)
        weight_mask[: min(acc_int + 1, B)] = 1.0
        weighted_lp = (action_lp[0] * weight_mask).sum()

        policy_loss = -advantage * weighted_lp

        # Entropy bonus for exploration
        entropy = -(probs * log_probs.exp().clamp(min=1e-8).log()).sum(-1).mean()
        loss = policy_loss - self.config.entropy_coef * entropy

        # ── Backprop ──────────────────────────────────────────────────────────
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.draft.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.global_step += 1

        t_ms = (time.perf_counter() - t0) * 1000.0

        return SPOTrainStep(
            step=self.global_step,
            reward=reward,
            loss=loss.item(),
            entropy=entropy.item(),
            baseline=self.baseline,
            grad_norm=float(grad_norm),
            lr=self.scheduler.get_last_lr()[0],
            t_ms=t_ms,
        )

    def save_checkpoint(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "draft_state_dict": self.draft.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.draft.config,
            "train_config": self.config,
            "baseline": self.baseline,
            "global_step": self.global_step,
        }, path / "spo_checkpoint.pt")

    def load_checkpoint(self, path: Path | str) -> None:
        ckpt = torch.load(Path(path) / "spo_checkpoint.pt", map_location="cpu", weights_only=False)
        self.draft.load_state_dict(ckpt["draft_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.baseline = ckpt["baseline"]
        self.global_step = ckpt["global_step"]


def load_spo_draft(checkpoint_path: str | Path, target: nn.Module) -> SPODraft:
    ckpt = torch.load(Path(checkpoint_path) / "spo_checkpoint.pt", map_location="cpu", weights_only=False)
    draft = SPODraft(ckpt["config"])
    draft.load_state_dict(ckpt["draft_state_dict"])
    draft.bind(target)
    draft.eval()
    return draft


# ─────────────────────────────────────────────────────────────────────────────
# Pre-generate target completions (training data)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def pregenerate_targets(
    target: nn.Module,
    tokenizer,
    dataset: list[dict],
    max_new_tokens: int,
    output_path: Path,
    enable_thinking: bool = False,
) -> Path:
    """Run AR on each prompt, save {input_ids, completion_ids} as JSONL."""
    from .benchmark import _apply_chat_template
    from tqdm import tqdm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path) as f:
            existing = sum(1 for _ in f)
        if existing >= len(dataset):
            return output_path

    device = next(target.parameters()).device
    eos_id = tokenizer.eos_token_id

    with open(output_path, "w") as f:
        for instance in tqdm(dataset, desc="Pre-generating"):
            messages = [{"role": "user", "content": instance["turns"][0]}]
            input_text = _apply_chat_template(tokenizer, messages, enable_thinking)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

            cache = DynamicCache()
            out = target(input_ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
            tok = int(out.logits[:, -1:].argmax(dim=-1)[0, 0])
            completion = [tok]

            for _ in range(max_new_tokens - 1):
                tok_t = torch.tensor([[tok]], dtype=torch.long, device=device)
                out = target(tok_t, past_key_values=cache, use_cache=True, logits_to_keep=1)
                tok = int(out.logits[:, -1:].argmax(dim=-1)[0, 0])
                completion.append(tok)
                if tok == eos_id:
                    break

            f.write(json.dumps({
                "input_ids": input_ids[0].tolist(),
                "completion_ids": completion,
            }) + "\n")

    return output_path


def load_pregenerated(path: Path | str) -> list[torch.LongTensor]:
    sequences = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            full = d["input_ids"] + d["completion_ids"]
            sequences.append(torch.tensor(full, dtype=torch.long))
    return sequences


# ─────────────────────────────────────────────────────────────────────────────
# Per-step stats for inference
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SPOStepStats:
    step: int
    acceptance_length: int
    block_size: int
    t_draft_ms: float
    t_verify_ms: float
    t_total_ms: float

    @property
    def utilization(self) -> float:
        return self.acceptance_length / self.block_size if self.block_size else 0.0

    @property
    def tps(self) -> float:
        return self.acceptance_length / (self.t_total_ms / 1000.0) if self.t_total_ms > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Generation (speculative decoding with SPO draft)
# ─────────────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def spo_generate(
    spo_draft: SPODraft,
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    return_stats: bool = False,
) -> SimpleNamespace:
    """Speculative decoding using the SPO-trained draft."""
    block_size = block_size or spo_draft.config.block_size
    num_input = input_ids.shape[1]
    device = input_ids.device
    target_cache = DynamicCache()

    # ── Prefill ───────────────────────────────────────────────────────────────
    t_prefill = _cuda_sync_time()
    t_out = target(input_ids, past_key_values=target_cache, use_cache=True, logits_to_keep=1)
    first_tok = int(t_out.logits[:, -1:].argmax(dim=-1)[0, 0]) if temperature < 1e-5 else \
        int(torch.multinomial(torch.softmax(t_out.logits[:, -1] / temperature, -1), 1)[0, 0])
    ttft = _cuda_sync_time() - t_prefill

    output = [first_tok]
    target_cache_pos = num_input
    step_stats: list[SPOStepStats] = []

    decode_start = _cuda_sync_time()

    while len(output) < max_new_tokens:
        curr_tok = output[-1]
        n = min(block_size, max_new_tokens - len(output))
        old_cache_pos = target_cache_pos

        # ── Draft ─────────────────────────────────────────────────────────────
        t0 = _cuda_sync_time()
        draft_tokens = spo_draft.propose(output, n, temperature=0.0)
        t_draft = (_cuda_sync_time() - t0) * 1000.0

        # ── Verify ────────────────────────────────────────────────────────────
        t0 = _cuda_sync_time()
        curr_t = torch.tensor([[curr_tok]], dtype=torch.long, device=device)
        draft_t = torch.tensor([draft_tokens[:n]], dtype=torch.long, device=device)
        verify_input = torch.cat([curr_t, draft_t], dim=1)

        t_out = target(verify_input, past_key_values=target_cache, use_cache=True)
        target_cache_pos += n + 1

        if temperature < 1e-5:
            t_preds = t_out.logits.argmax(dim=-1)[0].tolist()
        else:
            probs = torch.softmax(t_out.logits / temperature, dim=-1)
            t_preds = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(-1).tolist()

        # Acceptance
        accepted = 0
        for i in range(n):
            if draft_tokens[i] == t_preds[i]:
                accepted += 1
            else:
                break

        bonus = t_preds[accepted]
        new_toks = (draft_tokens[:accepted] + [bonus])[: max_new_tokens - len(output)]
        t_verify = (_cuda_sync_time() - t0) * 1000.0

        output.extend(new_toks)

        # Cache rollback
        target_cache.crop(old_cache_pos + accepted + 1)
        target_cache_pos = old_cache_pos + accepted + 1

        if return_stats:
            step_stats.append(SPOStepStats(
                step=len(step_stats),
                acceptance_length=len(new_toks),
                block_size=n + 1,
                t_draft_ms=t_draft,
                t_verify_ms=t_verify,
                t_total_ms=t_draft + t_verify,
            ))

        if stop_token_ids and any(t in new_toks for t in stop_token_ids):
            break

    # ── Assemble ──────────────────────────────────────────────────────────────
    out_len = min(len(output), max_new_tokens)
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

    total_decode = _cuda_sync_time() - decode_start

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input,
        num_output_tokens=out_len,
        time_to_first_token=ttft,
        time_per_output_token=total_decode / max(out_len, 1),
        acceptance_lengths=[s.acceptance_length for s in step_stats] if step_stats else [1] * out_len,
        step_stats=step_stats,
    )
