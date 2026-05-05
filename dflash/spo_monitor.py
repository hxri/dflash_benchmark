"""
Real-time Rich TUI for SPO training and inference.

Training panel shows: reward sparkline, loss curve, entropy, LR, baseline.
Inference panel shows: TPS sparkline, acceptance histogram, speedup vs AR.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from .spo import SPOTrainStep, SPOStepStats


_SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"
_SPARK_WIDTH = 20
_MAX_HISTORY = 30


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sparkline(values: list[float], width: int = _SPARK_WIDTH) -> str:
    if not values:
        return " " * width
    recent = values[-width:]
    lo, hi = min(recent), max(recent)
    span = hi - lo or 1.0
    chars = [_SPARKLINE_CHARS[int((v - lo) / span * (len(_SPARKLINE_CHARS) - 1))] for v in recent]
    return "".join(chars).ljust(width)


def _bar(frac: float, width: int = 12) -> str:
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


def _color_speedup(s: float) -> str:
    v = f"{s:.2f}×"
    if s >= 1.5:
        return f"[bold green]{v}[/bold green]"
    if s >= 1.0:
        return f"[green]{v}[/green]"
    return f"[red]{v}[/red]"


def _color_util(u: float) -> str:
    pct = f"{u * 100:.1f}%"
    if u >= 0.75:
        return f"[green]{pct}[/green]"
    if u >= 0.50:
        return f"[yellow]{pct}[/yellow]"
    return f"[red]{pct}[/red]"


# ─────────────────────────────────────────────────────────────────────────────
# Training monitor state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SPOTrainMonitorState:
    total_steps: int
    reward_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    entropy_history: list[float] = field(default_factory=list)
    lr_history: list[float] = field(default_factory=list)
    baseline_history: list[float] = field(default_factory=list)
    grad_norm_history: list[float] = field(default_factory=list)
    steps: list[SPOTrainStep] = field(default_factory=list)

    def record(self, s: SPOTrainStep) -> None:
        self.steps.append(s)
        self.reward_history.append(s.reward)
        self.loss_history.append(s.loss)
        self.entropy_history.append(s.entropy)
        self.lr_history.append(s.lr)
        self.baseline_history.append(s.baseline)
        self.grad_norm_history.append(s.grad_norm)

    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def mean_reward(self) -> float:
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history[-50:]))

    @property
    def max_reward(self) -> float:
        return max(self.reward_history) if self.reward_history else 0.0

    @property
    def mean_loss(self) -> float:
        if not self.loss_history:
            return 0.0
        return float(np.mean(self.loss_history[-50:]))

    @property
    def mean_entropy(self) -> float:
        if not self.entropy_history:
            return 0.0
        return float(np.mean(self.entropy_history[-50:]))


# ─────────────────────────────────────────────────────────────────────────────
# Inference monitor state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SPOInferMonitorState:
    block_size: int
    model_name: str
    ar_tps: float = 0.0

    steps: list[SPOStepStats] = field(default_factory=list)
    tps_history: list[float] = field(default_factory=list)
    util_history: list[float] = field(default_factory=list)
    accept_counts: collections.Counter = field(default_factory=collections.Counter)
    total_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record(self, s: SPOStepStats) -> None:
        self.steps.append(s)
        self.tps_history.append(s.tps)
        self.util_history.append(s.utilization)
        self.accept_counts[s.acceptance_length] += 1
        self.total_tokens += s.acceptance_length

    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def mean_tps(self) -> float:
        if not self.tps_history:
            return 0.0
        return float(np.mean(self.tps_history[-50:]))

    @property
    def mean_util(self) -> float:
        if not self.util_history:
            return 0.0
        return float(np.mean(self.util_history))

    @property
    def mean_acceptance(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.acceptance_length for s in self.steps]))

    @property
    def speedup(self) -> float:
        if self.ar_tps <= 0 or self.mean_tps <= 0:
            return 0.0
        return self.mean_tps / self.ar_tps

    @property
    def cumulative_tps(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return self.total_tokens / elapsed if elapsed > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_train_panel(state: SPOTrainMonitorState) -> Panel:
    lines: list[str] = []

    pct = state.n / state.total_steps * 100 if state.total_steps else 0
    lines.append(f"[bold cyan]SPO Training[/bold cyan]  step {state.n}/{state.total_steps}  ({pct:.0f}%)")
    lines.append("")

    lines.append("[bold]Reward[/bold]  (= acceptance length)")
    spark = _sparkline(state.reward_history)
    lines.append(f"  Mean (50) {state.mean_reward:5.2f}  max {state.max_reward:.0f}  {spark}")
    if state.baseline_history:
        lines.append(f"  Baseline  {state.baseline_history[-1]:5.2f}")
    lines.append("")

    lines.append("[bold]Loss[/bold]")
    spark_l = _sparkline(state.loss_history)
    lines.append(f"  Mean (50) {state.mean_loss:8.4f}  {spark_l}")
    lines.append("")

    lines.append("[bold]Entropy[/bold]")
    spark_e = _sparkline(state.entropy_history)
    lines.append(f"  Mean (50) {state.mean_entropy:5.3f}  {spark_e}")
    lines.append("")

    lines.append("[bold]Learning Rate[/bold]")
    if state.lr_history:
        lines.append(f"  Current   {state.lr_history[-1]:.2e}")
    lines.append("")

    lines.append("[bold]Grad norm[/bold]")
    if state.grad_norm_history:
        spark_g = _sparkline(state.grad_norm_history)
        lines.append(f"  Latest    {state.grad_norm_history[-1]:5.3f}  {spark_g}")

    body = "\n".join(lines)
    return Panel(body, title="[bold]SPO Train Monitor[/bold]", border_style="magenta", expand=True)


def _render_train_table(state: SPOTrainMonitorState) -> Table:
    t = Table(show_header=True, header_style="bold dim", show_lines=False,
              box=None, padding=(0, 1), expand=True)
    t.add_column("Step", style="dim", width=6, justify="right")
    t.add_column("Reward", width=7, justify="right")
    t.add_column("Loss", width=10, justify="right")
    t.add_column("Entropy", width=8, justify="right")
    t.add_column("∇norm", width=7, justify="right")
    t.add_column("LR", width=10, justify="right")
    t.add_column("ms", width=7, justify="right")

    recent = state.steps[-_MAX_HISTORY:]
    for s in recent:
        rew_color = "green" if s.reward >= 4 else ("yellow" if s.reward >= 2 else "red")
        t.add_row(
            str(s.step),
            f"[{rew_color}]{s.reward:.1f}[/{rew_color}]",
            f"{s.loss:.4f}",
            f"{s.entropy:.3f}",
            f"{s.grad_norm:.3f}",
            f"{s.lr:.2e}",
            f"{s.t_ms:.0f}",
        )
    return t


def _render_train_layout(state: SPOTrainMonitorState) -> Layout:
    root = Layout()
    root.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    root["left"].update(_render_train_panel(state))
    root["right"].update(Panel(
        _render_train_table(state),
        title="[bold]Step trace[/bold]",
        border_style="dim",
        expand=True,
    ))
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Inference renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_infer_panel(state: SPOInferMonitorState) -> Panel:
    lines: list[str] = []

    lines.append(f"[bold cyan]{state.model_name}[/bold cyan]  block={state.block_size}")
    lines.append("")

    lines.append("[bold]Throughput[/bold]")
    spark = _sparkline(state.tps_history)
    lines.append(f"  SPO      {state.mean_tps:7.1f} tok/s  {spark}")
    if state.ar_tps > 0:
        lines.append(f"  AR base  {state.ar_tps:7.1f} tok/s")
        lines.append(f"  Speedup  {_color_speedup(state.speedup)}")
    lines.append(f"  Cumul.   {state.cumulative_tps:7.1f} tok/s  ({state.total_tokens} tok)")
    lines.append("")

    lines.append("[bold]Acceptance[/bold]")
    spark_u = _sparkline(state.util_history)
    bar = _bar(state.mean_util)
    lines.append(f"  Mean util  {_color_util(state.mean_util)}  {bar}  {spark_u}")
    lines.append(f"  Mean accept  {state.mean_acceptance:.2f} / {state.block_size}")
    lines.append("")

    lines.append("[bold]Acceptance distribution[/bold]")
    for length in range(1, state.block_size + 2):
        cnt = state.accept_counts.get(length, 0)
        if cnt == 0 and length > max(state.accept_counts.keys(), default=1):
            break
        frac = cnt / state.n if state.n else 0.0
        b = _bar(frac, 10)
        lines.append(f"  len={length:2d}  {b}  {frac * 100:5.1f}%  ({cnt})")

    body = "\n".join(lines)
    return Panel(body, title="[bold]SPO Inference Monitor[/bold]", border_style="blue", expand=True)


def _render_infer_table(state: SPOInferMonitorState) -> Table:
    t = Table(show_header=True, header_style="bold dim", show_lines=False,
              box=None, padding=(0, 1), expand=True)
    t.add_column("Step", style="dim", width=5, justify="right")
    t.add_column("Accept", width=8, justify="center")
    t.add_column("Util", width=8, justify="right")
    t.add_column("Draft ms", width=9, justify="right")
    t.add_column("Verify ms", width=10, justify="right")
    t.add_column("TPS", width=8, justify="right")

    recent = state.steps[-_MAX_HISTORY:]
    for s in recent:
        t.add_row(
            str(s.step),
            f"[white]{s.acceptance_length:2d}[/white]/[dim]{s.block_size}[/dim]",
            _color_util(s.utilization),
            f"{s.t_draft_ms:5.1f}",
            f"{s.t_verify_ms:5.1f}",
            f"{s.tps:6.1f}",
        )
    return t


def _render_infer_layout(state: SPOInferMonitorState) -> Layout:
    root = Layout()
    root.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    root["left"].update(_render_infer_panel(state))
    root["right"].update(Panel(
        _render_infer_table(state),
        title="[bold]Step trace[/bold]",
        border_style="dim",
        expand=True,
    ))
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Public monitor classes
# ─────────────────────────────────────────────────────────────────────────────

class SPOTrainMonitor:
    """Context manager that drives a Rich Live display during SPO training."""

    def __init__(self, state: SPOTrainMonitorState, refresh_rate: int = 4,
                 console: Optional[Console] = None):
        self.state = state
        self._console = console or Console()
        self._live = Live(
            _render_train_layout(state),
            console=self._console,
            refresh_per_second=refresh_rate,
            vertical_overflow="visible",
        )

    def __enter__(self) -> "SPOTrainMonitor":
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._live.__exit__(*args)

    def update(self, step: SPOTrainStep) -> None:
        self.state.record(step)
        self._live.update(_render_train_layout(self.state))


class SPOInferMonitor:
    """Context manager that drives a Rich Live display during SPO inference."""

    def __init__(self, state: SPOInferMonitorState, refresh_rate: int = 4,
                 console: Optional[Console] = None):
        self.state = state
        self._console = console or Console()
        self._live = Live(
            _render_infer_layout(state),
            console=self._console,
            refresh_per_second=refresh_rate,
            vertical_overflow="visible",
        )

    def __enter__(self) -> "SPOInferMonitor":
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._live.__exit__(*args)

    def update(self, step: SPOStepStats) -> None:
        self.state.record(step)
        self._live.update(_render_infer_layout(self.state))


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────

def make_train_monitor(
    total_steps: int,
    refresh_rate: int = 4,
) -> tuple[SPOTrainMonitorState, SPOTrainMonitor]:
    state = SPOTrainMonitorState(total_steps=total_steps)
    return state, SPOTrainMonitor(state, refresh_rate=refresh_rate)


def make_infer_monitor(
    block_size: int,
    model_name: str,
    ar_tps: float = 0.0,
    refresh_rate: int = 4,
) -> tuple[SPOInferMonitorState, SPOInferMonitor]:
    state = SPOInferMonitorState(
        block_size=block_size,
        model_name=model_name,
        ar_tps=ar_tps,
    )
    return state, SPOInferMonitor(state, refresh_rate=refresh_rate)
