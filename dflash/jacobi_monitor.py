"""
Real-time Rich TUI for Jacobi decoding.

Shows per-step metrics in a scrolling table and live aggregate panels:
  - Tokens/sec (current and rolling average)
  - Iterations per block distribution
  - Block utilization (accepted / block_size)
  - Init strategy hit rates (n-gram, context, repeat)
  - Convergence and one-shot rates
  - Speedup vs AR baseline
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
from rich.text import Text

from .jacobi import JacobiStepStats


_SPARKLINE_CHARS = " ▁▂▃▄▅▆▇█"
_MAX_HISTORY = 30   # step rows to display
_SPARK_WIDTH = 20   # width of sparkline bars


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


def _color_util(u: float) -> str:
    pct = f"{u * 100:.1f}%"
    if u >= 0.75:
        return f"[green]{pct}[/green]"
    if u >= 0.50:
        return f"[yellow]{pct}[/yellow]"
    return f"[red]{pct}[/red]"


def _color_iters(n: int) -> str:
    if n == 1:
        return f"[green]{n}[/green]"
    if n <= 3:
        return f"[yellow]{n}[/yellow]"
    return f"[red]{n}[/red]"


def _color_speedup(s: float) -> str:
    v = f"{s:.2f}×"
    if s >= 1.5:
        return f"[bold green]{v}[/bold green]"
    if s >= 1.0:
        return f"[green]{v}[/green]"
    return f"[red]{v}[/red]"


def _color_init(name: str) -> str:
    colors = {"ngram": "cyan", "context": "blue", "repeat": "dim"}
    c = colors.get(name, "white")
    return f"[{c}]{name[:7]}[/{c}]"


# ─────────────────────────────────────────────────────────────────────────────
# Running state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JacobiMonitorState:
    block_size: int
    max_iters: int
    init_strategy: str
    model_name: str
    ar_tps: float = 0.0

    steps: list[JacobiStepStats] = field(default_factory=list)
    tps_history: list[float] = field(default_factory=list)   # per-step TPS
    util_history: list[float] = field(default_factory=list)  # per-step utilization
    iter_counts: collections.Counter = field(default_factory=collections.Counter)

    total_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record(self, s: JacobiStepStats) -> None:
        self.steps.append(s)
        self.tps_history.append(s.tps)
        self.util_history.append(s.utilization)
        self.iter_counts[s.n_iters] += 1
        self.total_tokens += s.n_accepted

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
    def mean_iters(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.n_iters for s in self.steps]))

    @property
    def one_shot_rate(self) -> float:
        if not self.steps:
            return 0.0
        return self.iter_counts[1] / self.n

    @property
    def ngram_hit_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.ngram_hit) / self.n

    @property
    def ctx_hit_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.ctx_hit) / self.n

    @property
    def repeat_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if not s.ngram_hit and not s.ctx_hit) / self.n

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
# Renderers
# ─────────────────────────────────────────────────────────────────────────────

def _render_step_table(state: JacobiMonitorState) -> Table:
    t = Table(
        title=None,
        show_header=True,
        header_style="bold dim",
        show_lines=False,
        box=None,
        padding=(0, 1),
        expand=True,
    )
    t.add_column("Step", style="dim", width=5, justify="right")
    t.add_column("Init",    width=8)
    t.add_column("Iters",   width=6, justify="center")
    t.add_column("Accepted",width=10, justify="center")
    t.add_column("Util",    width=8, justify="right")
    t.add_column("TPS",     width=8, justify="right")
    t.add_column("ms",      width=7, justify="right")

    recent = state.steps[-_MAX_HISTORY:]
    for s in recent:
        t.add_row(
            str(s.step),
            _color_init(s.init_strategy),
            _color_iters(s.n_iters),
            f"[white]{s.n_accepted:2d}[/white]/[dim]{s.block_size}[/dim]",
            _color_util(s.utilization),
            f"{s.tps:6.1f}",
            f"{s.t_ms:5.1f}",
        )
    return t


def _render_agg_panel(state: JacobiMonitorState) -> Panel:
    lines: list[str] = []

    # Header
    lines.append(f"[bold cyan]{state.model_name}[/bold cyan]  "
                 f"block={state.block_size}  max_iters={state.max_iters}  "
                 f"init=[cyan]{state.init_strategy}[/cyan]")
    lines.append("")

    # TPS
    lines.append(f"[bold]Throughput[/bold]")
    spark = _sparkline(state.tps_history)
    lines.append(f"  Jacobi   {state.mean_tps:7.1f} tok/s  {spark}")
    if state.ar_tps > 0:
        lines.append(f"  AR base  {state.ar_tps:7.1f} tok/s")
        lines.append(f"  Speedup  {_color_speedup(state.speedup)}")
    lines.append(f"  Cumul.   {state.cumulative_tps:7.1f} tok/s  ({state.total_tokens} tok)")
    lines.append("")

    # Block utilization
    lines.append(f"[bold]Block utilization[/bold]")
    spark_util = _sparkline(state.util_history)
    bar = _bar(state.mean_util)
    lines.append(f"  Mean  {_color_util(state.mean_util)}  {bar}  {spark_util}")
    lines.append(f"  1-shot rate  {state.one_shot_rate * 100:5.1f}%  (converged in 1 iter)")
    lines.append("")

    # Iterations distribution
    lines.append(f"[bold]Iterations per block[/bold]  (mean {state.mean_iters:.2f})")
    for iters in range(1, state.max_iters + 1):
        cnt = state.iter_counts.get(iters, 0)
        if cnt == 0 and iters > max((state.iter_counts or {1: 1}), default=1):
            break
        frac = cnt / state.n if state.n else 0.0
        b = _bar(frac, 10)
        lines.append(f"  iter={iters}  {b}  {frac * 100:5.1f}%  ({cnt})")
    lines.append("")

    # Init strategy breakdown
    lines.append(f"[bold]Init strategy[/bold]")
    lines.append(f"  [cyan]ngram  [/cyan] {state.ngram_hit_rate * 100:5.1f}%")
    lines.append(f"  [blue]context[/blue] {state.ctx_hit_rate * 100:5.1f}%")
    lines.append(f"  [dim]repeat [/dim] {state.repeat_rate * 100:5.1f}%")

    body = "\n".join(lines)
    return Panel(body, title="[bold]Jacobi Monitor[/bold]", border_style="blue", expand=True)


def _render_layout(state: JacobiMonitorState) -> Layout:
    root = Layout()
    root.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3),
    )
    root["left"].update(_render_agg_panel(state))
    root["right"].update(Panel(
        _render_step_table(state),
        title="[bold]Step trace[/bold]",
        border_style="dim",
        expand=True,
    ))
    return root


# ─────────────────────────────────────────────────────────────────────────────
# Public monitor class
# ─────────────────────────────────────────────────────────────────────────────

class JacobiMonitor:
    """
    Context manager that drives a Rich Live display alongside generation.

    Usage:
        with JacobiMonitor(state) as mon:
            for step_stats in generation_iterator:
                mon.update(step_stats)
    """

    def __init__(
        self,
        state: JacobiMonitorState,
        refresh_rate: int = 4,
        console: Optional[Console] = None,
    ):
        self.state = state
        self._console = console or Console()
        self._live = Live(
            _render_layout(state),
            console=self._console,
            refresh_per_second=refresh_rate,
            vertical_overflow="visible",
        )

    def __enter__(self) -> "JacobiMonitor":
        self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        self._live.__exit__(*args)

    def update(self, step: JacobiStepStats) -> None:
        self.state.record(step)
        self._live.update(_render_layout(self.state))


def make_monitor(
    block_size: int,
    max_iters: int,
    init_strategy: str,
    model_name: str,
    ar_tps: float = 0.0,
    refresh_rate: int = 4,
) -> tuple[JacobiMonitorState, JacobiMonitor]:
    """Convenience factory returning (state, monitor) ready for use as ctx manager."""
    state = JacobiMonitorState(
        block_size=block_size,
        max_iters=max_iters,
        init_strategy=init_strategy,
        model_name=model_name,
        ar_tps=ar_tps,
    )
    return state, JacobiMonitor(state, refresh_rate=refresh_rate)
