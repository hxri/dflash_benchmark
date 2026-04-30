"""
Real-time rich monitor for DFlash-SSD generation.

Displays a live table of per-step metrics and tells the user exactly
what to look for as decoding proceeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class StepStats:
    step: int
    t_verify_ms: float
    t_draft_ms: float
    t_wall_ms: float          # actual wall time for the step (verify ‖ draft)
    cache_hit: bool
    acceptance_len: int
    block_size: int
    h_cosine_sim: Optional[float]  # None for first step
    fan_out_lengths: list[int]     # which acceptance lengths were pre-generated


@dataclass
class RunningStats:
    steps: list[StepStats] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.steps)

    @property
    def hit_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(s.cache_hit for s in self.steps) / len(self.steps)

    @property
    def mean_accept(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.acceptance_len for s in self.steps]))

    @property
    def mean_h_sim(self) -> float:
        sims = [s.h_cosine_sim for s in self.steps if s.h_cosine_sim is not None]
        return float(np.mean(sims)) if sims else 0.0

    @property
    def draft_latency_saved_ms(self) -> float:
        """Average draft ms saved per step (only steps where draft ran in parallel)."""
        if not self.steps:
            return 0.0
        # On cache-hit steps, draft was concurrent → latency hidden = t_draft_ms
        saved = sum(s.t_draft_ms for s in self.steps if s.cache_hit)
        return saved / len(self.steps)

    @property
    def mean_t_verify_ms(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.t_verify_ms for s in self.steps]))

    @property
    def mean_t_draft_ms(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.t_draft_ms for s in self.steps]))

    @property
    def mean_t_wall_ms(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([s.t_wall_ms for s in self.steps]))

    def throughput_tps(self) -> float:
        """Tokens per second based on wall time and acceptance lengths."""
        total_ms = sum(s.t_wall_ms for s in self.steps)
        total_toks = sum(s.acceptance_len for s in self.steps)
        return (total_toks / total_ms * 1000) if total_ms > 0 else 0.0


def _color_sim(v: Optional[float]) -> str:
    if v is None:
        return "—"
    if v >= 0.95:
        return f"[green]{v:.3f}[/green]"
    if v >= 0.85:
        return f"[yellow]{v:.3f}[/yellow]"
    if v >= 0.70:
        return f"[orange1]{v:.3f}[/orange1]"
    return f"[red]{v:.3f}[/red]"


def _color_hit(hit: bool) -> str:
    return "[green]HIT [/green]" if hit else "[dim]MISS[/dim]"


def _color_accept(acc: int, block_size: int) -> str:
    frac = acc / block_size
    s = f"{acc:2d}/{block_size}"
    if frac >= 0.75:
        return f"[green]{s}[/green]"
    if frac >= 0.50:
        return f"[yellow]{s}[/yellow]"
    return f"[red]{s}[/red]"


def _color_hit_rate(r: float) -> str:
    if r >= 0.70:
        return f"[green]{r*100:.0f}%[/green]"
    if r >= 0.45:
        return f"[yellow]{r*100:.0f}%[/yellow]"
    return f"[red]{r*100:.0f}%[/red]"


class LiveMonitor:
    """
    Rich live display for DFlash-SSD generation.

    Usage:
        monitor = LiveMonitor(model_name="Qwen/Qwen3-4B",
                              draft_name="z-lab/Qwen3-4B-DFlash-b16",
                              block_size=16)
        with monitor:
            for step ...:
                # run step
                monitor.record(StepStats(...))
    """

    _TAIL = 20  # max rows shown in table

    def __init__(
        self,
        model_name: str,
        draft_name: str,
        block_size: int,
        fan_out: int,
        ar_tps: Optional[float] = None,
        dflash_tps: Optional[float] = None,
    ):
        self.model_name = model_name
        self.draft_name = draft_name
        self.block_size = block_size
        self.fan_out = fan_out
        self.ar_tps = ar_tps
        self.dflash_tps = dflash_tps
        self.stats = RunningStats()
        self._console = Console()
        self._live: Optional[Live] = None

    def __enter__(self):
        self._print_header()
        self._live = Live(self._render(), console=self._console, refresh_per_second=4)
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        if self._live:
            self._live.__exit__(*args)
        self._print_footer()

    def record(self, s: StepStats):
        self.stats.steps.append(s)
        if self._live:
            self._live.update(self._render())

    def _print_header(self):
        self._console.print()
        self._console.rule("[bold cyan]DFlash-SSD  Live Monitor[/bold cyan]")
        self._console.print(
            f"  Target : [cyan]{self.model_name}[/cyan]  (cuda:0)\n"
            f"  Draft  : [cyan]{self.draft_name}[/cyan]  (cuda:1)\n"
            f"  Block  : {self.block_size} tokens  |  Fan-out F = {self.fan_out}"
        )
        self._console.print()
        self._console.print(
            Panel(
                _guidance_text(),
                title="[bold]What to look for[/bold]",
                border_style="dim",
                expand=False,
            )
        )
        self._console.print()

    def _print_footer(self):
        rs = self.stats
        self._console.print()
        self._console.rule("[bold]Final Summary[/bold]")
        self._console.print(
            f"  Steps:              {rs.n}\n"
            f"  Cache hit rate:     {_color_hit_rate(rs.hit_rate)}\n"
            f"  Mean accept len:    {rs.mean_accept:.2f} / {self.block_size}\n"
            f"  Mean H cos_sim:     {rs.mean_h_sim:.4f}\n"
            f"  Mean T_verify:      {rs.mean_t_verify_ms:.1f} ms\n"
            f"  Mean T_draft:       {rs.mean_t_draft_ms:.1f} ms\n"
            f"  Mean T_wall:        {rs.mean_t_wall_ms:.1f} ms\n"
            f"  Draft latency saved:{rs.draft_latency_saved_ms:.1f} ms/step\n"
            f"  DFlash-SSD tput:    {rs.throughput_tps():.1f} tok/s"
        )
        if self.dflash_tps and rs.throughput_tps() > 0:
            speedup = rs.throughput_tps() / self.dflash_tps
            color = "green" if speedup > 1.05 else ("yellow" if speedup > 0.95 else "red")
            self._console.print(
                f"  vs DFlash:          [{color}]{speedup:.2f}x[/{color}]"
            )
        if self.ar_tps and rs.throughput_tps() > 0:
            self._console.print(
                f"  vs AR baseline:     {rs.throughput_tps() / self.ar_tps:.2f}x"
            )
        self._console.rule()

    def _render(self):
        rs = self.stats
        tail = self.stats.steps[-self._TAIL:]

        # Step table
        tbl = Table(box=None, show_header=True, header_style="bold dim",
                    pad_edge=False, collapse_padding=True)
        tbl.add_column("step",  style="dim",  width=5,  justify="right")
        tbl.add_column("T_ver", width=8,  justify="right")
        tbl.add_column("T_dft", width=8,  justify="right")
        tbl.add_column("T_wall",width=8,  justify="right")
        tbl.add_column("cached",width=6)
        tbl.add_column("accept",width=6)
        tbl.add_column("H_sim", width=7,  justify="right")

        for s in tail:
            tbl.add_row(
                str(s.step),
                f"{s.t_verify_ms:.1f}ms",
                f"{s.t_draft_ms:.1f}ms",
                f"{s.t_wall_ms:.1f}ms",
                _color_hit(s.cache_hit),
                _color_accept(s.acceptance_len, s.block_size),
                _color_sim(s.h_cosine_sim),
            )

        # Summary bar
        summary_lines = [
            f"  HIT {_color_hit_rate(rs.hit_rate)}"
            f"  |  ACCEPT [bold]{rs.mean_accept:.1f}[/bold]/{self.block_size}"
            f"  |  H_SIM [bold]{rs.mean_h_sim:.3f}[/bold]"
            f"  |  DRAFT SAVED [bold]{rs.draft_latency_saved_ms:.1f}ms[/bold]/step",
            f"  T_wall [bold]{rs.mean_t_wall_ms:.1f}ms[/bold]"
            f"  |  TPUT [bold]{rs.throughput_tps():.1f} tok/s[/bold]"
            + (f"  |  vs DFlash [bold]{rs.throughput_tps()/self.dflash_tps:.2f}x[/bold]"
               if self.dflash_tps else ""),
        ]
        summary = Panel("\n".join(summary_lines), expand=True,
                        border_style="cyan", title=f"[dim]step {rs.n}[/dim]")

        from rich.console import Group
        return Group(tbl, summary)


def _guidance_text() -> str:
    return (
        "[bold green]GREEN  (approach working well)[/bold green]\n"
        "  • Cache hit rate  > 60%  — stale-H prediction is effective\n"
        "  • H cos_sim       > 0.90 — consecutive hidden states are similar\n"
        "  • T_draft < T_verify     — pre-speculation fits inside verify window\n"
        "  • Accept retention> 85%  — minimal quality loss from stale H\n"
        "\n"
        "[bold yellow]YELLOW (marginal — tune these)[/bold yellow]\n"
        "  • Cache hit rate  40-60% — increase fan-out F or try running predictor\n"
        "  • H cos_sim   0.75-0.90  — consider adding fast-refinement DFlash pass\n"
        "  • Accept retention 70-85%— may still be net-positive for throughput\n"
        "\n"
        "[bold red]RED    (investigate)[/bold red]\n"
        "  • Cache hit rate  < 40%  — acceptance-length prediction is failing\n"
        "  • H cos_sim       < 0.75 — hidden states changing too fast (code/math?)\n"
        "  • T_draft > T_verify     — draft does not fit in verify window; reduce F\n"
        "  • Accept retention< 70%  — stale H degrades quality unacceptably\n"
        "\n"
        "[dim]KEY FORMULA: net speedup ≈ (T_verify + T_draft) / max(T_verify, F × T_draft)[/dim]"
    )
