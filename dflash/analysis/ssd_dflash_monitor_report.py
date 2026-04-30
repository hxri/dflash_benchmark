from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from rich import print
from rich.table import Table


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find_latest_run(base: Path) -> Path:
    if (base / "aggregate.json").exists():
        return base
    candidates = sorted(
        [p for p in base.glob("*") if p.is_dir() and (p / "aggregate.json").exists()],
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No run directory with aggregate.json under {base}")
    return candidates[-1]


def _report(run_dir: Path) -> None:
    aggregate = json.loads((run_dir / "aggregate.json").read_text())
    traces = []
    samples = []
    for p in sorted(run_dir.glob("trace_rank*.jsonl")):
        traces.extend(_load_jsonl(p))
    for p in sorted(run_dir.glob("samples_rank*.jsonl")):
        samples.extend(_load_jsonl(p))

    agg = aggregate["aggregate"]
    meta = aggregate["metadata"]

    table = Table(title="SSD+DFlash Monitor Report", show_header=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Model", meta["model"])
    table.add_row("Draft", meta["draft_model"])
    table.add_row("Dataset", meta["dataset"])
    table.add_row("World size", str(meta["world_size"]))
    table.add_row("Block size", str(meta["block_size"]))
    table.add_row("Mean decode tok/s", f"{agg['mean_decode_tps']:.2f}")
    table.add_row("Median decode tok/s", f"{agg['median_decode_tps']:.2f}")
    table.add_row("Mean TTFT (s)", f"{agg['mean_ttft']:.4f}")
    table.add_row("Acceptance utilization", f"{agg['acceptance_utilization'] * 100:.1f}%")
    drift = agg.get("mean_hidden_drift_cosine")
    table.add_row("Hidden drift cosine", "n/a" if drift is None else f"{drift:.4f}")
    print(table)

    if traces:
        step_ms = np.array([t["step_ms"] for t in traces], dtype=float)
        acc_ratio = np.array([t["acceptance_ratio"] for t in traces], dtype=float)
        tps = np.array([t["cumulative_decode_tps"] for t in traces], dtype=float)
        drift_rows = [t["hidden_drift_cosine"] for t in traces if t["hidden_drift_cosine"] is not None]

        dist = Table(title="Step Distribution", show_header=True)
        dist.add_column("Metric")
        dist.add_column("Value", justify="right")
        dist.add_row("P50 step ms", f"{np.percentile(step_ms, 50):.2f}")
        dist.add_row("P90 step ms", f"{np.percentile(step_ms, 90):.2f}")
        dist.add_row("P99 step ms", f"{np.percentile(step_ms, 99):.2f}")
        dist.add_row("Mean acceptance ratio", f"{np.mean(acc_ratio):.3f}")
        dist.add_row("P10 acceptance ratio", f"{np.percentile(acc_ratio, 10):.3f}")
        dist.add_row("P10 decode tps", f"{np.percentile(tps, 10):.2f}")
        if drift_rows:
            drift_arr = np.array(drift_rows, dtype=float)
            dist.add_row("P10 drift cosine", f"{np.percentile(drift_arr, 10):.4f}")
        print(dist)

    print("\nWhat to look for while decoding:")
    util = agg["acceptance_utilization"]
    if util < 0.45:
        print("  - Frequent low acceptance means draft mismatch; consider smaller block size or better matched draft.")
    elif util < 0.65:
        print("  - Mid acceptance is workable; monitor long prompts for degradation in later steps.")
    else:
        print("  - High acceptance indicates healthy SSD+DFlash coupling; focus next on throughput scaling.")

    if traces:
        p99 = float(np.percentile(step_ms, 99))
        p50 = float(np.percentile(step_ms, 50))
        if p99 > 2.0 * p50:
            print("  - Large p99/p50 step latency spread suggests kernel or scheduling jitter; check GPU utilization and thermals.")
        else:
            print("  - Step latency is stable; pipeline is compute-bound rather than jitter-bound.")

    if agg.get("mean_hidden_drift_cosine") is not None and agg["mean_hidden_drift_cosine"] < 0.85:
        print("  - Hidden drift is elevated; watch for acceptance collapse in long generations.")
    else:
        print("  - Hidden drift is within a stable range for stale-state reuse.")

    print(f"\nRun directory: {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze SSD+DFlash monitor logs and summarize decode health.")
    p.add_argument("--run-dir", default="results/ssd_dflash_accel")
    args = p.parse_args()

    run_dir = _find_latest_run(Path(args.run_dir))
    _report(run_dir)


if __name__ == "__main__":
    main()
