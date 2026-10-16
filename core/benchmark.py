"""Standardized micro-benchmarking, memory profiling, and FLOP counting."""

from dataclasses import dataclass
import gc
import statistics
import time
from typing import Any, Callable
import torch

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None  # type: ignore


@dataclass
class BenchmarkResult:
    name: str
    median_time_ms: float
    stdev_time_ms: float
    peak_ram_mb: float
    peak_vram_mb: float
    flops: int
    arithmetic_intensity: float = 0.0

    def as_row(self) -> list[Any]:
        return [
            self.name,
            f"{self.median_time_ms:.3f} ± {self.stdev_time_ms:.3f}",
            f"{self.peak_ram_mb:.2f}",
            f"{self.peak_vram_mb:.2f}",
            f"{self.flops:,}" if self.flops > 0 else "N/A",
            f"{self.arithmetic_intensity:.2f}" if self.arithmetic_intensity > 0 else "N/A",
        ]


def benchmark_fn(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 3,
    iters: int = 10,
    name: str = "Operation",
    **kwargs: Any,
) -> BenchmarkResult:
    """Accurately benchmarks latency, memory usage, and FLOP count of a callable.

    Runs warmup cycles, then measures wall-clock execution time and peak allocations.
    """
    gc.collect()
    is_cuda = torch.cuda.is_available() and any(
        isinstance(a, torch.Tensor) and a.is_cuda for a in args
    )

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize()

    # Latency timing
    durations: list[float] = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn(*args, **kwargs)
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        durations.append((end - start) * 1000.0)  # ms

    median_ms = statistics.median(durations)
    stdev_ms = statistics.stdev(durations) if len(durations) > 1 else 0.0

    # VRAM
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 * 1024) if is_cuda else 0.0
    )

    # Estimate FLOPs via PyTorch FlopCounterMode if available
    flops = 0
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as flop_counter:
            _ = fn(*args, **kwargs)
            flops = flop_counter.get_total_flops()
    except Exception:
        flops = 0

    return BenchmarkResult(
        name=name,
        median_time_ms=median_ms,
        stdev_time_ms=stdev_ms,
        peak_ram_mb=0.0,
        peak_vram_mb=peak_vram,
        flops=flops,
    )


def format_benchmark_table(results: list[BenchmarkResult]) -> str:
    """Formats benchmark results as a clean Markdown table."""
    headers = [
        "Model / Op",
        "Latency (ms)",
        "Peak RAM (MB)",
        "Peak VRAM (MB)",
        "FLOPs",
        "Arith. Int. (FLOP/B)",
    ]
    rows = [r.as_row() for r in results]
    if tabulate is not None:
        return tabulate(rows, headers=headers, tablefmt="github")

    # Pure Python fallback
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)
