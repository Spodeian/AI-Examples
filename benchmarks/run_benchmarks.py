"""Command-line cross-track benchmark runner emitting markdown performance summaries."""

from __future__ import annotations

import argparse
import sys

import torch

from core.benchmark import BenchmarkResult, benchmark_fn, format_benchmark_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-Examples Cross-Track Performance Benchmark Runner"
    )
    parser.add_argument("--category", type=str, default=None, help="Filter by category slug")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--iters", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    return parser.parse_args()


def run_baseline_benchmarks(device: str, warmup: int, iters: int) -> list[BenchmarkResult]:
    """Runs baseline micro-benchmarks on core tensor primitives."""
    results: list[BenchmarkResult] = []

    # 1. Linear matmul GEMM [512, 512] x [512, 512]
    A = torch.randn(512, 512, device=device)
    B = torch.randn(512, 512, device=device)
    res_gemm = benchmark_fn(
        lambda: A @ B,
        warmup=warmup,
        iters=iters,
        batch_size=512,
        name="Dense GEMM (512x512 FP32)",
    )
    results.append(res_gemm)

    # 2. LayerNorm [64, 128, 512]
    X_ln = torch.randn(64, 128, 512, device=device)
    ln = torch.nn.LayerNorm(512).to(device)
    res_ln = benchmark_fn(
        lambda: ln(X_ln),
        warmup=warmup,
        iters=iters,
        batch_size=64,
        name="LayerNorm (64x128x512)",
    )
    results.append(res_ln)

    # 3. Softmax Attention Score Matrix [4, 8, 128, 128]
    Q = torch.randn(4, 8, 128, 64, device=device)
    K = torch.randn(4, 8, 128, 64, device=device)
    scale = 1.0 / (64**0.5)

    def sdpa_score() -> torch.Tensor:
        scores = (Q @ K.transpose(-2, -1)) * scale
        return torch.softmax(scores, dim=-1)

    res_attn = benchmark_fn(
        sdpa_score,
        warmup=warmup,
        iters=iters,
        batch_size=4,
        name="SDPA Softmax Scores (4x8x128x128)",
    )
    results.append(res_attn)

    return results


def main() -> int:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU.", file=sys.stderr)
        device = "cpu"

    print(f"=== Running AI-Examples Benchmarks (Device: {device.upper()}) ===")
    results = run_baseline_benchmarks(device, args.warmup, args.iters)
    table_md = format_benchmark_table(results)
    print("\n" + table_md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
