# AI-Examples Benchmark Suite & Performance Dashboard

This directory contains standardized benchmarking harnesses for empirical performance comparison across:
1. **PyTorch 2 Eager** (`pytorch2/`)
2. **PyTorch 2 Compiled** (`torch.compile(fullgraph=True)`)
3. **JAX / Equinox JIT** (`jax/`)
4. **Candle Native Rust** (`candle/`)

---

## Metrics Tracked

* **Latency (ms):** High-resolution wall-clock execution time measured via `time.perf_counter_ns` across warm-up and timed iterations (reporting median p50, p90, and p99).
* **Peak RAM (MB):** Process resident memory allocation delta tracked via Python's `tracemalloc`.
* **Peak VRAM (MB):** GPU device memory allocations tracked via `torch.cuda.max_memory_allocated()`.
* **FLOPs & Arithmetic Intensity:** Estimated floating-point operations and operational intensity ($\text{FLOPs/byte}$).
* **Throughput:** Samples per second or tokens per second processed.

---

## Running Benchmarks

```bash
# Run baseline benchmarks on CPU
python benchmarks/run_benchmarks.py --device cpu

# Run with custom warmup and iterations
python benchmarks/run_benchmarks.py --device cpu --warmup 5 --iters 25
```
