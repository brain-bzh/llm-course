# Module 9 Lab — KV-Cached Decoding

This lab accompanies [Module 9: KV-cached decoding](../modules/09-kv-cache.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/09_kv_cache_bench.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/09_kv_cache_bench.py)
    - **Reference modules:** [`nanolm/kv_cache.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/kv_cache.py) · [`nanolm/model.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/model.py)
    - **Unit tests:** [`tests/test_module_09_kv_cache.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_09_kv_cache.py)

---

## Objectives

1. **Implement dynamic KV caching**: Store past Key and Value activation projections across autoregressive decode steps to eliminate repeated historical attention compute.
2. **Verify mathematical parity**: Check cached logits within numerical tolerances at every position and compare greedy tokens with full-sequence recomputation.
3. **Benchmark generation throughput**: Measure decode latency and memory consumption across growing sequence lengths.

---

## Quickstart

Run the unit tests and generation speedup benchmark:

```bash
uv run pytest tests/test_module_09_kv_cache.py -v
uv run python scripts/09_kv_cache_bench.py
```

## Session 26 protocol

- **0–15 min:** predict cache shape and bytes for a fixed prompt/output length.
- **15–35 min:** run tests covering every decode position, both attention paths, and the learned-position context boundary.
- **35–55 min:** run the benchmark for two prompt lengths and compare model TTFT with subsequent decode intervals.
- **55–75 min:** explain the result, report variability, and identify which memory and timing costs the benchmark excludes.

```bash
uv run python scripts/09_kv_cache_bench.py --prompt-tokens 16 --new-tokens 16
uv run python scripts/09_kv_cache_bench.py --prompt-tokens 64 --new-tokens 16
```

Use `--device cuda` on allocated GPUs. The default CPU run is a correctness and
measurement exercise, not evidence of GPU bandwidth saturation. The benchmark
warms up both paths, synchronizes accelerator execution, and repeats timings.
Model TTFT excludes request queues, networking, and tokenization. Decode intervals
include Python and greedy selection overhead. Logical cache bytes exclude temporary
allocations; the growing cache currently uses concatenation.

**Minimum evidence:** logit/greedy-token agreement, workload/configuration, prefill
and decode measurements, and predicted versus reported cache bytes. A speedup is
an observation, not a passing condition. The cached path rejects requests beyond
its learned-position window; it does not silently emulate sliding-window decoding.

**Fallback:** use the CPU run and explain the cache arithmetic without claiming a
GPU speedup. **Extension:** compare preallocation with concatenation at fixed outputs.
