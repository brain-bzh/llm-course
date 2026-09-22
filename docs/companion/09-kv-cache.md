# Module 9 Lab — KV-Cached Decoding

This lab accompanies [Module 9: KV-cached decoding](../modules/09-kv-cache.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/09_kv_cache_bench.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/09_kv_cache_bench.py)
    - **Reference modules:** [`minilm/kv_cache.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/kv_cache.py) · [`minilm/model.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/model.py)
    - **Unit tests:** [`tests/test_module_09_kv_cache.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_09_kv_cache.py)

---

## Objectives

1. **Implement dynamic KV caching**: Store past Key and Value activation projections across autoregressive decode steps to eliminate repeated historical attention compute.
2. **Verify mathematical parity**: Prove that cached generation yields identical logits and tokens compared to naive full-sequence recomputation.
3. **Benchmark generation throughput**: Measure decode latency and memory consumption across growing sequence lengths.

---

## Quickstart

Run the unit tests and generation speedup benchmark:

```bash
uv run pytest tests/test_module_09_kv_cache.py -v
uv run python scripts/09_kv_cache_bench.py
```
