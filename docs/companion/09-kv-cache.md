# Module 11 Lab — KV-Cached Decoding

This lab accompanies [Module 11: KV-cached decoding](../modules/11-kv-cache.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/11_kv_cache_bench.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/11_kv_cache_bench.py)
    - **Reference modules:** [`companion/minilm/kv_cache.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/kv_cache.py) · [`companion/minilm/model.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/model.py)
    - **Unit tests:** [`companion/tests/test_module_11_kv_cache.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/tests/test_module_11_kv_cache.py)

---

## Objectives

1. **Implement dynamic KV caching**: Store past Key and Value activation projections across autoregressive decode steps to eliminate repeated historical attention compute.
2. **Verify mathematical parity**: Prove that cached generation yields identical logits and tokens compared to naive full-sequence recomputation.
3. **Benchmark generation throughput**: Measure decode latency and memory consumption across growing sequence lengths.

---

## Quickstart

Run the unit tests and generation speedup benchmark:

```bash
cd companion
uv run pytest tests/test_module_11_kv_cache.py -v
uv run python scripts/11_kv_cache_bench.py
```
