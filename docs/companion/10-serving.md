# Module 12 Lab — Serving Systems

This lab accompanies [Module 12: Serving systems](../modules/12-serving.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/12_serving_benchmark.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/12_serving_benchmark.py)
    - **Reference module:** [`companion/minilm/serving_sim.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/serving_sim.py)
    - **Unit tests:** `companion/tests/test_module_12_serving.py` *(TODO)*

---

## Objectives

1. **Continuous batching**: Simulate iteration-level scheduling where arriving requests join the batch immediately and finished requests exit without blocking others.
2. **Paged KV-cache memory allocation**: Manage fragmented Key-Value memory via virtual-to-physical block tables.
3. **Serving metrics**: Measure Time to First Token (TTFT), Inter-Token Latency (ITL), request queue delays, and throughput under load.

---

## Quickstart

Run the discrete-event continuous batching simulation:

```bash
cd companion
uv run python scripts/12_serving_benchmark.py
```
