# Module 10 Lab — Serving Systems

This lab accompanies [Module 10: Serving systems](../modules/10-serving.md).

!!! info "Live Demonstration & Optional Exploration"
    In Session 28, serving architectures are demonstrated live by the instructor using production vLLM and educational implementations. There is no mandatory lab assignment or report to submit for this module, preserving student focus for [Reproduction Studio 4](../schedule.md#phase-3-serve-and-synthesize).
    
    The simulation scripts below and [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) are provided as clean, readable references for students who want to explore iteration-level scheduling and paged memory allocation in Python.

!!! tip "Exploration & Simulation Resources"
    - **Educational reference:** [nano-vllm (GitHub)](https://github.com/GeeeekExplorer/nano-vllm) — lightweight Python implementation of vLLM's PagedAttention and continuous batching scheduler.
    - **Lab simulation script:** [`scripts/10_serving_benchmark.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/10_serving_benchmark.py)
    - **Reference simulator module:** [`minilm/serving_sim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/serving_sim.py)

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
uv run python scripts/10_serving_benchmark.py
```
