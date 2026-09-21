# Module 10 Lab — Context and Pipeline Parallelism

This lab accompanies [Module 10: Context and pipeline parallelism](../modules/10-context-pipeline.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/10_parallelism_sizing.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/10_parallelism_sizing.py)
    - **Reference module:** [`companion/minilm/parallelism_calc.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/parallelism_calc.py)
    - **Unit tests:** `companion/tests/test_module_10_parallelism.py` *(TODO)*

---

## Objectives

1. **Pipeline bubble estimation**: Calculate idle bubbles for 1F1B (One Forward, One Backward) and interleaved schedules across micro-batch counts.
2. **Context Parallelism memory scaling**: Trace how Ring Attention and sequence chunking remove the quadratic attention activation memory bottleneck.
3. **Multidimensional cluster sizing**: Run analytical calculations that trade off TP, CP, PP, and DP given hardware network topologies (NVLink vs InfiniBand).

---

## Quickstart

Run the analytical parallelism sizing calculator:

```bash
cd companion
uv run python scripts/10_parallelism_sizing.py
```
