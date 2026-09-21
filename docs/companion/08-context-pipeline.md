# Module 8 Lab — Context, Pipeline, and Expert Parallelism

This lab accompanies [Module 8: Context, pipeline, and expert parallelism](../modules/08-context-pipeline.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/08_parallelism_sizing.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/08_parallelism_sizing.py)
    - **Reference module:** [`minilm/parallelism_calc.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/parallelism_calc.py)
    - **Unit tests:** `tests/test_module_08_parallelism.py` *(TODO)*

---

## Objectives

1. **Pipeline bubble estimation**: Calculate idle bubbles for 1F1B (One Forward, One Backward) schedules across micro-batch counts.
2. **Context Parallelism memory scaling**: Trace how Ring Attention and sequence chunking remove the quadratic attention activation memory bottleneck.
3. **Expert Parallelism & All-to-All volume**: Quantify token routing data transfers across sparse MoE experts.
4. **Multidimensional cluster sizing & hierarchy verification**: Run analytical calculations that order parallel dimensions from outer to inner ($\text{PP} \to \text{DP} \to \text{FSDP} \to \text{EP} \to \text{TP}$) grounded in communicated byte volume, frequency, and physical network topologies (NVLink vs InfiniBand).

---

## Quickstart

Run the analytical parallelism sizing calculator:

```bash
cd companion
uv run python scripts/08_parallelism_sizing.py
```
