# Module 8 Lab — Context, Pipeline, and Expert Parallelism

This lab accompanies [Module 8: Context, pipeline, and expert parallelism](../modules/08-context-pipeline.md).

!!! tip "Practical Lab & Simulator Resources"
    To explore multidimensional and pipeline parallelism concepts:

    - **Interactive simulator:** [Pipeline Parallelism Schedule Simulator](../demos/pipeline-parallelism.html){ target="_blank" } — interactive schedule comparison (Naive, AFAB, 1F1B, Interleaved), bubble ratios, memory bars, and boundary communication graph.
    - **Lab script:** [`scripts/08_parallelism_sizing.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/08_parallelism_sizing.py)
    - **Reference module:** [`nanolm/parallelism_calc.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/parallelism_calc.py)
    - **Unit tests:** `tests/test_module_08_parallelism.py`

---

## Objectives

1. **Pipeline bubble estimation**: Calculate idle bubbles for 1F1B (One Forward, One Backward) schedules across micro-batch counts.
2. **Context Parallelism memory scaling**: Trace how Ring Attention and sequence chunking remove the quadratic attention activation memory bottleneck.
3. **Expert Parallelism & All-to-All volume**: Quantify token routing data transfers across sparse MoE experts.
4. **Device and state accounting**: Count independent PP/TP/CP/DP axes, subdivide DP for FSDP or EP, and distinguish persistent state from peak memory. Network placement is a hypothesis to test, not a fixed hierarchy.

---

## Quickstart

Run the analytical parallelism sizing calculator:

```bash
uv run pytest tests/test_module_08_parallelism.py -q
uv run python scripts/08_parallelism_sizing.py
```

## Session 23 protocol

- **Prepare:** read the module's rank-group definitions and complete one bubble calculation.
- **0–15 min:** draw the eight-device MoE example; identify shared and expert replicas.
- **15–35 min:** run the calculator and compare DDP, FSDP, and EP at fixed device count. Explain why EP does not divide attention weights.
- **35–55 min:** use the pipeline simulator to compare AFAB and 1F1B at fixed stages and micro-batches. Report idle/total and activation residency separately.
- **55–75 min:** defend two network placements and name the memory and communication measurements needed to decide between them.

**Minimum evidence:** consistent devices and global tokens/update, explicit group
membership, persistent-state estimate with exclusions, and one falsifiable
placement hypothesis. The calculator does not predict peak VRAM or network time.

**Fallback:** use the module's worked rank layout and hand-calculate its device
count and pipeline fraction. Combined EP/FSDP layouts are an optional extension
requiring separate expert synchronization and shard-group definitions.
