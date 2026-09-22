# Module 4 Lab — Single-GPU Performance and Profiling

This lab accompanies [Module 4: Single-GPU performance](../modules/04-single-gpu.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/04_single_gpu_perf.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/04_single_gpu_perf.py)
    - **Reference module:** [`minilm/profile_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/profile_utils.py)
    - **Unit tests:** `tests/test_module_04_profile.py` *(TODO)*

---

## Objectives

1. **Profile execution breakdown**: Measure runtime spent in attention, MLP projections, normalization, and memory transfers using PyTorch Profiler.
2. **Compute arithmetic intensity & MFU**: Calculate theoretical FLOPs per token and assess Model FLOPs Utilization against accelerator peak specs.
3. **Ablate optimizations**: Isolate the impact of `bfloat16`, scaled dot-product attention (SDPA/FlashAttention), and `torch.compile`.

---

## Quickstart

Run the single-GPU profiling benchmark:

```bash
uv run python scripts/04_single_gpu_perf.py
```
