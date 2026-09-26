# Module 4 Lab — Single-GPU Performance and Profiling

This lab accompanies [Module 4: Single-GPU performance](../modules/04-single-gpu.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/04_single_gpu_perf.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/04_single_gpu_perf.py)
    - **Reference module:** [`nanolm/profile_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/profile_utils.py)
    - **Unit tests:** `tests/test_module_04_profile.py`

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

## Session 12 protocol

**Prepare:** state the model, token shape, precision, and the device's dense
peak throughput at that precision. A CPU run is only a workflow check.

```bash
uv run python scripts/04_single_gpu_perf.py --device cpu --modes manual sdpa
# On an allocated GPU:
uv run python scripts/04_single_gpu_perf.py --device cuda --modes manual sdpa \
  --trace-dir runs/profile-traces
uv run python scripts/04_single_gpu_perf.py --device cuda --modes bf16
uv run python scripts/04_single_gpu_perf.py --device cuda --modes compile
```

The modes form successive comparisons: manual FP32 → SDPA FP32 → SDPA BF16;
compile uses SDPA FP32 so compare it with SDPA FP32. Compilation happens during
warmup. Run correctness tests before interpreting speed. `--peak-tflops` is
optional and must match the measured precision and device; no peak is invented.
The FLOP estimate counts block projections, the vocabulary head, and attention
matrix products; it excludes optimizer and elementwise work.

- **0–15 min:** predict whether compute, activation memory, or input transfer limits the baseline.
- **15–40 min:** collect warmed-up repeated timings and peak allocated/reserved bytes for one change.
- **40–60 min:** inspect the separately collected Chrome profiler trace and test the bottleneck hypothesis.
- **60–75 min:** report configuration, timing samples, median, memory, and one limitation.

The script measures fixed synthetic batches already on the device. It does not
measure the end-to-end data loader or prove useful language quality. Profiler
collection is outside timed samples. Compare like workloads and keep a result
that is slower or inconclusive; speedup is not required to pass.

**Fallback:** inspect CPU traces and explain which conclusions need GPU evidence.
**Extension:** sweep batch/sequence length or measure input-pipeline overhead.
