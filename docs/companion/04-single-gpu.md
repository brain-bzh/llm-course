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

The companion pins PyTorch 2.9.1. For the `torch.compile` exercise, use the
companion's CPython 3.12 environment: this PyTorch version rejects compilation
on Python 3.14. From the companion repository, `uv sync --python 3.12` selects
that environment. The optional FlashAttention setup has additional Linux/CUDA
requirements in the companion README.

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
warmup. Increase `--warmup` if timing is still changing. The script does not
perform loss/gradient parity checks, so compare them separately before interpreting speed. Run the Module 1 model tests as a baseline; they do not establish eager/compiled parity. `--peak-tflops` is
optional and must match the measured precision and device; no peak is invented.
The FLOP estimate counts block projections, the vocabulary head, and attention
matrix products; it excludes optimizer and elementwise work.

- **0–15 min:** predict whether compute, activation memory, or input transfer limits the baseline.
- **15–40 min:** collect warmed-up repeated timings and peak allocated/reserved bytes for one change.
- **40–60 min:** inspect the separately collected Chrome profiler trace and test the bottleneck hypothesis.
- **60–75 min:** report configuration, timing samples, median, memory, and one limitation.

### Check eager and compiled outputs

The profiling script measures speed but does not compare results. With the
companion environment set to Python 3.12, run the following from the companion
repository. It compares identical SDPA FP32 models on one fixed batch; set
`device = "cuda"` when a CUDA GPU is available. The tolerances are an example
for this fixed FP32 check, not a universal threshold for BF16 or every GPU.

```python
import copy
import torch
from nanolm.model import GPTConfig, MiniGPT

torch.manual_seed(42)
device = "cpu"  # change to "cuda" on an allocated CUDA GPU
config = GPTConfig(vocab_size=512, block_size=64, n_layer=2,
                   n_head=4, n_embd=128, use_sdpa=True, dropout=0.0)
eager = MiniGPT(config).to(device)
compiled_raw = copy.deepcopy(eager)
compiled = torch.compile(compiled_raw)

tokens = torch.randint(0, config.vocab_size, (4, 65), device=device)
x, y = tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()
logits_e, loss_e, _ = eager(x, targets=y)
logits_c, loss_c, _ = compiled(x, targets=y)
loss_e.backward()
loss_c.backward()

torch.testing.assert_close(logits_c, logits_e, rtol=1e-3, atol=1e-4)
torch.testing.assert_close(loss_c, loss_e, rtol=1e-3, atol=1e-4)
for (name_e, p_e), (name_c, p_c) in zip(
        eager.named_parameters(), compiled_raw.named_parameters()):
    assert name_e == name_c
    if p_e.grad is None or p_c.grad is None:
        assert p_e.grad is None and p_c.grad is None, name_e
    else:
        torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-3, atol=1e-4)
print("Fixed-batch logits, loss and gradients agree.")
```

This is a starting check, not proof of equality over all batches, modes or
training trajectories. Separately check the precision and attention variants
you actually benchmark.

The script sets the CPU thread count to one and measures fixed synthetic batches already on the device. It does not
measure the end-to-end data loader or prove useful language quality. Profiler
collection is outside timed samples. Compare like workloads and keep a result
that is slower or inconclusive; speedup is not required to pass.

For a CPU/GPU comparison, run the same mode, shape and warmup/measurement
settings on both devices and record the hardware and thread count. The script's
CPU is single-threaded and its timing excludes the loader and transfers; treat
this as a controlled example, not a general comparison of the devices. CUDA
measurements and conclusions require an allocated GPU.

**Fallback:** inspect CPU traces and explain which conclusions need GPU evidence.
**Extension:** sweep batch/sequence length or measure input-pipeline overhead.
