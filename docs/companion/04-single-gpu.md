# Module 4 Lab — Single-GPU Performance and Profiling

This lab accompanies [Module 4: Single-GPU performance](../modules/04-single-gpu.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab scripts:** [`scripts/04_single_gpu_perf.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/04_single_gpu_perf.py) · [`scripts/04_check_execution.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/04_check_execution.py)
    - **Reference module:** [`nanolm/profile_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/profile_utils.py)
    - **Unit tests:** `tests/test_module_04_profile.py`

---

## Connection to the previous labs

The benchmark constructs the same `MiniGPT` family and executes a forward pass,
loss, backward pass and optimizer update. Its batches are synthetic and already
on the selected device: this isolates model execution, not the full baseline,
its checkpoint or the data loader from Module 3. Use the trace to investigate
operations in that update; measuring input stalls requires a separate end-to-end
experiment.

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

## Prepare the attention comparison

The Module 1 exercise deliberately implemented only manual attention. Before
comparing `manual` and `sdpa`, either add the SDPA path to your own model or use
the provided reference model. Merely setting `use_sdpa=True` does not change a
student implementation that ignores the flag: such a comparison would time
manual attention twice.

To extend your model, save `config.use_sdpa` and `config.dropout` in
`CausalSelfAttention`. After producing `q`, `k`, and `v` in
`B × n_head × T × d_head` layout, use this branch in place of the manual
score/mask/softmax/value multiplication when the flag is set:

```python
if self.use_sdpa:
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,
    )
else:
    # Keep your existing scaled scores, causal mask, softmax and value product.
    ...
# Both paths then merge heads and apply the output projection.
```

Compare outputs and gradients for identical manual and SDPA models with dropout
disabled, then rerun the causality tests:

```bash
uv run python scripts/04_check_execution.py --device cpu
uv run pytest tests/test_module_01_model.py -k causal -q
# On an allocated GPU, also check compilation:
uv run python scripts/04_check_execution.py --device cuda --compile
```

The script checks that each block actually calls SDPA as well as comparing
logits, loss and gradients. Small floating-point differences are expected; SDPA
may choose different backends on different hardware. If the
extension is unfinished, commit your model first, then restore the reference
version recorded at the start of Module 1:

```bash
git show "$(cat .course-reference-revision):nanolm/model.py" > nanolm/model.py
```

Record that choice. If the revision file is missing, obtain the matching
reference revision from the instructor; do not overwrite your work before it
is committed.

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

Mixed precision belongs here: `bf16` runs supported operations in bfloat16
while retaining appropriate higher-precision state where the implementation
requires it. This is a hardware and numerical trade-off to measure, not a
necessary feature of the FP32 loop built in Module 2. Check numerical behavior
as well as throughput; BF16 support depends on the GPU.

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

The check script above compares eager and compiled SDPA FP32 models with
identical weights on one fixed batch. Its example tolerances are not universal
thresholds for BF16 or every GPU. It does not establish equality over all inputs
or training trajectories; check the precision and attention variants used in
your experiment as well.

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
