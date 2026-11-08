# Module 4 Lab — Predict, then measure, one training step

This lab accompanies [Module 4: Single-GPU performance](../modules/04-single-gpu.md).
You will turn the module's derivations into code (FLOPs per token, MFU, static
memory), add a fused attention path to your model, prove that it computes the
same thing, and run **one controlled ablation** with a timing protocol you can defend.

!!! tip "Files used in this lab"
    - **Starter:** [`starter/04_profile_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/04_profile_utils.py)
    - **Your implementations:** [`nanolm/profile_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/profile_utils.py) · SDPA branch in `nanolm/model.py`
    - **Scripts:** [`scripts/04_check_execution.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/04_check_execution.py) (equivalence) · [`scripts/04_single_gpu_perf.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/04_single_gpu_perf.py) (timing, memory, traces)
    - **Tests:** [`tests/test_module_04_profile.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_04_profile.py)

## Connection to the previous labs

The benchmark builds the same `MiniGPT` family and times a forward pass, loss,
backward pass and optimizer update. Its batches are synthetic and already on
the device. This isolates model execution from the data loader of Module 3 and
the checkpointing of Module 2. Measuring input stalls requires a separate
end-to-end experiment.

## Before class (15 minutes)

1. Read [First-principles FLOPs accounting](../modules/04-single-gpu.md#first-principles-flops-accounting-attention-vs-mlp)
   and [Benchmark protocol](../modules/04-single-gpu.md#benchmark-protocol).
2. Use the companion's Python 3.12 environment (`uv sync`); PyTorch 2.9.1 does
   not support `torch.compile` on Python 3.14.
3. If you will use a GPU, find its **dense** peak TFLOP/s at the precision you
   will measure (FP32 and BF16 differ by an order of magnitude; ignore "sparse" figures).

```bash
git switch -c session-12-perf
cp starter/04_profile_utils.py nanolm/profile_utils.py
uv run pytest tests/test_module_04_profile.py -q   # 3 fail with NotImplementedError
```

## Session 12 protocol — 75 minutes

| Time | Activity | Evidence |
| --- | --- | --- |
| 0–15 min | Step 1: FLOPs per token, MFU and static memory | Tests pass; predictions for the benchmark model |
| 15–30 min | Step 2: SDPA branch in your attention; equivalence check | `04_check_execution.py` passes |
| 30–55 min | Step 3: one controlled ablation with repeated, warmed-up timings; one profiler trace | JSON report with samples, median, memory |
| 55–75 min | Step 4: read the trace as a causal story; write the report | Bottleneck hypothesis tested; one stated limitation |

### Step 1 — Count before you measure (15 minutes)

Implement the three functions from the module's derivations:

- **`compute_flops_per_token(config)`**: matrix-multiply parameters are
  $12\,L\,d^2$ (Q, K, V, output, and the two $4\times$ MLP projections) plus
  $V d$ for the tied output head; the embedding lookup is not a matmul.
  Training costs $6\times$ those (2 forward + 4 backward per weight), plus the
  attention products, $12\,L\,d\,T$.
- **`estimate_mfu(tokens_per_sec, flops_per_token, peak_tflops)`**: achieved
  TFLOP/s as a percentage of the peak; reject a non-positive peak.
- **`memory_breakdown(model, optimizer)`**: parameter, gradient and AdamW-moment
  bytes in MiB. For FP32 that totals 16 bytes per parameter.

```bash
uv run pytest tests/test_module_04_profile.py -q
```

Before running anything, predict for the benchmark model (vocabulary 512,
$T=64$, $L=2$, $d=128$, batch 4): FLOPs per training step, and static memory.
Which term dominates the FLOPs at this size: the blocks, the vocabulary head, or attention?

### Step 2 — Add fused attention, then prove it is the same (15 minutes)

Your Module 1 attention is explicit on purpose. To compare it with PyTorch's
`scaled_dot_product_attention` (SDPA), add a branch to `CausalSelfAttention`
that is taken when `config.use_sdpa` is set. Store `config.use_sdpa` and
`config.dropout` in `__init__`; after building `q`, `k`, `v` as
`B × n_head × T × d_head`:

```python
if self.use_sdpa:
    y = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=True,
    )
else:
    ...  # your existing scores, causal mask, softmax and value product
# both paths then merge heads and apply the output projection
```

Without this branch, setting `use_sdpa=True` changes nothing, and the "comparison"
would time manual attention twice. The check script detects that:

```bash
uv run python scripts/04_check_execution.py --device cpu
uv run pytest tests/test_module_01_model.py -k causal -q
# on an allocated GPU, also check compilation:
uv run python scripts/04_check_execution.py --device cuda --compile
```

It verifies that each block really calls SDPA and compares logits, loss and
gradients against the manual model with identical weights. Small differences
are expected: SDPA may select a different kernel on each device.

If your extension is unfinished, commit your model, then restore the reference
and record that choice:
`git show "$(cat .course-reference-revision):nanolm/model.py" > nanolm/model.py`.

### Step 3 — One controlled ablation (25 minutes)

Change **one** thing per comparison. The modes form a chain:
manual FP32 → SDPA FP32 → SDPA BF16. `compile` uses SDPA FP32, so compare it with SDPA FP32.

```bash
# CPU: checks the workflow only
uv run python scripts/04_single_gpu_perf.py --device cpu --modes manual sdpa

# On an allocated GPU (state the device and its peak for the measured precision)
uv run python scripts/04_single_gpu_perf.py --device cuda --modes manual sdpa \
  --trace-dir runs/profile-traces --peak-tflops <dense FP32 peak>
uv run python scripts/04_single_gpu_perf.py --device cuda --modes bf16 --peak-tflops <dense BF16 peak>
uv run python scripts/04_single_gpu_perf.py --device cuda --modes compile
```

Read `scripts/04_single_gpu_perf.py` before trusting it, and find:

- **warmup** steps excluded from timing (compilation happens there; increase
  `--warmup` if timings still drift);
- **`torch.cuda.synchronize()`** before reading the clock. GPU kernels run
  asynchronously, so without it you time the kernel *launches*;
- **repeats** and the **median**, so one slow sample does not decide the result;
- peak allocated vs reserved memory (the allocator caches blocks it no longer uses).

The profiler trace is collected **outside** the timed samples: profiling adds overhead.

### Step 4 — Explain the result (20 minutes)

Open a trace in [Perfetto](https://ui.perfetto.dev) or `chrome://tracing` and
follow one step. Then write a short report:

1. hardware, PyTorch version, mode, batch × sequence, warmup/steps/repeats;
2. timing samples, median tokens/s and, on GPU, peak memory and MFU;
3. your Step 1 prediction next to the measurement: does static memory explain
   the peak? If not, what is the rest (activations, workspace, allocator cache)?
4. one bottleneck hypothesis, and what in the trace supports or refutes it;
5. one limitation: what this benchmark does not measure.

A slower or inconclusive result is a valid result. On a tiny model, kernel
launch overhead can dominate, and a fused kernel may not help.

## Definition of done

- `tests/test_module_04_profile.py` passes with your accounting functions;
- `04_check_execution.py` confirms your SDPA branch matches manual attention;
- one ablation with repeated warmed-up timings, reported with hardware and exclusions;
- a prediction-versus-measurement comparison for memory, with the residual explained.

**Fallback:** without a GPU, run the CPU workflow and the CPU trace, and state
which conclusions need GPU evidence. A single-threaded CPU timing is not
evidence about GPU throughput.

**Extensions:** sweep batch size and sequence length and plot tokens/s and peak
memory; measure the data loader's contribution with real shards; try
activation checkpointing and quantify the memory/time trade-off.

!!! success "What to take away"
    - **Predict, then measure.** A FLOP and byte count tells you what to expect;
      a measurement is informative where it disagrees.
    - **Asynchronous execution lies to naive timers.** Warm up, synchronize, repeat,
      report the median, and keep profiling out of timed regions.
    - **Faster must still be correct.** Check a fused or compiled path against the
      explicit one before timing it.
    - **One change per comparison.** Precision, attention kernel and compilation
      each need their own controlled comparison.
    - **MFU is only as honest as its peak.** Use the dense peak for the exact
      device and precision you measured.
