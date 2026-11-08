# Module 4 — Single-GPU performance

## Purpose

Build a hardware model of one GPU, then use it to predict and explain the
performance of the course training step. Optimization begins with a bottleneck
hypothesis, not a bag of flags.

The main reference is the Scaling Book chapter
[How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). Its
hardware numbers are examples, not universal constants: always pair a formula
with the specification of the GPU actually used.

## Learning goals

By the end of the module, students should be able to:

- distinguish CUDA cores, Tensor Cores, SMs and GPU memory levels;
- explain warps, occupancy, divergence and memory coalescing at a useful level;
- calculate arithmetic intensity and place an operation on a roofline;
- decompose training memory into persistent state, activations and temporary
  workspaces;
- read a profiler trace as a timeline of CPU launch, GPU kernels and transfers;
- choose an optimization that targets the measured bottleneck;
- report throughput, memory and numerical behavior without changing the task.

## The course workload: from Python to kernels

The previous modules built a decoder-only GPT, its training loop, and token
batches. One update takes a batch through the model, computes next-token loss,
runs backward, and updates the weights. These familiar operations give us
something concrete to profile:

| Course operation | Work to look for | Question to investigate |
| :--- | :--- | :--- |
| Q/K/V, attention output, MLP projections | Matrix multiplications | Are dimensions and dtype suitable for an efficient kernel? |
| Attention | Matrix products, masking, softmax, weighted values | Which SDPA backend ran, and are intermediate scores materialized? |
| LayerNorm, GELU, residual additions | Reductions and elementwise work | Are many small kernels or intermediate tensors expensive? |
| Loss and backward | Softmax/reductions and gradient kernels | Which operations dominate the update? |
| Optimizer and batch preparation | Parameter updates, CPU work, possible transfers | Is the GPU busy, or waiting for launches or input? |

These are starting hypotheses, not labels that can be inferred from the Python
source alone. Kernel choice and cost depend on the workload and hardware.

### CPU versus GPU

A CPU has relatively few, powerful cores and is well suited to control flow,
input processing, and small or irregular work. A GPU has many execution lanes
and high device-memory bandwidth, and can process large amounts of regular
parallel tensor work efficiently. Both run tensor operations; neither is always
faster. A tiny model or batch may not amortize GPU launch and transfer costs.

In a CUDA training run, Python and CPU-side code prepare work and enqueue GPU
kernels. The GPU executes asynchronously using tensors in device memory.
Copying batches from host memory, reading results back, or requesting a scalar
can introduce transfers or synchronization. Time the full update and state
whether input preparation and transfers are included.

The companion benchmark is a deliberately limited model-step experiment: it
uses a **fixed synthetic batch already on the selected device**, sets one CPU
thread, and excludes data loading and host-to-device transfers from timing. Its
CPU mode checks the workflow. If you also run the same configuration on CUDA,
report that thread count and these exclusions; it is not a general CPU/GPU hardware
comparison. A separate end-to-end experiment must include the actual loader and
transfers on both paths.

## A GPU is not one fast processor

A modern NVIDIA training GPU contains many **streaming multiprocessors** (SMs).
Each SM combines:

- **Tensor Cores** specialized for dense low-precision matrix multiplication;
- **CUDA cores** for flexible scalar/vector arithmetic and reductions;
- warp schedulers that issue instructions to groups of 32 threads;
- registers and shared memory/L1 close to the compute units.

The full device also has a shared L2 cache and high-bandwidth memory (HBM).
Calling all of these pieces “GPU compute” hides the questions that matter:

- Is the kernel feeding Tensor Cores with supported shapes and precision?
- Are enough independent warps resident to hide memory latency?
- Are threads reading adjacent memory locations?
- Does the working set remain on chip or repeatedly travel to HBM?
- Are many tiny kernels dominated by launch overhead?

## GEMM and the operations in a Transformer

**GEMM** means general matrix–matrix multiplication:

\[
C \leftarrow \alpha AB + \beta C,
\quad A\in\mathbb{R}^{M\times K},\quad
B\in\mathbb{R}^{K\times N},\quad C\in\mathbb{R}^{M\times N}.
\]

Ordinary `A @ B` is the case \(\alpha=1,\beta=0\). In the course GPT,
linear projections flatten batch and token positions into rows: Q, K and V,
the attention output projection, the two MLP projections, and the vocabulary
head all involve matrix products. Attention also performs batched products
\(QK^\top\) and \(PV\) per head. These operations account for much of the
model's arithmetic, so their shape, dtype and implementation matter.

Tensor Cores accelerate supported matrix-tile multiply–accumulate operations;
general CUDA cores handle many scalar and vector instructions. Reductions and
special-function paths also contribute to operations such as normalization,
softmax and GELU. A kernel can combine several kinds of work, and matrix
multiplication is not guaranteed to use Tensor Cores: dispatch depends on the
GPU, precision, dimensions, layout and chosen backend. Similarly, avoiding a
slow pointwise kernel or unnecessary memory traffic can matter even when most
FLOPs belong to GEMMs. [NVIDIA's matrix multiplication guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
discusses tiling and shape effects.

## From thread to warp to SM

CUDA exposes a hierarchy:

```text
grid
  └── thread blocks
        └── warps of 32 threads
              └── individual threads
```

Threads in a warp follow a SIMT model: they normally execute the same
instruction on different data. If a branch sends some threads down one path and
the rest down another, the warp executes both paths with inactive lanes masked.
This **warp divergence** lowers useful work per issued instruction.

An SM keeps several warps resident and switches among ready warps when another
is waiting for data. More resident warps can hide latency, but occupancy is
limited by registers, shared memory and the maximum blocks/warps supported by
the SM. Maximum occupancy is not a goal by itself: a kernel using more registers
may do less memory traffic and run faster despite lower occupancy.

### Why shapes matter

Tensor Cores operate on tiles. Matrix dimensions, dtype, alignment and layout
determine whether an implementation can use efficient kernels. Padding a
vocabulary or hidden dimension to a friendly multiple can make a mathematically
larger operation faster.

This is why nanoGPT uses a padded vocabulary size for from-scratch training and
why benchmark results must report shapes, not just parameter counts.

## The memory hierarchy

Memory becomes larger and slower as it moves away from the execution units:

| Level | Scope | Typical role | Optimization question |
| --- | --- | --- | --- |
| Registers | one thread | scalars, accumulators | are spills occurring? |
| Shared memory / L1 | one SM/block | reused tiles | is reuse worth the limited capacity? |
| L2 | whole GPU | shared cache | is the access pattern cache-friendly? |
| HBM | whole GPU | weights, states, activations | are bytes dominating runtime? |
| Host memory | CPU | dataset, offloaded state | are transfers on the critical path? |

For an H100 example in the Scaling Book, HBM bandwidth is about 3.35 TB/s while
peak BF16 Tensor Core throughput is about 990 TFLOP/s. The ratio is the key:
the device can perform hundreds of floating-point operations in the time needed
to stream one byte from HBM.

This imbalance explains why recomputing a value can be faster than storing and
reloading it, and why kernel fusion can help even when it does not reduce FLOPs.

### CPU offload and the host–device link

Offload exchanges GPU-memory capacity for host memory and transfers. For
example, optimizer state and its update may live on the CPU, while the GPU
performs forward and backward. Copy time depends on the amount moved, link
bandwidth, and synchronization; it is separate from HBM traffic. Asynchronous
copies and pinned host memory can enable overlap when dependencies and hardware
permit, but neither guarantees it. In a profiler, look for copies and CPU work
on the critical path, and report peak GPU memory alongside end-to-end tokens/s.
See [Module 6](06-fsdp.md) for offload combined with sharding.

**Attention and saved activations.** A checkpointed attention operation can save
$Q,K,V$ rather than retaining the $N\times N$ intermediate for backward, and
recompute the intermediate when gradients are needed. Optionally offload the
saved inputs to CPU and reload them before recomputation. The extra computation
and host–device transfers may be exposed in the step time; actual saved tensors
and peak memory depend on the implementation. FlashAttention instead tiles
attention and uses recomputation to avoid materializing the full matrix in GPU
memory; CPU offload is not required for that benefit.

For one sequence and head, with equal element sizes, $Q,K,V$ occupy $3Ld_kb$
bytes while the matrix occupies $L^2b$ bytes. The matrix is larger for
$L>3d_k$: with $d_k=64$, this starts above 192 tokens. At $L=1024$ with BF16
($b=2$), the inputs total 0.375 MiB versus 2 MiB for one matrix per head.
These are selected tensors, not a peak-memory estimate; dtypes and saved tensors
can differ in a real kernel.

## Roofline reasoning

For an operation:

\[
I = \frac{\text{FLOPs}}{\text{bytes transferred}}
\]

is its arithmetic intensity. Let \(C\) be peak compute throughput and \(W\)
memory bandwidth. The roofline bound is

\[
P_{\max} = \min(C, I W),
\]

or equivalently

\[
t \ge \max\left(
\frac{\text{FLOPs}}{C},
\frac{\text{bytes}}{W}
\right).
\]

The **ridge point** is \(C/W\). Below it, the operation is bandwidth-bound;
above it, it can be compute-bound.

<figure markdown="span">
  ![A roofline chart with a sloped memory-bandwidth limit and a flat peak-compute limit. Vector addition lies below the ridge in the bandwidth-bound region, while a large tiled matrix multiplication reaches the compute-bound roof.](../assets/figures/gpu-roofline.svg){ loading=lazy }
  <figcaption>The roofline is a prediction: place an operation by arithmetic intensity, then look for the lower ceiling.</figcaption>
</figure>

### Worked example 1 — vector addition

Adding two FP32 vectors and writing the result performs one FLOP while moving
approximately 12 bytes per element: two 4-byte reads and one 4-byte write.

\[
I_{\text{add}} \approx \frac{1}{12}\ \text{FLOP/byte}.
\]

For large vectors, this model suggests bandwidth can dominate. Small vectors
can instead be limited by launch overhead. Fusion may avoid writing an
intermediate to device memory; caches and the chosen kernels affect actual traffic.

### Worked example 2 — matrix multiplication

For \(A\in\mathbb{R}^{M\times K}\) and
\(B\in\mathbb{R}^{K\times N}\), the product performs approximately
\(2MKN\) FLOPs. A simple BF16 byte estimate is

\[
2(MK + KN + MN)\ \text{bytes}.
\]

Larger matrices reuse each loaded element many times, raising arithmetic
intensity. Small batch or token dimensions can leave the same linear layer
bandwidth- or latency-bound even though its large-batch form is compute-bound.

## Why attention needs a specialized implementation

Scaled dot-product attention is
\(P=\operatorname{softmax}(QK^\top/\sqrt{d_h}+\text{mask})\), followed by
\(O=PV\). The two products are matrix operations, but masking and softmax add
pointwise work and row-wise reductions. A straightforward implementation also
materializes the \(T\times T\) scores and probabilities for each head. Writing
and rereading those intermediates costs HBM bandwidth and grows quadratically
with sequence length. Causal masking and limited on-chip storage add further
implementation constraints.

[FlashAttention](https://arxiv.org/abs/2205.14135) computes blocks of scores
and value products while maintaining online softmax statistics and partial
outputs in on-chip memory. It avoids writing the complete score and probability
matrices to HBM; during backward it recomputes tiles using saved statistics. It
computes the same dense attention mathematically, up to floating-point rounding:
the dense arithmetic still grows quadratically with sequence length. Its benefit
comes from reducing data movement, not from making attention sparse or linear.

In the lab, `torch.nn.functional.scaled_dot_product_attention` (SDPA) provides
an API that can select FlashAttention or another backend when eligible. It does
not promise a particular implementation for every GPU, dtype, shape or mask;
inspect the selected path and compare results and timings. See the
[PyTorch SDPA documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

## Training memory: what occupies VRAM?

Training memory has two main components: **model state**, which scales with
parameter count, and **activations**, which grow with batch and sequence length.
Temporary workspaces add to the peak.

### 1. Model state: parameters, gradients, and optimizer states

For mixed-precision AdamW, use this standard budget for $N$ parameters:

| State | Dtype | Bytes/parameter |
| --- | --- | ---: |
| Forward weights | BF16 / FP16 | 2 |
| Gradients | BF16 / FP16 | 2 |
| Master weights | FP32 | 4 |
| Adam first moment | FP32 | 4 |
| Adam second moment | FP32 | 4 |
| **Total** | | **16** |

Thus model state takes approximately **$16N$ bytes**. FP32 gradients raise
this to $18N$ bytes. The exact layout depends on the training implementation,
but $16N$ is a useful planning estimate.

### 2. Dynamic memory: activation accounting per layer

Let $B$ be the **resident micro-batch**, $T$ the sequence length, $d$ the hidden
width, $h$ the number of attention heads, and $L$ the layer count. Consider
ordinary multi-head attention and a two-linear-layer GELU MLP of width $4d$.

Assume standard unfused attention with dropout, 16-bit activations and
one-byte dropout masks. We count the main tensors saved for backward
([Korthikanti et al., Section 4.1](https://arxiv.org/abs/2205.05198)):

| Saved tensors | Bytes per layer |
| --- | ---: |
| Two normalization inputs | $4BTd$ |
| Shared input to Q/K/V projections, counted once | $2BTd$ |
| Q, K, V for attention backward | $6BTd$ |
| Softmax probabilities and attention-dropout output | $4BhT^2$ |
| Attention-dropout mask | $BhT^2$ |
| Output-projection input and residual-dropout mask | $3BTd$ |
| MLP up-projection input | $2BTd$ |
| GELU input and down-projection input | $16BTd$ |
| MLP residual-dropout mask | $BTd$ |
| **Total** | **$34BTd + 5BhT^2$** |

Softmax backward needs the probabilities, not the pre-softmax logits.
Across $L$ layers, the estimate is

\[
M_{\text{activations}} \approx LBT(34d+5hT)\text{ bytes}.
\]

The key distinction is **linear growth with $B$** and a **quadratic attention
term in $T$**. The vocabulary head and temporary workspaces add to this budget.

#### The impact of memory optimizations

- **FlashAttention:** avoids saving the full attention matrix, removing the
  quadratic storage term. Attention activation memory becomes $O(BTd)$ per
  layer, while attention compute remains quadratic in $T$. PyTorch SDPA can
  select this efficient backend on supported hardware.
- **Activation checkpointing:** saves block inputs and recomputes intermediate
  activations during backward. Checkpointing every block reduces saved inputs
  to roughly $2LBTd$ bytes, plus the current block's working memory. One extra
  forward pass changes block compute from about $3F$ to $4F$: roughly **33%
  more FLOPs**.
- **Gradient accumulation:** uses smaller micro-batches to reach the same
  effective batch size. Activation memory follows the resident micro-batch.

Measure both **allocated** memory (live tensors) and **reserved** memory
(allocator-managed storage) after warm-up. The measured peak includes temporary
workspaces as well as the tensors in these estimates.

---

## First-principles FLOPs accounting: Attention vs. MLP

To compute hardware utilization (MFU) and arithmetic intensity, we must count the theoretical floating-point operations required by the model.

In digital arithmetic, a matrix multiply-accumulate (MAC) operation involves **1 multiplication and 1 addition**, which counts as **$2\text{ FLOPs}$**. Therefore, multiplying a matrix of shape $(M \times K)$ by a matrix of shape $(K \times N)$ requires:

\[
\text{FLOPs}_{\text{GEMM}} = 2 M K N.
\]

### 1. Forward pass FLOPs per token

Count the dense matrix products for a training sequence and divide by its
$BT$ tokens. Assume ordinary multi-head attention with $d_h=d/h$, a $4d$ GELU
MLP, and count full $T\times T$ attention products. We focus on matrix
multiplications, which dominate the FLOPs:

| Component | Matrix products | Forward FLOPs/token |
| --- | --- | ---: |
| Q/K/V projections | Three $d\times d$ projections | $6d^2$ |
| Attention output projection | One $d\times d$ projection | $2d^2$ |
| Attention scores | $QK^\top$ across all heads | $2Td$ |
| Attention over values | $SV$ across all heads | $2Td$ |
| GELU MLP, width $4d$ | $d\to4d\to d$ | $16d^2$ |
| SwiGLU MLP, width $8d/3$ | Two up projections, one down projection | $16d^2$ |

For example, the batched Q/K/V projections cost $3(2BTd^2)$ FLOPs.
The score product has shapes $(T\times d_h)(d_h\times T)$ per head;
over $B$ sequences and $h$ heads it costs $2BhT^2d_h=2BT^2d$ FLOPs.
Divide each by $BT$ to obtain its table entry.

The SwiGLU row is an **alternative** to the GELU row, with
$d_{\mathrm{ff}}=8d/3$ before rounding: $6d\,d_{\mathrm{ff}}=16d^2$ FLOPs.
The main result is:

- **Attention linear projections:** $6d^2 + 2d^2 = 8d^2$ FLOPs/token.
- **MLP linear projections:** $8d^2 + 8d^2 = 16d^2$ FLOPs/token.
- **The $1/3$ to $2/3$ rule:** In this block, the feed-forward MLP performs **twice as many FLOPs** as the attention projections ($16d^2$ vs $8d^2$), and holds twice as many parameters ($8d^2$ vs $4d^2$).
- **The quadratic term:** The two attention products contribute $4Td$
  FLOPs/token, or $4T^2d$ per sequence. Their ratio to the projection cost is
  $T/(6d)$; they exceed all block projections when $T>6d$ under this convention.

### 2. The $2\times$ backward multiplier

For every linear layer $Y = X W$ in the forward pass, the backward pass must compute two distinct matrix derivatives via the chain rule:

**Gradient with respect to input activations:**

\[
\frac{\partial \mathcal{L}}{\partial X}
= \frac{\partial \mathcal{L}}{\partial Y} W^\top
\quad \implies 2MKN\text{ FLOPs}.
\]

**Gradient with respect to layer weights:**

\[
\frac{\partial \mathcal{L}}{\partial W}
= X^\top \frac{\partial \mathcal{L}}{\partial Y}
\quad \implies 2MKN\text{ FLOPs}.
\]

Backward therefore costs roughly twice as much as forward:

\[
\text{Backward FLOPs} \approx 2 \times \text{Forward FLOPs},
\]

Adding forward and backward gives:

\[
\text{Total Training FLOPs} = \text{Forward} + \text{Backward} \approx 3 \times \text{Forward FLOPs}.
\]

Embedding lookup is a gather, not a dense matrix multiply; token IDs do not
receive gradients. A dense vocabulary head still incurs its full matrix cost
when its weights are tied to the embedding table.

### 3. Total training FLOPs per token

Multiplying the forward matrix operations by 3 yields:

- **Transformer blocks (per layer):** $3 \times (8d^2 + 16d^2) = 72 d^2$ FLOPs/token.
- **Attention matrix products (per layer):** $3 \times (4 T d) = 12 T d$ FLOPs/token.
- **Output vocabulary head:** $3 \times (2 d V) = 6 d V$ FLOPs/token.

Summing across all $L$ layers:

\[
\text{Training FLOPs per token} \approx 72 L d^2 + 12 L d T + 6 d V.
\]

Ignoring biases and normalization parameters, the block matrix parameter count is $N_{\text{blocks}} = 12Ld^2$. The block projection term simplifies to:

\[
72 L d^2 = 6 N_{\text{blocks}}.
\]

Define $N_{\text{matrix}}=12Ld^2+dV$: the block matrix weights plus the
vocabulary-head matrix, counted once even when tied. Then

\[
\text{FLOPs/token} \approx 6N_{\text{matrix}} + 12LdT.
\]

This matches the companion's
[`compute_flops_per_token`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/profile_utils.py).
Ignoring the smaller bias, normalization and positional-embedding terms,
$N_{\text{matrix}}$ approximates the parameter count of a model with tied
embeddings.

For $D$ training tokens, the familiar rule of thumb is **$6ND$ FLOPs**, with
$12LdTD$ added for the attention matrix products. For example, $L=12$, $d=768$,
$V=50{,}304$, $T=1024$ gives about **855 million FLOPs/token**, including
**113 million (13%)** for attention products.

---

## FLOPs, tokens/s, MFU and time-to-train

Useful metrics answer different questions:

| Metric | What it answers |
| :--- | :--- |
| **tokens/s** | How fast does this exact workload progress? |
| **step time** | Where did latency change? |
| **peak memory** | What larger workload or micro-batch might fit? |
| **achieved FLOP/s** | How much arithmetic ran per second? |
| **MFU (Model FLOPs Utilization)** | What fraction of theoretical peak performed model-required FLOPs? |
| **HFU (Hardware FLOPs Utilization)** | What fraction includes executed arithmetic such as recomputation? |
| **time-to-target** | Did the end-to-end experiment improve? |

Compute MFU as model FLOPs/token times tokens/s divided by the GPU's peak
FLOP/s at the training precision. MFU excludes checkpoint recomputation;
HFU includes that extra arithmetic.

An optimization can raise hardware utilization while slowing time-to-target, for example by adding unnecessary recomputation. Tokens/s and validation behavior remain the primary outcomes for this course.

## Read a profiler trace as a causal story

Profile after warmup, over multiple representative steps, with the same batch
and synchronization points used in the benchmark. The companion collects a
separate trace after its timing samples, so profiling overhead is not included
in the reported timings.

Look for:

1. **CPU gaps:** the GPU waits because data loading or Python launch is late;
2. **many tiny kernels:** launch overhead or unfused pointwise work dominates;
3. **large GEMMs with low utilization:** shapes or precision miss an efficient
   kernel path;
4. **explicit device copies:** input transfer or accidental host round trips;
5. **attention matrix kernels:** a non-fused attention path may materialize
   \(T\times T\) scores;
6. **allocator churn:** repeated allocation or shape changes prevent reuse;
7. **synchronization:** logging `.item()`, timing without CUDA events, or memory
   queries can serialize asynchronous work.

Use top-down questions:

- Is the GPU busy?
- If busy, is it doing useful kernels?
- If useful, are those kernels compute-, bandwidth- or latency-bound?
- Which source operation launched the dominant kernel?

## Optimization ladder

Change one mechanism at a time and re-measure correctness, throughput and
memory.

### 1. Input pipeline

- prefetch and use pinned host memory when transfers warrant it;
- avoid tokenization or decompression on the critical path;
- check that batches are ready before the GPU asks for them.

### 2. Precision

A tensor stored in BF16 takes half the bytes of FP32 and retains FP32-like
exponent range. BF16 matrix products can use fast Tensor Core paths, but
autocast does not halve every tensor or convert persistent FP32 parameters.
FP16 may need loss scaling. Precision changes require a loss/gradient
comparison, not only a speed measurement.

### 3. Scaled-dot-product attention

Efficient SDPA/FlashAttention tiles Q, K and V through on-chip memory and avoids
writing the full attention matrix to HBM. It is exact attention with a different
IO schedule, not sparse or approximate attention.

### 4. Fused operations

Consider an illustrative pointwise chain on equal-sized tensors:

```python
u = x + b
v = torch.relu(u)
y = v + residual
```

A possible eager implementation launches a separate kernel for each operation,
writing `u` and `v` and reading them back. A fused kernel can read `x`, `b`,
and `residual`, compute the chain while intermediates stay in registers, and
write `y`. In an idealized element-transfer count that is eight transfers
versus four. This is an example, not a claim about the exact generated kernels
or HBM bytes: caches, broadcasting, kernel choice, and implementation matter.

The relevance to our GPT is the collection of elementwise operations around
normalization, activations, and residual additions. Fusion can reduce launches
and intermediate traffic; it does not mean the whole Transformer becomes a
single kernel. Efficient attention kernels are another example of combining work
and reducing memory traffic, but SDPA is a separate mechanism from compilation.

### 5. Compilation with `torch.compile`

```python
compiled_model = torch.compile(model)
logits, loss, _ = compiled_model(inputs, targets=targets)
loss.backward()
```

On execution, TorchDynamo captures supported operations into graphs guarded by
assumptions such as shapes. For training, PyTorch captures forward and backward
regions; the default TorchInductor backend generates optimized code that may
fuse operations or use optimized library kernels. The first calls can include
substantial compilation cost. Later calls reuse compiled code when their guards
hold. Unsupported code can cause a **graph break**: the captured region ends,
that code runs eagerly, and capture can resume afterwards. Changing shapes or
other guarded assumptions can lead to recompilation.

The potential benefits are fewer launches, less Python overhead, and lower
intermediate memory traffic. None is guaranteed: existing library kernels may
already dominate, compile overhead can outweigh savings, and graph breaks can
limit optimization. Compilation does not automatically choose FlashAttention or
replace a correctness check. The actual SDPA backend depends on the inputs,
implementation, and hardware. See the official [compiler documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile)
and [graph-break guide](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html).

In the companion lab, compare **SDPA FP32** with **compiled SDPA FP32** using
identical batch shapes and configuration. Warm up compilation before steady-state
timing, synchronize at boundaries, inspect the separate profiler traces, and
compare fixed-batch loss and gradients. The benchmark reports timing and memory;
it does not perform numerical parity checks or report compile time automatically. The pinned PyTorch 2.9.1 rejects `torch.compile` on Python 3.14; use the companion’s CPython 3.12 environment for this exercise.

### 6. Activation checkpointing

Checkpointing saves memory by discarding selected activations and recomputing
them during backward. It is valuable only if the released memory enables a
better micro-batch or prevents an OOM at acceptable time cost.

## Benchmark protocol

For every variant:

1. use the same model, batch tensors, precision target and optimizer semantics;
2. warm up until compilation and caches stabilize;
3. time enough steps to report a distribution, not one value;
4. synchronize correctly around timing boundaries;
5. record median step time, tokens/s and peak allocated/reserved memory;
6. compare loss and gradients on a fixed batch within declared tolerances;
7. save the profiler trace supporting the bottleneck explanation.

Report environment details: GPU model, driver/runtime, framework version,
power mode and whether other processes shared the device.

## In-class investigation

### Part A — roofline worksheet

For vector addition, LayerNorm and one course-model matrix multiplication:

1. estimate FLOPs and HBM bytes;
2. compute arithmetic intensity;
3. find the target GPU's \(C/W\) ridge point;
4. predict the bound;
5. name one optimization consistent with that prediction.

### Part B — memory prediction

Before running, estimate persistent model-state memory and activation growth as
micro-batch and sequence length change. Then compare with measured allocated
and reserved peaks. Explain the largest discrepancy.

### Part C — one-change ablation

Start with eager FP32 or the declared baseline. Add one of BF16, SDPA,
compilation or checkpointing. Produce a table with:

| Variant | tokens/s | median step ms | peak allocated | peak reserved | loss delta |
| --- | ---: | ---: | ---: | ---: | ---: |

The conclusion must name the bottleneck the change targeted and whether the
trace supports the explanation.

## Exit ticket

1. Why can a GPU with enormous peak TFLOP/s still be slow on vector addition?
2. What is the difference between bandwidth-bound and latency-bound?
3. Why can lower occupancy still produce a faster kernel?
4. How can fusion improve runtime without reducing mathematical FLOPs?
5. Why is peak reserved memory different from live tensor memory?

## Expected output

A measurement-first performance report containing a hardware-aware prediction,
a memory budget, baseline trace, one-change ablations and a justified final
single-GPU configuration.

## References

- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/)
  — primary conceptual spine for GPU execution, memory and rooflines;
- [PyTorch `torch.compile` reference](https://docs.pytorch.org/docs/stable/generated/torch.compile)
  — compilation options and backend behavior;
- [PyTorch compiler graph breaks](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.graph_breaks_index.html)
  — how unsupported code divides captured graphs;
- [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
  — trace collection and interpretation;
- [PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
  — practical tuning mechanisms;
- [FlashAttention](https://arxiv.org/abs/2205.14135)
  — IO-aware exact attention and tiling analysis.

---

[:material-file-pdf-box: View Lecture Slides (PDF)](../slides/04-single-gpu.pdf){ .md-button target="_blank" }
[:material-code-tags: Practical Companion Guide](../companion/04-single-gpu.md){ .md-button .md-button--primary }

