# Module 8 — Context, pipeline, and expert parallelism

## Purpose

Choose an optimal multidimensional parallelization strategy from model size,
context length, batch target, expert sparsity, and physical cluster topology.

Context parallelism (CP), pipeline parallelism (PP), expert parallelism (EP),
data parallelism (DP/FSDP), and tensor parallelism (TP) each target different
scaling bottlenecks—parameter capacity, activation memory, context length,
or compute throughput. Crucially, they exhibit wildly different communication
volumes, collective patterns, and network latency sensitivities.

The main reference is the
[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook),
extended here with expert parallelism and the physical network mapping rules
governing multi-thousand GPU clusters.

---

## Learning goals

By the end of the module, students should be able to:

- distinguish sequence parallelism from context parallelism;
- explain why attention makes sequence sharding non-local and trace ring attention with online softmax;
- explain causal load imbalance and zig-zag sequence placement;
- derive pipeline bubble ratios and activation-memory trade-offs for AFAB and 1F1B;
- trace Expert Parallelism (EP) token routing and All-to-All communication volumes;
- derive the complete cluster sizing identity: $G = P \times D \times F \times E_P \times T_P \times C$;
- defend the **outer-to-inner hierarchy** ($\text{PP} \to \text{DP} \to \text{FSDP} \to \text{EP} \to \text{TP}$) grounded in communicated byte volume, frequency, and network fabric domains.

---

## Two different sequence-axis ideas

The terminology is overloaded, so this course uses a strict distinction:

| Technique | Where sequence is sharded | Main purpose |
| :--- | :--- | :--- |
| **Sequence parallelism (SP)** | Only operations outside TP regions, such as normalization and residual/dropout | Remove replicated activation residency created by TP |
| **Context parallelism (CP)** | Through the full Transformer layer, including attention | Make very long sequences fit in VRAM and divide their quadratic compute |

SP is a direct companion to tensor parallelism within a single node. CP is an
orthogonal parallel axis used when the sequence length $T$ (e.g. 32k–1M tokens)
exceeds single-device memory.

---

## Ring attention for context parallelism

LayerNorm, RMSNorm, and MLP projections act independently on each token. If
a sequence is partitioned across $C$ context-parallel ranks, each rank holds $T/C$
tokens and evaluates these operations locally without communication.

Attention is different: token $i$ must attend to all preceding keys and values
$j \le i$. After sequence sharding, each rank holds local queries $Q_r$ but only
a subset of keys and values $(K_r, V_r)$.

In **ring attention**, ranks form a logical communication ring:

1. Each rank issues an asynchronous non-blocking send of its local $(K, V)$ chunk
   to rank $r+1$, and receives the chunk from rank $r-1$.
2. Concurrently, the rank computes scaled dot-product attention between its local
   queries $Q_r$ and its current $(K, V)$ chunk.
3. The running attention output and normalization factors are updated on the fly
   using the online softmax formula:
   
   $$m_{\text{new}} = \max(m_{\text{prev}}, m_{\text{curr}}), \quad l_{\text{new}} = e^{m_{\text{prev}} - m_{\text{new}}} l_{\text{prev}} + e^{m_{\text{curr}} - m_{\text{new}}} l_{\text{curr}}.$$
4. Repeat for $C-1$ shifts until all KV blocks have circulated.

### Causal load imbalance & zig-zag placement

In causal language modeling, tokens only attend to past positions. Contiguous
chunking ($[0, T/C)$ on rank 0, $[T/C, 2T/C)$ on rank 1, etc.) leaves rank 0
computing on only 1 chunk while rank $C-1$ must attend to all $C$ chunks—wasting
$50\%$ of cluster compute.

**Zig-zag placement** distributes paired non-contiguous sub-chunks across ranks
(e.g., assigning chunk $i$ and chunk $2C - 1 - i$ to the same rank), balancing
the causal attention mask exactly across all ranks.

---

## Pipeline parallelism (PP)

Pipeline parallelism partitions the model *by depth*: layers $1 \dots L$ are
divided across $P$ sequential stages.

### All-forward-all-backward (AFAB / GPipe)

All $m$ micro-batches execute forward through stage $1 \dots P$, then all $m$
micro-batches execute backward in reverse order.

- **Bubble fraction:** $F_{\text{bubble}} = \frac{P - 1}{m + P - 1}$. To keep the bubble small ($< 10\%$), one needs $m \ge 4P$.
- **Activation residency:** Stage 1 must hold activations for all $m$ micro-batches
  in memory until backward begins, causing severe activation memory spikes.

### One-forward-one-backward (1F1B)

After an initial warmup of $P$ micro-batches, each stage alternates executing
one forward micro-batch and one backward micro-batch:

- Frees activations immediately after each backward micro-step.
- Caps peak activation residency at $P$ micro-batches (the pipeline depth)
  instead of $m$, allowing arbitrarily large accumulation steps without OOM.
- Preserves the same steady-state bubble fraction as AFAB, but dramatically
  lowers peak VRAM.

### Interleaved and zero-bubble schedules

Interleaved 1F1B assigns multiple non-consecutive virtual stages to each
physical device (e.g. device 0 runs layers 1–4 and 17–20). This shrinks the
idle bubble by a factor of $v$ at the cost of $v \times$ more point-to-point
communication across stage boundaries.

---

## Expert Parallelism (EP)

In Mixture of Experts (MoE) models, dense MLP layers are replaced with $E$
sparsely activated experts. Each token is dynamically routed by a gating network
to its top-$k$ experts ($k \ll E$).

**Expert Parallelism (EP)** shards the $E$ expert networks across $E_P$ devices:
each rank hosts $E / E_P$ experts.

### Token routing and All-to-All communication

Because tokens in a batch on rank $r$ may be assigned to experts located on any
other rank, EP requires an **All-to-All** collective:

```
[Tokens on Rank 0..R-1] ──(Gate & Dispatch)──> [All-to-All] ──> [Local Experts compute]
                                                                     │
[Returned Hidden States] <──(Combine Weights)── <─── [All-to-All] ───┘
```

1. **Dispatch All-to-All:** Every rank packs tokens destined for remote experts
   and executes `all_to_all_single`, receiving tokens routed to its local experts.
2. **Local Computation:** Each rank processes incoming tokens through its assigned
   expert MLPs.
3. **Combine All-to-All:** Hidden state activations are transmitted back to their
   originating ranks via a second `all_to_all_single` collective and linearly
   weighted by the gate softmax probabilities.

### EP Communication volume

Per MoE layer, the data transmitted by each rank in forward is:

$$\text{Bytes}_{\text{EP, fwd}} = \left( \frac{B_\mu \cdot T}{C \cdot T_P} \right) \times k \times d \times \text{sizeof(dtype)}.$$

In the backward pass, an identical volume of activation gradients is exchanged.
Because All-to-All generates $E_P \times (E_P - 1)$ simultaneous point-to-point flows,
EP requires high bisection bandwidth to avoid network congestion and incast packet drops.

---

## The Outer-to-Inner Hierarchy: Grounded in Communicated Bytes

When mapping a 5D parallel execution plan ($P \times D \times F \times E_P \times T_P \times C$)
onto real cluster hardware, we must order parallel axes from **outermost**
(slowest, high-latency network links) to **innermost** (fastest, lowest-latency interconnect).

The order is:

$$\Large \mathbf{PP} \;\longrightarrow\; \mathbf{DP} \;\longrightarrow\; \mathbf{FSDP} \;\longrightarrow\; \mathbf{EP} \;\longrightarrow\; \mathbf{TP}$$

```
[ Outer Network: Cross-DC / Cross-Pod / Cross-Rack (100–400 Gbps, High Latency) ]
   └── PP: Point-to-point boundary activations only (B_μ * T * d bytes)
         └── DP: Gradient All-Reduce once per step (2 * params, fully overlapped)
               └── FSDP: All-Gather + Reduce-Scatter per layer (3 * params total)
                     └── EP: All-to-All token exchange per MoE layer
                           └── TP: Blocking All-Reduce inside layer inner loop (NVLink only)
[ Inner Network: Intra-Node NVLink / NVSwitch (900–1800 GB/s, Sub-Microsecond Latency) ]
```

### Comprehensive Systems & Communication Comparison

| Parallel Axis | Collective Pattern | Communicated Bytes per Step | Frequency & Trigger | Latency Sensitivity & Critical Path | Physical Domain Placement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PP** (Pipeline) | Point-to-Point (`send` / `recv`) | $\mathcal{O}(B_\mu \cdot T \cdot d)$ per micro-batch | Once per boundary layer per micro-batch | **Lowest.** P2P transfer, easily hidden behind 1F1B compute. Independent of parameter count. | **Outermost:** Inter-pod, cross-rack, or lower-bandwidth switches. |
| **DP** (Data Parallel) | All-Reduce | $2 \times \text{parameters}$ (gradients) | **Once per step** (at end of backward) | **Low.** Fully overlapped with backward pass by streaming gradient buckets from layer $L$ to 1. | **Outer:** Inter-node across spine-leaf fabric. |
| **FSDP / ZeRO-3** | All-Gather + Reduce-Scatter | $3 \times \text{parameters}$ ($1\times$ fwd weights, $1\times$ bwd weights, $1\times$ bwd grads) | **Every layer**, twice per step (fwd & bwd) | **Medium-High.** Bursty, high byte volume. Prefetching helps, but requires sustained high bisection bandwidth. | **Intermediate:** Intra-pod InfiniBand/RoCE fat-tree network. |
| **EP** (Expert Parallel) | All-to-All (`all_to_all_single`) | $\mathcal{O}(\text{tokens} \cdot k \cdot d)$ per MoE layer | **Every MoE layer**, fwd & bwd | **High.** All ranks exchange tokens simultaneously; highly sensitive to bisection bandwidth and incast. | **Intermediate:** Intra-pod non-blocking fabric. |
| **TP** (Tensor Parallel) | All-Reduce | $4L \times 2 \times (B_\mu \cdot T \cdot d)$ | **Every layer**, $2\times$ in fwd, $2\times$ in bwd | **Ultra-High.** Strictly blocking on the inner critical path. Zero overlap possible without specialized kernels. | **Innermost:** Intra-node NVLink / NVSwitch only ($T_P \le 8$). |

### Why this hierarchy is physically necessary:

1. **Why PP is Outermost:**
   PP communicates only activations at stage boundaries. The number of bytes is
   $B_\mu \cdot T \cdot d$, which is **completely independent of model parameter count $P_{\text{model}}$**.
   For a 70B model, transferring an activation chunk requires only tens of megabytes,
   compared to hundreds of gigabytes of parameter state. Because communication is
   point-to-point between adjacent ranks $(r, r+1)$ and hidden by the 1F1B schedule,
   PP is the only parallelism scheme that cleanly tolerates slow inter-rack or cross-datacenter links.

2. **Why DP is Outside FSDP:**
   Standard DDP communicates $2 \times \text{parameters}$ in gradients, but does
   so **only once per optimizer step**. PyTorch DDP overlaps this communication
   asynchronously with backward computation: while layer $l$ computes gradients,
   layers $l+1 \dots L$ are already all-reducing. FSDP, in contrast, must transmit
   $3 \times \text{parameters}$ **on demand layer-by-layer** (gathering weights right
   before execution). FSDP therefore requires substantially higher bandwidth and lower
   latency than pure DDP.

3. **Where EP Fits:**
   In an MoE model, EP replaces dense parameters with routed tokens. While token
   volume is generally lower than gathering full model weights, the **All-to-All**
   communication pattern creates dense, all-to-all cross-chatter that saturates
   network switches. EP must therefore be hosted within non-blocking network domains
   (e.g., within an InfiniBand island or rail-optimized pod).

4. **Why TP is Strictly Innermost:**
   Megatron-style TP splits weight matrices along rows and columns. In every single
   Transformer block, it executes two All-Reduces in forward (one after attention
   projection, one after MLP down-projection) and two in backward.
   Crucially, **computation cannot proceed until the All-Reduce finishes**.
   Running TP over a 400 Gbps network link degrades Model FLOPs Utilization (MFU)
   from $50\%$ to $< 15\%$ due to collective latency overhead. TP must stay on
   NVLink ($900\text{--}1800\text{ GB/s}$), limiting $T_P \le 8$.

---

## Compose the parallel dimensions

For a total cluster of $G$ GPUs, the parallel dimensions satisfy:

$$G = P \times D \times F \times E_P \times T_P \times C,$$

where:
- $P$: Pipeline-parallel stages (cross-rack / inter-pod);
- $D$: Replicated data-parallel groups (cross-pod / inter-node);
- $F$: Fully sharded data-parallel degree (intra-pod);
- $E_P$: Expert-parallel degree (intra-pod non-blocking fabric);
- $T_P$: Tensor-parallel degree (intra-node NVLink, $\le 8$);
- $C$: Context-parallel degree (ring attention, intra-node or intra-pod).

### Practical cluster selection recipe

1. **Fit one layer and preserve GEMM efficiency:** Choose intra-node $T_P \in \{1, 2, 4, 8\}$
   such that local matrix multiplication dimensions remain large enough to saturate Tensor Cores.
2. **Fit context length:** If $T > 32\text{k}$ tokens, add Context Parallelism ($C$)
   with ring attention.
3. **Handle MoE experts:** If using MoE, allocate $E_P$ such that $E / E_P$ experts
   fit in local VRAM, placing EP within the high-speed pod fabric.
4. **Fit total model parameters across nodes:** If the model does not fit in a single
   node, apply Pipeline Parallelism ($P$) across nodes or high-bandwidth FSDP ($F$).
5. **Scale to global batch:** Use Data Parallelism ($D$) across remaining devices
   until reaching the target global batch size:
   
   $$\text{Global tokens / step} = B_\mu \times T \times A \times D \times F.$$

---

## Strategy case studies

### Case A — Model fits in 1 GPU, short context, 64 GPUs available
- **Choice:** Pure DDP ($D=64$, $P=1, T_P=1, C=1$).
- **Rationale:** No memory bottleneck. DDP adds zero activation recomputation or pipeline bubbles and overlaps communication completely.

### Case B — 70B dense model on 64 $\times$ 8-GPU nodes (512 H100s)
- **Choice:** $T_P=8$ (intra-node NVLink), $P=4$ (across nodes), $F=16$ (intra-pod FSDP).
- **Rationale:** $T_P=8$ keeps latency-sensitive All-Reduces on NVLink. $P=4$ divides depth into manageable chunks. FSDP shards remaining state across 16 ranks with enough compute to hide weight gathers.

### Case C — 1M context on 32 GPUs
- **Choice:** $C=8$, $T_P=4$, $D=1$.
- **Rationale:** KV cache dominates memory ($2 \times L \times h_{kv} \times d_h \times T$). Ring attention shards KV activations across 8 ranks, while $T_P=4$ shards wide projections.

---

## In-class investigation

### Part A — Hierarchical communication calculation
For an 8-node cluster ($64 \times \text{H100}$ GPUs with 900 GB/s NVLink intra-node and 400 Gbps InfiniBand inter-node):
1. Compute the time spent in collective communication for $T_P=8$ intra-node vs $T_P=8$ inter-node.
2. Calculate the All-to-All byte volume for an MoE layer with $B_\mu=2, T=4096, d=4096, k=2, E_P=8$.
3. Prove why placing EP across nodes within a switch is viable, but placing TP across nodes collapses throughput.

### Part B — Ring-attention trace
For 4 ranks and 8 sequence chunks with causal masking:
1. Trace the zig-zag assignment to show equal computation across all 4 ranks.
2. Calculate the online softmax update factors between successive KV chunk shifts.

### Part C — Pipeline bubble derivation
Draw 1F1B timelines for $P=4, m=16$:
1. Calculate the exact bubble fraction $F_{\text{bubble}}$.
2. Compare peak activation memory between GPipe (AFAB) and 1F1B.

---

## Exit ticket

1. Why does pipeline communication volume not scale with parameter count $P_{\text{model}}$?
2. What makes tensor parallelism strictly intolerant of inter-node network latency?
3. How does zig-zag chunk placement eliminate the causal imbalance in ring attention?
4. In an MoE architecture, what collective does expert parallelism rely on, and why does it stress network bisection bandwidth?
5. Write the full outer-to-inner parallelism hierarchy and explain why FSDP is placed inside DP.

---

## References

- [The Ultra-Scale Playbook — Context Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook#context-parallelism)
  — ring attention, causal balancing, and scaling results;
- [The Ultra-Scale Playbook — Pipeline Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook#pipeline-parallelism)
  — AFAB, 1F1B, interleaving, and bubble analysis;
- [The Ultra-Scale Playbook — Finding the Best Training Configuration](https://huggingface.co/spaces/nanotron/ultrascale-playbook#finding-the-best-training-configuration)
  — multidimensional selection workflow;
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  — DualPipe overlapping schedule and multi-node EP/TP architecture;
- [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
  — foundational tensor and pipeline parallelism derivations.

---

[:material-file-pdf-box: View Lecture Slides (PDF)](../slides/08-context-pipeline.pdf){ .md-button target="_blank" }
[:material-code-tags: Practical Companion Guide](../companion/08-context-pipeline.md){ .md-button .md-button--primary }
