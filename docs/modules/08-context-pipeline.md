# Module 8 — Context, pipeline, and expert parallelism

## Purpose

Propose and test a feasible multidimensional parallelization strategy from model size,
context length, batch target, expert sparsity, and physical cluster topology.

Context parallelism (CP), pipeline parallelism (PP), expert parallelism (EP),
data parallelism (DP/FSDP), and tensor parallelism (TP) each target different
scaling bottlenecks—parameter capacity, activation memory, context length,
or compute throughput. Crucially, they exhibit wildly different communication
volumes, collective patterns, and network latency sensitivities.

The main reference is the
[Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook),
extended here with expert parallelism and explicit assumptions for mapping process groups onto hardware.

---

## Learning goals

By the end of the module, students should be able to:

- distinguish sequence parallelism from context parallelism;
- explain why attention makes sequence sharding non-local and trace ring attention with online softmax;
- explain causal load imbalance and zig-zag sequence placement;
- derive pipeline bubble ratios and activation-memory trade-offs for AFAB and 1F1B;
- trace Expert Parallelism (EP) token routing and All-to-All communication volumes;
- count devices from independent rank axes and distinguish subdivisions of DP;
- defend a network placement using payload, frequency, overlap, and measured bandwidth.

---

## Two different sequence-axis ideas

The terminology is overloaded, so this course uses a strict distinction:

| Technique | Where sequence is sharded | Main purpose |
| :--- | :--- | :--- |
| **Sequence parallelism (SP)** | Only operations outside TP regions, such as normalization and residual/dropout | Remove replicated activation residency created by TP |
| **Context parallelism (CP)** | Through the full Transformer layer, including attention | Make very long sequences fit in VRAM and divide their quadratic compute |

SP is a direct companion to tensor parallelism within a TP group. CP is an
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
processing fewer valid key blocks than rank $C-1$. The imbalance depends on
which masked blocks the kernel skips and whether transfers overlap compute.

**Zig-zag placement** distributes paired non-contiguous sub-chunks across ranks
(e.g., assigning chunk $i$ and chunk $2C - 1 - i$ to the same rank), balancing ideal valid-pair counts for equally sized chunks. Runtime also
depends on kernel efficiency and communication.

---

## Pipeline parallelism (PP)

Pipeline parallelism partitions the model *by depth*: layers $1 \dots L$ are
divided across $P$ sequential stages.

### All-forward-all-backward (AFAB / GPipe)

All $m$ micro-batches execute forward through stage $1 \dots P$, then all $m$
micro-batches execute backward in reverse order.

- **Bubble fraction:** $F_{\text{bubble}} = \frac{P - 1}{m + P - 1}$. Under balanced stage times and negligible communication, $F_{\text{bubble}}<0.1$
  requires $m>9(P-1)$. At $P=4,m=16$, the fraction is $3/19\approx15.8\%$.
- **Activation residency:** Stage 1 must hold activations for all $m$ micro-batches
  in memory until backward begins, causing severe activation memory spikes.

### One-forward-one-backward (1F1B)

After a stage-dependent warmup, each stage alternates executing
one forward micro-batch and one backward micro-batch:

- Frees activations immediately after each backward micro-step.
- Reduces the number of outstanding micro-batch activations to roughly the
  pipeline depth in the standard non-interleaved schedule. It does not guarantee
  that a model fits: weights, buffers, and activation sizes still matter.
- Has the same ideal bubble fraction as AFAB under the balanced schedule model,
  while reducing activation residency.

### Interleaved and zero-bubble schedules

Interleaved 1F1B assigns multiple non-consecutive virtual stages to each
physical device (e.g. device 0 runs layers 1–4 and 17–20). In an ideal balanced model, $v$ virtual stages can reduce the bubble term
roughly by $v$, while introducing more boundary transfers. Actual benefit depends
on schedule constraints, micro-batch count, stage balance, and communication.

### Interactive pipeline schedule simulator

To explore how these pipeline schedules behave under different numbers of stages ($p$) and micro-batches ($m$), launch the dedicated interactive simulator:

!!! tip "Interactive Visualization Lab"
    Experiment with Naive, GPipe / AFAB, 1F1B, and Interleaved 1F1B schedules in a visual step-by-step simulator. Distinguish idle/useful ratio $r=(p-1)/m$ from idle/total fraction $r/(1+r)$, track activation memory accumulation across GPUs, and watch peer-to-peer tensor transfers along stage boundaries.

    [:material-play-circle-outline: Launch Pipeline Parallelism Simulator](../demos/pipeline-parallelism.html){ .md-button .md-button--primary target="_blank" }

---

## Expert Parallelism (EP)

In Mixture of Experts (MoE) models, dense MLP layers are replaced with $E$
sparsely activated experts. Each token is dynamically routed by a gating network
to its top-$k$ experts ($k \ll E$).

**Expert Parallelism (EP)** shards the $E$ expert networks across $E_P$ devices:
each rank hosts $E / E_P$ experts. The router scores experts, dispatches each
token to the selected experts, and combines their outputs. Top-$k$ controls
active expert work, not total stored parameters. Uneven routing can overload one
rank; capacity limits or balancing policies change both runtime and model behavior.
Shared attention weights are not divided by EP merely because experts are.

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

### EP communication volume

Let $n$ be the number of source tokens owned by a rank, $k$ the selected experts
per token, $d$ the hidden width, and $b$ the bytes per activation element.
Dispatch plus combine has payload approximately $2n k d b$ in forward; backward
adds approximately the same again. Local routes do not use the network. With
uniform routing across $E_P$ ranks, a simple estimate multiplies by
$1-1/E_P$. Declare the token layout before setting $n$: TP/SP can change whether
tokens are replicated or partitioned. Metadata, padding, skew, and protocol
traffic are additional costs.

## Map communication to the network

There is no universally optimal nesting of PP, DP, FSDP, EP, TP, and CP.
Place groups according to the exposed communication on their critical path.

| Group | Main transfer | Placement question |
| --- | --- | --- |
| PP | Boundary activations and their gradients per micro-batch | Can transfer finish before the next stage needs it? |
| DP | Gradient buckets, normally once per optimizer update with accumulation | How much reduction remains after backward finishes? |
| FSDP | Parameter gathers and gradient reduce-scatter | Do gathering and prefetch fit the bandwidth and memory budget? |
| TP | Frequent activation collectives inside each layer | Are local matrix products large enough to amortize collective latency? |
| CP | KV blocks and attention-gradient exchanges | Can attention compute hide transfers at this context length? |
| EP | Routed-token dispatch and combine | Does the fabric support the routing pattern without one rank becoming a straggler? |

TP often benefits from the fastest local fabric; that is a starting hypothesis,
not an NVLink-only restriction or a universal eight-device limit. PP can reduce
the volume crossing a slower link, but its boundaries can still stall. DDP
bucket overlap hides some communication only when sufficient compute is available.
Measure achieved bandwidth and exposed time rather than dividing by an advertised
aggregate link rate and declaring the result optimal.

For an ideal ring all-reduce of payload $S$ bytes over $R$ ranks, bytes **sent per
rank** are $2(R-1)S/R$. This excludes received bytes and latency. A parameter
count becomes a payload only after specifying precision and sharding. For FSDP,
also specify how often weights are resharded and whether gradients synchronize
on each micro-batch; “three times parameters per optimizer step” is not universal.

## Compose the parallel dimensions

For this course's balanced teaching layout, define independent coordinates:

$$G=P\times T_P\times C\times D.$$

- $P$: pipeline stages;
- $T_P$: tensor-parallel degree;
- $C$: context-parallel degree;
- $D$: total data-parallel degree, counting independent micro-batches.

**FSDP is a state-sharding choice within $D$.** Let $F$ divide $D$. There are
$D/F$ replica groups, each sharding state over $F$ ranks. Ordinary DDP has $F=1$;
full sharding across the whole DP group has $F=D$. Do not multiply by $F$ again.

**EP can subdivide $D$.** In the simple MoE layout used here, $E_P$ divides $D$:
experts are distributed across $E_P$ ranks and each expert has $D/E_P$ replicas.
Shared and expert parameters have different synchronization groups. Do not
multiply the device count by $E_P$ again. Production layouts can use different
expert TP degrees or fold axes differently; derive their rank groups explicitly.
See [Megatron's parallel configuration](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.model_parallel_config.html).

The companion calculator supports either FSDP or EP under these assumptions.
It rejects combined EP/FSDP layouts rather than guessing the expert shard groups.
For micro-batch size $B_\mu$ and $A$ accumulation micro-batches per update:

$$\text{global tokens/update}=B_\mu\,T\,A\,D.$$

Neither CP nor TP creates extra training examples.

### Persistent state is not peak memory

For a declared mixed-precision AdamW layout, count parameter copies, gradients,
two moments, and any master weights separately. DDP replicates these states.
FSDP shards persistent state but temporarily materializes larger parameter units.
EP distributes expert parameters while shared layers remain replicated over EP.

Add activation residency, gathered weights, collective buffers, allocator reserve,
and the largest-stage imbalance before testing whether a run fits. The calculator
reports **average persistent state per rank**, not peak VRAM or a fit guarantee.

## Strategy case studies

### Case A — model fits, short context, 64 devices

Start with $D=64$ and the other independent degrees at one. Compare strong
scaling at fixed global batch with weak scaling at fixed local batch. DDP can
still be communication-bound; a model fitting on one GPU does not ensure speedup.

### Case B — 512 devices, dense model

One candidate is $T_P=8,P=4,D=16,C=1$, with $F=16$ for state sharding. It uses
512 devices, not 8192. Compare another layout at the same global token budget.
Neither the device identity nor static-state estimate proves memory fit.

### Case C — long-context training on 32 devices

One candidate is $C=8,T_P=4,D=P=1$. CP divides token activations and attention
work, but adds per-layer communication. Estimate training activations and
backward state; an inference KV-cache formula alone is not a training-memory budget.

### Case D — four experts within a data-parallel group

For $D=4,T_P=2,P=C=1$, the world contains eight devices. Choosing $E_P=4$ still
uses eight devices. Each expert-TP rank holds a slice of one expert, while
shared attention parameters remain replicated across the four EP ranks.

## In-class investigation

1. For $P=4,m=16$, calculate idle/total fraction and idle/useful ratio. Find the
   smallest integer $m$ giving an ideal bubble below 10%. Compare AFAB and 1F1B
   activation residency in the [simulator](../demos/pipeline-parallelism.html).
2. Trace four ranks and eight equally sized causal context chunks. Pair early
   and late chunks and explain which imbalance the pairing addresses.
3. For $n=8192,d=4096,k=2,b=2,E_P=8$, estimate EP dispatch/combine payload for
   forward and backward. State the routing and token-ownership assumptions.
4. Draw the rank coordinates for Case D. Identify which parameters EP distributes
   and which it replicates. Explain why multiplying by EP again is incorrect.
5. Propose two placements across eight 8-GPU nodes. Mark transfers crossing nodes,
   specify a bandwidth measurement, and state what evidence would reject your choice.

## Exit ticket

1. Which dimensions create new examples in the global batch?
2. Why can identical device counts have different persistent memory per rank?
3. When can pipeline transfers or DP reductions remain exposed?
4. Why does expert routing imbalance affect both memory and throughput?
5. Which omitted memory terms must be measured before claiming a model fits?

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
[:material-play-circle-outline: Interactive Simulator](../demos/pipeline-parallelism.html){ .md-button target="_blank" }
[:material-code-tags: Practical Companion Guide](../companion/08-context-pipeline.md){ .md-button .md-button--primary }
