# Distributed Primitives for Language Models

## Purpose

A foundational reference on distributed collective communication, autograd adjoints,
and network cost models. This primer bridges single-GPU mechanics
([Module 4](../modules/04-single-gpu.md)) and the multi-GPU parallelization curriculum:
Distributed Data Parallelism ([Module 5](../modules/05-ddp.md)), Fully Sharded Data Parallel
([Module 6](../modules/06-fsdp.md)), Tensor Parallelism ([Module 7](../modules/07-tensor-parallelism.md)),
and Context/Pipeline/Expert Parallelism ([Module 8](../modules/08-context-pipeline.md)).

---

## 1. Process Model and Topology Terminology

In distributed deep learning, a training job executes across multiple coordinated
worker processes:

- **World Size ($R$):** The total number of cooperating processes in the distributed group.
- **Global Rank ($r$):** Unique process identifier in $[0, R-1]$.
- **Node:** A physical server housing one or more GPUs (typically 4 or 8).
- **Local Rank:** Identifier of the process within its physical node (e.g. $[0, 7]$ on an 8-GPU node).
- **Process Group:** An explicit subset of ranks that communicate together via a backend (e.g. NCCL).

```
Node 0                                Node 1
┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│ GPU 0        GPU 1        GPU 2 │   │ GPU 3        GPU 4        GPU 5 │
│ Rank 0       Rank 1       Rank 2│   │ Rank 3       Rank 4       Rank 5│
│ LocalRank 0  LocalRank 1  Loc..2│   │ LocalRank 0  LocalRank 1  Loc..2│
└─────────────────────────────────┘   └─────────────────────────────────┘
        ▲                                     ▲
        └───────────────── Network ───────────┘
               (InfiniBand / RoCE / Ethernet)
```

In PyTorch, initialization under `torchrun` queries standard environment variables:

```python
import os
import torch
import torch.distributed as dist

def init_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size
```

---

## 2. Taxonomy of Communication Primitives

Distributed deep learning relies on three categories of communication:

1. **Point-to-Point (P2P):** Direct transfer between two specific ranks.
2. **One-to-All / All-to-One:** Broadcast, scatter, gather, and reduce.
3. **All-to-All Multi-Rank Collectives:** All-gather, reduce-scatter, all-reduce, and all-to-all.

---

### 2.1 Point-to-Point (P2P: Send / Recv)

Transfers a tensor directly from a single sender rank to a single receiver rank.

```
Rank 0:  [ Tensor A ] ──────────( send / recv )──────────> Rank 1: [ Tensor A ]
```

- **Semantics:** Rank $i$ sends; Rank $j$ receives.
- **PyTorch API:**
  - Synchronous: `dist.send(tensor, dst=1)`, `dist.recv(tensor, src=0)`
  - Asynchronous: `dist.isend(tensor, dst=1)`, `dist.irecv(tensor, src=0)`
  - Batching: `dist.batch_isend_irecv([P2POp(dist.isend, ...), P2POp(dist.irecv, ...)])`
- **Used in:**
  - **Pipeline Parallelism (PP):** Passing activation tensors forward and gradient tensors backward across adjacent pipeline stages.
  - **Context Parallelism (CP / Ring Attention):** Circulating key-value blocks in a ring.

---

### 2.2 Broadcast

Copies a tensor located on a designated root rank to all other ranks in the process group.

```
Initial State:                    After Broadcast (root=0):
Rank 0 (root): [ Data A ]         Rank 0: [ Data A ]
Rank 1:        [        ]   ==>   Rank 1: [ Data A ]
Rank 2:        [        ]         Rank 2: [ Data A ]
Rank 3:        [        ]         Rank 3: [ Data A ]
```

- **Tensor Shapes:** Input on root: $[N]$. Output on all ranks: $[N]$.
- **PyTorch API:** `dist.broadcast(tensor, src=0)`
- **Used in:**
  - Synchronizing initial model parameters and optimizer configuration at rank 0 to all workers at training launch.

---

### 2.3 Scatter and Gather

- **Scatter:** Splits a tensor on root into $R$ equal slices and sends slice $i$ to rank $i$.
- **Gather:** Collects slice $i$ from rank $i$ and concatenates them onto the root rank.

```
Scatter (root=0):                 Gather (root=0):
Rank 0 (root): [A | B | C | D]    Rank 0: [ A ] ──┐
Rank 1:        [             ]    Rank 1: [ B ] ──┼──> Rank 0: [A | B | C | D]
Rank 2:        [             ]    Rank 2: [ C ] ──┤
Rank 3:        [             ]    Rank 3: [ D ] ──┘
      │
      ▼
Rank 0: [ A ], Rank 1: [ B ], Rank 2: [ C ], Rank 3: [ D ]
```

- **Tensor Shapes:** Root: $[N] \longleftrightarrow$ All ranks: $[N/R]$.
- **PyTorch API:** `dist.scatter(tensor, scatter_list, src=0)`, `dist.gather(tensor, gather_list, dst=0)`.
- **Used in:** Checkpoint saving/loading, central evaluation metric logging.

---

### 2.4 All-Gather

Every rank begins with a local shard of size $[N/R]$. All ranks communicate so that
every rank ends up with the complete concatenated tensor $[N]$.

```
Initial State:                    After All-Gather:
Rank 0: [ A ]                     Rank 0: [ A | B | C | D ]
Rank 1: [ B ]               ==>   Rank 1: [ A | B | C | D ]
Rank 2: [ C ]                     Rank 2: [ A | B | C | D ]
Rank 3: [ D ]                     Rank 3: [ A | B | C | D ]
```

- **Tensor Shapes:** Input on each rank: $[N/R]$. Output on each rank: $[N]$.
- **PyTorch API:**
  - Modern flat buffer (recommended): `dist.all_gather_into_tensor(output_tensor, input_shard)`
  - Legacy list: `dist.all_gather(tensor_list, input_shard)`
- **Used in:**
  - **FSDP / ZeRO-3:** Materializing un-sharded layer parameters before forward and backward compute.
  - **Sequence Parallelism (SP):** Gathering activations along the sequence length before LayerNorm and Dropout.

---

### 2.5 Reduce-Scatter

Every rank begins with a full-sized tensor $[N]$. An elementwise reduction (e.g. SUM)
is computed across all ranks, and the reduced result is partitioned so that rank $i$
receives only the $i$-th reduced shard $[N/R]$.

```
Initial State:                    After Reduce-Scatter (op=SUM):
Rank 0: [ A0 | B0 | C0 | D0 ]     Rank 0: [ ΣA = A0+A1+A2+A3 ]
Rank 1: [ A1 | B1 | C1 | D1 ] ==> Rank 1: [ ΣB = B0+B1+B2+B3 ]
Rank 2: [ A2 | B2 | C2 | D2 ]     Rank 2: [ ΣC = C0+C1+C2+C3 ]
Rank 3: [ A3 | B3 | C3 | D3 ]     Rank 3: [ ΣD = D0+D1+D2+D3 ]
```

- **Tensor Shapes:** Input on each rank: $[N]$. Output on each rank: $[N/R]$.
- **PyTorch API:**
  - Modern flat buffer (recommended): `dist.reduce_scatter_tensor(output_shard, input_tensor, op=dist.ReduceOp.SUM)`
  - Legacy list: `dist.reduce_scatter(output_shard, input_tensor_list, op=dist.ReduceOp.SUM)`
- **Used in:**
  - **ZeRO-2 and FSDP:** Reducing parameter gradients across data-parallel ranks while leaving each rank only its owned optimizer shard.
  - **Sequence Parallelism (SP):** Reducing partial hidden states into sharded representations.

---

### 2.6 All-Reduce

Every rank begins with a full-sized tensor $[N]$. An elementwise reduction (typically SUM)
is computed across all ranks, and the complete reduced tensor $[N]$ is stored on every rank.

```
Initial State:                    After All-Reduce (op=SUM):
Rank 0: [ A0 | B0 | C0 | D0 ]     Rank 0: [ ΣA | ΣB | ΣC | ΣD ]
Rank 1: [ A1 | B1 | C1 | D1 ] ==> Rank 1: [ ΣA | ΣB | ΣC | ΣD ]
Rank 2: [ A2 | B2 | C2 | D2 ]     Rank 2: [ ΣA | ΣB | ΣC | ΣD ]
Rank 3: [ A3 | B3 | C3 | D3 ]     Rank 3: [ ΣA | ΣB | ΣC | ΣD ]
```

- **Equivalence:**
  $$\text{All-Reduce} \equiv \text{Reduce-Scatter} \xrightarrow{\quad} \text{All-Gather}.$$
- **Tensor Shapes:** Input on each rank: $[N]$. Output on each rank: $[N]$.
- **PyTorch API:** `dist.all_reduce(tensor, op=dist.ReduceOp.SUM)`
- **Used in:**
  - **DDP (Distributed Data Parallel):** Averaging gradients across data replicas after backward.
  - **Tensor Parallelism (TP):** Summing partial matrix multiplications across ranks at RowParallelLinear boundaries.

---

### 2.7 All-to-All (`all_to_all_single`)

Every rank begins with $R$ distinct slices (one destined for each rank). Rank $i$ sends its
$j$-th slice to rank $j$, and receives from rank $j$ the slice destined for rank $i$.
This is a **distributed matrix transpose**.

```
Initial State:                    After All-to-All:
Rank 0: [ A0 | B0 | C0 | D0 ]     Rank 0: [ A0 | A1 | A2 | A3 ]
Rank 1: [ A1 | B1 | C1 | D1 ] ==> Rank 1: [ B0 | B1 | B2 | B3 ]
Rank 2: [ A2 | B2 | C2 | D2 ]     Rank 2: [ C0 | C1 | C2 | C3 ]
Rank 3: [ A3 | B3 | C3 | D3 ]     Rank 3: [ D0 | D1 | D2 | D3 ]
```

- **Tensor Shapes:** Input: $[R \times M]$. Output: $[R \times M]$ (with chunks transposed).
- **PyTorch API:** `dist.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None)`
- **Used in:**
  - **Expert Parallelism (EP / MoE):** Dispatching tokens from data-parallel ranks to remote expert ranks, and combining expert outputs back to originating ranks.

---

## 3. Autograd Adjoints & Mathematical Duality

When writing distributed models, backward automatic differentiation through a
communication collective requires evaluating its **mathematical adjoint**.

### The Communication Adjoint Table

| Forward Operation | Forward Behavior | Backward Adjoint | Why? |
| :--- | :--- | :--- | :--- |
| **All-Gather** | Sharded $\to$ Replicated | **Reduce-Scatter** | Replicated upstream outputs accumulate gradients on all ranks; backward must sum and scatter gradients back to shard owners. |
| **Reduce-Scatter** | Partials $\to$ Sharded sum | **All-Gather** | Each rank has local gradient of its shard; backward gathers them to restore full gradient. |
| **All-Reduce** | Partials $\to$ Replicated sum | **Identity (No-Op)** | Upstream gradients are already identical and local to each rank. No network transfer needed! |
| **All-to-All** | Routing (Rank $i \to j$) | **All-to-All** | Transposing the backward gradients routes them back to originating tokens (Rank $j \to i$). |
| **P2P Send $\to$ Recv** | Forward stage transfer | **P2P Recv $\leftarrow$ Send** | Gradients travel in the exact opposite direction. |

### Implementing TP Adjoints with `torch.autograd.Function`

The Megatron-LM Tensor Parallelism module implements these adjoints explicitly:

```python
import torch
import torch.distributed as dist

class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass input through in forward, all-reduce gradients in backward.
    Used before ColumnParallelLinear layers.
    """
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        if dist.get_world_size() > 1:
            dist.all_reduce(grad_output, op=dist.ReduceOp.SUM)
        return grad_output


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce output in forward, pass gradients through in backward.
    Used after RowParallelLinear layers.
    """
    @staticmethod
    def forward(ctx, input_):
        if dist.get_world_size() > 1:
            dist.all_reduce(input_, op=dist.ReduceOp.SUM)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split tensor along last dim in forward, all-gather in backward."""
    @staticmethod
    def forward(ctx, input_):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return torch.chunk(input_, world_size, dim=-1)[rank].contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        world_size = dist.get_world_size()
        grad_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        dist.all_gather(grad_list, grad_output)
        return torch.cat(grad_list, dim=-1)
```

---

## 4. Communication Cost Models and Scaling Laws

### 4.1 Volume Transferred per Rank

Let $S$ be the total payload size in bytes, and $R$ be the world size.

| Collective | Bytes Sent per Rank | Bytes Received per Rank | Total Network Transfer (Sent) |
| :--- | :---: | :---: | :---: |
| **Point-to-Point** | $S$ | $S$ | $S$ |
| **Broadcast** | $S$ (root only) | $S$ (non-root) | $(R-1)S$ |
| **All-Gather** | $\frac{R-1}{R} S$ | $\frac{R-1}{R} S$ | $(R-1)S$ |
| **Reduce-Scatter** | $\frac{R-1}{R} S$ | $\frac{R-1}{R} S$ | $(R-1)S$ |
| **All-Reduce** | $2 \frac{R-1}{R} S$ | $2 \frac{R-1}{R} S$ | $2(R-1)S$ |
| **All-to-All** | $\frac{R-1}{R} S$ | $\frac{R-1}{R} S$ | $(R-1)S$ |

Notice that as $R \to \infty$:
- All-Gather, Reduce-Scatter, and All-to-All send roughly $S$ bytes per rank.
- All-Reduce sends roughly $2S$ bytes per rank. Adding more ranks **does not decrease** the per-rank gradient communication volume in DDP!

### 4.2 The Hockney Communication Model

A standard communication latency model expresses transfer time $T_{\text{comm}}$ as:

$$T_{\text{comm}}(S) \approx \alpha + \beta S,$$

where:
- $\alpha$: Connection startup and software launch latency (in seconds).
- $\beta = \frac{1}{\text{BW}_{\text{eff}}}$: Inverse effective transfer bandwidth (in seconds/byte).

For small payloads ($S < 100\text{ KB}$), latency $\alpha$ dominates. For large payloads ($S > 10\text{ MB}$), bandwidth $\beta S$ dominates.

### 4.3 Ring vs. Tree Collective Algorithms

Collectives like All-Reduce are implemented differently depending on payload size $S$ and rank count $R$:

```
Ring All-Reduce (Bandwidth-Optimal):
Each rank communicates only with its immediate neighbors (rank r -> r+1):
Total time: T_ring = 2(R - 1) * alpha + 2 * ((R - 1)/R) * beta * S

Tree All-Reduce (Latency-Optimal):
Ranks form a binomial or recursive doubling tree:
Total time: T_tree = 2 * log2(R) * alpha + 2 * log2(R) * beta * (S / tree_efficiency)
```

- **Ring Crossover:** When message size $S$ is large (e.g. 50 MB gradient buckets), the ring algorithm wins because its bandwidth multiplier $\frac{R-1}{R} \approx 1$ does not scale with $R$.
- **Tree Crossover:** When message size $S$ is small (e.g. TP activation vectors), tree all-reduce wins because it finishes in $\mathcal{O}(\log R)$ steps rather than $\mathcal{O}(R)$ ring steps.

---

## 5. Hardware Topologies and Mapping

### Physical Interconnect Hierarchy

Modern GPU clusters exhibit a steep hierarchy of communication bandwidth:

| Interconnect | Typical Bandwidth (per GPU) | Typical One-Way Latency | Scope |
| :--- | :---: | :---: | :--- |
| **NVLink 4 / 5 (NVSwitch)** | $450\text{--}900\text{ GB/s}$ (bidirectional) | $< 1\ \mu\text{s}$ | Intra-node (within 8 GPUs) |
| **PCIe Gen 5** | $64\text{ GB/s}$ (bidirectional) | $\sim 2\text{--}5\ \mu\text{s}$ | Intra-node CPU-GPU |
| **InfiniBand NDR / RoCE** | $25\text{--}100\text{ GB/s}$ ($200\text{--}800\text{ Gbps}$) | $5\text{--}15\ \mu\text{s}$ | Inter-node (across network fabric) |

### Mapping Parallelism Dimensions to Interconnects

```
High Bandwidth & Low Latency (NVLink) ────────────> Lower Bandwidth (InfiniBand)
             TP / CP                                       PP / DP / FSDP
[ Frequent layer-internal collectives ]           [ Infrequent step-boundary collectives ]
```

1. **Tensor Parallelism (TP):** Executes 2 All-Reduces **per Transformer layer**. Must stay inside a single node across NVLink. Placing TP across an Ethernet or InfiniBand network will stall compute.
2. **Context Parallelism (CP):** Circulates KV blocks in every attention head. Prefers NVLink or high-speed intra-rack fabrics.
3. **Pipeline Parallelism (PP):** Passes boundary activation tensors between stages. Payloads are small ($B_\mu \times T \times d$). Can cross inter-node InfiniBand links easily.
4. **Data Parallelism (DDP / FSDP):** All-reduces or reduce-scatters gradients once per step. Easily overlapped with backward pass compute across inter-node networks.

---

## 6. PyTorch Distributed Systems Engineering

### 6.1 CUDA Streams and Asynchronous Overlap

To overlap communication with computation, PyTorch executes collectives on a dedicated
internal **NCCL CUDA stream** distinct from the default compute stream.

```python
# Compute on default stream:
c = a @ b

# Launch async collective on NCCL stream:
work = dist.all_reduce(c, async_op=True)

# Compute independent operations on default stream while transfer proceeds:
d = compute_something_else()

# Block compute stream until communication finishes:
work.wait()

# Now safe to consume result of c:
result = c + d
```

### 6.2 Flat Contiguous Buffers vs. Python Lists

Legacy PyTorch collectives operated on lists of tensors (`dist.all_gather(tensor_list, shard)`).
This caused high Python overhead and memory fragmentation.

Always use modern flat tensor APIs:
- `dist.all_gather_into_tensor(output, input)`
- `dist.reduce_scatter_tensor(output, input)`

```python
# GOOD: Single flat memory allocation and one NCCL kernel launch
world_size = dist.get_world_size()
input_shard = torch.randn(1024, 768, device="cuda")
gathered_flat = torch.empty(world_size * 1024, 768, device="cuda")

dist.all_gather_into_tensor(gathered_flat, input_shard)
```

---

## 7. The Rosetta Stone: Parallelism Strategies ↔ Primitives

A summary matrix connecting every parallelization paradigm in the course
to its required communication primitives:

| Parallel Strategy | Module | Forward Collectives | Backward Collectives | Critical Network Constraint |
| :--- | :---: | :--- | :--- | :--- |
| **DDP** | [Mod 5](../modules/05-ddp.md) | None (local compute) | **All-Reduce** (or RS + AG) on gradients | Backward compute overlap |
| **ZeRO-1 / 2** | [Mod 6](../modules/06-fsdp.md) | None | **Reduce-Scatter** on gradients | Gradient bucket memory |
| **ZeRO-3 / FSDP** | [Mod 6](../modules/06-fsdp.md) | **All-Gather** (layer parameters) | **All-Gather** (params) + **Reduce-Scatter** (grads) | Parameter prefetching & resharding |
| **Tensor (TP)** | [Mod 7](../modules/07-tensor-parallelism.md) | Column: None<br>Row: **All-Reduce** | Column: **All-Reduce**<br>Row: None | Sub-microsecond latency (NVLink) |
| **Sequence (SP)** | [Mod 7](../modules/07-tensor-parallelism.md) | **All-Gather** $\to$ **Reduce-Scatter** | **All-Gather** $\to$ **Reduce-Scatter** | Activation memory footprint |
| **Context (CP)** | [Mod 8](../modules/08-context-pipeline.md) | **P2P Ring Send/Recv** (KV blocks) | **P2P Ring Send/Recv** (dK, dV blocks) | Online softmax & ring overlap |
| **Pipeline (PP)** | [Mod 8](../modules/08-context-pipeline.md) | **P2P Send/Recv** (activations) | **P2P Send/Recv** (gradients) | Pipeline bubble scheduling (1F1B) |
| **Expert (EP)** | [Mod 8](../modules/08-context-pipeline.md) | **All-to-All** (token dispatch & combine) | **All-to-All** (gradient routing) | Expert router load balancing |
