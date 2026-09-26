# PyTorch Primitives for Language Models

## Purpose

A foundational catch-up guide on PyTorch tensor mechanics, computation graphs,
and CUDA execution. This primer prepares you for implementing custom Transformer
layers from scratch ([Module 1](../modules/01-transformer.md)), optimizing single-GPU
throughput ([Module 4](../modules/04-single-gpu.md)), and writing custom distributed
autograd operators ([Modules 5–8](../modules/index.md)).

---

## 1. Tensor Memory Layout, Strides, and Contiguity

Every PyTorch tensor is a strided view over a 1D contiguous block of physical
memory called a `Storage`.

```
Physical Storage (1D flat memory in RAM or VRAM):
[ 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 ]
  0     1     2     3     4     5

Tensor View (Shape: [2, 3], Strides: (3, 1)):
Row 0: [ 1.0,  2.0,  3.0 ]  --> indices (0, 1, 2)
Row 1: [ 4.0,  5.0,  6.0 ]  --> indices (3, 4, 5)
```

### Strides and offset formula

For a tensor with dimensions $(d_0, d_1, \dots, d_{k-1})$ and strides
$(s_0, s_1, \dots, s_{k-1})$, the physical storage offset for multi-dimensional index
$(i_0, i_1, \dots, i_{k-1})$ is:

$$\text{offset} = \text{storage\_offset} + \sum_{j=0}^{k-1} i_j \times s_j.$$

```python
import torch

x = torch.arange(6).reshape(2, 3)
print(x.stride())  # (3, 1) -> jump 3 elements to move down a row, 1 to move right a column
```

### Contiguity and view vs. reshape

A tensor is **C-contiguous** (row-major) when its elements are laid out in memory
such that scanning along the last dimension corresponds to stepping through adjacent memory addresses.

When you transpose or permute a tensor, PyTorch **does not copy data**; it merely
adjusts the shape and strides metadata:

```python
x = torch.arange(6).reshape(2, 3)  # shape (2, 3), strides (3, 1)
y = x.t()                          # shape (3, 2), strides (1, 3) - NOT contiguous!
print(y.is_contiguous())           # False

# y.view(6) will raise RuntimeError:
# view size is not compatible with input tensor's size and stride
y_flat = y.contiguous().view(6)    # Copies data into new contiguous storage
# or equivalently:
y_flat = y.reshape(6)              # Calls .contiguous() internally if needed
```

### Attention head splitting pattern

In multi-head attention, you project an input `x` of shape $[B, T, d]$ into $Q, K, V$
and split the hidden dimension into $h$ heads of dimension $d_h = d / h$:

```python
B, T, d = 4, 1024, 768
h = 12
d_h = d // h  # 64

# Linear projection output: [B, T, d]
q = torch.randn(B, T, d)

# 1. Unpack head dimension: [B, T, h, d_h] (contiguous view)
q = q.view(B, T, h, d_h)

# 2. Permute heads before sequence: [B, h, T, d_h] (strided view, NOT contiguous)
q = q.transpose(1, 2)

# When merging heads back after attention:
# q is [B, h, T, d_h] -> transpose to [B, T, h, d_h] -> view to [B, T, d]
# You MUST call .contiguous() before .view():
out = q.transpose(1, 2).contiguous().view(B, T, d)
```

!!! warning "Why not just use `.reshape()` everywhere?"
    `.reshape()` silently clones memory if the tensor is non-contiguous. In large
    language models, unmonitored silent copies spike peak VRAM usage and trigger
    out-of-memory (OOM) errors during the backward pass. Always use explicit `.view()`
    when you intend zero-copy aliasing, or `.contiguous().view()` when you knowingly
    re-pack physical memory.

---

## 2. Broadcasting Semantics

Broadcasting allows arithmetic operations on tensors of different shapes without
allocating memory for duplicate copies.

### The two rules of broadcasting

Starting from the **trailing (rightmost) dimensions** and working backwards:

1. Two dimensions are compatible if they are **equal**, or if **one of them is 1**.
2. If one tensor has fewer dimensions than the other, its shape is prepended with
   dimensions of size `1` until lengths match.

### Common Transformer broadcasting patterns

#### 1. Causal attention mask addition
Attention scores have shape $[B, h, T, T]$. A triangular causal mask has shape $[1, 1, T, T]$:

```python
B, h, T = 2, 8, 4
scores = torch.randn(B, h, T, T)

# Upper triangular mask filled with -inf
mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)  # [T, T]
mask = mask.view(1, 1, T, T)                                      # [1, 1, T, T]

# Broadcast: [1, 1, T, T] expands to [B, h, T, T] with 0 memory allocation
masked_scores = scores + mask
```

#### 2. Embedding bias and LayerNorm affine parameters
```python
# hidden_states: [B, T, d]
# weight (gain): [d]
# bias:          [d]
# Broadcasting treats weight/bias as [1, 1, d]
normalized = (hidden_states - mean) / std * weight + bias
```

#### 3. Expanding vs Repeating
```python
x = torch.randn(1, 64)

# Zero-copy view with stride 0 along expanded dimension:
x_exp = x.expand(8, 64)
print(x_exp.stride())  # (0, 1) -> moving down rows reads the exact same memory!

# Physical copy allocating 8x memory:
x_rep = x.repeat(8, 1)
print(x_rep.stride())  # (64, 1)
```

---

## 3. Advanced Indexing: Gather, Scatter, and Masking

Language model architectures (especially Token Embeddings and Mixture-of-Experts)
rely heavily on dimension-specific gather and scatter operations.

### `torch.gather`

Extracts values from an input tensor along a specified `dim` using an index tensor:

$$\text{output}[i][j][k] = \text{input}[\text{index}[i][j][k]][j][k] \quad (\text{for } \text{dim}=0).$$

```python
# Vocabulary lookup: input logits [B, T, V], target labels [B, T]
B, T, V = 2, 4, 10
logits = torch.randn(B, T, V)
targets = torch.randint(0, V, (B, T))

# Gather the logit corresponding to each target token:
target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
# target_logits shape: [B, T]
```

### `index_add_` (used in MoE combine)

In Mixture-of-Experts (MoE), expert outputs must be accumulated back into their
original token positions:

```python
# total tokens = B*T
num_tokens, d_model = 8, 16
combined_output = torch.zeros(num_tokens, d_model)

# Suppose Expert 0 processed tokens 1, 3, and 7:
expert_out = torch.randn(3, d_model)
token_indices = torch.tensor([1, 3, 7], dtype=torch.long)

# Add expert output into combined buffer at specified row indices:
combined_output.index_add_(dim=0, index=token_indices, source=expert_out)
```

### Masked assignment: `masked_fill` and `torch.where`

```python
# Replace masked positions with a fill value (e.g. padding tokens)
mask = (targets == -100)  # [B, T] boolean mask
loss_weights = torch.ones_like(targets, dtype=torch.float)
loss_weights.masked_fill_(mask, 0.0)

# Branchless conditional selection:
clipped = torch.where(scores > 50.0, 50.0, scores)
```

---

## 4. Einstein Summation (`torch.einsum`)

`torch.einsum` evaluates multi-linear algebraic contractions using Einstein notation.
Repeated indices not present in the output label are summed over.

### Basic operations table

| Operation | Traditional PyTorch | `torch.einsum` |
| :--- | :--- | :--- |
| **Matrix multiplication** | `A @ B` | `einsum("i k, k j -> i j", A, B)` |
| **Batch matmul** | `torch.bmm(A, B)` | `einsum("b i k, b k j -> b i j", A, B)` |
| **Dot product** | `torch.dot(u, v)` | `einsum("i, i ->", u, v)` |
| **Batch trace** | `torch.diagonal(A, dim1=-2, dim2=-1).sum(-1)` | `einsum("b i i -> b", A)` |
| **Transpose** | `A.transpose(0, 1)` | `einsum("i j -> j i", A)` |

### Attention dot-product with einsum

Given Query $Q$ of shape $[B, h, T_q, d]$ and Key $K$ of shape $[B, h, T_k, d]$:

```python
# Contract over head dimension 'd', output attention matrix [B, h, T_q, T_k]
scores = torch.einsum("b h q d, b h k d -> b h q k", Q, K)

# Multiply attention probabilities P [B, h, T_q, T_k] with Value V [B, h, T_k, d]
# Contract over sequence dimension 'k', output [B, h, T_q, d]
context = torch.einsum("b h q k, b h k d -> b h q d", P, V)
```

*Note: In production training loops, use `F.scaled_dot_product_attention` to leverage
FlashAttention kernels, but `einsum` is invaluable for prototyping rotary embeddings
(RoPE) and multi-head latent attention (MLA).*

---

## 5. Autograd Mechanics and Custom Functions

PyTorch builds a directed acyclic graph (DAG) during the forward pass.
Tensors with `requires_grad=True` become leaf nodes. Intermediate operations
create `Node` objects (such as `MulBackward0`, `AddmmBackward0`) that hold gradient
evaluation closures.

```
       x (leaf) ──┐
                  ▼
              [Linear] ──> y ──> [GELU] ──> z ──> [Loss]
                  ▲
       w (leaf) ──┘
```

### Gradients and in-place operations

During backward evaluation, autograd relies on intermediate tensors saved during forward.
If an in-place operation modifies a saved buffer, autograd raises an error:

```python
a = torch.randn(5, requires_grad=True)
b = a * 2
b.add_(1.0)  # In-place modification!
# loss = b.sum()
# loss.backward() -> Raises RuntimeError: tensor was modified in place!
```

### Context managers: `no_grad` vs. `inference_mode`

- `torch.no_grad()`: Disables graph construction. Keeps tracking version counters.
  Usable for validation steps during training.
- `torch.inference_mode()`: Completely bypasses autograd bookkeeping and view tracking.
  Faster than `no_grad()`, ideal for production inference and KV-cache generation.

### Writing a custom `torch.autograd.Function`

Custom autograd functions are required when creating distributed operators
(such as Tensor Parallelism Column/Row boundaries or Context Parallelism ring shifts)
where PyTorch cannot infer backward derivatives through communication sockets.

```python
class ColumnParallelLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # Save tensors needed for backward
        ctx.save_for_backward(input, weight)
        ctx.has_bias = bias is not None
        output = input @ weight.t()
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        input, weight = ctx.saved_tensors
        
        # dL/d(input) = grad_output @ weight
        grad_input = grad_output @ weight
        # dL/d(weight) = grad_output^T @ input
        grad_weight = grad_output.t() @ input
        grad_bias = grad_output.sum(dim=0) if ctx.has_bias else None
        
        # Return gradients in the exact order of forward() arguments
        return grad_input, grad_weight, grad_bias
```

---

## 6. PyTorch Modules: Parameters, Buffers, and Hooks

An `nn.Module` manages two types of internal tensors:

| Tensor Type | Registered via | Stored in `state_dict()`? | Tracked by `optimizer.step()`? | Moved with `.to(device)`? |
| :--- | :--- | :---: | :---: | :---: |
| **`nn.Parameter`** | `nn.Parameter(...)` or assigned attribute | **Yes** | **Yes** | **Yes** |
| **Persistent Buffer** | `register_buffer("name", t, persistent=True)` | **Yes** | **No** | **Yes** |
| **Non-persistent Buffer** | `register_buffer("name", t, persistent=False)` | **No** | **No** | **Yes** |
| **Plain Attribute** | `self.attr = t` | **No** | **No** | **No** |

### The causal mask: register as buffer

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Causal mask is NOT a parameter, but must follow model to CUDA/BF16:
        mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("bias", mask, persistent=False)

    def forward(self, x):
        # self.bias is automatically on the same device as x!
        pass
```

### Hooks for gradient monitoring and bucketing

Hooks allow running custom logic during forward or backward passes.
Distributed Data Parallel (DDP) uses backward hooks to trigger asynchronous
all-reduce communication as soon as each parameter's gradient finishes computing:

```python
linear = nn.Linear(64, 64)

# Hook executed whenever gradient of parameter is computed
def grad_hook(grad):
    # DDP queues this gradient into an all-reduce bucket here!
    return grad

linear.weight.register_hook(grad_hook)
```

---

## 7. CUDA Execution Model and Memory Management

### Host launch queue vs. Device execution queue

All PyTorch CUDA operations are **asynchronous** by default. When you execute
`c = a @ b`, the CPU queues the kernel launch into the CUDA device stream and
immediately returns to the Python interpreter:

```python
# CPU queues kernel launches without waiting:
y = model(x)
loss = criterion(y, targets)
loss.backward()

# To measure true device execution time, you MUST synchronize:
torch.cuda.synchronize()  # Blocks CPU until GPU completes all queued kernels
```

### Accurate timing with `torch.cuda.Event`

Never use Python's `time.time()` without synchronization to benchmark GPU kernels:

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
out = model(x)
end_event.record()

# Wait for end_event to complete on device
end_event.synchronize()
elapsed_ms = start_event.elapsed_time(end_event)
print(f"Forward time: {elapsed_ms:.2f} ms")
```

### Pinned host memory and non-blocking data loading

Transferring data from CPU RAM to GPU VRAM over PCIe can be overlapped with compute
if host memory is page-locked (pinned):

```python
# In DataLoader:
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)

# In training loop:
# non_blocking=True allows asynchronous DMA copy over PCIe
inputs = inputs.to("cuda", non_blocking=True)
targets = targets.to("cuda", non_blocking=True)
```

### The PyTorch Caching Allocator

PyTorch manages VRAM through an internal caching allocator to avoid expensive
calls to `cudaMalloc` and `cudaFree`.

```
Total Device VRAM (e.g., 24 GiB)
│
├── Reserved Memory (Held by PyTorch allocator pool, e.g. 16 GiB)
│   ├── Active Allocated Memory (Tensors currently in use: 10 GiB)
│   └── Cached / Inactive Blocks (Available for immediate reuse: 6 GiB)
│
└── Free Device Memory (Unallocated OS VRAM: 8 GiB)
```

```python
allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Active tensors in MB
reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # Cached pool in MB
print(f"Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
```

!!! tip "Does `torch.cuda.empty_cache()` prevent OOM?"
    **No.** `torch.cuda.empty_cache()` releases inactive cached blocks back to the OS,
    but does not reduce the memory required by active tensors. It also incurs a major
    performance penalty due to GPU pipeline synchronization. Use it only when
    transitioning between training and evaluation phases, not inside training loops.
