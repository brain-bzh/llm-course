# Module 2 — Training-loop anatomy

## Purpose

Understand every state change between a batch of tokens and an optimizer update.

A deep learning training loop appears deceptively simple on the surface, but
getting operations in the correct order is remarkably fragile. Most mistakes in
PyTorch training loops do not throw errors or raise exceptions; instead, they
fail silently—causing runaway GPU memory consumption, stale parameter updates,
unintended gradient accumulation, or numerical divergence.

This module deconstructs the training loop from first principles, progressively
adding production LLM training features while grounding the mechanics in
[The Annotated PyTorch Training Loop](https://idlemachines.co.uk/essays/pytorch-training-loop).

## Key ideas

- autograd computation graph construction and vector-Jacobian backpropagation;
- decoupled weight decay (AdamW) and parameter grouping (2D weights vs 1D biases);
- gradient accumulation and exact loss normalization across micro-batches;
- global gradient norm clipping to prevent attention instability;
- warmup and cosine learning-rate schedules;
- strict isolation between training and evaluation states;
- deterministic checkpointing and full optimizer state recovery;
- automatic mixed precision (AMP) and host-to-device streaming efficiency.

## The minimal training loop

The core of any PyTorch training loop consists of five sequential steps:

```python
for x, y in dataloader:
    optimizer.zero_grad(set_to_none=True)  # 1. Clear gradients
    logits, loss = model(x, targets=y)      # 2. Forward pass & loss
    loss.backward()                        # 3. Backward pass (accumulate)
    optimizer.step()                       # 4. Update model parameters
```

To understand why this sequence works—and what breaks if any line moves—we must
examine the internal state transitions:

1. **Clearing the gradient buffer (`optimizer.zero_grad(set_to_none=True)`):**
   Every trainable `torch.nn.Parameter` has a `.grad` field. PyTorch's autograd
   engine *accumulates* gradients additively into this field (`param.grad += grad`).
   Setting `set_to_none=True` deallocates the memory buffer instead of writing
   zeros over every tensor element, saving memory bandwidth and reducing peak VRAM.
2. **The forward pass (`model(x)`):**
   As input tensors pass through each operation, PyTorch constructs a Directed
   Acyclic Graph (DAG) of `torch.autograd.Node` objects. Each node records its
   inputs, intermediate tensors needed for differentiation (saved via
   `ctx.save_for_backward`), and the vector-Jacobian product function.
3. **The loss function:**
   Computes a scalar loss from model predictions and ground-truth targets. For
   language models, this is categorical cross-entropy. The returned tensor
   is the root of the computation graph (`loss.grad_fn` points to the last node).
4. **The backward pass (`loss.backward()`):**
   Traverses the DAG backwards from `loss` to every leaf parameter. It computes
   derivatives via the chain rule and populates `.grad` on all parameters with
   `requires_grad=True`. Crucially, **`loss.backward()` does not modify weights**.
5. **The parameter update (`optimizer.step()`):**
   Reads `.grad` from every parameter and executes the optimizer update rule in place.
   **Parameters change here and nowhere else.**

<figure markdown="span">
  ![Data flow and state transitions in an LLM training loop: micro-batch forward and backward accumulation, gradient clipping, AdamW parameter updates, and gradient reset.](../assets/figures/training-loop-anatomy.svg){ loading=lazy }
  <figcaption>The anatomy of an LLM training step: micro-batch gradient accumulation, gradient norm clipping, and the AdamW update cycle.</figcaption>
</figure>

---

## Where order really matters: The silent failure catalog

The single most dangerous property of a PyTorch training loop is that
**misplaced lines rarely raise an exception**. The script executes cleanly, but the
mathematical or computational behavior is completely broken.

The table below catalogs these failure modes, grounded in the analysis from
[The Annotated PyTorch Training Loop](https://idlemachines.co.uk/essays/pytorch-training-loop):

| Operation | Wrong placement | Consequence (Silent Failure) |
| :--- | :--- | :--- |
| `model.to(device)` | **After** `optimizer = AdamW(...)` | Casting or moving a module creates new `nn.Parameter` tensors. The optimizer retains references to the discarded host parameters, updating dead memory while the active GPU model never trains. |
| `optimizer.zero_grad()` | **After** `loss.backward()` | Gradients from previous batches are not cleared; parameter updates use a growing sum of all historical batches, causing explosive divergence. |
| `clip_grad_norm_()` | **Before** `loss.backward()` | At this point, `param.grad` is `None`. The clipping function finds zero norms and executes as a silent no-op. Gradients remain unclipped. |
| `clip_grad_norm_()` | **After** `optimizer.step()` | Gradients are clipped *after* the optimizer has already applied unconstrained updates to weights. The clip is completely wasted. |
| `scheduler.step()` | **Inside** micro-batch loop | The learning rate decays once per micro-step instead of once per optimizer step, causing the learning rate to collapse orders of magnitude too early. |
| Omit `model.train()` | After running `model.eval()` | Dropout remains disabled and LayerNorm/BatchNorm tracking is frozen. The model continues training in evaluation mode without error. |
| Omit `torch.no_grad()` | During validation loop | Autograd constructs computation graphs for every evaluation batch, holding all validation activations in VRAM until an Out-Of-Memory (OOM) crash occurs. |
| Logging `loss` | Instead of `loss.item()` | Storing the tensor `loss` keeps a live reference to the entire computation graph, preventing Python garbage collection and steadily leaking memory across epochs. |

---

## Layering production features

A bare 5-line loop cannot train a modern Transformer stably or efficiently.
We progressively layer on the necessary features:

### 1. Decoupled weight decay (AdamW) & parameter grouping

Standard L2 regularization adds a penalty $\frac{\lambda}{2} \|W\|_2^2$ to the loss.
For SGD, this is mathematically equivalent to weight decay. However, for adaptive
gradient optimizers like Adam, L2 regularization scales the decay by the running
gradient variance $\sqrt{v_t}$:

$$W_{t+1} = W_t - \eta \frac{g_t + \lambda W_t}{\sqrt{v_t} + \epsilon}$$

Parameters with large historical gradients receive *less* decay than parameters
with sparse gradients. Loshchilov & Hutter (2017) demonstrated that true weight
decay must be **decoupled** from the gradient update:

$$W_{t+1} = W_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda W_t \right)$$

Furthermore, weight decay should **only** apply to matrix multiplications and
embedding weights. Decaying 1D tensors (such as LayerNorm/RMSNorm scale $\gamma$,
bias $\beta$, and linear biases) degrades training stability:

```python
def configure_optimizers(model, weight_decay=0.1, lr=6e-4, betas=(0.9, 0.95)):
    # Partition parameters: 2D weights get decay, 1D vectors do not
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)
```

### 2. Gradient accumulation

Language models require large batch sizes (e.g., $0.5\text{M}$ to $4\text{M}$ tokens)
for smooth gradient estimates and stable optimization. Because a single GPU's VRAM
can only hold a small micro-batch ($B_{\text{micro}} \times T$), we accumulate
gradients across $M$ micro-steps before updating weights:

$$\text{Tokens per update} = B_{\text{micro}} \times T \times M \times N_{\text{GPUs}}$$

Because autograd accumulates gradients additively, we must **scale the loss down by $M$**
before calling `.backward()`. Otherwise, the accumulated gradient will be $M$ times
too large:

```python
# Effective batch = grad_accum_steps * micro_batch_size
for micro in range(grad_accum_steps):
    logits, loss = model(x_micro, targets=y_micro)
    # Scale loss so the accumulated sum equals the true mean
    loss = loss / grad_accum_steps
    loss.backward()

# Update weights only once per accumulated batch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

### 3. Gradient norm clipping

Even with AdamW and pre-norm Transformers, self-attention layers periodically
encounter anomalous batches or pathological token sequences that cause extreme
gradient spikes. If left unclipped, a single spike can push parameters into a
divergent regime from which recovery is impossible.

We compute the global $L_2$ norm of all parameter gradients combined:

$$\|g\|_2 = \sqrt{\sum_{i=1}^P \|g_i\|_2^2}$$

If $\|g\|_2 > M_{\text{clip}}$, every gradient tensor is rescaled by
$\frac{M_{\text{clip}}}{\|g\|_2 + 10^{-6}}$:

```python
# Must occur strictly AFTER backward() and BEFORE optimizer.step()
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Tracking `total_norm` over time is a primary diagnostic for training health: a
healthy run maintains a stable norm ($0.1 \le \|g\|_2 \le 1.5$), whereas frequent
clipping saturation indicates learning rate instability.

### 4. Learning-rate schedules: Warmup and cosine decay

Randomly initialized attention layers produce nearly uniform attention distributions,
generating noisy gradients that would corrupt initial embeddings under a large
initial learning rate.

A standard LLM schedule incorporates two phases:

1. **Linear warmup:** Linearly increases the learning rate from $0$ to $\eta_{\text{max}}$
   over $T_{\text{warmup}}$ steps.
2. **Cosine decay:** Anneals the learning rate from $\eta_{\text{max}}$ down to
   $\eta_{\text{min}} \approx 0.1 \times \eta_{\text{max}}$ over the remaining steps:

$$\eta_t = \begin{cases}
\eta_{\text{max}} \cdot \frac{t + 1}{T_{\text{warmup}} + 1}, & t < T_{\text{warmup}} \\
\eta_{\text{min}} + \frac{1}{2} (1 + \cos(\pi \frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}})) (\eta_{\text{max}} - \eta_{\text{min}}), & T_{\text{warmup}} \le t \le T_{\text{max}} \\
\eta_{\text{min}}, & t > T_{\text{max}}
\end{cases}$$

```python
def get_lr_cosine_schedule(step, warmup_steps, max_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
```

In LLM pretraining, schedules are parameterized by **global optimizer steps**, not
epochs, because datasets consist of billions of non-repeating tokens.

### 5. Evaluation protocol and memory isolation

Validation monitors cross-entropy loss and Perplexity ($PPL = \exp(\mathcal{L})$)
on unseen tokens. Two distinct mechanisms must be used together during evaluation:

```python
@torch.no_grad()
def evaluate_loss(model, dataset, eval_iters=20):
    model.eval()  # 1. Disable dropout, freeze batch stats
    losses = []
    for _ in range(eval_iters):
        x, y = dataset.get_batch()
        _, loss = model(x, targets=y)
        losses.append(loss.item())  # 2. Extract float to prevent memory leak
    model.train()  # 3. Restore training mode!
    return sum(losses) / len(losses)
```

Notice the three critical invariants:
- `model.eval()` toggles module behavior (e.g. disables dropout); it does **not** stop autograd.
- `torch.no_grad()` disables gradient tracking and memory allocation for the DAG; it does **not** change module behavior.
- `model.train()` must be restored immediately after the evaluation loop finishes.

### 6. Checkpointing and exact recovery invariants

A production training run must be preemptible. A valid checkpoint must capture
the entire state required to reproduce the exact subsequent trajectory:

```python
def save_checkpoint(path, model, optimizer, config, step, val_loss):
    raw_model = model.module if hasattr(model, "module") else model
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),  # First and second moments!
        "config": config,
        "step": step,
        "val_loss": val_loss,
    }
    torch.save(state, path)
```

If `optimizer.state_dict()` is omitted, the momentum buffers ($m_t$ and $v_t$)
are lost. Upon resuming, the optimizer starts cold with zero momentum, causing
a sudden loss spike.

### 7. Automatic Mixed Precision (AMP) & GPU efficiency

Modern GPUs (NVIDIA Ampere/Hopper, Apple Silicon) deliver several times higher
compute throughput when executing matrix multiplications in half-precision
(`bfloat16` or `float16`) instead of `float32`.

- **`bfloat16` (Preferred for LLMs):** Possesses the exact same 8-bit dynamic exponent
  range as `float32`. Gradients do not underflow, so no gradient scaling is needed.
- **`float16` (Legacy/Inference):** Has a 5-bit exponent range. Small gradients
  underflow to zero unless multiplied by a dynamic scale factor via `GradScaler`.

When using `GradScaler`, the order of operations between unscaling and clipping
is critical:

```python
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

# Inside micro-step:
with torch.amp.autocast(device_type="cuda", dtype=dtype):
    logits, loss = model(x, targets=y)
    loss = loss / grad_accum_steps
scaler.scale(loss).backward()

# At optimizer step boundary:
# Must UNSCALE gradients BEFORE clipping, otherwise the norm is scaled!
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

Furthermore, data transfers between Host (CPU) and Device (GPU) can overlap with
compute if the source tensor is allocated in page-locked pinned memory:

```python
# Non-blocking transfer overlaps CUDA copy engine with tensor cores
x = x.to(device, non_blocking=True)
y = y.to(device, non_blocking=True)
```

---

## Practical task

Construct and verify the complete training harness:

1. **Parameter grouping:** Implement `configure_optimizers` and verify that all 2D
   weight matrices receive weight decay while 1D bias and norm parameters receive `0.0`.
2. **Step execution:** Build the step function combining gradient accumulation,
   gradient norm clipping, and the cosine schedule.
3. **Recovery test:** Train for $N$ steps, save a checkpoint, instantiate a fresh
   model and optimizer, resume from the checkpoint, and verify that step $N+1$
   computes the exact expected loss.
4. **Failure analysis:** Intentionally place `optimizer.zero_grad()` after `loss.backward()`
   or omit `scaler.unscale_()` to observe the numerical degradation firsthand.

## Expected output

A robust trainer that reports:
- exact parameter partitioning between decayed and non-decayed groups;
- stable gradient norms under accumulation across micro-batches;
- correct learning rate warmup and decay trajectories;
- exact state restoration from saved checkpoints.

---

## Practical companion guide

!!! tip "Practical Lab: Training-Loop Anatomy"
    To inspect and test the components covered in this module:

    👉 **Follow the companion implementation** to verify parameter grouping, step accumulation, and checkpoint round-trip invariants:

    - **Lab script:** [`companion/scripts/02_training_step.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/02_training_step.py)
    - **Optimizer module:** [`companion/minilm/optim.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/optim.py)
    - **Trainer module:** [`companion/minilm/train.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/train.py)
    - **Unit tests:** [`companion/tests/test_module_02_optim.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/tests/test_module_02_optim.py)

    Run the verification suite:
    ```bash
    cd companion
    uv run pytest tests/test_module_02_optim.py -v
    uv run python scripts/02_training_step.py
    ```

---

## References

- [The Annotated PyTorch Training Loop (idlemachines.co.uk)](https://idlemachines.co.uk/essays/pytorch-training-loop)
  — comprehensive line-by-line deconstruction of the training loop and placement hazards.
- [Loshchilov & Hutter (2017) — Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
  — introduces the AdamW formulation and explains why L2 regularization breaks in adaptive methods.
- [PyTorch Documentation — Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
  — execution details of the forward graph, saved tensors, and backward vector-Jacobian passes.
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
  — best practices for `set_to_none=True`, `pin_memory=True`, and mixed-precision execution.
