# Module 1 Lab — Inspect a Transformer & Reimplement Naive MHA

This lab accompanies [Module 1: Transformer from first principles](../modules/01-transformer.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/01_inspect_and_mha.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/01_inspect_and_mha.py) · [`companion/scripts/01_overfit.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/01_overfit.py)
    - **Reference module:** [`companion/minilm/model.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/model.py)
    - **Unit tests:** [`companion/tests/test_module_01_model.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/tests/test_module_01_model.py)

---

## Objective

1. **Inspect a reference Transformer**: Trace parameter names, tensor shapes, and state dictionaries to understand how weights map to conceptual architecture components.
2. **Reimplement a naive Multi-Head Attention (MHA) layer from scratch**: Write the Query, Key, Value projections, head splitting, causal masking, scaled dot-product attention, and output projection without helper libraries.
3. **Verify invariants**: Confirm that:
   - Output tensors maintain the expected shape $(B, T, d_{\text{model}})$.
   - Causality strictly holds (future tokens cannot alter previous token outputs).
   - The layer can overfit a tiny batch of data.

---

## Part 1 — Inspecting a Reference Transformer

Before implementing individual layers, inspect the complete model structure in Python:

```python
import torch
from minilm.model import MiniGPT, GPTConfig

config = GPTConfig(
    vocab_size=50257,
    block_size=256,
    n_layer=4,
    n_head=4,
    n_embd=128,
)
model = MiniGPT(config)
```

### Parameter Dictionary Inspection

Run through all parameters in `model.named_parameters()` and observe the naming convention:

```python
for name, param in model.named_parameters():
    print(f"{name:<35} | Shape: {str(list(param.shape)):<18} | Count: {param.numel():,}")
```

Notice the key structural patterns:
- `transformer.wte.weight`: Token embedding matrix of shape $[V, d_{\text{model}}]$.
- `transformer.wpe.weight`: Learned position embeddings of shape $[\text{block\_size}, d_{\text{model}}]$.
- `transformer.h.0.attn.c_attn.weight`: Single packed linear projection for $Q, K, V$, having shape $[3 \cdot d_{\text{model}}, d_{\text{model}}]$.
- `transformer.h.0.attn.c_proj.weight`: Output projection back to the residual stream, shape $[d_{\text{model}}, d_{\text{model}}]$.
- `transformer.h.0.mlp.c_fc.weight`: MLP expansion projection, shape $[4 \cdot d_{\text{model}}, d_{\text{model}}]$.
- `transformer.h.0.mlp.c_proj.weight`: MLP contraction projection, shape $[d_{\text{model}}, 4 \cdot d_{\text{model}}]$.
- `lm_head.weight`: Shares underlying storage with `transformer.wte.weight` (weight tying).

### Trace a Forward Pass

Pass a batch of synthetic token IDs and verify how shapes transform:

```python
B, T = 2, 8  # Batch size 2, Sequence length 8
idx = torch.randint(0, config.vocab_size, (B, T))

# Token IDs -> Embeddings
tok_emb = model.transformer.wte(idx)                     # [B, T, d_model]
pos = torch.arange(0, T, dtype=torch.long)
pos_emb = model.transformer.wpe(pos)                     # [T, d_model]
x = tok_emb + pos_emb                                    # [B, T, d_model]

print(f"Hidden state shape entering Layer 0: {list(x.shape)}")
```

---

## Part 2 — Reimplementing Naive Multi-Head Attention

Now, implement a naive multi-head attention module step-by-step.

### 1. The Mathematical Flow

Given input representations $X \in \mathbb{R}^{B \times T \times d_{\text{model}}}$:

1. **Linear Projections**:

    $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

    where $W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

2. **Reshape & Transpose into Multiple Heads**:

    $$B \times T \times d_{\text{model}} \;\longrightarrow\; B \times n_{\text{head}} \times T \times d_{\text{head}}$$

    where $d_{\text{head}} = d_{\text{model}} / n_{\text{head}}$.

3. **Scaled Dot-Product Attention**:

    $$S = \frac{Q K^T}{\sqrt{d_{\text{head}}}} \in \mathbb{R}^{B \times n_{\text{head}} \times T \times T}$$

4. **Causal Masking**:

    $$S_{i, j} = \begin{cases} S_{i, j} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

5. **Softmax & Value Mixing**:

    $$P = \text{softmax}(S, \text{dim}=-1), \quad O_{\text{heads}} = P V \in \mathbb{R}^{B \times n_{\text{head}} \times T \times d_{\text{head}}}$$

6. **Concatenation & Output Projection**:

    $$O_{\text{heads}} \;\longrightarrow\; O_{\text{concat}} \in \mathbb{R}^{B \times T \times d_{\text{model}}}$$

    $$Y = O_{\text{concat}} W_O$$

### 2. Implementation Template

Here is the naive, fully explicit implementation:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NaiveMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, block_size: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # 1. Individual linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # 2. Output projection
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # 3. Lower-triangular causal mask buffer
        # Shape: [1, 1, block_size, block_size]
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Step 1: Linear projections
        # Shape: [B, T, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Step 2: Split into heads and transpose
        # Shape: [B, n_head, T, d_head]
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # Step 3: Scaled dot-product scores (QK^T / sqrt(d_head))
        # Shape: [B, n_head, T, T]
        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))

        # Step 4: Apply causal mask
        causal_mask = self.mask[:, :, :T, :T] == 0
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Step 5: Softmax and weighted value aggregation
        attn_weights = F.softmax(scores, dim=-1)
        out_heads = attn_weights @ v  # [B, n_head, T, d_head]

        # Step 6: Concatenate heads and project output
        # [B, n_head, T, d_head] -> [B, T, n_head, d_head] -> [B, T, d_model]
        out_concat = out_heads.transpose(1, 2).contiguous().view(B, T, C)
        output = self.w_o(out_concat)

        return output
```

---

## Part 3 — Verification & Invariant Checks

Run the automated checks to prove the implementation is correct.

### Check 1: Shape Preservation

```python
mha = NaiveMultiHeadAttention(d_model=64, n_head=4, block_size=32)
x = torch.randn(2, 16, 64)
y = mha(x)

assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"
print("✓ Shape check passed:", y.shape)
```

### Check 2: Causal Masking Invariance Test

A future token must never change the representation or prediction of an earlier token:

```python
mha.eval()
seq1 = torch.randn(1, 6, 64)
seq2 = seq1.clone()
# Modify the last two tokens of seq2
seq2[:, 4:, :] = torch.randn(1, 2, 64)

with torch.no_grad():
    out1 = mha(seq1)
    out2 = mha(seq2)

# Prefix outputs (positions 0, 1, 2, 3) must be IDENTICAL
diff_prefix = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
assert diff_prefix < 1e-6, f"Causality leak detected! Difference: {diff_prefix}"
print(f"✓ Causal invariance verified! Max prefix diff: {diff_prefix:.2e}")
```

### Check 3: Tiny-Batch Overfitting Test (Exit Criterion)

Verify that when wrapped with a linear classification head, the attention module can overfit a tiny batch of random targets:

```bash
cd companion
uv run python scripts/01_inspect_and_mha.py
```

Expected output:
```text
=== Module 1: Inspect Transformer & Naive MHA ===
[Part 1] Model parameter inventory:
  transformer.wte.weight              | Shape: [50257, 128]
  transformer.wpe.weight              | Shape: [256, 128]
  transformer.h.0.attn.c_attn.weight   | Shape: [384, 128]
  ...
[Part 2] Naive MHA Shape check: Passed ([2, 16, 64])
[Part 3] Causal invariance check: Passed (diff: 0.00e+00)
[Part 4] Tiny-batch overfit:
  Step 10 | Loss: 2.2715
  Step 30 | Loss: 0.0806
  Step 60 | Loss: 0.0135
SUCCESS: Module 1 exit criterion satisfied!
```
