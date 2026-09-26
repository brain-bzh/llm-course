# Module 1 Lab — Build a decoder-only Transformer

This lab accompanies [Module 1: Transformer from first principles](../modules/01-transformer.md).
You will implement the model described there: first explicit causal multi-head
attention, then a Transformer block, and finally a complete decoder-only
language model.

!!! tip "Files used in this lab"
    - **Starter:** [`starter/01_transformer.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/01_transformer.py)
    - **Your implementation:** [`nanolm/model.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/model.py)
    - **Progressive tests:** [`tests/test_module_01_model.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_01_model.py)
    - **Provided weight converter:** [`nanolm/gpt2.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/gpt2.py)
    - **GPT-2 parity check:** [`scripts/01_gpt2_parity.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/01_gpt2_parity.py)
    - **Interactive visualization:** [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (live browser visualization of GPT-2 tensors and attention)

The repository contains a completed `nanolm/model.py` so that later modules run
from a fresh clone. For this lab, work on a branch and replace it with the
starter. Your implementation will then become the model you extend throughout
the course.

## Set up the exercise

From the root of your companion clone:

```bash
git switch -c session-01-transformer
cp starter/01_transformer.py nanolm/model.py
uv run pytest tests/test_module_01_model.py -q
```

The tests should initially fail with `NotImplementedError`. This is expected:
the starter defines the required public classes and method signatures, but not
their implementation.

Do not use `torch.nn.MultiheadAttention` or
`torch.nn.functional.scaled_dot_product_attention` in this first version. The
point is to make every projection, reshape, mask, and matrix multiplication
visible. A later module replaces this explicit implementation with optimized
attention.

## Step 1 — Implement causal multi-head attention

Implement `CausalSelfAttention` in `nanolm/model.py`.

For an input `x` with shape `B × T × d_model`:

1. Project `x` independently into `Q`, `K`, and `V`.
2. Reshape each projection to `B × n_head × T × d_head`, where
   `d_head = d_model / n_head`.
3. Compute the attention scores:

   $$S = \frac{QK^\top}{\sqrt{d_{\text{head}}}}.$$

4. Before the softmax, replace scores above the causal diagonal with
   $-\infty$.
5. Apply the softmax over key positions and multiply by `V`.
6. Join the heads back into `B × T × d_model` and apply the output projection.

Register the causal mask as a buffer so that it follows the module between
devices without becoming a trainable parameter.

Keep the projection names from the starter: `w_q`, `w_k`, `w_v`, and `c_proj`.
The provided checkpoint converter relies on this public structure when it
splits GPT-2's fused QKV tensor.

Run only the attention tests while iterating:

```bash
uv run pytest tests/test_module_01_model.py -k attention -q
```

Both tests must pass: attention must preserve the input shape, and changing
future input vectors must not change earlier outputs.

## Step 2 — Build the Transformer block

Implement `MLP` with the following data flow:

```text
d_model → 4 × d_model → GELU → d_model → dropout
```

Then implement `TransformerBlock` using the pre-LayerNorm residual convention
from the module:

```text
x = x + attention(norm_1(x))
x = x + mlp(norm_2(x))
```

The block must preserve `B × T × d_model` exactly.
Keep the component names `ln_1`, `attn`, `ln_2`, and `mlp`; the GPT-2 converter
uses those names to place the pretrained tensors.

```bash
uv run pytest tests/test_module_01_model.py -k transformer_block -q
```

## Step 3 — Assemble the decoder-only language model

Implement `MiniGPT` from the same components:

```text
token IDs
  → token embeddings + learned position embeddings
  → dropout
  → repeated Transformer blocks
  → final LayerNorm
  → vocabulary projection
  → logits
```

Keep these contracts unchanged:

- Input token IDs have shape `B × T` and must satisfy `T <= block_size`.
- Logits have shape `B × T × vocab_size`.
- Tie `lm_head.weight` to the token embedding weight.
- When `targets` are supplied, compute cross-entropy over all positions.
  The caller provides already-shifted next-token targets.
- Return `(logits, loss, None)`. The third value is reserved for the KV cache
  introduced later in the course.
- Store `wte`, `wpe`, `drop`, `h`, and `ln_f` inside `self.transformer`, and
  call the output projection `lm_head`. These names form the checkpoint-loading
  contract.

Run the model-level tests:

```bash
uv run pytest tests/test_module_01_model.py \
  -k "model_output or model_is_causal" -q
```

The causality test changes future token IDs and checks that earlier logits are
unchanged. Passing a shape test alone is not sufficient.

## Step 4 — Reproduce the official GPT-2 forward pass

First run the complete local Module 1 suite:

```bash
uv run pytest tests/test_module_01_model.py -q
```

Then install the optional reference dependency and run the capstone:

```bash
uv sync --extra gpt2
uv run --extra gpt2 python scripts/01_gpt2_parity.py
```

The first run downloads the official GPT-2 124M checkpoint. The provided
`nanolm/gpt2.py` converter handles two bookkeeping differences that are not the
focus of this lab:

- GPT-2 stores Q, K, and V in one fused projection, whereas your implementation
  keeps them separate;
- Hugging Face GPT-2 uses `Conv1D` tensors whose matrix axes must be transposed
  when copied into `nn.Linear`.

The script runs `"The capital of Germany is Berlin. The capital of France is"`
through both models. It compares every output logit, reports the
log-probability of the one-token continuation `" Paris"`, and prints the five
most likely next tokens. The maximum absolute logit difference must remain
within the script's numerical tolerance. On the pinned reference environment,
`" Paris"` is the top prediction with log-probability approximately `-0.334223`.

If parity fails, inspect the implementation in this order:

1. causal-mask orientation and placement before softmax;
2. head reshaping and transposition;
3. residual additions and pre-LayerNorm order;
4. GELU approximation and LayerNorm epsilon;
5. component names and projection dimensions expected by the converter.

## Definition of done

You are finished when:

- all six tests in `tests/test_module_01_model.py` pass;
- NanoLM reproduces the Hugging Face GPT-2 logits within tolerance;
- you can report and interpret the displayed log-probability of `" Paris"`;
- you can state the shape of `Q`, `K`, `V`, attention scores, hidden states, and
  logits without running the code;
- you can explain why changing a future token cannot alter an earlier logit.

Commit your implementation before moving to the training-loop module. From
this point onward, the rest of the companion assumes that `nanolm/model.py` is
your working Transformer implementation. Module 2 will initialize a much
smaller version from scratch and prove that it can learn by overfitting one
batch.
