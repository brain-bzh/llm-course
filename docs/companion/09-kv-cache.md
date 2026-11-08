# Module 9 Lab — Add a KV cache to your model

This lab accompanies [Module 9: KV-cached decoding](../modules/09-kv-cache.md).
Without a cache, generating each new token re-runs the whole prefix. You will
extend your Module 1 attention so that past keys and values are reused, write
the cached generation loop, prove that it produces the same logits and tokens,
and measure prefill and decode separately.

!!! tip "Files used in this lab"
    - **Starter:** [`starter/09_generate.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/09_generate.py)
    - **Your implementations:** `nanolm/model.py` (cache path in attention) · [`nanolm/generate.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/generate.py)
    - **Benchmark:** [`scripts/09_kv_cache_bench.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/09_kv_cache_bench.py) · [`nanolm/kv_cache.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/kv_cache.py)
    - **Tests:** [`tests/test_module_09_kv_cache.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_09_kv_cache.py)

## Before class (15 minutes)

Read [Cached attention, one layer](../modules/09-kv-cache.md#cached-attention-one-layer)
and [Position and mask correctness](../modules/09-kv-cache.md#position-and-mask-correctness).
Then predict, for the benchmark model (2 layers, 2 heads, $d=64$, fp32, batch 1)
with a 64-token prompt and 16 generated tokens:

1. the shape of one layer's K cache after the last decode step;
2. the total cache bytes, $2 \times L \times B \times H \times T \times d_{\text{head}} \times 4$.
   Careful with $T$: the final emitted token is never fed back into the model.

## Set up the exercise

Your Module 1 model returns `(logits, loss, None)` and has no cache. Branch
from your latest lab branch so that your model comes with you:

```bash
git switch -c session-26-kv-cache
cp starter/09_generate.py nanolm/generate.py
uv run pytest tests/test_module_09_kv_cache.py -q
```

## Session 26 protocol — 75 minutes

| Time | Activity | Evidence |
| --- | --- | --- |
| 0–10 min | Predict cache shape and bytes; trace one decode step on paper | Written prediction |
| 10–35 min | Step 1: cache path in attention, positions in the model | Per-position logits test passes |
| 35–50 min | Step 2: `generate_cached` | Greedy tokens identical to uncached |
| 50–65 min | Step 3: benchmark two prompt lengths | Prefill time, decode-interval median, cache bytes vs prediction |
| 65–75 min | Step 4: explain the result and its exclusions | Three-sentence interpretation |

### Step 1 — Make attention reuse past keys and values (25 minutes)

The model contract the tests use is:

```python
logits, loss, caches = model(idx, kv_caches=None, use_cache=True)     # prefill
logits, loss, caches = model(next_token, kv_caches=caches, use_cache=True)  # decode
```

where `caches` is a list with one `(K, V)` pair per layer, each of shape
$B \times H \times T_{\text{seen}} \times d_{\text{head}}$. Make three changes:

1. **Attention.** Add a method `forward_with_cache(x, kv_cache=None, use_cache=False)`
   to `CausalSelfAttention` that returns `(y, new_cache)`. After projecting and
   reshaping the new `k` and `v`, concatenate the past ones along the
   sequence dimension (`dim=2`). Keep `forward(x)` working by calling it and
   discarding the cache.
2. **Mask with an offset.** The $T$ new queries sit at absolute positions
   `past_len … past_len + T - 1`, where `past_len = total_T - T`. Slice the
   causal mask as `mask[:, :, past_len:past_len + T, :total_T]`. With one new
   token this row is all ones, so it is easy to forget. It matters when several
   tokens arrive at once (chunked prefill). If you added the SDPA branch in
   Module 4, take it only when there is no past cache: `is_causal=True` assumes
   queries and keys start at the same position.
3. **Positions and blocks.** In `MiniGPT.forward`, learned positions must start
   at `past_len` (read it from `kv_caches[0][0].size(2)`), not at zero. Give
   `TransformerBlock` a `forward_with_cache` that threads the cache through the
   attention branch, and collect each layer's new cache when `use_cache=True`.

```bash
uv run pytest tests/test_module_09_kv_cache.py -k "every_position or chunks" -q
```

The first test decodes one token at a time and compares the logits with a full
recomputation **at every position**, with manual attention and with SDPA. The
second feeds a prompt in two chunks and fails if the mask offset is missing.

!!! note "If your model is not ready"
    Commit your work, then restore the reference model, which already has this
    cache path, and continue with Step 2:
    `git show "$(cat .course-reference-revision):nanolm/model.py" > nanolm/model.py`.
    Record that you used it. The reference's `forward_with_cache` is also a good
    model answer to compare with after class.

### Step 2 — The cached generation loop (15 minutes)

`generate_uncached` in your new `nanolm/generate.py` is complete; read it first.
Implement `generate_cached`:

1. **Prefill:** run the whole prompt once with `use_cache=True`.
2. **Loop** `max_new_tokens` times: select the next token from the logits of the
   **last** position, exactly as the uncached version does (temperature, top-k,
   argmax when `temperature == 0`), and append it.
3. **Decode:** unless this was the final token, run only the new token
   (shape $B \times 1$) with the cache.

```bash
uv run pytest tests/test_module_09_kv_cache.py -q
```

Why does the last emitted token never need a forward pass? Count how many
times the uncached version processes the first prompt token, and how many
times your cached version does.

### Step 3 — Measure prefill and decode separately (15 minutes)

```bash
uv run python scripts/09_kv_cache_bench.py --prompt-tokens 16 --new-tokens 16
uv run python scripts/09_kv_cache_bench.py --prompt-tokens 64 --new-tokens 16
```

Example on a laptop CPU (yours will differ):

```text
"prompt_tokens": 64,  "new_tokens": 16,
"speedup_factor": 2.12,  "exact_token_match": true,
"model_ttft_ms": 0.333,  "decode_interval_median_ms": 0.162,
"logical_cache_bytes": 80896
```

- Does `logical_cache_bytes` match your prediction? (80,896 B is
  $2 \cdot 2 \cdot 1 \cdot 2 \cdot 79 \cdot 32 \cdot 4$: the prompt plus 15 fed-back tokens.)
- Model TTFT (prefill plus first token selection) grows with the prompt. How
  does the decode interval change between the two runs, and why so little?
- The benchmark warms up, repeats, synchronizes on accelerators, and refuses
  to time anything if cached and uncached tokens differ. Why should timing come
  after correctness?

### Step 4 — Interpret (10 minutes)

Write three sentences: what you measured, on which hardware, and what the
measurement excludes. This benchmark excludes request queueing, tokenization,
networking and batching, and it grows the cache by concatenation, which copies
the whole cache every step. On a GPU, decode for a model this small is
dominated by kernel-launch and Python overhead, not by the memory bandwidth
that dominates large models.

## Definition of done

- all tests in `tests/test_module_09_kv_cache.py` pass with your model and your `generate_cached`;
- benchmark runs at two prompt lengths, with prefill and decode reported separately;
- predicted cache bytes match the reported logical cache bytes;
- you can explain the mask offset and the position offset, and say what fails without each.

**Fallback:** restore the reference model as described above, complete Step 2,
and explain the cache arithmetic. A CPU run shows correctness, not GPU bandwidth behavior.

**Extensions:** replace concatenation with a preallocated cache of size
`block_size` plus a length counter, and compare the decode intervals at a long
prompt. Add grouped-query attention (fewer K/V heads than Q heads) and update
the byte formula. Run the benchmark with `--device cuda`.

!!! success "What to take away"
    - **Caching changes cost, not results.** The cached decoder must reproduce the
      uncached logits at every position. Check that first, then time it.
    - **Prefill and decode are different workloads.** Prefill processes many tokens
      in parallel. Decode processes one token per step and rereads the whole cache.
      Report them separately.
    - **Offsets are where incremental code breaks.** Positions and masks start at
      `past_len`. A one-token test cannot reveal a missing mask offset; a chunked one can.
    - **The cache is model state you pay for in memory.** Its size grows with
      layers × KV heads × sequence length × batch, which is why MQA, GQA and paging exist.
