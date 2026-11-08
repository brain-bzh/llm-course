# Module 3 Lab — Build the data path from documents to batches

This lab accompanies [Module 3: Pretraining data pipeline](../modules/03-data-pipeline.md).
You will implement the three steps between a pile of documents and a training
batch where silent bugs are most common: **splitting without leakage**,
**packing with document boundaries**, and **slicing shifted batches** from a
memory-mapped shard. The tokenizer, filters and loader constructor are provided.

!!! tip "Files used in this lab"
    - **Starter:** [`starter/03_data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/03_data.py)
    - **Your implementation:** [`nanolm/data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/data.py)
    - **Provided tokenizer:** [`nanolm/tokenizer.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/tokenizer.py) (byte-level BPE, trainable)
    - **Demo:** [`scripts/03_prepare_dataset.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/03_prepare_dataset.py)
    - **Tests:** [`tests/test_module_03_data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_03_data.py)

## Connection to the baseline

In Session 8 the baseline encoded UTF-8 bytes directly (257 symbols with the
end-of-text token) and wrote train and validation shards. This lab adds a
learned BPE vocabulary and implements the same boundary and shifted-batch ideas
yourself. Byte-level and BPE losses per token are **not** comparable: a BPE
token covers several bytes. Compare models on the same held-out text with a
normalized measure such as bits per byte.

## Before class (15 minutes)

Read [Byte-Pair Encoding from first principles](../modules/03-data-pipeline.md#byte-pair-encoding-bpe-from-first-principles)
and [Document boundaries & sequence packing](../modules/03-data-pipeline.md#document-boundaries-sequence-packing).
On paper, apply two BPE merges to `"low lower lowest"`: count adjacent byte
pairs, merge the most frequent, and repeat once.

```bash
git switch -c session-10-data
cp starter/03_data.py nanolm/data.py
uv run pytest tests/test_module_03_data.py -q   # 5 fail with NotImplementedError; BPE tests pass
```

## Session 10 protocol — 75 minutes

| Time | Activity | Evidence |
| --- | --- | --- |
| 0–10 min | Check the hand-traced merges against the provided tokenizer | Learned merges and a lossless Unicode round-trip |
| 10–25 min | Step 1: `assign_splits` (deduplicate before splitting) | Split test passes; list of dropped IDs |
| 25–45 min | Step 2: `pack_documents` with an end-of-text token after every document | Pack tests pass; printed IDs around a boundary |
| 45–60 min | Step 3: `get_batch` offsets and shifted targets | Batch test passes; largest valid offset stated |
| 60–75 min | Step 4: run the pipeline end to end; diagnose one deliberate bug | Corrected example with a causal explanation |

### Step 0 — Check your merges (10 minutes)

```python
# scratch.py — run with: uv run python scratch.py
from nanolm.tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train("low lower lowest", vocab_size=259)  # 256 bytes + EOT + 2 merges
for (a, b), new in tok.merges.items():
    print(tok.vocab[a], "+", tok.vocab[b], "->", tok.vocab[new])
text = "lowest café 🚀"
ids = tok.encode(text)
print(len(text.encode()), "bytes ->", len(ids), "tokens; round-trip:", tok.decode(ids) == text)
```

Do the merges match your paper trace? Why can a byte-level tokenizer encode
`🚀` even though it never saw it during training?

### Step 1 — Deduplicate, then split (15 minutes)

`assign_splits(records)` receives `(doc_id, proposed_split, text)` tuples in a
fixed order. The policy is: normalize whitespace (`normalize_text` is
provided), keep the **first** record of each identical text in its proposed
split, and drop every later copy, wherever it was proposed. Return the splits and the
`(dropped_id, kept_id)` pairs.

```bash
uv run pytest tests/test_module_03_data.py -k split -q
```

The test fixture has a validation document that repeats a training document,
up to whitespace. Without this step, part of your "held-out" loss would be measured on
training text. Why must deduplication also happen **before** the tokenizer is
trained on the training split? (Hint: what would the tokenizer's merges reveal?)
Exact matching misses near-duplicates; `nanolm/data_selection.py` has a
MinHash version for the extension.

### Step 2 — Pack documents with boundaries (20 minutes)

Implement `pack_documents(documents, tokenizer, path, eot_token_id, dtype)`:

1. encode every document and append `eot_token_id` after **each** one, including the last;
2. raise `ValueError` if any ID does not fit the dtype (`np.iinfo(dtype)`);
   a GPT-2 vocabulary fits `uint16`, while a 100k-token vocabulary does not;
3. create the parent directory, write with
   `np.asarray(ids, dtype=dtype).tofile(path)`, and return the token count.

```bash
uv run pytest tests/test_module_03_data.py -k pack -q
```

Print the IDs around one boundary. The delimiter tells the model where a
document ends; it does **not** stop attention from looking across it. A window
that spans two documents still lets the second attend to the first unless you
add a document mask. Which choice does GPT-2-style pretraining make, and why
is it usually acceptable?

### Step 3 — Shifted batches from a memory map (15 minutes)

`BinaryShardedDataset` maps the shard with `np.memmap`: the file is not read
into memory; the OS pages in what you slice. Implement `get_batch`:

- draw `batch_size` start offsets `i` so that the **target** window still fits:
  `x = data[i : i + block_size]`, `y = data[i + 1 : i + 1 + block_size]`;
- stack as `int64` tensors and move them to `device`.

```bash
uv run pytest tests/test_module_03_data.py -k batch -q
```

The offset test writes the shard `0, 1, …, 19` and checks that `y == x + 1`,
that offset `0` is reachable, and that the largest offset is exactly
`num_tokens - block_size - 1`. Explain the `- 1`. What error would one more
position cause, and which error would be *silent*?

### Step 4 — End to end, then break it (15 minutes)

```bash
uv run pytest tests/test_module_03_data.py -q
uv run python scripts/03_prepare_dataset.py
```

Pick one bug, introduce it in your copy, predict the symptom, and check:

| Bug | What to look for |
| --- | --- |
| No end-of-text token after the last document | Token count off by one; the next shard's first document is glued to this one |
| `y = data[i : i + block_size]` (no shift) | The model learns to copy its input. Loss collapses on train *and* validation, yet generation is useless |
| Offsets drawn up to `num_tokens - block_size` | Rare short target window: a shape error, or silent padding in other loaders |
| Validation copy kept | Validation loss looks better than it should, by an amount that depends on the overlap |

Restore your implementation and write the causal explanation in your log.

## Definition of done

- all tests in `tests/test_module_03_data.py` pass with your `nanolm/data.py`;
- a split report: kept IDs per split and dropped duplicates;
- printed token IDs around a document boundary;
- you can state the largest valid batch offset and explain one deliberate bug's symptom.

Compression ratio and loader throughput are observations for your corpus, not
pass/fail thresholds. A CPU timing of the loader says nothing about GPU input stalls.

**Fallback:** if packing is unfinished, inspect a shard produced by the demo
script and diagnose deliberately shifted slices by hand; record that the
implementation is incomplete.

**Extensions:** train the tokenizer on the training split only, then compare
tokens per byte on validation text; implement a quality filter and keep a few
rejected documents for inspection; use `minhash_deduplicate` from
`nanolm/data_selection.py` and count near-duplicates that exact matching missed.

!!! success "What to take away"
    - **Leakage happens before training.** Deduplicate, then split, then fit
      anything learned (tokenizer, filters) on the training split only.
    - **Boundaries are data.** An end-of-text token marks where a document ends;
      it does not isolate attention. Know which one your model needs.
    - **Off-by-one bugs rarely crash.** A missing shift makes the loss look
      excellent. Test invariants (`y[:, :-1] == x[:, 1:]`, offset ranges) instead
      of trusting the loss curve.
    - **Units matter when comparing tokenizers.** Loss per token depends on the
      vocabulary; compare models in bits per byte on the same text.
