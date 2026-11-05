# Module 3 Lab — Pretraining Data Pipeline

This lab accompanies [Module 3: Pretraining data pipeline](../modules/03-data-pipeline.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/03_prepare_dataset.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/03_prepare_dataset.py)
    - **Reference modules:** [`nanolm/tokenizer.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/tokenizer.py) · [`nanolm/data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/data.py)
    - **Unit tests:** [`tests/test_module_03_data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_03_data.py)

---

## Connection to the baseline

In session 8 the provided baseline encoded UTF-8 bytes directly, added an
end-of-text token and wrote train and validation shards. Its vocabulary has 257
entries. Here we inspect a learned BPE vocabulary and implement the same
boundary and shifted-batch ideas. The demo is a separate small corpus, not an
automatic replacement for the baseline data: a BPE run needs a matching
vocabulary size and tokenizer. Byte-token and BPE loss per token are not directly
comparable; compare a normalized measure on the same held-out text instead.

## Objectives

1. **Apply heuristic document hygiene**: Filter low-information documents using fast length, alphanumeric ratio, and line repetition rules to conserve GPU FLOPs.
2. **Train a Byte-Pair Encoding (BPE) tokenizer**: Trace merge rules, run the provided trainer, and verify lossless round-trip decoding.
3. **Pack documents contiguously**: Serialize token streams with `<|endoftext|>` delimiters into compact binary `uint16` memory-mapped files.
4. **Stream high-throughput batches**: Inspect the provided loader and measure batch production under declared conditions.

---

## Quickstart

Run the dataset preparation test suite and demo script:

```bash
uv run pytest tests/test_module_03_data.py -v
uv run python scripts/03_prepare_dataset.py
```

## Session 10 protocol — 75 minutes

**Prepare:** read the module sections on BPE, document boundaries, and shifted
batches. Run the quickstart before class. The reference tokenizer, filters, and
memory-mapped loader are provided; rebuilding all three is an optional extension.

| Time | Activity | Evidence |
| --- | --- | --- |
| 0–10 min | Inspect document IDs and a fixed train/validation split; identify an exact duplicate that would cross the split | A disjoint document manifest and explanation of the duplicate policy |
| 10–25 min | Hand-trace two BPE merges, then inspect the reference tokenizer on ASCII and Unicode examples | Decoded text equals the original; report bytes and token counts |
| 25–50 min | Implement packing with an explicit end-of-document token and shifted batch slices | Show the boundary token IDs and a batch with `Y[:, :-1] == X[:, 1:]` |
| 50–65 min | Inspect the provided memmap loader and time repeated batches | Batch size, sequence length, dtype, elapsed time, and tokens/s; no claim about GPU throughput from a CPU-only measurement |
| 65–75 min | Explain one failure: missing delimiter, off-by-one target, or leaked validation document | A corrected example and a short causal explanation |

The quickstart uses a toy corpus and writes a temporary shard; it does not
provide the disjoint manifest for this exercise. The instructor should supply a
small local corpus with stable IDs and a duplicate that crosses a proposed split,
or students can construct one explicitly. Show which IDs were retained in each
partition. Use a small local corpus with stable document IDs. Group identical documents
before splitting, and train learned tokenizer merges only on the training
partition. Pack training and validation into separate files. Document the
source and any filtering rule; preserve a few discarded examples for inspection.

### A small exercise you can run locally

If no course corpus is available, start with the following proposed split. The
first and fourth records have different IDs but identical text. Resolve the
cross-split duplicate before fitting the tokenizer; explain which record you
removed and why. This is a teaching fixture, not a representative corpus.

```python
records = [
    ("train-1", "train", "A small model learns from a sequence of tokens."),
    ("train-2", "train", "A token sequence can cross a document boundary."),
    ("val-1", "validation", "A held-out document measures a different example."),
    ("val-2", "validation", "A small model learns from a sequence of tokens."),
]
```

In a scratch Python file, implement the following steps. Use the provided
`BPETokenizer` from `nanolm.tokenizer` and NumPy; do not modify the reference
loader while you are learning its contract.

1. Group exact duplicate text, choose one partition for each group, and print
   the remaining document IDs in each split. Keep at least one document in
   each split.
2. Train a fresh `BPETokenizer` on the **training text only**, requesting a
   vocabulary size of 280. Print `encode` and `decode` for a held-out Unicode
   example and check the round-trip. The actual vocabulary may be smaller if
   the toy corpus has too few repeated pairs.
3. For each split independently, encode every document, append
   `tokenizer.eot_token_id` after each, and write the IDs with
   `numpy.asarray(ids, dtype=numpy.uint16).tofile(path)`. Print a few IDs around
   each boundary. Check that IDs fit the dtype before writing.
4. Read each shard with `BinaryShardedDataset`. Use `block_size=8` and
   `batch_size=2`, and check `y[:, :-1] == x[:, 1:]`. Look up how the loader
   draws its starting offsets and explain why the last target has no matching
   input inside the same window.

Compare your result with `pack_documents` in the reference implementation.
The delimiter marks a boundary; standard causal attention can still see tokens
from the preceding document unless an additional isolation mask is used.

**Minimum completion:** lossless tokenizer round-trip, visible document
boundaries, disjoint source-document partitions, and correct shifted batches.
Compression ratio and throughput are observations, not universal thresholds.

**Fallback:** if packing is unfinished, inspect a shard produced by the provided
script and diagnose deliberately shifted slices. Record that this establishes
understanding of the example, not completion of the implementation.

**Optional extension:** implement BPE training or a quality filter from scratch,
then compare retained documents or token counts. Change one component at a time.
