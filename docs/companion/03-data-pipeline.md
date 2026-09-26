# Module 3 Lab — Pretraining Data Pipeline

This lab accompanies [Module 3: Pretraining data pipeline](../modules/03-data-pipeline.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/03_prepare_dataset.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/03_prepare_dataset.py)
    - **Reference modules:** [`nanolm/tokenizer.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/tokenizer.py) · [`nanolm/data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/data.py)
    - **Unit tests:** [`tests/test_module_03_data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_03_data.py)

---

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

## Session 8 protocol — 75 minutes

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

Use a small local corpus with stable document IDs. Group identical documents
before splitting, and train learned tokenizer merges only on the training
partition. Pack training and validation into separate files. Document the
source and any filtering rule; preserve a few discarded examples for inspection.

**Minimum completion:** lossless tokenizer round-trip, visible document
boundaries, disjoint source-document partitions, and correct shifted batches.
Compression ratio and throughput are observations, not universal thresholds.

**Fallback:** if packing is unfinished, inspect a shard produced by the provided
script and diagnose deliberately shifted slices. Record that this establishes
understanding of the example, not completion of the implementation.

**Optional extension:** implement BPE training or a quality filter from scratch,
then compare retained documents or token counts. Change one component at a time.
