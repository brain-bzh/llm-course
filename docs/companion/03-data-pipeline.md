# Module 3 Lab — Pretraining Data Pipeline

This lab accompanies [Module 3: Pretraining data pipeline](../modules/03-data-pipeline.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/03_prepare_dataset.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/03_prepare_dataset.py)
    - **Reference modules:** [`minilm/tokenizer.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/tokenizer.py) · [`minilm/data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/data.py)
    - **Unit tests:** [`tests/test_module_03_data.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_03_data.py)

---

## Objectives

1. **Apply heuristic document hygiene**: Filter low-information documents using fast length, alphanumeric ratio, and line repetition rules to conserve GPU FLOPs.
2. **Train a Byte-Pair Encoding (BPE) tokenizer**: Train merge rules from raw character bytes and verify lossless round-trip decoding.
3. **Pack documents contiguously**: Serialize token streams with `<|endoftext|>` delimiters into compact binary `uint16` memory-mapped files.
4. **Stream high-throughput batches**: Build a loader that samples randomized chunks with zero accelerator starvation.

---

## Quickstart

Run the dataset preparation test suite and demo script:

```bash
uv run pytest tests/test_module_03_data.py -v
uv run python scripts/03_prepare_dataset.py
```
