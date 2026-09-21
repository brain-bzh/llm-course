# Module 5 Lab — Data Selection and Filtering

This lab accompanies [Module 5: Data selection through the FineWeb case study](../modules/05-data-selection.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/05_data_selection.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/05_data_selection.py)
    - **Reference module:** [`companion/minilm/data_selection.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/data_selection.py)
    - **Unit tests:** `companion/tests/test_module_05_selection.py` *(TODO)*

---

## Objectives

1. **Heuristic filtering**: Implement document length thresholds, symbol-to-word ratios, and language identification heuristics.
2. **MinHash deduplication**: Detect near-duplicate documents using MinHash LSH and compare with exact line/paragraph hashing.
3. **Controlled subset comparison**: Evaluate sample quality under identical token budgets.

---

## Quickstart

Run the data selection comparison pipeline:

```bash
cd companion
uv run python scripts/05_data_selection.py
```
