# Module 5 Lab — Distributed Data Parallelism

This lab accompanies [Module 5: Distributed data parallelism](../modules/05-ddp.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/05_train_ddp.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/05_train_ddp.py)
    - **Reference module:** [`minilm/distributed.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/distributed.py)
    - **Unit tests:** `tests/test_module_05_ddp.py` *(TODO)*

---

## Objectives

1. **Multi-process coordination**: Initialize distributed process groups with `torch.distributed` and configure distributed samplers across ranks.
2. **Gradient accumulation with `no_sync()`**: Verify all-reduce suppression during micro-batches to avoid premature communication.
3. **Equivalence check**: Prove that DDP with $R$ ranks across $S$ steps exactly mirrors a single-process run with identical global batch size.

---

## Quickstart

Launch the DDP distributed training simulation:

```bash
uv run python scripts/05_train_ddp.py
```
