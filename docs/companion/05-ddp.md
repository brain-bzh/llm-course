# Module 7 Lab — Distributed Data Parallelism

This lab accompanies [Module 7: Distributed data parallelism](../modules/07-ddp.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/07_train_ddp.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/07_train_ddp.py)
    - **Reference module:** [`companion/minilm/distributed.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/distributed.py)
    - **Unit tests:** `companion/tests/test_module_07_ddp.py` *(TODO)*

---

## Objectives

1. **Multi-process coordination**: Initialize distributed process groups with `torch.distributed` and configure distributed samplers across ranks.
2. **Gradient accumulation with `no_sync()`**: Verify all-reduce suppression during micro-batches to avoid premature communication.
3. **Equivalence check**: Prove that DDP with $R$ ranks across $S$ steps exactly mirrors a single-process run with identical global batch size.

---

## Quickstart

Launch the DDP distributed training simulation:

```bash
cd companion
uv run python scripts/07_train_ddp.py
```
