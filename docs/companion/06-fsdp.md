# Module 6 Lab — FSDP and ZeRO

This lab accompanies [Module 6: FSDP and ZeRO](../modules/06-fsdp.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/06_fsdp_experiment.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/06_fsdp_experiment.py)
    - **Reference module:** [`minilm/fsdp_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/fsdp_utils.py)
    - **Unit tests:** `tests/test_module_06_fsdp.py` *(TODO)*

---

## Objectives

1. **State sharding decomposition**: Understand where memory savings originate across ZeRO-1 (optimizer states), ZeRO-2 (gradients), and ZeRO-3 / FSDP (model weights).
2. **Auto-wrap policy**: Configure wrapping rules for Transformer blocks and observe collective communication (all-gather and reduce-scatter) timing.
3. **Peak memory measurement**: Compare memory consumption between unsharded DDP and sharded FSDP under identical batch configurations.

---

## Quickstart

Run the FSDP memory and communication scaling experiment:

```bash
uv run python scripts/06_fsdp_experiment.py
```
