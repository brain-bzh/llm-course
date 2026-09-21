# Module 13 Lab — Beyond Dense Transformers

This lab accompanies [Module 13: Beyond dense Transformers](../modules/13-beyond-transformers.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/13_moe_exploration.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/13_moe_exploration.py)
    - **Reference module:** [`companion/minilm/moe.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/moe.py)
    - **Unit tests:** [`companion/tests/test_module_13_moe.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/tests/test_module_13_moe.py)

---

## Objectives

1. **Sparse Mixture of Experts (MoE)**: Implement top-$k$ routing gating mechanics, expert dispatch, and auxiliary load-balancing loss.
2. **Compute vs Parameter decoupling**: Quantify how MoE decouples total parameter capacity from FLOPs per forward pass.
3. **State-space alternatives**: Contrast attention compute complexity with linear-time recurrent/state-space models (e.g. Mamba).

---

## Quickstart

Run the MoE unit tests and capacity exploration script:

```bash
cd companion
uv run pytest tests/test_module_13_moe.py -v
uv run python scripts/13_moe_exploration.py
```
