# Module 2 Lab — Training-Loop Anatomy

This lab accompanies [Module 2: Training-loop anatomy](../modules/02-training-loop.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/02_training_step.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/02_training_step.py)
    - **Reference modules:** [`companion/minilm/optim.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/optim.py) · [`companion/minilm/train.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/train.py)
    - **Unit tests:** [`companion/tests/test_module_02_optim.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/tests/test_module_02_optim.py)

---

## Objectives

1. **Implement training loop primitives**: Track parameter grouping (2D weights with decay, 1D biases/norms without decay), step accumulation, and gradient norm clipping.
2. **Verify accumulation invariance**: Confirm that accumulating $k$ micro-batches of size $B$ matches a single step with batch size $k \times B$.
3. **Verify checkpoint determinism**: Ensure training states can be saved and restored identically.

---

## Quickstart

Run the verification tests and the standalone demonstration script:

```bash
cd companion
uv run pytest tests/test_module_02_optim.py -v
uv run python scripts/02_training_step.py
```
