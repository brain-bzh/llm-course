# Module 2 Lab — Training-Loop Anatomy & Baseline GPT

This lab accompanies [Module 2: Training-loop anatomy and baseline GPT](../modules/02-training-loop.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab scripts:** [`scripts/02_training_step.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/02_training_step.py) · [`scripts/02_train_baseline.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/02_train_baseline.py)
    - **Reference modules:** [`minilm/optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/optim.py) · [`minilm/train.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/train.py) · [`minilm/generate.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/generate.py)
    - **Unit tests:** [`tests/test_module_02_optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_02_optim.py)

---

## Objectives

1. **Implement training loop primitives**: Track parameter grouping (2D weights with decay, 1D biases/norms without decay), step accumulation, and gradient norm clipping.
2. **Verify accumulation invariance**: Confirm that accumulating $k$ micro-batches of size $B$ matches a single step with batch size $k \times B$.
3. **Verify checkpoint determinism**: Ensure model parameters and optimizer states ($m_t, v_t$) can be saved and restored identically.
4. **Integrate end-to-end baseline training**: Launch the baseline run, log validation loss and learning rate decay to identify stability boundaries.
5. **Autoregressive sampling**: Generate text using temperature and top-$k$/top-$p$ nucleus sampling to evaluate qualitative progression.

---

## Quickstart

Run the verification tests and the standalone demonstration scripts:

```bash
cd companion

# 1. Verify training step invariants
uv run pytest tests/test_module_02_optim.py -v
uv run python scripts/02_training_step.py

# 2. Launch baseline training & sampling
uv run python scripts/02_train_baseline.py
```
