# Module 4 Lab — Train a Small GPT

This lab accompanies [Module 4: Train a small GPT](../modules/04-small-gpt.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`companion/scripts/04_train_baseline.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/scripts/04_train_baseline.py)
    - **Reference modules:** [`companion/minilm/train.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/train.py) · [`companion/minilm/generate.py`](https://github.com/jonathanlys01/llm-course/blob/main/companion/minilm/generate.py)
    - **Unit tests:** `companion/tests/test_module_04_baseline.py` *(TODO)*

---

## Objectives

1. **Integrate end-to-end components**: Connect the tokenizer, packed dataset pipeline, Transformer model, and optimizer into an executable baseline run.
2. **Monitor training dynamics**: Log validation loss and learning rate decay to identify stability boundaries.
3. **Autoregressive sampling**: Generate text using temperature and top-$k$/top-$p$ nucleus sampling to evaluate qualitative progression.

---

## Quickstart

Launch the baseline training demo:

```bash
cd companion
uv run python scripts/04_train_baseline.py
```
