# Module 2 Lab — Make a small NanoLM learn

This lab accompanies [Module 2: Training-loop anatomy and baseline GPT](../modules/02-training-loop.md).
Session 1 established that your architecture can reproduce GPT-2 inference.
Session 2 starts from random weights and establishes that your optimizer and
training loop can make a small version learn.

!!! tip "Files used in this lab"
    - **Starters:** [`starter/02_training/optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/02_training/optim.py) · [`starter/02_training/train.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/02_training/train.py)
    - **Your implementations:** [`nanolm/optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/optim.py) · [`nanolm/train.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/train.py)
    - **Progressive tests:** [`tests/test_module_02_optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_02_optim.py)
    - **Final experiment:** [`scripts/02_overfit.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/02_overfit.py)

The repository keeps completed implementations in `nanolm/` so later modules
run from a fresh clone. Work on a branch and replace them with the Session 2
starters:

```bash
git switch -c session-02-training
cp starter/02_training/optim.py nanolm/optim.py
cp starter/02_training/train.py nanolm/train.py
uv run pytest tests/test_module_02_optim.py -q
```

The tests should initially fail with `NotImplementedError`.

## Step 1 — Build the next-token batch

For a token sequence

```text
[1, 5, 9, 2, 6, 10, 3, 7, 11]
```

the model inputs are every token except the last and the targets are every token
except the first:

```python
inputs = tokens[:, :-1]
targets = tokens[:, 1:]
```

At position `t`, the causal model predicts `targets[:, t]` using only
`inputs[:, :t+1]`. This shift is part of the training pipeline rather than the
Transformer architecture itself.

## Step 2 — Configure AdamW and the learning-rate schedule

Implement `configure_optimizers` in `nanolm/optim.py` with two parameter groups:

- tensors with two or more dimensions receive weight decay;
- biases and normalization vectors receive zero weight decay.

Then implement linear warmup followed by cosine decay in
`get_lr_cosine_schedule`. The schedule advances once per optimizer update, not
once per micro-batch.

```bash
uv run pytest tests/test_module_02_optim.py \
  -k "parameter_grouping or cosine" -q
```

## Step 3 — Implement the state-changing operations in order

Implement `train_step` and `optimizer_step` in `nanolm/train.py`. Preserve this
order:

```text
forward pass
→ divide loss by the accumulation count
→ backward pass
→ unscale AMP gradients, when applicable
→ clip the global gradient norm
→ optimizer step
→ clear gradients
```

`train_step` accumulates gradients but does not update parameters.
`optimizer_step` performs exactly one update and clears the gradient buffers
with `set_to_none=True`.

## Step 4 — Overfit one batch

Run all four Session 2 tests, followed by the visible experiment:

```bash
uv run pytest tests/test_module_02_optim.py -q
uv run python scripts/02_overfit.py
```

The script creates a two-layer NanoLM with a 64-token vocabulary and repeatedly
trains on the one shifted sequence from Step 1. Its loss must fall below `0.1`
within 80 updates. This is deliberately not evidence of generalization: it is a
focused check that targets, backward propagation, parameter updates, and the
causal model are connected correctly.

## Step 5 — Prove checkpoint recovery

Implement `save_checkpoint` and `load_checkpoint`. A useful checkpoint contains:

- model parameters;
- AdamW moments and other optimizer state;
- the model configuration;
- the completed optimizer step;
- the last recorded loss.

The final test and `scripts/02_overfit.py` load the checkpoint into a fresh model
and require bit-identical logits on the fixed input. Reloading only the model
weights is not sufficient for resuming training.

## Optional extension — Move beyond one batch

Once the exit criterion passes, run:

```bash
uv run python scripts/02_train_baseline.py
```

This longer script repeatedly samples from a small synthetic corpus, tracks a
validation estimate, and generates a short continuation. It is an extension,
not the Session 2 definition of done.

## Definition of done

You are finished when:

- all four tests in `tests/test_module_02_optim.py` pass;
- `scripts/02_overfit.py` reaches loss `< 0.1`;
- a fresh model loaded from the saved checkpoint produces identical logits;
- you can explain why loss normalization happens before `backward()`, and why
  clipping happens before `optimizer.step()`.

The first two practical sessions now form one continuous result:

```text
Session 1: implement the architecture → reproduce GPT-2 inference
Session 2: implement the training loop → overfit a small NanoLM locally
```
