# Module 2 Lab — Make a small NanoLM learn

This lab accompanies [Module 2: Training-loop anatomy and baseline GPT](../modules/02-training-loop.md).
Module 1 established that your architecture can reproduce GPT-2 inference.
Module 2 starts from random weights and establishes that your optimizer and
training loop can make a small version learn.

!!! tip "Files used in this lab"
    - **Starters:** [`starter/02_training/optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/02_training/optim.py) · [`starter/02_training/train.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/02_training/train.py)
    - **Your implementations:** [`nanolm/optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/optim.py) · [`nanolm/train.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/train.py)
    - **Progressive tests:** [`tests/test_module_02_optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_02_optim.py)
    - **Final experiment:** [`scripts/02_overfit.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/02_overfit.py)

The repository keeps completed implementations in `nanolm/` so later modules
run from a fresh clone. Work on a branch and replace them with the Module 2
starters:

```bash
git switch -c session-07-training
cp starter/02_training/optim.py nanolm/optim.py
cp starter/02_training/train.py nanolm/train.py
uv run pytest tests/test_module_02_optim.py -q
```

The tests should initially fail with `NotImplementedError`.

## Session 7 protocol — 75 minutes

- **0–10 min:** build and inspect the shifted batch.
- **10–30 min:** configure AdamW and inspect its parameter groups; implement the schedule.
- **30–60 min:** implement forward, loss normalization, backward, clipping and the update; run the overfit experiment.
- **60–75 min:** explain the update order; list what a checkpoint must contain to resume training exactly (you implement it in Session 8).

Prioritize a working FP32 update and falling fixed-batch loss. Checkpoints are
the first task of Session 8, just before the recovery experiment that needs them.

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
→ clip the global gradient norm
→ optimizer step
→ clear gradients
```

For this exercise use ordinary FP32 execution (`scaler=None`, `dtype=torch.float32`).
The starter exposes optional mixed-precision arguments for later use; preserve the
signatures, but leave mixed precision to [Module 4](04-single-gpu.md).

`train_step` accumulates gradients but does not update parameters.
`optimizer_step` performs exactly one update and clears the gradient buffers
with `set_to_none=True`.

## Step 4 — Overfit one batch

Run the three Session 7 tests, followed by the visible experiment:

```bash
uv run pytest tests/test_module_02_optim.py -q
uv run python scripts/02_overfit.py
```

The script creates a two-layer NanoLM with a 64-token vocabulary and repeatedly
trains on the one shifted sequence from Step 1. Its loss must fall below `0.1`
within 80 updates. This is deliberately not evidence of generalization: it is a
focused check that targets, backward propagation, parameter updates, and the
causal model are connected correctly.

The script prints `PASS:` for the overfit, then reports that the checkpoint
round-trip is not implemented yet. That is expected until Session 8.

## Part B — Checkpoints and the baseline experiment (session 8)

The fixed-batch exercise is the session 7 milestone. Session 8 makes training
recoverable, then runs a separate baseline on disjoint training and validation
documents. The baseline calls your `train_step`, `optimizer_step`,
`save_checkpoint` and `load_checkpoint`.

### Step 5 — Implement checkpoints

Implement `save_checkpoint` and `load_checkpoint` in `nanolm/train.py`. A
checkpoint that can resume training contains:

- model parameters;
- AdamW moments and other optimizer state;
- the model configuration;
- the completed optimizer step and the last recorded loss;
- `extra_state`: the run identity and loss records the baseline passes in;
- the Python, NumPy and PyTorch RNG states, restored only when `restore_rng=True`.

The starter's TODOs name the dictionary keys the tests and the baseline read.

```bash
uv run pytest tests/test_module_02_baseline.py -k round_trip -q
uv run python scripts/02_overfit.py      # now also prints the checkpoint PASS
```

The round-trip loads the checkpoint into a fresh model and requires
bit-identical logits. Reloading only the model weights reproduces inference,
not training: the next update also depends on the AdamW moments, the step that
sets the learning rate, and the random state.

If your checkpoint is not ready after 20 minutes, commit your work, then restore
the reference training module from the revision you recorded in Module 1:

```bash
git show "$(cat .course-reference-revision):nanolm/train.py" > nanolm/train.py
```

Keep your committed
implementation for review, and record when the reference is used.
The baseline uses a fixed UTF-8 byte vocabulary plus EOT (257 tokens), keeping
this experiment independent of tokenizer training and network downloads.

The instructor should prepare the two documented JSONL splits and verify that
each contains more byte tokens than the chosen context length before class.
If the files are unavailable, use the smoke fixture and report the limitation.
Each command below needs a fresh output directory: an existing checkpoint is
protected against accidental overwrite.

First verify the workflow with the small synthetic smoke fixture:

```bash
uv run python scripts/02_train_baseline.py --output runs/smoke --steps 10
uv run pytest tests/test_module_02_baseline.py -q
```

For the course experiment, use two instructor-prepared JSONL files, one document per line:

```json
{"text": "A complete document from the declared source."}
```

Use different source documents in each file, retain the source/provenance record,
and remove duplicates before splitting. The script rejects whitespace-normalized
exact duplicates within or across splits; it does not detect near-duplicate or
semantic contamination. Supply the same files for all compared runs.

```bash
uv run python scripts/02_train_baseline.py \
  --train-jsonl data/train.jsonl --val-jsonl data/validation.jsonl \
  --output runs/baseline --steps 1000 --batch-size 4 --block-size 64
```

This configuration processes $1000 \times 4 \times 64=256000$ training tokens.
It is an example budget, not a guarantee of fluent text. Choose a corpus and
budget with the instructor before making quality comparisons.

### What to inspect

`report.json` records configuration, document hashes, token counts, fixed-window
held-out loss, training loss, gradient norm, learning rate, and code revision.
`checkpoint.pt` stores model/optimizer state, completed step, run identity, and
RNG states. A dirty revision requires preserving the working diff as well as the
commit. The two packed shards remain beside the report.

Validation uses the first full windows of the validation shard, up to
`--eval-batches` (default 4). This is a reproducible small diagnostic; increase
coverage for a final result and state which documents/windows it represents.
Loss is nats per byte-or-EOT token, not directly comparable to GPT-2 BPE loss.

### Recovery experiment

Keep the planned `--steps` unchanged so the learning-rate schedule stays fixed:

```bash
uv run python scripts/02_train_baseline.py --output runs/recovery \
  --steps 10 --stop-after 5
uv run python scripts/02_train_baseline.py --output runs/recovery \
  --steps 10 --resume runs/recovery/checkpoint.pt
```

For custom data, repeat both data-file flags on resume. Configuration or document
identity changes are rejected. Exact CPU continuation is tested against an
uninterrupted run, including final weights and optimizer states. CUDA execution
requires its own deterministic-kernel verification; equivalence across devices
or software releases is not promised.

### Session 8 protocol

- **0–20 min:** Step 5: implement checkpoints; round-trip test and overfit script.
- **20–30 min:** inspect disjoint document hashes, configuration, token budget, and initial loss.
- **30–45 min:** run the smoke test and the interrupt/resume comparison.
- **45–65 min:** launch the agreed baseline, inspect available loss/gradient records, and distinguish train from held-out evidence.
- **65–75 min:** explain one suspected failure and propose a controlled follow-up. Longer training continues between sessions.

**Minimum evidence:** `tests/test_module_02_baseline.py` passes with your
checkpoint functions; declared corpus/split, configuration and token budget,
held-out loss records, saved checkpoint, and recovery test. Fixed-prompt samples
can supplement the loss analysis; fluency is not a passing threshold.

**Fallback:** use the smoke fixture to verify the mechanics and record that
quality on a real corpus remains unmeasured. Avoid presenting smoke loss as a
pretraining result.

## Definition of done — Part A (session 7)

You are finished when:

- the three tests in `tests/test_module_02_optim.py` pass;
- `scripts/02_overfit.py` reaches loss `< 0.1`;
- you can list what a checkpoint must contain to resume training exactly;
- you can explain why loss normalization happens before `backward()`, and why
  clipping happens before `optimizer.step()`.

The first two modules form one continuous result:

```text
Module 1 (sessions 1–4): implement the architecture → reproduce GPT-2 inference
Module 2 (sessions 5–8): implement the training loop → overfit a small NanoLM locally
```

!!! success "What to take away"
    - **Overfit one batch first.** If a model cannot memorize eight tokens, no
      amount of data or compute will fix the loop.
    - **Order is semantics.** Normalize the loss before `backward()`, clip before
      `step()`, step the schedule per update rather than per micro-batch, and clear
      gradients after the update.
    - **A checkpoint is the whole training state.** Weights alone reproduce
      inference; resuming training also needs the optimizer, the step, the RNG
      states and the data position. Prove it by comparing an interrupted run with an uninterrupted one.
    - **Held-out means held out at the document level.** Report validation loss on
      documents the model never trained on, with the token budget and units stated.
