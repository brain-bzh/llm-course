# Module 5 Lab — Write data parallelism by hand

This lab accompanies [Module 5: Distributed data parallelism](../modules/05-ddp.md).
You will implement data parallelism with three collectives, prove that two
CPU processes produce exactly the update of one process on the whole batch,
and then check that PyTorch's `DistributedDataParallel` does the same with
`no_sync` accumulation.

!!! tip "Files used in this lab"
    - **Starter:** [`starter/05_ddp_manual.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/05_ddp_manual.py)
    - **Your implementation:** [`nanolm/ddp_manual.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/ddp_manual.py)
    - **Experiments:** [`scripts/05_manual_ddp.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/05_manual_ddp.py) · [`scripts/05_train_ddp.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/05_train_ddp.py)
    - **Tests:** [`tests/test_module_05_ddp.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_05_ddp.py)
    - **Background:** [Distributed primitives primer](../primers/distributed-primitives.md)

## Why write it yourself?

`DistributedDataParallel` hides three decisions behind one wrapper: every
replica must **start** identical, every replica must see a **different slice**
of the global batch, and every replica must apply the **same averaged gradient**.
Implement each decision once and DDP becomes something you can debug, not just call.

```text
rank 0:  shard 0 ─ forward/backward ─ local grad ─┐
                                                  ├─ all-reduce(SUM) / world ─ same update on every rank
rank 1:  shard 1 ─ forward/backward ─ local grad ─┘
```

## Before class (15 minutes)

1. Read [From accumulation to data parallelism](../modules/05-ddp.md#from-accumulation-to-data-parallelism)
   and [Derive the gradient average](../modules/05-ddp.md#derive-the-gradient-average).
2. Check that two CPU ranks can talk on your machine (see
   [supported platforms](index.md#supported-platforms)):

    ```bash
    uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_train_ddp.py
    ```

    The last line must start with `PASS:`. If it hangs, use the
    [troubleshooting table](index.md#troubleshooting-distributed-runs) before class.

3. Answer on paper: with 2 ranks, 2 accumulated micro-batches per rank, 2
   sequences per micro-batch and 8 tokens per sequence, how many tokens
   contribute to one optimizer update?

## Set up the exercise

```bash
git switch -c session-14-ddp
cp starter/05_ddp_manual.py nanolm/ddp_manual.py
uv run pytest tests/test_module_05_ddp.py -q
```

Six tests fail with `NotImplementedError`. The DDP-library test already passes:
it does not use your file.

## Session 14 protocol — 75 minutes

| Time | Activity | Evidence |
| --- | --- | --- |
| 0–10 min | Predict tokens/update and the number of collectives per update | Written prediction |
| 10–25 min | Step 1: implement `shard_batch` | Two single-process tests pass |
| 25–45 min | Steps 2–3: `broadcast_parameters` and `allreduce_gradients`; run on two ranks | `PASS:` and zero replica drift |
| 45–55 min | Step 4: break it on purpose | Failure explained in one sentence |
| 55–65 min | Step 5: compare with `DistributedDataParallel` and `no_sync` | Library run passes with `--accum 1` and `--accum 4` |
| 65–75 min | Step 6: design a strong-scaling measurement | Table of (devices, micro-batch, accumulation) at fixed tokens/update |

### Step 1 — Give each rank a disjoint slice

Every rank builds the **same** global batch from the same seed and keeps only
its rows. Implement `shard_batch(global_batch, rank, world_size)`:

- reject a batch whose size is not divisible by `world_size` (`ValueError`);
- return rows `[rank * local, (rank + 1) * local)`.

```bash
uv run pytest tests/test_module_05_ddp.py -k shard -q
```

Why reject uneven splits? Each rank computes a *mean* loss over its rows. The
average of per-rank means equals the global mean only when every rank has the
same number of tokens. With variable lengths or padding, weight by valid-token
counts instead.

### Step 2 — Start from identical replicas

Implement `broadcast_parameters(model, src=0)`: for every parameter **and
buffer**, call `dist.broadcast(tensor.data, src=src)`. Return immediately when
there is a single replica (no process group).

The experiment script initializes each rank with a *different* seed on purpose,
so the broadcast is the only thing that makes the replicas identical.

### Step 3 — Average the gradients

Implement `allreduce_gradients(model)`. For each parameter with a gradient,
all-reduce with `SUM`, then divide by the world size. Return the number of
collectives you issued.

```bash
uv run pytest tests/test_module_05_ddp.py -k "not torch_ddp" -q
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_manual_ddp.py
```

Expected output (rank 0, numbers may differ in the last digits):

```text
before broadcast: max replica difference = 1.237e-01
after broadcast:  max replica difference = 0.000e+00
world=2 accum=2 micro_batch=2 seq_len=8 -> global batch 8 sequences = 64 tokens/update
step 1: collectives= 19  sent≈ 109.2 kB/rank (ring estimate)  |param - reference|=7.19e-07  replica drift=0.00e+00
...
PASS: 2 hand-written replicas match one global-batch AdamW run (sync=per-param, accumulation=2).
```

Read the three numbers on each step line:

- **`|param - reference|`** compares your replica with one process trained on
  all 64 tokens. About `1e-6` is floating-point reduction order; `1e-2` is a bug.
- **`replica drift`** must be exactly zero: every rank applied the same update.
- **`collectives`** is one per parameter tensor (19 here). The ring estimate is
  $2\frac{D-1}{D}$ times the gradient bytes.

Notice where communication happens. The script accumulates two micro-batches
**without any communication**, then synchronizes once. That is what `no_sync`
does inside DDP.

### Step 4 — Break it on purpose

Run each failure, predict the output first, then explain what you see:

```bash
# 1. Replicas never synchronized at the start.
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_manual_ddp.py --skip-broadcast
```

2. In your `allreduce_gradients`, temporarily remove the division by the world
   size. Which number changes, and why does AdamW partly hide this bug compared
   with SGD? (Hint: AdamW normalizes the update by the gradient's running RMS.)
3. Make rank 1 skip one parameter in the loop (`if dist.get_rank() == 1 and ...`).
   What happens? Depending on which tensors get paired, you see a size-mismatch
   error or a **hang**: rank 0 waits forever for a collective that rank 1 never
   issues. Stop it with `Ctrl+C`, then restore your code.

### Step 5 — The library version

`DistributedDataParallel` registers hooks that all-reduce gradients in
**buckets** while backward is still running, overlapping communication with
computation. `no_sync()` disables those hooks for the accumulation
micro-steps; the hooks must fire on the last micro-step, during its backward.

```bash
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_train_ddp.py --accum 1
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_train_ddp.py --accum 4
```

Open `scripts/05_train_ddp.py` and find the `no_sync` context. Why must both
the forward **and** the backward of a micro-step be inside it?

**Extension:** implement `allreduce_gradients_flat` (one collective on a
concatenated buffer) and run `--sync flat`. The collective count drops from 19
to 1 while the bytes stay the same. This is why DDP uses buckets: each
collective pays a fixed latency.

### Step 6 — Design a scaling measurement

Correctness passes; the next question is speed. Fix the model and the
**global** tokens per update, then fill in a table for 1, 2, 4 and 8 devices:

| Devices $D$ | Micro-batch $B_\mu$ | Accumulation $A$ | Tokens/update $= D\,A\,B_\mu\,T$ |
| ---: | ---: | ---: | ---: |
| 1 | | | 65,536 |
| 2 | | | 65,536 |

Then state what you would time (training steps only, after warmup, with
`torch.cuda.synchronize()`), which quantity is the scaling efficiency, and
which trace event would show exposed all-reduce time. The 2-rank CPU run
checks correctness only: it shares one machine's memory bandwidth and cannot show speedup.

## Definition of done

- all tests in `tests/test_module_05_ddp.py` pass;
- `05_manual_ddp.py` prints `PASS:` on two ranks with `--accum 2` and `--accum 4`;
- you can explain each of the three failures in Step 4;
- your scaling table keeps tokens/update fixed and names the timing exclusions.

**Fallback:** if two processes cannot communicate on your machine, run
`uv run python scripts/05_manual_ddp.py` (one rank checks the code path), pair
with a neighbour for the two-rank run, and record that your own multi-rank
execution is unverified.

**GPU extension:** on an allocated multi-GPU node, `05_train_ddp.py --device cuda`
selects NCCL. Profile a training-only loop to measure exposed communication;
do not time the reference model the script also trains.

!!! success "What to take away"
    - **Data parallelism is the one-device computation, reorganized.** The global batch,
      the loss normalization and the optimizer are unchanged. Prove that with an
      exact comparison, not a falling loss curve.
    - **Identical start, identical gradient → identical replicas.** Broadcast once,
      average every step. Replica drift should be exactly zero, not small.
    - **Collectives are a contract.** Every rank must issue the same collectives in
      the same order. A mismatch usually hangs instead of raising an error.
    - **Count messages, not just bytes.** Bucketing cuts the number of collectives
      while the volume stays the same. Latency and bandwidth are separate costs.
