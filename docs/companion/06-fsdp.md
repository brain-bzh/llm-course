# Module 6 Lab — Shard the optimizer by hand, then watch FSDP2 work

This lab accompanies [Module 6: FSDP and ZeRO](../modules/06-fsdp.md). It has
three parts that follow the ZeRO stages:

1. **Predict** the persistent state of DDP, ZeRO-1/2 and ZeRO-3 with the calculator.
2. **Implement** a sharded AdamW (ZeRO-1/2) with reduce-scatter and all-gather,
   and prove that it matches `torch.optim.AdamW` exactly.
3. **Run real FSDP2** (`torch.distributed.fsdp.fully_shard`) on CPU ranks: watch
   parameters gather and reshard around each block, and measure what each rank stores.

The checkpoint-format part of this module runs in
[Session 21](07-tensor-parallelism.md#part-a-checkpoint-formats-25-minutes).

!!! tip "Files used in this lab"
    - **Starter:** [`starter/06_zero.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/starter/06_zero.py)
    - **Your implementation:** [`nanolm/zero.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/zero.py)
    - **Experiments:** [`scripts/06_fsdp_experiment.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/06_fsdp_experiment.py) (calculator) · [`scripts/06_zero_manual.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/06_zero_manual.py) · [`scripts/06_fsdp2.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/06_fsdp2.py)
    - **FSDP2 helpers:** [`nanolm/fsdp_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/fsdp_utils.py)
    - **Tests:** [`tests/test_module_06_zero.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_06_zero.py)

## The idea in one picture

DDP all-reduces gradients, then every rank runs the **same** AdamW update on the
**whole** model, holding every moment. That is $D$ identical copies of the
optimizer state. An all-reduce is a reduce-scatter followed by an all-gather,
so the update can be split in the middle:

```text
           DDP                               ZeRO-1/2 (this lab)
full grad ─ all-reduce ─ full grad      full grad ─ reduce-scatter ─ grad of MY slice
AdamW on everything (full moments)      AdamW on MY slice (1/D of the moments)
                                        my slice ─ all-gather ─ full parameters
```

The total communication volume is the same. Each rank now stores only $1/D$ of
the optimizer state. FSDP (ZeRO-3) goes further and shards the parameters and
gradients too, at the price of gathering each unit's parameters when it runs.

## Before class (15 minutes)

Read [ZeRO as progressive sharding](../modules/06-fsdp.md#zero-as-progressive-sharding)
and [One fully sharded unit, step by step](../modules/06-fsdp.md#one-fully-sharded-unit-step-by-step).
Then compute, for $N$ fp32 parameters trained with AdamW on $D$ ranks, the bytes
per rank for parameters, gradients and optimizer state under DDP, ZeRO-1,
ZeRO-2 and ZeRO-3. Write the four formulas down; you will check them twice.

## Set up the exercise

```bash
git switch -c session-19-zero
cp starter/06_zero.py nanolm/zero.py
uv run pytest tests/test_module_06_zero.py -q
```

The single-rank and two-rank sharded-AdamW tests fail with
`NotImplementedError`; the FSDP2 test already passes because it uses PyTorch's
implementation.

## Session 19 protocol — 75 minutes

| Time | Activity | Evidence |
| --- | --- | --- |
| 0–10 min | Part 1: run the calculator and reconcile it with your formulas | Four formulas checked against one printed row |
| 10–40 min | Part 2: implement the three sharded-update methods; two-rank run | `PASS:`; measured state = $8\lceil N/D\rceil$ bytes |
| 40–60 min | Part 3: run FSDP2; read the lifecycle and the measured bytes | Annotated lifecycle; measured/predicted table |
| 60–75 min | Compare `--keep-after-forward`; answer the transient-memory questions | Two sentences on what peak memory adds to the table |

### Part 1 — Predict (10 minutes)

```bash
uv run python scripts/06_fsdp_experiment.py
```

It prints persistent static memory for a 353M-parameter GPT-2-medium-sized model
(fp32 parameters, fp32 gradients, two fp32 AdamW moments: 16 bytes per parameter):

```text
Strategy     | N=1 (Single) | N=4        | N=8        | N=64
DDP          |    5410.0 MB |  5410.0 MB |  5410.0 MB |  5410.0 MB
ZeRO-1       |    5410.0 MB |  3381.3 MB |  3043.1 MB |  2747.3 MB
ZeRO-2       |    5410.0 MB |  2366.9 MB |  1859.7 MB |  1415.9 MB
FSDP/ZeRO-3  |    5410.0 MB |  1352.5 MB |   676.3 MB |    84.5 MB
```

Reproduce the `ZeRO-1, N=8` cell from your formula. Why does ZeRO-1 stop
improving beyond a certain $D$, while ZeRO-3 keeps shrinking? None of these
numbers includes activations or the temporarily gathered weights.

### Part 2 — Implement a sharded AdamW (30 minutes)

`ShardedAdamW` views all parameters as one flat vector of $N$ elements, padded
with zeros to a multiple of $D$. Rank $r$ owns elements
$[r\,s, (r+1)\,s)$ where $s = \lceil N/D\rceil$. Its `step()` is provided and
calls three methods you implement:

```python
grad_shard = self._reduce_scatter_mean(flat_grad)   # TODO 1
self._adamw_update(param_shard, grad_shard)          # TODO 2
full = self._all_gather(param_shard)                 # TODO 3
```

1. **`_reduce_scatter_mean`**: allocate `shard_size` elements and call
   `dist.reduce_scatter_tensor(out, flat_grad, op=dist.ReduceOp.SUM)`, then
   divide by the world size. With one rank, return a copy.
2. **`_adamw_update`**: decoupled weight decay, then the two moment updates,
   then the bias-corrected step, all in place on your slice. Use
   `self.step_count` as $t$:

    $$\theta \leftarrow \theta(1-\eta\lambda),\quad
    m \leftarrow \beta_1 m + (1-\beta_1)g,\quad
    v \leftarrow \beta_2 v + (1-\beta_2)g^2$$

    $$\theta \leftarrow \theta - \frac{\eta}{1-\beta_1^t}\,
    \frac{m}{\sqrt{v}/\sqrt{1-\beta_2^t} + \epsilon}$$

3. **`_all_gather`**: allocate `shard_size * world_size` elements and call
   `dist.all_gather_into_tensor(full, param_shard)`.

```bash
uv run pytest tests/test_module_06_zero.py -k single_rank -q    # exact AdamW on one rank
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_zero_manual.py
```

Expected (rank 0):

```text
parameters N=27,296  world=2  shard_size=13,648  padding=0
AdamW moments per rank: DDP 218,368 B  ->  sharded 109,184 B (prediction 8 * shard_size = 109,184 B)
step 1: |param - reference|=1.19e-06  replica drift=0.00e+00
...
PASS: 2 ranks with sharded AdamW state match one global-batch AdamW run.
```

Now run three ranks. Why is there one element of padding, and why does it
never become nonzero?

```bash
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=3 scripts/06_zero_manual.py
```

**Extension — clipping becomes a collective.** Gradient clipping needs the norm
of the **whole** gradient, but each rank only holds a slice. Implement `_clip_`:
sum the squares of your slice, all-reduce the sum, take the square root, and
scale. Then run with `--clip 0.5` and compare the printed norm with the
reference. Any per-parameter statistic (norms, NaN checks) needs the same treatment.
Its test is skipped by default; run `uv run pytest tests/test_module_06_zero.py -q --extensions`.

### Part 3 — Run real FSDP2 (20 minutes)

FSDP2 replaces each parameter with a `DTensor` sharded along its first
dimension. `fully_shard(block)` makes each Transformer block a **unit**: its
parameters are all-gathered just before it runs and freed after.

```bash
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_fsdp2.py
```

```text
step 1 lifecycle (rank 0):
  block 0 forward begins: mlp.c_fc.weight is Parameter local (128, 32) of (128, 32)
  block 0 forward ends:   mlp.c_fc.weight is DTensor   local (64, 32) of (128, 32)
  block 1 forward begins: mlp.c_fc.weight is Parameter local (128, 32) of (128, 32)
  block 1 forward ends:   mlp.c_fc.weight is DTensor   local (64, 32) of (128, 32)
  after backward: weight DTensor   local (64, 32) of (128, 32), grad DTensor   local (64, 32) of (128, 32)
step 1: |param - reference| = 1.19e-06
...
persistent state on rank 0 (fp32, AdamW), measured with DTensor.to_local():
                 replicated (DDP)   FSDP2 measured   ratio
  parameters              109,184           54,592    2.00
  gradients               109,184           54,592    2.00
  optimizer               218,368          109,184    2.00
```

Annotate this output:

1. Where did the all-gather happen, and where did the reshard happen?
2. During backward, block 1's weights are needed again, but they were freed.
   What must FSDP do before computing block 1's gradients? This is why ZeRO-3
   gathers each parameter **twice** per step.
3. The gradient is a `DTensor` with half the rows. Which collective produced it?
4. Compare the measured ratio with your Part 1 formula for ZeRO-3.

Then keep gathered weights between forward and backward:

```bash
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/06_fsdp2.py --keep-after-forward
```

The lifecycle now shows `Parameter` after each forward. Which collective did
this remove, and which memory term grew? The persistent-state table does not
change. Explain why it cannot show the difference: it measures state at rest,
not the transient peak.

## Definition of done

- `tests/test_module_06_zero.py` passes with your `nanolm/zero.py`;
- `06_zero_manual.py` passes on 2 and 3 ranks, with measured state equal to $8\lceil N/D\rceil$ bytes;
- an annotated FSDP2 lifecycle that marks every all-gather, reshard and reduce-scatter for one step;
- a table of predicted vs measured per-rank parameter, gradient and optimizer
  bytes, plus the list of terms it **excludes**: gathered unit weights,
  activations, communication buffers, allocator cache.

**Fallback:** if multi-process runs fail on your machine, pass the single-rank
test, run `uv run python scripts/06_zero_manual.py` (one rank), and annotate the
FSDP2 output above. Record that your own multi-rank execution is unverified.

**GPU extension:** on two GPUs, add `torch.cuda.max_memory_allocated()` around
one step of DDP and of FSDP2 at the same batch, and explain the gap between
measured peak and persistent state. A small model may run slower under FSDP:
it pays the extra collectives without needing the memory.

!!! success "What to take away"
    - **Sharding is a choice of who stores what, not new mathematics.** Your sharded
      AdamW and FSDP2 both reproduce one-process AdamW to about `1e-6`.
    - **All-reduce = reduce-scatter + all-gather.** Splitting it lets each rank keep
      only its slice of the optimizer, at no extra communication volume.
    - **ZeRO-3 trades memory for collectives.** Parameters are gathered on demand,
      twice per step, and the unit size sets the transient peak.
    - **Global quantities need collectives under sharding.** Gradient norms,
      clipping and NaN checks must all-reduce partial results.
    - **Persistent state is not peak memory.** Always list what a memory number excludes.
