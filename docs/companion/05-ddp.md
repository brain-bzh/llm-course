# Module 5 Lab — Distributed Data Parallelism

This lab accompanies [Module 5: Distributed data parallelism](../modules/05-ddp.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/05_train_ddp.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/05_train_ddp.py)
    - **Reference module:** [`nanolm/distributed.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/distributed.py)
    - **Unit tests:** `tests/test_distributed_experiments.py`

---

## Objectives

1. **Multi-process coordination**: Initialize distributed process groups with `torch.distributed` and configure distributed samplers across ranks.
2. **Gradient accumulation with `no_sync()`**: Verify all-reduce suppression during micro-batches to avoid premature communication.
3. **Equivalence check**: Prove that DDP with $R$ ranks across $S$ steps exactly mirrors a single-process run with identical global batch size.

---

## Quickstart

Launch the DDP distributed training simulation:

```bash
uv run torchrun --standalone --nproc_per_node=2 scripts/05_train_ddp.py
uv run pytest tests/test_distributed_experiments.py -q
```

## Session 16 protocol

**Prepare:** derive the effective batch and explain why `no_sync` must enclose
both forward and backward. The provided exercise uses equal-length, unmasked
batches; variable valid-token counts require token-weighted normalization.

- **0–15 min:** predict global tokens/update for two ranks, two accumulated micro-batches, local batch two, and sequence length eight.
- **15–35 min:** run the CPU Gloo command and inspect how each rank receives a disjoint slice of the same global batch.
- **35–55 min:** compare accumulated DDP gradients and SGD updates against a single-process global-batch reference. Repeat with `--accum 1` and `--accum 4`.
- **55–75 min:** design a strong-scaling comparison: fix global tokens/update and model, then choose local batch and accumulation for each device count.

On allocated GPUs, append `--device cuda` to select NCCL. The automated tests
verify two-rank CPU correctness. They do not establish CUDA speedup. This script
also computes a reference model for verification, so do not time its entire run
as a DDP performance benchmark.

**Minimum evidence:** gradient/update equivalence, correct globally reduced loss
and token count, and the intended sample partition. **Fallback:** a single-rank
run plus hand-derived partition; mark multi-rank execution as unverified.

**Scaling extension:** profile a training-only run after correctness passes,
including synchronized warmup/timing, fixed workload, and exposed all-reduce time.
