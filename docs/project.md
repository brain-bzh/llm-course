# Continuous project

## Objective

Every team owns one evolving language-model system and answers:

> **How can we build the best end-to-end system under fixed data, training
> compute and serving compute budgets?**

The final result is not merely a checkpoint. It is a reproducible technical
argument supported by code, measurements and controlled comparisons.

## Repository progression

```text
raw documents
    ↓
tokenizer and packed dataset
    ↓
correct minimal GPT
    ↓
baseline checkpoint
    ↓
selected training corpus
    ↓
optimized single-GPU trainer
    ↓
DDP and focused sharding experiments
    ↓
KV-cached decoder
    ↓
served model and benchmark report
```

## Milestones

| Milestone | Required evidence |
| --- | --- |
| Correctness | Tiny-batch overfit, masking tests and deterministic sample inspection |
| Data pipeline | Tokenizer statistics, split/boundary checks and loader throughput |
| Baseline | Configuration, curves, checkpoint and generated samples |
| Selection | Fixed-budget comparison against random selection |
| Performance | Before/after profiler evidence, tokens/s and peak memory |
| Distributed training | Equivalence checks, global accounting and scaling measurements |
| Inference | Cached/uncached equivalence plus prefill/decode benchmarks |
| Serving | Declared workload with TTFT, ITL, throughput and memory |

## Experimental rules

1. Change one material factor at a time unless the experiment explicitly tests
   an interaction.
2. Keep data, token count, model, optimizer and evaluation fixed when claiming
   a data-selection improvement.
3. Report failed experiments and negative results when they affect the final
   decision.
4. Distinguish wall-clock speed, throughput, utilization and theoretical FLOPs.
5. Preserve the exact configuration and commit associated with every reported
   result.
6. Never call a change an optimization without a measured baseline.

## Final technical defense

Each team presents four decisions:

- which data deserved the training budget;
- where the original training system wasted resources;
- which parallel strategy fit the tested hardware and model;
- what limited serving throughput or latency.

The defense must also identify the strongest remaining uncertainty and the next
experiment the team would run with additional compute.

