# Continuous project

This is the shared, in-class laboratory: every team builds the same evolving
language-model system, session after session, module after module. It is
where the course's techniques get implemented, tested and measured — it is
**not** the separately graded team project. That graded work, in which each
team instead verifies one claim from a paper of its choice using the skills
built here, is described on the [Reproduction project](reproduction-project.md)
page.

## Objective

Every team owns one evolving language-model system and answers:

> **How can we build the best end-to-end system under fixed data, training
> compute and serving compute budgets?**

The final result is not merely a checkpoint. It is a reproducible technical
argument supported by code, measurements and controlled comparisons, and the
evidence base the reproduction project and its individual defense draw on.

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
| Ingestion | Verified zero-copy streaming pipeline with lossless tokenization |
| Performance | Before/after profiler evidence, tokens/s and peak memory |
| Distributed training | Equivalence checks, global accounting and scaling measurements |
| Inference | Cached/uncached equivalence plus prefill/decode benchmarks |
| Serving | Declared workload with TTFT, ITL, throughput and memory |

## Experimental rules

1. Change one material factor at a time unless the experiment explicitly tests
   an interaction.
2. Keep data, token count, model, optimizer and evaluation fixed when claiming
   a dataset or preprocessing improvement.
3. Report failed experiments and negative results when they affect the final
   decision.
4. Distinguish wall-clock speed, throughput, utilization and theoretical FLOPs.
5. Preserve the exact configuration and commit associated with every reported
   result.
6. Never call a change an optimization without a measured baseline.

## Synthesis discussion

The course closes with a comparative postmortem on this shared system (see the
final block of the [schedule](schedule.md)), not a separate grade. Each team
should be ready to discuss four decisions:

- which data deserved the training budget;
- where the original training system wasted resources;
- which parallel strategy fit the tested hardware and model;
- what limited serving throughput or latency.

This discussion should also surface the strongest remaining uncertainty and
the next experiment the team would run with additional compute. It is common
material for the individual technical defense in the
[reproduction project](reproduction-project.md), since questions can be drawn
from either the shared system or the team's paper reproduction.

