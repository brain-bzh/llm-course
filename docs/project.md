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
minimal GPT & training loop
    ↓
pretraining data pipeline (filtering, BPE, binary shards)
    ↓
documented baseline checkpoint
    ↓
optimized single-GPU trainer (bf16, FlashAttention, compile)
    ↓
distributed scaling (DDP, FSDP, tensor parallelism)
    ↓
multidimensional parallelism & cluster sizing strategy
    ↓
KV-cached decoder
    ↓
continuous-batching served model and benchmark report
```

## Milestones

| Milestone | Required evidence |
| --- | --- |
| Model & loop correctness | GPT-2 logit parity, causal masking, tiny-batch overfit (`loss < 0.1`), and deterministic checkpoint recovery |
| Data pipeline & ingestion | Tokenizer fidelity/compression, contiguous packing with `<\|endoftext\|>`, zero-copy memmap loader throughput |
| Baseline checkpoint | Hyperparameter tuple, loss/grad-norm curves, verified resume determinism, and fluent generated samples |
| Single-GPU performance | Profiler trace comparison, bf16/SDPA/compile speedups, tokens/s, peak VRAM, and MFU calculation |
| Distributed training | Multi-rank synchronization checks, `no_sync` gradient accumulation, FSDP memory savings, and TP numerical equivalence |
| Multidimensional strategy | Cluster sizing identity verification ($G = P \times D \times F \times E_P \times T_P \times C$) and topology hierarchy defense |
| Cached inference | Cached vs uncached logit/token equivalence, prefill vs per-token decode latency, and memory bandwidth analysis |
| Serving engine | Declared dynamic workload with TTFT, ITL, throughput, and memory under continuous batching |

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
