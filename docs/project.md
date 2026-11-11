# Continuous project

The continuous project is the course laboratory. Every student builds the same
small language-model system, one practical session at a time, in their own
copy of the companion repository. By the end it trains, resumes, reads its own
data, runs on several processes, shards its optimizer, splits its layers and
generates with a KV cache, and every one of those steps has been checked
against a simpler version that you trust.

It is **not** a separately graded component. It is where the skills are built,
and the evidence base that the graded
[reproduction project](reproduction-project.md) and its individual defense draw
on (see [Assessment](assessment.md#what-the-continuous-project-is-graded-on)).

| Page | Role |
| --- | --- |
| [Module pages](modules/index.md) | Concepts, bottlenecks and derivations |
| [Companion guide](companion/index.md) | Step-by-step protocol for each practical session |
| This page | What the lab builds over the course, and what counts as finished |
| [Reproduction project](reproduction-project.md) | The graded team project that reuses this toolkit |

## The question behind the lab

> **Can you build each layer of a language-model system, prove that it
> computes what the simple version computes, and say what it costs?**

Production systems are layered: a model, a training loop, a data path, a fast
kernel, then data, optimizer and tensor sharding, then a cache for inference
and a scheduler for serving. Each layer is introduced for a cost reason and
each one is a new place for a silent bug. The lab adds them in that order, and
every addition must pass the same test: **same numbers as before, then a
measurement of what changed.**

## The companion repository

All labs run in the public companion repository
[`brain-bzh/llm-course-companion`](https://github.com/brain-bzh/llm-course-companion).
Fork or clone it once, in Session 2, and keep working in that clone for the
whole course.

```text
llm-course-companion/
├── nanolm/     complete, tested reference implementation (one module per topic)
├── starter/    the same modules with the core logic left as TODOs
├── scripts/    one runnable experiment per lab, printing PASS: or a measurement
└── tests/      the checks for each milestone (tests/test_module_XX_*.py)
```

In a practical session you **replace one reference module with its starter**,
implement it until its tests pass, then run the lab script. Because `nanolm/`
always contains a working reference, an unfinished lab never blocks the next
one: you can restore the reference for that file and record that you did.

Everything required runs on a laptop CPU, including the multi-process labs,
which use two or three CPU processes connected by PyTorch's Gloo backend. GPUs
are used only for optional measurement extensions. Linux and macOS are
supported. On Windows, use WSL2 (Ubuntu) or the school's Linux machines,
because `torchrun` does not run on native Windows. Check your setup before the
first distributed session as described in
[Supported platforms](companion/index.md#supported-platforms).

## What you build

Each row is a practical session. "Your file" is the module you replace with a
starter and implement; it stays yours for the rest of the course.

| Session | Your file | What it adds to the system | Proven against | Guide |
| :---: | --- | --- | --- | --- |
| 2–4 | `nanolm/model.py` | Explicit GPT: embeddings, causal multi-head attention, Pre-LN blocks, tied head | Official GPT-2 logits | [Module 1](companion/01-transformer.md) |
| 7 | `nanolm/optim.py`, `nanolm/train.py` | AdamW groups, schedule, ordered update | One-batch overfit | [Module 2](companion/02-training-loop.md) |
| 8 | checkpoints in `nanolm/train.py` | Full training state on disk; baseline on held-out documents; interrupt and resume | Identical logits after reload; uninterrupted run | [Module 2, part B](companion/02-training-loop.md#part-b-checkpoints-and-the-baseline-experiment-session-8) |
| 10 | `nanolm/data.py` | Dedupe-then-split, packing with end-of-text, shifted memmap batches | Boundary and offset invariants | [Module 3](companion/03-data-pipeline.md) |
| 12 | `nanolm/profile_utils.py`, SDPA path in `model.py` | FLOP and memory accounting; fused attention | Your explicit attention; your own predictions | [Module 4](companion/04-single-gpu.md) |
| 14 | `nanolm/ddp_manual.py` | Data parallelism by hand: shard, broadcast, all-reduce | One process on the global batch | [Module 5](companion/05-ddp.md) |
| 19 | `nanolm/zero.py` | Sharded AdamW: reduce-scatter, local update, all-gather; FSDP2 | `torch.optim.AdamW`; predicted per-rank bytes | [Module 6](companion/06-fsdp.md) |
| 21 | `nanolm/tensor_parallel.py` | Column/row-parallel layers and their autograd collectives; sharded checkpoints | Dense MLP; resharded reload | [Module 7](companion/07-tensor-parallelism.md) |
| 23 | `scripts/08_parallelism_layout.py` | A multidimensional rank layout (DP, TP, PP, CP, EP) | Device count, groups, state and bubble estimates | [Module 8](companion/08-context-pipeline.md) |
| 26 | KV path in `model.py` (pasted scaffold), `nanolm/generate.py` | KV-cached attention and generation | Uncached logits and greedy tokens | [Module 9](companion/09-kv-cache.md) |
| 28 | (demonstration) | Serving: continuous batching, paged KV, preemption | Simulated engine; live engine if available | [Module 10](companion/10-serving.md) |

Your model file is the thread through the course. It is written explicitly in
Sessions 2–4, gains an SDPA branch in Session 12 that must match the explicit
path, and gains a KV cache in Session 26 that must match the uncached model.
Sessions 14, 19 and 21 wrap the same model three ways (replicated, optimizer
sharded, layers split) and each must reproduce the single-process result.

## Milestones

A milestone is reached when you have its **minimum evidence**. Each guide's
*Definition of done* gives the exact list; this table is the summary.

| Milestone | Sessions | Minimum evidence |
| --- | :---: | --- |
| Model correctness | 2–4 | Module 1 tests pass; GPT-2 logits reproduced within tolerance; causality explained by a test that changes the future |
| Loop correctness | 7 | Session 7 tests pass; fixed batch reaches loss `< 0.1`; update order explained |
| Checkpoint and baseline | 8 | Session 8 tests pass (reloaded model gives identical logits); declared corpus and split, configuration and token budget; held-out loss; saved checkpoint; interrupted run matches the uninterrupted one |
| Data path | 10 | Module 3 tests pass; split report (kept and dropped documents); token IDs printed around a boundary; one deliberate bug diagnosed |
| Predict, then measure | 12 | Accounting tests pass; SDPA matches explicit attention; one warmed-up, repeated ablation with hardware and exclusions; memory prediction vs measurement (report finished after class) |
| Data parallelism | 14 | Module 5 tests pass; `05_manual_ddp.py` prints `PASS:` on two ranks; three deliberate failures explained; scaling table at fixed tokens per update (after class) |
| Sharding | 19 | Module 6 tests pass; `06_zero_manual.py` passes on 2 and 3 ranks with measured state equal to the prediction; annotated FSDP2 lifecycle |
| Tensor parallelism | 21 | Checkpoint formats script prints `PASS:`; Module 7 tests pass; two-rank TP prints `PASS:`; broken backward collective explained; shape ledger |
| Layout | 23 | Two valid layouts at the same device count and tokens per update; state and bubble estimates with exclusions; one layout defended |
| Cached decoding | 26 | Module 9 tests pass; prefill and decode timed separately at two prompt lengths; predicted cache bytes match |
| Serving | 28 | Explain TTFT, inter-token latency, throughput and memory for the declared workload; no separate report |

Throughout, **"tests pass" means with your own implementation**, not the
reference, and without `--extensions`: tests of optional extensions are
skipped by default. If you restored a reference file to keep going, the milestone is
reached for the system but not for you: write that down.

## Lab log

Keep one Markdown file in your repository (for example `LABLOG.md`) and add an
entry at the end of every practical session:

- the command and the revision (`git rev-parse --short HEAD`, plus "dirty" if uncommitted);
- hardware and software: CPU or GPU model, number of processes, PyTorch version;
- the prediction you made **before** running;
- the output that confirms or refutes it, with units;
- one sentence on what the result does **not** show;
- which parts you implemented, and which you restored from the reference.

This log is what you reach for in the reproduction project and in the
individual defense. A short honest entry is worth more than a long one written
afterwards.

## Experimental rules

1. **Correct before fast.** Check a new path against the simple one before you
   time it.
2. **One change per comparison**, unless the experiment explicitly tests an
   interaction.
3. **Hold the rest fixed.** Data, token count, model, optimizer and evaluation
   stay the same when you claim that one of them improved.
4. **Predict first.** Write the expected FLOPs, bytes or messages before running;
   the interesting result is the disagreement.
5. **Name the device and the exclusions.** A CPU timing is not GPU evidence, an
   estimate is not a measurement, and persistent state is not peak memory.
6. **Keep negative results.** A slower or inconclusive run is a result when it
   affects a decision.
7. **Make every number reproducible**: the exact command, configuration and
   commit.

## Falling behind

Labs are 75 minutes and are designed so that the minimum evidence fits in the
session; extensions are optional. If you are behind:

1. reach the minimum evidence first, even if it means skipping an extension;
2. use the guide's **fallback**, for example a single-rank run or a hand-traced
   computation when multi-process runs fail on your machine;
3. commit your partial work, restore the reference for that file
   ([how](companion/index.md#1-reference-code-starters-and-your-branch)) and
   record it in your lab log.

Never present a fallback as the full result: "multi-rank execution
unverified on my machine" is a correct statement; a single-rank `PASS:`
presented as a distributed result is not.

## Connection to the reproduction project

The reproduction project asks a different question (does a published claim
hold at a scale we can test?), but it uses the same toolkit:

- your training loop, baseline and checkpoints give a controlled starting point;
- the data path, profiling and parallelism labs give the measurements a systems
  claim needs;
- the experimental rules and lab log above are the minimum experimental
  expectations of the reproduction report.

The three Reproduction Studios (Sessions 17, 24 and 29–30) are working sessions
for that project, not continuous-project labs.

## Synthesis discussion

The course closes (Session 32, see the [schedule](schedule.md)) with a
comparative discussion of the system everyone built. Be ready to answer, with
evidence from your lab log:

- which check caught a bug that a loss curve or a plausible output would have hidden;
- where a technique saved memory or time, and what it cost in communication or complexity;
- which parallel layout fits a given model and cluster, and which measurement would confirm it;
- what limits serving throughput and latency for a stated workload.

Questions in the individual defense can be drawn from this shared system as
well as from your team's reproduction.
