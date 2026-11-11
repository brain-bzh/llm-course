# Companion Guide

This guide contains the step-by-step practical lab protocols that accompany each module in **Training and Scaling Language Models: From First Principles to Efficient Serving**, designed by the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) for [IMT Atlantique](https://www.imt-atlantique.fr/en).

The [module pages](../modules/index.md) explain the concepts, bottlenecks and
derivations. The lab guides turn them into code you write, a check that proves
it is correct, and a measurement you can interpret. Every practical session
follows the same rhythm:

```text
predict  →  implement the smallest faithful version  →  compare with a reference  →  measure  →  explain
```

---

## The companion codebase

All labs use the public companion repository
[**`brain-bzh/llm-course-companion`**](https://github.com/brain-bzh/llm-course-companion).

```bash
git clone https://github.com/brain-bzh/llm-course-companion.git
cd llm-course-companion
uv python install      # CPython 3.12, pinned by the repository
uv sync
uv run pytest tests/ -q
```

Run every command in the lab guides from the root of this clone. The runtime
dependencies are `torch`, `numpy` and `tiktoken`; `pytest` checks each milestone.

### Supported platforms

Every required exercise runs on a laptop CPU, including the distributed labs,
which start two or more CPU processes that communicate through PyTorch's Gloo
backend. GPUs are used only for measurement extensions in Modules 4, 5 and 9.

| Platform | Use | Notes |
| --- | --- | --- |
| Linux (x86-64 or aarch64) | Supported | Same environment as the teaching GPUs |
| macOS (Apple silicon or Intel) | Supported | |
| Windows | **WSL2 with Ubuntu**, or the school's Linux machines | Native Windows runs the single-process labs, but `torchrun` does not work there |

Check your machine before the first distributed session (Session 14):

```bash
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/05_train_ddp.py
```

It must print a line starting with `PASS:` within about a minute.

---

## How every lab works

### 1. Reference code, starters, and your branch

`nanolm/` always holds a complete, tested reference implementation, so any lab
can run from a fresh clone. In a hands-on session you **replace one reference
module with a starter** that keeps the public interface and leaves the core
logic as `TODO`s:

```bash
git switch -c session-14-ddp              # branch from your previous lab branch
cp starter/05_ddp_manual.py nanolm/ddp_manual.py
uv run pytest tests/test_module_05_ddp.py -q   # NotImplementedError until you finish
```

Tests of optional extensions are skipped by default. Add `--extensions` to the
pytest command when you attempt one.

The Module 1 setup (Session 2) records the starting revision in `.course-reference-revision`.
If you need the reference version of a file to continue (for example because a
previous lab is unfinished), commit your work first, then restore it:

```bash
git show "$(cat .course-reference-revision):nanolm/ddp_manual.py" > nanolm/ddp_manual.py
```

Record in your notes which parts you implemented and which you restored.

### 2. Evidence, not impressions

Each session ends with **minimum evidence**: an exact check that passed, a
prediction next to a measurement, or an explanation of a failure. Keep a
short lab log (a Markdown file in your branch is enough) with:

- the command and the code revision (`git rev-parse --short HEAD`, plus "dirty" if uncommitted);
- hardware and software: CPU/GPU model, number of processes, PyTorch version;
- the prediction you made **before** running;
- the output that confirms or refutes it, with units;
- one sentence on what the result does **not** show.

A slower or inconclusive result is a valid result. Never present a CPU timing
as a GPU result, or an analytical estimate as a measurement.

### 3. Timing

Sessions last 75 minutes. Each guide gives a timed protocol, a **fallback** if
the environment fails, and **extensions** for those who finish early. If you are
behind, reach the minimum evidence first; extensions are optional.

---

## Practical sessions at a glance

| Session | Module | You implement | You verify against | Guide |
| :---: | :---: | --- | --- | --- |
| 2–4 | 1 | Causal multi-head attention, Pre-LN block, GPT | Official GPT-2 logits | [Build a decoder-only Transformer](01-transformer.md) |
| 7 | 2 | AdamW grouping, schedule, update order | One-batch overfit | [Make a small NanoLM learn](02-training-loop.md) |
| 8 | 2 | Checkpoints with RNG state; baseline run and recovery test | Bit-identical reload; uninterrupted run | [Checkpoints and baseline](02-training-loop.md#part-b-checkpoints-and-the-baseline-experiment-session-8) |
| 10 | 3 | Dedupe-before-split, packing with EOT, shifted batches | Boundary and offset tests | [Pretraining data pipeline](03-data-pipeline.md) |
| 12 | 4 | FLOP/memory accounting, SDPA branch | Profiler trace and controlled timing | [Single-GPU performance](04-single-gpu.md) |
| 14 | 5 | Data parallelism by hand: broadcast and gradient all-reduce | One process on the global batch | [Distributed data parallelism](05-ddp.md) |
| 19 | 6 | Sharded AdamW: reduce-scatter, update, all-gather | `torch.optim.AdamW`; FSDP2 lifecycle | [FSDP and ZeRO](06-fsdp.md) |
| 21 | 6–7 | Column/row parallel layers and their autograd collectives | Dense MLP on two ranks; resharded checkpoint | [Tensor parallelism](07-tensor-parallelism.md) |
| 23 | 8 | A multidimensional rank layout | Device count, groups, bubble and state estimates | [Context, pipeline & expert parallelism](08-context-pipeline.md) |
| 26 | 9 | KV-cached attention and generation | Uncached logits and greedy tokens | [KV-cached decoding](09-kv-cache.md) |
| 28 | 10 | (demonstration) | Simulated and, if available, live engine | [Serving systems](10-serving.md) |

## Lessons that outlast the course

Each guide ends with a short *What to take away* box. Taken together, they form
the habits the course is designed to build:

1. **Keep a reference you trust.** A sharded, cached or compiled version is
   only correct when it agrees with the simplest version on the same inputs.
2. **Change one thing at a time.** One controlled difference per comparison
   tells you what caused the result.
3. **Every distributed trick is bookkeeping.** DDP, ZeRO, FSDP and TP perform the same
   arithmetic as one device, and differ only in *where* each number lives and *when* it moves.
4. **Count before you measure.** Predict FLOPs, bytes and messages first; a
   measurement is interesting when it disagrees with the prediction.
5. **Name what you did not measure.** Every result has exclusions: data loading,
   activations, network time, other hardware. Saying so is part of the result.

---

## Troubleshooting distributed runs

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `torchrun` hangs before any output | Launcher resolved your hostname to an unreachable address (VPN, Wi-Fi, IPv6) | Keep `--local-addr=127.0.0.1` in the command; disconnect the VPN and retry |
| Hangs after `init_process_group` | Gloo picked the wrong network interface | Linux: `export GLOO_SOCKET_IFNAME=lo`; macOS: `export GLOO_SOCKET_IFNAME=lo0` (the scripts do this unless you set a value) |
| `use_libuv was requested but PyTorch was built without libuv support` | `torchrun` on native Windows (`USE_LIBUV=0` does not help the launcher) | Use WSL2 or the school's Linux machines |
| `Address already in use` | A previous run is still alive | Wait a few seconds or stop old `python` processes; `--standalone` picks a free port |
| One rank raises, the other hangs | A collective was reached by only one rank | Read the first traceback; every rank must call the same collectives in the same order |
| Results differ slightly between runs | Floating-point reduction order | Compare with a tolerance, as the scripts do; exact equality is not expected across ranks or devices |

Running a distributed script with plain `python` starts a single rank. That
checks the code path, not the communication.
