# Module 8 Lab — Context, Pipeline, and Expert Parallelism

This lab accompanies [Module 8: Context, pipeline, and expert parallelism](../modules/08-context-pipeline.md).

!!! tip "Practical Lab & Simulator Resources"
    To explore multidimensional and pipeline parallelism concepts:

    - **Interactive simulator:** [Pipeline Parallelism Schedule Simulator](../demos/pipeline-parallelism.html){ target="_blank" } — interactive schedule comparison (Naive, AFAB, 1F1B, Interleaved), bubble ratios, memory bars, and boundary communication graph.
    - **Interactive composer:** [Parallelism Composer](../demos/parallelism-composer.html){ target="_blank" } — rank-to-GPU mapping on a multi-node cluster, process groups for TP/CP/EP/PP/DP, and per-GPU memory, communication, and step-time estimates.
    - **Layout exercise:** [`scripts/08_parallelism_layout.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/08_parallelism_layout.py) — student-defined rank mesh, NanoLM meta-device shape census, idealized sizing, and optional Gloo checks reusing DDP and TP components.
    - **Lab script:** [`scripts/08_parallelism_sizing.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/08_parallelism_sizing.py)
    - **Reference module:** [`nanolm/parallelism_calc.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/parallelism_calc.py)
    - **Unit tests:** `tests/test_module_08_parallelism.py`

---

## Objectives

1. **Pipeline bubble estimation**: Calculate idle bubbles for 1F1B (One Forward, One Backward) schedules across micro-batch counts.
2. **Context Parallelism memory scaling**: Trace how Ring Attention and sequence chunking remove the quadratic attention activation memory bottleneck.
3. **Expert Parallelism & All-to-All volume**: Quantify token routing data transfers across sparse MoE experts.
4. **Compose a layout in code**: Define PP/TP/CP/DP dimensions, derive rank coordinates and groups, and choose whether FSDP or EP subdivides DP.
5. **Model without allocating a large model**: Inspect NanoLM parameter and TP-shard shapes on PyTorch's [`meta` device](https://docs.pytorch.org/docs/stable/meta.html), then compare idealized persistent-state, pipeline, and communication-payload estimates. Meta tensors contain shape metadata, not parameter data, so they do not run collectives or numerical training.
6. **Separate model from measurement**: State what the calculator and pipeline simulator estimate, what they omit, and which claims still need real hardware measurements.

---

## Quickstart

From the companion repository root, run the student-editable layout script and
the analytical sizing calculator:

```bash
uv run python scripts/08_parallelism_layout.py
uv run python scripts/08_parallelism_sizing.py
uv run pytest tests/test_module_08_parallelism.py -q
```

In `scripts/08_parallelism_layout.py`, edit `MODEL` and `LAYOUT`, then rerun it.
The starting layout composes DP, TP, PP, CP, and EP across 32 logical devices.
Compare it with FSDP by setting `ep_size=1` and `fsdp_size=4`; the calculator
does not model EP and FSDP together. The meta census describes the dense NanoLM
parameters and illustrative TP MLP shards; the sizing model separately adds
the configured MoE expert count. Payload estimates assume bf16/fp16 activations
and uniform expert routing; EP dispatch assumes full-width hidden states before
TP sharding is accounted for.

For actual CPU communication checks, temporarily set a pure two-rank TP
layout (`dp_size=1`, `tp_size=2`, `pp_size=1`, `cp_size=1`, `ep_size=1`,
`fsdp_size=1`) and run the reused DDP all-reduce helper and TP equivalence
implementation:

```bash
uv run torchrun --standalone --local-addr=127.0.0.1 --nproc_per_node=2 scripts/08_parallelism_layout.py --gloo-check
```

This verifies the DDP averaging primitive and a small TP forward/backward on
real CPU tensors. The checks run in sequence on the same two processes; they do
not execute the full logical mesh or measure GPU/network performance.

## Before class (15 minutes)

1. Read the rank-group definitions in
   [Module 8](../modules/08-context-pipeline.md) and complete one
   pipeline-bubble calculation by hand: $P=4$ stages, $m=8$ micro-batches.
2. Run the quickstart commands above once, so that the environment is ready.

```bash
git switch -c session-23-layout
```

## Session 23 protocol — 75 minutes

| Time | Activity | Evidence |
| --- | --- | --- |
| 0–10 min | Edit `MODEL` and `LAYOUT`; predict the logical device count and global tokens/update | Written prediction |
| 10–25 min | Run the script; inspect rank coordinates and the DP/TP/PP/CP groups | One sentence per axis: does it split model state, activations or the batch? |
| 25–40 min | A second layout at the same device count: EP or FSDP subdivides DP | Two valid layouts with their persistent-state estimates |
| 40–55 min | [Pipeline simulator](../demos/pipeline-parallelism.html){ target="_blank" }: AFAB vs 1F1B at fixed stages and micro-batches | Idle fraction and activation residency, reported separately |
| 55–75 min | Defend one layout | Which group sits on the fastest links, one limitation, and the measurement that would refute the choice |

Everything in this session is an estimate. The calculator gives persistent
state and idealized payloads; it does not predict peak memory or step time.

## Definition of done

- two valid layouts with the same device count and global tokens/update, as printed maps;
- for each, the persistent-state estimate and the pipeline idle fraction, with what they exclude;
- one layout defended: which group goes on the fastest links, and which measurement would refute the choice.

**Fallback:** use the module's worked rank layout and hand-calculate its device
count and pipeline idle fraction.

**Extensions:** trace one training step through the configured groups and
estimate one communication payload with its assumptions; run the Gloo checks
at pure TP=2 (see Quickstart) and say what they prove and what they do not.
Combined EP/FSDP layouts require separate expert synchronization and
shard-group definitions.

!!! success "What to take away"
    - **A layout is a factorization of the device count.** Write $G = P \times T_P \times C \times D$
      explicitly; FSDP and EP subdivide data parallelism and add no devices.
    - **Each axis splits something different.** TP and PP split model state, CP
      splits the sequence's activations, DP splits the batch. Choose the axis for
      the resource that is actually short.
    - **Put the chattiest groups on the fastest links.** TP communicates every
      layer; pipeline stages communicate only at their boundaries, so they tolerate slower links.
    - **Bubbles are arithmetic.** For AFAB and 1F1B the idle fraction is
      $(P-1)/(m+P-1)$; reducing it to 10 % needs $m > 9(P-1)$ micro-batches. 1F1B
      changes activation residency, not this fraction.
    - **An estimate is a hypothesis.** The calculator gives persistent state and
      payloads, not peak VRAM or step time. Name the measurement that would choose between two layouts.
