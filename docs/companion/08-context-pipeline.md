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
uv run torchrun --standalone --nproc_per_node=2 scripts/08_parallelism_layout.py --gloo-check
```

This verifies the DDP averaging primitive and a small TP forward/backward on
real CPU tensors. The checks run in sequence on the same two processes; they do
not execute the full logical mesh or measure GPU/network performance.

## Session 23 protocol

- **Prepare:** read the rank-group definitions and complete one pipeline-bubble calculation.
- **0–10 min:** edit the model and layout dimensions in `scripts/08_parallelism_layout.py`; predict the logical device count and global tokens/update.
- **10–25 min:** run the script; inspect rank coordinates and DP/TP/PP/CP groups. Explain which dimensions split model state, activations, or input batches.
- **25–40 min:** compare EP and FSDP as alternative subdivisions of DP at the same device count; inspect the meta-device parameter/TP-shard shapes and idealized persistent-state estimates.
- **40–55 min:** use the pipeline simulator to compare AFAB and 1F1B at fixed stages and micro-batches. Report idle/total and activation residency separately.
- **55–70 min:** defend the chosen layout and one alternative, including network placement, an explicit limitation, and the measurements needed to choose between them.
- **70–75 min:** if the environment is ready, temporarily switch to pure TP=2 and run the Gloo DDP/TP checks; otherwise record them as unverified and use the reference results from Sessions 16 and 21.

**Definition of done:** submit a rank-layout diagram or printed map; show two
valid configurations with the same device count and global tokens/update;
include the persistent-state estimate, pipeline result, and one communication
payload estimate with assumptions and exclusions; trace one training step
through the configured groups; and defend one layout with a falsifiable
placement hypothesis. Explain what the optional Gloo checks prove and what they
do not. The calculator does not predict peak VRAM or network time.

**Fallback:** use the module's worked rank layout and hand-calculate its device
count and pipeline fraction. Combined EP/FSDP layouts are an optional extension
requiring separate expert synchronization and shard-group definitions.
