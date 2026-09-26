# Module 6 Lab — FSDP and ZeRO

This lab accompanies [Module 6: FSDP and ZeRO](../modules/06-fsdp.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/06_fsdp_experiment.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/06_fsdp_experiment.py)
    - **Reference module:** [`nanolm/profile_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/profile_utils.py) · [`nanolm/optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/optim.py)
    - **Unit tests:** [`tests/test_module_08_parallelism.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_08_parallelism.py)

---

## Objectives

1. **State sharding decomposition**: Understand where memory savings originate across ZeRO-1 (optimizer states), ZeRO-2 (gradients), and ZeRO-3 / FSDP (model weights).
2. **Auto-wrap policy**: Configure wrapping rules for Transformer blocks and observe collective communication (all-gather and reduce-scatter) timing.
3. **Peak memory measurement**: Compare memory consumption between unsharded DDP and sharded FSDP under identical batch configurations.

---

## Quickstart

Run the FSDP memory scaling analysis:

```bash
uv run python scripts/06_fsdp_experiment.py
```

## Session 19 protocol

**Prepare:** state model size, optimizer state multiplier (e.g. AdamW requires $8\text{ bytes/param}$ for $m$ and $v$ in FP32), and rank count $N$.

- **0–15 min:** predict static state memory per GPU for a 24-layer model under DDP vs ZeRO-1, ZeRO-2, and ZeRO-3 across $N \in \{1, 4, 8, 64\}$ ranks.
- **15–35 min:** execute `scripts/06_fsdp_experiment.py` and verify calculated memory footprints against analytical state formulas.
- **35–55 min:** analyze communication overhead: explain why ZeRO-3 requires all-gather before forward and backward, whereas DDP requires only gradient all-reduce.
- **55–75 min:** relate static state savings to dynamic activation memory. Formulate an auto-wrap policy on `TransformerBlock` boundaries to constrain peak materialized working memory.

**Minimum evidence:** verified parameter, gradient, and optimizer state byte breakdown table across sharding strategies, with explicit distinction between persistent state and transient forward activations.

**Fallback:** analytical derivation from parameter counts if script execution is unavailable. **Extension:** calculate the communication volume ratio between standard DDP and ZeRO-3 for one optimization step.
