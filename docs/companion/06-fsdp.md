# Module 6 Lab — FSDP and ZeRO

This lab accompanies [Module 6: FSDP and ZeRO](../modules/06-fsdp.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

    - **Lab script:** [`scripts/06_fsdp_experiment.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/06_fsdp_experiment.py)
    - **Reference module:** [`nanolm/profile_utils.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/profile_utils.py) · [`nanolm/optim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/optim.py)
    - **Unit tests:** [`tests/test_module_08_parallelism.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_08_parallelism.py) (analytical state-accounting checks)

---

## Objectives

1. **State sharding decomposition**: Understand where memory savings originate across ZeRO-1 (optimizer states), ZeRO-2 (gradients), and ZeRO-3 / FSDP (model weights).
2. **Auto-wrap policy**: Configure wrapping rules for Transformer blocks and observe collective communication (all-gather and reduce-scatter) timing.
3. **Peak memory reasoning**: Separate persistent-state estimates from gathered parameters, activations, and allocator reserve. The current script does not execute FSDP or measure peak VRAM.

---

## Quickstart

Run the analytical ZeRO state calculator (it does not launch FSDP):

```bash
uv run python scripts/06_fsdp_experiment.py
```

## Session 19 protocol

**Prepare:** read the DDP/FSDP comparison contract in Module 6 and calculate
state bytes per parameter for the declared optimizer and precision.

- **0–15 min:** predict DDP and ZeRO-1/2/3 persistent state for the supplied model and world sizes.
- **15–35 min:** run the calculator and reconcile each term against your worksheet.
- **35–55 min:** inspect `nanolm/fsdp_utils.py`; trace when each wrapped block's full parameters exist and when gradients are reduced.
- **55–75 min:** list the hardware, synchronization, warmup, model, batch, and measurement conditions needed for a fair DDP/FSDP peak-memory and throughput comparison.

**Minimum evidence:** state formulas, predicted values, a materialization
lifecycle sketch, and a list of excluded peak-memory terms. The current example
script is analytical: it constructs a model and prints formulas. It does not
launch distributed workers or call FSDP. No empirical FSDP result is required
from this script. If the teaching cluster later supports an actual FSDP run,
record that as a separate, currently unverified extension.

**Fallback:** use one rank and compare the formulas by hand. **Extension:** with
instructor-supplied measurements, compare predicted persistent state to observed
peak allocation and explain the residual; do not infer a speedup from the formula.
