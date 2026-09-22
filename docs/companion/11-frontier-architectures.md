# Module 11 Lab — Frontier Architectures & Efficient Generation

This lab accompanies [Module 11: Frontier architectures and efficient generation](../modules/11-frontier-architectures.md).

!!! info "Lecture & Optional Exploration"
    Session 30 is a theory lecture covering frontier architectures and inference acceleration, leading directly into final project presentations and defenses (Sessions 31–32). There is no mandatory lab assignment or report to submit for this module, preserving full team bandwidth for [Reproduction Project Defenses](../schedule.md#phase-3-serve-and-synthesize).

    The exploration scripts and unit tests below are provided as optional reference implementations for self-paced study or for teams incorporating frontier techniques into their reproduction project.

!!! tip "Optional Exploration & Reference Resources"
    - **Lab script:** [`scripts/11_frontier_exploration.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/11_frontier_exploration.py)
    - **Reference module:** [`minilm/moe.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/minilm/moe.py)
    - **Unit tests:** [`tests/test_module_11_frontier.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_11_frontier.py)

---

## Objectives

1. **Speculative decoding speedup dynamics**: Simulate rejection sampling token acceptance and quantify wall-clock acceleration as a function of draft length $K$ and acceptance rate $\alpha$.
2. **DeepSeek Multi-Head Latent Attention (MLA)**: Calculate KV-cache memory footprints comparing standard MHA, Grouped-Query Attention (GQA), and low-rank latent compression (MLA).
3. **Linear recurrence & constant-memory states**: Contrast quadratic attention cache growth with constant $\mathcal{O}(1)$ associative memory states (e.g. Gated Delta Networks and Mamba-2).
4. **Sparse Mixture of Experts (MoE)**: Evaluate expert routing, parameter capacity scaling, and active compute efficiency.

---

## Quickstart

Run the unit tests and the frontier architectures exploration script:

```bash
uv run pytest tests/test_module_11_frontier.py -v
uv run python scripts/11_frontier_exploration.py
```
