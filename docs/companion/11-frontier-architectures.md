# Module 11 Lab — Frontier Architectures

This page accompanies [Module 11: Frontier architectures](../modules/11-frontier-architectures.md).

!!! note "No required lab"
    Session 27 is a lecture with an in-class pen-and-paper exercise. The
    exploration script below predates it and covers neighbouring topics
    (speculative decoding, MLA memory, linear recurrence, MoE routing); it is
    optional material.

    The exploration scripts and unit tests below are provided as optional reference implementations for self-paced study or for teams incorporating frontier techniques into their reproduction project.

!!! tip "Optional Exploration & Reference Resources"
    - **Lab script:** [`scripts/11_frontier_exploration.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/11_frontier_exploration.py)
    - **Reference module:** [`nanolm/moe.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/moe.py)
    - **Unit tests:** [`tests/test_module_11_frontier.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_11_frontier.py)

---

## Objectives

1. **Speculative decoding speedup dynamics**: Simulate rejection sampling token acceptance and quantify wall-clock acceleration as a function of draft length $\gamma$ and acceptance rate $\alpha$.
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
