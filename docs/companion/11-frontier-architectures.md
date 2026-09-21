# Module 11 Lab — Frontier Architectures & Efficient Generation

This lab accompanies [Module 11: Frontier architectures and efficient generation](../modules/11-frontier-architectures.md).

!!! tip "Practical Lab Resources"
    To work through the hands-on implementation for this module:

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
cd companion
uv run pytest tests/test_module_11_frontier.py -v
uv run python scripts/11_frontier_exploration.py
```
