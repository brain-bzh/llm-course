# Companion Guide

This guide contains the step-by-step practical lab protocols that accompany each module in **Training Language Models: From First Principles to Efficient Serving**.

While the [Module material](../modules/index.md) defines the theoretical concepts, system bottlenecks, and mathematical foundations, the **Companion Guide** walks through the implementation, code inspection, measurement exercises, and testable exit criteria.

---

## The Companion Codebase

All practical modules are backed by the minimalist, self-contained companion repository located in the [`companion/`](https://github.com/jonathanlys01/llm-course/tree/main/companion) subdirectory of the course repository.

```bash
# Navigate to companion repository
cd companion

# Synchronize dependencies with uv
uv sync

# Run all test suites
uv run pytest tests/ -v
```

The companion codebase is built with strictly minimal dependencies:
- **PyTorch** (`torch`): Tensor operations and neural network modules.
- **NumPy** (`numpy`): Memory-mapped binary datasets and array packing.
- **Tiktoken** (`tiktoken`): Byte-Pair Encoding tokenizer for GPT models.
- **Pytest** (`pytest`): Milestone and invariant verification tests.

---

## Practical Module Roadmap

| Module | Lab Guide | Focus | Companion Scripts & Modules | Milestone / Exit Criterion |
| :---: | :--- | :--- | :--- | :--- |
| **1** | [**Inspect Transformer & Naive MHA**](01-transformer.md) | Inspect weights/shapes & implement causal Multi-Head Attention from first principles | `minilm/model.py`<br>`scripts/01_inspect_and_mha.py` | Causal masking invariance & tiny-batch overfit (`loss < 0.1`) |
| **2** | Training-Loop Anatomy *(coming next)* | Implement step execution, AdamW parameter grouping, cosine warmup, and checkpoint recovery | `minilm/optim.py`<br>`scripts/02_training_step.py` | Weight decay grouping verification & checkpoint round-trip |
| **3** | BPE & Data Pipeline | Tokenize raw text, pack sequences with `<\|endoftext\|>`, and create binary memmap shards | `minilm/data.py`<br>`scripts/03_prepare_dataset.py` | Target offset check & loader throughput |
| **4** | Train Small GPT | Launch baseline training run, track validation curves, and sample autoregressively | `minilm/train.py`<br>`scripts/04_train_baseline.py` | Baseline checkpoint & qualitative sampling |
| **5** | Data Selection | Implement heuristic filtering, exact & near deduplication, and controlled A/B subsets | `minilm/data_selection.py`<br>`scripts/05_data_selection.py` | Fixed-budget comparison against random baseline |
| **6** | Single-GPU Performance | Profile VRAM, calculate theoretical FLOPs, measure MFU, and benchmark PyTorch SDPA | `minilm/profile_utils.py`<br>`scripts/06_single_gpu_perf.py` | Before/after throughput and peak memory report |
| **7** | Distributed Data Parallelism | Multi-GPU scaling with `torchrun`, gradient all-reduce, and global token accounting | `minilm/distributed.py`<br>`scripts/07_train_ddp.py` | Multi-rank synchronization & global token count |
| **8** | FSDP and ZeRO | Model state sharding (ZeRO-1/2/3), auto-wrap policies, and activation checkpointing | `minilm/fsdp_utils.py`<br>`scripts/08_fsdp_experiment.py` | Per-GPU memory comparison against DDP |
| **9** | Tensor Parallelism | Megatron-style Column & Row parallel linear layers, and intermediate collective elimination | `minilm/tensor_parallel.py`<br>`scripts/09_tensor_parallel.py` | Mathematical equivalence to standard MLP |
| **10** | Context & Pipeline Parallelism | Analytical 3D/4D cluster sizing, memory estimation, and 1F1B bubble calculations | `minilm/parallelism_calc.py`<br>`scripts/10_parallelism_sizing.py` | Sizing defense for model & cluster scenarios |
| **11** | KV-Cached Decoding | Implement incremental Key-Value cache and profile prefill vs decode latency | `minilm/kv_cache.py`<br>`scripts/11_kv_cache_bench.py` | Exact token match & measured generation speedup |
| **12** | Serving Systems | Continuous batching scheduler simulation, paged KV-cache blocks, TTFT, and ITL | `minilm/serving_sim.py`<br>`scripts/12_serving_benchmark.py` | Serving benchmark report under dynamic workload |
| **13** | Beyond Dense Transformers | Sparse Mixture of Experts (MoE) layer, top-$k$ routing, and auxiliary load-balancing loss | `minilm/moe.py`<br>`scripts/13_moe_exploration.py` | Parameter expansion vs active compute analysis |
