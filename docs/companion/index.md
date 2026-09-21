# Companion Guide

This guide contains the step-by-step practical lab protocols that accompany each module in **Training and Scaling Language Models: From First Principles to Efficient Serving**.

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
| **2** | [**Training-Loop Anatomy & Baseline GPT**](02-training-loop.md) | Implement step execution, AdamW grouping, accumulation, and launch baseline GPT run | `minilm/optim.py`<br>`minilm/train.py`<br>`scripts/02_training_step.py`<br>`scripts/02_train_baseline.py` | Checkpoint round-trip, baseline checkpoint & qualitative sampling |
| **3** | [**Pretraining Data Pipeline**](03-data-pipeline.md) | Filter low-quality docs, train BPE tokenizer, pack sequences with `<\|endoftext\|>`, and create binary memmap shards | `minilm/data.py`<br>`minilm/tokenizer.py`<br>`scripts/03_prepare_dataset.py` | Target offset check & loader throughput |
| **4** | [**Single-GPU Performance**](04-single-gpu.md) | Profile VRAM, calculate theoretical FLOPs, measure MFU, and benchmark PyTorch SDPA | `minilm/profile_utils.py`<br>`scripts/04_single_gpu_perf.py` | Before/after throughput and peak memory report |
| **5** | [**Distributed Data Parallelism**](05-ddp.md) | Multi-GPU scaling with `torchrun`, gradient all-reduce, and global token accounting | `minilm/distributed.py`<br>`scripts/05_train_ddp.py` | Multi-rank synchronization & global token count |
| **6** | [**FSDP and ZeRO**](06-fsdp.md) | Model state sharding (ZeRO-1/2/3), auto-wrap policies, and activation checkpointing | `minilm/fsdp_utils.py`<br>`scripts/06_fsdp_experiment.py` | Per-GPU memory comparison against DDP |
| **7** | [**Tensor Parallelism**](07-tensor-parallelism.md) | Megatron-style Column & Row parallel linear layers, and intermediate collective elimination | `minilm/tensor_parallel.py`<br>`scripts/07_tensor_parallel.py` | Mathematical equivalence to standard MLP |
| **8** | [**Context, Pipeline & Expert Parallelism**](08-context-pipeline.md) | Analytical 5D cluster sizing, memory estimation, 1F1B bubble calculations, and EP All-to-All | `minilm/parallelism_calc.py`<br>`scripts/08_parallelism_sizing.py` | Defense of outer-to-inner hierarchy (PP->DP->FSDP->EP->TP) |
| **9** | [**KV-Cached Decoding**](09-kv-cache.md) | Implement incremental Key-Value cache and profile prefill vs decode latency | `minilm/kv_cache.py`<br>`scripts/09_kv_cache_bench.py` | Exact token match & measured generation speedup |
| **10** | [**Serving Systems**](10-serving.md) | Continuous batching scheduler simulation, paged KV-cache blocks, TTFT, and ITL | `minilm/serving_sim.py`<br>`scripts/10_serving_benchmark.py` | Serving benchmark report under dynamic workload |
| **11** | [**Frontier Architectures**](11-frontier-architectures.md) | Speculative decoding dynamics, DeepSeek MLA cache savings, linear attention, and MoE | `minilm/moe.py`<br>`scripts/11_frontier_exploration.py` | Speculative speedup modeling & MLA compression calculation |
