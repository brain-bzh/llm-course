# Modules

These pages are the conceptual spine of the course: 11 modules, each covering
one mechanism end to end. A module is not tied to a single 1h15 session — the
[schedule](../schedule.md) shows exactly which sessions cover which module, and
sessions range from two to five per module depending on how much theory and
practice the topic needs.

Every module follows the same contract: understand the mechanism, build or
test the smallest faithful version, and collect evidence that distinguishes a
real improvement from a plausible story. Concretely, each module:

1. introduces one concrete systems or modeling problem;
2. explains the minimum theory needed to reason about it;
3. implements or measures the corresponding mechanism;
4. adds the result to the continuous course project.

Slides and companion implementation guides are built from this material
without replacing it.

## 1. Transformer from first principles

**Understand:** causal attention, residual pathways, normalization, MLPs,
embeddings and the language-model head.

**Implement:** a minimal decoder-only Transformer without hiding the core logic
behind a high-level model library.

**Prove:** tensor shapes and causal masking pass, then the implementation
reproduces official GPT-2 logits and a verified next-token log-probability.

[Open the module page →](01-transformer.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/01-transformer.pdf){ target="_blank" }

## 2. Training-loop anatomy and baseline GPT

**Understand:** cross-entropy, autograd DAGs, AdamW, parameter grouping, learning-rate
schedules, gradient accumulation, clipping, mixed precision, initialization ($1/\sqrt{2L}$),
validation curves, and baseline checkpointing.

**Implement:** a complete training/checkpoint loop and use it to fit a small
NanoLM locally before attempting a longer baseline run.

**Prove:** one fixed shifted batch reaches loss below `0.1`, accumulation matches
the intended effective batch, and a restored checkpoint produces identical
logits.

[Open the module page →](02-training-loop.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/02-training-loop.pdf){ target="_blank" }

## 3. Pretraining data pipeline

**Understand:** raw web extraction (WARC vs WET), heuristic compute-conservation filtering,
vocabulary construction via byte-level BPE, document boundaries, packing,
binary sharding and zero-copy streaming without GPU starvation.

**Implement:** fast heuristic filters, BPE tokenizer training from scratch, and packed memory-mapped binary dataset shards.

**Prove:** filters discard noise without boundary corruption, tokenizer achieves lossless roundtrip fidelity, and the binary loader feeds GPUs at line rate.

[Open the module page →](03-data-pipeline.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/03-data-pipeline.pdf){ target="_blank" }

## 4. Single-GPU performance

**Understand:** FLOPs, memory decomposition, arithmetic intensity, profiler
traces, MFU, SDPA/FlashAttention, compilation and fused operations.

**Implement:** a measurement-first optimization pass.

**Prove:** report both throughput and peak memory, and isolate changes instead
of enabling every optimization simultaneously.

[Open the module page →](04-single-gpu.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/04-single-gpu.pdf){ target="_blank" }

## 5. Distributed data parallelism

**Understand:** all-reduce, process-per-GPU execution, distributed sampling,
global batches and communication overlap.

**Implement:** DDP with correct gradient accumulation.

**Prove:** parameters remain synchronized, samples are covered exactly as
intended and token accounting is global rather than rank-local.

[Open the module page →](05-ddp.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/05-ddp.pdf){ target="_blank" }

## 6. FSDP and ZeRO

**Understand:** which states are sharded, when all-gather and reduce-scatter
occur, and how checkpointing changes.

**Implement:** one focused FSDP comparison with DDP.

**Prove:** memory savings and throughput costs are measured, and the produced
checkpoint can be restored.

[Open the module page →](06-fsdp.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/06-fsdp.pdf){ target="_blank" }

## 7. Tensor and sequence parallelism

**Understand:** column- and row-parallel linear layers, attention/MLP sharding
and collective placement.

**Implement:** a minimal sharded MLP.

**Prove:** its forward results and parameter gradients match the unsharded
reference within the expected numerical tolerance.

[Open the module page →](07-tensor-parallelism.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/07-tensor-parallelism.pdf){ target="_blank" }

## 8. Context, pipeline, and expert parallelism

**Understand:** ring attention with online softmax, AFAB vs 1F1B pipeline schedules,
expert routing and All-to-All communication, multidimensional composition ($G = P \times D \times F \times E_P \times T_P \times C$),
and the physical outer-to-inner hierarchy ($\text{PP} \to \text{DP} \to \text{FSDP} \to \text{EP} \to \text{TP}$).

**Practice:** solve multidimensional memory and network communication cases across multi-node topologies.

**Prove:** every proposed strategy fits device memory and minimizes cross-network communication latency bottlenecks.

[Open the module page →](08-context-pipeline.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/08-context-pipeline.pdf){ target="_blank" } · [Interactive simulator :material-play-circle-outline:](../demos/pipeline-parallelism.html){ target="_blank" }

## 9. KV-cached decoding

**Understand:** prefill versus decode, cache shape and size, MHA/MQA/GQA and why
decode is often bandwidth-bound.

**Implement:** cached autoregressive generation in the course model.

**Prove:** cached and uncached logits agree and the benchmark separates prefill
from per-token decode.

[Open the module page →](09-kv-cache.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/09-kv-cache.pdf){ target="_blank" }

## 10. Serving systems

**Understand:** continuous batching, paged KV caches, chunked prefill, prefix
caching and scheduling.

**Explore:** inspect continuous batching, block-level KV cache allocation, and scheduler dynamics via live vLLM demonstrations and the educational [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) architecture.

**Observe:** measure time to first token (TTFT), inter-token latency (ITL), throughput saturation, and queue delays under open-loop and closed-loop workloads.

[Open the module page →](10-serving.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/10-serving.pdf){ target="_blank" } · [Interactive simulator :material-play-circle-outline:](../demos/continuous-batching.html){ target="_blank" }

## 11. Frontier architectures and efficient generation

**Understand:** speculative decoding and rejection sampling, diffusion models for text,
DeepSeek breakthroughs (Multi-Head Latent Attention, DeepSeekMoE, MTP), encoder-free
native multimodal architectures, and linear attention/SSMs (Mamba-2, Gated Delta Networks, KDA).

**Practice:** evaluate how architectural innovations break the memory-bandwidth wall
and quadratic context bottlenecks.

**Prove:** evaluate cache compression ratios, speculative speedup curves, and recurrent state update mechanics.

[Open the module page →](11-frontier-architectures.md) · [Lecture slides (PDF) :material-file-pdf-box:](../slides/11-frontier-architectures.pdf){ target="_blank" }
