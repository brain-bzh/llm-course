# Modules

These pages are the conceptual spine of the course: 13 modules, each covering
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

**Prove:** tensor shapes, causal masking and a tiny-batch overfit test all pass.

[Open the module page →](01-transformer.md)

## 2. Training-loop anatomy

**Understand:** cross-entropy, backward propagation, AdamW, schedules, gradient
accumulation, clipping and mixed precision.

**Implement:** a complete training/evaluation/checkpoint loop.

**Prove:** accumulation matches the intended effective batch and optimizer-step
count; checkpoints resume deterministically enough for the stated setup.

[Open the module page →](02-training-loop.md)

## 3. BPE and the data pipeline

**Understand:** vocabulary construction, document boundaries, packing,
shuffling, splits, streaming and loader throughput.

**Implement:** tokenizer training and packed dataset production from raw text.

**Prove:** boundaries and splits are valid, samples are reproducible and the
loader does not starve the accelerator.

[Open the module page →](03-data-pipeline.md)

## 4. Train a small GPT

**Understand:** initialization, validation loss, sampling, checkpoint selection
and common failure signatures.

**Implement:** the first end-to-end baseline run.

**Prove:** training beats trivial baselines, validation behaves coherently and
the checkpoint can generate samples.

[Open the module page →](04-small-gpt.md)

## 5. Data selection

**Understand:** filtering, deduplication, quality, loss, diversity, mixtures and
contamination.

**Implement:** random, heuristic and quality/diversity selection under a fixed
token budget.

**Prove:** comparisons use the same model, token budget, optimizer recipe and
evaluation protocol.

[Open the module page →](05-data-selection.md)

## 6. Single-GPU performance

**Understand:** FLOPs, memory decomposition, arithmetic intensity, profiler
traces, MFU, SDPA/FlashAttention, compilation and fused operations.

**Implement:** a measurement-first optimization pass.

**Prove:** report both throughput and peak memory, and isolate changes instead
of enabling every optimization simultaneously.

[Open the module page →](06-single-gpu.md)

## 7. Distributed data parallelism

**Understand:** all-reduce, process-per-GPU execution, distributed sampling,
global batches and communication overlap.

**Implement:** DDP with correct gradient accumulation.

**Prove:** parameters remain synchronized, samples are covered exactly as
intended and token accounting is global rather than rank-local.

[Open the module page →](07-ddp.md)

## 8. FSDP and ZeRO

**Understand:** which states are sharded, when all-gather and reduce-scatter
occur, and how checkpointing changes.

**Implement:** one focused FSDP comparison with DDP.

**Prove:** memory savings and throughput costs are measured, and the produced
checkpoint can be restored.

[Open the module page →](08-fsdp.md)

## 9. Tensor and sequence parallelism

**Understand:** column- and row-parallel linear layers, attention/MLP sharding
and collective placement.

**Implement:** a minimal sharded MLP.

**Prove:** its forward results and parameter gradients match the unsharded
reference within the expected numerical tolerance.

[Open the module page →](09-tensor-parallelism.md)

## 10. Context and pipeline parallelism

**Understand:** sequence-length pressure, pipeline bubbles, microbatches,
topology and multidimensional composition.

**Practice:** solve memory and communication cases rather than building a
production framework.

**Prove:** every proposed strategy fits memory and identifies its dominant
communication and utilization costs.

[Open the module page →](10-context-pipeline.md)

## 11. KV-cached decoding

**Understand:** prefill versus decode, cache shape and size, MHA/MQA/GQA and why
decode is often bandwidth-bound.

**Implement:** cached autoregressive generation in the course model.

**Prove:** cached and uncached logits agree and the benchmark separates prefill
from per-token decode.

[Open the module page →](11-kv-cache.md)

## 12. Serving systems

**Understand:** continuous batching, paged KV caches, chunked prefill, prefix
caching and scheduling.

**Implement:** serve the course checkpoint with an existing serving engine.

**Prove:** measure time to first token, inter-token latency, throughput and
memory under declared workloads.

[Open the module page →](12-serving.md)

## 13. Beyond the dense autoregressive Transformer

**Understand:** the pressures that motivate MoE, longer-context mechanisms,
state-space models, multimodality and diffusion-style language models.

**Practice:** compare architectures against the bottlenecks established during
the course.

**Prove:** claims are tied to concrete changes in compute, memory, data or
inference behavior rather than novelty alone.

[Open the module page →](13-beyond-transformers.md)
