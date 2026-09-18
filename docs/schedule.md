# Schedule

The course contains **13 theory periods** and **19 implementation, experiment or
presentation periods**. Practice therefore occupies 23h45 of the 40 scheduled
hours.

| Week | Days | Periods | Contact time | Main outcome |
| --- | ---: | ---: | ---: | --- |
| Build | 3 | 12 | 15h | Tokenizer, minimal GPT and baseline checkpoint |
| Select and scale | 3 | 12 | 15h | Selected data, optimized trainer and scaling evidence |
| Serve and synthesize | 2 | 8 | 10h | Cached decoder, serving benchmark and final defense |

## Week 1 — Build a language model

### Day 1 — Model and training loop

**5 periods · 6h15**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 1 | Theory | Transformer anatomy: embeddings, causal attention, residual stream, normalization, MLP and LM head | Architecture specification |
| 2 | Practice | Implement embeddings, causal attention and masking | Tested attention module |
| 3 | Practice | Assemble the Transformer block and minimal GPT | Forward pass and tiny-batch overfit |
| 4 | Theory | Loop anatomy: cross-entropy, AdamW, accumulation, clipping, scheduling and mixed precision | Training-loop checklist |
| 5 | Practice | Implement training, evaluation, logging and checkpointing | Minimal trainable GPT repository |

!!! success "Exit criterion"
    The model must overfit a tiny batch. If it cannot, the team does not proceed
    to larger training.

### Day 2 — From raw text to tensors

**2 periods · 2h30**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 6 | Theory | BPE, vocabulary trade-offs, document boundaries, packing, shuffling and splits | Dataset design |
| 7 | Practice | Train a tokenizer and convert documents into packed token sequences | Tokenizer and initial shards |

### Day 3 — Baseline training and data selection

**5 periods · 6h15**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 8 | Practice | Finish the dataloader; test shuffling, boundaries and throughput | Validated data pipeline |
| 9 | Theory | Initialization, validation loss, sampling, checkpoint selection and failure diagnosis | Baseline protocol |
| 10 | Practice | Launch and inspect the baseline run | Baseline checkpoint |
| 11 | Theory | Filtering, deduplication, quality, loss-based selection, diversity and mixtures | Selection hypotheses |
| 12 | Practice | Implement random, heuristic and deduplicated baselines | Candidate data subsets |

**Between teaching days:** score the corpus and run the controlled
baseline-versus-selection experiment with an identical token budget.

## Week 2 — Select and scale

### Day 4 — Controlled data experiments and profiling

**2 periods · 2h30**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 13 | Practice | Compare selection strategies at fixed token and compute budgets | Selected corpus and evidence |
| 14 | Theory | FLOPs, memory, arithmetic intensity, MFU, profiling, SDPA and compilation | Performance model |

### Day 5 — Single-GPU optimization and DDP

**5 periods · 6h15**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 15 | Practice | Profile the baseline and identify the actual bottleneck | Baseline performance report |
| 16 | Practice | Apply and measure bf16, SDPA, compilation or fusion one change at a time | Optimized trainer |
| 17 | Theory | DDP, all-reduce, distributed sampling, accumulation, `no_sync` and global batches | DDP execution model |
| 18 | Practice | Convert the trainer to DDP and verify equivalence and accounting | Correct multi-GPU trainer |
| 19 | Theory | FSDP and ZeRO: sharding, collectives, memory and checkpointing | Sharding plan |

### Day 6 — Sharded and multidimensional parallelism

**5 periods · 6h15**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 20 | Practice | Compare a small FSDP run with DDP | Memory, throughput and checkpoint evidence |
| 21 | Theory | Tensor and sequence parallelism: row/column sharding and communication placement | TP derivation |
| 22 | Practice | Implement and validate a toy sharded MLP | Minimal TP implementation |
| 23 | Theory | Context parallelism, pipeline parallelism, bubbles and topology | Strategy rules |
| 24 | Practice | Select and defend strategies for several model and cluster configurations | Scaling design report |

!!! note "Implementation boundary"
    Students implement DDP, a focused FSDP experiment and a toy TP layer. CP,
    PP and multidimensional parallelism are taught through cost models, design
    exercises and production-code reading—not a fragile reimplementation of a
    production stack.

## Week 3 — Serve and synthesize

### Day 7 — Autoregressive decoding

**3 periods · 3h45**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 25 | Theory | Naive decoding, prefill/decode, KV cache, MHA/MQA/GQA and memory bandwidth | Decode cost model |
| 26 | Practice | Add a KV cache to the course model | Cached decoder |
| 27 | Practice | Test equivalence and benchmark latency, memory and tokens per second | Decode benchmark |

### Day 8 — Serving and final demonstration

**5 periods · 6h15**

| Period | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 28 | Theory | Continuous batching, paged KV cache, chunked prefill, prefix caching and scheduling | Serving-system model |
| 29 | Practice | Serve the checkpoint and measure TTFT, ITL, throughput and memory | Serving benchmark |
| 30 | Theory | MoE, long context, state-space models, multimodality and diffusion-style LMs | Architecture comparison |
| 31 | Practice | Final team demonstrations and technical defense | Presentation |
| 32 | Practice | Comparative postmortem and course synthesis | Final report and repository |

