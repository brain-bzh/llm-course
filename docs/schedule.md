# Schedule

The course runs as **32 sessions of 1h15** (40 contact hours): 12 theory
sessions, 16 guided-implementation sessions, 2 mid-course paper-presentation
sessions and 2 final presentation/synthesis sessions. Sessions are grouped into
**8 blocks**, and blocks into three phases. A block groups sessions that belong
to the same module or milestone — it is not a fixed calendar unit. Depending on
the institutional timetable, a block may occupy part of a day, a full day, or
be split across more than one day.

| Phase | Blocks | Sessions | Contact time | Main outcome |
| --- | --- | ---: | ---: | --- |
| Build | 3 | 12 | 15h | End-to-end data pipeline, minimal GPT, baseline checkpoint, optimized trainer |
| Scale | 3 | 12 | 15h | Mid-course presentation, distributed DDP/FSDP scaling, tensor parallelism evidence |
| Serve and synthesize | 2 | 8 | 10h | Cached decoder, serving benchmark and final defense |

Each module (the 11 topics listed on the [module pages](modules/index.md))
spans two to five sessions, matched to how much theory and practice the topic
needs — a module is a unit of content, not a unit of time. The table below
shows exactly which sessions cover which module.

## Phase 1 — Build a language model

### Block 1 — Model and training loop

**5 sessions · 6h15 · [Module 1](modules/01-transformer.md) ([slides](slides/01-transformer.pdf){ target="_blank" }, sessions 1–3), [Module 2](modules/02-training-loop.md) ([slides](slides/02-training-loop.pdf){ target="_blank" }, sessions 4–5)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 1 | Theory | Transformer anatomy: embeddings, causal attention, residual stream, normalization, MLP and LM head | Architecture specification |
| 2 | Practice | Implement embeddings, causal attention and masking | Tested attention module |
| 3 | Practice | Assemble the Transformer block and minimal GPT | Forward pass and tiny-batch overfit |
| 4 | Theory | Loop anatomy: cross-entropy, AdamW, accumulation, clipping, scheduling and mixed precision | Training-loop checklist |
| 5 | Practice | Implement training, evaluation, logging and checkpointing | Minimal trainable GPT repository |

!!! success "Exit criterion"
    The model must overfit a tiny batch. If it cannot, the team does not proceed
    to larger training.

!!! note "Assessment milestone"
    Teams begin identifying candidate papers for the mid-course presentation
    and reproduction project.

### Block 2 — From raw text to tensors

**3 sessions · 3h45 · [Module 3](modules/03-data-pipeline.md) ([slides](slides/03-data-pipeline.pdf){ target="_blank" }, sessions 6–8)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 6 | Theory | Raw web extraction (WARC vs WET), heuristic compute-conservation filtering, BPE tokenization, vocabulary trade-offs, packing and binary memmap sharding | Ingestion pipeline design |
| 7 | Practice | Implement heuristic filters, train BPE tokenizer, and pack clean documents into binary shards | Tokenizer and initial shards |
| 8 | Practice | Build and test memory-mapped DataLoader; verify boundaries, prefetching, and throughput without GPU starvation | Validated high-throughput data pipeline |

!!! note "Assessment milestone"
    Paper shortlist due.

### Block 3 — Baseline training, profiling and single-GPU optimization

**4 sessions · 5h00 · [Module 2](modules/02-training-loop.md) continued (sessions 9–10), [Module 4](modules/04-single-gpu.md) ([slides](slides/04-single-gpu.pdf){ target="_blank" }, sessions 11–12)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 9 | Theory | Initialization ($1/\sqrt{2L}$), validation loss, sampling, checkpoint selection and failure diagnosis | Baseline protocol |
| 10 | Practice | Launch and inspect the baseline training run | Baseline checkpoint |
| 11 | Theory | Hardware model of one GPU: FLOPs, memory hierarchy, rooflines, arithmetic intensity, and MFU | Bottleneck hypothesis |
| 12 | Practice | Profile baseline with PyTorch Profiler / Nsight; apply bf16, SDPA/FlashAttention, and `torch.compile` | Optimized single-GPU trainer |

**Between sessions:** profile the optimized trainer across batch sizes and verify throughput and memory improvements before distributed scaling.

!!! note "Assessment milestone"
    Paper and one or two target reproduction claims approved.

## Phase 2 — Scale

### Block 4 — Mid-course paper presentations

**2 sessions · 2h30**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 13 | Presentation | Group paper presentations: problem, central claims, mechanism, evidence and reproduction proposal | Presentation and peer questions |
| 14 | Presentation | Remaining group presentations and cross-group discussion | Approved reproduction protocol per team |

!!! success "Assessment milestone"
    **Mid-course paper presentation** (see [Reproduction project](reproduction-project.md)).

### Block 5 — Distributed data parallelism and memory sharding

**5 sessions · 6h15 · [Module 5](modules/05-ddp.md) ([slides](slides/05-ddp.pdf){ target="_blank" }, sessions 15–17), [Module 6](modules/06-fsdp.md) ([slides](slides/06-fsdp.pdf){ target="_blank" }, sessions 18–19)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 15 | Theory | DDP execution model: all-reduce algorithms, process-per-GPU execution, distributed sampling, global batches and communication overlap | DDP architecture |
| 16 | Practice | Convert trainer to DDP with `torchrun`; verify rank synchronization and gradient accumulation via `no_sync` | Verified multi-GPU trainer |
| 17 | Practice | Benchmark DDP scaling efficiency across multiple GPUs; quantify communication overhead and linear speedup limits | DDP scaling report |
| 18 | Theory | FSDP and ZeRO: state decomposition (ZeRO-1/2/3), all-gather and reduce-scatter collective timing, memory scaling, and checkpointing | Sharding plan |
| 19 | Practice | Implement FSDP auto-wrap policy on course model; benchmark per-GPU memory savings against DDP | Initial FSDP benchmark |

!!! note "Assessment milestone"
    Reproduction protocol and baseline fixed.

### Block 6 — Advanced sharding and multidimensional parallelism

**5 sessions · 6h15 · [Module 6](modules/06-fsdp.md) continued (session 20), [Module 7](modules/07-tensor-parallelism.md) ([slides](slides/07-tensor-parallelism.pdf){ target="_blank" }, sessions 21–22), [Module 8](modules/08-context-pipeline.md) ([slides](slides/08-context-pipeline.pdf){ target="_blank" }, sessions 23–24)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 20 | Practice | Distributed checkpointing (save/resume with sharded states) and recovery overhead analysis | Resilient FSDP checkpoint pipeline |
| 21 | Theory | Tensor and sequence parallelism: column/row sharding, collective placement, and activation recomputation | TP derivation |
| 22 | Practice | Implement and validate a toy sharded MLP using Gloo; verify numerical equivalence to unsharded linear layers | Minimal TP implementation |
| 23 | Theory | Context, pipeline & expert parallelism: ring attention, 1F1B bubbles, All-to-All communication and outer-to-inner hierarchy | Strategy rules |
| 24 | Practice | Select and defend multidimensional strategies ($G = P \times D \times F \times E_P \times T_P \times C$) across physical cluster topologies | Scaling design report |

!!! note "Implementation boundary"
    Students implement DDP, a focused FSDP experiment and a toy TP layer. CP,
    PP, EP and multidimensional parallelism are taught through cost models, design
    exercises and production-code reading—not a fragile reimplementation of a
    production stack.

!!! note "Assessment milestone"
    Intermediate reproduction check-in.

### Block 7 — Autoregressive decoding

**3 sessions · 3h45 · [Module 9](modules/09-kv-cache.md) ([slides](slides/09-kv-cache.pdf){ target="_blank" }, sessions 25–27)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 25 | Theory | Naive decoding, prefill/decode separation, KV cache, MHA/MQA/GQA and memory bandwidth bottlenecks | Decode cost model |
| 26 | Practice | Implement dynamic KV cache in the course model | Cached decoder |
| 27 | Practice | Test mathematical equivalence and benchmark latency, memory and decode throughput | Decode benchmark report |

!!! note "Assessment milestone"
    Reproduction results and limitations reviewed.

## Phase 3 — Serve and synthesize

### Block 8 — Serving and final demonstration

**5 sessions · 6h15 · [Module 10](modules/10-serving.md) ([slides](slides/10-serving.pdf){ target="_blank" }, sessions 28–29), [Module 11](modules/11-frontier-architectures.md) ([slides](slides/11-frontier-architectures.pdf){ target="_blank" }, session 30), synthesis (sessions 31–32)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 28 | Theory | Continuous batching, paged KV cache, chunked prefill, prefix caching and serving schedulers | Serving-system model |
| 29 | Practice | Serve the course checkpoint with a high-throughput engine; benchmark TTFT, ITL, throughput and memory | Serving benchmark |
| 30 | Theory | Speculative decoding, DeepSeek MLA/MoE/MTP, encoder-free multimodality, linear attention/SSMs | Frontier architecture analysis |
| 31 | Presentation | Final team demonstrations and technical defense | Presentation |
| 32 | Presentation | Comparative postmortem and course synthesis | Final report and repository |

!!! success "Assessment milestone"
    **Code submission and final project presentation** (see
    [Reproduction project](reproduction-project.md)).
