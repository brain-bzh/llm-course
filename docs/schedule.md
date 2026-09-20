# Schedule

The course runs as **32 sessions of 1h15** (40 contact hours): 12 theory
sessions, 16 guided-implementation sessions, 2 mid-course paper-presentation
sessions and 2 final presentation/synthesis sessions. Sessions are grouped into
**8 blocks**, and blocks into three phases. A block groups sessions that belong
to the same module or milestone — it is not a fixed calendar unit. Depending on
the institutional timetable, a block may occupy part of a day, a full day, or
be split across more than one day.

| Phase | Blocks | Sessions | Contact time | Main outcome |
| --- | ---: | ---: | ---: | --- |
| Build | 3 | 12 | 15h | Tokenizer, minimal GPT, baseline checkpoint and approved paper claim |
| Select and scale | 3 | 12 | 15h | Mid-course presentation, selected data, optimized trainer and scaling evidence |
| Serve and synthesize | 2 | 8 | 10h | Cached decoder, serving benchmark and final defense |

Each module (the 13 topics listed on the [module pages](modules/index.md))
spans two to five sessions, matched to how much theory and practice the topic
needs — a module is a unit of content, not a unit of time. The table below
shows exactly which sessions cover which module.

## Phase 1 — Build a language model

### Block 1 — Model and training loop

**5 sessions · 6h15 · Module 1 (sessions 1-3), Module 2 (sessions 4-5)**

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

**2 sessions · 2h30 · Module 3 (sessions 6-7)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 6 | Theory | BPE, vocabulary trade-offs, document boundaries, packing, shuffling and splits | Dataset design |
| 7 | Practice | Train a tokenizer and convert documents into packed token sequences | Tokenizer and initial shards |

!!! note "Assessment milestone"
    Paper shortlist due.

### Block 3 — Baseline training and data selection

**5 sessions · 6h15 · Module 3 continued (session 8), Module 4 (sessions 9-10), Module 5 (sessions 11-12)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 8 | Practice | Finish the dataloader; test shuffling, boundaries and throughput | Validated data pipeline |
| 9 | Theory | Initialization, validation loss, sampling, checkpoint selection and failure diagnosis | Baseline protocol |
| 10 | Practice | Launch and inspect the baseline run | Baseline checkpoint |
| 11 | Theory | Filtering, deduplication, quality, loss-based selection, diversity and mixtures | Selection hypotheses |
| 12 | Practice | Implement random, heuristic and deduplicated baselines | Candidate data subsets |

**Between sessions:** score the corpus and run the controlled
baseline-versus-selection experiment with an identical token budget; results
are discussed at the start of Block 5.

!!! note "Assessment milestone"
    Paper and one or two target reproduction claims approved.

## Phase 2 — Select and scale

### Block 4 — Mid-course paper presentations

**2 sessions · 2h30**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 13 | Presentation | Group paper presentations: problem, central claims, mechanism, evidence and reproduction proposal | Presentation and peer questions |
| 14 | Presentation | Remaining group presentations and cross-group discussion | Approved reproduction protocol per team |

!!! success "Assessment milestone"
    **Mid-course paper presentation** (see [Reproduction project](reproduction-project.md)).

### Block 5 — Data experiments, profiling and single-GPU optimization

**5 sessions · 6h15 · Module 5 continued (discussion), Module 6 (sessions 15-16), Module 7 (sessions 17-18), Module 8 (session 19)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 15 | Practice | Discuss the selection-strategy comparison; FLOPs, memory, arithmetic intensity, MFU and profiler tooling; profile the baseline and identify the actual bottleneck | Selected corpus, evidence and baseline performance report |
| 16 | Practice | Apply and measure bf16, SDPA, compilation or fusion one change at a time | Optimized trainer |
| 17 | Theory | DDP, all-reduce, distributed sampling, accumulation, `no_sync` and global batches | DDP execution model |
| 18 | Practice | Convert the trainer to DDP and verify equivalence and accounting | Correct multi-GPU trainer |
| 19 | Theory | FSDP and ZeRO: sharding, collectives, memory and checkpointing | Sharding plan |

!!! note "Assessment milestone"
    Reproduction protocol and baseline fixed.

### Block 6 — Sharded and multidimensional parallelism

**5 sessions · 6h15 · Module 8 continued (session 20), Module 9 (sessions 21-22), Module 10 (sessions 23-24)**

| Session | Mode | Focus | Output |
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

!!! note "Assessment milestone"
    Intermediate reproduction check-in.

### Block 7 — Autoregressive decoding

**3 sessions · 3h45 · Module 11 (sessions 25-27)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 25 | Theory | Naive decoding, prefill/decode, KV cache, MHA/MQA/GQA and memory bandwidth | Decode cost model |
| 26 | Practice | Add a KV cache to the course model | Cached decoder |
| 27 | Practice | Test equivalence and benchmark latency, memory and tokens per second | Decode benchmark |

!!! note "Assessment milestone"
    Reproduction results and limitations reviewed.

## Phase 3 — Serve and synthesize

### Block 8 — Serving and final demonstration

**5 sessions · 6h15 · Module 12 (sessions 28-29), Module 13 (session 30), synthesis (sessions 31-32)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 28 | Theory | Continuous batching, paged KV cache, chunked prefill, prefix caching and scheduling | Serving-system model |
| 29 | Practice | Serve the checkpoint and measure TTFT, ITL, throughput and memory | Serving benchmark |
| 30 | Theory | MoE, long context, state-space models, multimodality and diffusion-style LMs | Architecture comparison |
| 31 | Presentation | Final team demonstrations and technical defense | Presentation |
| 32 | Presentation | Comparative postmortem and course synthesis | Final report and repository |

!!! success "Assessment milestone"
    **Code submission and final project presentation** (see
    [Reproduction project](reproduction-project.md)).
