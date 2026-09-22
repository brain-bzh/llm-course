# Schedule

The course runs as **32 sessions of 1h15** (40 contact hours): 12 theory
sessions, 12 guided-implementation and systems showcase sessions, 4 dedicated
reproduction project studio sessions, 2 mid-course paper-presentation sessions,
and 2 final presentation/synthesis sessions. Sessions are grouped into
**8 blocks**, and blocks into three phases. A block groups sessions that belong
to the same module or milestone — it is not a fixed calendar unit. Depending on
the institutional timetable, a block may occupy part of a day, a full day, or
be split across more than one day.

| Phase | Blocks | Sessions | Contact time | Main outcome |
| --- | --- | ---: | ---: | --- |
| Build | 3 | 12 | 15h | Minimal GPT architecture, end-to-end pretraining data pipeline, documented baseline checkpoint, and profiled/optimized single-GPU trainer |
| Scale | 3 | 12 | 15h | Mid-course presentation, distributed DDP/FSDP scaling, multidimensional parallelism strategy, and Project Studios 1 & 2 |
| Serve and synthesize | 2 | 8 | 10h | KV-cached decoder, serving systems & live vLLM demonstration, frontier architecture analysis, Project Studios 3 & 4, and final defense |

Each module (the 11 topics listed on the [module pages](modules/index.md))
spans two to five sessions, matched to how much theory and practice the topic
needs — a module is a unit of content, not a unit of time. The table below
shows exactly which sessions cover which module.

## Phase 1 — Build a language model

!!! tip "Course presentation"
    Download the [Course presentation slides (Lecture 0 PDF)](slides/00-introduction.pdf){ target="_blank" } covering course objectives, schedule, continuous laboratory, reproduction project, and assessment.

### Block 1 — Model and training loop

**6 sessions · 7h30 · [Module 1: Transformer from first principles](modules/01-transformer.md) ([slides](slides/01-transformer.pdf){ target="_blank" }, sessions 1–4), [Module 2: Training-loop anatomy and baseline GPT](modules/02-training-loop.md) ([slides](slides/02-training-loop.pdf){ target="_blank" }, sessions 5–6)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 1 | Theory | Next-token prediction objective $P(x_1, \dots, x_T)$, recurrent sequential bottlenecks vs Transformer parallel revolution; Transformer anatomy: token and positional embeddings, causal self-attention, residual stream, normalization (pre-LN/RMSNorm), MLP, and LM head with weight tying | Architecture specification & mathematical formulation |
| 2 | Practice | PyTorch tensor mechanics & attention primitives: tensor broadcasting, stride/layout, `nn.Module` inspection; implement token and learned positional embeddings, scaled dot-product attention, and causal triangular masking | PyTorch foundations & tested causal attention primitives |
| 3 | Practice | Multi-Head Attention (MHA) & Transformer sublayers: implement head projection splitting and output projection ($W_O$); build pre-LN LayerNorm / RMSNorm and the MLP block ($4d$ expansion, GELU/SwiGLU, contraction) | Verified Multi-Head Attention and Transformer sublayer modules |
| 4 | Practice | Assemble minimal GPT & checkpoint parity: integrate Transformer blocks, residual highways, and a tied LM head; load official GPT-2 weights through the provided converter; reproduce reference logits and a verified next-token log-probability | Student-built GPT implementation reproducing the official GPT-2 forward pass |
| 5 | Theory | Training-loop anatomy: autograd DAG, operation order & silent failure catalog, cross-entropy loss, AdamW parameter grouping (2D weights vs 1D biases/norms), micro-batch accumulation ($1/A$), gradient norm clipping, warmup & cosine LR scheduling, and mixed precision (AMP) | Training-loop checklist & state transition diagram |
| 6 | Practice | Implement target shifting, optimizer grouping, backward/update ordering, gradient clipping, and deterministic checkpoint recovery; overfit a small NanoLM on one fixed batch (`loss < 0.1`) | Minimal trainable NanoLM with verified checkpoint round-trip |

!!! success "Exit criterion"
    The student implementation must first reproduce GPT-2 inference, then its
    randomly initialized small configuration must overfit one fixed batch. If
    either check fails, the team does not proceed to larger training.

!!! note "Assessment milestone"
    Teams begin identifying candidate papers for the mid-course presentation
    and reproduction project.

### Block 2 — From raw text to tensors

**2 sessions · 2h30 · [Module 3: Pretraining data pipeline](modules/03-data-pipeline.md) ([slides](slides/03-data-pipeline.pdf){ target="_blank" }, sessions 7–8)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 7 | Theory | Raw web extraction (WARC vs WET, boilerplate tax), heuristic compute-conservation filtering, subword tokenization trade-offs, byte-level BPE from scratch, regex pre-tokenization splitting, sequence packing with `<|endoftext|>`, and zero-copy binary sharding with `np.memmap` | Ingestion pipeline design & tokenization trade-off analysis |
| 8 | Practice | Implement heuristic quality filters, train byte-level BPE tokenizer from scratch, measure compression ratio and roundtrip fidelity, pack clean documents into binary shards, and build/verify high-throughput memory-mapped DataLoader ($Y_{b,t} = X_{b,t+1}$) | Validated end-to-end data pipeline and binary shards |

!!! note "Assessment milestone"
    Paper shortlist due.

### Block 3 — Baseline training, profiling and single-GPU optimization

**4 sessions · 5h00 · [Module 2: Training-loop anatomy and baseline GPT](modules/02-training-loop.md) continued (sessions 9–10), [Module 4: Single-GPU performance](modules/04-single-gpu.md) ([slides](slides/04-single-gpu.pdf){ target="_blank" }, sessions 11–12)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 9 | Theory | GPT parameter scaling ($12Ld^2$), depth-scaled initialization ($1/\sqrt{2L}$ to preserve residual stream), pre-flight checks ($\mathcal{L}_{\text{init}} \approx \log V$), token accounting, curve triage, and sampling diagnostics | Baseline training protocol & pre-flight checklist |
| 10 | Practice | Launch baseline training run on packed dataset shards; log training/validation curves; verify checkpoint resume determinism; evaluate qualitative samples across prompt suite | Documented baseline checkpoint & curves |
| 11 | Theory | Hardware model of one GPU (SMs, Tensor Cores, warps, divergence, memory hierarchy), roofline model (arithmetic intensity, compute vs memory bound), memory decomposition, and profiler traces | Bottleneck hypothesis & profiler plan |
| 12 | Practice | Profile baseline trainer with PyTorch Profiler / Nsight; systematically benchmark bf16, SDPA/FlashAttention-2, and `torch.compile`; measure throughput, peak VRAM, and MFU | Optimized single-GPU trainer & profiling report |

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

### Block 5 — Distributed data parallelism, memory sharding & Project Studio 1

**5 sessions · 6h15 · [Module 5: Distributed data parallelism](modules/05-ddp.md) ([slides](slides/05-ddp.pdf){ target="_blank" }, sessions 15–16), Project Studio 1 (session 17), [Module 6: FSDP and ZeRO](modules/06-fsdp.md) ([slides](slides/06-fsdp.pdf){ target="_blank" }, sessions 18–19)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 15 | Theory | DDP execution model: one process per GPU, parameter broadcast, synchronous SGD gradient averaging derivation, ring all-reduce, gradient bucketing, backward overlap, and global token accounting | DDP architecture specification |
| 16 | Practice | DDP showcase & scaling benchmark: launch multi-GPU trainer with `torchrun`, verify rank synchronization and gradient accumulation via `no_sync`, and measure linear speedup limits | Verified multi-GPU trainer & scaling report |
| 17 | Project Studio | **Reproduction Studio 1 — Baseline & Environment Setup**: Dedicated in-class team hackathon; set up paper codebases, download/preprocess initial datasets, verify baseline compute requirements, and unblock execution bottlenecks with instructor mentoring | Functioning reproduction repository & baseline test run |
| 18 | Theory | FSDP and ZeRO: state redundancy in DDP ($16\Psi$ bytes/param), progressive state decomposition (ZeRO-1/2/3), all-gather and reduce-scatter collective timing, sharded residency vs temporary materialization, and auto-wrapping | Sharding plan & memory analysis |
| 19 | Practice | FSDP showcase & memory comparison: apply FSDP auto-wrap policy on course model; benchmark per-GPU peak VRAM and throughput against DDP across ZeRO stages | FSDP benchmark report |

!!! note "Assessment milestone"
    Reproduction protocol and baseline fixed.

### Block 6 — Advanced sharding, multidimensional parallelism & Project Studio 2

**5 sessions · 6h15 · [Module 6: FSDP and ZeRO](modules/06-fsdp.md) continued (session 20), [Module 7: Tensor parallelism](modules/07-tensor-parallelism.md) ([slides](slides/07-tensor-parallelism.pdf){ target="_blank" }, session 21), [Module 8: Context, pipeline, and expert parallelism](modules/08-context-pipeline.md) ([slides](slides/08-context-pipeline.pdf){ target="_blank" }, sessions 22–23), Project Studio 2 (session 24)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 20 | Practice | FSDP distributed checkpointing & TP showcase: verify sharded state save/resume determinism; inspect minimal two-rank column/row sharded MLP using Gloo/NCCL and verify numerical equivalence | Resilient sharded checkpointing & TP equivalence verification |
| 21 | Theory | Tensor and sequence parallelism: column/row matrix partitioning, intermediate collective elimination, GQA attention head divisibility, and sequence parallelism (SP) activation reduction | TP derivation & layout ledger |
| 22 | Theory | Context, pipeline & expert parallelism: Sequence Parallelism vs Context Parallelism, Ring Attention with online softmax, 1F1B vs AFAB bubble ratios ([interactive simulator](demos/pipeline-parallelism.html){ target="_blank" }), Expert Parallelism (EP) All-to-All communication, and outer-to-inner hierarchy ($\text{PP} \to \text{DP} \to \text{FSDP} \to \text{EP} \to \text{TP}$) | Parallelism strategy rules & cost models |
| 23 | Practice | Multidimensional cluster sizing: solve 5D scaling cases ($G = P \times D \times F \times E_P \times T_P \times C$); map parallel dimensions to physical cluster network topologies; defend an optimal configuration | Multidimensional scaling design report |
| 24 | Project Studio | **Reproduction Studio 2 — Scaling Experiments & Failure Diagnosis**: Dedicated team hackathon; launch scaling or ablation experiments on target claims, debug CUDA/distributed errors, and triage unexpected loss curves or throughput anomalies | Preliminary experimental results & ablation checkpoints |

!!! note "Implementation boundary"
    Students implement DDP, a focused FSDP experiment and a toy TP layer. CP,
    PP, EP and multidimensional parallelism are taught through cost models, design
    exercises and production-code reading—not a fragile reimplementation of a
    production stack.

!!! note "Assessment milestone"
    Intermediate reproduction check-in.

### Block 7 — Autoregressive decoding & Project Studio 3

**3 sessions · 3h45 · [Module 9: KV-cached decoding](modules/09-kv-cache.md) ([slides](slides/09-kv-cache.pdf){ target="_blank" }, sessions 25–26), Project Studio 3 (session 27)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 25 | Theory | Autoregressive decoding without cache, causality invariant, prefill vs decode separation, memory-bandwidth bound ($\sim 1\text{ FLOP/byte}$), and KV-cache shapes/byte accounting for MHA, MQA, and GQA | Decode cost model & arithmetic intensity analysis |
| 26 | Practice | KV-cached decoding showcase & benchmark: implement incremental KV-cache in the course model; assert exact logit/token equivalence to uncached generation; benchmark TTFT, inter-token latency, and memory bandwidth | Cached decoder & equivalence verification report |
| 27 | Project Studio | **Reproduction Studio 3 — Results Synthesis & Defense Prep**: Dedicated team hackathon; analyze reproduction claims against paper targets, characterize discrepancies and limitations, format evidence charts, and prepare final defense presentation | Draft presentation slides & verified result tables |

!!! note "Assessment milestone"
    Reproduction results and limitations reviewed.

## Phase 3 — Serve and synthesize

### Block 8 — Serving and final demonstration

**5 sessions · 6h15 · [Module 10: Serving systems](modules/10-serving.md) ([slides](slides/10-serving.pdf){ target="_blank" }, session 28), Project Studio 4 (session 29), [Module 11: Frontier architectures and efficient generation](modules/11-frontier-architectures.md) ([slides](slides/11-frontier-architectures.pdf){ target="_blank" }, session 30), synthesis (sessions 31–32)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 28 | Theory | Continuous batching, request lifecycle state machine, paged KV-cache (block tables, fragmentation), decode priority vs chunked prefill, prefix caching, serving metrics (TTFT, ITL, goodput under SLOs), [interactive serving simulator](demos/continuous-batching.html){ target="_blank" }, and instructor-led live vLLM serving demonstration | Serving-system architecture model & live benchmark observation |
| 29 | Project Studio | **Reproduction Studio 4 — Final Presentation & Defense Prep**: Dedicated team studio time; rehearse team presentation timing (8 min presentation + 4 min questions), finalize reproduction claim evidence charts, and anticipate individual technical defense questions with instructor coaching | Rehearsed presentation slides & defense readiness |
| 30 | Theory | Frontier architectures and efficient generation: speculative decoding & rejection sampling, diffusion models for text, DeepSeek innovations (Multi-Head Latent Attention MLA, DeepSeekMoE, MTP), encoder-free multimodality, and linear attention/SSMs (Mamba-2, GDN) | Frontier architecture analysis & compression evaluation |
| 31 | Presentation | Final team demonstrations and technical defenses: present paper reproduction evidence, systems measurements, and failure postmortems | Project presentation & technical defense |
| 32 | Presentation | Comparative course postmortem, cross-team architecture and parallelism synthesis, and final repository submission | Final project report and repository submission |

!!! success "Assessment milestone"
    **Code submission and final project presentation** (see
    [Reproduction project](reproduction-project.md)).
