# Schedule

The course runs as **32 sessions of 1h15** (40 contact hours): 11 scheduled theory sessions, one open session, 12 guided-implementation and systems showcase sessions, 4 dedicated
reproduction project studio sessions, 2 mid-course paper-presentation sessions,
and 2 final presentation/synthesis sessions. Sessions are grouped into
**8 blocks**, and blocks into three phases. A block groups sessions that belong
to the same module or milestone — it is not a fixed calendar unit. Depending on
the institutional timetable, a block may occupy part of a day, a full day, or
be split across more than one day.

| Phase | Blocks | Sessions | Contact time | Main outcome |
| --- | --- | ---: | ---: | --- |
| Build | 3 | 12 | 15h | Minimal GPT architecture, end-to-end pretraining data pipeline, documented baseline checkpoint, and profiled/optimized single-GPU trainer |
| Scale | 3 | 12 | 15h | Mid-course presentation, hand-written DDP and sharded AdamW, FSDP2 on CPU ranks, multidimensional parallelism strategy, and Project Studios 1 & 2 |
| Serve and synthesize | 2 | 8 | 10h | KV-cached decoder, serving systems and a prepared demonstration, an open content slot, Project Studio 3 (Parts 1 & 2), and final defense |

Each module (the 11 topics listed on the [module pages](modules/index.md))
spans one to four sessions, matched to how much theory and practice the topic
needs — a module is a unit of content, not a unit of time. The table below
shows exactly which sessions cover which module.

## January 2027 calendar

<div class="schedule-calendar-scroll">
<table class="schedule-calendar">
  <caption>Session dates and times · weekdays, January 4–22</caption>
  <thead><tr><th scope="col">Mon</th><th scope="col">Tue</th><th scope="col">Wed</th><th scope="col">Thu</th><th scope="col">Fri</th></tr></thead>
  <tbody>
    <tr><td><div class="calendar-day"><time datetime="2027-01-04">4</time></div></td><td><div class="calendar-day"><time datetime="2027-01-05">5</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>09:30–10:45</time><span class="calendar-session">S01</span>Kickoff &amp; Transformer map</li><li><time>11:00–12:15</time><span class="calendar-session">S02</span>Build the GPT foundation</li></ul></div><div class="calendar-half-day"><div class="calendar-period-label">Afternoon</div><ul class="calendar-events"><li><time>14:00–15:15</time><span class="calendar-session">S03</span>Implement multi-head attention</li><li><time>15:30–16:45</time><span class="calendar-session">S04</span>Assemble the GPT</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-06">6</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>09:30–10:45</time><span class="calendar-session">S05</span>Training-loop theory</li><li><time>11:00–12:15</time><span class="calendar-session">S06</span>Baseline training · theory</li></ul></div><div class="calendar-half-day"><div class="calendar-period-label">Afternoon</div><ul class="calendar-events"><li><time>14:00–15:15</time><span class="calendar-session">S07</span>Train a tiny NanoLM</li><li><time>15:30–16:45</time><span class="calendar-session">S08</span>Run baseline training</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-07">7</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>09:30–10:45</time><span class="calendar-session">S09</span>Pretraining data pipeline · theory</li><li><time>11:00–12:15</time><span class="calendar-session">S10</span>Build the data pipeline</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-08">8</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>09:30–10:45</time><span class="calendar-session">S11</span>Profile a GPU · theory</li><li><time>11:00–12:15</time><span class="calendar-session">S12</span>Optimize the single-GPU trainer</li></ul></div></td></tr>
    <tr><td><div class="calendar-day"><time datetime="2027-01-11">11</time></div></td><td><div class="calendar-day"><time datetime="2027-01-12">12</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>09:30–10:45</time><span class="calendar-session">S13</span>Distributed foundations &amp; DDP · theory</li><li><time>11:00–12:15</time><span class="calendar-session">S14</span>Write DDP by hand</li></ul></div><div class="calendar-half-day"><div class="calendar-period-label">Afternoon</div><ul class="calendar-events"><li><time>14:00–15:15</time><span class="calendar-session">S15</span>Lightning paper pitches</li><li><time>15:30–16:45</time><span class="calendar-session">S16</span>Whole-class discussion</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-13">13</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>11:00–12:15</time><span class="calendar-session">S17</span>Reproduction Studio 1</li></ul></div><div class="calendar-half-day"><div class="calendar-period-label">Afternoon</div><ul class="calendar-events"><li><time>14:00–15:15</time><span class="calendar-session">S18</span>FSDP and ZeRO · theory</li><li><time>15:30–16:45</time><span class="calendar-session">S19</span>Shard AdamW; run FSDP2</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-14">14</time></div></td><td><div class="calendar-day"><time datetime="2027-01-15">15</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>08:00–09:15</time><span class="calendar-session">S20</span>Tensor &amp; sequence parallelism · theory</li><li><time>09:30–10:45</time><span class="calendar-session">S21</span>Checkpoint formats &amp; implement TP</li></ul></div></td></tr>
    <tr><td><div class="calendar-day"><time datetime="2027-01-18">18</time></div><div class="calendar-half-day"><div class="calendar-period-label">Afternoon</div><ul class="calendar-events"><li><time>15:30–16:45</time><span class="calendar-session">S22</span>CP, PP &amp; EP · theory</li><li><time>17:00–18:15</time><span class="calendar-session">S23</span>Multidimensional cluster sizing</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-19">19</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>11:00–12:15</time><span class="calendar-session">S24</span>Reproduction Studio 2</li></ul></div><div class="calendar-half-day"><div class="calendar-period-label">Afternoon</div><ul class="calendar-events"><li><time>15:30–16:45</time><span class="calendar-session">S25</span>KV-cached decoding · theory</li><li><time>17:00–18:15</time><span class="calendar-session">S26</span>KV-cache showcase &amp; benchmark</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-20">20</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>09:30–10:45</time><span class="calendar-session">S27</span>Open · content to be decided</li><li><time>11:00–12:15</time><span class="calendar-session">S28</span>Serving systems</li></ul></div><div class="calendar-half-day"><div class="calendar-period-label">Afternoon</div><ul class="calendar-events"><li><time>14:00–15:15</time><span class="calendar-session">S29</span>Reproduction Studio 3 · Part 1</li><li><time>15:30–16:45</time><span class="calendar-session">S30</span>Reproduction Studio 3 · Part 2</li></ul></div></td><td><div class="calendar-day"><time datetime="2027-01-21">21</time></div></td><td><div class="calendar-day"><time datetime="2027-01-22">22</time></div><div class="calendar-half-day"><div class="calendar-period-label">Morning</div><ul class="calendar-events"><li><time>09:30–10:45</time><span class="calendar-session">S31</span>Final project presentations</li><li><time>11:00–12:15</time><span class="calendar-session">S32</span>Course synthesis &amp; submission</li></ul></div></td></tr>
  </tbody>
</table>
</div>

## Phase 1 — Build a language model

!!! tip "Course presentation"
    The [Course presentation slides (Lecture 0 PDF)](slides/00-introduction.pdf){ target="_blank" } are available for an overview of course objectives, schedule, continuous laboratory, reproduction project, and assessment.

### Block 1 — Build a GPT

**4 sessions · 5h · [Module 1: Transformer from first principles](modules/01-transformer.md) ([slides](slides/01-transformer.pdf){ target="_blank" }, sessions 1–4)**

#### Working block 1 — Kickoff, model map, and start building (sessions 1–2 · 2h30)

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 1 | Kickoff + theory | Brief course presentation; next-token prediction ([interactive demo](demos/next-token.html){ target="_blank" }), RNN vs Transformer, embeddings, attention, residual stream, normalization, MLP and LM head | Course workflow and Transformer architecture map |
| 2 | Practice | Set up the starter; build token and learned position embeddings, pre-LN residual structure, and the GPT-2 MLP | Non-attention components with shape checks |

#### Working block 2 — Multi-head attention through logits (sessions 3–4 · 2h30)

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 3 | Practice | Implement QKV projections, scaled dot-product attention, causal masking, head splitting/merging, and output projection | Tested multi-head attention module |
| 4 | Practice | Assemble the Transformer and tied LM head; inspect logits and compare with official GPT-2 reference logits | Student-built GPT reproducing reference logits |

### Block 2 — Training loop and baseline

**4 sessions · 5h · [Module 2: Training-loop anatomy and baseline GPT](modules/02-training-loop.md) ([slides](slides/02-training-loop.pdf){ target="_blank" }, sessions 5–8)**

On January 6, the morning is one continuous theory block; the afternoon combines
implementation and baseline experiments. Use the same Module 2 deck throughout
the morning.

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 5 | Theory | Optimizer 101: loss, activations, gradients, SGD, learning rate and Adam state; forward/backward and the base loop | Trace from one batch to one update |
| 6 | Theory | Accumulation, clipping and learning-rate scheduling; FLOP/memory estimates; baseline design, token budgets, held-out evaluation and recovery | Update-order diagram and baseline protocol |
| 7 | Practice | Implement shifted targets, optimizer grouping, backward/update ordering and clipping; overfit one fixed NanoLM batch and check a checkpoint round-trip | Tiny NanoLM and verified round-trip |
| 8 | Practice | Use instructor-prepared disjoint JSONL documents to run the provided byte-token baseline; inspect losses and token budget and test CPU checkpoint recovery | Baseline report and checkpoint, or documented smoke-test fallback |

!!! note "Data dependency"
    The baseline script prepares its own byte-token shards from the supplied
    documents. The next morning's pipeline lab teaches students how to build
    and validate a richer data pipeline; it is not a prerequisite for this run.
    Longer baseline training can continue between sessions.

!!! note "Assessment milestone"
    Teams begin identifying candidate papers for the mid-course presentation
    and reproduction project.

### Block 3 — Data pipeline and single-GPU performance

**4 sessions · 5h · [Module 3: Pretraining data pipeline](modules/03-data-pipeline.md) ([slides](slides/03-data-pipeline.pdf){ target="_blank" }, sessions 9–10), [Module 4: Single-GPU performance](modules/04-single-gpu.md) ([slides](slides/04-single-gpu.pdf){ target="_blank" }, sessions 11–12)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 9 | Theory | Document sourcing and filtering; byte-level BPE, pre-tokenization, vocabulary trade-offs, packing, boundaries and binary shards | Pipeline design and tokenizer trade-offs |
| 10 | Practice | Check hand-traced BPE merges; implement deduplicate-then-split, packing with end-of-text boundaries, and shifted memmap batches; diagnose one deliberate data bug | Tested data path, split report and binary shards |
| 11 | Theory | CPU vs GPU execution of the course GPT, memory and roofline reasoning, kernel fusion, `torch.compile`, profiler traces and benchmarking | Bottleneck hypothesis and profiler plan |
| 12 | Practice | Implement FLOP, MFU and static-memory accounting; add and verify an SDPA path; run one warmed-up, synchronized ablation (precision, SDPA or compilation) with a profiler trace | Prediction-versus-measurement report for one controlled change |

**Between sessions:** continue the baseline as resources permit, then profile the
trainer across batch sizes before distributed scaling.

!!! note "Assessment milestone"
    Paper shortlist due; paper and one or two target reproduction claims are approved before the mid-course discussion.

## Phase 2 — Scale

### Block 4 — Distributed data parallelism and paper discussion

**4 sessions · 5h00 · [Module 5: Distributed data parallelism](modules/05-ddp.md) ([slides](slides/05-ddp.pdf){ target="_blank" }, sessions 13–14), mid-course paper presentation (sessions 15–16)**

Start with distributed foundations and DDP theory, followed by a short implementation showcase. Students then have
the distributed-training vocabulary and execution model to frame their paper
pitches and discuss reproduction claims.

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 13 | Theory | Distributed motivation, nodes/ranks/process groups, collective and communication costs; then DDP replicas, gradient averaging, global batch, buckets and overlap. Ring algorithms are optional depth. | Distributed vocabulary and DDP update specification |
| 14 | Practice | Implement data parallelism by hand (batch sharding, parameter broadcast, gradient all-reduce) on two CPU ranks; match a single-process global-batch run; break it on purpose; compare with `DistributedDataParallel` and `no_sync`; design a scaling benchmark | Exact update equivalence, failure explanations and scaling plan |
| 15 | Lightning presentations | A few-minute pitch per paper: problem, central claim, proposed mechanism, and key evidence; capture one candidate reproduction question per group | Shared set of candidate claims and questions |
| 16 | Whole-class discussion | Compare the claims together: what is worth understanding or testing, what evidence would settle it, and which questions are feasible to reproduce; teams agree on a direction | Reproduction question and initial protocol per team |

For session 13, budget approximately 20 minutes for foundations and 55 minutes
for DDP. Ring and tree algorithms remain optional reading on the Module 5 page; session
14 retains its full practical slot. For session 18, allow roughly 10 minutes to revisit DDP state, 45 minutes for
sharding, collectives and the unit lifecycle, and 20 minutes for wrapping and
comparison. Use precision and checkpoint slides as needed for the guided
exercise and later checkpoint discussion.

!!! success "Assessment milestone"
    **Mid-course paper presentation** (see [Reproduction project](reproduction-project.md)).

### Block 5 — Reproduction Studio 1 and memory sharding

**3 sessions · 3h45 · Project Studio 1 (session 17), [Module 6: FSDP and ZeRO](modules/06-fsdp.md) ([slides](slides/06-fsdp.pdf){ target="_blank" }, sessions 18–19)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 17 | Project Studio | **Reproduction Studio 1 — Baseline & Environment Setup**: Dedicated in-class team hackathon; set up paper codebases, download/preprocess initial datasets, verify baseline compute requirements, and unblock execution bottlenecks with instructor mentoring | Functioning reproduction repository & baseline test run |
| 18 | Theory | DDP state redundancy, progressive ZeRO sharding, all-gather and reduce-scatter, one FSDP unit through forward/backward, wrapping and transient memory. Precision and checkpoint details are reference material. | Sharding plan & memory analysis |
| 19 | Practice | Predict DDP/ZeRO state; implement a sharded AdamW with reduce-scatter and all-gather and match `torch.optim.AdamW`; run FSDP2 (`fully_shard`) on CPU ranks to observe the gather/reshard lifecycle and measure per-rank state | Verified sharded update and predicted-versus-measured state table with exclusions |

!!! note "Assessment milestone"
    Reproduction protocol and baseline fixed.

### Block 6 — Advanced sharding, multidimensional parallelism & Project Studio 2

**5 sessions · 6h15 · [Module 6: FSDP and ZeRO](modules/06-fsdp.md) continued (session 21), [Module 7: Tensor parallelism](modules/07-tensor-parallelism.md) ([slides](slides/07-tensor-parallelism.pdf){ target="_blank" }, sessions 20–21), [Module 8: Context, pipeline, and expert parallelism](modules/08-context-pipeline.md) ([slides](slides/08-context-pipeline.pdf){ target="_blank" }, sessions 22–23), Project Studio 2 (session 24)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 20 | Theory | Tensor and sequence parallelism: column/row matrix partitioning, intermediate collective elimination, GQA attention head divisibility, and sequence parallelism (SP) activation reduction | TP derivation & layout ledger |
| 21 | Practice | Save full and sharded checkpoints from two FSDP2 ranks and reload both in one process; implement column/row-parallel layers and their autograd collectives; verify two-rank forward/backward equivalence with Gloo | Resharded checkpoint check, format decisions and verified TP gradients |
| 22 | Theory | Context, pipeline & expert parallelism: Sequence Parallelism vs Context Parallelism, Ring Attention with online softmax, 1F1B vs AFAB bubble ratios ([interactive simulator](demos/pipeline-parallelism.html){ target="_blank" }), Expert Parallelism (EP) All-to-All communication, and placement based on exposed communication and topology ([parallelism composer](demos/parallelism-composer.html){ target="_blank" }) | Parallelism strategy rules & cost models |
| 23 | Practice | Multidimensional cluster sizing: count independent rank axes in scaling cases ($G = P \times T_P \times C \times D$); map parallel dimensions to physical cluster network topologies; defend a feasible configuration with explicit assumptions | Multidimensional scaling design report |
| 24 | Project Studio | **Reproduction Studio 2 — Scaling Experiments & Failure Diagnosis**: Dedicated team hackathon; launch scaling or ablation experiments on target claims, debug CUDA/distributed errors, and triage unexpected loss curves or throughput anomalies | Preliminary experimental results & ablation checkpoints |

!!! note "Implementation boundary"
    Students implement DDP, a sharded AdamW (ZeRO-1/2) and a toy TP layer by
    hand, and run PyTorch FSDP2 on CPU ranks; all of these run on a laptop with
    Gloo. CP, PP, EP and multidimensional parallelism are taught through cost
    models, design exercises and production-code reading—not a fragile
    reimplementation of a production stack. GPU measurements are extensions.

!!! note "Assessment milestone"
    Intermediate reproduction check-in.

## Phase 3 — Serve and synthesize

### Block 7 — Autoregressive decoding

**2 sessions · 2h30 · [Module 9: KV-cached decoding](modules/09-kv-cache.md) ([slides](slides/09-kv-cache.pdf){ target="_blank" }, sessions 25–26)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 25 | Theory | Full-prefix recomputation, prefill vs decode, KV-cache shapes, memory and bandwidth accounting for MHA, MQA and GQA | Decode cost model |
| 26 | Practice | Add a KV-cache path (with position and mask offsets) to the student model and write cached generation; check logits at every position and greedy tokens; benchmark prefill and decode separately | Cached decoder, equivalence tests and prefill/decode report |

### Block 8 — Serving, open slot and final demonstration

**6 sessions · 7h30 · [Module 10: Serving systems](modules/10-serving.md) ([slides](slides/10-serving.pdf){ target="_blank" }, session 28), open slot (session 27; content to be decided), Project Studio 3 (sessions 29–30), synthesis (sessions 31–32)**

| Session | Mode | Focus | Output |
| ---: | --- | --- | --- |
| 27 | Open | Content to be decided; [Module 11](modules/11-frontier-architectures.md) and its existing slides are a placeholder | To be defined |
| 28 | Theory | Request lifecycle, continuous batching, paged KV allocation, TTFT, ITL and throughput in a prepared demonstration ([simulator](demos/continuous-batching.html){ target="_blank" }); live vLLM is conditional on preflight | Annotated scheduling trace and benchmark interpretation |
| 29 | Project Studio | Reproduction Studio 3, Part 1: synthesize results and limitations, prepare evidence charts and draft the defense | Verified result tables and draft slides |
| 30 | Project Studio | Reproduction Studio 3, Part 2: rehearse the presentation, finalize evidence and prepare for questions | Rehearsed defense |
| 31 | Presentation | Final team presentations and technical defenses | Project presentation and technical defense |
| 32 | Presentation | Course synthesis and final repository submission | Final report and repository submission |

!!! success "Assessment milestone"
    **Code submission and final project presentation** (see
    [Reproduction project](reproduction-project.md)).
