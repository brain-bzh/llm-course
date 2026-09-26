<div class="course-hero" markdown>

<p class="course-eyebrow">BRAIN team · IMT Atlantique</p>

# Training and Scaling Language Models

<p class="course-lead">From first principles to efficient serving.</p>

<ul class="course-pipeline">
  <li><span>Raw text</span></li>
  <li><span>Transformer</span></li>
  <li><span>Single GPU</span></li>
  <li><span>DDP · FSDP · TP</span></li>
  <li><span>Serving</span></li>
</ul>

</div>

An intensive, implementation-led course on the complete language-model
pipeline, designed by the [BRAIN team](https://www.imt-atlantique.fr/en/research-innovation/teams/brain) for [IMT Atlantique](https://www.imt-atlantique.fr/en).
We begin with raw text and a minimal Transformer, make training fast
and distributed, and finish by serving the resulting model efficiently.

The course follows one question throughout:

> **Given finite compute, memory bandwidth, and interconnect limits: where is our
> system bottlenecked, and how do we prove an optimization actually scales?**

## Course at a glance

<div class="course-grid" markdown>

<div class="course-card" markdown>

### 40 hours

32 × 1h15 sessions across several intensive teaching days.

</div>

<div class="course-card" markdown>

### 11 modules

Every concept becomes code or a controlled measurement.

</div>

<div class="course-card" markdown>

### One evolving system

Raw text → trained model → distributed training → efficient serving.

</div>

</div>

## Learning outcomes

By the end of the course, students should be able to:

- trace the path from raw documents to next-token loss;
- implement and debug a small decoder-only Transformer and training loop;
- trace raw-text ingestion and build validated token shards with a memory-mapped loader;
- measure memory, throughput and model FLOP utilization before optimizing;
- explain and test DDP, FSDP and a minimal tensor-parallel layer;
- choose a parallelization strategy from model size, sequence length and
  interconnect constraints;
- implement a KV cache and explain why autoregressive decode is memory-bound;
- understand a serving system using time to first token, inter-token latency,
  throughput and memory.

## Scope and preparation

This course focuses on pretraining and the systems used to train and serve
language models. Post-training turns a pretrained model into a task- or
instruction-following model; SFT, preference optimization, retrieval, and agents
are outside the implementation scope here.

Students should already be comfortable with Python, matrix multiplication,
probability and cross-entropy, differentiation, and basic PyTorch training.
The opening practical sessions reinforce tensor shapes and module composition,
but do not replace an introductory deep-learning course.

## Teaching principle

Theory and practice alternate throughout the course. A concept is not complete
when students can repeat its definition. It is complete when they can implement
it, validate it, measure it and explain the trade-off it introduces.

Long-running jobs execute between sessions. Contact time is reserved for
implementation, diagnosis and interpretation—not watching progress bars.

Alongside this shared system, each team also selects a recent paper, presents
it mid-course, and spends the second half of the course reproducing or
verifying one of its claims — see the
[reproduction project](reproduction-project.md).

[Course presentation (PDF)](slides/00-introduction.pdf){ .md-button .md-button--primary target="_blank" }
[View the complete schedule](schedule.md){ .md-button }
[Read the project specification](project.md){ .md-button }
