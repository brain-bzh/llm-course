# Training and Scaling Language Models

## From first principles to efficient serving

An intensive, implementation-led course on the complete language-model
pipeline. We begin with raw text and a minimal Transformer, make training fast
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
- build a zero-copy data ingestion pipeline from raw web extraction to memory-mapped tokens;
- measure memory, throughput and model FLOP utilization before optimizing;
- explain and test DDP, FSDP and a minimal tensor-parallel layer;
- choose a parallelization strategy from model size, sequence length and
  interconnect constraints;
- implement a KV cache and explain why autoregressive decode is memory-bound;
- benchmark a serving system using time to first token, inter-token latency,
  throughput and memory.

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

[View the complete schedule](schedule.md){ .md-button .md-button--primary }
[Read the project specification](project.md){ .md-button }
