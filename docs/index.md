# Training Language Models

## From first principles to efficient serving

An intensive, implementation-led course on the complete language-model
pipeline. We begin with raw text and a minimal Transformer, make training fast
and distributed, and finish by serving the resulting model efficiently.

The course follows one question throughout:

> **Given limited data and compute, where are we wasting resources, and how can
> we prove that an improvement is real?**

## Course at a glance

<div class="course-grid" markdown>

<div class="course-card" markdown>

### 40 hours

32 × 1h15 across eight intensive teaching days.

</div>

<div class="course-card" markdown>

### 13 modules

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
- construct controlled data-selection experiments under a fixed token budget;
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

Long-running jobs execute between teaching days. Contact time is reserved for
implementation, diagnosis and interpretation—not watching progress bars.

[View the complete schedule](schedule.md){ .md-button .md-button--primary }
[Read the project specification](project.md){ .md-button }
