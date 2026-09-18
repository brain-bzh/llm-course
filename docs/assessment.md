# Assessment

Raw model quality is deliberately not the primary grade. A lucky run or a large
checkpoint is weaker evidence than a correct implementation and a controlled
technical argument.

| Component | Weight | What is evaluated |
| --- | ---: | --- |
| Correctness | 25% | Model, data pipeline, training loop, distributed behavior and checkpoint recovery |
| Experimental discipline | 25% | Controls, metrics, reproducibility, interpretation and treatment of failed runs |
| Data-selection study | 20% | Quality of hypotheses, fairness of comparison and evidence for the chosen subset |
| Systems performance | 20% | Profiling, training efficiency, memory analysis and serving benchmarks |
| Final technical defense | 10% | Precision of explanations, trade-offs, limitations and proposed next experiment |

## Minimum passing requirements

A project cannot pass solely through presentation quality. It must demonstrate:

- a training loop that learns and resumes from a checkpoint;
- a valid held-out evaluation that does not overlap the selected training data;
- at least one controlled data-selection comparison;
- one measured training-performance improvement;
- a correct distributed or sharded-training experiment;
- a KV-cache equivalence test;
- a serving benchmark with a declared workload.

## What does not count as evidence

- screenshots without the configuration and commit that produced them;
- GPU utilization presented as equivalent to model FLOP utilization;
- faster training obtained by silently changing model, sequence length or token
  budget;
- a data-selection result evaluated on contaminated or overlapping data;
- a distributed run whose rank-local metrics are reported as global values;
- cached generation that is faster but does not reproduce uncached logits;
- throughput numbers without batch, prompt and generation-length distributions.

