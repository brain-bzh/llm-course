# Assessment

Raw model quality is deliberately not the primary grade. A lucky run or a large
checkpoint is weaker evidence than a correct implementation and a controlled
technical argument.

| Component | Weight | Purpose |
| --- | ---: | --- |
| Pre-session quizzes | 15% | Encourage preparation and reveal misconceptions early |
| Mid-course paper presentation | 20% | Assess understanding of a large-scale LLM training or systems technique |
| Reproduction project | 65% | Assess the ability to verify a precise claim with code and controlled evidence |

The paper presentation and reproduction project are two stages of the same
work: a group first explains a paper and identifies a claim worth testing,
then reproduces or verifies a carefully scoped part of that paper. Full
requirements and rubrics are on the [reproduction project](reproduction-project.md)
page.

## Pre-session quizzes — 15%

Quizzes are mandatory, and weighted enough to matter, but they stay well below
either project component — they check preparation, not mastery.

- One short online quiz before each teaching day (a day may contain several
  1h15 sessions).
- Approximately 5-10 minutes and 3-5 questions.
- Questions check prerequisite reading, core concepts and interpretation of a
  small trace, figure or result.
- All but the lowest one or two scores count, to allow for one absence or
  technical problem.

## Paper presentation and reproduction project — 85%

See [Reproduction project](reproduction-project.md) for the full breakdown of
the mid-course paper presentation (20%) and the final reproduction project
(65%), including deliverables, minimum experimental expectations and rubrics.

## What the continuous project is graded on

The [continuous project](project.md) is the shared in-class system every team
builds; it is the laboratory where the skills assessed above are learned and
exercised, not a separately weighted component. Correctness and evidence from
that system still matter because the reproduction project and its defense draw
on it directly:

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
- throughput numbers without batch, prompt and generation-length distributions;
- fabricated experiments, logs, citations or results — this is academic
  misconduct, whether produced manually or by an AI system.
