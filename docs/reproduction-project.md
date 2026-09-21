# Reproduction project

This page describes the two assessed team milestones that sit on top of the
[continuous project](project.md): the mid-course paper presentation and the
final reproduction project. The continuous project is the shared laboratory
where everyone builds the same system; the reproduction project is where each
team applies those skills to verify one claim from a paper of its choice.

```text
select a paper and a claim
        ↓
mid-course paper presentation (20%)
        ↓
scale the claim down and reproduce it with the course toolkit
        ↓
reproduction project: code, report, defense (65%)
```

## Paper selection

Groups select a recent conference paper involving a technique used in
large-scale LLM training or systems — data selection, optimization, memory
reduction, parallelism, communication, checkpointing or training efficiency are
all suitable. Selection happens gradually over the first few sessions (see the
[schedule](schedule.md) for exact milestones):

1. teams begin identifying candidate papers while building the baseline model;
2. a paper shortlist is due once the data pipeline is under way;
3. one paper and one or two target claims are approved before the mid-course
   presentation.

## Mid-course paper presentation — 20%

Each group presents its selected paper before starting the main scaling and
serving work, so that the rest of the course gives them concrete tools to
revisit the claim. The presentation should answer five questions:

1. What problem does the paper address?
2. What are its one or two central claims?
3. How does the proposed technique work?
4. What evidence supports the claims, and what are its limitations?
5. Which claim could the group test meaningfully with the available resources?

A tentative format is **8 minutes of presentation and 4 minutes of questions**
per group, adjusted once enrollment and group count are known. The
presentation should be understandable to classmates who have not read the
paper.

| Criterion | Weight within component |
| --- | ---: |
| Group explanation of the claims and mechanism | 25% |
| Group analysis of evidence and limitations | 15% |
| Group reproduction proposal | 10% |
| Individual answers and technical discussion | 50% |

This stage evaluates understanding, not experimental results. Its main output
is an approved, testable reproduction question.

## Reproduction project — 65%

Each group reproduces or verifies **one precise claim** from its selected
paper. The goal is not to retrain a frontier model or reproduce every table.
Students must scale the claim down intelligently while preserving the
mechanism under study. Valid approaches include:

- testing a systems mechanism on a small model or minimal distributed setup;
- measuring a memory or runtime claim with controlled synthetic workloads;
- reproducing a trend with a smaller model, dataset or training budget;
- implementing a focused operator or scheduling idea and comparing it with a
  baseline;
- checking a theoretical implication through a toy derivation and numerical
  experiment.

### In-class reproduction studios

To ensure teams make steady progress and receive hands-on mentoring under
instructor guidance, four dedicated **Reproduction Project Studio** sessions
are embedded directly into the second half of the course (see [schedule](schedule.md)):

- **Studio 1 (Session 17):** Codebase setup, baseline verification, and dataset pipeline sanity checks.
- **Studio 2 (Session 24):** Scaling runs, cluster job troubleshooting, and experimental failure triage.
- **Studio 3 (Session 27):** Claim verification, ablation synthesis, and figure generation.
- **Studio 4 (Session 29):** Presentation timing rehearsals, slide polish, and technical defense dry runs.

### Deliverables

1. **Code repository** with setup instructions, configurations and commands
   needed to reproduce the reported result.
2. **Short technical report** describing the claim, protocol, controls,
   results, limitations and relation to the course. A tentative limit is five
   pages excluding references and appendices.
3. **Final presentation and technical defense** showing what was reproduced,
   what differed from the paper, and what the group learned.

### Minimum experimental expectations

- an explicit target claim and a justified reduced-scale setting;
- a working baseline and at least one meaningful comparison;
- correctness checks before performance or quality claims;
- declared hardware, software, seeds, configurations and measurement method;
- at least one ablation, stress test or documented failure mode;
- plots or tables with interpretation, not only raw logs;
- a candid account of negative results and discrepancies;
- clear attribution of external code and a statement of any AI-assisted work.

One strong, well-controlled experiment is preferable to several superficial
ones. A rigorous negative result can be fully successful if the group explains
why the original claim did not transfer to the reduced setting.

| Criterion | Weight within component | Share of final grade |
| --- | ---: | ---: |
| Team artifact: correctness and faithfulness | 15% | 9.75% |
| Team artifact: experimental design and controls | 15% | 9.75% |
| Team artifact: results and interpretation | 10% | 6.5% |
| Team artifact: reproducibility and code quality | 10% | 6.5% |
| Individual technical defense | 25% | 16.25% |
| Individual diagnosis and adaptation task | 15% | 9.75% |
| Individual decision record and reflection | 10% | 6.5% |

See [Assessment](assessment.md) for how this fits into the overall course
grade, and [Continuous project](project.md) for the shared system that
provides the skills and code this project builds on.
