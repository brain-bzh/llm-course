# Course objectives, organization and assessment

> Working note for the teaching team. The schedule, formats and weights below
> are proposals and can still be adjusted to the cohort size and available
> compute.

## Course objective

The course should teach students to reason about the complete lifecycle of a
language model, from data and model construction to distributed training and
serving. The central question is:

> Given limited data and compute, where are resources being wasted, and what
> evidence would show that an intervention actually helps?

By the end of the course, students should be able to:

- connect the main components of an LLM system: data, architecture, training,
  parallelism and inference;
- implement and debug the smallest faithful version of an important mechanism;
- turn a claim from the literature into a testable question;
- design a controlled experiment under limited compute;
- measure correctness, quality, memory, throughput and latency without
  confusing these quantities;
- interpret negative results and discrepancies rather than hiding them;
- communicate a technical claim, its evidence and its limitations clearly.

The aim is not to reward the largest training run. It is to develop the ability
to understand a large-scale technique, scale it down intelligently, and produce
convincing evidence about what it does.

## Course format and time allocation

The course contains **40 contact hours**, organized as **32 sessions of
1h15**. Sessions are grouped into **8 blocks**, spread across several teaching
days. The exact number of days depends on the institutional calendar and can
change without changing the course itself — what stays fixed is the 32
sessions and the 13 technical modules they cover (see the
[schedule](docs/schedule.md) and [modules](docs/modules/index.md) pages for the
session-by-session and module-by-module breakdown).

| Activity | Sessions | Time | Share |
| --- | ---: | ---: | ---: |
| Concepts and technical foundations | 12 | 15h | 38% |
| Guided implementation and experiments | 16 | 20h | 50% |
| Mid-course paper presentations | 2 | 2h30 | 6% |
| Final project presentations and synthesis | 2 | 2h30 | 6% |
| **Total** | **32** | **40h** | **100%** |

Long-running experiments should happen between sessions, not only between
days. Contact time is better used for implementation, diagnosis, comparison and
discussion than for waiting for jobs to finish.

For assessment purposes, a **teaching day** is the unit that matters, not the
1h15 session: a day may contain several sessions. The current proposal
therefore has one short pre-day quiz per teaching day, rather than 32 separate
quizzes; the best scores count, dropping the lowest one or two to allow for an
absence or technical problem.

## Tentative schedule

The schedule keeps one technical progression while making the two assessed
presentations explicit. Blocks are numbered in teaching order; how many
calendar days they span is a scheduling detail, not a course property.

| Block | Sessions | Main topics and activities | Assessment milestone |
| --- | ---: | --- | --- |
| 1 | 5 | Transformer anatomy; causal attention; minimal GPT; training-loop foundations | Teams begin identifying possible papers |
| 2 | 2 | Tokenization, document boundaries, packing, shuffling and data splits | Paper shortlist |
| 3 | 5 | Data pipeline; baseline training; evaluation; data selection | Paper and 1-2 target claims approved |
| 4 | 2 | Group paper presentations and discussion | **Mid-course paper presentation** |
| 5 | 5 | Controlled data experiments; profiling; single-GPU optimization; DDP foundations | Reproduction protocol and baseline fixed |
| 6 | 5 | DDP; FSDP/ZeRO; tensor parallelism; equivalence and scaling measurements | Intermediate reproduction check-in |
| 7 | 3 | Context/pipeline strategy; autoregressive decoding; KV-cache implementation and tests | Results and limitations reviewed |
| 8 | 5 | Serving systems; serving benchmark; architectures beyond dense Transformers; project presentations and synthesis | **Code submission and final project presentation** |

The paper presentation is deliberately placed before the main scaling and
serving work. It gives each group a precise claim to revisit as the relevant
course concepts and measurement tools are introduced.

## Assessment overview

| Component | Proposed weight | Purpose |
| --- | ---: | --- |
| Pre-session quizzes | 15% | Encourage preparation and reveal misconceptions early |
| Mid-course paper presentation | 20% | Assess understanding of a large-scale LLM training or systems technique |
| Reproduction project | 65% | Assess the ability to verify a precise claim with code and controlled evidence |
| **Total** | **100%** | |

The paper presentation and reproduction project are two stages of the same
work. A group first explains a paper and identifies a claim worth testing. It
then reproduces or verifies a carefully scoped part of that paper. This adapts
the spirit of the
[Poster Presentation Project](https://web.archive.org/web/20260410071224/https://training-large-models-course.github.io/hw/poster.pdf),
but separates understanding and experimental verification into two moments.

### 1. Pre-session quizzes - 15%

- One short online quiz before each teaching day (not before each session).
- Approximately 5-10 minutes and 3-5 questions.
- Questions check prerequisite reading, core concepts and interpretation of a
  small trace, figure or result; they should not require lengthy calculations.
- All but the lowest one or two scores count. This keeps the quizzes low stakes
  and allows for one absence or technical problem.
- Quiz results should inform the start of the session: common errors can be
  addressed briefly before new material begins.

The quizzes assess preparation and conceptual continuity. They should not
become a parallel examination system.

### 2. Mid-course paper presentation - 20%

Groups select a recent conference paper involving a technique used in
large-scale LLM training or systems. Suitable topics include data selection,
optimization, memory reduction, parallelism, communication, checkpointing or
training efficiency.

The presentation should answer five questions:

1. What problem does the paper address?
2. What are its one or two central claims?
3. How does the proposed technique work?
4. What evidence supports the claims, and what are its limitations?
5. Which claim could the group test meaningfully with the available resources?

A tentative format is **8 minutes of presentation and 4 minutes of questions**
per group, adjusted once enrollment and group count are known. The presentation
should be understandable to classmates who have not read the paper.

| Criterion | Weight within component |
| --- | ---: |
| Group explanation of the claims and mechanism | 25% |
| Group analysis of evidence and limitations | 15% |
| Group reproduction proposal | 10% |
| Individual answers and technical discussion | 50% |

This stage evaluates understanding, not experimental results. Its main output
is an approved, testable reproduction question.

### 3. Reproduction project - 65%

Each group reproduces or verifies **one precise claim** from its selected paper.
The goal is not to retrain a frontier model or reproduce every table. Students
must scale the claim down intelligently while preserving the mechanism under
study. Valid approaches include:

- testing a systems mechanism on a small model or minimal distributed setup;
- measuring a memory or runtime claim with controlled synthetic workloads;
- reproducing a trend with a smaller model, dataset or training budget;
- implementing a focused operator or scheduling idea and comparing it with a
  baseline;
- checking a theoretical implication through a toy derivation and numerical
  experiment.

#### Deliverables

1. **Code repository** with setup instructions, configurations and commands
   needed to reproduce the reported result.
2. **Short technical report** describing the claim, protocol, controls,
   results, limitations and relation to the course. A tentative limit is five
   pages excluding references and appendices.
3. **Final presentation and technical defense** showing what was reproduced,
   what differed from the paper, and what the group learned.

#### Minimum experimental expectations

The project should include:

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

## Relationship between the course and the project

The continuous in-class implementation provides shared exercises in building,
training, scaling and serving a small language model. It is the laboratory in
which students learn the techniques needed for the assessed reproduction. It
need not force every group into the same final project.

The three assessment components test complementary levels:

```text
prepare and recall
        ↓
understand and explain a published claim
        ↓
verify the claim with code and controlled evidence
```

This progression keeps the evaluation aligned with the course objective: not
merely knowing the vocabulary of large-scale training, but being able to decide
whether a technique works, under which conditions, and with what evidence.

## Evaluating students in AI-assisted work

The assessment should not try to prove that every line of code was typed by a
student. That is difficult to establish and is not the most useful boundary.
Instead, it should distinguish between **producing an artifact** and **owning
the reasoning behind it**.

Coding agents may help produce code, tests, explanations or plots. The student
is still responsible for being able to:

- explain what the system does and trace the relevant data flow;
- justify the main design and experimental decisions;
- predict how a change should affect correctness, memory or performance;
- diagnose a failure using evidence rather than guesses;
- modify the approach when an assumption or experimental condition changes;
- identify limitations, unsupported claims and suspicious results.

The goal is therefore not AI detection. It is to collect enough direct evidence
of understanding that a polished generated artifact cannot substitute for the
student's competence.

### Recommended grading split

Within the existing 15% / 20% / 65% structure, the detailed rubrics above make
more than half of the final grade individually attributable:

| Assessment evidence | Team or individual | Share of final grade |
| --- | --- | ---: |
| Pre-session quizzes | Individual | 15% |
| Paper presentation content | Team | 10% |
| Paper presentation questions | Individual | 10% |
| Reproduction artifact: code, report and evidence | Team | 32.5% |
| Project defense, diagnosis and adaptation | Individual | 32.5% |
| **Total** | | **100%** |

This preserves collaborative project work while making it impossible for a
student to earn a strong grade solely from a repository or presentation created
by other people or tools. A minimum individual-defense score could also be a
passing requirement.

### Practical assessment formats

#### Individual oral defense

After each group presentation, address at least one question to every student.
For the final project, use a short individual defense with questions drawn from
a common bank. Examples include:

- Trace one batch from input data to the reported metric.
- Why is this baseline fair? What would make it unfair?
- Which result most strongly supports the target claim?
- Which result could be an artifact of the measurement setup?
- If the available memory were halved, what would you change first?
- What failed during the project, and how did the evidence change your mind?

Questions should target the group's actual submission. Generic rehearsed
knowledge is weaker evidence of ownership than explaining a concrete choice,
plot, configuration or failure from the project.

#### Small adaptation task

Give each student a modest variation of the submitted work and ask them to
reason through it. This need not be a speed-coding exercise. Students could
receive preparation time and access to their repository and documentation, but
not to a coding agent during the assessed portion.

Possible tasks include:

- change the batch size or sequence length and predict which measurements must
  also change;
- add or remove one control and explain how it affects the conclusion;
- locate and diagnose a deliberately introduced bug;
- interpret an unfamiliar profiler trace or failed run;
- design a reduced experiment for different hardware;
- explain which parts of the implementation would need to change for a new
  model or dataset.

The rubric should reward the reasoning process, choice of checks and use of
evidence. Typing speed and memory for library syntax should carry little or no
weight. Equivalent accommodations should be available when a live task creates
an accessibility barrier.

#### Code and result walkthrough

Select a small part of the submission rather than asking students to present
the whole repository. Ask a student to:

1. explain the purpose and assumptions of the selected section;
2. connect it to one reported result;
3. identify a plausible failure mode;
4. describe a test that would expose that failure.

The selected section can be announced at the defense. This encourages shared
ownership of the group repository rather than dividing it into isolated parts
that only one member understands.

#### Prediction before execution

At major project checkpoints, groups record a brief prediction before running
an experiment:

- what they expect to happen;
- which mechanism motivates that expectation;
- which result would contradict it;
- what they would test next in either case.

These short, timestamped decision notes make scientific reasoning visible and
reduce the value of a retrospective story written after seeing the result. They
also provide good material for individual defense questions.

#### Reproduction exchange

Near the end of the project, another group receives the repository and attempts
one documented command or small experiment. The assessors can evaluate:

- whether the result can be reproduced from the instructions;
- which assumptions or missing information the second group discovers;
- how the original group responds to the reproduction report.

This tests documentation and reproducibility more directly than asking whether
a README looks complete. The exchange should remain small enough that students
are not asked to debug another group's entire project.

#### Short individual reflection

Each student submits a concise reflection alongside the group deliverables:

- one decision they personally contributed to;
- one claim they initially believed but revised;
- one failure or surprising result they can explain;
- one next experiment they would prioritize;
- how coding agents or other external tools were used.

The reflection should be a starting point for questioning, not accepted as
proof of authorship by itself.

### Role of quizzes

Unsupervised online quizzes are weak evidence of unaided individual knowledge,
because both classmates and agents can answer them. They can still be useful as
a low-weight preparation mechanism. If the teaching team wants quizzes to
contribute stronger individual evidence, possible formats are:

- administer a very short retrieval quiz at the start of the teaching day;
- follow the pre-session quiz with one related in-class question;
- ask students to correct and explain one of their quiz errors;
- sample quiz concepts again during the individual defense.

The quiz grade should remain small. Higher-stakes individual evidence should
come from explanation, transfer and diagnosis rather than from surveillance or
increasingly elaborate anti-cheating measures.

### AI-use policy

A workable policy is to permit AI assistance during project work while making
the following expectations explicit:

- students disclose the tools used and the parts of the workflow they affected;
- external and generated code receives the same scrutiny as any other
  dependency;
- students remain responsible for correctness, licensing, attribution and
  reproducibility;
- any team member may be questioned about any central part of the submission;
- coding agents are unavailable only during clearly identified individual
  assessment moments;
- fabricated experiments, logs, citations or results are academic misconduct,
  whether produced manually or by an AI system.

Requiring complete prompt histories is unlikely to be reliable or
proportionate: logs can be incomplete, contain private information and say
little about actual understanding. A brief disclosure plus direct assessment
of ownership is more useful.

### Approaches to avoid

- **AI-text or AI-code detectors:** their output is not reliable evidence of
  authorship and can create false accusations.
- **Large amounts of live coding from memory:** this often measures speed,
  syntax recall and stress tolerance more than the course objectives.
- **Grading repository polish as understanding:** a clean artifact can be
  generated or produced by one teammate.
- **Banning all assistance during project work:** this is difficult to enforce
  and prevents students from learning how to use tools critically.
- **Giving the same group mark to every member without individual evidence:**
  this makes contribution and understanding impossible to distinguish.

## Decisions still to make

- Confirm the number and size of groups, then fix presentation durations.
- Decide whether paper selection is open, based on an instructor shortlist, or
  a mixture of both.
- Define the available compute budget and whether all groups receive the same
  allocation.
- Decide whether the report is exactly five pages or a shorter artifact such as
  a reproducibility note.
- Decide whether peer feedback contributes to the paper-presentation grade or
  remains formative.
- Specify the policy for late quizzes, absences, external code and generative
  AI use.
- Decide which individual assessment moments are supervised and whether a
  minimum individual-defense score is required to pass.
