# Course Lectures: Training and Scaling Language Models

LaTeX Beamer slide decks for **Training and Scaling Language Models: From First Principles to Efficient Serving**.

Styled with the IMT Atlantique theme adapted from `llm4code`.

## Teaching Format

The course has 11 modules and 32 teaching sessions of 75 minutes. Lecture 0
introduces the course. Modules 1–10 have developed slide sources; Module 11
remains a scaffold reserved for later authoring. The website chapters are the
conceptual source, and the [schedule](../docs/schedule.md) assigns each module
to one or more sessions.

A module number is not a session number. Some existing slide titles still use
the older session numbering and need reconciliation before delivery. Do not
interpret a compiled PDF as evidence that its content is complete or current.

The developed decks combine guiding questions, worked estimates, diagrams,
and reasoning checkpoints. Session timing and required practical outputs are
defined by the schedule and companion guides.

## Directory Structure

```
lectures/
├── common/                     # Shared theme, definitions, and assets
│   ├── style.sty               # IMT Atlantique / Beamer theme package
│   ├── colors.tex              # Semantic color definitions (blues, terracotta, alerts)
│   ├── preamble.tex            # Shared packages, TikZ, listings, and macros
│   └── figs/                   # Shared logos and graphics
├── 01-transformer/             # Module 1: Transformer from first principles
├── 02-training-loop/           # Module 2: Training-loop anatomy & baseline GPT
├── 03-data-pipeline/           # Module 3: Pretraining data pipeline
├── 04-single-gpu/              # Module 4: Single-GPU performance
├── 05-ddp/                     # Module 5: Distributed data parallelism
├── 06-fsdp/                    # Module 6: FSDP and ZeRO
├── 07-tensor-parallelism/      # Module 7: Tensor parallelism
├── 08-context-pipeline/        # Module 8: Context, pipeline & expert parallelism
├── 09-kv-cache/                # Module 9: KV-cached decoding
├── 10-serving/                 # Module 10: Serving systems
├── 11-frontier-architectures/  # Module 11: Frontier architectures
└── Makefile                    # Batch compilation and cleaning
```

## Compilation

To compile the introduction and all 11 module slide decks:
```bash
make all
```

To compile an individual module (e.g. Module 1):
```bash
make 01-transformer
# or manually:
cd 01-transformer && pdflatex 01-transformer.tex && pdflatex 01-transformer.tex
```

To clean intermediate auxiliary files (`.aux`, `.log`, `.nav`, etc.):
```bash
make clean
```
