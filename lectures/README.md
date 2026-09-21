# Course Lectures: Training and Scaling Language Models

LaTeX Beamer slide decks for **Training and Scaling Language Models: From First Principles to Efficient Serving**.

Styled with the IMT Atlantique theme adapted from `llm4code`.

## Teaching Format

Modules 1--12 are complete lecture decks, one per course module. Each module
spans two to five 1h15 sessions of contact time (see the course
[schedule](../docs/schedule.md) for the exact session count per module — a
module is a topic, not a fixed-length time slot). Each deck uses the same
teaching rhythm:

1. a guiding question and four explicit learning goals;
2. three concept sections with equations, diagrams, or worked estimates;
3. short checkpoints for individual reasoning or peer discussion;
4. a final slide containing the durable ideas students should retain.

The decks contain 19--23 slides each. Section-outline slides provide natural
pauses, while the checkpoints reserve time for active reasoning rather than
continuous exposition. Module 13 remains the original scaffold, as requested,
for later authoring.

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

To compile all 11 module slide decks:
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
