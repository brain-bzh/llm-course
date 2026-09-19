# Course Lectures: Training Language Models

LaTeX Beamer slide decks for **Training Language Models: From First Principles to Efficient Serving**.

Styled with the IMT Atlantique theme adapted from `llm4code`.

## Teaching Format

Sessions 1--12 are complete lecture decks for a 1h15 teaching period. Each one
uses the same teaching rhythm:

1. a guiding question and four explicit learning goals;
2. three concept sections with equations, diagrams, or worked estimates;
3. short checkpoints for individual reasoning or peer discussion;
4. a final slide containing the durable ideas students should retain.

The decks contain 19--23 slides each. Section-outline slides provide natural
pauses, while the checkpoints reserve time for active reasoning rather than
continuous exposition. Session 13 remains the original scaffold, as requested,
for later authoring.

## Directory Structure

```
lectures/
├── common/                     # Shared theme, definitions, and assets
│   ├── style.sty               # IMT Atlantique / Beamer theme package
│   ├── colors.tex              # Semantic color definitions (blues, terracotta, alerts)
│   ├── preamble.tex            # Shared packages, TikZ, listings, and macros
│   └── figs/                   # Shared logos and graphics
├── 01-transformer/             # Session 1: Transformer from first principles
├── 02-training-loop/           # Session 2: Training-loop anatomy
├── 03-data-pipeline/           # Session 3: BPE and the data pipeline
├── 04-small-gpt/               # Session 4: Train a small GPT
├── 05-data-selection/          # Session 5: Data selection & corpus filtering
├── 06-single-gpu/              # Session 6: Single-GPU performance
├── 07-ddp/                     # Session 7: Distributed data parallelism
├── 08-fsdp/                    # Session 8: FSDP and ZeRO
├── 09-tensor-parallelism/      # Session 9: Tensor parallelism
├── 10-context-pipeline/        # Session 10: Context and pipeline parallelism
├── 11-kv-cache/                # Session 11: KV-cached decoding
├── 12-serving/                 # Session 12: Serving systems
├── 13-beyond-transformers/     # Session 13: Beyond dense Transformers
└── Makefile                    # Batch compilation and cleaning
```

## Compilation

To compile all 13 session slides:
```bash
make all
```

To compile an individual session (e.g. Session 1):
```bash
make 01-transformer
# or manually:
cd 01-transformer && pdflatex 01-transformer.tex && pdflatex 01-transformer.tex
```

To clean intermediate auxiliary files (`.aux`, `.log`, `.nav`, etc.):
```bash
make clean
```
