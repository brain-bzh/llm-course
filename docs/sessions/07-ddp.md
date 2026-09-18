# Session 7 — Distributed data parallelism

## Purpose

Scale a correct single-GPU trainer across replicated model copies without
changing the intended optimization.

## Key ideas

- process-per-GPU execution;
- gradient all-reduce;
- distributed sampling and global batches;
- accumulation, `no_sync` and communication overlap.

## Practical task

Convert the course trainer to DDP and compare it with the single-GPU reference.

## Expected output

A multi-GPU run with verified synchronization, sample coverage and global token
accounting.

## Material to add later

- collective-communication diagrams;
- DDP starter patch;
- distributed correctness checks;
- scaling benchmark worksheet.

