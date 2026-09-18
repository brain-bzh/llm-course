# Session 10 — Context and pipeline parallelism

## Purpose

Choose a parallelization strategy from model, sequence and cluster constraints
rather than from framework fashion.

## Key ideas

- context-length memory pressure;
- splitting sequence work across devices;
- pipeline stages, microbatches and bubbles;
- topology and multidimensional parallelism.

## Practical task

Solve several model-and-cluster design cases using explicit memory and
communication estimates.

## Expected output

A defended strategy for each case, including the dominant cost and expected
utilization loss.

## Material to add later

- cluster-topology diagrams;
- pipeline timeline exercise;
- memory and communication calculator;
- production code-reading excerpts.

