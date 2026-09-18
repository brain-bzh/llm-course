# Session 6 — Single-GPU performance

## Purpose

Measure where one-GPU training spends time and memory before attempting to make
it faster.

## Key ideas

- FLOPs, throughput and model FLOP utilization;
- parameters, optimizer states, gradients and activations;
- compute-bound versus memory-bound work;
- SDPA, fused operations, compilation and mixed precision.

## Practical task

Profile the baseline trainer and apply one optimization at a time.

## Expected output

A before/after report containing throughput, peak memory and profiler evidence.

## Material to add later

- memory-accounting worksheet;
- profiler walkthrough;
- optimization toggles;
- benchmark template.

