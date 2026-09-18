# Session 8 — FSDP and ZeRO

## Purpose

Understand how sharding model states changes the memory and communication costs
of training.

## Key ideas

- parameter, gradient and optimizer-state memory;
- all-gather and reduce-scatter;
- sharding stages and wrapping choices;
- distributed checkpointing.

## Practical task

Run one focused FSDP experiment and compare it with DDP on the same model and
hardware.

## Expected output

A comparison of peak memory, throughput and checkpoint behavior.

## Material to add later

- state-sharding diagram;
- FSDP configuration example;
- memory prediction exercise;
- checkpoint recovery test.

