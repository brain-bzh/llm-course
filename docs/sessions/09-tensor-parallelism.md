# Session 9 — Tensor parallelism

## Purpose

See how a single layer can be split across devices when model replication is no
longer sufficient.

## Key ideas

- column- and row-parallel linear layers;
- collective placement;
- attention and MLP sharding;
- sequence-parallel activations.

## Practical task

Implement a toy sharded MLP using basic collective operations.

## Expected output

A small tensor-parallel layer whose outputs and gradients match an unsharded
reference.

## Material to add later

- sharded-linear derivation;
- collective-operation exercises;
- toy distributed implementation;
- equivalence tests.

