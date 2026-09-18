# Session 3 — BPE and the data pipeline

## Purpose

Trace how raw documents become the token tensors consumed by the model.

## Key ideas

- BPE training and vocabulary size;
- document boundaries and special tokens;
- packing, shuffling and train/validation splits;
- dataset shards and loader throughput.

## Practical task

Train a tokenizer, encode a small corpus and build packed training sequences.

## Expected output

A reproducible tokenizer and dataset pipeline with basic boundary, split and
throughput checks.

## Material to add later

- small raw corpus;
- tokenizer notebook;
- packing visualization;
- data-pipeline validation script.

