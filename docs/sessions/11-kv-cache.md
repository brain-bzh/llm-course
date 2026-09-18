# Session 11 — KV-cached decoding

## Purpose

Derive why naive autoregressive generation repeats work and remove that waste
without changing model outputs.

## Key ideas

- prefill versus decode;
- key/value cache structure and memory;
- MHA, MQA and GQA;
- memory bandwidth during token-by-token decoding.

## Practical task

Add a KV cache to the course model and compare it with uncached generation.

## Expected output

A cached decoder with an equivalence test and separate prefill/decode timing.

## Material to add later

- cache-shape derivation;
- incremental-attention implementation;
- correctness fixtures;
- latency benchmark.

