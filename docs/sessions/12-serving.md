# Session 12 — Serving systems

## Purpose

Move from one efficient decoding request to a system that schedules many
requests under latency and memory constraints.

## Key ideas

- continuous batching;
- paged KV-cache management;
- chunked prefill and prefix caching;
- time to first token, inter-token latency and throughput.

## Practical task

Serve the course checkpoint with an existing engine and benchmark declared
request workloads.

## Expected output

A serving report that connects workload, scheduler behavior, latency,
throughput and memory.

## Material to add later

- serving timeline diagram;
- workload generator;
- vLLM or SGLang configuration;
- benchmark-report template.

