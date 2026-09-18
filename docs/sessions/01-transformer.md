# Session 1 — Transformer from first principles

## Purpose

Build a shared mental model of the decoder-only Transformer and turn that model
into the smallest implementation students can fully inspect.

## Key ideas

- token representations and position information;
- causal self-attention;
- residual connections, normalization and the MLP;
- logits and next-token prediction.

## Practical task

Implement a minimal Transformer block, assemble a small language model and test
the causal mask and tensor shapes.

## Expected output

A readable model implementation that completes a forward pass and can overfit a
tiny batch.

## Material to add later

- architecture diagram;
- annotated implementation;
- shape and masking exercises;
- tiny-batch debugging checklist.

