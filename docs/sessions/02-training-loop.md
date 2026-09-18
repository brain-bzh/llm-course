# Session 2 — Training-loop anatomy

## Purpose

Understand every state change between a batch of tokens and an optimizer update.

## Key ideas

- cross-entropy and backpropagation;
- AdamW, weight decay and learning-rate schedules;
- gradient accumulation and clipping;
- mixed precision, evaluation and checkpointing.

## Practical task

Implement the complete training loop around the model from Session 1, including
metrics, validation and checkpoint recovery.

## Expected output

A correct baseline trainer whose optimizer-step count, effective batch size and
resume behavior can be explained.

## Material to add later

- training-loop diagram;
- deliberately broken loop examples;
- accumulation and scheduling exercises;
- checkpoint test.

