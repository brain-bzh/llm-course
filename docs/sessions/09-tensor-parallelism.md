# Session 9 — Tensor parallelism

## Purpose

Split the work inside a Transformer layer when replicating the whole model on
every GPU is no longer possible or efficient.

## Key ideas

- derive column and row parallelism from matrix multiplication;
- keep intermediate activations in the layout needed by the next operation;
- place collectives at explicit layout transitions;
- extend the same pattern from the MLP to attention;
- use sequence parallelism to avoid storing replicated activations outside the
  tensor-parallel regions.

## Start from matrix multiplication

Flatten the batch and sequence dimensions into `B`. A linear layer receives
`X ∈ B × d` and stores `W ∈ d × m`. There are two useful ways to partition the
matrix product.

**Partition the output features.** Write `W = [W[0] | W[1]]`. Each rank can
compute one output shard independently:

`XW = [XW[0] | XW[1]]`.

**Partition the contracted dimension.** Write `X = [X[0] | X[1]]` and stack the
matching weight shards as `W = [W[0]; W[1]]`. Each rank computes a partial result:

`XW = X[0]W[0] + X[1]W[1]`.

The arithmetic is simple. Tensor parallelism is mainly about tracking the layout
of every activation and choosing when it must change.

## Column-parallel first linear layer

The first MLP weight is partitioned along its output features. Every rank starts
with the same input `X`, but stores only one column shard of `W1`. Rank `i`
computes `H_i = φ(X W1_i)`.

<figure markdown="span">
  ![Two ranks receive the same X and compute different output-feature shards of H.](../assets/figures/tp-column-linear.svg){ loading=lazy }
  <figcaption>Column parallelism partitions the output features.</figcaption>
</figure>

A standalone column-parallel layer needs an all-gather if its caller requires a
complete `H`. The MLP does not: the following layer can consume the shards
directly, so no forward collective is needed at this boundary.

## Row-parallel second linear layer

The second weight is partitioned along its input features. Its row shards align
with the hidden-feature shards produced by the first layer. Rank `i` computes a
partial output `Ŷ_i = H_i W2_i`.

<figure markdown="span">
  ![Each rank computes a partial output, then an all-reduce sums and replicates Y.](../assets/figures/tp-row-linear.svg){ loading=lazy }
  <figcaption>Row parallelism sums partial results with one all-reduce.</figcaption>
</figure>

The complete result is `Y = Σ_i Ŷ_i`. An all-reduce performs that sum and leaves
the same `Y` on every rank. A standalone row-parallel layer would first need its
input scattered into compatible shards; composition gives it those shards for
free.

## Column followed by row parallelism

The two layouts are deliberately paired:

`X replicated → W1 column-parallel → H sharded → φ local → W2 row-parallel → all-reduce Y`

<figure markdown="span">
  ![A materialized layer boundary uses an all-gather, while a composed MLP carries hidden shards directly to the row-parallel layer.](../assets/figures/tp-mlp-chain.svg){ loading=lazy }
  <figcaption>Keeping `H` sharded removes the intermediate all-gather.</figcaption>
</figure>

| Stage | Activation layout | Forward collective |
| --- | --- | --- |
| MLP input `X` | replicated | none |
| after column-parallel `W1` | sharded over hidden features | none |
| after pointwise `φ` | same hidden-feature shards | none |
| after row-parallel `W2` | partial `B × d` on each rank | all-reduce |
| MLP output `Y` | replicated | — |

This removes one intermediate collective; it does **not** make tensor
parallelism communication-free. The backward pass has the complementary
communication pattern, and collectives occur in every Transformer block.

## Inside a Transformer block

The same pairing appears in both major sublayers:

- **MLP:** the expansion projection is column-parallel and the contraction
  projection is row-parallel;
- **attention:** the Q, K and V projections are column-parallel, naturally
  assigning attention heads to ranks, while the output projection is
  row-parallel.

Each rank can run its local pointwise activation or its local attention heads
without communication between the paired projections. Grouped-query and
multi-query attention need extra care because KV heads may be shared across
multiple query heads.

## Sequence parallelism

Tensor parallelism alone leaves operations such as normalization, dropout and
residual updates replicated. Their activations can become a substantial memory
cost. Sequence parallelism shards those regions along the sequence dimension.

At a tensor-parallel boundary, the final all-reduce can become a reduce-scatter:
it both sums the partial results and keeps only a sequence shard on each rank.
Before the next tensor-parallel region, an all-gather restores the replicated
input expected by the column-parallel projection.

Sequence parallelism primarily reduces activation residency. It changes the
layout transitions, not the model's arithmetic.

## When tensor parallelism helps

Tensor parallelism reduces the parameter and activation footprint of individual
layers, but introduces frequent collectives on the critical path. It is usually
kept within a node where the GPUs have high-bandwidth links, then combined with
data, context or pipeline parallelism across slower links. The useful tensor-
parallel degree is a measurement, not a constant: benchmark communication and
matrix efficiency on the target hardware.

## Practical task

Implement the two-rank MLP from this page using basic collective operations:

1. build a dense reference MLP;
2. shard the first projection by output features;
3. shard the second projection by input features;
4. keep the hidden activation local and all-reduce only the final partial output;
5. compare the sharded and dense forward results and parameter gradients;
6. record every tensor shape and the communicated bytes.

## Expected output

A small tensor-parallel MLP whose outputs and gradients match the unsharded
reference, plus a trace that makes each layout transition explicit.

## References

- [The Ultra-Scale Playbook — Tensor Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook#tensor-parallelism), the main conceptual sequence for this session;
- [Picotron](https://github.com/huggingface/picotron), a compact implementation to read after the toy exercise;
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053), the original Transformer tensor-parallel formulation.

## Material to add later

- an attention-head sharding figure;
- a sequence-parallel layout exercise;
- a communication-volume worksheet;
- a small hardware benchmark comparing tensor-parallel degrees.
