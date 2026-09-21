# Module 9 — Tensor parallelism

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

## Forward and backward are conjugate

Every layout-changing collective has a complementary backward operation. If a
forward all-gather concatenates shards, backward reduce-scatters the gradient
back to the owners. If a forward all-reduce sums partial outputs, backward can
often avoid another identical reduction because each input-gradient shard is
already determined by its local weight shard.

This is why a tensor-parallel layer should be reasoned about as an autograd
pair, not only as a forward matrix multiplication:

| Forward layout change | Backward counterpart |
| --- | --- |
| all-gather shards → replicated tensor | reduce-scatter replicated gradient → shards |
| reduce-scatter partials → shards | all-gather gradient shards → replicated gradient |
| all-reduce partials → replicated sum | identity or local shard computation, depending on boundary |

The exact placement depends on which tensors the neighboring layer expects.
Writing the layout beside every tensor is safer than memorizing a framework's
operator names.

## Communication model

Let \(p\) be the TP degree and let one activation at a block boundary contain
\(A = B T d\) elements. A ring all-reduce of an \(A\)-element tensor moves
approximately

\[
2\frac{p-1}{p}Aq
\]

bytes per rank, where \(q\) is bytes per element. A classic tensor-parallel
Transformer has synchronization points in both attention and MLP during
forward, with conjugate communication during backward.

The ratio between communication and matrix compute improves with hidden width:
the payload grows roughly as \(BTd\), while dense projection work grows as
\(BTd^2\). Wide layers amortize communication better than narrow ones. Larger
TP degree, however, makes each local matrix smaller and can reduce Tensor Core
efficiency while adding synchronization participants.

Sequence parallelism replaces an all-reduce with a reduce-scatter plus a later
all-gather. In bandwidth terms those two ring phases are comparable to an
all-reduce; the benefit is lower activation residency between them, not free
communication.

## Attention head constraints

Column-sharding Q, K and V naturally assigns whole heads to ranks. Check these
divisibility constraints before choosing a TP degree:

- query heads should divide across TP ranks;
- with GQA, KV heads are the tighter constraint;
- if \(p\) exceeds the number of KV heads, KV heads may need replication;
- head dimension must remain intact unless an additional communication scheme
  is introduced.

For example, a model with 32 query heads and 8 KV heads maps cleanly to TP=8.
TP=16 can shard query heads but cannot give every rank a distinct KV head;
replication or a more complex layout is required.

## Correctness invariants

A tensor-parallel implementation is credible only if it verifies:

1. the sharded weights reconstruct the dense reference weights;
2. forward outputs match the dense layer;
3. input gradients match;
4. each parameter-shard gradient matches the corresponding dense slice;
5. optimizer updates preserve the match;
6. collectives occur in the same order on every rank;
7. dropout and other stochastic operations use a deliberate RNG policy.

Test at least one non-square shape and more than one batch/sequence size. A
single symmetric example can hide a wrong split dimension.

## When tensor parallelism helps

Tensor parallelism reduces the parameter and activation footprint of individual
layers, but introduces frequent collectives on the critical path. It is usually
kept within a node where the GPUs have high-bandwidth links, then combined with
data, context or pipeline parallelism across slower links. The useful tensor-
parallel degree is a measurement, not a constant: benchmark communication and
matrix efficiency on the target hardware.

Use these design rules as hypotheses:

- keep TP inside the fastest interconnect domain when possible;
- use the smallest TP degree that makes the layer fit and preserves efficient
  local matrices;
- pair TP with sequence parallelism when replicated non-TP activations dominate;
- do not use TP merely because several GPUs are available—DDP gives independent
  compute and often better throughput when the model already fits;
- benchmark the exact hidden size, head layout, dtype and micro-batch.

## Practical task

Implement the two-rank MLP from this page using basic collective operations:

1. build a dense reference MLP;
2. shard the first projection by output features;
3. shard the second projection by input features;
4. keep the hidden activation local and all-reduce only the final partial output;
5. compare the sharded and dense forward results and parameter gradients;
6. record every tensor shape and the communicated bytes.

## In-class investigation

### Part A — layout ledger

For one attention sublayer and one MLP sublayer, fill a table with:

| Boundary | Global shape | Local shape | Layout | Collective |
| --- | --- | --- | --- | --- |

Include both forward and backward. Any row labeled only “sharded” is
incomplete—state the dimension and process group.

### Part B — degree selection

For a model with \(d=4096\), 32 query heads, 8 KV heads, BF16 activations and
an 8-GPU NVLink node:

1. evaluate TP degrees 1, 2, 4, 8 and 16 for divisibility;
2. estimate local projection shapes;
3. identify which degree crosses a node boundary;
4. predict memory and throughput trends;
5. select a degree and state the measurement that could overturn the choice.

### Part C — communication worksheet

Using the course batch and sequence length, estimate the payload and optimistic
time of one activation all-reduce. Compare it with the adjacent matrix
multiplication time from Module 6's roofline model.

## Exit ticket

1. Why does column-then-row parallelism avoid an intermediate all-gather?
2. Which collective turns row-parallel partial outputs into a replicated sum?
3. What memory does sequence parallelism save?
4. Why is TP usually more sensitive to interconnect bandwidth than pipeline
   parallelism?
5. Why can GQA limit the useful TP degree?

## Expected output

A small tensor-parallel MLP whose outputs and gradients match the unsharded
reference, plus a trace that makes each layout transition explicit.

## References

- [The Ultra-Scale Playbook — Tensor Parallelism](https://huggingface.co/spaces/nanotron/ultrascale-playbook#tensor-parallelism)
  — the main conceptual sequence and scaling evidence for this module;
- [Picotron](https://github.com/huggingface/picotron)
  — compact implementations of tensor and sequence parallelism;
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
  — the original Transformer tensor-parallel formulation;
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)
  — sequence parallelism and selective activation recomputation.

---

[:material-file-pdf-box: View Lecture Slides (PDF)](../slides/09-tensor-parallelism.pdf){ .md-button target="_blank" }
[:material-code-tags: Practical Companion Guide](../companion/09-tensor-parallelism.md){ .md-button .md-button--primary }

