# Module 4 — Train a small GPT

## Purpose

Turn the components from Modules 1–3 into the first controlled language-model
experiment. The goal is not merely to make the loss decrease: it is to know
what was trained, why the run is credible, and which checkpoint should become
the baseline for every later comparison.

The module uses two complementary references:

- the [GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  explains the modeling hypothesis, data choices and architecture;
- [nanoGPT](https://github.com/karpathy/nanoGPT) exposes those choices in a
  compact model definition and training loop.

GPT-2 is the scientific object; nanoGPT is the readable map. The course model
is a smaller experiment that preserves the same causal language-modeling
mechanism without pretending to reproduce GPT-2's result or budget.

## Learning goals

By the end of the module, students should be able to:

- connect a GPT configuration to tensor shapes and a parameter count;
- explain which details make the course model GPT-2-like;
- specify a training run in tokens, optimizer updates and wall-clock budget;
- interpret training and validation loss without over-reading noisy samples;
- diagnose common failure signatures from curves and logged quantities;
- choose and document a baseline checkpoint before changing the data or system.

## The GPT-2 hypothesis

For a token sequence \(x_1, \ldots, x_T\), a causal language model factorizes
the joint probability as

\[
p(x_1, \ldots, x_T) = \prod_{t=1}^{T} p(x_t \mid x_{<t}).
\]

Training minimizes mean next-token negative log-likelihood:

\[
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}
\log p_\theta(x_i \mid x_{<i}).
\]

The interesting claim in the GPT-2 report is not the loss formula itself. The
claim is that a sufficiently capable language model trained on varied natural
text encounters demonstrations of many tasks inside the data. Translation,
question answering and summarization can all be represented as text followed
by text. Improving next-token prediction can therefore produce task behavior
without a separate supervised objective for every task.

This motivates three experimental choices that remain relevant at small scale:

1. use a diverse corpus rather than one narrow task dataset;
2. preserve the causal next-token objective from pretraining through sampling;
3. evaluate generalization on held-out data, not only memorization of training
   sequences.

## What exactly is GPT-2-like?

The GPT-2 report describes four model sizes. The smallest published model has
12 layers, width 768 and 12 attention heads; the largest has 48 layers, width
1600 and 25 heads. All use a context length of 1024 and a byte-level BPE
vocabulary of 50,257 tokens.

The architectural recipe is more important than the exact scale:

- decoder-only causal self-attention;
- learned token and absolute position embeddings;
- pre-normalization: LayerNorm before attention and before the MLP;
- an additional final LayerNorm;
- an MLP with an expansion ratio of four;
- residual projections initialized more conservatively as depth increases;
- a language-model head that produces one logit per vocabulary item.

nanoGPT keeps this structure visible in roughly one model file. It also ties
the token-embedding and output-head weights, uses PyTorch scaled-dot-product
attention when available, and pads the vocabulary size from 50,257 to 50,304
for more efficient matrix shapes during from-scratch training.

!!! note "Reference, reproduction and adaptation"
    Loading the published GPT-2 weights, retraining a 124M model on OpenWebText,
    and training a smaller GPT-shaped model on the course corpus are three
    different experiments. State clearly which one is being performed.

## Read the configuration as a model

Let:

- \(L\) be the number of Transformer blocks;
- \(d\) the model width;
- \(h\) the number of attention heads;
- \(d_h = d/h\) the head width;
- \(V\) the vocabulary size;
- \(T\) the maximum context length.

Ignoring biases and normalization vectors, one GPT-2-style block contains:

| Component | Approximate parameters |
| --- | ---: |
| Q, K and V projections | \(3d^2\) |
| attention output projection | \(d^2\) |
| MLP expansion | \(4d^2\) |
| MLP contraction | \(4d^2\) |
| **Total per block** | **\(12d^2\)** |

With tied input/output embeddings, a useful estimate is therefore

\[
N \approx 12Ld^2 + Vd + Td.
\]

The \(12Ld^2\) term dominates as the network grows. Vocabulary embeddings can
still be a large fraction of a small model, which is why changing the tokenizer
can change both the data representation and the model size.

### Worked example

For \(L=12\), \(d=768\), \(V=50{,}257\), and \(T=1024\):

- blocks: \(12 \times 12 \times 768^2 \approx 84.9\) million parameters;
- token embeddings: \(50{,}257 \times 768 \approx 38.6\) million;
- position embeddings: \(1024 \times 768 \approx 0.8\) million.

This already explains the familiar 124M scale before accounting for small
vectors and implementation details. Students should perform the same estimate
for the course configuration before instantiating it.

## Initialization: preserve the residual stream

nanoGPT initializes linear and embedding weights from a normal distribution
with standard deviation 0.02. It then scales residual-output projection weights
by

\[
\frac{1}{\sqrt{2L}}.
\]

Each block adds two branches to the residual stream—attention and MLP—so their
variance can accumulate with depth. The scaled initialization keeps early
residual updates from overwhelming the stream. This is a concrete example of a
small-looking detail that belongs in the experimental specification: removing
it changes the optimization problem.

Initialization checks before a long run:

- logits are finite and not saturated;
- initial loss is close to, though not necessarily exactly, \(\log V\);
- gradient norms are finite after the first backward pass;
- one tiny batch can be overfit;
- two independent batches do not accidentally contain the same packed region.

For \(V=50{,}257\), a uniform predictor has loss
\(\log(50{,}257) \approx 10.82\) nats. A lower initial loss is not automatically
suspicious—the model, tokenizer, batch composition and tied weights affect the
exact value—but this gives a useful order-of-magnitude check.

## Specify the experiment before launching it

A reproducible run is a tuple, not a checkpoint filename:

\[
\text{run} = (
\text{code}, \text{data}, \text{tokenizer}, \text{model},
\text{optimizer}, \text{schedule}, \text{seed}, \text{hardware}
).
\]

Record at least:

| Category | Required fields |
| --- | --- |
| Data | corpus version, train/validation split, number of tokens, packing rule |
| Tokenizer | vocabulary artifact and special-token IDs |
| Model | \(L, d, h, T, V\), dropout, bias, weight tying, parameter count |
| Optimization | AdamW betas, weight decay, peak/minimum LR, warmup, clipping |
| Batch | micro-batch size, accumulation steps, sequence length, tokens/update |
| Budget | maximum updates, maximum tokens, evaluation interval |
| Runtime | precision, device, software versions, seed |

The fundamental accounting identity is

\[
\text{tokens per update} =
B_{\text{micro}} \times T \times A \times D,
\]

where \(A\) is the number of gradient-accumulation steps and \(D\) the number
of data-parallel replicas. For this module \(D=1\), but keeping it explicit
prevents confusion when the same run is moved to DDP in Module 7.

Total training tokens are

\[
\text{training tokens} =
\text{tokens per update} \times \text{optimizer updates}.
\]

Compare runs at the same number of training tokens, not merely the same number
of dataloader iterations.

## Read the curves as evidence

Training loss answers: *is the optimizer fitting the batches it sees?*
Validation loss answers: *does that improvement transfer to held-out tokens?*
Neither alone is enough.

| Observation | Plausible interpretation | Next check |
| --- | --- | --- |
| Train and validation loss both fall | healthy learning | verify token accounting and samples |
| Train falls; validation flattens or rises | overfitting or split mismatch | inspect split, duplicates and regularization |
| Both remain near initial loss | no learning or data/target bug | tiny-batch overfit, LR, gradients, target shift |
| Sudden loss spike with gradient spike | unstable update or bad batch | LR, clipping, batch contents, numerical range |
| Smooth curve but incoherent samples | undertraining, sampling issue or data issue | compare prompt, temperature and held-out loss |
| Validation improves implausibly fast | leakage or duplicate contamination | audit document-level split and deduplication |
| Loss changes after resume | incomplete checkpoint or changed data order | restore optimizer, scheduler, RNG and iterator state |

Evaluation loss is an estimate. Report the number of evaluation batches or
tokens, because a short estimate can move enough to change an apparent
checkpoint winner.

### Perplexity, with a warning

For average loss \(\mathcal{L}\) in natural-log units,

\[
\mathrm{PPL} = e^{\mathcal{L}}.
\]

Perplexity is only directly comparable when tokenization and preprocessing are
the same. A tokenizer that represents the same text using different prediction
units changes the numerical value even if the underlying text modeling is not
meaningfully better.

## Sampling is a diagnostic, not a score

Sampling catches errors that a scalar loss can hide: broken decoding,
inconsistent tokenizer artifacts, forgotten evaluation mode, pathological
repetition or a checkpoint that cannot be restored.

Use a fixed prompt suite and seed at every evaluation. Include:

- an empty or beginning-of-document prompt;
- a short in-domain prompt;
- a held-out passage prefix;
- an out-of-domain prompt;
- one adversarial formatting or rare-token prompt.

Generate with at least two settings:

- near-greedy or low temperature to expose the dominant continuation;
- moderate temperature with a declared top-k or top-p rule to inspect variety.

Do not select a checkpoint because one sample is charming. Select by the
predeclared validation criterion, then use samples to describe behavior and
find failures.

## Checkpoint selection and resume integrity

A training checkpoint should contain enough state to continue the same
optimization trajectory:

- model parameters;
- optimizer moments;
- scheduler position or optimizer-update count;
- gradient-scaler state when FP16 scaling is used;
- random-number-generator states;
- model and data configuration;
- best validation metric and the rule used to compute it.

Keep two concepts separate:

- **latest checkpoint:** best for fault recovery;
- **best checkpoint:** lowest credible validation loss under the fixed
  evaluation protocol.

After saving, restore the checkpoint in a fresh process, run one validation
batch, and generate one sample. A file that was written successfully is not yet
a verified checkpoint.

## In-class investigation

### Part A — configuration audit

Given the course configuration:

1. derive the head width and approximate parameter count;
2. identify every deliberate difference from GPT-2 124M;
3. compute tokens per optimizer update and total planned tokens;
4. predict the uniform-reference loss \(\log V\);
5. estimate the bytes needed just for BF16 parameters.

### Part B — curve triage

For three supplied runs—healthy, overfitting and unstable—write a diagnosis
containing:

1. the evidence visible in the curves;
2. two competing explanations;
3. the cheapest measurement that distinguishes them;
4. the action that should *not* be taken before that measurement.

### Part C — baseline decision

Complete a one-page model card for the chosen checkpoint:

- intended use and known limitations;
- exact training-token budget;
- best and latest checkpoint IDs;
- validation protocol;
- two representative samples and one failure;
- the next controlled experiment.

## Exit ticket

Answer without running code:

1. Why does tied embedding/output weight change the parameter count?
2. Why is a falling training loss insufficient evidence of a correct run?
3. If sequence length doubles and all other batch terms stay fixed, what happens
   to tokens per optimizer update?
4. Why should sampling settings be fixed when comparing checkpoints?
5. Which checkpoint state is needed for resume but not for inference?

## Expected output

A documented baseline checkpoint with its configuration, training and
validation curves, verified resume path, fixed-prompt samples and a short
decision record explaining why it is the baseline.

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
  — the GPT-2 report: motivation, WebText, architecture and zero-shot analysis;
- [nanoGPT](https://github.com/karpathy/nanoGPT)
  — readable GPT-2-shaped model and training-loop reference;
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
  — an end-to-end walkthrough connecting the report to an implementation.

---

[:material-file-pdf-box: View Lecture Slides (PDF)](../slides/04-small-gpt.pdf){ .md-button target="_blank" }
[:material-code-tags: Practical Companion Guide](../companion/04-small-gpt.md){ .md-button .md-button--primary }

