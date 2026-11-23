# Module 11 — Frontier architectures: reading a 2026 model config

## Purpose

Use the course's cost models to read the design of two very recent open-weight
models, line by line from their `config.json`:

- **Mistral Large 4** (October 2026, 1T total / 52B active): sparse attention in
  every layer, with indexer reuse and cross-layer KV sharing;
- **GLM-5.3-Flash** (August 2026, 320B / 18B): a "super hybrid" in which three
  of every four attention layers are recurrent, and the remaining layers are
  sparse.

Both models target a context window of 1,048,576 tokens. At that length the
dense-Transformer costs from earlier modules stop being abstract: a KV cache of
tens of GiB per request (Module 9), quadratic prefill (Module 4), expert
traffic (Module 8). Each architecture choice in these configs reduces one of
those costs and creates a new one. The session's goal is to name both.

!!! note "Session 27 · no required lab"
    This is a lecture with an in-class pen-and-paper exercise. It adds no
    coding assignment or report. Speculative decoding is covered in
    [Module 10](10-serving.md).

## Learning goals

By the end of the session, students should be able to:

- read an MoE/MLA `config.json` and recover total and active parameters, the
  layer pattern and the cached bytes per token;
- explain what DeepSeek Sparse Attention removes from attention cost and what
  it still has to scan;
- explain the indexer sharing and cross-layer KV sharing in Mistral Large 4,
  and what each one saves;
- compare a growing KV cache with the fixed recurrent state of Kimi Delta
  Attention, and compute the context length at which one exceeds the other;
- explain why hybrid models keep some attention layers, and why those layers
  can be NoPE;
- state what a four-stream residual (mHC) adds to per-token state;
- relate expert granularity to compute per dispatched byte (Module 8).

## Session plan

| Time | Block | Builds on |
| --- | --- | --- |
| 0–10 min | Where the cost goes at a million tokens | Modules 4, 8, 9 |
| 10–25 min | Four mechanisms: MLA without RoPE, DSA, KDA, mHC | Module 9 |
| 25–42 min | Case study: Mistral Large 4 | Modules 8, 9 |
| 42–60 min | Case study: GLM-5.2 → GLM-5.3-Flash | Modules 8, 9 |
| 60–75 min | Exercise: read an unknown config | everything |

---

## 1. Where the cost goes at a million tokens

For one decode step of one request at cached length \(T\), the bytes read are
roughly

\[
\underbrace{\frac{\text{weights}_{\text{active}}}{B_{\text{eff}}}}_{\text{amortized over the batch}}
\;+\;
\underbrace{L_{\text{attn}}\;T\;e\;b}_{\text{KV cache, per request}}
\;+\;
\underbrace{\text{state}}_{\text{per request}} ,
\]

with \(e\) elements cached per token and layer (Module 9). Batching amortizes
the first term. With MoE it amortizes only partly, because different tokens
select different experts. The cache term grows with \(T\) and belongs to a
single request. At the full window \(T=2^{20}=1{,}048{,}576\), a 7B-scale GQA model with 32 layers, 8 KV heads
of width 128 and BF16 needs 128 KiB per token, i.e. 128 GiB for one request.

Prefill has the matching problem in compute: dense causal attention does
\(\mathcal{O}(T^2)\) work per layer.

Frontier designs attack the per-request terms in two ways:

1. **Make every attention layer cheaper.** Store less per token (MLA), and
   attend to a selected subset of positions instead of all of them (DSA).
   Mistral Large 4 takes this route.
2. **Replace most attention layers with a fixed-size recurrent state.** Keep a
   few attention layers for exact retrieval. GLM-5.3-Flash takes this route
   (and also makes its remaining attention layers sparse).

<figure markdown="span">
  ![Layer-by-layer attention type for GLM-5.2, Mistral Large 4 and GLM-5.3-Flash. GLM-5.2 has 78 MLA layers with an indexer in one of every four. Mistral Large 4 has 72 MLA layers, 44 with their own indexer and 9 that reuse the KV cache of layer 35. GLM-5.3-Flash alternates three KDA layers and one MLA layer.](../assets/figures/frontier-layer-patterns.svg){ loading=lazy }
  <figcaption>Layer patterns read from each model's <code>config.json</code>.</figcaption>
</figure>

## 2. Four mechanisms

### MLA, and MLA without RoPE

Module 9 introduced multi-head latent attention: cache one latent
\(c_t\in\mathbb{R}^{d_c}\) per token and layer, plus a small positional key of
width \(d_R\). All three models in this session use \(d_c=512\).

- Mistral Large 4 and GLM-5.2 keep the decoupled RoPE key, \(d_R=64\), so a
  layer caches \(576\) elements per token.
- GLM-5.3-Flash sets `qk_rope_head_dim = 0`: its MLA layers are **NoPE** and
  cache only the \(512\)-wide latent. Order information comes from the
  recurrent layers around them, which see tokens in sequence. Kimi Linear made
  the same choice.
- Mistral Large 4 uses NoPE in one layer in four (`no_rope_layer_interval: 4`).
  The other layers use YaRN-scaled RoPE, which Mistral calls LeRoPE because it
  adds a learned scale per frequency.

### DeepSeek Sparse Attention (DSA)

[DeepSeek-V3.2](https://arxiv.org/abs/2512.02556) added a **lightning
indexer** in front of MLA. For query position \(t\) and each earlier position
\(s\), a few small indexer heads give a relevance score

\[
I_{t,s}=\sum_{j=1}^{H^I} w_{t,j}\,\operatorname{ReLU}\!\left(q^{I}_{t,j}\cdot k^{I}_{s}\right),
\]

and MLA attends only to the top-\(k\) positions:

\[
\mathcal{S}_t=\operatorname{top-}k_s\;I_{t,s},
\qquad
o_t=\operatorname{Attn}\!\big(q_t,\{c_s: s\in\mathcal{S}_t\}\big).
\]

All three DSA models here use \(H^I=32\) indexer heads of width 128 and
\(k=2048\).

| Cost | Dense MLA | MLA + DSA |
| --- | --- | --- |
| Core attention per query | \(\mathcal{O}(T)\) | \(\mathcal{O}(k)\) |
| Indexer per query | — | \(\mathcal{O}(T)\), but cheap per pair |
| Prefill, per layer | \(\mathcal{O}(T^2)\) | \(\mathcal{O}(Tk)\) attention + \(\mathcal{O}(T^2)\) indexer |
| Extra cache per token | — | one indexer key (128 values) |

The indexer is still quadratic over a whole sequence. What makes DSA pay is the
cost of one query–key pair. With weight absorption, a Mistral Large 4 MLA head
does about \(576+512=1088\) multiply–adds per pair, and 64 heads give about
\(70\)k. The indexer does \(32\times128\approx4\)k, often in FP8. That is about
17 times less work per pair, applied to all \(T\) positions, while full
attention is applied to only \(k\) of them.

!!! warning "DSA is trained in"
    In DeepSeek-V3.2, the indexer is first trained to match the dense attention
    distribution, and then the model continues training with sparse selection.
    As with MLA, top-\(k\) selection cannot be switched on in an arbitrary
    pretrained model without that training.

### Kimi Delta Attention (KDA)

[Kimi Linear](https://arxiv.org/abs/2510.26692) replaces softmax attention in
most layers with a gated delta-rule recurrence. Each head keeps a matrix state
\(S_t\in\mathbb{R}^{d_k\times d_v}\):

\[
S_t=\big(I-\beta_t\,k_tk_t^{\top}\big)\operatorname{Diag}(\alpha_t)\,S_{t-1}
+\beta_t\,k_t v_t^{\top},
\qquad
o_t=S_t^{\top}q_t .
\]

- \(\alpha_t\in(0,1)^{d_k}\) is a forget gate **per channel**. Gated DeltaNet
  uses one scalar per head.
- \(\beta_t\) is a write strength. The term \((I-\beta_tk_tk_t^{\top})\)
  erases the value previously stored along \(k_t\) before writing the new one.
- Training uses chunk-wise parallel kernels, so the recurrence still turns into
  matrix multiplications.

The state does not grow with \(T\). A KDA layer can only keep what fits in
\(d_k\times d_v\) numbers per head, however. Exact retrieval of an arbitrary
old token is what the remaining attention layers are for.

### Manifold-constrained hyper-connections (mHC)

A standard residual stream carries one vector of width \(C\) per token.
[Hyper-connections](https://arxiv.org/abs/2409.19606) widen it to \(n\)
parallel streams \(X_l\in\mathbb{R}^{n\times C}\), with learned mixing:

\[
X_{l+1}=H^{\text{res}}_l X_l+\big(H^{\text{post}}_l\big)^{\top}
F\!\big(H^{\text{pre}}_l X_l\big).
\]

The block \(F\) (attention or MLP) still sees one vector of width \(C\).
[mHC](https://arxiv.org/abs/2512.24880) projects \(H^{\text{res}}\) onto
doubly stochastic matrices with Sinkhorn iterations. Mixing then cannot amplify
the signal across depth, which addresses the training instability of
unconstrained hyper-connections. GLM-5.3-Flash uses \(n=4\) streams
(`hc_mult`) and 20 Sinkhorn iterations. The authors report about 6.7% extra
training time at \(n=4\). The residual state each token carries between
blocks is four times wider. That matters for activation memory (Module 4) and
for any communication that moves the residual stream.

---

## 3. Case study: Mistral Large 4 — sparse attention everywhere

| Field (`text_config`) | Value | Reading |
| --- | --- | --- |
| `hidden_size`, `num_hidden_layers` | 6144, 72 | |
| `num_attention_heads` | 64 | MLA, no query compression |
| `kv_lora_rank`, `qk_rope_head_dim` | 512, 64 | 576 cached elements per layer |
| `index_n_heads`, `index_head_dim`, `index_topk` | 32, 128, 2048 | DSA in all 72 layers |
| `indexer_types` | 44 `full`, 28 `shared` | 28 layers reuse the previous indexer's top-\(k\) |
| `kv_source_layers` | 63 distinct sources | layers 39, 43, …, 71 read layer 35's latent |
| `no_rope_layers` | 18 NoPE, 54 RoPE | one layer in four is NoPE |
| `n_routed_experts`, `num_experts_per_tok`, `n_shared_experts` | 128, 4, 1 | softmax router; layer 0 dense |
| `moe_intermediate_size` | 6144 | each expert ≈ 113M parameters |
| `final_logit_softcapping` | 30 | |
| `vocab_size`, `max_position_embeddings` | 204,800, 1,048,576 | |

Counting from these fields gives about 1.05T total parameters and about 52B
active per token (about 50B without embeddings and output layer), which
matches the published 1T-A52B.

**Two different kinds of sharing.**

- *Indexer sharing* reuses the **selection**. A `shared` layer attends to the
  same 2,048 positions as the most recent `full` layer, but with its own
  queries over its own cached latents. This saves the \(\mathcal{O}(T)\)
  indexer scan, and the indexer key cache, in 28 layers.
- *KV sharing* reuses the **cache**. The nine NoPE layers after layer 35 write
  no latent of their own and attend over layer 35's cache, which is also NoPE.
  Each of these layers keeps its own indexer.

**Cache per token.** Use BF16 latents and the FP8 indexer key used by vLLM's
DSA cache (128 bytes and a 4-byte scale):

\[
\underbrace{63\times576\times2}_{72{,}576\ \text{B latent}}
+\underbrace{44\times132}_{5{,}808\ \text{B indexer}}
=78{,}384\ \text{B}\approx76.5\ \text{KiB per token}.
\]

Without KV sharing the latent alone would be \(72\times1152=82{,}944\) B. With
an FP8 latent cache, vLLM reports 47,136 B per token. One request at
\(T=2^{20}\) still holds about 77 GiB in BF16. MLA, DSA and sharing reduce the
cache by constant factors. They do not stop it growing with \(T\).

**What remains at long context.** Per decode step and request at
\(T=2^{20}\):

- sparse attention reads about \(63\times2048\times1152\approx0.15\) GB of
  latents;
- the 44 indexers scan about \(44\times2^{20}\times132\approx6.1\) GB of keys.

Weights (about 52 GB of active FP8 weights) are shared across the batch, but
the indexer scan belongs to each request. At long context the indexer becomes
the dominant per-request decode traffic, and this is the term that indexer
sharing reduces.

**MoE shape.** Each token goes to 4 large experts plus one shared expert.
Compared with fine-grained designs, this means fewer, larger expert GEMMs and
less dispatch traffic per FLOP (see the comparison in §5).

## 4. Case study: GLM-5.2 → GLM-5.3-Flash — the super hybrid

GLM-5.2 is already a DSA model. GLM-5.3-Flash keeps its MoE recipe, shrinks it,
and replaces three of every four attention layers with KDA.

| Field | GLM-5.2 | GLM-5.3-Flash |
| --- | --- | --- |
| Parameters (total / active) | 744B / 40B | 320B / 18B |
| `hidden_size` | 6144 | 4096 |
| Layers | 78 (+1 MTP) | 45 (+1 MTP) |
| Attention pattern | 78 × MLA + DSA | 34 × KDA + 11 × MLA + DSA, repeating `KKKD` |
| `kv_lora_rank` / `qk_rope_head_dim` | 512 / 64 | 512 / 0 (NoPE) |
| Indexer | 21 `full`, 57 `shared` (IndexShare) | 11 `full`, pooled selection (`index_kpool: 4`) |
| KDA heads × \(d_k\) × \(d_v\) | — | 64 × 128 × 128, short convolution of width 4 |
| Experts (routed / active / shared) | 256 / 8 / 1 | 288 / 8 / 1 |
| `moe_intermediate_size` | 2048 | 2048 |
| Dense layers before MoE | 3 | 3 |
| Router | sigmoid, bias-based balancing (`noaux_tc`) | same |
| Residual | standard | mHC, 4 streams |
| Vision | — | native ViT encoder (24 layers) |

Counting from the GLM-5.3-Flash config gives about 313B, plus about 7B for the
MTP layer, with about 17–18B active.

**Cache and state per request.** Use the same convention as for Mistral Large 4:
BF16 latent and a 132-byte indexer key. GLM-5.3-Flash caches

\[
\underbrace{11\times512\times2}_{11{,}264\ \text{B latent}}
+\underbrace{11\times132}_{1{,}452\ \text{B indexer}}
=12{,}716\ \text{B}\approx12.4\ \text{KiB per token},
\]

against about 90.5 KiB for GLM-5.2. In exchange, every request holds a **fixed**
KDA state:

\[
34\ \text{layers}\times64\times128\times128\times4\ \text{B (FP32)}
=136\ \text{MiB},
\]

plus about 5 MiB of convolution cache. The reference implementation keeps the
recurrent state in FP32.

| Context \(T\) | Mistral Large 4 | GLM-5.3-Flash (cache + state) |
| ---: | ---: | ---: |
| 4k | 0.30 GiB | 0.18 GiB |
| 32k | 2.4 GiB | 0.52 GiB |
| 128k | 9.6 GiB | 1.7 GiB |
| 1M | 76.5 GiB | 12.5 GiB |

The state equals the cache at \(136\ \text{MiB}/12{,}716\ \text{B}\approx11\)k
tokens. Below that, GLM-5.3-Flash's per-request memory is mostly fixed state.
Short-prompt, high-concurrency serving therefore gains much less than the
million-token headline suggests. That is the workload Module 10 measures.

**Why keep MLA layers, and why make them sparse.**

- The 11 attention layers give exact access to any earlier token. The KDA state
  is a compressed summary and cannot provide that.
- These layers are NoPE: the KDA layers already encode order, and a NoPE
  latent has no RoPE frequencies to extrapolate to a million positions.
- With only 11 indexers, each scoring pools of 4 tokens, the \(\mathcal{O}(T)\)
  indexer term is much smaller than in Mistral Large 4. The four-token pooling
  is an inference from the config and the Transformers code; Z.ai does not
  describe it.

**What it costs.**

- KDA is lossy memory. Retrieval-heavy long-context evaluations, not perplexity,
  are the evidence to ask for.
- There are two kinds of layer to implement, fuse and parallelize. Context
  parallelism (Module 8) must pass a recurrent state between ranks instead of
  ring-passing K/V blocks.
- mHC makes the residual state four times wider.

---

## 5. Side by side

| | Mistral Large 4 | GLM-5.3-Flash |
| --- | --- | --- |
| Main idea | every layer sparse; share selections and caches | most layers recurrent; the rest sparse |
| Cache per token | ≈76.5 KiB | ≈12.4 KiB + 136 MiB fixed |
| Per-request long-context traffic | indexer scan, 44 layers | indexer scan, 11 layers (pooled), plus state |
| Experts | 128, top-4 + 1 shared, \(d_{ff}=6144\) | 288, top-8 + 1 shared, \(d_{ff}=2048\) |
| Dispatch per token and MoE layer (BF16) | \(4\times6144\times2=48\) KiB | \(8\times4096\times2=64\) KiB |
| Expert FLOPs per dispatched byte | ≈18k | ≈6k |
| Position | 3:1 RoPE : NoPE | NoPE attention, order from KDA |
| Residual | standard | mHC, 4 streams |

The last MoE rows connect to Module 8. Fine-grained experts give more expert
combinations per token, but they do less compute for each byte sent through
All-to-All. Overlapping that communication is therefore harder.

## 6. In-class exercise: read an unknown config

The following model is **invented** for the exercise. Treat it like a new
release.

| Field | Value |
| --- | --- |
| `hidden_size` | 5120 |
| `num_hidden_layers` | 48, pattern `GGGM` repeated (G: gated DeltaNet, M: MLA) |
| MLA | `kv_lora_rank` 512, `qk_rope_head_dim` 64, dense (no indexer) |
| Linear layers | 40 heads, \(d_k=d_v=128\), FP32 state |
| MoE | 256 routed experts, top-8, 1 shared, `moe_intermediate_size` 1536 |
| Dense layers before MoE | 2, `intermediate_size` 12288 |
| Vocabulary | 160,000, untied input and output embeddings |
| Attention and recurrent projections | 6.0B parameters in total (given) |

1. KV-cache bytes per token in BF16.
2. Recurrent state per request.
3. The context length at which cache and state are equal, and the per-request
   total at 128k.
4. Total and active parameters (SwiGLU experts have \(3\,C\,d_{ff}\)
   parameters).
5. Which cost does this design target? What evidence would you ask for before
   believing that it does not lose quality?

??? success "One possible answer"
    1. 12 MLA layers × 576 × 2 B = 13,824 B = 13.5 KiB per token. The same
       model with MLA in all 48 layers would need 54 KiB.
    2. 36 × 40 × 128 × 128 × 4 B = 90 MiB.
    3. 90 MiB / 13.5 KiB ≈ 6.8k tokens. At 128k: 1.69 GiB cache + 0.09 GiB
       state ≈ 1.8 GiB, against 6.75 GiB for all-MLA.
    4. One expert: 3 × 5120 × 1536 ≈ 23.6M. MoE layers: 46 × 257 × 23.6M ≈
       278.9B total; 46 × 9 × 23.6M ≈ 9.8B active. Add dense layers 0.38B,
       embeddings 1.64B and projections 6.0B: ≈ 287B total, ≈ 17.8B active.
    5. Per-request memory and bandwidth at long context. Ask for long-context
       retrieval results (multi-needle, multi-hop) against an all-attention
       baseline trained the same way, and for measured decode throughput
       against context length, not only the parameter count.

## Exit ticket

1. DSA keeps an \(\mathcal{O}(T^2)\) term in prefill. Why is it still worth
   paying?
2. What does Mistral Large 4 share in `shared` indexer layers, and what in
   layers 39, 43, …, 71?
3. Below roughly what context does GLM-5.3-Flash use **more** per-request
   memory for KDA state than for its KV cache?
4. Why can GLM-5.3-Flash's MLA layers be NoPE?
5. Which of the two models do you expect to be easier to serve with expert
   parallelism across nodes, and why?

## Expected output

An in-class worksheet for the exercise, and one sentence per design choice
stating the cost it reduces and the cost it adds. There is no required report.

## A note on sources

Architecture fields come from the published `config.json` files. Parameter
counts and byte counts are computed from them in this module. Cache sizes depend
on the implementation (latent dtype, indexer key format, paging), so always
state your convention. Statements about *why* a choice was made are marked
as interpretation when no source gives a reason.

## References

- [Mistral Large 4 `config.json`](https://huggingface.co/mistralai/Mistral-Large-4-1T-A52B/blob/main/config.json),
  [Hugging Face release post](https://huggingface.co/blog/mistral-large4) and
  [vLLM day-0 post](https://vllm.ai/blog/2026-10-29-mistral-large-4)
  — architecture, indexer and KV sharing, cache sizes in production;
- [GLM-5.3-Flash `config.json`](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json),
  [Transformers model page](https://huggingface.co/docs/transformers/en/model_doc/glm5_next) and
  [GLM-5.2 `config.json`](https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json);
- [Raschka (2026) — GLM-5.3-Flash architecture notes](https://sebastianraschka.com/blog/2026/glm-5-3-flash-architecture-notes.html)
  and the [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
  — compact side-by-side summaries of recent models;
- [GLM-5: from Vibe Coding to Agentic Engineering (2026)](https://arxiv.org/abs/2602.15763)
  — the GLM-5 family and its adoption of DSA;
- [DeepSeek-V3.2 (2025)](https://arxiv.org/abs/2512.02556)
  — DeepSeek Sparse Attention and the lightning indexer;
- [Kimi Linear (2025)](https://arxiv.org/abs/2510.26692)
  — Kimi Delta Attention and a 3:1 hybrid with NoPE MLA;
- [Gated Delta Networks (2024)](https://arxiv.org/abs/2412.06464)
  — the gated delta rule that KDA refines;
- [Hyper-Connections (2024)](https://arxiv.org/abs/2409.19606) and
  [mHC: Manifold-Constrained Hyper-Connections (2025)](https://arxiv.org/abs/2512.24880)
  — multi-stream residuals and their doubly stochastic constraint;
- [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434)
  — multi-head latent attention, as used in Module 9.

---

[:material-file-pdf-box: View Lecture Slides (PDF)](../slides/11-frontier-architectures.pdf){ .md-button target="_blank" }
[:material-code-tags: Optional companion material](../companion/11-frontier-architectures.md){ .md-button .md-button--primary }
