# Module 11 — Frontier architectures and efficient generation

## Purpose

Examine the frontiers of model architecture and inference acceleration that go beyond
standard dense pre-norm Transformers.

As language models scale to multi-million token contexts and high-throughput production
serving, two fundamental physical walls emerge:

1. **The memory-bandwidth wall during generation:** Standard autoregressive decoding is
   often bandwidth-bound at small batch sizes, with weight-only arithmetic intensity $\sim 1\text{ FLOP/byte}$.
2. **The quadratic scaling wall of softmax attention:** Standard causal self-attention
   scales $\mathcal{O}(T^2)$ in computation and $\mathcal{O}(T)$ in KV-cache memory.

This module analyzes the modern architectural innovations breaking through these walls:
**speculative decoding**, **discrete and continuous diffusion for text**,
**DeepSeek architectural breakthroughs (MLA, fine-grained MoE, MTP)**,
**multimodal input architectures**, and
**linear attention / recurrent state-space models (Mamba, GDN, KDA)**.

---

## Session 30: required core and optional reading

Use the 75-minute session for three comparisons: speculative verification
(25 minutes), MLA cache accounting (20 minutes), and recurrent state versus
KV cache (20 minutes), followed by a 10-minute synthesis discussion. For each,
identify the original bottleneck, the new cost, and an experiment that could
show a benefit. Diffusion, detailed MoE/MTP, multimodality, and individual
recurrent variants are optional reading. There is no additional coding report.

## Key ideas

- speculative decoding mechanics, rejection sampling verification, and tree-based speculation;
- diffusion models for text generation: score matching, iterative denoising, and continuous vs discrete spaces;
- Multi-Head Latent Attention (MLA) and low-rank KV-cache compression with decoupled RoPE;
- fine-grained mixture of experts (DeepSeekMoE), isolated shared experts, and aux-loss-free balancing;
- multi-token prediction (MTP) architectures;
- multimodal input architectures: vision encoders, discrete image tokenizers, and direct patch projection;
- linear attention, selective state-space models (Mamba/Mamba-2), and Gated Delta Networks (GDN).

---

## Speculative decoding: Breaking the memory-bandwidth wall

In standard autoregressive decoding, generating each token requires reading all model
parameters $P$ from high-bandwidth memory (HBM) into SRAM to perform a single vector-matrix
multiplication:

$$\text{Arithmetic Intensity} = \frac{2P \text{ FLOPs}}{2P \text{ Bytes}} = 1 \text{ FLOP / Byte}.$$

This estimate assumes one sequence, dense weights stored in two bytes per
parameter, and weight traffic dominating the step. Batching amortizes weight
reads; long contexts add substantial KV traffic. Use the measured workload
and device roofline before concluding that compute is underutilized.

```
[ Small Draft Model ] ──(Proposes K draft tokens)──> [ x₁, x₂, ..., x_K ]
                                                              │
                                                     (Parallel Forward)
                                                              ▼
[ Large Target Model ] ──(Verifies all K in 1 step)──> [ Accept k ≤ K + 1 tokens ]
```

### 1. The draft-and-verify paradigm

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) decouples draft proposal
from target verification:

1. **Draft Phase:** A small, fast draft model (e.g. 1B) autoregressively generates $K$
   candidate tokens: $[x_1, \dots, x_K]$.
2. **Verification Phase:** The large target model (e.g. 70B) runs a **single parallel forward pass**
   evaluating all $K$ tokens simultaneously:
   
   $$p_{\text{target}}(x_t \mid x_{< t}) \quad \text{for all } t \in \{1, \dots, K\}.$$
   
   Evaluating several positions together can reuse target weights and improve
   arithmetic intensity. Verification cost still depends on draft length, context,
   batch size, and kernels; it must be measured rather than assumed equal to one
   ordinary decode step.

### 2. Provably unbiased rejection sampling

To ensure the output text matches the exact target model distribution without any bias,
each draft token $x_t$ is evaluated sequentially with rejection sampling:

- Accept $x_t$ with probability:
  
  $$\min\left(1, \frac{p_{\text{target}}(x_t \mid x_{< t})}{p_{\text{draft}}(x_t \mid x_{< t})}\right).$$

- If $x_t$ is rejected, stop and sample the replacement token from the residual distribution:
  
  $$p_{\text{residual}}(x) = \frac{\max(0, p_{\text{target}}(x) - p_{\text{draft}}(x))}{\sum_{x'} \max(0, p_{\text{target}}(x') - p_{\text{draft}}(x'))}.$$

- If all $K$ tokens are accepted, sample one bonus token from $p_{\text{target}}(x_{K+1} \mid x_{\le K})$.

### 3. Expected speedup & advanced speculation

In a simplified model with constant conditional acceptance probability
$\alpha$ at each draft position, the expected emitted tokens per verification
step (including the replacement or bonus token) is:

$$\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}.$$

For $\alpha = 0.8$ and $K=5$, $\mathbb{E} \approx 3.69$ emitted tokens.
That is not yet a wall-clock speedup. Let $t_{\mathrm{base}}$ be the ordinary
target decode time, $t_{\mathrm{draft}}(K)$ the time to propose $K$ tokens, and
$t_{\mathrm{verify}}(K)$ the verification time. A simple cost model is

$$S \approx \frac{\mathbb{E}[N] \, t_{\mathrm{base}}}
{t_{\mathrm{draft}}(K) + t_{\mathrm{verify}}(K) + t_{\mathrm{overhead}}}.$$

For illustrative costs of 10 ms baseline, 8 ms drafting, 12 ms verification,
and 2 ms overhead, the speedup is about $1.68\times$. If drafting takes 30 ms,
it falls below $1\times$. Increasing draft length is useful only while its
extra accepted tokens outweigh its extra cost. At $\alpha=1$, use the limit
$\mathbb{E}[N]=K+1$.

- **Alternative draft mechanisms:** Auxiliary heads or feature-based draft
  networks can replace a separate standalone language model. Their training,
  acceptance rates, and verification cost remain part of the comparison.
- **Tree-based speculation:** Generates candidate trees instead of linear sequences, verifying
  multiple token branches concurrently using tree attention masks.

---

## Diffusion models for language generation

While autoregressive models generate text strictly left-to-right, **diffusion models**
formulate generation as iterative denoising:

$$x_T \sim \mathcal{N}(0, I) \quad \longrightarrow \quad x_{T-1} \quad \longrightarrow \quad \dots \quad \longrightarrow \quad x_0.$$

### Continuous vs discrete diffusion for text

Language is discrete (categorical tokens), creating a fundamental challenge for diffusion:

1. **Continuous diffusion in embedding space (e.g. Diffusion-LM, Plaid):**
   Tokens are mapped to continuous embedding vectors $e(x) \in \mathbb{R}^d$. Gaussian noise
   is added in embedding space according to a forward SDE:
   
   $$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I).$$
   
   A Transformer neural network predicts the denoised embeddings, which are projected back
   to vocabulary logits via a rounding operation: $p(w) \propto \exp(-\|w - \hat{x}_0\|^2)$.
2. **Discrete state-space diffusion:**
   Noise is defined directly over discrete vocabulary states via continuous-time Markov chains
   or transition matrices $Q_t$. Tokens are masked or uniformly corrupted, and the model
   learns reverse transition probabilities.

### Non-autoregressive trade-offs

| Dimension | Autoregressive (Causal GPT) | Diffusion / Non-Autoregressive |
| :--- | :--- | :--- |
| **Generation order** | Strictly sequential left-to-right | Global, bidirectional iterative refinement |
| **Latency scaling** | $\mathcal{O}(N)$ (proportional to generated length) | $\mathcal{O}(S)$ (fixed $S$ denoising steps, independent of length) |
| **Controllability** | Prefix-guided only | Infilling, attribute guidance, and arbitrary token conditioning |
| **Per-token quality** | Gold standard (monotonic chain-rule probability) | Prone to local inconsistencies and error accumulation |

---

## DeepSeek architectural lineage: Overcoming scaling limits

DeepSeek's recent architectures (DeepSeek-V2, V3, and Flash variants) introduced several
defining systems-level modifications to standard Transformer layers.

```
                  Standard Multi-Head Attention (MHA)
Query ───> [H Heads] \
Key   ───> [H Heads]  ├──> Cache Size = 2 * L * H * d_h * T  (Memory Crisis!)
Value ───> [H Heads] /

                  Multi-Head Latent Attention (MLA)
Key/Val ─(Low-rank Down-proj)─> Latent Vector c_t^{KV}  (Cache Size: d_c * T, 80%+ smaller!)
                                       │
                                (Decoupled RoPE k_t^R)
```

### 1. Multi-Head Latent Attention (MLA)

In long-context inference ($T \ge 32\text{k}$), the KV-cache dominates GPU memory.
Grouped-Query Attention (GQA) reduces head count, but degrades model expressivity when
$H_{kv} \ll H_q$.

**Multi-Head Latent Attention (MLA)** compresses the Key and Value vectors into a single
low-dimensional latent vector $c_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll d$):

1. **Down-projection during caching:**
   
   $$c_t^{KV} = W^{DKV} x_t, \quad c_t^{KV} \in \mathbb{R}^{d_c}.$$
   
   **Only $c_t^{KV}$ is stored in the KV-cache.** This achieves an $80\%\text{--}90\%$ reduction
   in inference memory compared to standard MHA.
2. **Decoupled Rotary Positional Embeddings (RoPE):**
   Standard RoPE mixes position information into $K$, preventing linear absorption.
   MLA decouples positional representations into a separate small head:
   
   $$q_{t, i} = [ W^{UQ}_i c_t^Q, \text{RoPE}(W^{QR} c_t^Q) ], \quad k_{t, i} = [ W^{UK}_i c_t^{KV}, \text{RoPE}(W^{KR} x_t) ].$$

During inference, $W^{UK}$ and $W^{UV}$ can be absorbed directly into the query and output
projection matrices, eliminating key and value reconstruction entirely.

### 2. DeepSeekMoE: Fine-grained experts & isolated shared experts

Traditional MoE (like Switch Transformer or Mixtral 8x7B) routes tokens among a few large
experts (e.g. 8 experts, activating top-2).

DeepSeekMoE alters the expert granularity:
- **Fine-grained expert division:** Instead of 8 large experts of size $4d^2$, the model
  instantiates 256 fine experts of size $0.25d^2$, activating 32. This enables combinatorial
  specialization ($\binom{256}{32}$ vs $\binom{8}{2}$).
- **Isolated shared experts:** Several experts are kept permanently active for all tokens.
  These shared experts capture common language modeling features and grammar, preventing
  specialized experts from redundantly storing general knowledge.

### 3. Auxiliary-Loss-Free Load Balancing

Standard MoE models rely on auxiliary load-balancing losses to prevent routing collapse
(where one expert receives all tokens). However, strong auxiliary losses degrade primary
cross-entropy learning.

DeepSeek replaces auxiliary loss with **dynamic expert bias adjustment**:
- A learnable bias $b_e$ is added to each expert's routing score: $s_{t, e} = \text{softmax}(x_t^\top w_e) + b_e$.
- If expert $e$ is overloaded, $b_e$ is decreased; if underloaded, $b_e$ is increased.
- Gradients from the task loss do not update $b_e$, preserving training stability without
  compromising modeling quality.

### 4. Multi-Token Prediction (MTP)

Rather than predicting only token $x_{t+1}$, Multi-Token Prediction trains sequential prediction
modules that forecast $k$ future tokens ($x_{t+1}, \dots, x_{t+k}$) concurrently:

- Each MTP module consists of a Transformer layer that combines the previous layer's hidden
  state with future token representations.
- Enhances representation learning during pretraining by requiring deeper multi-step planning.
- At inference time, the trained MTP modules can be reused directly as native draft heads for
  **speculative decoding**, eliminating the need to train and host an independent draft model.

---

## Multimodal input architectures (optional)

Separate three decisions: how an image becomes tokens or features, where those
representations enter the language model, and which components are trainable.
Early fusion does not by itself imply an encoder-free input path.

- **Vision encoder plus language model:** Gemma 3 uses a SigLIP vision encoder;
  it is not a direct-patch encoder-free example. See the
  [Gemma 3 developer guide](https://developers.googleblog.com/introducing-gemma3/).
- **Discrete image tokens and early fusion:** Chameleon represents images with
  a learned image tokenizer and models mixed image/text token sequences. This
  is distinct from feeding raw patches through a single linear projection.
  See the [Chameleon paper](https://arxiv.org/abs/2405.09818).
- **Direct patch projection:** A patch embedding can map raw patches into the
  shared model's hidden dimension. Whether later layers are shared or specialized,
  and which weights are frozen, must be checked for the particular architecture.

For a systems comparison, count image tokens, preprocessing/encoder latency,
prefill work, and cache growth. Compare quality at a declared image resolution;
a shorter representation can reduce cost while losing useful detail.

---

## Linear attention & modern recurrent architectures

Standard self-attention computes:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V.$$

The softmax operator couples queries and keys non-linearly, enforcing quadratic $\mathcal{O}(T^2)$
complexity and requiring the entire history to be stored in the KV cache.

### 1. The linear attention formulation

By replacing the softmax with a kernel feature map $\phi(x)$, attention becomes associative:

$$\text{Attn}(Q, K, V) = \frac{(\phi(Q) \phi(K)^\top) V}{\phi(Q) \sum \phi(K)^\top} = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \sum \phi(K)^\top}.$$

During decoding, the term $S_t = \sum_{\tau=1}^t \phi(K_\tau)^\top V_\tau \in \mathbb{R}^{d \times d}$
acts as a **constant-size recurrent state**:

$$S_t = S_{t-1} + \phi(K_t)^\top V_t.$$

Inference requires **$\mathcal{O}(1)$ memory per step**, completely eliminating the growing KV-cache!

### 2. Mamba & Mamba-2 (Selective State-Space Models)

Continuous-time State-Space Models map an input $x(t)$ to an output $y(t)$ through a hidden state $h(t)$:

$$h'(t) = A h(t) + B x(t), \quad y(t) = C h(t).$$

**Mamba (Gu & Dao, 2023)** made SSMs input-dependent:
- The parameters $\Delta, B, C$ are parameterized as functions of the current input $x_t$.
- This enables the model to selectively remember or filter out information based on content.
- A hardware-aware **parallel associative scan** enables fast parallel training on GPUs.
- **Mamba-2 (State Space Duality):** Proves that selective SSMs and 1-semiseparable structured
  masked attention are mathematically dual representations of the same operation.

### 3. Gated Delta Networks (GDN / DeltaNet)

Standard linear attention suffers from memory capacity limits: $S_t = S_{t-1} + K_t^\top V_t$
continually accumulates new associations without discarding stale information.

**Gated Delta Networks (GDN)** apply the classical **delta rule** to associative memory:
the model predicts the current retrieved value $\hat{V}_t = S_{t-1} K_t$ and updates
the memory using the prediction error $(V_t - \hat{V}_t)$:

$$S_t = \alpha_t S_{t-1} + \beta_t (V_t - S_{t-1} K_t) K_t^\top = \alpha_t S_{t-1} (I - \beta_t K_t K_t^\top) + \beta_t V_t K_t^\top,$$

where:
- $\alpha_t \in (0, 1)$ acts as a dynamic **forget/decay gate**, controlling the persistence of historical memory;
- $\beta_t \in (0, 1)$ is a data-dependent **write gate** (learning rate) that controls the magnitude of the associative update;
- the $(I - \beta_t K_t K_t^\top)$ term projects along the key vector to **actively erase stale associations**, solving the capacity saturation problem of un-gated linear RNNs.

### 4. Kimi Delta Attention (KDA)

[Kimi Linear](https://arxiv.org/abs/2510.26692) introduces Kimi Delta Attention,
which extends Gated DeltaNet with finer-grained gating. Kimi Linear is a hybrid
architecture: fixed-size recurrent state in some layers does not imply that
the entire model has constant cache memory if other layers retain attention.

---

## Architectural Comparison Matrix

| Architecture | Attention / Sequence Operator | Training Complexity | Inference KV-Memory | Key Strength | Dominant Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Transformer** | Causal Softmax Attention | $\mathcal{O}(T^2)$ | $\mathcal{O}(T)$ (Large MHA/GQA) | Exact associative recall | Memory bandwidth & quadratic context |
| **DeepSeek (MLA + MoE)** | Low-Rank Latent Attention | $\mathcal{O}(T^2)$ | $\mathcal{O}(T)$ ($80\%\text{--}90\%$ smaller) | Extreme parameter and cache efficiency | High All-to-All network communication |
| **Speculative Decoding** | Target MHA + Draft verification | Standard | $\mathcal{O}(T)$ (Target + draft caches) | Empirical $1.5\text{--}2.5\times$ decode latency reduction | Draft acceptance rate $\alpha$, drafting latency, and verification kernel overhead |
| **Mamba-2 (SSD)** | Selective State-Space / 1-SS | $\mathcal{O}(T)$ | $\mathcal{O}(1)$ (Fixed state $h$) | Long-context linear throughput | Bounded memory capacity on complex multi-needle retrieval |
| **Gated Delta Net (GDN)** | Delta-Rule Recurrent Memory | $\mathcal{O}(T)$ | $\mathcal{O}(1)$ (Fixed state $S$) | Dynamic associative erasure and update | Matrix state dimension $d \times d$ compute |
| **Multimodal token sequence** | Causal Transformer with image representations | $\mathcal{O}(T^2)$ | $\mathcal{O}(T)$ | End-to-end multimodal alignment | Long sequence lengths from vision patches |

---

## Optional exploration

1. **MLA memory calculation:** Derive the precise byte savings of Multi-Head Latent Attention
   versus standard MHA and GQA across sequence lengths from $4\text{k}$ to $128\text{k}$.
2. **Speculative decoding simulation:** Model expected token acceleration $\mathbb{E}[\text{tokens/step}]$
   and the wall-clock cost model as a function of draft length $K \in \{1, \dots, 8\}$ and empirical acceptance rate $\alpha \in [0.5, 0.95]$.
3. **Linear recurrent step:** Implement a minimal Gated Delta Network update step and compare
   its memory footprint against an autoregressive KV-cache during generation.

---

## Expected output

An in-class explanation of the three core comparisons: what cost is reduced,
what cost remains, and what evidence would establish a useful trade-off. The
explorations above are optional and do not add a required report.

---

## References

- [Leviathan et al. (2023) — Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
  — foundational draft-and-verify algorithm and rejection sampling proof;
- [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437)
  — Multi-Head Latent Attention (MLA), DeepSeekMoE, Aux-loss-free balancing, and MTP;
- [Gu & Dao (2023) — Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
  — selective state-space mechanics and hardware-aware scan;
- [Dao & Gu (2024) — Transformers are SSMs: Generalized Models and State Space Duality](https://arxiv.org/abs/2405.21060)
  — Mamba-2 and theoretical unification of attention and SSMs;
- [Schlag et al. (2021) — Linear Transformers with Learnable Kernel Functions (Delta Net)](https://arxiv.org/abs/2102.11174)
  — delta-rule memory update mechanics;
- [Chameleon Team (2024) — Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)
  — discrete image tokenization and mixed-modal early fusion.
- [Kimi Team (2025) — Kimi Linear](https://arxiv.org/abs/2510.26692)
  — Kimi Delta Attention and hybrid recurrent/attention layers.

---

[:material-file-pdf-box: View Lecture Slides (PDF)](../slides/11-frontier-architectures.pdf){ .md-button target="_blank" }
[:material-code-tags: Practical Companion Guide](../companion/11-frontier-architectures.md){ .md-button .md-button--primary }
