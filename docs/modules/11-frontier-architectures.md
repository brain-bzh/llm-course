# Module 11 — Frontier architectures and efficient generation

## Purpose

Examine the frontiers of model architecture and inference acceleration that go beyond
standard dense pre-norm Transformers.

As language models scale to multi-million token contexts and high-throughput production
serving, two fundamental physical walls emerge:

1. **The memory-bandwidth wall during generation:** Standard autoregressive decoding is
   strictly memory-bound with arithmetic intensity $\sim 1\text{ FLOP/byte}$.
2. **The quadratic scaling wall of softmax attention:** Standard causal self-attention
   scales $\mathcal{O}(T^2)$ in computation and $\mathcal{O}(T)$ in KV-cache memory.

This module analyzes the modern architectural innovations breaking through these walls:
**speculative decoding**, **discrete and continuous diffusion for text**,
**DeepSeek architectural breakthroughs (MLA, fine-grained MoE, MTP)**,
**encoder-free multimodal architectures**, and
**linear attention / recurrent state-space models (Mamba, GDN, KDA)**.

---

## Key ideas

- speculative decoding mechanics, rejection sampling verification, and tree-based speculation;
- diffusion models for text generation: score matching, iterative denoising, and continuous vs discrete spaces;
- Multi-Head Latent Attention (MLA) and low-rank KV-cache compression with decoupled RoPE;
- fine-grained mixture of experts (DeepSeekMoE), isolated shared experts, and aux-loss-free balancing;
- multi-token prediction (MTP) architectures;
- encoder-free multimodal architectures: direct patch projection vs frozen vision towers (CLIP/SigLIP);
- linear attention, selective state-space models (Mamba/Mamba-2), and Gated Delta Networks (GDN).

---

## Speculative decoding: Breaking the memory-bandwidth wall

In standard autoregressive decoding, generating each token requires reading all model
parameters $P$ from high-bandwidth memory (HBM) into SRAM to perform a single vector-matrix
multiplication:

$$\text{Arithmetic Intensity} = \frac{2P \text{ FLOPs}}{2P \text{ Bytes}} = 1 \text{ FLOP / Byte}.$$

Modern accelerators (like H100) deliver $> 1000\text{ TFLOPs/s}$ of compute but only
$3.35\text{ TB/s}$ of memory bandwidth. Consequently, GPU compute units sit $> 99\%$ idle
during single-sequence token generation.

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
   
   Because evaluating $K$ tokens in parallel converts memory-bound vector-matrix operations
   into compute-bound matrix-matrix multiplications (GEMMs), the target verification takes
   nearly the exact same time as generating a single token!

### 2. Provably unbiased rejection sampling

To ensure the output text matches the exact target model distribution without any bias,
each draft token $x_t$ is evaluated sequentially with rejection sampling:

- Accept $x_t$ with probability:
  
  $$\min\left(1, \frac{p_{\text{target}}(x_t \mid x_{< t})}{p_{\text{draft}}(x_t \mid x_{< t})}\right).$$

- If $x_t$ is rejected, stop and sample the replacement token from the residual distribution:
  
  $$p_{\text{residual}}(x) = \frac{\max(0, p_{\text{target}}(x) - p_{\text{draft}}(x))}{\sum_{x'} \max(0, p_{\text{target}}(x') - p_{\text{draft}}(x'))}.$$

- If all $K$ tokens are accepted, sample one bonus token from $p_{\text{target}}(x_{K+1} \mid x_{\le K})$.

### 3. Expected speedup & advanced speculation

For an average acceptance rate $\alpha$, the expected accepted tokens per step is:

$$\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}.$$

For $\alpha = 0.8$ and $K=5$, $\mathbb{E} \approx 3.36$ tokens per step, yielding a
$2\text{--}3\times$ wall-clock speedup.

- **Medusa / EAGLE:** Eliminate the external draft model entirely. Medusa adds $M$ lightweight
  MLP prediction heads atop the target model. EAGLE conditions draft prediction on the target
  model's top-layer hidden states, reaching acceptance rates $\alpha > 0.85$.
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

Rather than predicting only token $x_{t+1}$, Multi-Token Prediction adds sequential prediction
modules that forecast $k$ future tokens ($x_{t+1}, \dots, x_{t+k}$) concurrently:

- Each MTP module consists of a Transformer layer that combines the previous layer's hidden
  state with future token representations.
- Enhances representation learning during pretraining.
- The MTP modules can be detached and reused as native draft heads for **speculative decoding**
  at zero additional training cost!

---

## Encoder-free multimodal architectures

First-generation vision-language models (e.g. LLaVA, Flamingo) pair a pretrained causal LLM
with a **separate, frozen vision encoder** (such as CLIP or SigLIP) connected via a cross-attention
adapter or MLP projector:

$$\text{Image} \;\longrightarrow\; \boxed{\text{Vision Tower (CLIP/SigLIP)}} \;\longrightarrow\; \boxed{\text{Projector}} \;\longrightarrow\; \boxed{\text{Causal Decoder}}.$$

```
                     Traditional Multimodal Architecture
Image ──> [Frozen ViT (CLIP)] ──> [MLP Adapter] ──\
                                                   ├──> [Causal LLM Decoder]
Text  ──> [Token Embedding] ──────────────────────/

                     Encoder-Free Architecture (e.g. Gemma 3)
Image ──> [Linear Patch Projection] ──────────────\
                                                   ├──> [Unified Causal Decoder]
Text  ──> [Token Embedding] ──────────────────────/
```

### Limitations of dual-encoder designs

1. **Resolution bottleneck & information loss:** Vision encoders compress images into a fixed
   grid of vectors trained on contrastive image-text matching, stripping fine-grained spatial
   and document-layout details.
2. **Modality misalignment:** Visual and textual representations reside in different geometric
   spaces, requiring expensive alignment stages.
3. **Inference overhead:** Running a separate heavy vision backbone increases TTFT and latency.

### Native encoder-free architectures (e.g. Gemma 3 / Chameleon)

Modern architectures tokenize raw images directly into the unified causal model:

1. **Patch projection:** The input image is divided into $P \times P$ non-overlapping patches
   and mapped directly to the model's hidden dimension $d$ via a single linear or lightweight
   convolutional layer:
   
   $$e_{\text{patch}} = \text{Conv2d}(\text{Image}, \text{kernel}=P, \text{stride}=P) \in \mathbb{R}^{N_{\text{patches}} \times d}.$$

2. **Unified causal autoregression:** Visual patch tokens and text tokens are placed in the
   exact same token sequence and processed through the exact same Transformer blocks.
3. **End-to-end multimodal gradients:** Every parameter in the model updates with respect to
   both visual and textual loss, enabling native interleaved image-text reasoning and generation.

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
continually accumulates new data without discarding stale information.

**Gated Delta Networks (GDN)** apply the classical **delta rule** to associative memory:

$$S_t = S_{t-1} (I - \beta_t K_t K_t^\top) + \beta_t V_t K_t^\top,$$

where $\beta_t$ is a dynamic learning rate or decay gate.
If a key $K_t$ already exists in memory, the $(I - \beta_t K_t K_t^\top)$ term actively
**erases the old associated value** before writing the new one, solving the catastrophic
forgetting and capacity saturation problem of linear RNNs.

### 4. Kernelized Dynamic Attention (KDA) / Kangaroo

KDA architectures combine lightweight dynamic state tracking with selective attention
anchor points, allowing sub-quadratic processing of multi-million token streams while
retaining precise token retrieval capability.

---

## Architectural Comparison Matrix

| Architecture | Attention / Sequence Operator | Training Complexity | Inference KV-Memory | Key Strength | Dominant Bottleneck |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Transformer** | Causal Softmax Attention | $\mathcal{O}(T^2)$ | $\mathcal{O}(T)$ (Large MHA/GQA) | Exact associative recall | Memory bandwidth & quadratic context |
| **DeepSeek (MLA + MoE)** | Low-Rank Latent Attention | $\mathcal{O}(T^2)$ | $\mathcal{O}(T)$ ($80\%\text{--}90\%$ smaller) | Extreme parameter and cache efficiency | High All-to-All network communication |
| **Speculative Decoding** | Target MHA + Draft verification | Standard | Standard | $2\text{--}3\times$ lower decode latency | Draft model alignment & acceptance variance |
| **Mamba-2 (SSD)** | Selective State-Space / 1-SS | $\mathcal{O}(T)$ | $\mathcal{O}(1)$ (Fixed state $h$) | Infinite-context linear throughput | Sub-quadratic recall on in-context needle retrieval |
| **Gated Delta Net (GDN)** | Delta-Rule Recurrent Memory | $\mathcal{O}(T)$ | $\mathcal{O}(1)$ (Fixed state $S$) | Dynamic memory erasure and update | Matrix state dimension $d \times d$ compute |
| **Encoder-Free Multimodal** | Unified Causal Transformer | $\mathcal{O}(T^2)$ | $\mathcal{O}(T)$ | End-to-end multimodal alignment | Long sequence lengths from vision patches |

---

## Practical task

1. **MLA memory calculation:** Derive the precise byte savings of Multi-Head Latent Attention
   versus standard MHA and GQA across sequence lengths from $4\text{k}$ to $128\text{k}$.
2. **Speculative decoding simulation:** Model expected token acceleration $\mathbb{E}[\text{tokens/step}]$
   as a function of draft length $K \in \{1, \dots, 8\}$ and empirical acceptance rate $\alpha \in [0.5, 0.95]$.
3. **Linear recurrent step:** Implement a minimal Gated Delta Network update step and compare
   its memory footprint against an autoregressive KV-cache during generation.

---

## Expected output

A comparative systems analysis evaluating how MLA, speculative decoding, Mamba-2, and GDN
shift the compute, memory, and communication bottlenecks established throughout this course.

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
  — encoder-free unified multimodal tokenization.

---

[:material-file-pdf-box: View Lecture Slides (PDF)](../slides/11-frontier-architectures.pdf){ .md-button target="_blank" }
[:material-code-tags: Practical Companion Guide](../companion/11-frontier-architectures.md){ .md-button .md-button--primary }
