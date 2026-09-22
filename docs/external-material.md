# External material

This page collects the references behind the course. We thank the many authors of the papers, codebases and videos that are listed below.

## Course spine

- **[Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)**
  — The main video walkthrough for building and training the initial model.
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** — A compact, readable
  reference for the model and training loop. The repository is now deprecated,
  but remains useful pedagogically.
- **[nanochat](https://github.com/karpathy/nanochat)** — A more modern
  end-to-end reference covering tokenization, pretraining, evaluation and
  inference.
- **[Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language-models.pdf)**
  — The GPT-2 technical report and the target architecture for the first part
  of the course.

### GPT-2 from scratch

<div class="video-embed">
  <iframe
    src="https://www.youtube-nocookie.com/embed/l8pRSuU81PU"
    title="Let's reproduce GPT-2 (124M) by Andrej Karpathy"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

## Model and training foundations

_Sessions 1, 2 and 4._

### Papers

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** — The
  original Transformer architecture and scaled dot-product attention.
- **[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)**
  — The motivation behind AdamW and the distinction between weight decay and an
  L2 penalty under adaptive optimization.
- **[Mixed Precision Training](https://arxiv.org/abs/1710.03740)** — Master
  weights, reduced-precision computation and loss scaling.

### Blogs and websites

- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**
  — A line-by-line PyTorch implementation of the original encoder-decoder
  Transformer; especially useful for attention shapes, masking, residual paths
  and normalization.
- **[Transformer Explainer](https://poloclub.github.io/transformer-explainer/)**
  — An interactive visual walkthrough of a live GPT-2 model showing token
  embeddings, attention heads, LayerNorm, and MLP activations directly in the
  browser.

## Tokenization and data

_Sessions 3 and 5._

- **[minBPE](https://github.com/karpathy/minbpe)** — A minimal implementation
  and exercise set for byte-level BPE.
- **[The FineWeb Datasets](https://arxiv.org/abs/2406.17557)** — A documented
  large-scale filtering and deduplication pipeline with ablations.
- **[FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)**
  — The long-form report behind the paper, including extraction choices,
  unsuccessful global-deduplication strategies, custom filters and FineWeb-Edu.
- **[DataTrove](https://github.com/huggingface/datatrove)** — The open-source
  processing library and reproducible configurations used to build FineWeb.
- **[DataComp-LM](https://arxiv.org/abs/2406.11794)** — A controlled framework
  for comparing language-model dataset construction strategies.

## Single-GPU efficiency

_Session 6._

- **[How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/)** —
  The main hardware reference for SMs, warps, Tensor Cores, the memory
  hierarchy, collectives and compute/communication rooflines.
- **[PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)**
  — Practical starting points for measuring and improving PyTorch workloads.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** — The IO-aware view of
  exact attention and why reducing memory traffic changes performance.

## Distributed training

_Sessions 7–10._

- **[The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)**
  — A visual, interactive guide from single-GPU memory accounting through data,
  tensor, pipeline and context parallelism, backed by scaling experiments.
- **[Picotron](https://github.com/huggingface/picotron)** — The Playbook's
  minimal educational implementation of data, tensor, pipeline and context
  parallelism, organized into short, readable modules. Follow it with
  Ferdinand Mom's **[video playlist](https://www.youtube.com/playlist?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S)**
  and the accompanying **[tutorial code](https://github.com/huggingface/picotron_tutorial)**.
- **[Building a distributed training framework from first principles](https://www.youtube.com/watch?v=XoGvCBRnwLs)**
  — Umar Jamil's roughly 20-hour implementation walkthrough, from collectives
  and device meshes through dense and expert multidimensional parallelism.
- **[TorchTitan](https://github.com/pytorch/torchtitan)** — A PyTorch-native
  training platform that composes FSDP2, tensor, pipeline and context
  parallelism with checkpointing, compilation and lower-precision training.
- **[PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)**
  — A map of data, sharded-data, tensor and pipeline parallelism in PyTorch.
- **[DistributedDataParallel tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)**
  — Process-per-GPU training and gradient synchronization.
- **[FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)**
  — Parameter, gradient and optimizer-state sharding in practice.
- **[ZeRO](https://arxiv.org/abs/1910.02054)** — The memory accounting and
  sharding stages behind zero-redundancy training.
- **[Megatron-LM model parallelism](https://arxiv.org/abs/1909.08053)** — The
  column- and row-parallel linear-layer construction used in Session 9.
- **[PyTorch tensor-parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)**
  — A current implementation of column-wise, row-wise and sequence-parallel
  layouts.
- **[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)**
  — Sequence parallelism and selective activation recomputation.
- **[PyTorch context-parallel tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html)**
  — Splitting scaled dot-product attention over the sequence dimension.
- **[PyTorch pipeline-parallel tutorial](https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html)**
  — Pipeline stages, microbatches and schedules.
- **[Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473)**
  — How data, tensor and pipeline parallelism compose at scale.

We recomend watching those videos to get a better understanding of distributed training:

### Picotron tutorial playlist

<div class="video-embed">
  <iframe
    src="https://www.youtube-nocookie.com/embed/videoseries?list=PL-_armZiJvAnhcRr6yTJ0__f3Oi-LLi9S"
    title="Picotron distributed-training tutorial playlist by Ferdinand Mom"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

### Distributed training from first principles

<div class="video-embed">
  <iframe
    src="https://www.youtube-nocookie.com/embed/XoGvCBRnwLs"
    title="Building a distributed training framework from first principles by Umar Jamil"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

### TorchTitan talk

<div class="video-embed">
  <iframe
    src="https://www.youtube-nocookie.com/embed/WsNEBxPDljU"
    title="TorchTitan: large-scale LLM training using native PyTorch 3D parallelism"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
</div>

## Efficient inference and serving

_Sessions 25–28 (Modules 9 and 10)._

- **[KV Cache Explained Intuitively](https://medium.com/@saad.ahmed1926q/kv-cache-explained-intuitively-2b425a36dfc7)**
  — A visual introduction to repeated autoregressive work, prefill and cached
  token generation.
- **[Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)**
  — Multi-query attention and the memory-bandwidth cost of incremental
  decoding.
- **[GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)**
  — Grouped-query attention as a quality/efficiency compromise between MHA and
  MQA.
- **[Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)**
  — The paper behind vLLM's paged KV-cache design.
- **[Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm)**
  — A guided tour of the vLLM V1 engine core, scheduler, KV-cache manager,
  advanced features and distributed serving stack.
- **[Continuous batching from first principles](https://huggingface.co/blog/continuous_batching)**
  — A visual derivation of ragged batching, chunked prefill and dynamic
  scheduling from attention and KV caching.
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**
  — A lightweight, educational re-implementation of vLLM's scheduler, block tables,
  and PagedAttention mechanism in clean, readable Python.
- **[vLLM documentation](https://docs.vllm.ai/)** — The serving engine used to
  study continuous batching, chunked prefill, prefix caching and production
  metrics.

## Frontier architectures and efficient generation

_Session 30 (Module 11)._

- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**
  — Foundational draft-and-verify paradigm and proof of provably unbiased rejection sampling.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)**
  — Multi-Head Latent Attention (MLA) for low-rank KV-cache compression, fine-grained DeepSeekMoE with isolated shared experts, auxiliary-loss-free load balancing, and Multi-Token Prediction (MTP).
- **[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)**
  — Selective state-space models, input-dependent conditioning, and hardware-aware associative scan.
- **[Transformers are SSMs: Generalized Models and State Space Duality](https://arxiv.org/abs/2405.21060)**
  — Mamba-2 and theoretical unification of structured masked attention and state-space duality.
- **[Linear Transformers with Learnable Kernel Functions (Delta Net)](https://arxiv.org/abs/2102.11174)**
  — Applying the classical delta rule to associative recurrent memory to prevent catastrophic saturation.
- **[Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)**
  — Masked diffusion formulation of non-autoregressive language modeling.
- **[Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818)**
  — Direct patch projection and unified tokenization without frozen vision encoders.
- **[Switch Transformers](https://arxiv.org/abs/2101.03961)**
  — Sparse mixture-of-experts routing and its computation/communication trade-offs.

## Acknowledgements

This course draws inspiration from the **[MVA Training Large Models Course](https://training-large-models-course.github.io)** by Edouard Oyallon.