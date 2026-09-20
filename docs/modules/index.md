# Session material

These pages are the conceptual spine of the course. Each one develops the
mechanism, derives the quantities students need to reason about it, and ends
with an evidence-producing investigation. Slides and implementation guides can
be built from this material without replacing it.

Every session follows the same rhythm:

1. introduce one concrete systems or modeling problem;
2. explain the minimum theory needed to reason about it;
3. implement or measure the corresponding mechanism;
4. add the result to the continuous course project.

| Session | Topic | Main practical outcome |
| ---: | --- | --- |
| 1 | Transformer from first principles | Minimal decoder-only model |
| 2 | Training-loop anatomy | Correct training and checkpoint loop |
| 3 | BPE and the data pipeline | Tokenizer and packed dataset |
| 4 | Train a small GPT | Baseline checkpoint |
| 5 | Data selection | Controlled selected-data experiment |
| 6 | Single-GPU performance | Profiled and optimized trainer |
| 7 | Distributed data parallelism | Correct DDP training |
| 8 | FSDP and ZeRO | Sharding comparison |
| 9 | Tensor parallelism | Toy sharded layer |
| 10 | Context and pipeline parallelism | Parallelism design exercise |
| 11 | KV-cached decoding | Correct cached decoder |
| 12 | Serving systems | Latency and throughput benchmark |
| 13 | Beyond dense Transformers | Architecture comparison and synthesis |

Use the navigation to open a session page.
