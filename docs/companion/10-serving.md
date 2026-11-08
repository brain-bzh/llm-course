# Module 10 Lab — Read a serving engine through a simulator

This lab accompanies [Module 10: Serving systems](../modules/10-serving.md).
Session 28 is a demonstration session: the instructor runs the simulator below
and, if the preflight succeeded, a live engine. There is no implementation
deliverable. Use this page during the session to follow the trace, and after it
to rerun the comparisons yourself (about 30 minutes).

!!! tip "Files used in this lab"
    - **Simulator:** [`nanolm/serving_sim.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/nanolm/serving_sim.py)
    - **Script:** [`scripts/10_serving_benchmark.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/scripts/10_serving_benchmark.py)
    - **Tests:** [`tests/test_module_10_serving.py`](https://github.com/brain-bzh/llm-course-companion/blob/main/tests/test_module_10_serving.py)
    - **Interactive version:** [continuous-batching simulator](../demos/continuous-batching.html){ target="_blank" }
    - **Readable engine:** [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), a small Python implementation of paged attention and continuous batching

## The model in the simulator

Eight requests arrive over 120 ms with different prompt and output lengths.
Each engine iteration costs

$$t_{\text{iter}} = 15\ \text{ms} + 0.2\ \text{ms} \times (\text{prompt tokens prefilled in this iteration})$$

so a long prompt delays its own first token **and** slows every request that
is decoding in the same iteration. KV memory comes in blocks of 16 tokens. Four
policies are compared:

| | **reserve** KV: hold prompt + 32 tokens from admission | **paged** KV: grow one block at a time, preempt if full |
| --- | --- | --- |
| **static** batching: admit only when the batch is empty | | |
| **continuous** batching: admit whenever a slot frees | | |

The simulator is deterministic and has no GPU, no attention cost that grows
with context, no chunked prefill and no prefix caching. It teaches the
scheduling logic; it does not predict a real engine's numbers.

## Run it

```bash
uv run python scripts/10_serving_benchmark.py
```

```text
workload: 8 requests; batch <= 4; 24 KV blocks x 16 tokens
times in ms of simulated clock

policy                   TTFT mean    TTFT max    ITL mean   out tok/s    peak run peak blocks     preempt
static + reserve            392.65       708.6        15.0       126.7           4          18           0
static + paged              392.65       708.6        15.0       126.7           4          13           0
continuous + reserve        138.45       338.6        15.9       223.5           4          20           0
continuous + paged          138.45       338.6        15.9       223.5           4          17           0
```

### Questions (15 minutes)

1. **Batching.** Continuous batching cuts the mean TTFT by about 3× and raises
   output throughput. Where does a static batch waste time? (Look at what a
   finished short request's slot does while the longest request runs.)
2. **Interference.** The mean ITL rises from 15.0 to 15.9 ms with continuous
   batching. Which term of $t_{\text{iter}}$ causes it? What would chunked
   prefill change?
3. **Memory.** With plenty of memory, the allocation policy does not change the
   latencies, only the peak blocks. Why?

Now make memory scarce and allow larger batches:

```bash
uv run python scripts/10_serving_benchmark.py --blocks 10 --batch 8
```

4. Which policy now runs the most requests at once, and what did it pay for that?
   (A preempted request loses its KV cache and recomputes its prefill.)
5. Compute the internal fragmentation of the **reserve** policy for request 3
   (16-token prompt, 8 output tokens): blocks reserved vs blocks it ever needs.

### Follow one request (10 minutes)

```bash
uv run python scripts/10_serving_benchmark.py --trace
```

Each line is one engine iteration: running requests as `id:generated/output`,
arrived-but-waiting requests, and KV blocks in use. Find:

- the iteration that admits requests 2 and 3, and why it lasts 27.8 ms instead of 15 ms;
- the first iteration after request 3 finishes, and when request 5 joins;
- the TTFT of request 5, decomposed into queueing time and its own prefill.

### Read the metrics as a system (5 minutes)

- **TTFT** includes queueing; a low *median* TTFT can hide a long tail (`TTFT max`).
- **ITL** is what a streaming user feels; it rises when prefill shares iterations with decode.
- **Throughput** counts tokens per second of the whole engine; it can rise while
  individual latency gets worse.

State one change you would make for a named target (for example "TTFT p99 below
300 ms at this arrival rate"), its trade-off, and the measurement that would
show whether it worked.

## Optional: change the engine

Every policy is a few lines in `nanolm/serving_sim.py`. Try one change, predict
its effect, and keep the tests green (`uv run pytest tests/test_module_10_serving.py -q`):

- **chunked prefill:** cap the prompt tokens prefilled per iteration (for
  example 32) and carry the rest to the next iteration. How do ITL and TTFT move?
- **shortest-prompt-first admission** instead of first-come first-served. Who wins, and who can starve?
- **swap instead of recompute** on preemption: keep the victim's generated tokens and charge a fixed transfer time.

!!! success "What to take away"
    - **Schedule iterations, not requests.** Continuous batching refills a freed
      slot at the next iteration instead of waiting for the slowest request in the batch.
    - **Prefill and decode compete.** A long prompt raises the latency of every
      request decoding beside it; chunked prefill trades TTFT for smoother ITL.
    - **Paging turns reserved worst cases into memory on demand.** More concurrent
      requests fit, at the price of preemption when the optimism fails.
    - **Latency is a distribution and throughput is a system property.** Report
      tails and state the workload; a mean alone hides who waited.
