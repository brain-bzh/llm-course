# GPU fundamentals (optional)

For readers curious about what happens between a PyTorch operation and the
hardware. You can read this alongside [Module 4](../modules/04-single-gpu.md).
The examples use NVIDIA CUDA terminology and require no GPU.

## How CPUs and GPUs use parallel hardware

Both processors need control logic to fetch and decode instructions, arithmetic
units to execute them, and storage to supply operands and hold results.
A CPU typically has fewer, more complex cores that can make fast progress on
instruction streams with branches and dependencies. It also exploits
data parallelism. For example, a 256-bit **SIMD** vector add can add eight
FP32 pairs in one instruction, with operands held in vector registers.

A GPU devotes more hardware to arithmetic lanes and groups independent
software threads into **warps**. Under NVIDIA's **SIMT** model, a single
instruction can run across 32 threads, each with its own registers and
logical index. A warp of threads can be ready to issue while another waits
for data. This organization helps the GPU sustain high throughput on large,
regular workloads.

## From Python to the GPU

A GPU is a processor designed to sustain many operations in parallel. A
training GPU has many **streaming multiprocessors** (SMs), each with execution
units, schedulers, and small pools of fast on-chip storage. The SMs share
an L2 cache and access large device memory, typically HBM or GDDR.

Suppose `y = x + 1` runs in eager PyTorch and `x` already lives on a CUDA device.
PyTorch dispatches the addition to a GPU **kernel**, a program launched across
many threads. The CPU normally enqueues the launch and can continue while the
GPU executes it. Kernels on the same CUDA stream run in order, so a later kernel
on that stream can consume `y`. Asking for a GPU scalar in Python with
`.item()` can make the CPU wait for the value.

## How the GPU divides the work

Imagine a kernel that assigns one element to each thread. Adding one to 1,000
elements with 128 threads per block would launch a **grid** of eight **thread
blocks**. Each block runs on one SM, which can host multiple blocks when it has
enough resources. Each 128-thread block contains four **warps** of 32 threads.
For this example, threads with indices 1,000 through 1,023 skip the addition and
store. Only eight lanes of the last warp do that work.

Warps allow an SM to issue an instruction to a group of threads operating on
different data. Each participating thread can have its own index, registers,
and result. If threads in the same warp take different paths through a branch
and both paths require instructions, the warp executes those instructions with
the lanes for the other path masked off. This is **warp divergence**.

### What happens while a warp waits?

Once a warp issues a load, its dependent arithmetic has to wait for the data.
A scheduler can issue an instruction from another ready **resident** warp during
that wait. This is **latency hiding**. It keeps execution units busy while the
load completes. Registers and shared memory are limited, so part of kernel
design is deciding how much on-chip storage each thread or block can use while
leaving room for other warps.

## How data reaches the execution units

| Storage | How a kernel uses it |
| --- | --- |
| Registers | Each thread keeps values such as indices and accumulators close to the execution units. |
| Shared memory | Threads in a block can explicitly stage and reuse a tile of data. |
| L1 and L2 caches | The hardware can serve repeated accesses from on-chip copies. L1 is local to an SM and L2 is shared across the GPU. |
| Device memory (HBM or GDDR) | Holds large tensors such as weights, activations, and gradients. |

The hardware behind this hierarchy explains the capacity differences. A
typical **SRAM** bit uses a six-transistor cell that holds its value while
powered. SRAM is fast and takes more silicon area per bit, which suits the
on-chip caches. A typical **DRAM** bit uses a transistor and a capacitor,
which makes it denser and requires periodic refresh. HBM is stacked DRAM
with a wide interface designed for high aggregate bandwidth. An L2 hit saves
a trip through the memory controllers to those stacks.

When a warp loads 32 adjacent FP32 values from a 32-byte-aligned address, those
128 bytes fit in four 32-byte sectors. If lane `i` instead reads element `8 * i`,
each lane requests a different sector. Both patterns request 32 values, but
the strided pattern touches eight times as many sectors. Grouping the lanes'
requests into the sectors they share is **memory coalescing**. Caches and reuse
across warps affect how much of this traffic reaches device memory.

## Inside an NVIDIA H100 SXM

| Level | What the hardware contains |
| --- | --- |
| GPU | 8 GPU Processing Clusters (GPCs), with 66 enabled Texture Processing Clusters (TPCs) in total. Each TPC contains 2 SMs, for 132 enabled SMs. The GPU has 50 MB of on-chip L2 cache and 80 GB of HBM3 across five stacks. |
| One streaming multiprocessor | 4 processing partitions, a 256 KiB pool for L1, texture cache and shared memory, and capacity for up to 64 resident warps. Up to 228 KiB of the pool can be configured as shared memory. |
| One processing partition | A warp scheduler, a portion of the register file, 32 FP32 CUDA cores, and one Tensor Core. Other execution units handle work such as integer arithmetic and loads and stores. |

A CUDA thread is a piece of work with its own state. An FP32 CUDA core is an
arithmetic lane that executes FP32 instructions issued by a warp scheduler.
Many warps can stay resident and take turns issuing instructions to the same
execution units. A Tensor Core executes matrix multiply–accumulate
instructions cooperatively across threads.
[NVIDIA Hopper architecture, Figures 3–4](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#nvidia_h100_gpu_architecture_in-depth) ·
[Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#streaming-multiprocessor)

### How register use limits resident warps

An H100 SM has 65,536 32-bit registers. With 128 threads per block and 64
registers per thread, each block needs 8,192 registers. The register budget
therefore allows at most eight blocks, or 32 warps, on one SM. That gives a
theoretical occupancy ceiling of 50% relative to the hardware limit of 64
warps. If each block also needs 48 KiB of shared memory, even the 228 KiB
maximum pool fits at most four blocks. The ceiling falls to 16 warps, or 25%.
Allocation granularity and launch settings can further constrain residency.
[Hopper tuning guide — occupancy](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy)

## Explore a warp's work

[:material-play-circle-outline: Open the GPU execution explorer](../demos/gpu-execution.html){ .md-button .md-button--primary }

1. Run the same four warps with room for one and then four at a time. Which
   idle slots disappear? Does an individual load return sooner?
2. Keep all 32 lanes active and move the starting address by one FP32 word.
   Why does a contiguous load now touch a fifth sector?
3. Compare the first 16 lanes with alternating lanes using aligned, contiguous
   addresses. Why do the two loads touch different numbers of sectors?
4. In the projection tile, change the number of tokens and output features.
   Which change increases the reuse of one weight?

## How this shows up in a Transformer

An elementwise residual addition reads its inputs, performs little arithmetic
per value, and writes the output. Laying out neighboring elements for
neighboring lanes helps it use bandwidth efficiently.

In a projection `Y = X @ W`, each activation in `X` contributes to several
output features and each weight in `W` is used across multiple token rows.
Take one 16 × 16 output tile and one 16-element slice of the reduction
dimension. The corresponding activation and weight tiles have 256 elements
each. Together they supply 4,096 multiply–accumulates, with each input
element used 16 times. An optimized kernel can stage input tiles in shared
memory for reuse and keep partial output sums in registers while it works
through the next slices. Supported matrix instructions can run on Tensor Cores.

[Module 4](../modules/04-single-gpu.md) builds on this distinction to explain
tiling, fusion, and when a kernel is limited by memory bandwidth or compute.

## Further reading

- [IMT Atlantique — Embedded Software and Hardware for DL (session 7)](https://github.com/brain-bzh/efficient-deep-learning/blob/master/cours7.pdf).
  The SIMD, SRAM/DRAM, CPU/GPU diagrams, and data-reuse slides (5–7, 11–12,
  and 20–21) informed the examples above.
- [NVIDIA CUDA Programming Guide — Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [NVIDIA CUDA Best Practices Guide — Coalesced Access to Global Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)
