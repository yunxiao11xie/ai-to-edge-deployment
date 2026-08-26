# 10_GPU_Kernel与FlashAttention

> **所属路线**：AI 学习路线 · 第三部分  
> **定位**：从 AI Compiler / TensorIR / Schedule 继续向下进入真正的 GPU Kernel 层，理解一个高性能算子为什么快，以及 GEMM、Softmax、Attention 如何映射到 GPU 的 Thread / Warp / Block、Register / Shared Memory / Global Memory 与 Tensor Core  
> **核心仓库**：[`triton-lang/triton`](https://github.com/triton-lang/triton) + [`Dao-AILab/flash-attention`](https://github.com/Dao-AILab/flash-attention) + [`NVIDIA/cutlass`](https://github.com/NVIDIA/cutlass)  
> **核心资料**：NVIDIA CUDA C++ Programming Guide、Triton Tutorials、FlashAttention / FlashAttention-2、CUTLASS GEMM / CuTe 文档  
> **学习边界**：本章重点建立 GPU Kernel Mental Model；不要求立即成为 CUDA 性能专家。后续 Rockchip NPU 章节将把这里的 Memory Hierarchy / Tiling / Tensor Unit / Data Movement 思维迁移到 NPU。  
> **资料检查日期**：2026-08-25

---

# 0. 本章最重要的问题

上一章已经理解：

```text
Model
↓
Relax
↓
TensorIR
↓
Schedule
↓
Lowering
↓
CodeGen
↓
Kernel
```

但还有一个问题：

> **什么样的 Kernel 才是一个“好 Kernel”？**

为什么：

```text
同样计算：
C = A @ B
```

一个实现可能：

```text
1 ms
```

另一个：

```text
20 ms？
```

为什么：

```text
FlashAttention
```

没有减少 Attention 的数学结果精度，却可以：

```text
显著减少显存访问
降低中间结果内存
提高性能？
```

为什么：

```text
Triton
```

只写几十行 Python-like Kernel，就可以接近高度优化 CUDA 实现？

为什么：

```text
CUTLASS
```

要把 GEMM 拆成：

```text
CTA Tile
↓
Warp Tile
↓
Instruction Tile
```

并专门设计：

```text
Mainloop
Epilogue
Data Movement
Tensor Core MMA？
```

本章就是回答这些问题。

---

# 1. 本章最终主线

```text
Tensor Operation
      ↓
GPU Kernel
      ↓
Grid
      ↓
Thread Block / CTA
      ↓
Warp
      ↓
Thread
      ↓
Instruction
```

同时另一条线：

```text
DRAM / HBM
    ↓
L2 Cache
    ↓
Shared Memory
    ↓
Register
    ↓
ALU / Tensor Core
```

两条线合起来：

```text
Work Partition
+
Data Movement
+
Computation
=
GPU Kernel Performance
```

---

# 2. 本章学习目标

学完以后，应能解释：

1. GPU Kernel 是什么？
2. CPU Function 和 GPU Kernel 有什么不同？
3. Grid / Block / Thread 是什么？
4. CUDA 中 CTA 与 Thread Block 是什么关系？
5. Warp 是什么？
6. NVIDIA GPU 为什么通常以 32 Thread 为一个 Warp 执行？
7. SIMT 是什么？
8. SIMT 与 SIMD 有什么区别？
9. Warp Divergence 是什么？
10. 为什么 Branch Divergence 可能降低性能？
11. SM / Streaming Multiprocessor 是什么？
12. Register、Shared Memory、L2、Global Memory 分别是什么？
13. 为什么 GPU 性能优化大量工作其实是在优化 Memory？
14. Global Memory Coalescing 是什么？
15. Shared Memory Bank Conflict 是什么？
16. Register Spill 是什么？
17. Occupancy 是什么？
18. 为什么 Occupancy 不是越高越好？
19. Latency Hiding 是什么？
20. 为什么 GPU 需要很多 Warp？
21. Memory-Bound 与 Compute-Bound 如何在 Kernel 层判断？
22. Arithmetic Intensity 如何应用到 GPU Kernel？
23. Roofline 与 Kernel Optimization 的关系是什么？
24. GEMM 是什么？
25. GEMV 与 GEMM 为什么性能特征差别很大？
26. Naive GEMM 为什么慢？
27. GEMM Tiling 是什么？
28. CTA Tile / Warp Tile / Instruction Tile 是什么？
29. Shared Memory Tiling 为什么快？
30. Register Blocking 是什么？
31. Double Buffering / Multi-stage Pipeline 是什么？
32. Async Copy 是什么？
33. Tensor Core 是什么？
34. MMA 指令是什么？
35. FP16 / BF16 / TF32 / FP8 / INT8 对 Tensor Core 有什么意义？
36. 为什么 Tensor Core 不等于“自动使用就快”？
37. Epilogue 是什么？
38. 为什么 Bias / Activation / Quantization 常适合 Fuse 到 GEMM Epilogue？
39. Triton 是什么？
40. Triton 和 CUDA C++ 的区别是什么？
41. Triton 的 Program Instance 是什么？
42. `program_id` 是什么？
43. `tl.arange`、`tl.load`、`tl.store` 是什么？
44. Triton 为什么采用 Tile-based Programming Model？
45. Triton `@triton.autotune` 在调什么？
46. Triton Softmax 为什么是学习 Kernel Fusion 的好例子？
47. 为什么一个 Fused Softmax 可以显著减少 DRAM Traffic？
48. Triton MatMul 如何表达 Tiling？
49. Triton `tl.dot` 与 Tensor Core 有什么关系？
50. FlashAttention 为什么出现？
51. 标准 Attention 的主要 Memory IO 问题是什么？
52. 为什么保存完整 `N × N` Attention Matrix 很昂贵？
53. FlashAttention 是否近似 Attention？
54. FlashAttention 的 IO-aware 到底是什么意思？
55. Online Softmax 是什么？
56. 为什么 Softmax 可以分块在线计算？
57. FlashAttention 如何避免保存完整 Score Matrix？
58. FlashAttention 为什么需要对 Q/K/V 分块？
59. FlashAttention 如何在 SRAM / Shared Memory 中完成局部计算？
60. FlashAttention 的 Forward 主循环是什么？
61. FlashAttention 为什么能减少 HBM / DRAM IO？
62. FlashAttention-2 主要在优化什么？
63. FlashAttention-3 为什么更关注 Hopper 的异步能力与新硬件特性？
64. FlashAttention 与 PagedAttention 有什么区别？
65. FlashAttention 与 KV Cache 有什么关系？
66. Prefill 和 Decode 是否使用相同 Attention Kernel？
67. FlashAttention 为什么对 Prefill 尤其重要？
68. Decode Attention 的主要瓶颈是什么？
69. CUTLASS 是什么？
70. CUTLASS 和 cuBLAS 有什么区别？
71. CUTLASS 的 GEMM hierarchy 是什么？
72. CUTLASS Mainloop 是什么？
73. CUTLASS Epilogue 是什么？
74. CuTe 是什么？
75. CuTe Layout Algebra 为什么重要？
76. CUTLASS 与 Triton 的学习价值有什么区别？
77. Triton、CUTLASS、TVM 三者是什么关系？
78. Nsight Systems 和 Nsight Compute 分别看什么？
79. Kernel Profiling 应重点看哪些指标？
80. 为什么最终应该形成“计算 + 数据搬运 + 并行映射”的统一分析方法？

---

# 3. 核心资料

---

## 3.1 NVIDIA CUDA Programming Guide

官方：

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

这一部分重点学习：

```text
Thread
Warp
Block
Grid
SM
Register
Shared Memory
Global Memory
Occupancy
Coalescing
Bank Conflict
Synchronization
```

不要一开始：

```text
从 CUDA API 大全学起。
```

---

# 4. Triton

GitHub：

- [triton-lang/triton](https://github.com/triton-lang/triton)

官方定位：

> 一个用于编写高效 Deep Learning Primitive 的语言与 Compiler。

推荐源码 / 教程：

- [Triton README](https://github.com/triton-lang/triton/blob/main/README.md)
- [Vector Add Tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py)
- [Fused Softmax Tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py)
- [Matrix Multiplication Tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)
- [Layer Norm Tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py)
- [Fused Attention Tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)

学习顺序：

```text
Vector Add
↓
Softmax
↓
MatMul
↓
LayerNorm
↓
Attention
```

---

# 5. FlashAttention

GitHub：

- [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

核心论文：

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf)

核心思想：

```text
不是：
Approximate Attention

而是：
Exact Attention
+
IO-aware Algorithm
```

---

# 6. CUTLASS

GitHub：

- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

推荐：

- [CUTLASS README](https://github.com/NVIDIA/cutlass/blob/main/README.md)
- [CUTLASS GEMM API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api.md)
- [CUTLASS 3.x Design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cutlass_3x_design.md)
- [CuTe GEMM Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0x_gemm_tutorial.md)
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass/tree/main/examples)

---

# 7. 第一部分：GPU 执行模型

CPU 常见：

```text
几个到几十个高性能 Core
```

GPU：

```text
大量较轻量计算单元
```

GPU 设计目标：

```text
Massive Parallelism
+
High Throughput
```

---

# 8. CPU vs GPU

CPU：

```text
复杂控制
强单线程
大 Cache
低延迟
```

GPU：

```text
大量线程
高吞吐
高 Memory Bandwidth
通过并发隐藏延迟
```

---

# 9. GPU 最重要的思维转变

CPU 优化经常问：

```text
单个线程多久完成？
```

GPU 更常问：

```text
整个设备有多少并行 Work？
数据怎么分块？
多少 Warp 可以同时工作？
Memory Pipeline 是否被填满？
```

---

# 10. GPU Kernel

CUDA 中：

```cpp
__global__ void kernel(...) {
    ...
}
```

这是：

```text
Device Kernel。
```

由 Host：

```text
launch。
```

---

# 11. Kernel Launch

概念：

```text
CPU
↓
Launch Kernel
↓
GPU Grid
↓
Blocks
↓
Threads
```

---

# 12. Grid

一次 Kernel Launch：

```text
生成一个 Grid。
```

Grid：

```text
由很多 Thread Block 组成。
```

---

# 13. Thread Block / CTA

```text
Thread Block
```

也常叫：

```text
CTA
=
Cooperative Thread Array。
```

同一个 Block 内 Thread：

```text
可以共享 Shared Memory
可以进行 Block-level Synchronization。
```

---

# 14. Thread

Thread：

```text
GPU 最小编程级执行单元之一。
```

每个 Thread 有：

```text
threadIdx。
```

---

# 15. Block ID

每个 Block：

```text
blockIdx。
```

Grid / Block Dimension：

```text
gridDim
blockDim。
```

---

# 16. 一个 Vector Add

```text
C[i] = A[i] + B[i]
```

可以：

```text
1 Thread
→
1 Element。
```

---

# 17. Thread Index

典型：

```cpp
int i =
    blockIdx.x * blockDim.x
    +
    threadIdx.x;
```

---

# 18. 为什么需要这么多 Thread？

GPU：

```text
一个 Thread 等 Memory
```

可以：

```text
切换执行另一个 Ready Warp。
```

---

# 19. Warp

NVIDIA GPU：

```text
32 Threads
```

组成：

```text
1 Warp。
```

硬件执行通常以：

```text
Warp
```

为调度粒度。

---

# 20. Warp 不是 CUDA Source 中显式创建的

你创建：

```text
Thread Block。
```

硬件自动：

```text
把 Thread 分成 Warp。
```

---

# 21. 为什么 Block Size 常取 Warp Size 的整数倍？

例如：

```text
128
256
512。
```

避免：

```text
最后一个 Warp
只有少数 Thread 有效。
```

---

# 22. SIMT

GPU 常说：

```text
SIMT
=
Single Instruction
Multiple Threads。
```

---

# 23. SIMT 和 SIMD

SIMD：

```text
一个指令
操作一个 Vector。
```

SIMT：

```text
程序看起来是多个 Thread
但硬件 Warp 同步执行指令。
```

---

# 24. Warp Divergence

```cpp
if (threadIdx.x % 2 == 0) {
    A();
} else {
    B();
}
```

同一 Warp：

```text
部分 Thread 走 A
部分走 B。
```

---

# 25. Divergence 发生什么？

硬件可能：

```text
执行 A 路径
屏蔽 B Thread

再执行 B 路径
屏蔽 A Thread。
```

于是：

```text
有效并行度下降。
```

---

# 26. 所有 Branch 都很坏吗？

不是。

如果：

```text
Warp 内所有 Thread
走同一个 Branch。
```

没有明显 Divergence。

---

# 27. SM

NVIDIA：

```text
Streaming Multiprocessor
```

可以理解：

> GPU 上真正承载多个 Block / Warp 的计算单元。

---

# 28. 一个 GPU 有多个 SM

每个 SM 拥有：

```text
Warp Scheduler
Registers
Shared Memory
ALU
Tensor Core
Load/Store Unit
...
```

---

# 29. Block 被调度到 SM

一个 Thread Block：

```text
整个生命周期
通常驻留在一个 SM。
```

---

# 30. 同一 SM 可以同时驻留多个 Block

前提：

```text
资源足够。
```

---

# 31. 资源包括

```text
Registers
Shared Memory
Thread Slots
Block Slots
Warp Slots。
```

---

# 32. Occupancy

Occupancy：

```text
Active Warps
/
Maximum Supported Warps
```

可以粗略表示：

```text
SM 被多少 Warp 填充。
```

---

# 33. 为什么需要高 Occupancy？

一个 Warp：

```text
等待 Global Memory。
```

Scheduler：

```text
切换到另一个 Ready Warp。
```

这叫：

```text
Latency Hiding。
```

---

# 34. Occupancy 高一定快吗？

不一定。

非常重要：

```text
High Occupancy
≠
High Performance。
```

---

# 35. 为什么？

一个 Kernel 可能：

```text
50% Occupancy
```

已经：

```text
把 Tensor Core / Memory Bandwidth
吃满。
```

再提高：

```text
没有收益。
```

---

# 36. 甚至可能反过来

为了提高 Occupancy：

```text
减少 Register
缩小 Tile。
```

导致：

```text
Data Reuse ↓
Instruction ↑
Memory Traffic ↑。
```

反而慢。

---

# 37. 所以 Occupancy 是约束 / 指标

不是：

```text
最终优化目标。
```

最终目标：

```text
Latency
Throughput
Useful FLOPS
Bandwidth Utilization。
```

---

# 38. 第二部分：GPU Memory Hierarchy

性能优化核心之一：

```text
Memory Hierarchy。
```

---

# 39. 简化层级

```text
Slow / Large

HBM / GDDR / Global Memory
           ↓
          L2
           ↓
    Shared Memory / L1
           ↓
        Register

Fast / Small
```

不同架构细节会变化。

---

# 40. Global Memory

也就是：

```text
GPU Device DRAM。
```

特点：

```text
容量最大
带宽高
但延迟也高。
```

---

# 41. Register

每个 Thread：

```text
拥有 Register。
```

特点：

```text
最快
容量最有限。
```

---

# 42. Shared Memory

Block 内：

```text
Thread 共享。
```

On-chip：

```text
低延迟
高带宽。
```

---

# 43. Shared Memory 最大价值

```text
Global Memory
↓
Load once
↓
Shared Memory
↓
Reuse many times。
```

---

# 44. 这就是 Tiling

上一章的：

```text
cache_read("shared")
```

最终硬件意义：

```text
把 Tile 搬到 Shared Memory。
```

---

# 45. L2 Cache

整个 GPU：

```text
共享较大的 Cache。
```

很多：

```text
Global Memory Request
```

会经过 L2。

---

# 46. GPU 优化的经典目标

```text
尽可能少访问 Global Memory

尽可能多复用：
Shared / Register。
```

---

# 47. Memory Coalescing

一个 Warp：

```text
32 Threads
```

同时访问：

```text
Global Memory。
```

如果地址：

```text
连续 / 合理对齐
```

硬件可以：

```text
合并 Memory Transaction。
```

---

# 48. Coalesced Access

概念：

```text
Thread 0 → A[0]
Thread 1 → A[1]
...
Thread 31 → A[31]
```

通常非常友好。

---

# 49. Uncoalesced

例如：

```text
Thread 0 → A[0]
Thread 1 → A[1024]
Thread 2 → A[2048]
...
```

可能需要：

```text
很多 Memory Transactions。
```

---

# 50. 为什么 Layout 如此重要？

同一个数学 Tensor：

```text
数据排列
```

决定：

```text
Thread 是否能 Coalesced Load。
```

所以：

```text
Layout
=
性能变量。
```

---

# 51. Shared Memory Bank

Shared Memory：

```text
内部划分为多个 Bank。
```

多个 Bank：

```text
可以并行访问。
```

---

# 52. Bank Conflict

如果一个 Warp：

```text
多个 Thread
访问同一个 Bank 的不同地址。
```

可能：

```text
访问被序列化。
```

---

# 53. 因此 Shared Memory 也不是随便用

要考虑：

```text
Layout
Padding
Transpose
Bank Mapping。
```

---

# 54. Register Spill

如果每个 Thread：

```text
需要太多 Register。
```

Compiler 可能：

```text
把部分变量 spill 到 local memory。
```

---

# 55. Local Memory 名字很容易误导

CUDA 中：

```text
local memory
```

并不意味着：

```text
在芯片旁边很快。
```

很多时候：

```text
实际位于 Device Memory
并通过 Cache。
```

所以 Spill：

```text
可能很贵。
```

---

# 56. Register Pressure

一个 Kernel：

```text
Tile 越大
Accumulator 越多
```

可能：

```text
Register / Thread ↑。
```

---

# 57. Register 多的代价

SM Register 总量有限。

每 Thread 多：

```text
同一时间能驻留的 Warp ↓。
```

于是：

```text
Occupancy ↓。
```

---

# 58. Shared Memory 也是类似

Block：

```text
Shared Memory 用太多。
```

同一 SM：

```text
能驻留 Block 数下降。
```

---

# 59. Kernel Optimization 永远是 Tradeoff

```text
Tile Bigger
→ Reuse ↑
→ Register / Shared ↑
→ Occupancy ↓。
```

不存在：

```text
永远最大的 Tile 最快。
```

---

# 60. 第三部分：Compute-bound 与 Memory-bound

上一章已经讲：

```text
Arithmetic Intensity
=
Operations / Bytes Moved。
```

现在把它落到 Kernel。

---

# 61. Vector Add

```text
C = A + B
```

每个元素：

```text
Read A
Read B
1 Add
Write C。
```

计算：

```text
极少。
```

Memory：

```text
很多。
```

典型：

```text
Memory-Bound。
```

---

# 62. GEMM

```text
C = A @ B
```

同一个：

```text
A Tile
B Tile
```

可以：

```text
重复做大量 Multiply-Accumulate。
```

如果 Tiling 好：

```text
Arithmetic Intensity 很高。
```

更可能：

```text
Compute-Bound。
```

---

# 63. Softmax

包括：

```text
Max Reduction
Subtract
Exp
Sum Reduction
Divide。
```

FLOPs：

```text
不算特别巨大。
```

但 naïve 实现会：

```text
反复读写 DRAM。
```

所以常：

```text
Bandwidth-sensitive。
```

---

# 64. 为什么 Fusion 对 Softmax 很有效？

原来多个 Framework Op：

```text
max
sub
exp
sum
div
```

每个可能：

```text
独立 Kernel
+
DRAM round trip。
```

---

# 65. Fused Softmax

一次：

```text
Load Row
↓
Keep in SRAM / Register
↓
Max
↓
Exp
↓
Sum
↓
Normalize
↓
Store。
```

显著减少：

```text
DRAM IO。
```

---

# 66. 这正是 Triton 官方 Softmax 教程的核心

不是：

```text
发明新的 Softmax 数学。
```

而是：

```text
Kernel Fusion
+
Memory Traffic Reduction。
```

---

# 67. 第四部分：GEMM

GEMM：

```text
General Matrix-Matrix Multiplication。
```

经典：

```text
D =
alpha × A × B
+
beta × C。
```

---

# 68. 为什么 GEMM 是 AI Kernel 的核心？

Transformer：

```text
Q/K/V Projection
Output Projection
FFN Up
FFN Gate
FFN Down
LM Head
```

大量都是：

```text
GEMM / GEMV。
```

---

# 69. LLM 大部分参数在哪里？

```text
Linear Weight。
```

而 Linear：

```text
本质是矩阵乘。
```

---

# 70. GEMM vs GEMV

Prefill：

```text
M 较大
```

往往：

```text
GEMM。
```

Decode：

```text
M ≈ Batch
```

单用户时：

```text
M ≈ 1。
```

更接近：

```text
GEMV / Skinny GEMM。
```

---

# 71. 为什么 GEMV 难吃满 GPU？

矩阵：

```text
只有一个 / 少量 Row。
```

并行 Work：

```text
比大型 GEMM 少很多。
```

同时 Weight：

```text
仍要大量读取。
```

因此容易：

```text
Memory-Bound。
```

---

# 72. Naive GEMM

```python
for i:
    for j:
        for k:
            C[i,j] += A[i,k] * B[k,j]
```

如果每次：

```text
A / B
都从 Global Memory 读。
```

效率极低。

---

# 73. GEMM 的第一层 Tiling

把输出 C：

```text
分成 CTA Tile。
```

例如：

```text
128 × 128。
```

每个 Block：

```text
负责一个 C Tile。
```

---

# 74. CTA Tile

```text
CTA
=
Thread Block。
```

所以：

```text
CTA Tile
=
一个 Thread Block 负责的输出区域。
```

---

# 75. K Dimension Tiling

MatMul：

```text
K
```

也分块。

例如：

```text
K tile = 32。
```

---

# 76. Mainloop

```text
for each K Tile:
    load A tile
    load B tile
    compute partial C
```

这就是 GEMM：

```text
Mainloop。
```

---

# 77. Shared Memory Tiling

每一轮：

```text
A global tile
→ shared

B global tile
→ shared。
```

---

# 78. 然后：

```text
多个 Thread / Warp
重复使用 Shared Tile。
```

---

# 79. 为什么快？

一次 Global Load：

```text
参与大量 MAC。
```

---

# 80. Warp Tile

CTA Tile：

```text
再分给多个 Warp。
```

例如：

```text
CTA 128×128

4 Warp
每个负责部分 Tile。
```

---

# 81. Thread / MMA Tile

Warp：

```text
再拆成更小计算片。
```

最终：

```text
映射到 Tensor Core MMA instruction。
```

---

# 82. GEMM Hierarchy

建议直接记：

```text
GEMM
↓
CTA Tile
↓
Warp Tile
↓
Instruction Tile
↓
MMA。
```

---

# 83. 这和 GPU Hardware Hierarchy 完美对应

```text
Grid
↓
Block
↓
Warp
↓
Tensor Core Instruction。
```

---

# 84. CUTLASS 为什么采用 Hierarchical Decomposition？

因为：

```text
硬件本来就是层级化的。
```

高性能 GEMM：

```text
必须把 Work 和 Data
映射到各级硬件。
```

---

# 85. Accumulator

MatMul：

```text
C Tile
```

通常部分结果：

```text
保存在 Register。
```

---

# 86. 为什么不每次写 Shared / Global？

因为：

```text
Reduction K
```

需要反复累加。

Register：

```text
最快。
```

---

# 87. Mainloop 最终做什么？

不断：

```text
Load A/B Tile
↓
MMA
↓
Accumulate Register C。
```

---

# 88. Epilogue

Mainloop 结束后：

```text
Accumulator
```

需要：

```text
写 Output。
```

---

# 89. 为什么 Epilogue 单独设计？

写回前可能还要：

```text
alpha / beta
Bias
Activation
Scale
Quantize
Clamp
Aux Output。
```

---

# 90. Fusion 最适合发生在哪里？

例如：

```text
GEMM
↓
Bias
↓
SiLU。
```

如果 Bias / SiLU：

```text
Fusion 到 Epilogue
```

就可以：

```text
避免 GEMM Output
先写 DRAM
再读回来。
```

---

# 91. LLM 中非常常见

例如：

```text
Linear
+
Bias

GEMM
+
Activation

GEMM
+
Quantize。
```

---

# 92. Tensor Core

现代 NVIDIA GPU 中：

```text
专门执行矩阵 Multiply-Accumulate
```

的硬件单元。

---

# 93. Tensor Core 输入不是任意 Shape

通常对应：

```text
特定 Tile / dtype / layout。
```

Kernel 必须：

```text
把数据组织成硬件需要的 Fragment。
```

---

# 94. MMA

```text
Matrix Multiply-Accumulate。
```

概念：

```text
D = A × B + C。
```

硬件一次：

```text
处理一小块矩阵。
```

---

# 95. Tensor Core 性能为什么很高？

因为：

```text
硬件专门为 Matrix MAC
做了大量并行数据通路。
```

---

# 96. 但 Tensor Core 不是免费性能

如果：

```text
数据搬不进来
Tile 不匹配
Layout 错
Occupancy 太低
```

Tensor Core：

```text
可能吃不满。
```

---

# 97. Kernel 的目标

不是：

```text
“用了 Tensor Core”
```

而是：

> **持续给 Tensor Core 喂足数据。**

---

# 98. Double Buffering

当：

```text
正在计算 Tile 0
```

同时：

```text
预取 Tile 1。
```

---

# 99. 为什么？

隐藏：

```text
Memory Load Latency。
```

---

# 100. Pipeline

```text
Load Tile N+1
```

与：

```text
Compute Tile N
```

重叠。

---

# 101. Multi-stage Pipeline

进一步：

```text
多个 Shared Memory Stage。
```

---

# 102. Async Copy

现代 GPU 可以：

```text
异步搬 Global → Shared
```

减少：

```text
Thread 手工同步 / 等待。
```

---

# 103. Hopper / Blackwell 的数据搬运能力更进一步

例如：

```text
TMA
WGMMA
Warp Specialization
```

等会让：

```text
Producer Warp
Consumer Warp
```

角色更加明显。

---

# 104. 但学习顺序不要反

先理解：

```text
Basic GEMM Tiling
Shared Memory
Register
```

再进入：

```text
cp.async
TMA
WGMMA。
```

---

# 105. 第五部分：Triton

Triton 官方定义：

> 用于编写高效 Deep Learning Primitive 的语言和 Compiler。

---

# 106. 为什么 Triton 出现？

CUDA C++：

```text
控制力非常强。
```

但高性能 Kernel：

```text
开发成本高。
```

很多 DL Kernel 其实是：

```text
Tile-based Tensor Program。
```

Triton 希望用户描述：

```text
一个 Tile 怎么算。
```

Compiler 负责：

```text
Thread Layout
Register
Shared Memory
Instruction Selection
部分 Data Movement。
```

---

# 107. Triton 不是“Python 在 GPU 上直接执行”

虽然源码：

```python
@triton.jit
def kernel(...):
    ...
```

看起来像 Python。

实际：

```text
Triton DSL
↓
Compiler IR
↓
GPU Backend
↓
PTX / AMD Code
↓
GPU。
```

---

# 108. Triton Program Instance

CUDA：

```text
你显式思考 Thread。
```

Triton：

```text
更常以一个 Program
处理一个 Tile。
```

---

# 109. `program_id`

```python
pid = tl.program_id(axis=0)
```

类似于：

```text
确定当前 Program
负责哪块数据。
```

---

# 110. Triton 与 CUDA Block

可以粗略类比：

```text
Triton Program Instance
≈
一个 Block / CTA 级计算任务。
```

但不要做严格一一对应。

---

# 111. `tl.arange`

生成：

```text
Tile 内的向量化 Index。
```

例如：

```python
offsets =
    pid * BLOCK_SIZE
    +
    tl.arange(0, BLOCK_SIZE)
```

---

# 112. `tl.load`

从：

```text
Global Memory
```

加载一个：

```text
Tensor Block。
```

---

# 113. Mask

边界：

```python
mask = offsets < n
```

避免：

```text
越界访问。
```

---

# 114. `tl.store`

把 Block 结果：

```text
写回。
```

---

# 115. Vector Add

Triton 思想：

```text
一个 Program
处理 BLOCK_SIZE 个 Element。
```

而不是：

```text
你逐个管理每个 CUDA Thread。
```

---

# 116. Triton 的真正优势

抽象层正好位于：

```text
PyTorch Operator
```

和：

```text
CUDA Thread-level Code
```

之间。

---

# 117. 这和 TVM TensorIR 很像吗？

有相似点：

```text
都属于 Tensor / Kernel Program 层。
```

---

# 118. 但不同

TVM：

```text
更完整 ML Compiler
Graph → TensorIR → CodeGen。
```

Triton：

```text
专注 GPU Kernel Language / Compiler。
```

---

# 119. Triton Kernel 与 PyTorch

典型：

```python
y = custom_triton_kernel(x)
```

上层仍然：

```text
PyTorch。
```

---

# 120. Triton Fused Softmax

这是最适合学习 Fusion 的例子。

朴素 PyTorch：

```text
max
↓
sub
↓
exp
↓
sum
↓
divide。
```

---

# 121. 每个 Op 都可能产生中间 Tensor

```text
X
↓
Max Output

Z
↓
Exp Output

Sum
↓
Output。
```

---

# 122. DRAM Traffic 很大

Triton 教程会直接统计：

```text
naive softmax
```

需要多次：

```text
读写 M×N 元素。
```

---

# 123. Fused Kernel

```text
Read row once
↓
Compute max
↓
Compute exp
↓
reduce
↓
divide
↓
write once。
```

---

# 124. 关键不是 Python vs C++

而是：

```text
Fusion
+
On-chip Reuse。
```

---

# 125. Triton MatMul

核心参数通常：

```text
BLOCK_SIZE_M
BLOCK_SIZE_N
BLOCK_SIZE_K
num_warps
num_stages。
```

---

# 126. 这些是什么？

```text
BLOCK_M/N/K
=
Tile Size

num_warps
=
一个 Program 使用多少 Warp

num_stages
=
Software Pipeline / Buffer Stage。
```

---

# 127. `tl.dot`

表达：

```text
Block Matrix Dot Product。
```

Compiler：

```text
根据 dtype / architecture
映射到合适 Matrix Instruction。
```

---

# 128. Triton AutoTune

```python
@triton.autotune(
    configs=[...],
    key=[...]
)
```

---

# 129. AutoTune 在调什么？

例如：

```text
BLOCK_M
BLOCK_N
BLOCK_K
num_warps
num_stages。
```

---

# 130. 为什么 Shape 是 key？

不同：

```text
M/N/K
```

最佳 Tile：

```text
可能不同。
```

---

# 131. Kernel Specialization

例如：

```text
M=1
```

和：

```text
M=4096
```

完全是不同 Workload。

---

# 132. LLM 中非常重要

```text
Prefill GEMM
```

与：

```text
Decode GEMV-like
```

不应该盲目使用：

```text
同一个 Kernel Config。
```

---

# 133. Triton 编译栈

当前实现基于：

```text
MLIR-based compiler infrastructure。
```

可以粗略：

```text
Triton Python DSL
↓
Triton IR
↓
TritonGPU / Backend IR
↓
LLVM-related lowering
↓
PTX / AMD ISA path。
```

具体名称会随版本演进。

---

# 134. 为什么看 IR 有价值？

因为：

```text
Python Kernel
```

只是高层。

最终性能来自：

```text
Compiler
如何分配 Layout / Memory / Instruction。
```

---

# 135. Triton 当前还有更低层探索

仓库中现在还可以看到：

```text
Gluon
```

这种更低层 GPU programming language 教程。

但本章：

```text
不需要深入。
```

只需要知道：

> Triton 本身仍在继续探索“生产力 vs 硬件控制力”的边界。

---

# 136. Triton 源码学习顺序

```text
README
↓
01-vector-add
↓
02-fused-softmax
↓
03-matrix-multiplication
↓
05-layer-norm
↓
06-fused-attention
↓
triton.language
↓
compiler.py
↓
backend
↓
MLIR passes。
```

---

# 137. 不要从 Compiler C++ 开始

先：

```text
会写 3 个 Kernel。
```

再追：

```text
它怎么 Lower。
```

---

# 138. 第六部分：为什么 FlashAttention 出现？

标准 Self-Attention：

```text
Q
K
V
```

计算：

```text
S = QKᵀ
P = softmax(S)
O = PV。
```

---

# 139. Shape

假设：

```text
Q:
[N, d]

K:
[N, d]

V:
[N, d]
```

则：

```text
S:
[N, N]。
```

---

# 140. 问题在哪里？

当：

```text
N = 8192
```

Score Matrix：

```text
8192 × 8192
```

非常大。

---

# 141. 标准 Materialized Attention

概念：

```text
Q,K
↓
GEMM
↓
S = QKᵀ
↓
write S to HBM
↓
read S
↓
softmax
↓
write P
↓
read P,V
↓
GEMM
↓
O。
```

---

# 142. 发生了什么？

大量：

```text
N×N
```

中间数据：

```text
HBM 写出
再读回。
```

---

# 143. 算力不是唯一问题

Attention：

```text
FLOPs 很多。
```

但：

```text
Memory IO
```

同样非常关键。

---

# 144. FlashAttention 的核心问题

不是问：

```text
如何减少 Attention FLOPs？
```

而是：

> **如何减少 HBM ↔ SRAM 的 Data Movement？**

---

# 145. 所以叫：

```text
IO-aware Attention。
```

---

# 146. FlashAttention 是不是近似 Attention？

不是。

核心目标：

```text
计算与标准 Attention
数学等价的 exact result
```

允许：

```text
浮点数数值误差范围内差异。
```

---

# 147. 最重要的技巧

```text
Tiling
+
Online Softmax
+
Recomputation / Fusion。
```

---

# 148. 第一件事：不保存完整 S

而是：

```text
Q Tile
×
K Tile
↓
局部 Score Tile。
```

---

# 149. Score Tile 在哪里？

尽量：

```text
On-chip SRAM / Register。
```

而不是：

```text
写完整 N×N 到 HBM。
```

---

# 150. 然后局部 Softmax

问题来了：

```text
Softmax
```

需要整行：

```text
max
sum。
```

怎么只看一块？

---

# 151. Online Softmax

关键：

> 可以逐块维护当前已经看到数据的最大值与指数和。

---

# 152. 普通 Softmax

一行：

```text
x_1 ... x_N
```

为了数值稳定：

```text
m = max(x)

l = Σ exp(x_i - m)

p_i =
exp(x_i - m) / l。
```

---

# 153. 如果分成两个 Block

Block A：

```text
m_A
l_A。
```

Block B：

```text
m_B
l_B。
```

---

# 154. 合并最大值

```text
m =
max(m_A, m_B)。
```

---

# 155. 旧 Block 的 Sum 需要重新缩放

因为之前：

```text
l_A
```

是以：

```text
m_A
```

为基准。

现在基准变：

```text
m。
```

所以：

```text
l_A'
=
l_A × exp(m_A - m)。
```

---

# 156. 同理：

```text
l_B'
=
l_B × exp(m_B - m)。
```

于是：

```text
l
=
l_A'
+
l_B'。
```

---

# 157. 这就是 Online Softmax 的关键

只需要维护：

```text
running max
running sum。
```

不需要一次保存整行 Score。

---

# 158. 更进一步

Attention 最终需要：

```text
O =
softmax(S) V。
```

可以在每个 K/V Tile 处理时：

```text
同时累计 Output。
```

---

# 159. 维护什么状态？

对每个 Q Row：

```text
m_i
=
running max

l_i
=
running softmax denominator

O_i
=
running output accumulator。
```

---

# 160. 当新 Score Tile 到来

更新：

```text
new max
new denominator
new output accumulator。
```

---

# 161. 因此：

```text
不需要保存完整 Attention Matrix。
```

---

# 162. FlashAttention Forward 高层流程

```text
for each Q block:

    load Q block

    initialize:
        m
        l
        O

    for each K/V block:

        load K block
        load V block

        S = Q_block @ K_block.T

        apply mask

        update online softmax stats

        update O using V_block

    normalize O

    store O
```

---

# 163. 这就是 FlashAttention 最核心的算法骨架

建议：

```text
自己手画一次。
```

---

# 164. 为什么 FlashAttention Memory 更省？

传统：

```text
需要 Materialize
N×N Score / Probability。
```

Flash：

```text
只保留 Tile + running statistics。
```

---

# 165. Memory Complexity

中间 Attention Matrix：

```text
O(N²)
```

不再需要完整保存。

实际工作内存：

```text
大幅降低。
```

---

# 166. 计算 Complexity 变了吗？

核心 Attention：

```text
仍然 O(N² d)
```

并没有：

```text
变成 O(N)。
```

---

# 167. 为什么还能更快？

因为：

```text
GPU 不只是在算。
```

还大量：

```text
搬数据。
```

FlashAttention：

```text
显著减少昂贵 HBM Traffic。
```

---

# 168. 这是“IO complexity”思维

算法性能：

```text
不仅看 FLOPs。
```

还看：

```text
Memory Access Complexity。
```

---

# 169. 这和前面所有章节串起来

06：

```text
Memory-Bound。
```

10：

```text
Tiling / Data Reuse。
```

11：

```text
FlashAttention
把 IO-aware 做到 Attention 算法级。
```

---

# 170. FlashAttention 与 Kernel Fusion

标准：

```text
QK GEMM
↓
Softmax
↓
PV GEMM
```

Flash：

```text
把它们更紧密地融合在一个 Tile Pipeline 中。
```

---

# 171. 为什么两个 GEMM 中间有 Softmax 很难？

因为：

```text
GEMM → Reduction → GEMM。
```

不是简单 Elementwise Fusion。

---

# 172. FlashAttention 的价值就在这里

它重新组织算法：

```text
让这个跨 GEMM 的 Fusion
在 Tile 层可行。
```

---

# 173. Causal Mask

Decoder：

```text
Token i
只能看 <= i。
```

所以 Score Tile：

```text
部分区域需要 Mask。
```

---

# 174. FlashAttention 会跳过无效 Tile 吗？

Causal 情况：

```text
可根据 Tile Position
避免计算完全位于未来区域的 Tile。
```

边界 Tile：

```text
应用 Mask。
```

---

# 175. Variable Length

真实 Batch：

```text
每个 Sequence 长度不同。
```

高性能实现还要处理：

```text
varlen
padding
packed sequence。
```

---

# 176. MQA / GQA

Attention Kernel：

```text
Q Head 数
```

和：

```text
KV Head 数
```

可能不同。

Kernel 需要：

```text
正确复用 K/V。
```

---

# 177. FlashAttention-2

FA2 的核心目标：

> 在 FlashAttention 的 IO-aware 基础上，进一步改善并行度和 Work Partitioning。

---

# 178. 为什么还需要 FA2？

FA1：

```text
已经大幅减少 IO。
```

但 GPU：

```text
仍可能存在 Work Partition 不均
non-matmul FLOPs 较多
Warp 协作不够理想。
```

---

# 179. FA2 大方向

```text
更好的 Sequence Parallelism
更好的 Warp Work Partition
减少非 MatMul FLOPs
提高 Tensor Core 利用。
```

本章掌握：

```text
设计目标即可。
```

---

# 180. FlashAttention-3

当前官方仓库还提供：

```text
FlashAttention-3
```

面向：

```text
Hopper。
```

重点利用：

```text
更先进异步执行
Warp Specialization
FP8
新矩阵与数据搬运能力。
```

---

# 181. 为什么 FA3 强调 Hardware-aware？

算法：

```text
已经不只写数学。
```

而是：

> 根据某一代 GPU 的 Pipeline / Instruction / Memory 系统重新安排 Work。

---

# 182. 这就是系统级优化

```text
Algorithm
+
Kernel
+
Hardware
```

协同设计。

---

# 183. FlashAttention 与 PagedAttention

非常容易混淆。

---

# 184. FlashAttention

核心：

```text
Attention Computation
怎么减少 IO。
```

---

# 185. PagedAttention

核心：

```text
KV Cache
怎么按 Block / Page 管理。
```

---

# 186. 两者不是竞争关系

Serving Runtime 可以：

```text
Paged KV Cache
+
Flash / Paged Attention Kernel。
```

---

# 187. FlashAttention 与 KV Cache

Prefill：

```text
Q/K/V 都是较长 Sequence。
```

适合：

```text
FlashAttention-like tiled attention。
```

---

# 188. Decode

通常：

```text
Q Length = 1
K/V Length = Context。
```

---

# 189. Decode Attention 的核心

```text
Q_new
×
K_cache
```

然后：

```text
Softmax
×
V_cache。
```

---

# 190. 为什么 Decode 与 Prefill Kernel 不同？

Prefill：

```text
很多 Q Row
```

具有：

```text
大矩阵并行。
```

Decode：

```text
Q 只有 1 / 少数 Row。
```

主要：

```text
读取长 KV Cache。
```

---

# 191. 所以：

```text
Prefill Attention
```

和：

```text
Decode Attention
```

经常使用：

```text
不同 Kernel 设计。
```

---

# 192. FlashAttention 对 Prefill 尤其自然

因为：

```text
N×N Attention
```

中间矩阵问题最突出。

---

# 193. Decode 更关注

```text
KV Cache Layout
Memory Bandwidth
Batch
Paged KV
GQA/MQA。
```

---

# 194. 第七部分：CUTLASS

CUTLASS：

> NVIDIA 提供的高性能 CUDA Linear Algebra 模板与 DSL 体系。

核心：

```text
GEMM
Convolution
Data Movement
Tensor Core
Layout
Epilogue
```

---

# 195. CUTLASS 和 cuBLAS

cuBLAS：

```text
调用已有高性能库 API。
```

例如：

```text
cublasGemmEx(...)
```

---

# 196. CUTLASS

更像：

```text
构建自己的 GEMM Kernel
所使用的高性能积木。
```

---

# 197. 为什么需要 CUTLASS？

如果：

```text
标准 GEMM
```

cuBLAS 很好。

如果想：

```text
Custom Dtype
Custom Layout
Custom Fusion
Custom Epilogue
Specialized Shape
```

CUTLASS：

```text
更灵活。
```

---

# 198. CUTLASS 的核心思想

```text
Hierarchical Decomposition
+
Data Movement。
```

---

# 199. GEMM 分解

```text
Device GEMM
↓
CTA / Threadblock GEMM
↓
Warp GEMM
↓
Thread / Instruction GEMM。
```

---

# 200. 这就是硬件映射

```text
Problem
↓
Grid
↓
Block
↓
Warp
↓
Instruction。
```

---

# 201. CUTLASS Mainloop

官方 GEMM 设计：

```text
Mainloop
```

主要：

```text
沿 K Tile 循环
↓
Load A/B
↓
Shared Memory / Pipeline
↓
MMA
↓
Accumulator。
```

---

# 202. CUTLASS Epilogue

Mainloop 后：

```text
Accumulator
↓
Epilogue
↓
Final Output。
```

---

# 203. Epilogue 可以做

```text
alpha * AB
+
beta * C
+
Bias
+
Activation
+
Scale
+
Auxiliary Output。
```

---

# 204. 为什么 Epilogue 非常适合 Fusion？

Accumulator：

```text
本来就在 Register / On-chip。
```

如果直接：

```text
做 Bias / Activation
```

避免：

```text
先写 Global
再重新读取。
```

---

# 205. CuTe

现代 CUTLASS 中非常重要：

```text
CuTe。
```

核心：

> 用形式化 Layout Algebra 描述 Tensor 和 Thread 到 Data 的映射。

---

# 206. 为什么 Layout Algebra 有意义？

高性能 Kernel 大量代码本质：

```text
Thread 17
应该加载 A 的哪个元素？

Warp 2
负责 C 的哪个 Tile？

Shared Memory 如何布局？
```

---

# 207. 传统做法

大量：

```text
手工 Index Arithmetic。
```

难：

```text
理解
验证
组合。
```

---

# 208. CuTe

把：

```text
Shape
Stride
Layout
Thread Mapping
```

变成：

```text
可组合的代数对象。
```

---

# 209. CUTLASS 3.x 设计核心变化之一

就是：

```text
使用 cute::Tensor / Layout
统一表达各种 Memory / Thread Mapping。
```

---

# 210. Mainloop + Epilogue

现代 CUTLASS 可以把 GEMM Kernel 看成：

```text
Collective Mainloop
+
Collective Epilogue
+
Tile Scheduler。
```

---

# 211. Tile Scheduler

尤其现代 GPU：

```text
Work Tile 如何分配给 CTA
```

本身也可能：

```text
影响性能。
```

---

# 212. Persistent Kernel

有些 Kernel：

```text
CTA 长时间驻留
不断领取 Tile。
```

而不是：

```text
一个 CTA 只做一个 Tile 就结束。
```

---

# 213. 为什么？

减少：

```text
Launch / Scheduling overhead
```

并改善：

```text
Work Balance。
```

---

# 214. Grouped GEMM

MoE：

```text
很多 Expert
```

每个 Expert：

```text
一个不同 Shape 的 GEMM。
```

---

# 215. 如果每个 Expert 单独 Launch

```text
大量小 Kernel。
```

效率可能很差。

---

# 216. Grouped GEMM

把：

```text
很多 GEMM Problem
```

放进：

```text
统一调度 / Kernel。
```

---

# 217. 为什么 CUTLASS 对 MoE 很重要？

因为：

```text
MoE
=
Routing
+
Grouped GEMM。
```

---

# 218. CUTLASS 当前资料里也大量包含

```text
FP8
Narrow Precision
Grouped GEMM
Blackwell / Hopper
Fusion
```

这直接对应现代 LLM。

---

# 219. 第八部分：Triton vs CUTLASS vs CUDA

---

# 220. CUDA C++

```text
控制力：
★★★★★

开发复杂度：
★★★★★
```

适合：

```text
理解硬件
极致定制
底层能力。
```

---

# 221. Triton

```text
控制力：
★★★★

生产力：
★★★★☆
```

适合：

```text
快速写 DL Kernel
Fusion
MatMul
Softmax
Norm
Attention。
```

---

# 222. CUTLASS

```text
针对 NVIDIA GEMM / Tensor Core：
★★★★★
```

但：

```text
Template / Layout 系统复杂。
```

---

# 223. 三者学习定位

```text
CUDA
=
理解 GPU Hardware

Triton
=
快速实践 Kernel Optimization

CUTLASS
=
理解工业级 GEMM / Tensor Core Kernel。
```

---

# 224. TVM 又在哪里？

TVM：

```text
Compiler 负责生成 / 调用 Kernel。
```

---

# 225. 整体层次

```text
PyTorch
↓
Compiler / Runtime
    ├── TVM
    ├── torch.compile
    └── vLLM
↓
Kernel
    ├── Triton
    ├── CUTLASS
    └── CUDA
↓
Hardware
```

---

# 226. `torch.compile` 与 Triton

PyTorch Compiler：

```text
Graph
↓
Inductor
↓
Triton Kernel
↓
GPU。
```

这是 Triton 极其重要的现实应用。

---

# 227. vLLM 与 Triton / CUTLASS

vLLM：

```text
Scheduler / Serving。
```

底层：

```text
可以调用
Triton / CUTLASS / FlashAttention
等 Kernel。
```

---

# 228. MLC / TVM 与 Kernel

MLC：

```text
Relax / TIR
↓
Generated / Library Kernel。
```

---

# 229. 一个成熟 LLM Stack

```text
Model
↓
Graph Compiler
↓
Runtime Scheduler
↓
Kernel Library / Generated Kernel
↓
GPU。
```

---

# 230. 第九部分：Kernel Fusion

最值得建立的一条原则：

> **如果多个 Operator 之间的中间数据只需要被下一步立刻消费，就思考能否不落 DRAM。**

---

# 231. Elementwise Fusion

例如：

```text
x
↓
add
↓
relu
↓
mul。
```

最容易 Fusion。

---

# 232. Reduction Fusion

例如：

```text
Softmax
LayerNorm
RMSNorm。
```

更复杂。

---

# 233. GEMM Epilogue Fusion

例如：

```text
MatMul
+
Bias
+
Activation。
```

非常常见。

---

# 234. Attention Fusion

```text
QK
+
Softmax
+
PV。
```

就是：

```text
FlashAttention。
```

属于：

```text
更高级 Fusion / Algorithm Restructuring。
```

---

# 235. Fusion 不是越多越好

过度 Fusion：

```text
Register Pressure ↑
Shared Memory ↑
Occupancy ↓
Code Complexity ↑。
```

---

# 236. Fusion Decision 也是 Tradeoff

```text
Memory Traffic ↓
```

vs：

```text
Resource Pressure ↑。
```

---

# 237. 第十部分：Profiling

没有 Profiling：

```text
GPU Optimization
很容易变成猜。
```

---

# 238. Nsight Systems

更适合：

```text
Timeline。
```

看：

```text
CPU
Kernel Launch
GPU Kernel
Memcpy
Synchronization
Overlap。
```

---

# 239. Nsight Compute

更适合：

```text
单个 Kernel 深度分析。
```

---

# 240. 重点指标

建议关注：

```text
Kernel Duration
Memory Throughput
DRAM Read / Write
L2 Hit Rate
Achieved Occupancy
Register Usage
Shared Memory Usage
Warp Stall Reasons
Tensor Core Utilization
Instructions
Launch Configuration。
```

---

# 241. 第一问永远是：

```text
Compute-Bound
还是
Memory-Bound？
```

---

# 242. Memory-Bound

看：

```text
Memory Throughput
DRAM Transactions
L2 Hit
Coalescing
Bytes / Element。
```

---

# 243. Compute-Bound

看：

```text
Tensor Core
FP Units
Instruction Throughput
Occupancy
Pipeline Stall。
```

---

# 244. 如果 Occupancy 低

先看：

```text
Registers / Thread
Shared Memory / Block
Threads / Block。
```

---

# 245. 但不要立刻优化 Occupancy

先问：

```text
它真的限制性能了吗？
```

---

# 246. Warp Stall

例如：

```text
Memory Dependency
Execution Dependency
Barrier
Not Selected
Long Scoreboard
Short Scoreboard。
```

可以帮助：

```text
判断 GPU 为什么在等。
```

---

# 247. Memory Coalescing

如果 DRAM Transaction：

```text
远大于理论需要。
```

可能：

```text
Access Pattern 差。
```

---

# 248. L2 Hit

如果：

```text
需要复用 Weight / Data
```

但 L2 Hit 很低：

```text
Layout / Work Scheduling
```

可能需要调整。

---

# 249. Tensor Core Utilization

MatMul：

```text
理论上应该用 Tensor Core。
```

但利用率低：

```text
Shape / dtype / tile / pipeline
```

可能有问题。

---

# 250. Kernel Benchmark 正确方式

一定：

```text
Warmup
多次重复
Synchronize
使用 GPU Timer。
```

---

# 251. 不要这样

```python
t0 = time.time()
kernel()
t1 = time.time()
```

GPU：

```text
异步执行。
```

CPU Timer：

```text
可能只量到 launch。
```

---

# 252. Triton

可以使用：

```text
triton.testing.do_bench
```

做基础 Benchmark。

---

# 253. PyTorch

可以使用：

```text
torch.cuda.Event。
```

---

# 254. Benchmark 必须验证 Correctness

```text
fast
```

但：

```text
答案错
```

没有意义。

---

# 255. Numerical Tolerance

FP16 / BF16：

```text
不应要求 bitwise identical。
```

使用：

```text
atol
rtol。
```

---

# 256. FlashAttention 官方测试也是类似思想

比较：

```text
Reference Attention
```

并允许：

```text
数值误差容忍。
```

---

# 257. 第十一部分：从 PyTorch 到 Kernel

一个：

```python
y = torch.nn.functional.silu(x @ w)
```

---

# 258. PyTorch Eager

可能：

```text
GEMM Kernel
↓
write y0
↓
SiLU Kernel
↓
write y。
```

---

# 259. Compiled / Fused

可能：

```text
GEMM
↓
Epilogue SiLU
↓
write y。
```

---

# 260. 这就是 Compiler + Kernel 协同

Graph Compiler：

```text
发现 Fusion Opportunity。
```

Kernel：

```text
实现 Fusion。
```

---

# 261. 再看 Attention

```text
QKᵀ
↓
Softmax
↓
PV。
```

Graph Compiler：

```text
看到 Pattern。
```

Runtime：

```text
可以 Dispatch FlashAttention Kernel。
```

---

# 262. 所以 Compiler 不一定自己生成 FlashAttention

可以：

```text
Pattern Match
↓
Call FlashAttention Library。
```

---

# 263. 这和 TVM BYOC / Library Dispatch 很像

```text
识别 Pattern
↓
选择高性能 External Implementation。
```

---

# 264. 第十二部分：FlashAttention 的更深一层

Attention：

```text
Q_i
```

需要和：

```text
所有 K_j
```

计算 Score。

---

# 265. 分块

把：

```text
Q
```

切：

```text
Q Block。
```

把：

```text
K/V
```

切：

```text
K/V Block。
```

---

# 266. 内层：

```text
Q Block
×
K Blockᵀ
```

得到：

```text
Score Tile。
```

---

# 267. Score Tile 不落 HBM

直接：

```text
Mask
↓
Online Softmax
↓
× V Tile
↓
Update O。
```

---

# 268. Running Statistics

每个 Q Row：

```text
m
=
当前最大 Score

l
=
当前 exp sum

O
=
当前 Output numerator-like accumulator。
```

---

# 269. 新 Block 到来

假设旧：

```text
m_old
l_old
O_old。
```

新 Score：

```text
S_new。
```

---

# 270. 新最大值

```text
m_new
=
max(
    m_old,
    max(S_new)
)。
```

---

# 271. Rescale Old Sum

```text
alpha
=
exp(m_old - m_new)。
```

---

# 272. New Tile Probability Numerator

```text
P_new
=
exp(S_new - m_new)。
```

---

# 273. New Sum

```text
l_new
=
alpha * l_old
+
sum(P_new)。
```

---

# 274. Output Accumulator 也要 Rescale

```text
O_new
=
alpha * O_old
+
P_new @ V_new。
```

最终：

```text
O = O_new / l_new。
```

具体实现形式可能对 accumulator 定义做不同安排，但核心：

```text
旧贡献必须按新 max 重新缩放。
```

---

# 275. 这是理解 FlashAttention 的数学核心

不是：

```text
记 CUDA 源码。
```

而是：

> **为什么 Softmax 可以 Block-wise 算且保持数学正确。**

---

# 276. 为什么不能简单分块各算各的 Softmax？

因为：

```text
softmax(block A)
+
softmax(block B)
```

不等于：

```text
softmax(A ∪ B)。
```

---

# 277. Online Softmax 解决：

```text
跨 Block 的 Global Normalization。
```

---

# 278. 这是非常漂亮的 Algorithm-System Co-design

算法：

```text
重新组织。
```

目的：

```text
适应有限 SRAM。
```

最终：

```text
减少 HBM IO。
```

---

# 279. FlashAttention 的性能并非所有 Shape 都一样

取决于：

```text
Sequence Length
Head Dim
Causal
Dtype
GPU Architecture
Forward / Backward
MQA/GQA
Variable Length。
```

---

# 280. 所以不要说：

```text
FlashAttention 永远快 X 倍。
```

正确做法：

```text
Benchmark Current Shape / GPU。
```

---

# 281. 第十三部分：Prefill / Decode 再联系

Prefill：

```text
Q length = N
K/V length = N。
```

---

# 282. Attention Shape

```text
QK:
[N,d] @ [d,N]
```

是：

```text
大 MatMul。
```

---

# 283. Decode

```text
Q length = 1。
```

---

# 284. Decode Attention

```text
[1,d]
@
[d,N]
```

更像：

```text
Matrix-Vector / Skinny MatMul。
```

---

# 285. Decode 主要读

```text
K Cache
V Cache。
```

---

# 286. 所以长 Context Decode

```text
Memory Traffic
```

非常关键。

---

# 287. GQA/MQA 的价值再出现

KV Head 少：

```text
KV Cache ↓
Memory Traffic ↓。
```

---

# 288. Kernel Design 也随之变化

MQA / GQA：

```text
多个 Q Head
共享 K/V Head。
```

可以：

```text
提高 K/V reuse。
```

---

# 289. 第十四部分：LLM 常见 Kernel

除了 GEMM / Attention：

```text
RMSNorm
LayerNorm
RoPE
Softmax
Activation
Elementwise
Embedding
Sampling
Top-K
Quantize / Dequantize
MoE Routing
Grouped GEMM
```

---

# 290. RMSNorm

典型：

```text
Reduction
+
Elementwise Scale。
```

通常：

```text
Memory-Bound。
```

适合：

```text
Fusion。
```

---

# 291. RoPE

```text
Elementwise / Pairwise rotation。
```

通常：

```text
FLOPs 不高。
```

---

# 292. 为什么会 Fuse RoPE？

避免：

```text
Q/K
写一次 Global
再读做 RoPE。
```

---

# 293. SwiGLU

```text
gate = silu(xW_gate)

up = xW_up

y = gate * up。
```

包含：

```text
两个 GEMM
+
Elementwise。
```

---

# 294. 可以优化什么？

```text
GEMM Epilogue
Elementwise Fusion
Grouped / Fused Projection
Quantized GEMM。
```

---

# 295. Quantized GEMM

Weight：

```text
INT4。
```

Kernel：

```text
Load Packed INT4
↓
Decode / Scale
↓
MMA / Dot
↓
Accumulate。
```

---

# 296. 量化 Kernel 的核心

不是：

```text
先把整个模型 Dequant 成 FP16。
```

否则：

```text
失去 Memory Benefit。
```

---

# 297. 更好的方式

```text
Tile-by-Tile Dequant
+
Compute Fusion。
```

---

# 298. CUTLASS 中也有 Mixed Dtype / Dequant Fusion

这就是现代 LLM Kernel 的重要方向。

---

# 299. MoE Kernel

```text
Router
↓
Token Permute
↓
Grouped GEMM
↓
Unpermute
↓
Combine。
```

---

# 300. MoE 的难点

```text
Expert Load 不均
Small GEMM
Irregular Memory
Dispatch / Communication。
```

---

# 301. Kernel 设计与 Runtime Scheduler 会合

vLLM：

```text
负责 Request / Token。
```

MoE Runtime：

```text
还要考虑 Expert Dispatch。
```

底层：

```text
Grouped GEMM。
```

---

# 302. 第十五部分：Kernel 性能分析模板

面对任何 Kernel：

先问 8 个问题。

---

# 303. 问题 1

```text
输入 / 输出 Shape 是什么？
```

---

# 304. 问题 2

```text
总 FLOPs 是多少？
```

---

# 305. 问题 3

```text
至少要搬多少 Byte？
```

---

# 306. 问题 4

```text
Arithmetic Intensity？
```

---

# 307. 问题 5

```text
理论更偏 Compute-bound
还是 Memory-bound？
```

---

# 308. 问题 6

```text
数据复用在哪里？
```

```text
L2？
Shared？
Register？
```

---

# 309. 问题 7

```text
并行维度怎么映射：
Grid
Block
Warp
Thread？
```

---

# 310. 问题 8

```text
Profiling 结果验证了理论吗？
```

---

# 311. 一个 Softmax 分析

FLOPs：

```text
相对较少。
```

Memory：

```text
读 / 写很多。
```

→：

```text
Memory-Bound。
```

优化：

```text
Fusion
Keep row on-chip。
```

---

# 312. 一个 GEMM 分析

FLOPs：

```text
2MNK。
```

Data：

```text
A + B + C。
```

如果 Tile Reuse 高：

```text
Compute-Bound。
```

优化：

```text
Tiling
Tensor Core
Pipeline。
```

---

# 313. 一个 Decode Attention

Q：

```text
很小。
```

KV：

```text
很长。
```

→：

```text
Memory-Bound 倾向。
```

优化：

```text
KV Layout
GQA
Paged KV
Batch。
```

---

# 314. 第十六部分：与 TVM Schedule 的映射

上一章：

```text
split
```

Kernel 层：

```text
Tile Dimension。
```

---

# 315. `bind`

Kernel：

```text
blockIdx / threadIdx。
```

---

# 316. `cache_read`

Kernel：

```text
Global → Shared / Local。
```

---

# 317. `vectorize`

CPU：

```text
SIMD。
```

GPU：

```text
Vectorized Memory Access / CodeGen。
```

---

# 318. `tensorize`

Kernel：

```text
Tensor Core MMA。
```

---

# 319. MetaSchedule

Kernel：

```text
搜索 Tile / Warp / Stage / Layout。
```

---

# 320. 所以 Schedule 不再抽象

你现在可以看到：

```text
每个 Schedule Primitive
都对应真实 Hardware Decision。
```

---

# 321. 第十七部分：与 llama.cpp 的映射

llama.cpp：

```text
ggml Op:
MUL_MAT。
```

---

# 322. Backend

CUDA Backend：

```text
选择 Quant / GEMM Kernel。
```

---

# 323. Prompt Processing

```text
大型 GEMM
```

可能：

```text
非常适合 GPU / BLAS / Tensor Core。
```

---

# 324. Decode

```text
Skinny GEMM / GEMV
```

需要：

```text
专门 Kernel
Quantized Weight streaming。
```

---

# 325. llama.cpp 为什么有大量 Quant Kernel？

因为：

```text
Q4_K_M
```

只有：

```text
对应 Kernel 高效执行
```

才有价值。

---

# 326. 第十八部分：与 vLLM 的映射

vLLM：

```text
Scheduler
↓
Model Runner
↓
Attention / GEMM Kernel。
```

---

# 327. Scheduler 可以让 Kernel 更高效吗？

间接。

例如：

```text
增加 Decode Batch
```

让：

```text
GEMV-like
→
更像 GEMM。
```

提高：

```text
Arithmetic Intensity。
```

---

# 328. Chunked Prefill

把：

```text
Compute-heavy Prefill
```

和：

```text
Memory-heavy Decode
```

组合。

也是：

```text
系统层帮助硬件利用。
```

---

# 329. 第十九部分：与 MLC / TVM 的映射

MLC：

```text
Relax
↓
TIR
↓
Target Code。
```

---

# 330. Triton / CUTLASS 在这里是什么？

它们代表：

```text
Target Kernel Implementation。
```

---

# 331. Compiler 可以：

```text
自己生成
```

也可以：

```text
Dispatch 到 CUTLASS。
```

---

# 332. 第二十部分：与 NPU 的迁移

虽然 RK3576 NPU：

```text
不是 NVIDIA GPU。
```

但这一章很多核心思想仍成立：

```text
Memory Hierarchy
Tiling
Data Reuse
Tensor Unit
Layout
Dtype
DMA
Fusion。
```

---

# 333. NPU 也要问

```text
Tensor 是否放得进 SRAM？
Tile 多大？
DMA 怎么搬？
算子是否 Fusion？
INT8 / INT4 Matrix Unit 怎么用？
```

---

# 334. 不同的是

GPU：

```text
用户 / Compiler
拥有相对开放 Programming Model。
```

NPU：

```text
很多细节封装在 Vendor Compiler。
```

---

# 335. 所以学 GPU Kernel 的价值

你能更清楚理解：

> Vendor NPU Compiler 在黑盒里大概必须做哪些工作。

---

# 336. 本章实验路线

---

# 337. 实验 1：CUDA Vector Add

写：

```text
CPU Vector Add
CUDA Vector Add。
```

理解：

```text
Grid
Block
Thread。
```

---

# 338. 实验 2：Block Size

测试：

```text
32
64
128
256
512。
```

记录：

```text
Latency。
```

---

# 339. 观察

不是：

```text
Thread 越多越快。
```

---

# 340. 实验 3：Memory Coalescing

版本 A：

```text
连续访问。
```

版本 B：

```text
stride access。
```

比较：

```text
Bandwidth。
```

---

# 341. 实验 4：Shared Memory

做一个简单：

```text
Matrix Transpose。
```

用：

```text
Global only
```

和：

```text
Shared Memory。
```

比较。

---

# 342. 实验 5：Bank Conflict

设计：

```text
有冲突的 Shared Layout
```

vs：

```text
Padding 后无冲突。
```

---

# 343. 实验 6：Naive MatMul

自己写：

```text
1 Thread
→
1 C element。
```

Benchmark。

---

# 344. 实验 7：Tiled MatMul

加入：

```text
Shared Memory Tile。
```

比较：

```text
GFLOPS。
```

---

# 345. 实验 8：Register Blocking

一个 Thread：

```text
计算多个 C element。
```

---

# 346. 实验 9：cuBLAS Baseline

同 Shape：

```text
自己的 GEMM
vs
cuBLAS。
```

---

# 347. 不要目标一开始就是打败 cuBLAS

目标：

```text
理解差距来自哪里。
```

---

# 348. 实验 10：Triton Vector Add

对应：

```text
CUDA Vector Add。
```

比较代码量。

---

# 349. 实验 11：Triton Softmax

运行官方：

```text
02-fused-softmax.py。
```

---

# 350. 记录

```text
PyTorch Eager

Triton Fused
```

对不同：

```text
Row Length。
```

性能。

---

# 351. 实验 12：Triton MatMul

运行：

```text
03-matrix-multiplication.py。
```

理解：

```text
BLOCK_M/N/K
num_warps
num_stages
autotune。
```

---

# 352. 实验 13：修改 Tile

手工减少 AutoTune Config。

测试：

```text
32×32
64×64
128×64
128×128。
```

---

# 353. 实验 14：看 Triton IR

打开：

```text
IR dump。
```

至少知道：

```text
Python DSL
不是最终执行形式。
```

---

# 354. 实验 15：标准 Attention

PyTorch：

```python
scores = q @ k.transpose(-2, -1)
probs = softmax(scores)
out = probs @ v
```

测：

```text
Peak Memory
Latency。
```

---

# 355. 实验 16：FlashAttention

同样：

```text
B/H/N/D。
```

测：

```text
Latency
Peak Memory。
```

---

# 356. 实验 17：Sequence Length Sweep

```text
N =
128
512
1024
2048
4096
8192。
```

观察：

```text
标准 Attention
vs
FlashAttention。
```

---

# 357. 实验 18：Causal

```text
causal=False
vs
causal=True。
```

---

# 358. 实验 19：Head Dim

```text
64
128
256。
```

---

# 359. 实验 20：Triton Fused Attention

阅读：

```text
06-fused-attention.py。
```

先：

```text
画出算法
```

再：

```text
读代码。
```

---

# 360. 实验 21：CUTLASS GEMM

先运行：

```text
Simple GEMM Example。
```

---

# 361. 实验 22：看 GEMM Hierarchy

定位：

```text
CTA Tile
Warp Tile
Instruction Tile。
```

---

# 362. 实验 23：Epilogue

尝试：

```text
GEMM
+
Activation / Bias。
```

---

# 363. 实验 24：Nsight Systems

分析：

```text
PyTorch MLP。
```

看：

```text
有多少 Kernel Launch。
```

---

# 364. 实验 25：torch.compile

对同一个 MLP：

```text
Eager
vs
torch.compile。
```

观察：

```text
Kernel 数量变化。
```

---

# 365. 实验 26：Nsight Compute

选择：

```text
一个 GEMM
一个 Softmax。
```

对比：

```text
Memory Throughput
Tensor Core
Occupancy
Warp Stall。
```

---

# 366. 推荐实验仓库结构

```text
gpu-kernel-learning/
│
├── 01_cuda_vector_add/
├── 02_cuda_coalescing/
├── 03_shared_memory/
├── 04_bank_conflict/
├── 05_naive_matmul/
├── 06_tiled_matmul/
├── 07_register_blocking/
├── 08_triton_vector_add.py
├── 09_triton_softmax.py
├── 10_triton_matmul.py
├── 11_attention_baseline.py
├── 12_flash_attention.py
├── 13_cutlass_gemm/
├── 14_epilogue_fusion/
└── profiling/
    ├── nsys/
    └── ncu/
```

---

# 367. 每个实验记录

```text
GPU
Compute Capability

dtype

Shape

Block Size

Tile Size

num_warps

num_stages

Latency

GB/s

TFLOPS

Registers

Shared Memory

Occupancy

L2 Hit

DRAM Throughput。
```

---

# 368. 本章推荐学习顺序

---

## Stage 1：GPU 基础

```text
Thread
Warp
Block
SM
SIMT。
```

---

## Stage 2：Memory

```text
Global
L2
Shared
Register
Coalescing
Bank Conflict。
```

---

## Stage 3：Performance Model

```text
Arithmetic Intensity
Memory-bound
Compute-bound
Occupancy
Latency Hiding。
```

---

## Stage 4：GEMM

```text
Naive
↓
Tiling
↓
Shared
↓
Register
↓
Tensor Core。
```

---

## Stage 5：Triton

```text
Vector Add
↓
Softmax
↓
MatMul。
```

---

## Stage 6：FlashAttention

先：

```text
数学 / IO。
```

后：

```text
源码。
```

---

## Stage 7：CUTLASS

```text
GEMM hierarchy
Mainloop
Epilogue
CuTe。
```

---

## Stage 8：Profiling

```text
Nsight Systems
↓
Nsight Compute。
```

---

# 369. 推荐优先级

| 内容 | 优先级 |
|---|---|
| Thread / Warp / Block | ★★★★★ |
| SM | ★★★★★ |
| Memory Hierarchy | ★★★★★ |
| Coalescing | ★★★★★ |
| Shared Memory | ★★★★★ |
| Register | ★★★★★ |
| Occupancy | ★★★★ |
| Arithmetic Intensity | ★★★★★ |
| GEMM Tiling | ★★★★★ |
| Register Blocking | ★★★★★ |
| Tensor Core / MMA | ★★★★★ |
| Kernel Fusion | ★★★★★ |
| Triton | ★★★★★ |
| Triton Softmax | ★★★★★ |
| Triton MatMul | ★★★★★ |
| FlashAttention | ★★★★★ |
| Online Softmax | ★★★★★ |
| FlashAttention-2 | ★★★★ |
| FlashAttention-3 | ★★★ |
| CUTLASS GEMM | ★★★★★ |
| CuTe | ★★★★ |
| Nsight Compute | ★★★★★ |
| Warp Specialization | ★★★ |
| TMA / WGMMA | ★★★ |

---

# 370. 为什么 TMA / WGMMA 不先学？

因为：

```text
它们是 Hopper+ 现代优化能力。
```

如果还不懂：

```text
Tile
Shared Memory
Warp
Pipeline
```

直接学：

```text
TMA / WGMMA
```

容易只背 API。

---

# 371. 为什么 FlashAttention 是本章核心？

它同时包含：

```text
Algorithm
Tiling
Memory Hierarchy
Online Reduction
Fusion
MatMul
Softmax
GPU Scheduling。
```

---

# 372. 它是 AI Systems 很典型的案例

不是：

```text
仅仅换了一个 CUDA Kernel。
```

而是：

```text
重新设计 Algorithm
以适应 Hardware。
```

---

# 373. 常见误区

---

## 误区 1

```text
GPU Thread 越多越快。
```

错误。

还受：

```text
Resource
Work Size
Memory
Occupancy。
```

---

## 误区 2

```text
Occupancy 100%
=
性能最好。
```

错误。

---

## 误区 3

```text
Shared Memory 比 Global 快
所以越多越好。
```

错误。

Shared Memory：

```text
占 SM Resource。
```

过多：

```text
Occupancy 降。
```

---

## 误区 4

```text
Tensor Core 有很高 TFLOPS
所以 LLM Decode 一定很快。
```

错误。

Decode：

```text
常受 Memory Bandwidth 限制。
```

---

## 误区 5

```text
Tiling 减少 MatMul FLOPs。
```

通常不。

它主要：

```text
提高 Data Reuse。
```

---

## 误区 6

```text
Triton 是 Python GPU Interpreter。
```

错误。

Triton：

```text
是 DSL + Compiler。
```

---

## 误区 7

```text
Triton 比 CUDA 抽象高
所以一定比 CUDA 慢。
```

错误。

高层 Tile Abstraction：

```text
仍可编译成高效 GPU Code。
```

---

## 误区 8

```text
FlashAttention 是近似 Attention。
```

错误。

核心：

```text
Exact IO-aware Attention。
```

---

## 误区 9

```text
FlashAttention 把 O(N²) 变成 O(N)。
```

错误。

主要：

```text
减少 IO / Memory。
```

不是把核心 Attention FLOPs 降成线性。

---

## 误区 10

```text
FlashAttention = PagedAttention。
```

错误。

---

## 误区 11

```text
FlashAttention 只是在 Softmax 上做 Fusion。
```

不完整。

它是：

```text
QK
+
Online Softmax
+
PV
```

整体 Tile Pipeline。

---

## 误区 12

```text
CUTLASS = cuBLAS。
```

错误。

cuBLAS：

```text
现成 Library API。
```

CUTLASS：

```text
构建 / 定制高性能 Kernel 的组件。
```

---

## 误区 13

```text
Kernel 快只看 TFLOPS。
```

错误。

还要：

```text
Bandwidth
Latency
Launch
Memory Traffic
Occupancy。
```

---

# 374. 本章知识树

```text
GPU Kernel
│
├── Execution Model
│   ├── Grid
│   ├── Block / CTA
│   ├── Warp
│   ├── Thread
│   ├── SIMT
│   ├── SM
│   └── Divergence
│
├── Memory Hierarchy
│   ├── Global / HBM
│   ├── L2
│   ├── Shared Memory
│   ├── Register
│   ├── Coalescing
│   ├── Bank Conflict
│   └── Register Spill
│
├── Performance
│   ├── Arithmetic Intensity
│   ├── Compute-bound
│   ├── Memory-bound
│   ├── Occupancy
│   ├── Latency Hiding
│   └── Warp Stall
│
├── GEMM
│   ├── CTA Tile
│   ├── Warp Tile
│   ├── Instruction Tile
│   ├── Shared Tiling
│   ├── Register Blocking
│   ├── Pipeline
│   ├── Tensor Core
│   ├── MMA
│   ├── Mainloop
│   └── Epilogue
│
├── Triton
│   ├── Program
│   ├── program_id
│   ├── arange
│   ├── load/store
│   ├── Tile
│   ├── tl.dot
│   ├── num_warps
│   ├── num_stages
│   └── autotune
│
├── FlashAttention
│   ├── QKᵀ
│   ├── Softmax
│   ├── PV
│   ├── IO-aware
│   ├── Tiling
│   ├── Online Softmax
│   ├── Running Max
│   ├── Running Sum
│   ├── Running Output
│   ├── FA2
│   └── FA3
│
├── CUTLASS
│   ├── CuTe
│   ├── Layout
│   ├── GEMM Hierarchy
│   ├── Collective Mainloop
│   ├── Collective Epilogue
│   ├── Tile Scheduler
│   ├── Grouped GEMM
│   └── Mixed Precision
│
└── Profiling
    ├── Nsight Systems
    ├── Nsight Compute
    ├── Memory Throughput
    ├── L2 Hit
    ├── Occupancy
    ├── Registers
    ├── Shared Memory
    ├── Tensor Core
    └── Warp Stall
```

---

# 375. 本章最重要的十个思维模型

## 1

```text
GPU Performance
=
Work Partition
+
Data Movement
+
Compute。
```

---

## 2

```text
Warp
=
GPU 实际重要执行 / 调度粒度。
```

---

## 3

```text
Fast Memory
容量小

Slow Memory
容量大。
```

---

## 4

```text
Tiling
=
把数据搬进快 Memory
并尽可能复用。
```

---

## 5

```text
High Occupancy
≠
High Performance。
```

---

## 6

```text
GEMM Performance
=
Tiling
+
Pipeline
+
Tensor Core
+
Data Movement。
```

---

## 7

```text
Fusion
=
减少中间 Tensor
落回 DRAM。
```

---

## 8

```text
FlashAttention
=
Exact Attention
+
IO-aware Tiling
+
Online Softmax。
```

---

## 9

```text
Triton
=
Tile-based GPU Kernel DSL + Compiler。
```

---

## 10

```text
CUTLASS
=
工业级 CUDA GEMM / Tensor Core
分层实现体系。
```

---

# 376. 本章完成标准

## GPU 基础

- [ ] 能解释 Kernel
- [ ] 能解释 Grid
- [ ] 能解释 Block / CTA
- [ ] 能解释 Warp
- [ ] 能解释 Thread
- [ ] 能解释 SIMT
- [ ] 能解释 Warp Divergence
- [ ] 能解释 SM

## Memory

- [ ] 能解释 Global Memory
- [ ] 能解释 L2
- [ ] 能解释 Shared Memory
- [ ] 能解释 Register
- [ ] 能解释 Coalescing
- [ ] 能解释 Bank Conflict
- [ ] 能解释 Register Spill
- [ ] 能解释 Register Pressure

## Performance

- [ ] 能解释 Arithmetic Intensity
- [ ] 能判断简单 Kernel 是 Memory-bound 还是 Compute-bound
- [ ] 能解释 Occupancy
- [ ] 能解释为什么 Occupancy 不是最终目标
- [ ] 能解释 Latency Hiding
- [ ] 能解释 Warp Stall

## GEMM

- [ ] 能写 Naive GEMM
- [ ] 能解释为什么 Naive GEMM 慢
- [ ] 能解释 CTA Tile
- [ ] 能解释 Warp Tile
- [ ] 能解释 Instruction Tile
- [ ] 能解释 Shared Memory Tiling
- [ ] 能解释 Register Blocking
- [ ] 能解释 Mainloop
- [ ] 能解释 Epilogue
- [ ] 能解释 Double Buffering
- [ ] 能解释 Tensor Core
- [ ] 能解释 MMA

## Triton

- [ ] 能解释 Triton 定位
- [ ] 能解释 Triton vs CUDA
- [ ] 能解释 Program Instance
- [ ] 能使用 `tl.program_id`
- [ ] 能使用 `tl.arange`
- [ ] 能使用 `tl.load/store`
- [ ] 能完成 Vector Add
- [ ] 能看懂 Fused Softmax
- [ ] 能解释 `BLOCK_M/N/K`
- [ ] 能解释 `num_warps`
- [ ] 能解释 `num_stages`
- [ ] 能解释 Autotune
- [ ] 能运行 MatMul Tutorial

## FlashAttention

- [ ] 能写出标准 Attention
- [ ] 能解释 `N×N` Score Matrix
- [ ] 能解释标准 Attention 的 HBM IO
- [ ] 能解释 FlashAttention 为什么是 exact
- [ ] 能解释 IO-aware
- [ ] 能解释 Q/K/V Tiling
- [ ] 能解释 Online Softmax
- [ ] 能推导 Running Max 更新
- [ ] 能解释 Running Sum Rescale
- [ ] 能解释 Output Accumulator 更新
- [ ] 能画出 FlashAttention Forward Loop
- [ ] 能解释 FA2 主要优化并行与 Work Partition
- [ ] 知道 FA3 面向更现代 Hopper 能力
- [ ] 能区分 FlashAttention 与 PagedAttention
- [ ] 能解释 Prefill vs Decode Attention 差异

## CUTLASS

- [ ] 能解释 CUTLASS
- [ ] 能解释 CUTLASS vs cuBLAS
- [ ] 能解释 GEMM Hierarchy
- [ ] 能解释 Mainloop
- [ ] 能解释 Epilogue
- [ ] 能解释 CuTe
- [ ] 能解释 Layout Algebra 的意义
- [ ] 能运行一个 CUTLASS GEMM Example
- [ ] 能解释 Grouped GEMM 与 MoE 的关系

## Profiling

- [ ] 能解释 Nsight Systems
- [ ] 能解释 Nsight Compute
- [ ] 能正确 Benchmark GPU Kernel
- [ ] 能查看 Memory Throughput
- [ ] 能查看 Achieved Occupancy
- [ ] 能查看 Register
- [ ] 能查看 Shared Memory
- [ ] 能查看 L2 Hit
- [ ] 能查看 Warp Stall
- [ ] 能查看 Tensor Core 使用情况

---

# 377. 暂时不要求深入

当前暂时不要求：

- [ ] 精通 CUDA ISA
- [ ] 手写 PTX
- [ ] 精通 SASS
- [ ] 自己实现 FlashAttention-3
- [ ] 精通 Backward FlashAttention
- [ ] 精通所有 CUTLASS Template
- [ ] 精通 CuTe Layout Algebra 全部语法
- [ ] 精通 TMA
- [ ] 精通 WGMMA
- [ ] 精通 Warp Specialization
- [ ] 实现 Triton Backend
- [ ] 自己写 MLIR Pass
- [ ] 打败 cuBLAS
- [ ] 打败官方 FlashAttention
- [ ] 精通 Blackwell Kernel

先做到：

```text
会分析
会写基础 Kernel
会 Benchmark
会 Profile
会解释为什么快。
```

---

# 378. 核心参考链接

## CUDA

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

重点：

```text
Warp
Block
SM
Memory Hierarchy
Occupancy
Coalescing
Shared Memory
Synchronization。
```

---

## Triton

- [Triton Repository](https://github.com/triton-lang/triton)
- [README](https://github.com/triton-lang/triton/blob/main/README.md)
- [Installation](https://github.com/triton-lang/triton/blob/main/docs/getting-started/installation.rst)
- [Triton Language API](https://github.com/triton-lang/triton/blob/main/docs/python-api/triton.language.rst)
- [Vector Add](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py)
- [Fused Softmax](https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py)
- [Matrix Multiplication](https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py)
- [Layer Norm](https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py)
- [Fused Attention](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py)
- [Triton Compiler Source](https://github.com/triton-lang/triton/blob/main/python/triton/compiler/compiler.py)
- [Backend Compiler Interface](https://github.com/triton-lang/triton/blob/main/python/triton/backends/compiler.py)

---

## FlashAttention

- [FlashAttention Repository](https://github.com/Dao-AILab/flash-attention)
- [README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://tridao.me/publications/flash2/flash2.pdf)
- [FlashAttention-3 Paper](https://tridao.me/publications/flash3/flash3.pdf)

重点：

```text
IO-aware
Tiling
Online Softmax
Exact Attention
Work Partitioning
Hardware-aware Pipeline。
```

---

## CUTLASS

- [CUTLASS Repository](https://github.com/NVIDIA/cutlass)
- [CUTLASS README](https://github.com/NVIDIA/cutlass/blob/main/README.md)
- [CUTLASS GEMM API](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/gemm_api.md)
- [CUTLASS 3.x Design](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cutlass_3x_design.md)
- [CuTe GEMM Tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0x_gemm_tutorial.md)
- [CUTLASS Examples](https://github.com/NVIDIA/cutlass/blob/main/examples/README.md)

重点：

```text
CTA Tile
Warp Tile
Instruction Tile
Mainloop
Epilogue
Data Movement
CuTe
Tensor Core。
```

---

# 379. 推荐源码阅读顺序

## Triton

```text
01-vector-add
↓
02-fused-softmax
↓
03-matmul
↓
05-layernorm
↓
06-attention
↓
triton.language
↓
compiler
↓
backend。
```

---

## FlashAttention

```text
Paper Figure / Algorithm
↓
README
↓
Python Interface
↓
Forward Kernel
↓
Backward Kernel
↓
FA2 work partition
↓
FA3 Hopper implementation。
```

不要：

```text
第一天直接啃 CUDA Template。
```

---

## CUTLASS

```text
README
↓
GEMM API
↓
CuTe GEMM Tutorial
↓
Simple GEMM Example
↓
Mainloop
↓
Epilogue
↓
Layout / CuTe
↓
Hopper / Blackwell Advanced Example。
```

---

# 380. 与上一章的映射

```text
09_AI_Compiler与TVM

split
↓
Tile

bind
↓
Block / Thread

cache_read
↓
Shared Memory

tensorize
↓
Tensor Core MMA

MetaSchedule
↓
Search Kernel Configuration。
```

---

# 381. 与 LLM 推理的映射

```text
06_LLM推理原理

Prefill
↓
Large GEMM
+
FlashAttention

Decode
↓
Quantized GEMV / Skinny GEMM
+
KV Cache Attention。
```

---

# 382. 与 vLLM 的映射

```text
vLLM
Scheduler
↓
Batch Shape
↓
Kernel Shape
↓
FlashAttention / Paged Attention / GEMM。
```

---

# 383. 与 llama.cpp 的映射

```text
GGML Op
↓
Backend
↓
Quantized Kernel
↓
CPU / GPU。
```

---

# 384. 与 RK3576 的最终迁移

后面即使进入：

```text
RKNPU
```

仍然要保留本章的四个问题：

```text
1. Work 怎么切？

2. Data 放哪里？

3. Data 怎么搬？

4. Matrix Unit 怎么吃满？
```

---

# 385. 最终大图

```text
                       AI Model
                          │
                          ▼
                       Compiler
                          │
                          ▼
                       Schedule
                          │
                          ▼
                    GPU Kernel
                          │
       ┌──────────────────┼──────────────────┐
       ▼                  ▼                  ▼
 Work Partition       Data Movement       Compute
       │                  │                  │
       ▼                  ▼                  ▼
 Grid / Block          HBM / L2          CUDA Core
 Warp / Thread       Shared / Reg        Tensor Core
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                       Performance
```

GEMM：

```text
Matrix
↓
CTA Tile
↓
Warp Tile
↓
Instruction Tile
↓
Tensor Core MMA
```

FlashAttention：

```text
Q Tile
↓
K/V Tile
↓
QKᵀ Tile
↓
Online Softmax
↓
PV Accumulate
↓
No Full N×N Materialization
↓
Less HBM IO
```

---

# 386. 一句话总结

本章真正需要建立的是：

```text
GPU Kernel Performance
不是“算得多快”一个问题。

而是：

数据怎么搬
+
任务怎么分
+
计算单元怎么利用。
```

其中：

```text
Tiling
=
Data Reuse

Shared Memory
=
Block-level Fast Storage

Register
=
Thread-level Fast Storage

Tensor Core
=
Matrix Compute Engine

Triton
=
Tile-based Kernel DSL + Compiler

CUTLASS
=
工业级 GEMM / Tensor Core 构建体系

FlashAttention
=
把 IO-aware 思维应用到完整 Attention。
```

如果可以真正解释：

```text
为什么 Naive MatMul 慢？

为什么 Tiled GEMM 快？

为什么 Softmax Fusion 快？

为什么 FlashAttention 不保存完整 N×N Matrix？

为什么 Online Softmax 仍然是 exact？

为什么 Occupancy 高不代表一定快？
```

那么就已经真正从：

```text
AI Compiler
```

走到了：

```text
GPU Kernel / Hardware Optimization。
```

下一步就可以把这种思维继续迁移到：

> **Rockchip RKNN / RKNPU / RKLLM**

开始研究：

```text
模型为什么需要 Vendor Compiler？
.rknn 到底是什么？
NPU Operator Mapping 怎么发生？
INT8 / INT4 为什么是 NPU 的关键？
DDR / SRAM / DMA 对 NPU 性能有什么影响？
RKNN-Toolkit2 和 RKLLM 为什么是两条不同工具链？
```

这会把整条路线最终落到：

```text
RK3576 Edge AI Hardware。
```
