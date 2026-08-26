# 08_vLLM与MLC-LLM

> **所属路线**：AI 学习路线 · 第二部分  
> **定位**：在理解 LLM 推理原理与 llama.cpp 之后，进一步学习两种非常典型的高性能 LLM Runtime 思想：**vLLM 的 Serving / Scheduling 路线**与 **MLC-LLM 的 ML Compilation / Cross-platform Deployment 路线**  
> **核心仓库**：`vllm-project/vllm` + `mlc-ai/mlc-llm`  
> **底层关联**：PyTorch、CUDA/HIP、FlashAttention/FlashInfer、Paged KV Cache、Apache TVM、Relax、TIR、Vulkan、Metal、WebGPU、OpenCL  
> **学习边界**：本章重点理解 Runtime Architecture、Scheduler、KV Cache Management 与 Compiler-driven Deployment；TVM Compiler、TIR、GPU Kernel、FlashAttention、CUTLASS 在后续章节独立深入。  
> **资料检查日期**：2026-08-25

---

# 0. 本章学习目标

完成本章以后，需要能够回答：

1. vLLM 到底是什么？它和 llama.cpp 的主要目标为什么完全不同？
2. 为什么普通 Hugging Face `model.generate()` 不适合直接做大规模在线 Serving？
3. 为什么 LLM Serving 的核心问题不只是“把 MatMul 算快”？
4. Continuous Batching 是什么？
5. 为什么传统 Static Batching 会浪费 GPU？
6. vLLM Scheduler 到底在调度什么？
7. 当前 vLLM V1 为什么说“Scheduler 中没有严格的 Prefill Phase / Decode Phase”？
8. `max_num_batched_tokens` 与 `max_num_seqs` 分别代表什么？
9. Chunked Prefill 为什么能改善 ITL，同时提升 GPU 利用率？
10. 为什么当前 V1 会优先调度 Decode，再利用剩余 Token Budget 调度 Prefill？
11. PagedAttention 为什么出现？
12. KV Cache 为什么要按照 Block / Page 管理？
13. Block Table 是什么？
14. Logical KV Block 与 Physical KV Block 有什么区别？
15. Paged KV Cache 和操作系统 Virtual Memory 的类比在哪里成立、哪里不完全成立？
16. vLLM 的 KVCacheManager 在 Scheduler 与实际 KV Block 之间承担什么角色？
17. Prefix Caching 是怎样复用 KV Block 的？
18. 为什么 Prefix Cache 通常通过 Token Prefix Hash 来识别可复用 Block？
19. Prefix Cache 为什么可以降低 TTFT，却不能直接提升正常 Decode 的每 Token 算力？
20. Preemption 是什么？
21. 为什么 KV Cache 不够时需要重新安排请求？
22. Scheduler Watermark / Admission Control 为什么重要？
23. vLLM 的 Engine、Scheduler、KV Cache Manager、Model Runner、Executor 大致是什么关系？
24. Offline `LLM.generate()` 与 Online `vllm serve` 有什么区别？
25. OpenAI-compatible Server 为什么在工程上很重要？
26. vLLM 当前支持哪些常见并行思想？
27. Tensor Parallel、Pipeline Parallel、Data Parallel、Expert Parallel 分别解决什么问题？
28. MoE 为什么让 Serving 进一步复杂？
29. Speculative Decoding 在 vLLM 中处于什么层？
30. FlashAttention / FlashInfer / CUTLASS 这类 Kernel 和 vLLM 的关系是什么？
31. vLLM 是 AI Compiler 吗？
32. `torch.compile` 为什么已经进入现代 vLLM 优化栈？
33. MLC-LLM 到底是什么？
34. 为什么 MLC-LLM 的核心不是 GGUF，而是“Weight + Compiled Model Library”？
35. MLC-LLM 为什么依赖 TVM？
36. TVM Relax 在 MLC-LLM 中处于什么位置？
37. MLC 中的 Model Definition 如何变成 `IRModule`？
38. `mlc_llm convert_weight`、`gen_config`、`compile` 分别做什么？
39. `mlc-chat-config.json` 里为什么包含 Context、Quantization、Prefill Chunk 等编译信息？
40. MLC Model Library 是什么？
41. `.so`、`.dylib`、`.dll`、`.tar`、`.wasm` 为什么都可能成为 MLC 编译产物？
42. 为什么 MLC 的 Weight 可以跨某些平台共享，而 Model Library 要针对 Target 编译？
43. JIT Compile 与 Ahead-of-Time Compile 有什么区别？
44. `MLCEngine` 是什么？
45. MLC 的 `local / interactive / server` mode 有什么差别？
46. 为什么 MLC 也有 Paged KV Cache？
47. 为什么 MLC 也需要 Scheduler / Serving Engine？
48. MLC 的 `q4f16_1` 与 llama.cpp 的 `Q4_K_M` 为什么是两套不同量化体系？
49. MLC-LLM 如何支持 CUDA、Vulkan、Metal、WebGPU、iOS、Android？
50. MLC-LLM 为什么非常接近真正的 AI Compiler 学习入口？
51. llama.cpp、vLLM、MLC-LLM 三者应该怎么选？
52. 在 RK3576 上，vLLM 为什么通常不是主要部署方案？
53. MLC-LLM 能否直接等于 Rockchip NPU Runtime？
54. MLC Vulkan/OpenCL 路线和 RKLLM NPU 路线有什么根本区别？
55. 为什么学习完 MLC-LLM 后自然应该进入 TVM？

本章最终建立三条路线：

```text
路线 A：llama.cpp

HF Model
↓
GGUF
↓
llama.cpp
↓
ggml Graph
↓
CPU / GPU Backend
↓
Local / Edge Inference
```

```text
路线 B：vLLM

HF Model
↓
vLLM Engine
↓
Scheduler
↓
Continuous Batching
↓
Paged KV Cache
↓
Optimized GPU Kernels
↓
High-throughput Serving
```

```text
路线 C：MLC-LLM

HF Model
↓
Weight Conversion
↓
MLC Model Definition
↓
TVM Relax IR
↓
Compiler Pass
↓
Target-specific Model Library
↓
MLCEngine
↓
CUDA / Vulkan / Metal / WebGPU / ...
```

---

# 1. 为什么把 vLLM 和 MLC-LLM 放在一章？

因为这三个项目：

```text
llama.cpp
vLLM
MLC-LLM
```

都能：

```text
运行 LLM
```

但它们真正代表的是三种不同的系统设计思想。

---

# 2. 三种 Runtime 思想

## llama.cpp

```text
重点：
Local / Edge / CPU / Heterogeneous

核心：
C/C++
GGUF
Quantized Tensor
Hand-written Runtime
Backend Abstraction
```

---

## vLLM

```text
重点：
GPU Server / High Concurrency / High Throughput

核心：
Scheduler
Continuous Batching
Paged KV Cache
Prefix Cache
Optimized GPU Kernel
Distributed Serving
```

---

## MLC-LLM

```text
重点：
Compile Once / Cross-platform Native Deployment

核心：
ML Compilation
TVM Relax
Target Code Generation
Compiled Model Library
Unified Runtime
```

---

# 3. 一张总对比图

```text
                    LLM Runtime
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
    llama.cpp          vLLM           MLC-LLM
        │               │                │
     GGUF          HF / PyTorch       HF Model
        │               │                │
     ggml           Scheduler        TVM Relax
        │               │                │
 Quant Kernel       Paged KV        Compile Pass
        │               │                │
 CPU/GPU          CUDA/HIP/...      Target Library
        │               │                │
 Local/Edge        GPU Serving       Cross-platform
```

---

# 4. 第一部分：vLLM

GitHub：

- [vllm-project/vllm](https://github.com/vllm-project/vllm)

官方文档：

- [vLLM Documentation](https://docs.vllm.ai/)
- [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [vLLM V1 Guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)

---

# 5. vLLM 的定位

vLLM 可以定义为：

> **面向高吞吐、低延迟 LLM Inference / Serving 的 GPU-first Runtime Engine。**

它重点解决：

```text
Many Requests
+
Dynamic Sequence Length
+
KV Cache Memory
+
Batching
+
Scheduling
+
Kernel Efficiency
+
Distributed GPUs
```

---

# 6. 为什么不能直接用 Hugging Face `generate()` 做大型 Serving？

最简单：

```python
model.generate(...)
```

非常适合：

```text
实验
单请求
功能验证
```

但服务器环境可能：

```text
100
1000
10000
```

个 Request 并发。

问题变成：

```text
谁先 Prefill？
谁在 Decode？
如何组成 Batch？
KV Cache 放在哪里？
某个用户完成后 Batch 怎么补人？
GPU Memory 不够怎么办？
长 Prompt 会不会阻塞其他用户？
```

这些都不是单纯：

```text
model.forward()
```

的问题。

---

# 7. vLLM 的核心价值

可以压缩成：

```text
Model Execution
+
KV Cache Management
+
Request Scheduling
+
Batching
+
Serving
```

---

# 8. vLLM V1

当前学习 vLLM：

> 应该以 V1 为主线。

当前官方已经明确：

```text
V0 已完全 deprecated
```

V1 对：

```text
Scheduler
KV Cache Manager
Worker
Sampler
API Server
```

等核心系统做了重新架构。

推荐：

- [vLLM V1 Guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)

---

# 9. 为什么要知道 V1？

很多旧教程会讲：

```text
BlockSpaceManager
V0 Scheduler
SequenceGroup
Swap
```

这些概念和当前实现：

```text
可能已经变化。
```

学习系统项目一定要：

> 区分历史论文思想和当前工程实现。

---

# 10. vLLM 核心架构

可以先建立：

```text
Client / API
      ↓
AsyncLLM / LLM
      ↓
EngineCore
      ↓
Scheduler
      │
      ├── Request Queue
      ├── Token Budget
      └── KV Allocation
      ↓
Executor / Worker
      ↓
Model Runner
      ↓
GPU Kernels
```

旁边：

```text
KVCacheManager
```

持续参与：

```text
Block Allocation
Prefix Cache
Free / Reuse
Admission
```

---

# 11. EngineCore

源码入口：

- [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)

`EngineCore` 可以理解：

> V1 Engine 的核心内部循环。

它把：

```text
Request
Scheduler
Executor
KV Cache
Model Runner Output
```

组织起来。

---

# 12. Scheduler

当前源码：

- [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [Scheduler Interface](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/interface/)

Scheduler 解决：

> **下一次 Model Forward 应该处理哪些 Request、每个 Request 处理多少 Token。**

---

# 13. 这是一个非常重要的定义

不要把 Scheduler 理解成：

```text
“给 Request 排队”
```

这么简单。

实际上每个 Engine Step：

```text
Scheduler
↓
决定：
Request A → 1 token
Request B → 1 token
Request C → 512 tokens
Request D → 0
```

然后：

```text
组成一次 GPU Forward。
```

---

# 14. V1 Unified Scheduling

当前 vLLM V1 有一个特别重要的变化：

> Scheduler 内部不再必须严格区分“这是 Prefill Request”还是“这是 Decode Request”。

它更统一地看：

```text
num_computed_tokens
```

和：

```text
num_tokens_with_spec
```

差多少。

---

# 15. 换句话说

Request：

```text
已经计算 100 Tokens

应该计算到 500 Tokens
```

那么：

```text
还欠 400 Tokens。
```

Scheduler 决定这一 Step：

```text
给它 256 Tokens
```

就形成：

```text
Chunked Prefill。
```

---

# 16. Decode Request

如果 Request：

```text
已经算到当前最后一个 Token
```

下一步只需要：

```text
1 Token
```

Scheduler：

```text
num_tokens = 1。
```

---

# 17. 所以 V1 的统一模型

```text
Every Request
=
Need to Catch Up
from
num_computed_tokens
to
num_tokens
```

这个抽象非常漂亮。

因为同一套 Scheduler 可以自然支持：

```text
Normal Prefill
Decode
Chunked Prefill
Prefix Cache
Speculative Decode
```

---

# 18. Token Budget

Scheduler 每一轮有：

```text
Token Budget。
```

核心配置：

```text
max_num_batched_tokens
```

表示：

> 一次 Scheduler Step / Model Forward 最多处理多少 Token。

---

# 19. `max_num_seqs`

另一个：

```text
max_num_seqs
```

限制：

> 一次 Iteration 最多同时包含多少个 Sequence / Request。

所以：

```text
max_num_batched_tokens
=
Token 维度预算

max_num_seqs
=
Sequence 数量预算
```

---

# 20. 为什么两个都需要？

例如：

```text
128 个 Sequence
```

每个 Decode：

```text
1 Token
```

共：

```text
128 Tokens。
```

但一个长 Prefill：

```text
可能单 Request 就 4096 Tokens。
```

所以：

```text
Sequence Count
```

和：

```text
Token Count
```

必须分别限制。

---

# 21. Static Batching

传统：

```text
Request A
Request B
Request C
Request D
↓
一起组成 Batch
↓
全部完成
↓
下一 Batch
```

问题：

```text
A 输出 20 Tokens
B 输出 200 Tokens
```

A：

```text
早结束
```

但 Batch Slot：

```text
可能被浪费。
```

---

# 22. Continuous Batching

核心：

```text
某个 Sequence 结束
↓
马上移出
↓
新 Request 加入
```

不需要：

```text
等整个 Batch 完成。
```

---

# 23. 为什么 LLM 特别适合 Continuous Batching？

因为 Decode 本来就是：

```text
Step 1
Step 2
Step 3
...
```

每一步之间：

```text
都可以重新组织 Batch。
```

---

# 24. Iteration-level Scheduling

这就是：

```text
Iteration-level scheduling。
```

每次 Model Forward 前：

```text
重新决定 Batch。
```

---

# 25. Continuous Batching 的核心收益

```text
GPU Idle ↓
Batch Utilization ↑
Throughput ↑
```

特别是：

```text
输出长度差异很大的请求。
```

---

# 26. 但 Continuous Batching 也有成本

Scheduler 每一步要处理：

```text
Request State
KV Mapping
Batch Metadata
Output Routing
```

所以：

```text
Runtime Complexity ↑
```

---

# 27. Prefill 和 Decode 的冲突

上一章知道：

```text
Prefill
=
Compute-heavy

Decode
=
Memory-heavy
```

Server 同时存在：

```text
Long Prefill Request
+
Many Decode Requests
```

怎么办？

---

# 28. 最简单做法

一次把：

```text
4096-token Prompt
```

全部 Prefill。

问题：

```text
GPU 被长 Prefill 占住
```

正在聊天的其他用户：

```text
下一个 Token 要等待。
```

ITL：

```text
变差。
```

---

# 29. Chunked Prefill

把：

```text
4096 Prompt
```

拆：

```text
512
512
512
...
```

---

# 30. 当前 vLLM V1 的 Chunked Prefill

当前官方优化文档中：

```text
Chunked Prefill
```

在可用情况下默认开启。

Scheduler 会优先：

```text
Decode Requests
```

然后使用：

```text
剩余 Token Budget
```

调度 Prefill。

---

# 31. 为什么优先 Decode？

因为 Decode：

```text
直接决定正在输出用户的 ITL。
```

如果一轮推迟：

```text
用户看到输出卡顿。
```

---

# 32. Chunked Prefill 的另一个收益

可以把：

```text
Compute-bound Prefill
```

和：

```text
Memory-bound Decode
```

更合理地组合在同一批工作中。

理论上：

```text
提升硬件利用率。
```

---

# 33. `max_num_batched_tokens` 的调优

较小：

```text
Prefill Chunk 更小
ITL 更友好
TTFT 可能变差
GPU 大 GEMM 效率可能降低
```

较大：

```text
Prefill 吞吐更好
TTFT 可能更好
但 Decode 可能受到干扰
```

---

# 34. 所以它是经典 Tradeoff

```text
TTFT
vs
ITL
vs
Throughput
```

---

# 35. PagedAttention 从哪里来？

vLLM 最著名的设计之一：

```text
PagedAttention。
```

它最开始的核心动机：

> **高效管理动态增长、长度不一致的 KV Cache。**

---

# 36. 传统 KV Cache 分配的问题

假设：

```text
Request A:
128 Tokens

Request B:
2048 Tokens

Request C:
37 Tokens
```

如果每个 Request：

```text
提前申请完整 Max Context
```

极其浪费。

---

# 37. 另一问题：Memory Fragmentation

每个 Request：

```text
长度不断增加
结束时间不同
```

如果用：

```text
大块连续 Buffer
```

很容易：

```text
出现 Fragmentation。
```

---

# 38. Page / Block 思想

把 KV Memory：

```text
切成固定大小 Block。
```

例如概念上：

```text
Block 0
Block 1
Block 2
...
```

每个 Block：

```text
保存固定数量 Token 的 KV。
```

---

# 39. Request 不再需要一整块连续 KV

例如 Sequence A：

```text
Logical Blocks:
0
1
2
```

可以映射到：

```text
Physical Blocks:
17
3
42
```

---

# 40. Block Table

负责：

```text
Logical Block
→
Physical Block
```

映射。

这和：

```text
Virtual Memory Page Table
```

非常类似。

---

# 41. PagedAttention 名字中的 Paged

不是说：

```text
Attention 数学变成 Page Attention。
```

而是：

> Attention Kernel 需要按照 Page / Block Table 访问非连续 KV Cache。

---

# 42. 数学仍然是

```text
Q
×
K_cache
↓
Softmax
↓
V_cache
```

变化的是：

```text
K/V 在内存中的管理与访问。
```

---

# 43. KV Cache Block 的价值

```text
按需分配
按需释放
减少内部浪费
减少外部碎片
方便 Prefix Sharing
支持动态 Batch
```

---

# 44. KVCacheManager

当前源码：

- [vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
- [Hybrid KV Cache Manager Design](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)

它位于：

```text
Scheduler
↕
KV Block System
```

之间。

---

# 45. KVCacheManager 负责什么？

概念：

```text
Request 需要更多 KV
↓
Allocate Blocks

Request 完成
↓
Free Blocks

Request 有 Prefix
↓
Find Cached Blocks

Memory 不够
↓
Admission / Preemption related decisions
```

---

# 46. 为什么当前还有 Hybrid KV Cache Manager？

现代 Model 不一定所有 Layer 都是：

```text
Full Attention。
```

可能：

```text
Full Attention
+
Sliding Window
+
Mamba
+
其他 Efficient Attention。
```

这些 Layer：

```text
KV / State 需求不同。
```

因此 Runtime 需要：

```text
更通用的 Cache Manager。
```

---

# 47. 这和 llama.cpp 的 Memory Abstraction 是同一个趋势

现代 LLM 已经不再是：

```text
所有模型 = 标准 LLaMA Transformer。
```

Runtime 必须支持：

```text
Hybrid Architecture。
```

---

# 48. Prefix Caching

场景：

```text
所有用户共享 2000 Token System Prompt。
```

每次 Prefill：

```text
2000 Tokens
```

很浪费。

---

# 49. Automatic Prefix Caching

第一次：

```text
Prefix
↓
Prefill
↓
KV Blocks
↓
Cache
```

第二个 Request：

```text
Same Prefix
↓
Reuse KV Blocks
```

---

# 50. Prefix Cache 怎么判断 Same Prefix？

vLLM 的经典实现：

```text
对 KV Block 对应 Token Prefix
做 Hash。
```

如果：

```text
Hash 匹配
```

则可以找到：

```text
已存在的 Block。
```

---

# 51. 为什么 Hash 不能只看当前 Block Token？

因为 Block 1：

```text
在 Prefix A 后
```

和相同 Token：

```text
在 Prefix B 后
```

它们的 K/V：

```text
可能不同。
```

所以 Hash 必须：

```text
包含前缀上下文关系。
```

---

# 52. Prefix Cache 的典型价值

```text
Long System Prompt
Few-shot Prompt
Shared Documents
Multi-turn Conversation Prefix
Agent Repeated Context
```

---

# 53. Prefix Cache 主要优化什么？

```text
Prefill Compute ↓
TTFT ↓
```

它不意味着：

```text
模型 Decode Matrix 更快。
```

---

# 54. Prefix Cache 与 Prompt Cache

概念上都在：

```text
复用已计算上下文。
```

不同 Runtime：

```text
具体命名 / 实现不同。
```

---

# 55. Scheduler Admission

假设 KV Cache：

```text
只剩 10 Blocks。
```

新 Request：

```text
可能需要 100 Blocks。
```

不能简单：

```text
立即接受。
```

---

# 56. Over-admission

如果 Scheduler：

```text
接太多 Request
```

随后：

```text
KV 不够
```

就会频繁：

```text
Preempt
Evict
Recompute
```

导致：

```text
Thrashing。
```

---

# 57. Watermark

当前 SchedulerConfig 提供：

```text
watermark
```

思路：

> 保留一定比例 KV Cache Headroom。

避免：

```text
内存刚好用满
→
下一步不断抢占。
```

---

# 58. `scheduler_reserve_full_isl`

当前 V1 还可以：

> 在接纳新 Request 时检查完整 Input Sequence 是否能放进 KV，而不只是第一块 Chunk。

目的：

```text
避免 Chunked Prefill 过度接纳请求。
```

---

# 59. Preemption

当资源不足：

```text
某些 Request
```

可能被：

```text
暂停 / 重新排队。
```

后面再恢复。

---

# 60. 当前 V1 和旧版 Swap

旧资料经常讲：

```text
GPU KV
↔
CPU KV Swap。
```

当前 V1 Guide 已明确：

```text
不再依赖旧式 GPU↔CPU KV Swap
来处理 Preemption。
```

所以读旧论文时要知道：

> 设计已经演进。

---

# 61. Request 生命周期

可以粗略：

```text
WAITING
↓
RUNNING
↓
FINISHED
```

资源不足时：

```text
RUNNING
↓
PREEMPTED / WAITING
```

然后：

```text
重新调度。
```

---

# 62. Scheduler Policy

常见：

```text
FCFS
Priority
```

即：

```text
First Come First Served
```

或：

```text
优先级调度。
```

---

# 63. Scheduler 的真正目标

不是单一：

```text
最大 Tokens/s。
```

而是平衡：

```text
Throughput
TTFT
ITL
Fairness
KV Memory
Starvation
```

---

# 64. vLLM Model Runner

Scheduler 决定：

```text
谁跑
多少 Token
```

Model Runner：

> 真正准备 Tensor 并执行 Model Forward。

---

# 65. Executor / Worker

多 GPU 环境：

```text
Executor
```

协调：

```text
Worker / GPU Process。
```

比如：

```text
Tensor Parallel
Pipeline Parallel。
```

---

# 66. vLLM 并不自己手写所有 Kernel

底层可以依赖 / 集成：

```text
FlashAttention
FlashInfer
CUTLASS
Triton
TRTLLM-GEN
...
```

具体随硬件和版本变化。

---

# 67. vLLM 的价值在更高一层

即使有最快：

```text
GEMM Kernel
Attention Kernel
```

如果：

```text
Batch 差
KV 管理差
Scheduler 差
```

Server：

```text
仍然跑不快。
```

---

# 68. Runtime Optimization ≠ Kernel Optimization

vLLM 强大的地方很大一部分是：

```text
Runtime-level Optimization。
```

---

# 69. vLLM 当前首页的重要能力

当前官方列出：

```text
PagedAttention
Continuous Batching
Chunked Prefill
Prefix Caching
CUDA/HIP Graph
Many Quantizations
Optimized Attention Kernels
Optimized GEMM / MoE Kernels
Speculative Decoding
torch.compile
Disaggregated Prefill/Decode
Distributed Parallelism
```

所以现代 vLLM 已远超过最早：

```text
PagedAttention 一个点。
```

---

# 70. Speculative Decoding

Scheduler 需要支持：

```text
一次 Request
可能在一个 Step 处理 >1 Output Token。
```

因此 V1 的统一：

```text
num_tokens
```

抽象非常适合。

---

# 71. CUDA Graph

Decode：

```text
Shape 相对稳定
Kernel Launch 多
```

CUDA Graph 可以：

```text
减少 CPU Launch Overhead。
```

---

# 72. `torch.compile`

当前 vLLM 也会利用：

```text
torch.compile
```

进行：

```text
Graph-level transformation
Operator Fusion
Generated Kernel
```

所以：

> vLLM 正在同时吸收 Runtime 与 Compiler 技术。

---

# 73. vLLM 是 AI Compiler 吗？

最好说：

```text
不是主要定位。
```

它是：

> **LLM Serving Runtime / Engine。**

但内部会使用：

```text
torch.compile
Kernel Generation
Graph Transform
```

等 Compiler 技术。

---

# 74. vLLM Offline API

```python
from vllm import LLM

llm = LLM(
    model="..."
)

outputs = llm.generate(
    ["Hello"]
)
```

适合：

```text
Offline Inference
Batch Evaluation
Dataset Generation
```

---

# 75. vLLM Server

典型：

```bash
vllm serve MODEL
```

启动：

```text
OpenAI-compatible API Server。
```

---

# 76. 为什么 OpenAI-compatible API 很重要？

上层 Application：

```text
LangChain
Agent
RAG
OpenAI SDK
```

不需要知道：

```text
后端是 OpenAI
还是 vLLM。
```

只要：

```text
API Compatible。
```

---

# 77. Agent + vLLM

上一章：

```text
Agent
↓
LLM API
```

可以直接：

```text
Agent
↓
vLLM Server
```

用于：

```text
私有部署
高并发。
```

---

# 78. Parallelism

当模型：

```text
单 GPU 放不下
```

或者希望：

```text
扩大 Throughput
```

需要 Multi-GPU。

---

# 79. Tensor Parallel

```text
一个 Layer 的 Weight
```

拆到：

```text
多个 GPU。
```

例如：

```text
W
↓
GPU0 shard
GPU1 shard
GPU2 shard
GPU3 shard
```

---

# 80. TP 适合

```text
单模型太大
```

或：

```text
希望每 GPU 留更多 KV 空间。
```

---

# 81. TP 的代价

每层可能需要：

```text
AllReduce
AllGather
```

所以：

```text
GPU Interconnect
```

很重要。

---

# 82. Pipeline Parallel

```text
Layer 0-19 → GPU0
Layer 20-39 → GPU1
...
```

---

# 83. PP 优点

对：

```text
很深模型
```

可以按 Layer 切。

---

# 84. PP 代价

```text
Stage dependency
Pipeline bubble
Communication
```

尤其单 Token Decode：

```text
延迟链长。
```

---

# 85. Data Parallel

完整模型：

```text
复制多份。
```

每份：

```text
处理不同 Request Batch。
```

---

# 86. DP 主要提升

```text
Serving Throughput
```

而不是：

```text
让单个模型装得下。
```

---

# 87. Expert Parallel

MoE：

```text
不同 Expert
```

分布到不同 GPU。

---

# 88. 为什么 MoE 特别适合 EP？

每个 Token：

```text
只访问少数 Expert。
```

如果：

```text
Expert 分布合理
```

可以减少单卡 Weight 压力。

---

# 89. EP 最大挑战

Router 输出：

```text
Token → Different Expert GPU
```

需要：

```text
All-to-All Communication。
```

---

# 90. MoE Serving 不是简单 MatMul

还包括：

```text
Routing
Load Balance
Dispatch
All-to-All
Expert GEMM
Combine
```

---

# 91. vLLM 推荐源码阅读顺序

不要从 Kernel 开始。

推荐：

```text
1. docs/usage/v1_guide.md

2. vllm/entrypoints
   or LLM API

3. vllm/v1/engine/core.py

4. vllm/v1/core/sched/scheduler.py

5. vllm/v1/core/kv_cache_manager.py

6. model runner

7. attention layer

8. GPU kernel
```

---

# 92. 第一阶段：Scheduler

重点跟：

```text
schedule()
```

观察它返回：

```text
Request → num_tokens
```

---

# 93. 第二阶段：KV

跟：

```text
get_computed_blocks
allocate_slots
free
```

理解：

```text
Request
↔
KV Blocks。
```

---

# 94. 第三阶段：Model Runner

观察：

```text
SchedulerOutput
↓
Input Tensor
↓
Forward
↓
Logits
↓
Sample
```

---

# 95. 第四阶段：Attention Backend

最后再看：

```text
Paged Attention Kernel
FlashAttention
FlashInfer。
```

---

# 96. vLLM 必做实验 1：Offline

安装环境允许时：

```python
from vllm import LLM

llm = LLM(
    model="small-model"
)
```

运行：

```text
单 Prompt
多 Prompt。
```

---

# 97. 实验 2：OpenAI Server

```bash
vllm serve MODEL
```

然后：

```text
OpenAI SDK
```

调用。

---

# 98. 实验 3：Concurrency

并发：

```text
1
2
4
8
16
32
```

记录：

```text
TTFT
TPOT
ITL
Output t/s
Request/s
```

---

# 99. 实验 4：`max_num_seqs`

不同：

```text
32
64
128
```

观察：

```text
Throughput
KV Memory
Latency。
```

---

# 100. 实验 5：`max_num_batched_tokens`

改变 Token Budget：

```text
小
中
大
```

比较：

```text
TTFT
ITL
Throughput。
```

---

# 101. 实验 6：Long Prefill + Active Decode

同时：

```text
一个 8K Prompt
+
多个正在 Decode 的 Request
```

观察：

```text
不开 / 开 Chunked Prefill
```

ITL。

---

# 102. 实验 7：Prefix Cache

构造：

```text
固定 4K Prefix
+
不同 User Query。
```

第一批：

```text
Cold
```

第二批：

```text
Hit。
```

比较：

```text
TTFT。
```

---

# 103. 实验 8：Tensor Parallel

2 GPU：

```text
TP=1
TP=2
```

比较：

```text
Model Capacity
KV Capacity
Single-request Latency
Throughput。
```

---

# 104. 第二部分：MLC-LLM

GitHub：

- [mlc-ai/mlc-llm](https://github.com/mlc-ai/mlc-llm)

官方：

- [MLC LLM Documentation](https://llm.mlc.ai/)
- [Introduction](https://llm.mlc.ai/docs/get_started/introduction.html)
- [Compile Model Libraries](https://llm.mlc.ai/docs/compilation/compile_models.html)
- [Python MLCEngine](https://llm.mlc.ai/docs/deploy/python_engine.html)

---

# 105. MLC 是什么缩写？

这里更重要的是思想：

```text
Machine Learning Compilation。
```

MLC-LLM：

> **使用机器学习编译技术，把 LLM 针对具体硬件编译成高性能 Native Runtime / Model Library。**

---

# 106. 官方定位

MLC-LLM：

```text
Machine Learning Compiler
+
High-performance LLM Deployment Engine
```

核心目标：

```text
Universal Deployment。
```

---

# 107. 与 llama.cpp 最大区别

llama.cpp：

```text
Runtime 在运行时构建 ggml Graph
+
不同 Backend 手写 Kernel / Dispatch。
```

MLC：

```text
模型先进入 Compiler IR
↓
做 Pass / Lowering / Codegen
↓
生成目标平台 Model Library。
```

---

# 108. 一张核心图

```text
Hugging Face Model
      │
      ├──────── Weights ───────────────┐
      │                                │
      ▼                                ▼
MLC Model Architecture            convert_weight
      │                                │
      ▼                                ▼
TVM Relax IR                    Quantized Weights
      │
      ▼
Compiler Pipeline
      │
      ▼
Target-specific Code
      │
      ▼
Model Library
.so / dylib / wasm / tar
      │
      └──────────────┐
                     ▼
                 MLCEngine
                     │
              loads weights
                     │
                     ▼
          CUDA / Vulkan / Metal /
          WebGPU / OpenCL / ...
```

---

# 109. MLC 的两个部署产物

官方文档强调：

运行一个 Model 需要：

```text
1. Converted / Quantized Model Weights

2. Model Library
```

---

# 110. Model Weights

例如：

```text
Llama-...-q4f16_1-MLC/
```

存：

```text
转换 / 量化后的模型参数
Tokenizer / Config。
```

---

# 111. Model Library

包含：

> **模型的推理逻辑经过编译后的目标代码。**

例如：

```text
Linux:
.so

Windows:
.dll

macOS:
.dylib

Web:
.wasm

iOS / Android:
.tar / static library related artifacts
```

---

# 112. 这是 MLC 与 GGUF 最大的认知差别之一

GGUF：

```text
Weight + Metadata
```

Graph 主要由：

```text
llama.cpp Runtime 代码
```

在运行时构建。

MLC：

```text
Inference Logic
```

会提前：

```text
编译成 Model Library。
```

---

# 113. 为什么模型库可以独立于 Weight？

如果：

```text
Architecture
参数 Shape
Quantization Layout
```

一样。

只换：

```text
Fine-tuned Weight。
```

有些情况下：

```text
同一个 Model Library
```

可以继续使用。

---

# 114. 这非常像传统 Compiler

```text
Program Logic
↓
Compile
↓
Binary

Data
↓
Runtime Input
```

MLC：

```text
Model Logic
↓
Compile
↓
Model Library

Weight
↓
Runtime Data
```

当然 Weight 不是普通输入，但这个类比非常有帮助。

---

# 115. TVM 是 MLC-LLM 的编译基础

MLC-LLM 编译模型时需要：

```text
TVM Compiler。
```

如果要真正理解：

```text
MLC 为什么能跨 CUDA / Metal / Vulkan
```

最终必须学 TVM。

---

# 116. Relax

当前 MLC Model Definition 使用：

```python
from tvm.relax.frontend import nn
```

例如：

- [Llama Model Source](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/llama/llama_model.py)

可以看到：

```text
nn.Module
Tensor
op
PagedKVCache
```

---

# 117. 这和 PyTorch 很像

PyTorch：

```python
class LlamaModel(nn.Module):
    ...
```

MLC：

```python
from tvm.relax.frontend import nn

class LlamaModel(nn.Module):
    ...
```

看起来熟悉。

但最终目的不同：

```text
PyTorch:
Eager / Training / Runtime

MLC:
Generate Compiler IR
```

---

# 118. MLC Model Definition

例如：

```text
LlamaDecoderLayer
LlamaModel
LlamaForCausalLM
```

代码里：

```text
RMSNorm
Attention
MLP
Residual
```

仍然是我们熟悉的 Transformer。

---

# 119. 最关键的一步

MLC Frontend：

```text
Model Definition
↓
Export / Convert
↓
TVM IRModule
```

---

# 120. IRModule

可以先理解：

> TVM 编译器中承载模型函数和低层函数的一组 IR。

里面可能包含：

```text
Relax Function
TIR PrimFunc
Runtime Function Metadata
```

---

# 121. Relax 负责什么？

粗略：

```text
High-level Tensor Program
```

例如：

```text
Linear
Attention
Paged KV Cache
Sampling
```

---

# 122. TIR 负责什么？

更低层：

```text
Loop
Buffer
Thread Binding
Memory Access
Tile
```

后面的：

> TVM 章节会深入。

---

# 123. MLC Compiler Pipeline

源码：

- [python/mlc_llm/compiler_pass/pipeline.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler_pass/pipeline.py)

它会在：

```text
IRModule
```

上执行一系列：

```text
Compiler Pass。
```

---

# 124. Compiler Pass 是什么？

Compiler 中：

```text
IR
↓
Pass 1
↓
IR'
↓
Pass 2
↓
IR''
...
```

每个 Pass：

```text
分析或修改程序。
```

---

# 125. MLC 当前 Compiler Pipeline 可以看到

例如：

```text
Attach GPU Sampling
Attach Logit Processor
Attach Softmax with Temperature
Attach Spec Decode Functions
CUDA Graph related pass
Optimization / Lowering
```

等。

---

# 126. 为什么 Sampling 都能被 Compiler Attach？

llama.cpp：

```text
Sampling
主要是 Runtime Sampler Chain。
```

MLC：

```text
部分 Sampling 逻辑
可以被编译成 GPU Function。
```

例如源码：

- [attach_sampler.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler_pass/attach_sampler.py)

---

# 127. 这体现编译驱动路线的特点

不仅：

```text
Transformer Forward
```

可以编译。

甚至：

```text
Sampling
Logit Processing
Spec Decode Helper
```

也可以进入编译优化。

---

# 128. Target

Compiler 必须知道：

```text
目标硬件。
```

例如：

```text
cuda
metal
vulkan
webgpu
android
iphone
```

---

# 129. 为什么 Target 如此重要？

同一个：

```text
MatMul
```

针对：

```text
NVIDIA CUDA
Apple Metal
WebGPU
```

需要：

```text
不同 Kernel
Memory Scope
Thread Mapping
Codegen。
```

---

# 130. Compile-time Optimization

Compiler 可以提前知道：

```text
Model Architecture
Hidden Size
Layer Count
Quantization
Target
Context Metadata
```

从而：

```text
生成更加 target-specific 的代码。
```

---

# 131. llama.cpp 的优点与 MLC 的优点不同

llama.cpp：

```text
Runtime 灵活
Backend 手写优化
GGUF 易部署。
```

MLC：

```text
Compiler 可系统地做 Target Optimization
Cross Compile
Native Library Packaging。
```

---

# 132. MLC 完整 Workflow

官方当前典型：

```text
1. convert_weight

2. gen_config

3. compile

4. MLCEngine / chat / serve
```

---

# 133. `convert_weight`

典型：

```bash
mlc_llm convert_weight \
    ./HF_MODEL \
    --quantization q4f16_1 \
    -o dist/MODEL-q4f16_1-MLC
```

---

# 134. 这一步做什么？

```text
读取 HF Weight
↓
Tensor Name Mapping
↓
Quantize
↓
Layout Conversion
↓
保存 MLC-compatible Weight。
```

---

# 135. Weight Mapping

不同 Framework：

```text
q_proj.weight
```

在 MLC 内部：

```text
可能对应不同参数名 / layout。
```

Loader 需要：

```text
ExternMapping。
```

---

# 136. On-the-fly Quantization

MLC Loader 设计里支持：

```text
读取原始 FP16/BF16 Weight
↓
加载时 Quantize
```

或者：

```text
读取已有 AutoAWQ / GPTQ 等预量化权重。
```

---

# 137. `q4f16_1`

MLC 的典型 Quant 名。

当前源码中：

```text
q4f16_1
```

大致表示：

```text
Weight:
group-wise int4

Model compute dtype:
float16

group_size:
32

specific linear layout:
NK
```

---

# 138. 它不是 llama.cpp `Q4_K_M`

必须牢记：

```text
q4f16_1
≠
Q4_K_M
```

虽然：

```text
都可以叫“4-bit”。
```

---

# 139. 为什么不能互换？

它们定义：

```text
Block Layout
Scale
Packing
Compute dtype
Kernel Contract
```

完全不同。

---

# 140. 量化格式和 Runtime Kernel 是一个生态

```text
GGUF Q4_K_M
↔
ggml Quant Kernel

MLC q4f16_1
↔
TVM / MLC Generated Kernel
```

---

# 141. `gen_config`

典型：

```bash
mlc_llm gen_config \
    ./HF_MODEL \
    --quantization q4f16_1 \
    --conv-template ... \
    -o dist/MODEL-MLC
```

---

# 142. `mlc-chat-config.json`

包含：

```text
Model Architecture
Quantization
Conversation Template
Context Window
Prefill Chunk Size
Tokenizer information
Runtime Metadata
```

---

# 143. 为什么 Context 会进入 Compile Config？

因为：

```text
Context
Prefill Chunk
Batch
```

会影响：

```text
Memory Planning
Dynamic Shape Bounds
KV Cache Configuration。
```

---

# 144. 这和普通 Python Runtime 很不一样

Compiler：

```text
希望尽可能知道 Shape Bound。
```

这样可以：

```text
提前规划 Memory。
```

---

# 145. `compile`

典型：

```bash
mlc_llm compile \
    dist/MODEL-MLC/mlc-chat-config.json \
    --device cuda \
    -o dist/libs/model-cuda.so
```

---

# 146. `compile` 真正做什么？

粗略：

```text
Load Model Definition
↓
Apply Quantized Model Transform
↓
Export Relax IR
↓
Apply Compiler Pipeline
↓
Lower Operators
↓
Schedule / Optimize
↓
Target Codegen
↓
Link
↓
Model Library
```

---

# 147. JIT Compile

如果：

```text
Python MLCEngine
```

启动时没有找到：

```text
model_lib
```

MLC 可以：

```text
自动 JIT compile。
```

---

# 148. JIT 的体验

第一次运行：

```text
Download / Load Weight
Compile Model
Start Runtime
```

后续：

```text
使用 Cache
```

避免重复编译。

---

# 149. AOT Compile

你也可以提前：

```bash
mlc_llm compile ...
```

生成：

```text
model library。
```

Deployment Device：

```text
不需要完整 Compiler 依赖。
```

---

# 150. 为什么 AOT 对端侧重要？

Android / iOS / Web：

```text
不希望在用户设备上
带完整 TVM Compiler。
```

所以：

```text
Host 编译
↓
Target Library
↓
Device Runtime。
```

---

# 151. Cross Compilation

Compiler 所在机器：

```text
不一定是最终 Device。
```

只要：

```text
Target Toolchain
```

存在。

例如：

```text
PC
↓
compile
↓
Android library。
```

---

# 152. MLC 的 Universal Deployment

当前官方主页列出：

```text
NVIDIA GPU:
CUDA / Vulkan

AMD:
ROCm / Vulkan

Intel:
Vulkan

Apple:
Metal

Web:
WebGPU + WASM

iOS:
Metal

Android:
OpenCL on Adreno / Mali
```

具体支持持续变化，以当前文档为准。

---

# 153. 为什么 WebGPU 很特别？

浏览器：

```text
不能直接运行 CUDA。
```

MLC 可以：

```text
Compile
↓
WebGPU / WASM
```

于是：

```text
Browser Local LLM。
```

---

# 154. WebLLM

MLC 生态：

```text
WebLLM
```

就是：

> 将 MLC 编译能力带到浏览器端。

---

# 155. iOS

Apple A-series / M-series：

```text
Metal。
```

可以：

```text
编译成 iOS native library。
```

---

# 156. Android

Android GPU 路线：

```text
OpenCL / Vulkan
```

依 GPU 和当前支持而异。

---

# 157. 这就是 MLC 为什么值得 Edge AI 学

它把：

```text
同一个 LLM
```

经过：

```text
Target Compilation
```

部署到：

```text
Desktop
Mobile
Browser。
```

---

# 158. MLC Runtime

编译完 Model Library 后：

```text
仍然需要 Runtime。
```

即：

```text
MLCEngine。
```

---

# 159. MLCEngine

Python：

```python
from mlc_llm import MLCEngine

engine = MLCEngine(model)
```

它负责：

```text
Load Model Library
Load Weights
Manage KV
Schedule Requests
Execute
Sample
Return OpenAI-compatible results。
```

---

# 160. MLC 不只是 Compiler

这是重要边界：

```text
MLC-LLM
=
Compiler
+
Runtime / Serving Engine。
```

---

# 161. MLCEngine OpenAI-compatible API

MLC Python API：

```text
Chat Completions
Completions
```

设计成：

```text
OpenAI API style。
```

这使上层：

```text
Agent / RAG
```

易于接入。

---

# 162. MLCEngine Modes

当前源码中：

```text
local
interactive
server
```

三种预设模式。

---

# 163. `interactive`

适合：

```text
1 concurrent request。
```

目标：

```text
个人交互。
```

---

# 164. `local`

适合：

```text
低并发本地服务。
```

默认：

```text
小 Batch。
```

---

# 165. `server`

适合：

```text
较多并发 Request。
```

Runtime 会：

```text
尽量利用 GPU Memory
推断更大的 Batch / KV Capacity。
```

---

# 166. 说明 MLC 也在解决 Serving

并不是：

```text
编译完一个静态 Graph
然后只跑单用户。
```

MLCEngine 也有：

```text
Scheduler
KV Cache
Request Management。
```

---

# 167. MLC 的 PagedKVCache

MLC Model Source 中：

```python
PagedKVCache
```

直接出现在：

```text
Llama Attention / Model Spec。
```

---

# 168. 为什么编译路线也需要 Paged KV？

因为问题来自：

```text
LLM Serving 本身。
```

不是 vLLM 独有。

```text
Variable-length requests
Dynamic Context
Batch
```

都需要：

```text
灵活 KV 管理。
```

---

# 169. MLC Llama Model Spec

当前源代码中可以看到：

```text
batch_prefill
batch_decode
batch_verify
create_paged_kv_cache
```

这些函数：

> 在编译阶段就被定义成明确 Entry Point。

---

# 170. 这和 llama.cpp 很不一样

llama.cpp：

```text
llama_decode
```

统一入口，Runtime 动态建 Graph。

MLC：

```text
batch_prefill
batch_decode
batch_verify
```

等：

```text
Compiler-visible Function。
```

---

# 171. 为什么这样好？

Compiler 可以针对：

```text
Prefill Shape
Decode Shape
Verify Shape
```

分别：

```text
生成最合适代码。
```

---

# 172. Speculative Decode

`batch_verify`：

```text
就是 Speculative Decode
```

主模型批量验证候选 Token 的重要入口。

---

# 173. MLC Compiler 能看到更多全局信息

因为它处理：

```text
完整模型 IR。
```

可以做：

```text
Operator Fusion
Memory Planning
Kernel Selection
Target-specific Optimization。
```

---

# 174. MLC 与 TVM 的关系

可以画：

```text
MLC-LLM
│
├── LLM Model Definitions
├── Weight Conversion
├── Quantization
├── LLM-specific Compiler Pass
├── Serving Engine
└── App APIs
        │
        ▼
      TVM
│
├── Relax
├── TIR
├── Target
├── Scheduling
├── Codegen
└── Runtime
```

---

# 175. MLC 是 TVM 的 Vertical Application

可以理解：

> **MLC-LLM 是基于 TVM 打造的 LLM 专用 Compiler + Runtime Stack。**

---

# 176. 为什么这章后面自然进入 TVM？

因为看到：

```text
Relax
IRModule
Compiler Pass
Target
TIR
Codegen
```

以后，自然会问：

```text
这些到底是什么？
```

这就是：

> AI Compiler。

---

# 177. MLC 推荐源码阅读顺序

第一阶段：

```text
README / Docs
```

---

# 178. 第二阶段：Model Definition

例如：

- [python/mlc_llm/model/llama/llama_model.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/llama/llama_model.py)

重点：

```text
Config
Attention
DecoderLayer
LlamaModel
LlamaForCausalLM
ModuleSpec。
```

---

# 179. 第三阶段：Quantization

- [python/mlc_llm/quantization/quantization.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/quantization/quantization.py)

看：

```text
q0f16
q4f16_1
AWQ
Group Quant。
```

---

# 180. 第四阶段：Weight Loader

看：

```text
ExternMapping
QuantizeMapping。
```

理解：

```text
HF Weight
→
MLC Parameter。
```

---

# 181. 第五阶段：Compile Interface

重点：

```text
mlc_llm compile
```

对应：

```text
Python compile interface。
```

---

# 182. 第六阶段：Compiler Pipeline

- [compiler_pass/pipeline.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler_pass/pipeline.py)

观察：

```text
Pass 顺序。
```

---

# 183. 第七阶段：Serving Engine

- [python/mlc_llm/serve/engine.py](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/serve/engine.py)
- [cpp/serve/engine.cc](https://github.com/mlc-ai/mlc-llm/blob/main/cpp/serve/engine.cc)

理解：

```text
Python API
↓
C++ Engine
↓
Compiled Functions。
```

---

# 184. 第八阶段：TVM

进入：

```text
Relax
TIR
Target
Schedule
DLight
Codegen。
```

---

# 185. MLC 必做实验 1

准备小模型。

运行：

```text
mlc_llm chat
```

先体验：

```text
JIT Compile。
```

---

# 186. 实验 2：Weight Conversion

```text
HF FP16
↓
q4f16_1
```

记录：

```text
Model Size。
```

---

# 187. 实验 3：AOT Compile

同一 Model：

```text
CUDA
```

生成：

```text
.so。
```

---

# 188. 实验 4：Vulkan Compile

如果机器支持：

```text
同一个模型
```

编译：

```text
Vulkan library。
```

---

# 189. 观察重点

Weight：

```text
可共享。
```

Model Library：

```text
不同 Target。
```

---

# 190. 实验 5：MLCEngine

自己写：

```python
from mlc_llm import MLCEngine
```

使用：

```text
Chat Completion。
```

---

# 191. 实验 6：Mode

比较：

```text
interactive
local
server
```

观察：

```text
Max Batch
Memory
Throughput。
```

---

# 192. 实验 7：Compile Log

开启更详细日志。

观察：

```text
Relax
TIR
Target
Pass
Build。
```

---

# 193. 实验 8：Print IR

如果准备进入 Compiler：

```text
导出 / 打印 IRModule。
```

看看：

```text
Llama Python Model
```

如何变成：

```text
Relax Function。
```

---

# 194. 实验 9：读 TIR

选一个：

```text
MatMul / RMSNorm
```

Lower 后的：

```text
TIR PrimFunc。
```

先不优化。

只理解：

```text
Loop + Buffer。
```

---

# 195. 实验 10：跨平台

如果有条件：

```text
Windows/Linux GPU
↓
Vulkan

Android
↓
OpenCL/Vulkan

Browser
↓
WebGPU
```

至少体验两种 Target。

---

# 196. 第三部分：三大 Runtime 深度对比

---

# 197. Model Input Format

## llama.cpp

```text
GGUF。
```

---

## vLLM

主要：

```text
Hugging Face-compatible model / weights。
```

当前也支持：

```text
GGUF 等多种量化 / load 形式，
```

但其核心生态仍是：

```text
HF + GPU Serving。
```

---

## MLC

```text
HF Model
↓
Convert Weight
↓
MLC Weight
+
Compiled Model Library。
```

---

# 198. Graph Representation

llama.cpp：

```text
Runtime 依据 architecture
动态构建 ggml Graph。
```

vLLM：

```text
PyTorch Model / vLLM Model Implementation
+
torch.compile / optimized execution。
```

MLC：

```text
Model
↓
Relax IR
↓
TIR
↓
Compiled Target Code。
```

---

# 199. Runtime Philosophy

```text
llama.cpp
=
Runtime-centric

vLLM
=
Serving-centric

MLC
=
Compiler-centric
```

---

# 200. Quantization Philosophy

llama.cpp：

```text
GGUF quant
Q4_K_M / IQ...
```

vLLM：

```text
AWQ
GPTQ
FP8
INT8
INT4
GGUF
compressed-tensors
...
```

主要：

```text
使用硬件高性能 Kernel。
```

MLC：

```text
q4f16_1
q4f16_awq
...
```

量化和：

```text
Compiler / Model Transformation
```

结合。

---

# 201. Edge Friendliness

llama.cpp：

```text
★★★★★
```

尤其：

```text
CPU
ARM
local binary
GGUF。
```

---

# 202. vLLM Edge Friendliness

传统定位：

```text
★★
```

更适合：

```text
NVIDIA / AMD GPU Server。
```

虽然当前硬件支持不断扩展，但学习主线仍应：

```text
GPU Serving。
```

---

# 203. MLC Edge Friendliness

```text
★★★★★
```

特别是：

```text
Android
iOS
Web
Vulkan
Metal
OpenCL。
```

---

# 204. Server Throughput

vLLM：

```text
★★★★★
```

是其主战场。

---

# 205. llama.cpp Server

可以：

```text
Serve。
```

但核心优势：

```text
不是数据中心超高并发 GPU Serving。
```

---

# 206. MLC Server

也有：

```text
MLCEngine server mode。
```

但 MLC 最独特优势仍然：

```text
Compiler + Universal Deployment。
```

---

# 207. Source Code Learning Value

llama.cpp：

```text
学习：
C++ Runtime
Tensor
Backend
Quant Kernel。
```

---

# 208. vLLM

学习：

```text
Serving Systems
Scheduler
Memory Manager
Distributed Systems。
```

---

# 209. MLC

学习：

```text
AI Compiler
IR
Target
Codegen
Cross-platform Runtime。
```

---

# 210. 三者和用户路线的关系

```text
llama.cpp
↓
理解 Runtime

vLLM
↓
理解 Serving

MLC-LLM
↓
理解 Compiler

TVM
↓
深入 AI Compiler

FlashAttention / CUTLASS
↓
深入 Kernel

RKLLM
↓
进入 Vendor NPU。
```

---

# 211. llama.cpp vs vLLM：最核心区别

llama.cpp 最先问：

```text
怎样在这台机器上
尽可能轻量地跑模型？
```

vLLM 最先问：

```text
怎样让 GPU 同时高效服务
大量动态 Request？
```

---

# 212. vLLM 为什么不是“更快的 llama.cpp”？

因为它们：

```text
Workload Assumption
```

不同。

llama.cpp 常见：

```text
Batch 1
Local Chat
CPU / Consumer GPU。
```

vLLM 常见：

```text
Many Concurrent Requests
Multi-GPU
Server。
```

---

# 213. MLC 为什么不是“编译版 llama.cpp”？

MLC：

```text
有独立 IR / Compiler Pipeline / Model Library。
```

它的系统哲学：

```text
从模型程序开始做编译。
```

而 llama.cpp：

```text
更多依赖预实现 architecture + backend。
```

---

# 214. vLLM 与 MLC 的共同点

二者都有：

```text
Paged KV
Serving Engine
Batch
Scheduler
OpenAI-style API
Optimized GPU Path。
```

---

# 215. 但不同点

vLLM：

```text
Serving Scheduler 是灵魂。
```

MLC：

```text
Compiler / Target Codegen 是最大特色。
```

---

# 216. MLC 也能做 Server，vLLM 也有 Compiler 技术

所以：

> 不要把项目贴成绝对单一标签。

这里说的是：

```text
最核心设计重心。
```

---

# 217. 一个选型表

| 场景 | 更优先考虑 |
|---|---|
| CPU 本地推理 | llama.cpp |
| ARM Linux 本地 LLM | llama.cpp |
| GGUF 生态 | llama.cpp |
| NVIDIA 多用户 Serving | vLLM |
| 高吞吐 API Server | vLLM |
| Multi-GPU 大模型 Serving | vLLM |
| Paged KV / Scheduler 学习 | vLLM |
| Android GPU Native LLM | MLC-LLM |
| iOS / Metal Native | MLC-LLM |
| Browser WebGPU | MLC / WebLLM |
| 学 AI Compiler | MLC → TVM |
| Rockchip RKNPU | RKLLM / RKNN-LLM |

---

# 218. 在 RK3576 上怎么理解这三条路线？

---

# 219. llama.cpp

```text
RK3576
↓
Ubuntu
↓
ARM CPU
↓
llama.cpp
```

适合作为：

```text
Baseline
学习
CPU 推理。
```

---

# 220. MLC

理论可探索：

```text
RK3576 GPU
↓
Vulkan / OpenCL
↓
MLC-LLM
```

前提：

```text
GPU Driver
Compiler Target
Runtime Backend
Operator Support
```

均满足。

---

# 221. 但是 MLC 不等于 Rockchip NPU

当前通用 MLC：

```text
Vulkan / OpenCL
```

主要针对：

```text
GPU。
```

Rockchip NPU：

```text
RKNPU
```

需要：

```text
Vendor Compiler / Runtime。
```

---

# 222. 所以真正 NPU 路线

```text
HF Model
↓
RKLLM Toolchain
↓
RKLLM Model
↓
RKLLM Runtime
↓
RKNPU。
```

---

# 223. vLLM 在 RK3576 呢？

不是主要路线。

因为 vLLM 的主设计场景：

```text
高性能 Server Accelerator
Large GPU Memory
Concurrent Serving。
```

RK3576：

```text
Edge SoC
Shared DDR
ARM CPU
Embedded GPU
RKNPU。
```

Workload 完全不同。

---

# 224. 因此学习 vLLM 的目的

不是：

```text
把 vLLM 装到 RK3576。
```

而是：

> **理解高性能 LLM Serving 的系统设计。**

---

# 225. 学习 MLC 的目的

对 Edge 方向更直接：

```text
理解：
模型怎样从 Python
进入 Compiler IR
最后变成设备 Native Library。
```

---

# 226. 这正是进入 AI Compiler 的桥

在 MLC 中已经出现：

```text
Relax
TIR
Compiler Pass
Target
Schedule
Codegen。
```

下一步：

```text
TVM。
```

---

# 227. vLLM 性能 Debug 思路

如果：

```text
TTFT 高
```

检查：

```text
Queue
Prefix Cache
Prompt Length
Chunked Prefill
Prefill Kernel
Scheduler Budget。
```

---

# 228. 如果 ITL 高

检查：

```text
Decode Batch
Long Prefill interference
Scheduler
KV Memory
Kernel
GPU Utilization。
```

---

# 229. 如果 Throughput 低

检查：

```text
Concurrency
max_num_seqs
max_num_batched_tokens
KV Capacity
Tensor Parallel
Kernel
Batch Utilization。
```

---

# 230. 如果频繁 Preemption

检查：

```text
KV Cache Size
max_model_len
Concurrent Requests
Watermark
Admission。
```

---

# 231. 如果 Prefix Cache Hit 低

检查：

```text
Prompt 是否真的共享 Prefix
Chat Template
Tokenizer
Block Boundary
LoRA / Multimodal Context
Hash。
```

---

# 232. MLC 性能 Debug 思路

如果：

```text
Compile 失败
```

检查：

```text
Model Architecture
TVM Version
Target
Operator
Compiler Pass
Toolchain。
```

---

# 233. 如果 Runtime Load 失败

检查：

```text
Model Library
Weight Config
Target Runtime
ABI
Device。
```

---

# 234. 如果能跑但慢

检查：

```text
Target 选错
Quantization
Kernel Schedule
Prefill Chunk
KV
Driver
Generated Code。
```

---

# 235. 如果不同 Target 表现差异大

这是正常的。

因为：

```text
CUDA
Metal
Vulkan
```

拥有：

```text
不同硬件
不同 Kernel
不同 Compiler Schedule。
```

---

# 236. 本章建议学习顺序

---

## Stage 1：复习 06

必须先会：

```text
Prefill
Decode
KV Cache
TTFT
TPOT
Batch。
```

---

## Stage 2：vLLM 外部使用

```text
LLM()
vllm serve
OpenAI API。
```

---

## Stage 3：Continuous Batching

理解：

```text
为什么 Batch 每 Step 都会变化。
```

---

## Stage 4：Scheduler

重点读：

```text
V1 schedule()
Token Budget。
```

---

## Stage 5：Paged KV

理解：

```text
Block
Block Table
KVCacheManager。
```

---

## Stage 6：Prefix Cache

理解：

```text
Block Hash
Reuse。
```

---

## Stage 7：Chunked Prefill

自己做：

```text
长 Prompt + Decode 并发实验。
```

---

## Stage 8：Parallelism

理解：

```text
TP
PP
DP
EP。
```

---

## Stage 9：MLC Quick Start

体验：

```text
JIT compile。
```

---

## Stage 10：Weight / Config / Compile

亲手：

```text
convert_weight
gen_config
compile。
```

---

## Stage 11：读 MLC Model

看：

```text
Llama Model
PagedKVCache
ModuleSpec。
```

---

## Stage 12：读 Compiler Pass

看：

```text
IRModule
Relax Pass。
```

---

## Stage 13：读 MLCEngine

理解：

```text
Compiled Model
怎样被 Serving Runtime 驱动。
```

---

## Stage 14：进入 TVM

下一章：

```text
AI Compiler / TVM。
```

---

# 237. 本章必做实验清单

## vLLM

- [ ] 安装并运行 vLLM
- [ ] 使用 `LLM.generate`
- [ ] 启动 `vllm serve`
- [ ] 用 OpenAI SDK 调用
- [ ] 测 Concurrency 1/2/4/8
- [ ] 记录 TTFT
- [ ] 记录 TPOT
- [ ] 记录 ITL
- [ ] 记录 Output Throughput
- [ ] 修改 `max_num_seqs`
- [ ] 修改 `max_num_batched_tokens`
- [ ] 测 Long Prompt
- [ ] 测 Chunked Prefill
- [ ] 测 Prefix Cache
- [ ] 阅读 V1 Scheduler
- [ ] 阅读 KVCacheManager
- [ ] 能画出 Block Table
- [ ] 有多 GPU 时测试 TP

## MLC

- [ ] 安装 MLC-LLM
- [ ] 运行 `mlc_llm chat`
- [ ] 观察 JIT Compile
- [ ] `convert_weight`
- [ ] 理解 `q4f16_1`
- [ ] 生成 `mlc-chat-config.json`
- [ ] 运行 `mlc_llm compile`
- [ ] 找到生成的 Model Library
- [ ] 使用 MLCEngine
- [ ] 阅读 Llama Model Definition
- [ ] 找到 `PagedKVCache`
- [ ] 阅读 Compiler Pipeline
- [ ] 阅读 `attach_sampler`
- [ ] 阅读 C++ Serve Engine
- [ ] 最好尝试两个不同 Target
- [ ] 打印一次 Relax IR

---

# 238. 常见误区

---

## 误区 1

```text
vLLM = 一个更快的 Hugging Face generate
```

不完整。

vLLM 的核心价值：

```text
Serving Runtime + Scheduling + KV Management。
```

---

## 误区 2

```text
PagedAttention = 新 Attention 算法
```

不是重点。

核心：

```text
Paged KV Memory Management。
```

---

## 误区 3

```text
Continuous Batching = 把 Batch Size 调大
```

错误。

它意味着：

```text
Batch 成员可以在每个 Decode Iteration 动态变化。
```

---

## 误区 4

```text
Prefix Cache = KV Cache
```

不完全。

KV Cache：

```text
当前请求历史状态。
```

Prefix Cache：

```text
跨请求复用部分 KV。
```

---

## 误区 5

```text
vLLM V1 仍严格分 Prefill Queue 和 Decode Queue
```

当前核心 Scheduler 的抽象已经更加统一：

```text
基于每个 Request 还需计算多少 Token。
```

---

## 误区 6

```text
MLC-LLM = TVM
```

错误。

MLC-LLM：

```text
基于 TVM 的 LLM Compiler + Runtime Application Stack。
```

TVM：

```text
更通用的 ML Compiler Framework。
```

---

## 误区 7

```text
MLC Model = 一个类似 GGUF 的文件
```

不准确。

MLC 部署通常包括：

```text
Converted Weights
+
Compiled Model Library。
```

---

## 误区 8

```text
q4f16_1 = Q4_K_M
```

错误。

不同生态。

---

## 误区 9

```text
MLC compile 只是把文件格式转换一下
```

错误。

它包含真正：

```text
IR
Compiler Pass
Target Lowering
Code Generation。
```

---

## 误区 10

```text
MLC 可以跨平台，所以同一个 .so 到处跑
```

错误。

Weight 可在某些情况下共享。

但：

```text
Model Library
```

针对 Target 编译。

---

## 误区 11

```text
vLLM 只做 PagedAttention
```

早已不是。

现代 vLLM 包含：

```text
Scheduler
Continuous Batching
Prefix Cache
Chunked Prefill
Spec Decode
Parallelism
Kernel / Compile Integration。
```

---

## 误区 12

```text
MLC Vulkan = Rockchip NPU
```

完全错误。

Vulkan：

```text
GPU API。
```

RKNPU：

```text
专用 NPU。
```

---

# 239. 本章知识树

```text
LLM Runtime Systems
│
├── vLLM
│   │
│   ├── API
│   │   ├── LLM
│   │   └── OpenAI Server
│   │
│   ├── Engine
│   │   └── EngineCore
│   │
│   ├── Scheduler
│   │   ├── Token Budget
│   │   ├── max_num_batched_tokens
│   │   ├── max_num_seqs
│   │   ├── FCFS
│   │   ├── Priority
│   │   └── Preemption
│   │
│   ├── Batching
│   │   ├── Continuous Batching
│   │   └── Chunked Prefill
│   │
│   ├── KV Cache
│   │   ├── Block
│   │   ├── Block Table
│   │   ├── KVCacheManager
│   │   ├── Paged Attention
│   │   └── Prefix Cache
│   │
│   ├── Execution
│   │   ├── Model Runner
│   │   ├── FlashAttention
│   │   ├── FlashInfer
│   │   ├── CUTLASS
│   │   └── torch.compile
│   │
│   └── Distributed
│       ├── TP
│       ├── PP
│       ├── DP
│       └── EP
│
└── MLC-LLM
    │
    ├── Model
    │   ├── Relax Frontend nn
    │   ├── Llama Model
    │   ├── PagedKVCache
    │   └── ModuleSpec
    │
    ├── Weight
    │   ├── HF Mapping
    │   ├── convert_weight
    │   └── q4f16_1
    │
    ├── Config
    │   └── mlc-chat-config.json
    │
    ├── Compiler
    │   ├── IRModule
    │   ├── Relax
    │   ├── Compiler Pass
    │   ├── TIR
    │   ├── Target
    │   └── Codegen
    │
    ├── Artifact
    │   └── Model Library
    │       ├── .so
    │       ├── .dylib
    │       ├── .dll
    │       ├── .wasm
    │       └── .tar
    │
    ├── Engine
    │   ├── MLCEngine
    │   ├── AsyncMLCEngine
    │   ├── local
    │   ├── interactive
    │   └── server
    │
    └── Target
        ├── CUDA
        ├── ROCm
        ├── Vulkan
        ├── Metal
        ├── WebGPU
        └── OpenCL
```

---

# 240. 本章最重要的思维模型

## 思维模型 1

```text
vLLM
=
LLM Serving System
```

---

## 思维模型 2

```text
High Throughput
不仅靠更快 Kernel

还靠：
Scheduler
Batching
KV Management
```

---

## 思维模型 3

```text
Continuous Batching
=
Batch Membership
can change
every iteration
```

---

## 思维模型 4

```text
PagedAttention
核心：
KV Cache Page / Block Management
```

---

## 思维模型 5

```text
V1 Scheduler
=
Allocate Token Budget
to Requests
per Engine Step
```

---

## 思维模型 6

```text
MLC-LLM
=
LLM
+
ML Compiler
+
Runtime
```

---

## 思维模型 7

```text
MLC Deployment
=
Converted Weights
+
Compiled Model Library
```

---

## 思维模型 8

```text
Model
↓
Relax IR
↓
Compiler Pass
↓
TIR / Target
↓
Native Code
```

---

## 思维模型 9

```text
llama.cpp
Runtime-centric

vLLM
Serving-centric

MLC
Compiler-centric
```

---

## 思维模型 10

```text
通用 GPU Runtime
≠
Vendor NPU Runtime
```

---

# 241. 本章完成标准

如果下面大部分都可以自己解释，本章就可以结束。

## vLLM 基础

- [ ] 能解释 vLLM 的定位
- [ ] 能解释为什么 HF `generate` 不够做大规模 Serving
- [ ] 能画出 vLLM Engine 结构
- [ ] 知道当前应以 V1 为主线
- [ ] 能解释 EngineCore
- [ ] 能解释 Model Runner

## Scheduler

- [ ] 能解释 Scheduler 调度的是 Token Work
- [ ] 能解释 `max_num_batched_tokens`
- [ ] 能解释 `max_num_seqs`
- [ ] 能解释 V1 unified scheduling
- [ ] 能解释 FCFS / Priority
- [ ] 能解释 Admission
- [ ] 能解释 Preemption
- [ ] 能解释 Watermark

## Batching

- [ ] 能解释 Static Batching
- [ ] 能解释 Continuous Batching
- [ ] 能解释 Iteration-level Scheduling
- [ ] 能解释为什么输出长度不同会浪费 Static Batch
- [ ] 能解释 Chunked Prefill
- [ ] 能解释 Chunked Prefill 的 TTFT / ITL Tradeoff

## Paged KV

- [ ] 能解释为什么 KV Cache 需要 Block 管理
- [ ] 能解释 Logical Block
- [ ] 能解释 Physical Block
- [ ] 能解释 Block Table
- [ ] 能解释 PagedAttention
- [ ] 能解释 KVCacheManager
- [ ] 能解释 Hybrid KV Cache
- [ ] 能解释 Fragmentation

## Prefix Cache

- [ ] 能解释 Prefix Cache
- [ ] 能解释 Prefix Block Hash
- [ ] 能解释为什么 Hash 需要包含 Prefix
- [ ] 能解释 Cache Hit 对 TTFT 的影响
- [ ] 能区分 Prefix Cache 与普通 KV Cache

## Serving

- [ ] 能启动 `vllm serve`
- [ ] 能使用 OpenAI API Client
- [ ] 能解释 TTFT
- [ ] 能解释 TPOT
- [ ] 能解释 ITL
- [ ] 能解释 Throughput
- [ ] 能完成一次 Concurrency Benchmark

## Parallelism

- [ ] 能解释 Tensor Parallel
- [ ] 能解释 Pipeline Parallel
- [ ] 能解释 Data Parallel
- [ ] 能解释 Expert Parallel
- [ ] 能解释 TP Communication
- [ ] 能解释 MoE All-to-All

## MLC 基础

- [ ] 能解释 MLC-LLM 的定位
- [ ] 能解释 MLC 与 TVM 的关系
- [ ] 能解释 Converted Weight
- [ ] 能解释 Model Library
- [ ] 能区分 JIT / AOT Compile
- [ ] 能解释 Cross Compilation

## MLC Workflow

- [ ] 能使用 `convert_weight`
- [ ] 能解释 `gen_config`
- [ ] 能解释 `mlc-chat-config.json`
- [ ] 能使用 `mlc_llm compile`
- [ ] 能找到编译后的 `.so/.dll/...`
- [ ] 能运行 MLCEngine

## MLC Compiler

- [ ] 能解释 Relax
- [ ] 能解释 IRModule
- [ ] 知道 TIR 的位置
- [ ] 能解释 Compiler Pass
- [ ] 能解释 Target
- [ ] 能解释 Codegen
- [ ] 看过 `compiler_pass/pipeline.py`
- [ ] 看过一个 MLC Model Definition

## MLC Runtime

- [ ] 能解释 MLCEngine
- [ ] 能解释 local mode
- [ ] 能解释 interactive mode
- [ ] 能解释 server mode
- [ ] 能解释 MLC PagedKVCache
- [ ] 能解释 batch_prefill / batch_decode
- [ ] 能解释 batch_verify

## 对比

- [ ] 能解释 llama.cpp vs vLLM
- [ ] 能解释 llama.cpp vs MLC
- [ ] 能解释 vLLM vs MLC
- [ ] 能根据场景做基本选型
- [ ] 能解释为什么 vLLM 不适合作为 RK3576 主路线
- [ ] 能解释为什么 MLC GPU 路线不等于 RKNPU
- [ ] 能画出 RK3576 的 llama.cpp / MLC GPU / RKLLM NPU 三条路线

---

# 242. 本章暂时不要求深入

暂时不要求：

- [ ] 自己实现 vLLM Scheduler
- [ ] 自己实现 PagedAttention CUDA Kernel
- [ ] 精通所有 vLLM V1 代码
- [ ] 精通 Ray
- [ ] 精通 Multi-node Serving
- [ ] 精通 DeepEP
- [ ] 精通所有 MoE All-to-All Backend
- [ ] 自己实现 TVM Relax
- [ ] 自己实现 TIR Pass
- [ ] 自己写 CUDA Codegen
- [ ] 精通 MLC 所有 Target
- [ ] 自己实现新的 MLC Backend

这些属于后续系统 / Compiler 深入内容。

---

# 243. 核心参考链接

## vLLM

- [vLLM Repository](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM V1 Guide](https://docs.vllm.ai/en/stable/usage/v1_guide/)
- [Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization/)
- [Scheduler Config](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)
- [Scheduler Interface](https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/interface/)
- [Scheduler Source](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [EngineCore](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)
- [KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)
- [Hybrid KV Cache Design](https://github.com/vllm-project/vllm/blob/main/docs/design/hybrid_kv_cache_manager.md)
- [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- [Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM Serve](https://docs.vllm.ai/en/latest/cli/serve/)
- [Benchmark](https://docs.vllm.ai/en/latest/benchmarking/cli/)

---

## MLC-LLM

- [MLC-LLM Repository](https://github.com/mlc-ai/mlc-llm)
- [MLC LLM Documentation](https://llm.mlc.ai/)
- [Introduction](https://llm.mlc.ai/docs/get_started/introduction.html)
- [Quick Start](https://llm.mlc.ai/docs/get_started/quick_start.html)
- [Compile Model Libraries](https://llm.mlc.ai/docs/compilation/compile_models.html)
- [Package Libraries and Weights](https://llm.mlc.ai/docs/compilation/package_libraries_and_weights.html)
- [Python MLCEngine](https://llm.mlc.ai/docs/deploy/python_engine.html)
- [Install TVM](https://github.com/mlc-ai/mlc-llm/blob/main/docs/install/tvm.rst)
- [Llama Model Source](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/model/llama/llama_model.py)
- [Quantization Source](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/quantization/quantization.py)
- [Compiler Pipeline](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler_pass/pipeline.py)
- [GPU Sampler Compiler Pass](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/compiler_pass/attach_sampler.py)
- [Python Serving Engine](https://github.com/mlc-ai/mlc-llm/blob/main/python/mlc_llm/serve/engine.py)
- [C++ Serving Engine](https://github.com/mlc-ai/mlc-llm/blob/main/cpp/serve/engine.cc)

---

# 244. 推荐源码路线：vLLM

```text
V1 Guide
   ↓
LLM / serve API
   ↓
EngineCore
   ↓
Scheduler
   ↓
KVCacheManager
   ↓
Model Runner
   ↓
Attention Backend
   ↓
GPU Kernel
```

重点：

> **先学 Scheduler，再读 Kernel。**

---

# 245. 推荐源码路线：MLC

```text
Introduction
   ↓
Llama Model
   ↓
Quantization
   ↓
Weight Loader
   ↓
Compile Interface
   ↓
Compiler Pipeline
   ↓
Relax IR
   ↓
MLCEngine
   ↓
C++ Engine
   ↓
TVM / TIR
```

重点：

> **先看模型如何成为 IR，再进入 TVM。**

---

# 246. 与 06_LLM推理原理 的映射

```text
06 理论                 vLLM

Prefill        → Chunked Prefill
Decode         → Scheduled token step
KV Cache       → KVCacheManager
Context        → KV Blocks
Batch          → Continuous Batching
TTFT           → Scheduler / Prefix / Prefill
ITL            → Decode Priority
Serving        → EngineCore
```

---

# 247. 与 06 的 MLC 映射

```text
06 理论                 MLC

Prefill        → batch_prefill
Decode         → batch_decode
KV Cache       → PagedKVCache
Spec Decode    → batch_verify
Model Graph    → Relax
Kernel         → TIR / Target Code
Runtime        → MLCEngine
```

---

# 248. 与 07_llama.cpp 的映射

```text
llama.cpp               vLLM

llama_context      ↔    Engine Request State
KV Memory          ↔    KVCacheManager
n_batch            ↔    Token Budget / Batch
PP Graph           ↔    Prefill Work
TG Graph           ↔    Decode Work
Backend            ↔    GPU Model Runner / Kernel
llama-server       ↔    vllm serve
```

---

# 249. 与 MLC 的映射

```text
llama.cpp                   MLC

GGUF                    ↔   Converted Weights + Config
Architecture code       ↔   Relax Frontend Model
ggml Graph              ↔   Relax IR
Backend Kernel          ↔   TIR / Target Code
llama_context           ↔   MLCEngine State
llama_decode            ↔   compiled prefill/decode funcs
```

---

# 250. 三种路线最终总图

```text
                          HF Model
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
      llama.cpp             vLLM             MLC-LLM
          │                  │                  │
       GGUF               HF Weight          convert_weight
          │                  │                  │
       ggml              PyTorch Model      MLC Model
          │                  │                  │
    Runtime Graph         torch.compile       Relax IR
          │                  │                  │
     Backend Op         Scheduler / KV     Compiler Pass
          │                  │                  │
 Quant / CPU/GPU       CUDA/HIP Kernel        TIR
          │                  │                  │
 Local / Edge         GPU Serving          Target Code
                                               │
                                               ▼
                                  CUDA / Vulkan / Metal /
                                  WebGPU / OpenCL / ...
```

---

# 251. 从职业技术栈看这三者

如果目标偏：

```text
Embedded / Edge AI Runtime
```

优先：

```text
llama.cpp
+
MLC / TVM
+
RKNN/RKLLM。
```

---

# 252. 如果目标偏：

```text
LLM Infrastructure / Serving
```

优先：

```text
vLLM
+
CUDA
+
Distributed Systems
+
FlashAttention / CUTLASS。
```

---

# 253. 如果目标偏：

```text
AI Compiler
```

优先：

```text
MLC
↓
TVM
↓
Relax
↓
TIR
↓
GPU Kernel。
```

---

# 254. 对本学习路线最重要的连接

我们不是为了：

```text
把三个 Runtime 都背熟。
```

而是为了理解：

```text
同一个 LLM 推理问题
```

可以在三种抽象层解决。

---

# 255. Runtime-centric

```text
手写 Runtime
手写 Backend
手写 Quant Kernel
```

代表：

```text
llama.cpp。
```

---

# 256. Serving-centric

```text
Schedule Requests
Manage KV
Batch Dynamically
Scale GPUs
```

代表：

```text
vLLM。
```

---

# 257. Compiler-centric

```text
Model → IR → Optimize → Codegen
```

代表：

```text
MLC。
```

---

# 258. 这三层以后还会合流

未来高性能系统：

```text
Runtime
+
Scheduler
+
Compiler
+
Kernel
```

往往同时存在。

所以这些项目：

```text
边界正在不断融合。
```

---

# 259. 一句话总结 vLLM

```text
vLLM
不是靠“一个更快 Attention Kernel”
成为高吞吐 Runtime。

它的核心是：

Request
↓
Scheduler
↓
Continuous Batching
↓
Paged KV Cache
↓
Optimized Model Execution
↓
GPU
```

---

# 260. 一句话总结 MLC-LLM

```text
MLC-LLM
不是简单模型格式转换工具。

它的核心是：

Model
↓
Relax IR
↓
Compiler Pass
↓
Target-specific Code
↓
Model Library
↓
MLCEngine
↓
Hardware
```

---

# 261. 本章最终完成标准

最终如果可以不看笔记画出：

```text
vLLM:

API
↓
EngineCore
↓
Scheduler
↓
KVCacheManager
↓
Continuous Batch
↓
Model Runner
↓
GPU
```

以及：

```text
MLC:

HF
↓
convert_weight
↓
Model Definition
↓
Relax
↓
Compiler Pass
↓
TIR
↓
Model Library
↓
MLCEngine
↓
Target
```

并且可以解释：

```text
llama.cpp
vLLM
MLC-LLM
```

为什么不是简单的：

```text
“谁比谁快”
```

而是面向不同系统问题的设计。

那么这一章就完成了。

---

# 262. 下一阶段

到这里：

```text
06
LLM 推理原理

07
llama.cpp

08
vLLM + MLC-LLM
```

已经完成第二部分最重要的：

```text
LLM Runtime Systems
```

学习。

接下来正式进入第三部分：

```text
AI Compiler
↓
TVM
↓
Graph / IR
↓
Lowering
↓
TensorIR
↓
Schedule
↓
Codegen
↓
GPU Kernel
↓
Hardware
```

也就是：

> **09_AI_Compiler与TVM.md**

从这里开始，学习重点将从：

```text
“Runtime 怎么运行模型”
```

继续下钻到：

```text
“Compiler 怎么把一个模型算子真正变成硬件可执行的高性能代码”。
```
