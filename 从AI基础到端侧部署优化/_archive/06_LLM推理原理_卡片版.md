# 06_LLM推理原理

> **所属路线**：AI 学习路线 · 第二部分  
> **定位**：从“一个训练完成的 Decoder-only LLM”出发，深入理解模型加载、Prompt Processing、Prefill、KV Cache、Decode、Sampling、性能指标、内存占用与硬件瓶颈  
> **核心参考项目**：`ggml-org/llama.cpp` + `vllm-project/vllm` + Hugging Face Transformers + NVIDIA LLM Inference Optimization  
> **学习边界**：本章重点解决“LLM 推理为什么这样运行”；`llama.cpp`、`vLLM`、`MLC-LLM` 的具体源码和工程实现分别放在后续独立章节。  
> **资料检查日期**：2026-08-25

---

# 0. 本章学习目标

完成本章以后，需要能够回答：

1. 一个 `.safetensors` / GGUF 模型被 Runtime 加载后，内存里到底有什么？
2. LLM 推理为什么和 Training 完全不同？
3. Prompt 从字符串开始，经历哪些步骤才得到第一个输出 Token？
4. Tokenizer 在推理链路里处于什么位置？
5. 什么是 Prompt Processing / Prefill？
6. 什么是 Decode / Token Generation？
7. 为什么 Prefill 可以高度并行，而 Decode 必须自回归逐 Token 生成？
8. 为什么 Prefill 和 Decode 是两种完全不同的硬件 Workload？
9. 什么是 TTFT？
10. 什么是 TPOT / ITL？
11. 为什么只说“20 tokens/s”往往不够描述推理性能？
12. 什么是 KV Cache？
13. KV Cache 缓存的是哪一层、哪些 Tensor？
14. 为什么缓存 K/V，而不是缓存 Q？
15. 不使用 KV Cache 会重复计算什么？
16. KV Cache 的内存大小由哪些参数决定？
17. MHA / GQA / MQA 为什么会直接影响 KV Cache 大小？
18. 为什么 Context Length 越大，推理内存占用越高？
19. 为什么 Decode 通常是 Memory-Bound？
20. 为什么 Prefill 通常更接近 Compute-Bound？
21. 什么是 Arithmetic Intensity？
22. Roofline Model 在理解 LLM 推理时有什么价值？
23. 为什么 INT4 量化不仅减少模型大小，还可能提升 Decode Tokens/s？
24. 为什么量化不一定同样大幅提高 Prefill？
25. Weight Memory、KV Cache、Activation、Workspace 分别是什么？
26. 为什么“模型文件只有 1.2GB”不代表运行时只需要 1.2GB RAM？
27. Batch Size 对 Prefill 和 Decode 有什么不同影响？
28. 单用户本地推理和服务器多用户 Serving 有什么不同？
29. Continuous Batching 为什么对服务器推理很重要？
30. Paged KV Cache / PagedAttention 想解决什么问题？
31. Prefix Cache / Prompt Cache 是什么？
32. Chunked Prefill 是什么？
33. Speculative Decoding 的基本思想是什么？
34. FlashAttention 在 Prefill / Attention 中主要优化什么？
35. Sampling 属于模型 Forward 还是 Forward 之后的逻辑？
36. Greedy、Temperature、Top-K、Top-P 的成本和行为有什么区别？
37. 为什么 Tokenizer 不同的两个模型不能简单只比较 Tokens/s？
38. CPU、GPU、NPU 在 LLM 推理中分别受哪些瓶颈限制？
39. 为什么内存带宽对 RK3576 这类端侧 SoC 特别重要？
40. llama.cpp 中的 `pp` / `tg` benchmark 对应什么？
41. vLLM 为什么需要专门管理 KV Cache 与 Scheduler？
42. 后面的量化、Runtime、Compiler、Kernel 优化到底都在优化这一条链中的哪一步？

本章最终需要建立如下完整推理链：

```text
User Text
   ↓
Chat Template
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Model Runtime
   ↓
Prefill
   │
   ├── Embedding
   ├── Transformer × N
   ├── Build KV Cache
   └── Last-position Logits
   ↓
Sampling
   ↓
First Output Token
   ↓
Decode Loop
   │
   ├── New Token Embedding
   ├── New Q/K/V
   ├── Append K/V Cache
   ├── Attention over Past KV
   ├── FFN
   ├── Logits
   └── Sampling
   ↓
Next Token
   ↓
Repeat
```

---

# 1. 本章对应的核心资料

本章不是某一个仓库的“教程复述”。

它主要把几个高价值 Runtime / Framework 中反复出现的概念抽出来。

---

## 1.1 llama.cpp

GitHub：

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

重点源码 / 文档：

- [llama.cpp CLI README](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
- [KV Cache Header](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.h)
- [llama-bench Source](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)
- [Batched Bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md)

llama.cpp 中非常值得关注的几个词：

```text
Prompt Processing
Token Generation
KV Cache
Batch
UBatch
Context Size
Flash Attention
Cache Type K
Cache Type V
GPU Offload
```

本章先理解原理。

后续：

> `07_llama.cpp.md`

再深入源码。

---

# 2. vLLM

GitHub：

- [vllm-project/vllm](https://github.com/vllm-project/vllm)

官方文档：

- [vLLM Documentation](https://docs.vllm.ai/)
- [Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [Paged Attention / KV Cache related source](https://docs.vllm.ai/en/latest/)

vLLM 让我们进一步看到：

```text
Single Request Inference
        ↓
Many Concurrent Requests
        ↓
KV Cache Management
        ↓
Scheduler
        ↓
Continuous Batching
        ↓
Paged Attention
```

所以本章会先建立：

> 为什么 Serving Runtime 需要这些机制。

---

# 3. Hugging Face Transformers KV Cache

官方：

- [Transformers Cache Strategies](https://huggingface.co/docs/transformers/kv_cache)
- [Caching Explanation](https://huggingface.co/docs/transformers/main/cache_explanation)

这里非常适合学习：

```text
Dynamic Cache
Static Cache
Quantized Cache
Cache Offloading
```

以及：

```text
为什么 K/V 可以复用。
```

---

# 4. NVIDIA LLM Inference Optimization

推荐：

- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [Hardware-Friendly LLM Design](https://developer.nvidia.com/blog/ai-model-co-design-hardware-friendly-llm-design/)

这类资料非常适合建立：

```text
Compute-Bound
Memory-Bound
Arithmetic Intensity
Memory Bandwidth
Batch
```

之间的关系。

---

# 5. 从 Training 切换到 Inference

前面已经学过 Training：

```text
Input
↓
Forward
↓
Loss
↓
Backward
↓
Gradient
↓
Optimizer
↓
Update Weight
```

Inference：

```text
Input
↓
Forward
↓
Output
```

没有：

```text
Backward
Gradient
Optimizer
Parameter Update
```

所以：

> LLM 推理只需要执行模型的 Forward Graph。

---

# 6. Training Runtime 需要什么？

Training 时内存通常包括：

```text
Model Weights
+
Gradients
+
Optimizer State
+
Forward Activations
+
Temporary Workspace
```

例如 Adam / AdamW：

```text
除了 Weight
还需要 Momentum 等 Optimizer State
```

---

# 7. Inference Runtime 需要什么？

LLM Inference 主要：

```text
Model Weights
+
KV Cache
+
Temporary Activations
+
Workspace
+
Runtime Metadata
```

因此：

> 推理内存虽然比 Training 少很多，但并不等于模型文件大小。

---

# 8. 一个 LLM Runtime 的基本职责

一个 Runtime 至少需要做：

```text
Load Model
↓
Allocate Memory
↓
Tokenize Input
↓
Build / Execute Graph
↓
Maintain KV Cache
↓
Generate Logits
↓
Sample Token
↓
Update Sequence
↓
Repeat
```

高性能 Server Runtime 还需要：

```text
Batch
Schedule
Cache Management
Parallelism
Request Queue
Streaming
```

---

# 9. 从用户输入开始

用户输入：

```text
“解释一下 RK3576 的 NPU。”
```

真正送进模型前通常先：

```text
System Prompt
+
Conversation History
+
User Message
```

经过：

```text
Chat Template
```

形成完整 Prompt。

---

# 10. Chat Template

例如概念上：

```text
<system>
You are a helpful assistant.
</system>

<user>
解释一下 RK3576 的 NPU。
</user>

<assistant>
```

然后：

```text
Tokenizer
```

将其转成 Token ID。

---

# 11. Tokenization 是推理的一部分，但不是 Transformer Forward

链路：

```text
String
↓
Tokenizer
↓
Token IDs
↓
Transformer Runtime
```

所以：

```text
Tokenizer Time
```

通常不属于模型 Kernel 本身。

---

# 12. 为什么 Benchmark 要明确是否包含 Tokenization？

例如：

```text
llama-bench
```

主要关注：

```text
Prompt Processing
Token Generation
```

而某些 Benchmark 可能还包含：

```text
Tokenization
HTTP
Queue
Streaming
Sampling
Detokenization
```

如果测量范围不同：

> 两个数字不能直接比较。

---

# 13. 模型加载

假设：

```text
Qwen 2B
```

Weight 文件在：

```text
SSD
```

Runtime 首先：

```text
Read / mmap
↓
Map Tensor
↓
Allocate Backend Buffers
↓
Load / Map Weights
```

到：

```text
System RAM
GPU VRAM
NPU accessible memory
```

---

# 14. Model Loading 不等于 Inference

模型启动时间可能包括：

```text
Disk IO
Model Parsing
Memory Mapping
Weight Upload
Kernel Initialization
Graph Compilation
```

这些通常是：

```text
Cold Start
```

成本。

---

# 15. Warm Model

模型已经：

```text
Resident in RAM / VRAM
```

再执行请求时：

```text
不需要重新加载 Weight
```

这才是正常在线服务性能测量的典型状态。

---

# 16. Memory Mapping

llama.cpp 常使用：

```text
mmap
```

思想：

> 将模型文件映射到进程地址空间，而不是一次性传统 read 全部复制。

这能：

```text
简化加载
利用 OS Page Cache
降低启动 Copy
```

但 Runtime 行为仍受：

```text
Page Fault
RAM
Swap
Storage Speed
```

影响。

---

# 17. 模型 Load 到底 Load 什么？

至少：

```text
Embedding Weight
Attention Weight
FFN Weight
Norm Weight
LM Head
Tokenizer Metadata
Architecture Metadata
```

对于 GGUF：

```text
Weight
+
Metadata
+
Tokenizer-related information
```

通常被统一打包。

---

# 18. Prefill 是什么？

Prefill：

> Runtime 第一次处理完整输入 Prompt 的阶段。

例如：

```text
Prompt = 1024 Tokens
```

模型会对：

```text
1024 Tokens
```

进行完整 Forward。

---

# 19. Prefill 的目标有两个

第一：

```text
计算第一个 Output Token 的 Logits
```

第二：

```text
为所有 Prompt Token 构建 KV Cache
```

所以：

```text
Prefill
=
Prompt Computation
+
KV Cache Construction
```

---

# 20. llama.cpp 里的 Prompt Processing

llama.cpp benchmark 常用：

```text
pp
=
prompt processing
```

例如：

```text
pp512
```

代表：

> 处理一个长度约 512 Token 的 Prompt。

可以把：

```text
Prompt Processing
≈
Prefill
```

理解。

---

# 21. Prefill 的 Tensor Shape

假设：

```text
B = 1
T = 1024
D = 2048
```

输入 Hidden：

```text
[1, 1024, 2048]
```

Linear：

```text
X @ W
```

其中：

```text
M ≈ 1024
```

因此属于比较大的：

```text
Matrix-Matrix Multiplication
```

---

# 22. 为什么 Prefill 并行度高？

因为整个 Prompt：

```text
Token 1
Token 2
...
Token 1024
```

都是已知的。

虽然使用：

```text
Causal Mask
```

每个位置只能看过去。

但各位置 Q/K/V、FFN 等可以：

```text
批量并行计算
```

---

# 23. Prefill 中 Attention

Q：

```text
[T,d]
```

K：

```text
[T,d]
```

Attention Score：

```text
[T,T]
```

因此 Context Length 较长时：

```text
Attention Compute
```

会非常显著。

---

# 24. Prefill 为什么通常更接近 Compute-Bound？

Prefill：

```text
一次处理很多 Token
```

同一个 Weight：

```text
被大量 Token 重复使用
```

所以：

```text
每从内存读取一块 Weight
可以完成更多 FLOPs
```

Arithmetic Intensity 较高。

---

# 25. 什么是 Compute-Bound？

如果硬件：

```text
算力忙满
```

而：

```text
Memory Bandwidth 还不是限制
```

性能主要受：

```text
Peak Compute
```

影响。

这叫：

```text
Compute-Bound
```

---

# 26. 什么是 Memory-Bound？

如果：

```text
Compute Unit 等数据
```

主要时间花在：

```text
从内存搬 Weight / KV / Activation
```

那么：

```text
Memory Bandwidth
```

成为主要瓶颈。

这叫：

```text
Memory-Bound
```

---

# 27. Arithmetic Intensity

定义：

```text
Arithmetic Intensity
=
Operations
/
Bytes Moved
```

即：

> 每搬 1 Byte 数据能做多少计算。

---

# 28. Arithmetic Intensity 高

意味着：

```text
搬一次数据
做很多运算
```

更有机会：

```text
Compute-Bound
```

---

# 29. Arithmetic Intensity 低

意味着：

```text
搬很多数据
只做少量计算
```

更容易：

```text
Memory-Bound
```

---

# 30. Roofline Model

可以理解：

```text
Performance
│
│                 ───────── Peak Compute
│              /
│            /
│          /
│        /
│______/
└──────────────────────── Arithmetic Intensity
```

左边：

```text
Memory-Bound
```

右边：

```text
Compute-Bound
```

---

# 31. Decode 是什么？

Prefill 生成第一个 Token 后：

```text
开始逐 Token 生成
```

例如：

```text
Output Token 1
↓
Output Token 2
↓
Output Token 3
...
```

这叫：

```text
Decode
```

或者 llama.cpp 中常叫：

```text
Token Generation
TG
```

---

# 32. Decode 为什么不能像 Prefill 一样一次算 100 个未来 Token？

因为：

```text
Token 2
```

依赖：

```text
Token 1
```

而：

```text
Token 1
```

在模型运行前未知。

所以：

```text
future output token
```

无法提前并行计算。

---

# 33. Decode Loop

```text
Current Sequence
      ↓
Model Forward
      ↓
Logits
      ↓
Sampling
      ↓
New Token
      ↓
Append Token
      ↓
Next Forward
```

循环：

```text
N times
```

---

# 34. Decode 时每步输入多少 Token？

使用 KV Cache 后：

```text
通常只需要输入新 Token
```

所以当前 Query Length：

```text
1
```

或者服务器批量时：

```text
每个 active sequence 1 个新 Token
```

---

# 35. Decode 的 Linear Layer

Prefill：

```text
[1024,D] @ [D,H]
```

Decode：

```text
[1,D] @ [D,H]
```

即：

```text
Matrix-Matrix
```

变得更接近：

```text
Vector-Matrix / Small GEMM
```

---

# 36. 为什么 Decode GPU 利用率容易低？

因为：

```text
Work per Request
```

很小。

但：

```text
Model Weight
```

仍然要被访问。

因此：

```text
Compute Units
```

可能没有足够工作填满。

---

# 37. Decode 为什么典型 Memory-Bound？

每生成 1 个 Token：

```text
模型的大量 Weight
```

仍然需要参与 Forward。

但只有：

```text
1 Token
```

共享这些 Weight。

于是：

```text
FLOPs / Byte
```

比较低。

---

# 38. 一个非常重要的直觉

假设模型 Weight：

```text
4 GB
```

单用户 Decode 一个 Token：

```text
本质上接近：
把模型大部分 Weight 从内存系统“流过”一次
```

所以：

> Decode Tokens/s 往往和可实现的 Memory Bandwidth 强相关。

这不是严格公式，但非常有用。

---

# 39. 一个粗略带宽上限直觉

如果：

```text
Model Weight = 2 GB
```

设备有效带宽：

```text
20 GB/s
```

极度粗略地看：

```text
20 / 2
≈
10 token/s
```

实际还要：

```text
KV Cache
Activation
Kernel Efficiency
Cache Hit
Quant Metadata
Operator Overhead
```

所以会不同。

但这个估算非常适合判断：

> 为什么端侧 Decode 不可能只看 TOPS。

---

# 40. TOPS 为什么不能直接等于 LLM Tokens/s？

TOPS：

```text
Peak Arithmetic Throughput
```

但 Decode 可能：

```text
Memory-Bound
```

即使：

```text
NPU 6 TOPS
```

也可能因为：

```text
DDR Bandwidth
Weight Format
Operator Support
Runtime Efficiency
```

只得到有限 Tokens/s。

---

# 41. Prefill 和 Decode 是两种不同 Benchmark

这件事必须牢牢记住。

```text
Prefill
=
Prompt Processing Speed

Decode
=
Token Generation Speed
```

所以一个 Runtime 可以：

```text
Prefill 很快
Decode 一般
```

也可以反过来。

---

# 42. llama.cpp benchmark

`llama-bench` 通常区分：

```text
-p
Prompt Processing

-n
Token Generation

-pg
Prompt + Generation
```

因此 benchmark 时应分别看：

```text
pp
tg
```

---

# 43. 为什么只说“模型 25 t/s”信息不完整？

需要问：

```text
Prompt Processing 还是 Generation？

Prompt Length 多长？

Output Length 多长？

Batch Size？

Context Depth？

Quantization？

Threads？

GPU Offload？

KV Cache Type？
```

---

# 44. TTFT

TTFT：

```text
Time To First Token
```

从用户发出 Request：

```text
到收到第一个 Output Token
```

的时间。

---

# 45. TTFT 包含什么？

Serving Benchmark 中可能包括：

```text
Queue Time
Tokenization
Prefill
Scheduling
Network / Streaming
First Sampling
```

具体要看 Benchmark 定义。

所以：

> TTFT 不是严格等于 Prefill Kernel Time。

但通常：

```text
Prefill
```

是重要组成部分。

---

# 46. 为什么 Prompt 越长 TTFT 通常越高？

因为 Prefill 要处理：

```text
更多 Token
```

尤其长 Context：

```text
Attention
```

成本增加。

---

# 47. TPOT

TPOT：

```text
Time Per Output Token
```

代表：

> 第一个 Token 以后，每生成一个输出 Token 平均需要多长时间。

vLLM 的常见定义：

```text
TPOT
=
(E2E Latency - TTFT)
/
(Output Tokens - 1)
```

---

# 48. Tokens/s 与 TPOT

如果：

```text
TPOT = 50 ms
```

粗略：

```text
1000 / 50
=
20 token/s
```

对于单序列 Decode：

```text
Tokens/s
≈
1 / seconds_per_token
```

---

# 49. ITL

ITL：

```text
Inter-Token Latency
```

即：

```text
相邻 streamed outputs
```

之间的时间间隔。

---

# 50. TPOT vs ITL

标准逐 Token Streaming 时：

```text
二者通常接近
```

但如果：

```text
Speculative Decoding
一次返回多个 Token
```

或者复杂调度：

```text
ITL 和 TPOT
```

可能不同。

---

# 51. End-to-End Latency

总响应时间：

```text
Request
↓
First Token
↓
All Decode
↓
Last Token
```

可以粗略：

```text
E2E
≈
TTFT
+
TPOT × (N_output - 1)
```

不考虑额外开销时。

---

# 52. 交互式体验最重要的两个指标

通常：

```text
TTFT
+
ITL / TPOT
```

用户最直接感受到：

```text
多久开始说
+
后面说得快不快
```

---

# 53. Server Throughput 又是另一回事

服务器还关心：

```text
Requests/s
Output Tokens/s
Total Tokens/s
Concurrency
```

所以：

> 单用户最快 和 系统总吞吐最高 不一定是同一个配置。

---

# 54. Latency vs Throughput

低 Batch：

```text
单用户 Latency 好
```

高 Batch：

```text
硬件利用率高
总 Throughput 好
```

但：

```text
单用户可能等待更久
```

---

# 55. KV Cache：Decode 的核心

进入这一章最重要的内部状态：

```text
KV Cache
```

它缓存：

```text
每一层 Attention
历史 Token 对应的 K 和 V
```

---

# 56. Attention 回顾

每层：

```text
Q = XWq

K = XWk

V = XWv
```

Attention：

```text
softmax(QK^T / sqrt(d)) V
```

---

# 57. 第一次 Prefill

Prompt：

```text
Token 1 ... Token T
```

所有 Token：

```text
K_1...K_T
V_1...V_T
```

都会被计算出来。

---

# 58. 生成新 Token 时

新 Token：

```text
Token T+1
```

计算：

```text
Q_new
K_new
V_new
```

然后：

```text
Q_new
```

需要 Attention：

```text
K_1 ... K_T K_new
```

以及：

```text
V_1 ... V_T V_new
```

---

# 59. 如果没有 KV Cache

下一步：

```text
Token T+2
```

如果重新从头 Forward：

```text
K_1...K_T
V_1...V_T
```

又被重新计算。

这极其浪费。

---

# 60. 使用 KV Cache

保存：

```text
K_1 ... K_T
V_1 ... V_T
```

下一 Token：

```text
只计算：
K_new
V_new
Q_new
```

然后 Append：

```text
KV Cache
```

---

# 61. 为什么缓存 K 和 V？

因为历史 Token 的：

```text
K
V
```

一旦计算完成：

> 在标准 causal autoregressive Transformer 中，不会因为未来 Token 出现而发生变化。

所以可复用。

---

# 62. 为什么不缓存历史 Q？

历史 Query：

```text
只在历史 Token 当时计算输出时使用
```

未来生成 Token 时：

```text
需要的是新 Token 的 Query
```

去查询：

```text
所有历史 K
```

所以历史 Q 没有主要复用价值。

---

# 63. KV Cache 是 Layer-specific

Transformer：

```text
Layer 0
Layer 1
...
Layer N-1
```

每一层都有自己的：

```text
K Cache
V Cache
```

不是整个模型只有一份 K/V。

---

# 64. KV Cache Shape

典型可以理解：

```text
K:
[L, B, H_kv, T, D_head]

V:
[L, B, H_kv, T, D_head]
```

实际 Runtime Layout 会不同。

其中：

```text
L = Layers
B = Batch / Sequences
H_kv = Number of KV Heads
T = Cached Tokens
D_head = Head Dimension
```

---

# 65. KV Cache Memory 粗略公式

```text
KV bytes
≈
2
×
Layers
×
Sequence Length
×
KV Heads
×
Head Dim
×
Bytes Per Element
×
Batch
```

其中：

```text
2
=
K + V
```

---

# 66. 举一个 KV Cache 例子

假设：

```text
Layers = 28
KV Heads = 4
Head Dim = 128
Context = 8192
dtype = FP16 = 2 bytes
Batch = 1
```

粗略：

```text
2 × 28 × 4 × 128 × 8192 × 2
```

约：

```text
448 MiB
```

只是一份 KV Cache。

---

# 67. Context Length 翻倍

如果其他不变：

```text
KV Cache
```

基本：

```text
线性翻倍
```

所以：

```text
8K → 16K
```

KV Memory：

```text
约 ×2
```

---

# 68. Batch 翻倍

同时处理：

```text
更多独立 Sequence
```

KV Cache：

```text
也近似线性增加
```

---

# 69. GQA 为什么重要？

MHA：

```text
H_q = H_kv
```

GQA：

```text
H_q > H_kv
```

减少：

```text
KV Heads
```

直接减少：

```text
KV Cache
```

和：

```text
Decode KV Memory Traffic
```

---

# 70. MQA

MQA：

```text
H_kv = 1
```

进一步降低 KV Cache。

但：

```text
模型结构与质量折中
```

由模型设计阶段决定。

---

# 71. KV Cache 的 dtype

不一定必须：

```text
FP16
```

现代 Runtime 可以尝试：

```text
FP8
INT8
Q8
Q4
```

等 Cache 类型。

例如 llama.cpp CLI 中可以单独配置：

```text
cache type K
cache type V
```

---

# 72. KV Cache Quantization

目标：

```text
KV Memory ↓
KV Bandwidth ↓
```

尤其：

```text
Long Context
High Concurrency
```

时价值明显。

---

# 73. KV Cache 量化的代价

可能：

```text
Accuracy / Quality 变化
Quant / Dequant 开销
Kernel Support Requirement
```

所以：

> 不是越低 bit 越好。

---

# 74. Cache Offloading

如果 GPU VRAM 不够：

```text
一部分 KV Cache
```

可以放：

```text
CPU RAM
```

需要时：

```text
Prefetch
```

---

# 75. Cache Offload 的本质

用：

```text
PCIe / Interconnect Transfer
```

换：

```text
GPU Memory Capacity
```

所以：

```text
能跑更长 Context
```

但性能可能下降。

---

# 76. Weight Memory

模型参数：

```text
P Parameters
```

如果：

```text
FP16
2 bytes / parameter
```

粗略：

```text
Weight Memory
≈
2P bytes
```

---

# 77. 2B 模型 FP16

```text
2 billion
×
2 bytes
≈
4 GB
```

这是：

```text
裸 Weight
```

粗略值。

---

# 78. INT8

理论：

```text
≈ 1 byte / Weight
```

2B：

```text
≈ 2 GB
```

---

# 79. INT4

理论裸 Weight：

```text
≈ 0.5 byte / Weight
```

2B：

```text
≈ 1 GB
```

---

# 80. 为什么实际量化模型会比理论裸 Weight 大？

还需要：

```text
Scale
Zero Point
Block Metadata
Tensor Metadata
Alignment
Model Metadata
Tokenizer
```

所以：

```text
INT4
```

并不是精确：

```text
0.5 byte × parameters
```

---

# 81. Runtime Memory 总体组成

```text
Runtime Memory
=
Weights
+
KV Cache
+
Activations
+
Workspace
+
Runtime Buffers
+
Allocator Overhead
```

---

# 82. Activation Memory

Inference 虽然不保存 Training Backward Activations。

但当前 Layer 仍需要：

```text
Hidden State
Q/K/V
Attention Output
FFN Intermediate
```

等临时 Tensor。

---

# 83. Workspace

高性能 Kernel：

```text
GEMM
Attention
Quantization
```

可能需要：

```text
temporary workspace
```

Runtime 通常会预分配。

---

# 84. 为什么峰值内存重要？

某个瞬间：

```text
FFN intermediate
+
Attention buffer
+
KV
+
Weight
```

同时存在。

所以：

```text
Peak Memory
```

比平均内存更重要。

---

# 85. Weight Quantization 为什么能加速 Decode？

Decode Memory-Bound：

```text
主要成本之一
=
读取 Weight
```

FP16 → INT4：

```text
Weight Bytes
约减少 4 倍
```

理论上：

```text
Memory Traffic 大幅下降
```

于是：

```text
Tokens/s 可以上升
```

---

# 86. 量化速度收益不是固定 4 倍

因为还有：

```text
Dequantization
KV Cache
Other Operators
Memory Efficiency
Kernel Efficiency
Metadata
CPU SIMD / GPU Kernel
```

所以：

```text
Memory 大幅减小
```

不等于：

```text
速度严格 ×4
```

---

# 87. 为什么 Prefill 量化收益可能和 Decode 不同？

Prefill：

```text
更高 Arithmetic Intensity
```

更接近：

```text
Compute-Bound
```

所以：

```text
Weight Bandwidth
```

不一定是唯一限制。

量化还要看：

```text
低精度 Compute Kernel
Tensor Core / NPU Support
```

才能真正提高速度。

---

# 88. 量化的两个性能价值

第一：

```text
Memory Capacity
```

让模型：

```text
能装进设备
```

第二：

```text
Memory Bandwidth
```

让 Decode：

```text
搬更少 Weight
```

---

# 89. 这也是端侧 INT4 的关键

RK3576 之类 SoC：

```text
DDR Capacity 有限
DDR Bandwidth 有限
```

INT4：

```text
不仅让模型放得下
```

还可能：

```text
降低每 Token 的 DDR Traffic
```

---

# 90. CPU Decode

CPU：

```text
Model Weight
```

通常在系统 RAM。

Decode 性能高度依赖：

```text
Memory Bandwidth
SIMD
Cache
NUMA
Threading
Quantized Kernel
```

---

# 91. 为什么 CPU 增加很多线程后 Tokens/s 不一定线性上升？

一开始：

```text
更多 Core
→
更高 Memory Request
```

到达：

```text
Memory Bandwidth 饱和
```

后：

```text
再增加线程
```

收益很小。

甚至：

```text
Thread Overhead
Cache Contention
```

导致变慢。

---

# 92. GPU Decode

GPU 有：

```text
高显存带宽
```

但单序列 Decode：

```text
并行度不足
```

可能不能充分使用：

```text
峰值 FLOPS
```

---

# 93. 为什么 Batch 可以提升 GPU Decode Throughput？

多个 Sequence 同时：

```text
每个都生成 1 Token
```

合并：

```text
B Tokens
```

一起执行。

于是：

```text
Weight 读取一次
```

能服务：

```text
多个 Sequence
```

Arithmetic Intensity 提高。

---

# 94. Batch = 更高 Hardware Utilization

但代价：

```text
每用户 Latency
Queue
KV Memory
```

可能增加。

所以服务器要做：

```text
Latency / Throughput Tradeoff
```

---

# 95. Static Batching

最简单：

```text
等一批 Request
↓
一起跑
↓
全部完成
↓
下一批
```

问题：

```text
Output Length 不同
```

短请求要等长请求。

---

# 96. Continuous Batching

当某个 Request：

```text
完成
```

Runtime 可以立即：

```text
把新 Request 插入 Batch
```

不用等整个 Batch 完成。

这叫：

```text
Continuous Batching
```

---

# 97. 为什么 vLLM 需要 Scheduler？

因为同时有：

```text
Request A:
Prefill

Request B:
Decode

Request C:
Decode

Request D:
Waiting
```

Runtime 必须决定：

```text
下一步 GPU 跑谁？
```

---

# 98. Serving Scheduler 的目标

通常需要平衡：

```text
TTFT
TPOT
Throughput
Fairness
Memory
```

不是简单：

```text
先来先服务
```

就一定最好。

---

# 99. Prefill 会影响 Decode Latency

大 Prefill：

```text
GPU 占用时间长
```

可能让正在 Decode 的用户：

```text
下一个 Token 等更久
```

导致：

```text
ITL Spike
```

---

# 100. Chunked Prefill

把一个长 Prompt：

```text
4096 Tokens
```

切成：

```text
512
512
512
...
```

逐块 Prefill。

---

# 101. Chunked Prefill 的目标

让：

```text
长 Prefill
```

不要一次霸占设备太久。

这样可以：

```text
在 Prefill Chunk 之间插入 Decode
```

改善：

```text
ITL / fairness
```

---

# 102. Chunk 太小也不好

因为：

```text
Kernel Launch
Scheduling
Extra Overhead
```

增加。

所以：

```text
Chunk Size
```

也是性能调优参数。

---

# 103. KV Cache Management

服务器同时几十 / 几百 Request：

```text
每个 Sequence
```

有自己的 KV Cache。

如果每个都预先申请：

```text
Max Context Length
```

非常浪费。

---

# 104. 一个 KV 内存浪费例子

配置：

```text
Max Context = 32K
```

实际 Request：

```text
只用了 1K
```

如果预分配完整 32K：

```text
31K 空间浪费
```

---

# 105. Paged KV Cache

类似 OS Virtual Memory：

```text
KV Cache
```

被切成：

```text
Fixed-size Blocks / Pages
```

Sequence：

```text
需要多少
分配多少 Block
```

---

# 106. PagedAttention 的核心思想

不是说：

```text
Attention 数学公式变了
```

而是：

> KV Cache 的存储和访问变成分块 / 页式管理。

---

# 107. Paged KV 的价值

```text
降低 Memory Fragmentation
提高 KV Memory Utilization
更灵活支持 Dynamic Sequences
方便 Continuous Batching
```

---

# 108. Logical Block vs Physical Block

Sequence 认为自己的 KV：

```text
Token 0...N
```

是连续的逻辑空间。

实际：

```text
Block Table
```

把逻辑 Block 映射到：

```text
Physical KV Blocks
```

---

# 109. 为什么需要特殊 Attention Kernel？

因为 K/V：

```text
物理上可能不连续
```

Attention Kernel 需要根据：

```text
Block Table
```

找到对应数据。

---

# 110. Prefix Cache

假设很多 Request 都有相同：

```text
System Prompt
```

例如：

```text
2000 Tokens
```

每次重新 Prefill：

```text
非常浪费
```

---

# 111. Prefix Caching

把相同前缀的：

```text
KV State
```

缓存。

下一 Request：

```text
复用前缀
```

只 Prefill：

```text
新增部分
```

---

# 112. Prompt Cache

llama.cpp 也有：

```text
Prompt Cache
```

概念上是：

> 保存某个 Prompt 处理后的模型状态，以便后续复用。

---

# 113. Prefix Cache 适合什么？

```text
固定 System Prompt
长文档 Prefix
Multi-turn Conversation
共享模板
```

---

# 114. Multi-turn Chat

第一轮：

```text
System
User1
Assistant1
```

第二轮：

```text
System
User1
Assistant1
User2
```

如果重新从头 Prefill：

```text
大量重复
```

Runtime 可以通过 Cache：

```text
复用已有上下文状态
```

---

# 115. Conversation Context 仍然会增长

即使复用 Prefix：

```text
KV Cache
```

仍要保存越来越多历史 Token。

所以：

```text
Long Conversation
```

最终仍受：

```text
Context Limit
Memory
```

限制。

---

# 116. Context Truncation

如果超过 Max Context：

Runtime / Application 可能：

```text
丢弃旧 Token
Summarize
Sliding Window
RoPE Scaling
```

具体取决于模型和应用。

---

# 117. Sliding Window Attention

不是所有 Layer 永远关注：

```text
所有历史 Token
```

有些模型只保留：

```text
最近 W Tokens
```

作为 Attention Window。

---

# 118. Sliding Window 的意义

长 Sequence：

```text
KV / Attention Cost
```

可以控制在：

```text
固定 Window
```

但代价：

```text
不能直接关注很久之前的 Token
```

---

# 119. FlashAttention

标准 Attention 概念实现：

```text
QK^T
↓
Store Score Matrix
↓
Softmax
↓
Store
↓
× V
```

会产生大量：

```text
Memory IO
```

---

# 120. FlashAttention 核心

通过：

```text
Tiling
Kernel Fusion
Online Softmax
```

减少：

```text
HBM / DRAM ↔ On-chip Memory
```

的数据搬运。

---

# 121. FlashAttention 不是改变模型结果目标

它仍然计算：

```text
Attention
```

核心优化：

```text
Memory IO
```

而不是简单：

```text
删掉 Attention
```

---

# 122. FlashAttention 对 Prefill 特别重要

Prefill：

```text
T 比较大
```

Attention：

```text
T × T
```

中间数据巨大。

所以：

```text
IO-aware Attention Kernel
```

价值高。

---

# 123. Decode Attention 也不同

Decode：

```text
Q length = 1
```

但需要访问：

```text
Long KV Cache
```

所以瓶颈更偏：

```text
KV Memory Read
```

---

# 124. Prefill Kernel 与 Decode Kernel 可以不同

因为 Shape 完全不同：

```text
Prefill:
Q length = T

Decode:
Q length = 1
```

所以高性能 Runtime 会：

```text
使用不同 Kernel Path
```

---

# 125. Sampling 在哪里？

Model Forward 输出：

```text
Logits
```

然后 Runtime / Application：

```text
Sampling
```

得到 Token ID。

所以：

```text
Sampling
```

通常是 Forward 之后的逻辑。

---

# 126. Logits Processor

在 Sampling 前可能：

```text
Temperature
Repetition Penalty
Frequency Penalty
Top-K
Top-P
Grammar Mask
Bad Word Mask
```

修改 Logits。

---

# 127. Greedy

```text
argmax(logits)
```

理论上最简单。

---

# 128. Temperature

```text
logits /= temperature
```

改变：

```text
概率分布尖锐程度
```

---

# 129. Top-K

只保留：

```text
最高 K 个候选
```

---

# 130. Top-P

保留累计概率达到：

```text
p
```

的候选集合。

---

# 131. Sampling 的计算量大吗？

相较：

```text
整个 Transformer Forward
```

通常不是主要计算量。

但如果：

```text
Vocabulary 很大
Complex Grammar
```

也有一定成本。

---

# 132. Structured Generation

如果 Agent 要输出：

```text
JSON
Tool Call
Grammar-constrained Text
```

Runtime 可以在每步 Sampling 时：

```text
Mask Invalid Tokens
```

---

# 133. 为什么 Structured Output 会影响 Decode？

每 Token：

```text
需要判断哪些 Token 合法
```

可能增加：

```text
CPU / sampling overhead
```

但通常远小于大型模型 Forward。

---

# 134. Detokenization

生成：

```text
Token IDs
```

需要：

```text
Tokenizer Decode
```

变成 String。

Streaming 时：

```text
逐步输出 Text Pieces
```

---

# 135. Tokens/s 为什么不是跨模型绝对公平指标？

不同模型：

```text
Tokenizer Vocabulary
Merge Rules
```

不同。

同一句话可能：

```text
Model A = 10 Tokens
Model B = 14 Tokens
```

所以：

```text
20 Tokens/s
```

不一定意味着相同字符 / 字数输出速度。

---

# 136. 更公平的用户体验指标

有时可以同时看：

```text
Tokens/s
Characters/s
Words/s
TTFT
TPOT
```

---

# 137. Speculative Decoding

Decode 最大问题：

```text
一次只能确定一个 Token
```

Speculative Decoding：

> 先用一个较便宜 Draft Model 一次猜多个 Token，再让主模型批量验证。

---

# 138. Speculative Decoding 基本链

```text
Draft Model
↓
Guess:
t1 t2 t3 t4
↓
Target Model
一次验证
↓
Accept prefix
↓
Continue
```

---

# 139. 为什么可能加速？

主模型原本：

```text
4 次 Sequential Decode
```

现在可能：

```text
1 次更大批量验证
```

如果 Draft 命中率高：

```text
减少主模型 Sequential Steps
```

---

# 140. Speculative Decoding 的关键

```text
Acceptance Rate
```

如果 Draft 很差：

```text
经常猜错
```

额外 Draft 成本可能不划算。

---

# 141. Draft Model 不一定是另一完整模型

现代方法还包括：

```text
Medusa Heads
EAGLE
Lookahead
N-gram speculation
```

本章只知道位置。

---

# 142. Model Parallelism

如果模型 Weight：

```text
一张 GPU 放不下
```

需要分到多个 Device。

常见：

```text
Tensor Parallel
Pipeline Parallel
Expert Parallel
```

---

# 143. Tensor Parallel

一个大的 Matrix：

```text
W
```

拆到多个 GPU。

每个 GPU：

```text
算一部分
```

再进行：

```text
Collective Communication
```

---

# 144. Tensor Parallel 的代价

需要：

```text
GPU-GPU Communication
```

所以：

```text
Interconnect Bandwidth
```

非常重要。

---

# 145. Pipeline Parallel

不同 Layer：

```text
GPU0:
Layer 0-9

GPU1:
Layer 10-19
```

数据按层流动。

---

# 146. Pipeline Parallel 对单 Token Decode

每个 Token：

```text
需要依次经过多个 Stage
```

会有：

```text
Pipeline Bubble / communication
```

所以部署策略需权衡。

---

# 147. Edge Device 一般不是这个路线

RK3576：

```text
单 SoC
CPU + NPU + GPU
```

更常见：

```text
Heterogeneous Offload
```

而不是大型集群 Parallelism。

---

# 148. Offload

比如 llama.cpp：

```text
部分 Layer / Tensor
```

放 GPU。

其余：

```text
CPU
```

叫：

```text
GPU Offload
```

---

# 149. Partial Offload 的问题

如果每层都频繁：

```text
CPU ↔ GPU
```

传数据：

```text
Interconnect Cost
```

会限制性能。

---

# 150. Weight Residency

高性能推理最好：

```text
Weight 尽量常驻计算 Device 可高效访问的 Memory
```

减少：

```text
反复传输
```

---

# 151. Unified Memory / Shared Memory SoC

很多端侧 SoC：

```text
CPU
GPU
NPU
```

共享：

```text
DDR
```

虽然不一定需要传统 PCIe Copy。

但仍然：

```text
争用 Memory Bandwidth
```

---

# 152. 这对 RK3576 很重要

即使：

```text
NPU Compute 很快
```

如果：

```text
CPU
NPU
GPU
ISP
```

同时访问 DDR：

```text
Effective Bandwidth
```

会下降。

---

# 153. Memory Capacity vs Bandwidth

这两个必须分开。

Capacity：

```text
能不能装下模型
```

Bandwidth：

```text
每秒能搬多少数据
```

---

# 154. 一个模型“能装下”不等于“跑得快”

例如：

```text
8GB RAM
```

可以装：

```text
4GB Model
```

但内存带宽低：

```text
Decode 仍然慢
```

---

# 155. Bandwidth 估算思维

端侧部署先问：

```text
Model Weight Size?

Effective Memory Bandwidth?

Quantization?

KV Size?

Runtime Kernel Efficiency?
```

而不是只问：

```text
TOPS?
```

---

# 156. Batch Size 在端侧

本地 Chat：

```text
通常 Batch = 1 Sequence
```

目标：

```text
Interactive Latency
```

---

# 157. Batch Size 与 Prompt Processing

llama.cpp 还有：

```text
batch-size
ubatch-size
```

它们主要影响：

```text
Prompt Processing
```

时一次处理多少 Token。

---

# 158. Logical Batch vs Physical UBatch

可以粗略理解：

```text
Logical Batch
=
上层希望一起处理的 Token 数

Physical UBatch
=
底层单次真正送入计算 Graph 的 Token Chunk
```

具体 llama.cpp 实现后续章节展开。

---

# 159. 为什么 UBatch 太大可能内存增加？

Prefill：

```text
一次处理更多 Token
```

临时 Activation：

```text
更大
```

所以：

```text
Memory ↑
```

---

# 160. 为什么 UBatch 太小性能也可能下降？

```text
并行度不足
Kernel Launch 次数增加
```

所以需要调优。

---

# 161. Context Size

Runtime 参数：

```text
ctx-size
```

通常决定：

```text
最大上下文容量
```

也影响：

```text
KV Cache Allocation
```

---

# 162. Context 配得过大

即使实际只用短 Prompt：

某些 Runtime：

```text
可能预分配更大 Cache
```

增加 Memory。

具体视 Cache 策略而定。

---

# 163. Dynamic KV Cache

随着 Token：

```text
逐渐增长
```

优点：

```text
少用少分
```

---

# 164. Static KV Cache

提前：

```text
分配固定最大 Shape
```

优点：

```text
更适合 Compile
Memory Address 稳定
```

缺点：

```text
可能浪费容量
```

---

# 165. 为什么 Static Cache 有利于 Graph Compile？

Decode 时：

```text
Sequence Length 每步 +1
```

如果 Tensor Shape 每步变化：

```text
Compiler 可能需要重新 Trace / Specialize
```

Static Cache：

```text
保持 Buffer Shape
```

更加 AOT-friendly。

---

# 166. Quantized KV Cache

目的：

```text
Context Capacity ↑
Concurrent Requests ↑
Bandwidth ↓
```

代价：

```text
Quant/Dequant
Quality / Kernel Support
```

---

# 167. Prefix Cache 与 KV Cache 的区别

KV Cache：

```text
当前 Sequence 推理必须保存的历史 K/V
```

Prefix Cache：

```text
跨 Request 复用相同前缀的 KV 状态
```

---

# 168. Model Cache / Prompt Cache 不要混淆

Model Weight Cache：

```text
权重驻留
```

Prompt / Prefix Cache：

```text
已计算的 Context State
```

KV Cache：

```text
Attention 历史状态
```

术语要看具体 Runtime。

---

# 169. Prefill Complexity

Transformer 每层大致：

```text
Linear / FFN
+
Attention
```

长 Context 时：

```text
Attention 部分
≈ O(T²)
```

但大量 Linear：

```text
≈ O(T)
```

乘模型维度。

---

# 170. Decode 每 Token Complexity

使用 KV Cache 后：

```text
不用重新计算过去 Token 的所有 Layer Hidden
```

但 Attention 当前 Query 仍需：

```text
访问历史 K/V
```

所以随 Context：

```text
Attention per token
```

仍会增长。

---

# 171. Decode 不是 O(1)

有 KV Cache：

```text
避免过去完整 Transformer 重算
```

但：

```text
Attention over cache
```

仍与 Context Length 有关。

---

# 172. Context 越长，Decode 也可能越慢

因为：

```text
读取更多 K/V
```

Attention：

```text
Q_new × K_past
```

长度增加。

---

# 173. 为什么小模型长 Context 也可能内存很大？

Weight：

```text
固定
```

但 KV：

```text
随 Context 增长
```

所以：

```text
小 Weight
+
128K Context
```

KV 可能成为显著部分。

---

# 174. Attention 与 FFN 的性能占比

取决于：

```text
Model Shape
Context Length
Batch
Hardware
```

短 Context：

```text
FFN / Weight MatMul
```

可能占很大部分。

超长 Context：

```text
Attention / KV
```

占比上升。

---

# 175. MoE 推理

MoE：

```text
每 Token 只激活部分 Expert
```

所以：

```text
Active FLOPs
```

较少。

但所有 Expert Weight：

```text
可能都需要存储
```

---

# 176. MoE 对 Decode 的挑战

不同 Token：

```text
路由到不同 Expert
```

会导致：

```text
Irregular Memory Access
Load Imbalance
Expert Weight Traffic
```

---

# 177. Active Parameters vs Total Parameters

MoE 模型：

```text
Total = 30B
Active = 3B
```

并不代表：

```text
内存只需要 3B
```

因为大量 Expert Weight：

```text
仍然要存
```

---

# 178. NPU 推理与 GPU 推理有什么差异？

模型数学：

```text
基本相同
```

但 NPU：

```text
Operator Set
Tensor Layout
On-chip SRAM
Quantization Format
Compiler
Runtime
```

更固定。

---

# 179. NPU 为什么经常更喜欢 INT8 / INT4？

专用 MAC Array：

```text
对低精度整数
```

通常提供：

```text
更高吞吐
更低功耗
更低内存
```

---

# 180. 但 LLM NPU 部署难在哪里？

LLM 有：

```text
Dynamic Decode Loop
KV Cache
Dynamic Context
RoPE
Attention
Sampling
Long Sequence
```

比固定 CNN Graph：

```text
复杂得多
```

---

# 181. 因此 RKLLM 与 RKNN 分开

普通视觉 / CNN：

```text
RKNN-Toolkit2
```

LLM：

```text
RKLLM / RKNN-LLM
```

因为 Runtime 模式完全不同。

---

# 182. LLM Runtime 不是简单执行一次静态 Graph

实际上：

```text
Load Model
↓
Prefill Graph
↓
Sample
↓
Decode Graph
↓
Sample
↓
Decode Graph
...
```

还要管理：

```text
KV State
Sequence
Position
Stop Condition
```

---

# 183. Prefill Graph 和 Decode Graph 可以不同

虽然使用：

```text
同一套 Weight
```

但 Tensor Shape：

```text
完全不同
```

所以：

```text
Compiler / Runtime
```

可以专门优化两个阶段。

---

# 184. NPU 上这尤其重要

固定编译器可能分别生成：

```text
Prefill Execution Plan
```

和：

```text
Decode Execution Plan
```

甚至不同 Context / Batch：

```text
不同 Graph Variant
```

---

# 185. Token Generation 的最终停止条件

常见：

```text
EOS

Max New Tokens

Stop String

Grammar Finish

Application Condition
```

---

# 186. EOS 检测在哪里？

一般：

```text
Sampling 后
```

得到 Token ID。

Runtime / Generation Loop：

```text
判断是否 EOS
```

---

# 187. Streaming

每得到 Token：

```text
decode to text piece
```

然后：

```text
发送给客户端
```

让用户看到实时输出。

---

# 188. Streaming 不一定改变模型速度

它主要改变：

```text
Perceived Latency
```

用户不用等完整答案结束。

---

# 189. Streaming Server 还有额外成本

```text
HTTP
SSE / WebSocket
Serialization
Network
Frontend Render
```

所以：

```text
Server TTFT
```

不等于纯模型 TTFT。

---

# 190. Benchmark 层次

建议分三层。

---

# 191. Layer 1：Kernel Benchmark

例如：

```text
GEMM
Attention
Softmax
```

测：

```text
μs
TFLOPS
Bandwidth
```

---

# 192. Layer 2：Runtime Benchmark

例如 llama-bench：

```text
Prompt Processing
Token Generation
```

测：

```text
tokens/s
```

---

# 193. Layer 3：Serving Benchmark

例如 vLLM：

```text
HTTP Requests
Concurrency
Queue
Scheduler
Streaming
```

测：

```text
TTFT
TPOT
ITL
Throughput
```

---

# 194. 三层 Benchmark 不能混为一谈

Kernel 快：

```text
不保证 Runtime 快
```

Runtime 单请求快：

```text
不保证 Server 高并发快
```

---

# 195. Benchmark 要记录 Prompt / Output Length

例如：

```text
Input = 128
Output = 128
```

和：

```text
Input = 8192
Output = 128
```

完全是不同 Workload。

---

# 196. 推荐 Benchmark 格式

```text
Model:
Qwen...

Quant:
Q4_K_M

Device:
...

Runtime:
...

Input Tokens:
512

Output Tokens:
128

Context:
4096

Batch:
1

PP:
xxx tok/s

TG:
xxx tok/s

TTFT:
xxx ms

TPOT:
xxx ms

Peak Memory:
xxx MB
```

---

# 197. 为什么必须记录 Runtime Version？

Runtime：

```text
Kernel
Scheduler
Quantization
Backend
```

持续优化。

不同版本：

```text
性能可能明显不同
```

---

# 198. 为什么必须记录 Quant Type？

例如：

```text
FP16
Q8_0
Q4_0
Q4_K_M
```

不仅：

```text
模型文件大小不同
```

Kernel 路径也不同。

---

# 199. 为什么必须记录 KV Cache Type？

Weight：

```text
Q4
```

不代表 KV：

```text
也是 Q4
```

KV 可能：

```text
FP16
```

或者单独量化。

---

# 200. 为什么必须记录 GPU Offload？

llama.cpp：

```text
CPU only
```

和：

```text
all layers GPU
```

性能完全不同。

---

# 201. 为什么 PP 和 TG 要分别 Benchmark？

它们优化方向不同：

```text
PP:
Compute / Attention / Batch

TG:
Memory Bandwidth / Quantized MatVec / KV
```

---

# 202. 一种典型错误

用户看到：

```text
Prompt Processing = 500 t/s
```

就认为：

```text
聊天输出也是 500 t/s
```

错误。

实际：

```text
Token Generation
```

可能只有：

```text
20 t/s
```

---

# 203. 另一种典型错误

只看：

```text
TG = 30 t/s
```

却忽略：

```text
长 Prompt Prefill 需要 10 秒
```

真实 Chat：

```text
TTFT 很差
```

---

# 204. Agent 场景为什么特别关注 TTFT？

Agent 每个 Tool Result 后：

```text
可能再次调用 LLM
```

大量：

```text
短 Prefill + Decode
```

不断发生。

所以：

```text
LLM Call Latency
```

会累积。

---

# 205. RAG 为什么让 Prefill 更重要？

RAG 会加入：

```text
多个 Retrieved Chunks
```

Prompt 可能从：

```text
200 Tokens
```

变：

```text
4000 Tokens
```

于是：

```text
TTFT ↑
KV Cache ↑
```

---

# 206. Context Engineering 其实也是 Inference Optimization

减少无关 Context：

```text
不仅提高回答质量
```

还可以：

```text
Prefill ↓
KV Memory ↓
TTFT ↓
```

---

# 207. Agent Memory 也会影响推理性能

如果每次把：

```text
所有历史 Memory
```

塞入 Prompt：

```text
Prefill 越来越慢
```

所以：

```text
Retrieve relevant memory
```

也是系统性能设计。

---

# 208. Hardware Utilization

理想：

```text
Compute Unit
一直忙
```

现实：

```text
Waiting for Memory
Waiting for Kernel Launch
Waiting for Sync
Small Tensor
```

都会降低 Utilization。

---

# 209. Operator Fusion

例如：

```text
RMSNorm
+
Quant
+
MatMul
```

如果可以融合：

```text
减少中间 Tensor 写回 DRAM
```

可能明显提升 Decode。

---

# 210. Kernel Fusion 为什么对 Decode 重要？

Decode Tensor 小：

```text
Kernel Launch Overhead
Memory Roundtrip
```

相对更显著。

所以：

```text
Fusion
```

可以减少：

```text
Launch
Memory Traffic
```

---

# 211. Weight Repacking

量化 Weight：

```text
原始文件布局
```

不一定是 CPU / GPU Kernel 最喜欢的布局。

Runtime 可以：

```text
Repack
```

成：

```text
SIMD / Tensor Core friendly
```

格式。

---

# 212. Repacking 的代价

启动：

```text
多一步转换
```

但长期运行：

```text
Kernel 更快
```

属于：

```text
Load-time vs Run-time tradeoff
```

---

# 213. Memory Layout

同样 Tensor：

```text
逻辑 Shape 一样
```

但：

```text
内存排列不同
```

会影响：

```text
Vectorization
Coalescing
Cache
```

---

# 214. CPU Cache

CPU：

```text
L1
L2
L3
RAM
```

速度差很多。

但 LLM Weight：

```text
远大于 Cache
```

Decode 大量访问：

```text
DRAM
```

所以 Memory Bandwidth 很关键。

---

# 215. GPU Memory Hierarchy

```text
Register
Shared Memory
L2
HBM / GDDR
```

FlashAttention：

```text
就是在想办法
更多利用近端高速存储
减少 HBM IO
```

---

# 216. NPU On-chip SRAM

NPU 通常：

```text
小容量高速 SRAM
+
外部 DDR
```

Compiler 要：

```text
Tile
Schedule
Reuse Data
```

减少 DDR 访问。

---

# 217. 这就是为什么 AI Compiler 很重要

同一个 Operator：

```text
MatMul
```

怎么：

```text
Tile
Fuse
Schedule
Layout
Buffer
```

决定真实性能。

---

# 218. 但 Runtime 层也很重要

即使 Kernel 很快：

```text
KV Cache 管理差
Batching 差
Scheduling 差
Memory Fragmentation
```

整体仍然慢。

---

# 219. LLM 推理优化的四层

可以建立：

```text
Model Level
Runtime Level
Kernel Level
Hardware Level
```

---

# 220. Model Level

例如：

```text
GQA
MQA
MoE
Sliding Window
Smaller Hidden
Quantization-aware architecture
```

---

# 221. Runtime Level

例如：

```text
Continuous Batching
Paged KV
Prefix Cache
Chunked Prefill
Speculative Decoding
Offload
```

---

# 222. Kernel Level

例如：

```text
FlashAttention
Fused RMSNorm
Quantized GEMM
Fused MLP
Tensor Core Kernel
SIMD MatVec
```

---

# 223. Hardware Level

例如：

```text
HBM Bandwidth
DDR Bandwidth
Tensor Core
NPU MAC Array
Cache
SRAM
Interconnect
```

---

# 224. 为什么后续要分章节学？

因为：

```text
同一个 20 t/s
```

可能慢在：

```text
Model
Runtime
Kernel
Memory
Compiler
```

必须逐层定位。

---

# 225. 本章建议做的第一张图

自己画：

```text
            Prompt
              ↓
          Tokenizer
              ↓
           Prefill
              │
      ┌───────┴────────┐
      ↓                ↓
   KV Cache         Logits
                       ↓
                    Sample
                       ↓
                  First Token
                       ↓
                    Decode
                       │
      ┌────────────────┼─────────────┐
      ↓                ↓             ↓
   Read KV          Read Weight   New KV
      │                │             │
      └────────────────┴─────────────┘
                       ↓
                    Logits
                       ↓
                    Sample
                       ↓
                   Next Token
```

---

# 226. 第二张图：硬件瓶颈

```text
Prefill
│
├── Many Tokens
├── Large GEMM
├── High Parallelism
└── Often Compute-heavy

Decode
│
├── One Token / Sequence
├── Small GEMM / GEMV-like
├── Read Huge Weights
├── Read KV Cache
└── Often Memory-bandwidth-heavy
```

---

# 227. 第三个思维图：优化对应关系

```text
TTFT 太高
↓
看：
Prompt Length
Prefill Kernel
FlashAttention
Batch / UBatch
Compute Throughput
Prefix Cache

TPOT 太高
↓
看：
Weight Size
Quantization
Memory Bandwidth
KV Cache
Decode Kernel
Fusion

OOM
↓
看：
Weight
KV Cache
Context
Batch
Workspace
Offload
```

---

# 228. 为什么性能 Debug 要先分 Prefill / Decode？

如果不分：

```text
“模型慢”
```

信息几乎没有。

先问：

```text
First Token 慢？

还是后面 Token 慢？
```

就能迅速缩小范围。

---

# 229. First Token 慢

可能：

```text
模型加载
Tokenization
Queue
长 Prompt
Prefill Compute
Prefix Cache Miss
```

---

# 230. 后续 Token 慢

可能：

```text
Memory Bandwidth
Weight Quantization
KV Cache
Decode Kernel
CPU Threads
GPU Utilization
Device Offload
```

---

# 231. 长 Context 后 Token 越来越慢

重点检查：

```text
KV Cache Read
Attention Context Length
Sliding Window
Cache Layout
Memory Pressure
```

---

# 232. 输出突然一卡一卡

Server 环境：

```text
Prefill interruption
Scheduler
Other Requests
GC / memory
Network
```

都会影响 ITL。

---

# 233. 本地模型突然变慢

可能：

```text
Thermal Throttling
Swap
Memory Pressure
CPU Frequency
GPU Frequency
Background Process
```

端侧尤其明显。

---

# 234. Thermal Throttling

连续运行：

```text
NPU / CPU / GPU
```

温度升高。

设备可能：

```text
降频
```

于是：

```text
前 30 秒 20 t/s
后面 12 t/s
```

---

# 235. Edge Benchmark 必须测稳定状态

不能只记录：

```text
第一次最快数字
```

应观察：

```text
1 min
5 min
10 min
```

持续推理性能。

---

# 236. Power

端侧系统还需要：

```text
Watt
```

甚至：

```text
tokens / Joule
```

---

# 237. Tokens per Joule

可以帮助比较：

```text
GPU
NPU
CPU
```

能效。

这对电池设备非常重要。

---

# 238. LLM 推理中的三个“速度”

需要区分：

```text
Prompt Processing Speed
Generation Speed
Serving Throughput
```

---

# 239. Prompt Processing Speed

单位：

```text
prompt tokens/s
```

---

# 240. Generation Speed

单位：

```text
output tokens/s
```

---

# 241. Serving Throughput

单位：

```text
total output tokens/s
requests/s
```

是在：

```text
多用户并发
```

环境。

---

# 242. 单用户 30 t/s vs Server 3000 t/s

不矛盾。

Server：

```text
100 users
```

可能每人：

```text
30 t/s
```

总计：

```text
3000 output t/s
```

---

# 243. Throughput 不能替代用户 Latency

系统：

```text
5000 t/s
```

但单用户：

```text
TTFT 5 秒
```

交互体验仍可能很差。

---

# 244. SLO

生产系统会设置：

```text
TTFT < 1s
TPOT < 50ms
```

之类目标。

然后最大化：

```text
Throughput
```

---

# 245. vLLM Benchmark 的意义

vLLM 当前 benchmark 会明确报告：

```text
TTFT
TPOT
ITL
Request Throughput
Output Token Throughput
```

帮助区分：

```text
用户体验
vs
系统吞吐
```

---

# 246. llama.cpp Benchmark 的意义

更适合：

```text
本地 Runtime
```

快速比较：

```text
Quant
Backend
GPU Layers
Threads
PP
TG
```

---

# 247. 后面 07 章会怎么用？

`07_llama.cpp.md` 会具体看：

```text
GGUF
ggml Tensor
Graph
Backend
Quantized Kernel
KV Cache
Batch / UBatch
llama_decode
Sampling
llama-bench
```

---

# 248. 后面 vLLM 章会怎么用？

重点：

```text
Scheduler
Continuous Batching
PagedAttention
Block Manager
Prefix Caching
Chunked Prefill
Tensor Parallel
Serving Metrics
```

---

# 249. 后面量化章会怎么用？

现在已经知道：

```text
Decode Memory-Bound
```

所以量化章会回答：

```text
FP16
INT8
INT4
Q4_K_M
Block Quant
Scale
Zero Point
Dequant
Quantized MatMul
```

到底怎么改变：

```text
Memory
Bandwidth
Compute
Quality
```

---

# 250. 后面 TVM / Compiler 章会怎么用？

推理 Graph：

```text
RMSNorm
MatMul
RoPE
Attention
Softmax
FFN
```

Compiler：

```text
如何优化 Graph
如何 Lower
如何 Tile
如何 Fuse
如何映射硬件
```

---

# 251. 后面 FlashAttention 章会怎么用？

深入：

```text
标准 Attention
为什么 Memory IO 巨大
```

以及：

```text
Tiling
Online Softmax
SRAM
HBM
Kernel Fusion
```

---

# 252. 后面 RKLLM 章会怎么用？

实际测：

```text
RK3576
```

的：

```text
Prefill t/s
Decode t/s
TTFT
Memory
Context
Quantization
NPU Utilization
```

---

# 253. 本章必做实验 1：手写最小 Generation Loop

使用 Hugging Face 模型：

```python
input_ids = tokenizer(
    prompt,
    return_tensors="pt"
).input_ids

for _ in range(max_new_tokens):

    outputs = model(
        input_ids
    )

    logits = outputs.logits[:, -1, :]

    next_token = torch.argmax(
        logits,
        dim=-1,
        keepdim=True
    )

    input_ids = torch.cat(
        [input_ids, next_token],
        dim=-1
    )
```

先不使用 Cache。

观察：

```text
Sequence 越长
每步越来越慢
```

---

# 254. 实验 2：加入 KV Cache

使用：

```text
past_key_values
```

或 Hugging Face Cache API。

逻辑：

```text
First:
full prompt
↓
past KV

Next:
only new token
+
past KV
```

比较：

```text
Decode Time
```

---

# 255. 实验 3：观察 KV Shape

打印每一层：

```text
K shape
V shape
```

记录：

```text
Layers
Heads
Sequence
Head Dim
```

然后自己算：

```text
KV Memory
```

---

# 256. 实验 4：Context Length 对 KV Memory

测试：

```text
512
1024
2048
4096
```

记录：

```text
Peak GPU / RAM
```

画表：

```text
Context
vs
Memory
```

应该近似看到：

```text
线性增长部分
```

---

# 257. 实验 5：Prefill vs Decode Timing

分别测：

```text
Prompt = 128
512
2048
4096
```

Output：

```text
固定 128
```

记录：

```text
TTFT
TPOT
```

观察：

```text
Prompt Length
```

主要影响谁。

---

# 258. 实验 6：llama-bench

后面配置 GGUF 后：

```text
Prompt Processing
Token Generation
```

分别测。

建立表：

| Quant | PP t/s | TG t/s | Memory |
|---|---:|---:|---:|
| F16 | | | |
| Q8 | | | |
| Q4_K_M | | | |

---

# 259. 实验 7：量化与 Decode

同一个模型：

```text
FP16
Q8
Q4
```

比较：

```text
Model Size
RAM
TG t/s
PP t/s
```

验证：

> Decode 对 Weight Size / Bandwidth 的敏感度。

---

# 260. 实验 8：CPU Threads

llama.cpp CPU：

```text
1
2
4
8
...
```

线程。

记录：

```text
TG t/s
```

观察何时：

```text
收益开始饱和
```

尝试联系：

```text
Memory Bandwidth
```

---

# 261. 实验 9：Batch / UBatch

调整：

```text
batch-size
ubatch-size
```

记录：

```text
Prompt Processing
Memory
```

理解：

```text
并行度 vs 临时内存
```

---

# 262. 实验 10：KV Cache Type

支持 Runtime 中：

```text
FP16
Q8
Q4
```

比较：

```text
Memory
Speed
Output Quality
```

---

# 263. 实验 11：Prefix Cache

固定：

```text
2000 Token System / Document Prefix
```

第一次：

```text
Cold
```

第二次：

```text
Cache Hit
```

比较：

```text
TTFT
```

---

# 264. 实验 12：简单 Serving Benchmark

使用：

```text
vLLM / llama-server
```

并发：

```text
1
2
4
8
```

记录：

```text
TTFT
TPOT
Total Output t/s
```

观察：

```text
Latency / Throughput Tradeoff
```

---

# 265. 实验 13：端侧持续运行 Benchmark

RK3576：

```text
连续 Generate 10 分钟
```

每隔：

```text
30 秒
```

记录：

```text
TG t/s
CPU Temp
NPU Temp
Frequency
RAM
Power
```

观察 Thermal。

---

# 266. 本章建议学习顺序

---

## Stage 1：Generation Loop

先理解：

```text
Logits
Sampling
Append
Repeat
```

不要先学 vLLM。

---

## Stage 2：Prefill vs Decode

理解：

```text
为什么两阶段 Shape 不同
为什么瓶颈不同
```

这是本章最重要的分界。

---

## Stage 3：KV Cache

使用 Hugging Face Cache 文档：

- [Cache Strategies](https://huggingface.co/docs/transformers/kv_cache)
- [Cache Explanation](https://huggingface.co/docs/transformers/main/cache_explanation)

必须亲手观察：

```text
Cache Shape
Memory
Speed
```

---

## Stage 4：Performance Metrics

理解：

```text
TTFT
TPOT
ITL
PP t/s
TG t/s
Throughput
```

---

## Stage 5：Roofline / Memory Bandwidth

阅读：

- [NVIDIA Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [Hardware-Friendly LLM Design](https://developer.nvidia.com/blog/ai-model-co-design-hardware-friendly-llm-design/)

建立：

```text
Arithmetic Intensity
Compute-Bound
Memory-Bound
```

---

## Stage 6：llama.cpp Benchmark

阅读：

- [CLI README](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
- [llama-bench Source](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)
- [Batched Bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md)

理解：

```text
pp
tg
batch
ubatch
ctx
KV type
```

---

## Stage 7：Serving Concepts

开始读 vLLM：

- [vLLM](https://github.com/vllm-project/vllm)
- [Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)

先只理解：

```text
Scheduler
Continuous Batching
Paged KV
Chunked Prefill
```

不要在本章深入源码。

---

# 267. 推荐学习优先级

| 内容 | 优先级 |
|---|---|
| Prefill | ★★★★★ |
| Decode | ★★★★★ |
| KV Cache | ★★★★★ |
| TTFT / TPOT | ★★★★★ |
| Prompt Processing vs Token Generation | ★★★★★ |
| Memory-Bound vs Compute-Bound | ★★★★★ |
| Arithmetic Intensity | ★★★★★ |
| Weight Memory | ★★★★★ |
| Quantization 对 Decode 的作用 | ★★★★★ |
| Context Length | ★★★★★ |
| Batch | ★★★★ |
| KV Cache Quantization | ★★★★ |
| Prefix Cache | ★★★★ |
| Continuous Batching | ★★★★ |
| Paged KV Cache | ★★★★ |
| Chunked Prefill | ★★★★ |
| FlashAttention | ★★★★ |
| Speculative Decoding | ★★★ |
| Tensor Parallel | ★★★ |
| Pipeline Parallel | ★★ |

---

# 268. 常见误区

---

## 误区 1：LLM 推理就是一次 Forward

错误。

生成 100 个 Token：

```text
Prefill
+
100 次左右 Decode Step
```

---

## 误区 2：Prefill = Decode

错误。

它们：

```text
Shape
Parallelism
Arithmetic Intensity
Hardware Bottleneck
```

都不同。

---

## 误区 3：KV Cache 保存整个 Hidden State

不准确。

标准 Attention Cache 的核心：

```text
每层历史 K / V
```

---

## 误区 4：有 KV Cache 后 Decode 与 Context Length 无关

错误。

新 Q：

```text
仍需 Attention over past K/V
```

---

## 误区 5：TOPS 越高 LLM 一定越快

错误。

Decode 常受：

```text
Memory Bandwidth
```

限制。

---

## 误区 6：INT4 只是省空间

错误。

它还：

```text
减少 Weight Memory Traffic
```

因此可能提升 Decode。

---

## 误区 7：量化后一定快 4 倍

错误。

还取决于：

```text
Kernel
Dequant
Other Operators
Bandwidth
Hardware
```

---

## 误区 8：模型文件 1GB 就只需要 1GB RAM

错误。

还需要：

```text
KV
Workspace
Activation
Runtime
```

---

## 误区 9：30 tokens/s 可以直接跨模型比较

不严格。

Tokenizer 不同。

---

## 误区 10：Server 3000 tokens/s = 每用户 3000 tokens/s

错误。

可能是：

```text
多用户总 Throughput
```

---

## 误区 11：PagedAttention 改变了 Attention 数学

核心并不是。

它主要改变：

```text
KV Memory Management / Access
```

---

## 误区 12：FlashAttention 是 Sparse Attention

不是。

FlashAttention 核心：

```text
IO-aware exact attention implementation
```

---

# 269. 本章知识树

```text
LLM Inference
│
├── Input
│   ├── Chat Template
│   ├── Tokenizer
│   └── Token IDs
│
├── Model Loading
│   ├── Weight
│   ├── mmap
│   ├── Device Memory
│   └── Backend Buffer
│
├── Prefill
│   ├── Prompt Processing
│   ├── Large GEMM
│   ├── Attention
│   ├── KV Construction
│   ├── First Logits
│   └── TTFT
│
├── Decode
│   ├── One New Token
│   ├── Small GEMM / GEMV-like
│   ├── Read Weights
│   ├── Read KV
│   ├── New Logits
│   ├── Sampling
│   └── TPOT / ITL
│
├── KV Cache
│   ├── K
│   ├── V
│   ├── Layers
│   ├── Context
│   ├── GQA / MQA
│   ├── Quantization
│   └── Offloading
│
├── Performance
│   ├── PP t/s
│   ├── TG t/s
│   ├── TTFT
│   ├── TPOT
│   ├── ITL
│   ├── Throughput
│   └── Latency
│
├── Hardware
│   ├── Compute
│   ├── Memory Capacity
│   ├── Memory Bandwidth
│   ├── Arithmetic Intensity
│   └── Roofline
│
├── Runtime Optimization
│   ├── Quantization
│   ├── Prefix Cache
│   ├── Continuous Batching
│   ├── Paged KV
│   ├── Chunked Prefill
│   ├── FlashAttention
│   └── Speculative Decode
│
└── Serving
    ├── Scheduler
    ├── Request Queue
    ├── Concurrency
    ├── Batch
    └── Streaming
```

---

# 270. 本章最重要的思维模型

## 思维模型 1

```text
LLM Generation
=
Prefill
+
Decode Loop
```

---

## 思维模型 2

```text
Prefill
=
Many Tokens at Once
=
High Parallelism
=
Often Compute-heavy
```

---

## 思维模型 3

```text
Decode
=
One Token at a Time
=
Low Arithmetic Intensity
=
Often Memory-bandwidth-heavy
```

---

## 思维模型 4

```text
KV Cache
=
Reuse Past Attention K/V
```

---

## 思维模型 5

```text
Context Length ↑
=
KV Memory ↑
+
Prefill Work ↑
+
Decode Attention Work ↑
```

---

## 思维模型 6

```text
Quantization
=
Memory Capacity Optimization
+
Memory Bandwidth Optimization
+
Low-precision Compute Optimization
```

---

## 思维模型 7

```text
TOPS
≠
Tokens/s
```

---

## 思维模型 8

```text
Interactive Performance
=
TTFT
+
TPOT / ITL
```

---

## 思维模型 9

```text
Server Performance
=
Latency
+
Throughput
+
Concurrency
+
Scheduler
+
KV Management
```

---

## 思维模型 10

```text
LLM Optimization
=
Model
+
Runtime
+
Compiler
+
Kernel
+
Hardware
```

---

# 271. 本章完成标准

如果下面大部分都可以自己解释，本章就可以结束。

## 推理流程

- [ ] 能画出完整 LLM Generation Loop
- [ ] 能解释 Chat Template
- [ ] 能解释 Tokenizer 所在位置
- [ ] 能解释 Model Loading
- [ ] 能解释 Prefill
- [ ] 能解释 Decode
- [ ] 能解释 Sampling
- [ ] 能解释为什么 Generation 必须 Autoregressive

## Prefill / Decode

- [ ] 能解释为什么 Prefill 可以并行
- [ ] 能解释为什么 Decode 逐 Token
- [ ] 能解释 Prefill 与 Decode Tensor Shape 的差异
- [ ] 能解释为什么 Prefill 常有大 GEMM
- [ ] 能解释为什么 Decode 更像 GEMV / Small GEMM
- [ ] 能解释为什么两阶段需要不同 Kernel

## KV Cache

- [ ] 能解释 KV Cache 缓存什么
- [ ] 能解释为什么缓存 K / V
- [ ] 能解释为什么不需要缓存历史 Q
- [ ] 能画出 Layer-specific KV
- [ ] 能写出 KV Cache Memory 粗略公式
- [ ] 能解释 Context 对 KV 的影响
- [ ] 能解释 Batch 对 KV 的影响
- [ ] 能解释 GQA 为什么减少 KV
- [ ] 能解释 KV Quantization
- [ ] 能解释 KV Offload

## Performance Metrics

- [ ] 能解释 Prompt Processing t/s
- [ ] 能解释 Token Generation t/s
- [ ] 能解释 TTFT
- [ ] 能解释 TPOT
- [ ] 能解释 ITL
- [ ] 能解释 End-to-End Latency
- [ ] 能解释 Throughput
- [ ] 能说明单用户 t/s 和 Server Throughput 的区别

## Hardware

- [ ] 能解释 Compute-Bound
- [ ] 能解释 Memory-Bound
- [ ] 能解释 Arithmetic Intensity
- [ ] 能解释 Roofline Model
- [ ] 能解释为什么 Decode 常受 Memory Bandwidth 限制
- [ ] 能解释为什么 Prefill 更容易利用计算单元
- [ ] 能解释 Memory Capacity vs Bandwidth
- [ ] 能解释为什么 TOPS 不能直接推导 Tokens/s

## Quantization

- [ ] 能解释 FP16 / INT8 / INT4 对 Weight Memory 的粗略影响
- [ ] 能解释为什么 INT4 可能提升 Decode
- [ ] 能解释为什么速度不会严格按 bit 数比例提升
- [ ] 能解释量化对 Prefill 和 Decode 影响可能不同
- [ ] 能解释 Weight Quant 与 KV Quant 是两件事

## Runtime

- [ ] 能解释 Prefix Cache
- [ ] 能解释 Prompt Cache
- [ ] 能解释 Continuous Batching
- [ ] 能解释 Paged KV Cache
- [ ] 能解释 PagedAttention 的核心目的
- [ ] 能解释 Chunked Prefill
- [ ] 能解释 FlashAttention 在优化什么
- [ ] 能解释 Speculative Decoding

## Benchmark

- [ ] 知道 llama.cpp `pp` 与 `tg` 的含义
- [ ] 能记录 Input / Output Length
- [ ] 能记录 Quant Type
- [ ] 能记录 Context Size
- [ ] 能记录 Runtime Version
- [ ] 能分别 Benchmark Prefill / Decode
- [ ] 能测 Peak Memory
- [ ] 最好完成一次 Context vs KV Memory 实验
- [ ] 最好完成一次 Quant vs Decode Speed 实验

## Edge AI

- [ ] 能解释为什么端侧 LLM 特别关注 DDR Bandwidth
- [ ] 能解释为什么模型能装下不代表速度快
- [ ] 能解释共享 DDR SoC 的带宽竞争
- [ ] 能解释 Thermal Throttling
- [ ] 能解释为什么需要持续性能测试
- [ ] 能说明 RKLLM 为什么需要特殊 LLM Runtime

---

# 272. 本章暂时不要求深入

暂时不要求：

- [ ] 手写完整 Attention CUDA Kernel
- [ ] 精通 FlashAttention 数学实现
- [ ] 精通 vLLM Scheduler 源码
- [ ] 精通 PagedAttention CUDA Kernel
- [ ] 精通 llama.cpp C++ 源码
- [ ] 精通 GGUF
- [ ] 精通 Q4_K_M Block Layout
- [ ] 精通 Tensor Parallel
- [ ] 精通 Pipeline Parallel
- [ ] 精通 Speculative Decoding 所有算法
- [ ] 精通 RKLLM
- [ ] 精通 AI Compiler

这些后续逐章展开。

---

# 273. 核心参考链接

## llama.cpp

- [llama.cpp Repository](https://github.com/ggml-org/llama.cpp)
- [CLI README](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md)
- [KV Cache Source](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.h)
- [llama-bench Source](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)
- [Batched Bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md)

重点关键词：

```text
ctx-size
batch-size
ubatch-size
flash-attn
cache-type-k
cache-type-v
kv-offload
parallel
prompt processing
token generation
```

---

## vLLM

- [vLLM Repository](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Benchmark CLI](https://docs.vllm.ai/en/latest/benchmarking/cli/)
- [Chunked Prefill / Paged Decode Source Docs](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/ops/chunked_prefill_paged_decode/)

重点：

```text
TTFT
TPOT
ITL
Scheduler
Paged KV
Continuous Batching
Chunked Prefill
```

---

## Hugging Face Transformers

- [KV Cache Strategies](https://huggingface.co/docs/transformers/kv_cache)
- [Caching Explanation](https://huggingface.co/docs/transformers/main/cache_explanation)

重点：

```text
Dynamic Cache
Static Cache
Quantized Cache
Offloaded Cache
```

---

## NVIDIA

- [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
- [AI Model Co-Design: Hardware-Friendly LLM Design](https://developer.nvidia.com/blog/ai-model-co-design-hardware-friendly-llm-design/)
- [Long-context Inference Optimization](https://developer.nvidia.com/blog/accelerating-long-context-inference-with-skip-softmax-in-nvidia-tensorrt-llm/)

重点：

```text
Prefill
Decode
Memory Bound
Compute Bound
Arithmetic Intensity
KV Cache
Memory Bandwidth
```

---

# 274. 推荐阅读组合

```text
Hugging Face Cache Docs
=
把 KV Cache 机制看清

NVIDIA Inference Optimization
=
把 Prefill / Decode 的硬件特征看清

llama.cpp
=
把单机 Runtime 参数与实际 Benchmark 对应起来

vLLM
=
把单请求推理扩展到多用户 Serving
```

---

# 275. 与上一章 Agent 的关系

上一章：

```text
Agent
↓
调用 LLM
↓
得到结果
```

本章：

> 把“调用一次 LLM”拆开。

```text
Agent LLM Call
      ↓
Prompt
      ↓
Prefill
      ↓
First Token
      ↓
Decode × N
      ↓
Response
```

因此 Agent：

```text
调用次数越多
```

底层推理成本：

```text
累积越大
```

---

# 276. 与下一章 llama.cpp 的关系

本章回答：

```text
推理为什么需要：
KV Cache
Quantization
Batch
Backend
FlashAttention
```

下一章开始回答：

> llama.cpp 到底是怎么把这些东西实现出来的？

---

# 277. 与 vLLM 的关系

本章已经建立：

```text
单用户：
Prefill + Decode
```

vLLM 要解决：

```text
100 个用户
都有自己的 Prefill + Decode
```

怎么：

```text
排队
批处理
管理 KV
利用 GPU
```

---

# 278. 与 AI Compiler / Kernel 的关系

Runtime 看到：

```text
MatMul
RMSNorm
RoPE
Attention
Softmax
FFN
```

下一层：

```text
Compiler
```

决定：

```text
如何组织 Graph
```

再下一层：

```text
Kernel
```

决定：

```text
怎么在硬件上高速执行
```

---

# 279. 最后建立一张完整大图

```text
                      LLM Inference
                           │
                           ▼
                      Input Text
                           │
                           ▼
                    Chat Template
                           │
                           ▼
                       Tokenizer
                           │
                           ▼
                      Token IDs
                           │
                           ▼
                    ┌────────────┐
                    │  Prefill   │
                    └─────┬──────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    Large GEMM       Attention          Build KV
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
                       Logits
                          │
                          ▼
                       Sampling
                          │
                          ▼
                    First Token
                          │
                          ▼
                    ┌────────────┐
                    │   Decode   │◄──────────────┐
                    └─────┬──────┘               │
                          │                      │
             ┌────────────┼────────────┐         │
             ▼            ▼            ▼         │
        Read Weight    Read KV      New K/V      │
             │            │            │         │
             └────────────┼────────────┘         │
                          ▼                      │
                       Logits                    │
                          │                      │
                          ▼                      │
                       Sampling                  │
                          │                      │
                          ▼                      │
                      Next Token ────────────────┘
                          │
                          ▼
                    Stop Condition
                          │
                    ┌─────┴─────┐
                    ▼           ▼
                  Stop        Continue
```

---

# 280. 一句话总结

本章真正需要彻底理解的只有五件事。

第一：

```text
LLM Generation
=
Prefill
+
Decode
```

第二：

```text
Prefill
=
一次处理很多 Prompt Token
=
高并行度
=
通常更偏 Compute
```

第三：

```text
Decode
=
一个 Token 一个 Token
=
每步仍要访问大量 Weight
=
通常更偏 Memory Bandwidth
```

第四：

```text
KV Cache
=
保存历史 Token 的 K/V
=
避免重复计算
=
用 Memory 换 Compute
```

第五：

```text
LLM 推理性能
不是一个 Tokens/s 数字

而是：

TTFT
+
TPOT / ITL
+
Throughput
+
Memory
+
Context
+
Quantization
+
Hardware
```

当这五件事真正理解以后，接下来学习：

> **07_llama.cpp.md**

就不再是在学习一堆命令。

而是在学习：

> **一个真正的 LLM Runtime，是如何用 C/C++、量化 Tensor、计算图、KV Cache、Backend 和高性能 Kernel，把这一整套推理机制在 CPU / GPU 上实现出来的。**
