# 07_llama.cpp

> **所属路线**：AI 学习路线 · 第二部分  
> **定位**：从 LLM 推理原理进入真正的 C/C++ Runtime，理解 GGUF、ggml、量化 Tensor、计算图、Backend、KV Cache、Sampling、CPU/GPU Offload 与 Benchmark 是如何在 `llama.cpp` 中落地的  
> **核心仓库**：`ggml-org/llama.cpp`  
> **底层项目**：`ggml-org/ggml`（代码已作为 llama.cpp 的核心子目录/基础库体系存在）  
> **学习边界**：本章重点理解 llama.cpp Runtime 本身；`vLLM / MLC-LLM`、AI Compiler、FlashAttention/CUTLASS、RKNN/RKLLM 分别在后续章节展开。  
> **资料检查日期**：2026-08-25

---

# 0. 本章学习目标

完成本章以后，需要能够回答：

1. `llama.cpp` 到底是什么？它不是模型、不是量化格式、也不是 GGUF。
2. `ggml`、`GGUF`、`llama.cpp`、`Q4_K_M` 分别是什么关系？
3. 为什么 `llama.cpp` 能不依赖 PyTorch 就直接运行 LLM？
4. Hugging Face 模型为什么通常需要转换成 GGUF？
5. GGUF 文件里到底存了什么？
6. GGUF 为什么不仅是 Weight 文件？
7. GGUF 的 Metadata、Tensor Info、Tensor Data 分别是什么？
8. `Q4_0`、`Q4_K_M`、`Q8_0` 等名字在 llama.cpp 生态中代表什么？
9. `Q4_K_M` 为什么不是“模型架构”？
10. `convert_hf_to_gguf.py` 与 `llama-quantize` 分别做什么？
11. 为什么推荐从 FP16/BF16 高质量 GGUF 再量化，而不是反复 Re-quantize？
12. `ggml_tensor` 是什么？
13. `ggml` 为什么既像 Tensor Library，又像轻量计算图系统？
14. `ggml_cgraph` 是什么？
15. `Operator` 在 ggml 中如何表示？
16. llama.cpp 的 Model Loader 如何从 GGUF 中读取 Metadata 与 Tensor？
17. 为什么当前源码已经拆成 `llama-model-loader.cpp`、`llama-model.cpp`、`llama-context.cpp`、`llama-graph.cpp` 等多层？
18. `llama_model` 和 `llama_context` 有什么区别？
19. 为什么一个 Model 可以对应多个 Context？
20. `llama_decode()` 到底处在什么位置？
21. `llama_batch` 是什么？
22. `n_batch`、`n_ubatch`、`n_ctx` 分别解决什么问题？
23. llama.cpp 如何区分 Prompt Processing 与 Token Generation？
24. 为什么源码会为 PP Graph 和 TG Graph 分别 Reserve Buffer？
25. KV Cache 在 llama.cpp 中由谁管理？
26. `--cache-type-k` / `--cache-type-v` 为什么可以单独设置？
27. `--no-kv-offload` / KV offload 的含义是什么？
28. llama.cpp 的 Sampling 为什么被拆成 sampler chain？
29. Top-K、Top-P、Temperature、Grammar 等为什么可以组合成 chain？
30. `llama-cli` 和 `llama-server` 的定位有什么区别？
31. `llama-simple` 为什么是源码学习最好的入口之一？
32. `llama-bench` 中 `pp512`、`tg128` 分别表示什么？
33. 为什么 llama.cpp 的 CPU 推理特别依赖量化 Kernel 和 Memory Bandwidth？
34. BLAS 为什么对 Prompt Processing 常有价值，但不一定主导单 Token Decode？
35. CPU Backend、CUDA、Metal、Vulkan、SYCL、OpenCL 等 Backend 处于哪一层？
36. Backend 和 Operator 有什么区别？
37. ggml Backend Scheduler 在解决什么？
38. GPU Offload 的 `-ngl / --gpu-layers` 大致意味着什么？
39. 为什么 Offload 并不是“把 Python 模型搬到 GPU”？
40. llama.cpp 为什么可以在 ARM Linux 上成为很好的 Edge AI 学习 Runtime？
41. 为什么 llama.cpp 并不等于 Rockchip NPU 路线？
42. 在 RK3576 上跑 llama.cpp 与使用 RKLLM 的根本区别是什么？
43. 怎样按照正确顺序阅读 llama.cpp 源码，而不是一上来陷入几千行代码？
44. 如何建立从 CLI 参数 → Context Params → Graph → Backend → Kernel 的完整映射？

本章最终要建立以下完整软件栈：

```text
Hugging Face Model
        ↓
convert_hf_to_gguf.py
        ↓
High-quality GGUF
        ↓
llama-quantize
        ↓
Quantized GGUF
        ↓
llama.cpp
        │
        ├── Model Loader
        ├── Vocabulary / Tokenizer
        ├── llama_model
        ├── llama_context
        ├── KV / Memory
        ├── Graph Builder
        ├── Sampler
        └── Backend Scheduler
        ↓
       ggml
        │
        ├── Tensor
        ├── Operator
        ├── Graph
        ├── Allocator
        └── Backend
        ↓
CPU / CUDA / Metal / Vulkan / SYCL / ...
        ↓
Kernel
        ↓
Hardware
```

---

# 1. 核心 GitHub 仓库

主仓库：

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

建议首先收藏以下文件。

---

## 1.1 README

- [llama.cpp README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)

README 可以看到：

```text
Supported Models
Supported Backends
Build
Model Conversion
Quantization
CLI
Server
Benchmark
Examples
```

它负责：

> 建立整个项目的外部地图。

---

## 1.2 最小示例

- [examples/simple/simple.cpp](https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/simple.cpp)

这是本章最推荐先读的源码。

因为它把完整流程压缩成：

```text
load backend
↓
load model
↓
get vocab
↓
tokenize
↓
create context
↓
create sampler
↓
decode prompt
↓
sample token
↓
decode next token
↓
repeat
```

不要一开始看 `llama-model.cpp`。

先看：

> `simple.cpp`

---

## 1.3 公共 API

- [include/llama.h](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)

它相当于：

> llama.cpp 对外暴露的核心 C API。

看到源码实现前，先知道外部用户实际调用哪些接口。

重点搜索：

```text
llama_model_load_from_file

llama_model_default_params

llama_context_default_params

llama_init_from_model

llama_tokenize

llama_decode

llama_sampler

llama_token_to_piece
```

---

# 2. 先彻底分清四个概念

```text
llama.cpp
ggml
GGUF
Q4_K_M
```

这是最容易长期混淆的一组概念。

---

# 3. llama.cpp 是什么？

可以把 llama.cpp 定义为：

> **一个以 C/C++ 为核心、面向本地和跨硬件 LLM 推理的 Runtime / Library / Tool Ecosystem。**

它负责：

```text
Model Loading
Tokenizer
Model Architecture
Graph Construction
KV Cache
Inference
Sampling
Backend Dispatch
CLI
Server
Benchmark
Quantization Tools
```

所以：

```text
llama.cpp
≠
模型
```

也：

```text
llama.cpp
≠
量化格式
```

更：

```text
llama.cpp
≠
GGUF
```

---

# 4. ggml 是什么？

ggml 可以先理解为：

> **llama.cpp 底层使用的轻量 Tensor + Operator + Computation Graph + Backend 基础设施。**

它向上支持：

```text
llama.cpp
```

向下连接：

```text
CPU
CUDA
Metal
Vulkan
SYCL
OpenCL
...
```

可以粗略类比：

```text
PyTorch:
Tensor + ATen + Dispatcher + Backend

llama.cpp:
ggml Tensor + Graph + Backend
```

当然二者规模和目标完全不同。

---

# 5. GGUF 是什么？

GGUF：

> **ggml 生态使用的二进制模型容器格式。**

它保存：

```text
Header
Metadata
Tensor Descriptions
Tensor Binary Data
```

因此：

```text
GGUF
=
File Format / Container
```

不是 Runtime。

---

# 6. Q4_K_M 是什么？

`Q4_K_M`：

> llama.cpp / ggml 生态中的一种 K-quant 混合量化方案。

它描述：

```text
模型中不同 Tensor
以怎样的低比特方式保存
```

所以：

```text
Q4_K_M
=
Quantization Scheme
```

不是：

```text
GGUF
```

也不是：

```text
llama.cpp
```

---

# 7. 四者关系

```text
                llama.cpp
                    │
      Runtime / Model Implementation
                    │
                    ▼
                   ggml
       Tensor / Graph / Backend / Kernel
                    │
            ┌───────┴────────┐
            ▼                ▼
          GGUF          Quant Types
      Model Container     Q4_K_M
                          Q8_0
                          ...
```

---

# 8. 一个更加准确的数据流

```text
Qwen / LLaMA / Gemma
Hugging Face Checkpoint
        ↓
convert_hf_to_gguf.py
        ↓
F16 / BF16 GGUF
        ↓
llama-quantize
        ↓
Q4_K_M GGUF
        ↓
llama.cpp Runtime
        ↓
ggml Graph
        ↓
Backend
        ↓
Kernel
        ↓
Hardware
```

---

# 9. 为什么 llama.cpp 不直接读取普通 PyTorch Model？

PyTorch Model：

```text
Python Model Code
+
Config
+
Safetensors
+
Framework Semantics
```

llama.cpp 不带完整：

```text
PyTorch Runtime
Python Interpreter
ATen Training Framework
```

它使用自己实现的：

```text
Model Architecture
Tensor Representation
Graph
Kernel
```

所以需要：

```text
转换成 llama.cpp 能直接解释的模型容器。
```

---

# 10. Hugging Face 模型通常包含什么？

典型：

```text
config.json

model.safetensors

tokenizer.json

tokenizer_config.json

generation_config.json
```

而转换到 GGUF：

```text
Architecture Metadata
Tokenizer Metadata
Tensor Name / Shape / Type
Tensor Data
```

被重新组织。

---

# 11. `convert_hf_to_gguf.py`

核心脚本：

- [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)

它负责：

```text
Read HF Config
Read Tokenizer
Read Weights
Map Tensor Names
Convert Tensor Layout / Type if needed
Write GGUF Metadata
Write GGUF Tensors
```

---

# 12. Conversion 和 Quantization 不是一回事

Conversion：

```text
HF Format
→
GGUF
```

Quantization：

```text
High Precision Tensor
→
Low-bit Quantized Tensor
```

所以通常：

```text
HF
↓ conversion
F16/BF16 GGUF
↓ quantization
Q4/Q5/Q8 GGUF
```

---

# 13. 为什么先得到高质量 GGUF 再量化？

因为 Quantization：

```text
会损失信息
```

如果：

```text
Q4
↓
再 Q2
```

不是从原始 FP16 直接 Q2。

误差可能：

```text
累积
```

llama.cpp 的 quantize 文档也明确警告：

> Re-quantizing 已量化 Tensor 可能显著降低质量。

所以最佳实践：

```text
Original FP16/BF16
↓
Target Quant
```

---

# 14. llama-quantize

官方文档：

- [tools/quantize/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

典型：

```bash
./build/bin/llama-quantize \
    model-f16.gguf \
    model-Q4_K_M.gguf \
    Q4_K_M
```

这一步：

```text
读取 GGUF
↓
按 Quant Scheme 重新编码 Tensor
↓
写新的 GGUF
```

---

# 15. Quantization 仍然输出 GGUF

非常重要：

```text
F16.gguf
```

和：

```text
Q4_K_M.gguf
```

都是：

```text
GGUF 文件
```

区别：

```text
Tensor dtype / quant format
```

不同。

---

# 16. GGUF 的基本文件结构

官方 gguf.h 对结构定义非常清楚。

可以概括：

```text
GGUF File
│
├── Magic
│   └── "GGUF"
│
├── Version
│
├── Tensor Count
│
├── KV Metadata Count
│
├── Key-Value Metadata
│
├── Tensor Infos
│   ├── Name
│   ├── Dimension
│   ├── Shape
│   ├── ggml_type
│   └── Data Offset
│
├── Padding / Alignment
│
└── Tensor Binary Data
```

参考：

- [ggml/include/gguf.h](https://github.com/ggml-org/ggml/blob/master/include/gguf.h)
- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)

---

# 17. GGUF Magic

文件开始：

```text
"GGUF"
```

用于：

```text
识别文件格式
```

当前规范中：

```text
GGUF_VERSION
```

也记录格式版本。

---

# 18. Metadata

GGUF 不只是：

```text
Weight
```

它还可以保存：

```text
general.architecture
general.name
general.quantization_version
context length
embedding length
layer count
attention head count
rope parameters
tokenizer tokens
special token IDs
chat template
...
```

---

# 19. 为什么 Metadata 很重要？

Runtime 加载 Model 前必须知道：

```text
这是 Qwen 还是 LLaMA？

多少 Layer？

Hidden Size？

Head 数？

RoPE 参数？

Tokenizer 是什么？

EOS Token 是多少？
```

这些信息：

```text
不能只从裸 Weight 数组猜。
```

---

# 20. Tensor Info

每个 Tensor：

```text
name
shape
dtype / quant type
offset
```

例如概念上：

```text
blk.0.attn_q.weight

shape:
[...]
type:
Q4_K

offset:
...
```

---

# 21. Tensor Data

真正 Weight：

```text
binary bytes
```

放在 GGUF 后部。

每个 Tensor Data：

```text
按 alignment 对齐
```

便于 Runtime 读取 / mmap。

---

# 22. 为什么 GGUF 适合 mmap？

因为 Tensor：

```text
在文件中有明确 offset
```

并按：

```text
alignment
```

组织。

Runtime 可以：

```text
把文件映射进进程地址空间
```

减少传统 Copy。

---

# 23. GGUF ≠ GGML

历史上有人说：

```text
ggml model
```

现在需要更准确：

```text
ggml
=
计算库 / tensor ecosystem

GGUF
=
模型文件格式
```

---

# 24. GGUF 为什么适合 llama.cpp？

它把 llama.cpp 推理需要的重要信息统一到：

```text
一个或多个模型文件
```

包括：

```text
Weights
Architecture
Tokenizer
Metadata
Quantization Information
```

因此 Deployment：

```text
更简单。
```

---

# 25. GGUF 可以分片

大型模型可能：

```text
model-00001-of-00004.gguf
...
```

Model Loader 会处理：

```text
split files
```

当前 `llama-model-loader.cpp` 已包含：

```text
split count
split no
多个 GGUF context
```

等逻辑。

---

# 26. Model Loader

源码：

- [src/llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp)

它负责：

```text
Open GGUF
↓
Read Metadata
↓
Detect Architecture
↓
Index Tensor
↓
Handle Split Files
↓
mmap / read
↓
Prepare Tensor Loading
```

---

# 27. Model Loader 的第一阶段

典型：

```text
gguf_init_from_file
```

先读取：

```text
Header
Metadata
Tensor descriptors
```

而不一定立刻分配所有 Tensor Data。

---

# 28. `no_alloc`

GGUF 初始化：

```text
可以只建立 Tensor metadata
```

而不马上：

```text
allocate tensor data
```

这是 Graph / Tensor 系统很常见的设计。

---

# 29. 为什么先 Metadata 后 Weight？

因为 Runtime 要先知道：

```text
Model Architecture
Tensor Names
Tensor Shapes
Tensor Types
Backend Placement
```

才能决定：

```text
怎么分配 Buffer。
```

---

# 30. `llama_model`

可以把：

```text
llama_model
```

理解为：

> 一个已加载的、基本不可变的模型本体。

它包含：

```text
Architecture
Hyperparameters
Vocabulary
Weight Tensors
Backend Buffers
Model Metadata
```

---

# 31. `llama_context`

`llama_context`：

> 一次或一组推理 Session 的运行状态。

它包含更多：

```text
Context Params
KV / Memory State
Scheduler
Compute Buffers
Outputs
Sampling-related runtime state
```

---

# 32. Model vs Context

非常重要：

```text
llama_model
=
Weights / Architecture

llama_context
=
Inference Runtime State
```

---

# 33. 为什么要分 Model 与 Context？

同一个：

```text
Model Weight
```

可以服务：

```text
多个 Context / Session
```

概念上：

```text
              llama_model
            /      |      \
           ↓       ↓       ↓
       context A context B context C
```

每个有自己：

```text
KV State
Sequence
Context
```

---

# 34. 对比 Hugging Face

可以粗略：

```text
HF model
≈
llama_model

generation state / past_key_values
≈
llama_context 中的一部分运行状态
```

并非严格一一对应，但很有帮助。

---

# 35. Model Loading API

`simple.cpp` 中：

```cpp
llama_model_params model_params =
    llama_model_default_params();

llama_model * model =
    llama_model_load_from_file(
        model_path.c_str(),
        model_params
    );
```

最值得先理解：

```text
Model Params
↓
Load GGUF
↓
llama_model
```

---

# 36. `n_gpu_layers`

Model Params 中非常常见：

```text
n_gpu_layers
```

CLI：

```text
-ngl
--gpu-layers
```

概念：

> 尝试将一定数量的模型 Layer / Tensor 放到 GPU Backend。

---

# 37. GPU Offload 并不是“GPU 运行 Python”

llama.cpp 自己：

```text
Build ggml Graph
```

然后：

```text
Backend Scheduler
```

决定 Graph Node / Tensor：

```text
CPU
GPU
```

上的执行。

---

# 38. 为什么叫 Offload？

因为默认：

```text
CPU
```

是基本执行 Backend。

然后把：

```text
部分计算 / Weight
```

交给：

```text
GPU Backend
```

---

# 39. Vocabulary

```cpp
const llama_vocab * vocab =
    llama_model_get_vocab(model);
```

Vocabulary：

```text
来自 GGUF Tokenizer Metadata
```

用于：

```text
tokenize
token → piece
special tokens
```

---

# 40. Tokenize

`simple.cpp`：

```text
String Prompt
↓
llama_tokenize
↓
std::vector<llama_token>
```

所以：

```text
llama_token
```

本质上是：

```text
Token ID
```

---

# 41. 为什么先调用一次 tokenize 获取 size？

C API 常见模式：

```text
第一次：
buffer = null
得到需要的 token 数量

第二次：
分配 vector
真正 tokenize
```

---

# 42. Token → Piece

生成 Token 后：

```cpp
llama_token_to_piece(...)
```

将：

```text
Token ID
```

恢复为：

```text
Text Piece
```

然后 Streaming 输出。

---

# 43. Context Params

```cpp
llama_context_params ctx_params =
    llama_context_default_params();
```

重要参数：

```text
n_ctx
n_batch
n_ubatch
threads
flash attention
kv cache type
...
```

---

# 44. `n_ctx`

```text
Context Size
```

用于决定：

```text
可容纳多少 Token 的上下文 / Memory。
```

会直接影响：

```text
KV Cache
```

---

# 45. `n_batch`

`simple.cpp` 中注释非常明确：

```text
maximum number of tokens
that can be processed
in a single call to llama_decode
```

可以理解成：

> 一次 logical decode call 允许处理的最大 Token 数。

---

# 46. `n_ubatch`

Physical UBatch：

> Scheduler / Graph 实际一次处理的 Token Chunk 上限。

可以粗略：

```text
n_batch
=
逻辑 Batch 上限

n_ubatch
=
物理 Micro-batch 上限
```

---

# 47. 为什么要有 UBatch？

假设 Prompt：

```text
4096 Tokens
```

并不一定一次建一个：

```text
4096-token compute graph
```

可以：

```text
512
512
512
...
```

分块执行。

---

# 48. UBatch 的价值

平衡：

```text
Compute Parallelism
vs
Activation Memory
vs
Graph Buffer Size
```

---

# 49. 创建 Context

```cpp
llama_context * ctx =
    llama_init_from_model(
        model,
        ctx_params
    );
```

这一步会：

```text
根据 Model
+
Runtime Params
```

创建推理运行环境。

---

# 50. Context 创建时发生什么？

概念上包括：

```text
Initialize Memory Module
Allocate KV / recurrent memory
Initialize Backend Scheduler
Reserve Compute Buffers
Prepare Output Buffers
Resolve Backend Capability
```

---

# 51. 当前源码非常值得注意的一点

`llama-context.cpp` 中：

```text
会分别 Reserve：
pp graph
tg graph
```

即：

```text
Prompt Processing
Token Generation
```

图。

这正好对应上一章：

```text
Prefill
Decode
```

---

# 52. 为什么分别 Reserve PP / TG Graph？

因为两者：

```text
Tensor Shape
Node Count
Memory Need
Graph Split
```

不同。

所以 Runtime 在初始化阶段：

```text
提前为 worst case graph reserve buffer
```

减少推理过程中：

```text
反复 realloc
```

---

# 53. 这是非常重要的 Runtime 思维

高性能 Runtime：

```text
不是每生成一个 Token
都临时 new 一堆 Tensor Buffer。
```

而是：

```text
提前规划 / reserve
循环复用
```

---

# 54. `llama_batch`

推理输入不是简单：

```text
vector<int>
```

而使用：

```text
llama_batch
```

因为 Runtime 需要表达：

```text
Token
Position
Sequence ID
Logits Requirement
Embedding
```

等信息。

---

# 55. 为什么 Batch 里需要 Position？

RoPE / Position Encoding：

```text
需要知道每个 Token
处在哪个位置。
```

---

# 56. 为什么需要 Sequence ID？

一个 Context / Batch：

```text
可能处理多个 Sequence。
```

Runtime 要知道：

```text
哪个 Token 属于哪个 Sequence。
```

---

# 57. Prompt Processing

Prompt：

```text
N Tokens
```

构造 Batch：

```text
Token 0
Token 1
...
Token N-1
```

送入：

```text
llama_decode
```

---

# 58. `llama_decode()`

这是整个 llama.cpp 最关键的 API 之一。

从外部看：

```text
Batch
↓
llama_decode
↓
Model Graph Execution
↓
Logits / Embeddings
↓
Memory / KV Update
```

---

# 59. `llama_decode` 不是“只解码文字”

名字很容易误导。

它不是：

```text
Tokenizer decode
```

而是：

> 执行模型 Transformer Forward / Generation State Update 的核心推理入口。

---

# 60. Tokenizer Decode 是另一个概念

```text
Token ID
↓
llama_token_to_piece
↓
Text
```

这才是：

```text
detokenization
```

---

# 61. `llama_decode` 的核心内部阶段

可以概括：

```text
validate batch
↓
split into ubatches
↓
prepare memory positions
↓
build graph
↓
allocate / schedule graph
↓
execute backend
↓
collect logits / outputs
↓
update memory
```

---

# 62. Graph Builder

源码：

- [src/llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp)

这里会根据：

```text
Model Architecture
Context
UBatch
Memory
```

构建：

```text
ggml Computation Graph
```

---

# 63. 为什么 Model Architecture 需要专门实现？

Qwen、LLaMA、Gemma、Mistral 等虽然都是：

```text
Decoder-only Transformer
```

但细节不同：

```text
Norm
RoPE
Attention
GQA
FFN
MoE
Sliding Window
Special Layer
```

因此：

```text
Graph Builder
```

必须知道架构。

---

# 64. 当前源码中的模型架构组织

`src/CMakeLists.txt` 会把：

```text
models/*.cpp
```

加入 llama library。

这意味着：

> 现代 llama.cpp 已经把不同 Model Architecture 实现进一步拆分到独立文件。

源码阅读时不要假设：

```text
所有模型都写在一个巨大 switch 里。
```

---

# 65. `llama-model.cpp`

源码：

- [src/llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)

它主要处理：

```text
Model Lifecycle
Hyperparameters
Tensor Creation / Loading
Backend Placement
Model Info
```

不是最适合第一眼开始读的文件。

---

# 66. 推荐阅读顺序

```text
examples/simple/simple.cpp

↓
include/llama.h

↓
src/llama.cpp

↓
src/llama-model-loader.cpp

↓
src/llama-model.cpp

↓
src/llama-context.cpp

↓
src/llama-graph.cpp

↓
src/llama-kv-cache.cpp / memory modules

↓
src/llama-sampler.cpp

↓
ggml/*
```

---

# 67. 为什么这样读？

先看：

```text
API 调用链
```

再看：

```text
内部实现
```

否则：

```text
几千行 Graph / Model 代码
```

没有上层地图。

---

# 68. ggml Tensor

核心数据结构：

```text
ggml_tensor
```

可以理解：

```text
Data Type
Shape
Stride
Operation
Source Tensor
Backend Data
Name
```

的组合。

---

# 69. ggml Tensor 和 PyTorch Tensor 的共同点

都有：

```text
dtype
shape
stride
data
```

---

# 70. 但 ggml Tensor 还可以直接表示 Graph Node

例如：

```text
C = ggml_mul_mat(ctx, A, B)
```

返回：

```text
一个 ggml_tensor
```

这个 Tensor 同时记录：

```text
它由什么 Operator
依赖哪些 Source Tensor
```

---

# 71. Tensor as Graph Node

概念：

```text
A ──┐
    MatMul ── C
B ──┘
```

`C`：

```text
既代表 Result Tensor
也代表 MatMul Node
```

---

# 72. ggml Operator

官方支持大量 Op：

- [docs/ops.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

例如：

```text
ADD
MUL
MUL_MAT
SOFT_MAX
ROPE
RMS_NORM
RESHAPE
VIEW
...
```

---

# 73. 为什么 `docs/ops.md` 很值得看？

它直接列：

```text
每种 ggml Op
```

在不同 Backend：

```text
CPU
CUDA
Metal
Vulkan
SYCL
OpenCL
...
```

的支持情况。

这是理解：

> Operator Coverage

的真实材料。

---

# 74. Operator 与 Backend

Operator：

```text
MUL_MAT
```

描述：

```text
做什么
```

Backend：

```text
CPU
CUDA
Vulkan
```

提供：

```text
怎么做
```

---

# 75. ggml Computation Graph

多个 Tensor/Op：

```text
Embedding
↓
RMSNorm
↓
MatMul Q/K/V
↓
RoPE
↓
Attention
↓
FFN
...
```

最终组成：

```text
ggml_cgraph
```

---

# 76. Graph Build vs Graph Compute

Build：

```text
创建 Tensor Node
描述 Dependency
```

Compute：

```text
真正分配 / 调度 / 执行 Kernel
```

---

# 77. 为什么 Build Graph 每次可能不同？

Prompt Processing：

```text
T = 512
```

Token Generation：

```text
T = 1
```

Output Requirements：

```text
也可能不同。
```

因此 Graph Shape：

```text
会变化。
```

---

# 78. Backend

当前官方 README 列出了多种 Backend。

包括：

```text
CPU / BLAS / BLIS
Metal
CUDA
HIP
SYCL
Vulkan
CANN
OpenCL
WebGPU
RPC
...
```

具体列表持续变化，以项目 README 为准。

---

# 79. Backend 的职责

```text
Allocate Device Buffer
Copy Tensor
Check Op Support
Execute Op
Synchronize
Measure
```

---

# 80. ggml Backend

底层代码：

- [ggml/src/ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)

这里是理解：

```text
Backend Abstraction
Buffer
Event
Scheduler
```

的重要入口。

---

# 81. 为什么需要 Backend Scheduler？

一个 Graph：

```text
有些 Node 在 CPU
有些 Node 在 GPU
```

需要：

```text
Partition
Allocate
Copy
Execute
Synchronize
```

这就是：

```text
Scheduler
```

需要解决的事情。

---

# 82. Backend Scheduler 的核心问题

```text
这个 Node 放哪个 Backend？

Tensor Buffer 放哪里？

跨 Backend 如何复制？

Graph 如何分 Split？

什么时候同步？
```

---

# 83. CPU Backend

CPU 是 llama.cpp 最基础的 Backend。

优势：

```text
无额外 GPU 依赖
跨平台
对量化格式支持完整
适合本地 / ARM
```

---

# 84. CPU 为什么是 llama.cpp 的灵魂之一？

llama.cpp 最早成功的核心之一就是：

> 在普通 CPU 上使用高度优化的低比特量化矩阵计算实现实用 LLM 推理。

所以 CPU 路线不是：

```text
“没有 GPU 的退化模式”
```

而是 llama.cpp 的核心能力。

---

# 85. ARM CPU

官方 feature matrix 当前包含：

```text
CPU ARM NEON
```

说明 ARM 是重要目标之一。

这对：

```text
Linux ARM SoC
```

很有意义。

---

# 86. SIMD

CPU Quantized MatMul：

```text
不是一个 Weight 一个 Weight 用 C 循环慢慢算。
```

而要利用：

```text
AVX / AVX2 / AVX512
NEON
dot product instructions
```

等 SIMD。

---

# 87. ARM KleidiAI

当前 build 文档还提供：

```text
Arm KleidiAI
```

CPU microkernel 路线。

它强调：

```text
为 Arm CPU 提供优化 AI microkernel。
```

这也是非常值得关注的 Edge AI 方向。

---

# 88. BLAS

CPU build 可启用：

```text
OpenBLAS
BLIS
...
```

BLAS 特别擅长：

```text
Dense Matrix-Matrix Multiply
```

---

# 89. 为什么 BLAS 更容易帮助 Prompt Processing？

Prefill：

```text
M 较大
```

更接近：

```text
GEMM
```

适合 BLAS。

---

# 90. Decode 为什么情况不同？

单 Token：

```text
M ≈ 1
```

更接近：

```text
Matrix-Vector
```

量化 Weight：

```text
还需要专门 quant kernel
```

所以高性能路径与普通 FP32 BLAS：

```text
不同。
```

---

# 91. CUDA Backend

Build：

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

参考：

- [docs/build.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)

---

# 92. CUDA Backend 干什么？

提供：

```text
GPU Tensor Buffer
CUDA Kernels
GEMM / quant kernels
Attention
Copy
Scheduling
```

---

# 93. Metal

主要：

```text
Apple Silicon
```

统一内存架构下：

```text
llama.cpp
```

表现非常突出。

---

# 94. Vulkan

Vulkan Backend：

```text
面向更广泛 GPU
```

Build：

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

---

# 95. SYCL

主要可用于：

```text
Intel GPU
```

并持续加入：

```text
Flash Attention
Quantized KV
Fusion
```

等优化路径。

---

# 96. OpenCL

当前 llama.cpp OpenCL Backend 官方文档主要面向：

```text
Qualcomm Adreno GPU
```

同时由于 OpenCL 可移植性，也可能支持其他设备。

参考：

- [docs/backend/OPENCL.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md)

---

# 97. Backend 不代表自动最快

例如：

```text
Vulkan supported
```

不等于：

```text
Vulkan 一定比 CPU 快。
```

取决于：

```text
Model
Quant
Operator Coverage
Driver
Kernel
Memory Bandwidth
Offload Ratio
```

---

# 98. Feature Matrix

官方 Wiki：

- [Feature Matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)

可以看到：

```text
K-quants
I-quants
KV Cache Quant
MoE
Flash Attention
Multi-GPU
```

在不同 Backend：

```text
支持程度不同。
```

---

# 99. Backend Support 是动态的

所以学习时：

> 不要把某一年的 Backend 支持状态当永久事实。

需要：

```text
查当前 README / docs / feature matrix。
```

---

# 100. RK3576 与 llama.cpp

这里必须明确。

llama.cpp 当前官方 Backend：

```text
没有 Rockchip RKNPU Backend。
```

所以：

```text
llama.cpp
```

不能直接等同：

```text
RKNPU LLM Runtime。
```

---

# 101. RK3576 上 llama.cpp 可以是什么路线？

概念上：

```text
RK3576 Linux
↓
llama.cpp
↓
ARM CPU Backend
```

如果 GPU Driver / Backend 条件满足：

```text
可能进一步探索 Vulkan / OpenCL 等 GPU 路径
```

具体以设备驱动和 Backend 支持为准。

---

# 102. 但 NPU 是另一条路线

```text
Qwen
↓
RKLLM / RKNN-LLM Toolchain
↓
Rockchip NPU Runtime
↓
RKNPU
```

不是：

```text
GGUF → llama.cpp → RKNPU
```

---

# 103. llama.cpp vs RKLLM

```text
llama.cpp

优势：
开放 Runtime
源码透明
GGUF 生态
CPU/GPU 跨平台
非常适合学习推理系统

RKLLM

优势：
Rockchip Vendor Toolchain
针对 RKNPU
专用 NPU 编译与 Runtime
```

---

# 104. 为什么两条路线都值得学？

llama.cpp：

> 学“通用 LLM Runtime 原理”。

RKLLM：

> 学“Vendor NPU 部署”。

二者结合才能真正理解：

```text
通用 Runtime
vs
专用 Accelerator Runtime。
```

---

# 105. Quantization Types

llama.cpp 生态中：

```text
F32
F16
BF16
Q8_0
Q6_K
Q5_K_M
Q4_K_M
Q4_K_S
IQ4_XS
IQ3
IQ2
...
```

种类很多。

不要试图一次背完。

---

# 106. 第一阶段只掌握

```text
F16
Q8_0
Q4_0
Q4_K_M
Q5_K_M
Q6_K
```

以及：

```text
K-quants
```

的基本思想。

---

# 107. K-quants

K-quants：

> 基于 block / super-block 的量化方案族。

不同：

```text
Q2_K
Q3_K
Q4_K
Q5_K
Q6_K
```

在：

```text
Bits per Weight
Scale Representation
Quality
Speed
```

上不同。

---

# 108. `_S / _M / _L`

对于一些 K Quant 名：

```text
S
M
L
```

通常表示：

> 不同 Tensor 采用不同精度混合策略的档位。

不能简单理解为：

```text
M = 单一 4-bit。
```

---

# 109. Q4_K_M

最重要的认识：

```text
“Q4”
```

不代表模型里所有 Tensor：

```text
严格每个 Weight 4 bits。
```

实际：

```text
不同 Tensor
可能使用不同量化类型
```

以质量和速度做平衡。

---

# 110. Bits Per Weight

官方 quantize 文档使用：

```text
bits/weight
```

比简单说：

```text
4 bit
```

更加准确。

例如 Q4_K_M：

```text
实际平均 BPW > 4。
```

---

# 111. 为什么有 Metadata Overhead？

量化 Block 需要保存：

```text
Scale
Min
Quantized Codes
```

所以：

```text
有效 BPW
```

大于理论裸 bit。

---

# 112. 为什么量化需要专门 Kernel？

如果 Weight：

```text
Q4_K
```

不能直接把 bit pattern：

```text
当 FP32 乘。
```

需要：

```text
Load Quant Block
Decode Scale
Unpack Quant Values
Dot Product / MatMul
```

---

# 113. Dequantize-first vs Fused Quantized Compute

最简单：

```text
Q4
↓
Dequant to FP32
↓
MatMul
```

会增加：

```text
Memory / temporary traffic。
```

高性能 Kernel 更希望：

```text
边解码 Quant Block
边参与 dot product。
```

---

# 114. 这就是 llama.cpp 的关键技术之一

不仅：

```text
模型存成 Q4
```

更重要：

> **有针对 Q4/Q5/Q8 等格式优化的 CPU/GPU Kernel。**

---

# 115. Importance Matrix

Quantize Tool 支持：

```text
--imatrix
```

用于：

```text
importance-aware quantization。
```

核心思想：

> 不同 Weight / Channel 对模型输出的重要性不同，量化可以利用代表性数据统计来降低质量损失。

本章知道位置即可。

---

# 116. 为什么不能只按文件大小选 Quant？

还要考虑：

```text
Perplexity / Quality
Prompt Processing
Token Generation
Backend Support
Kernel Speed
Memory
```

官方 quantize README 本身就给：

```text
size
pp t/s
tg t/s
```

对比。

---

# 117. 一个很重要的现象

更低 BPW：

```text
不一定 Prompt Processing 更快。
```

因为：

```text
Kernel Efficiency
Compute
Dequantization
```

不同。

---

# 118. 但 Text Generation 常对 Weight Size 非常敏感

这和上一章：

```text
Decode Memory-Bound
```

完全对应。

---

# 119. Sampling System

源码：

- [src/llama-sampler.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)

llama.cpp 不是把：

```text
top-k / top-p / temperature
```

硬编码在一个巨大函数里。

它使用：

```text
Sampler Abstraction
```

---

# 120. Sampler Chain

`simple.cpp`：

```cpp
llama_sampler * smpl =
    llama_sampler_chain_init(...);

llama_sampler_chain_add(
    smpl,
    llama_sampler_init_greedy()
);
```

更复杂可以组合：

```text
Repetition Penalty
Top-K
Top-P
Min-P
Temperature
Distribution
Grammar
```

---

# 121. 为什么 Chain 很好？

每个 Sampler：

```text
只负责一个 Logit Transformation / Selection 规则。
```

然后：

```text
Composable
```

---

# 122. Sampling Pipeline

概念：

```text
Raw Logits
↓
Penalty
↓
Grammar Mask
↓
Top-K
↓
Top-P
↓
Temperature
↓
Distribution Sample
↓
Token
```

具体顺序：

```text
取决于配置。
```

---

# 123. Grammar

llama.cpp 支持：

```text
Grammar-constrained generation。
```

这让：

```text
JSON
Structured Output
```

更可靠。

---

# 124. Grammar 本质在做什么？

在每个 Decode Step：

```text
根据当前 Grammar State
```

把：

```text
不允许的 Token
```

设为不可选。

---

# 125. Tool Calling 与 llama.cpp

Tool Calling 最终仍依赖：

```text
Model Chat Template
+
Structured Generation
+
Application Parsing
```

llama.cpp Runtime：

```text
负责模型推理。
```

Agent Framework：

```text
负责 Tool Loop。
```

---

# 126. `llama-cli`

主要用于：

```text
交互式命令行推理
测试模型
Chat
参数调试
```

---

# 127. `llama-server`

主要：

```text
把 llama.cpp 包装成 Server。
```

用于：

```text
HTTP
OpenAI-compatible style API
Concurrent Requests
Embedding
Reranking 等
```

具体功能随版本持续变化。

---

# 128. 为什么 Server 不是 vLLM？

二者都能：

```text
Serve LLM
```

但设计目标不同。

llama.cpp：

```text
本地 / 多硬件 / GGUF / CPU-GPU
```

vLLM：

```text
GPU High-throughput Serving
Scheduler / Paged KV
```

后面专门对比。

---

# 129. `llama-simple`

这个程序特别值得：

> 自己抄一遍并重新实现。

因为只要理解它，就已经理解：

```text
80% 外部 API 主线。
```

---

# 130. `llama-bench`

README 中：

```bash
llama-bench -m model.gguf
```

输出：

```text
model
size
params
backend
threads
test
t/s
```

---

# 131. `pp512`

```text
Prompt Processing
512 Tokens
```

对应：

```text
Prefill Benchmark。
```

---

# 132. `tg128`

```text
Token Generation
128 Tokens
```

对应：

```text
Decode Benchmark。
```

---

# 133. 为什么 README 同一个模型会有两个 t/s？

因为：

```text
pp
```

和：

```text
tg
```

是完全不同 Workload。

---

# 134. llama.cpp Benchmark 最应该记录

```text
Model
Quant
Model Size
Backend
Threads
GPU Layers
Context
Batch
UBatch
KV Type
Flash Attention
PP
TG
```

---

# 135. 不要只看 llama-cli 体感

Benchmark：

```text
有固定 test shape
```

更适合：

```text
版本对比
参数对比
量化对比
```

---

# 136. Build：CPU

官方当前基本：

```bash
cmake -B build
cmake --build build --config Release
```

---

# 137. Debug Build

读源码时很建议：

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

然后：

```text
gdb
lldb
```

逐步跟。

---

# 138. 为什么自己编译比只下载 Binary 更适合学习？

可以：

```text
加 Log
打断点
看 Call Stack
切 Backend
修改源码
```

---

# 139. Build：CUDA

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

---

# 140. Build：Vulkan

```bash
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release
```

---

# 141. Build：OpenBLAS

```bash
cmake -B build \
    -DGGML_BLAS=ON \
    -DGGML_BLAS_VENDOR=OpenBLAS

cmake --build build --config Release
```

---

# 142. Build：ARM CPU Optimized Microkernel

当前文档包含：

```bash
cmake -B build \
    -DGGML_CPU_KLEIDIAI=ON

cmake --build build --config Release
```

具体平台支持和性能应以当前文档为准。

---

# 143. 推荐实验环境

学习源码最方便：

```text
Linux x86 CPU
```

或者：

```text
Ubuntu VM
```

先跑 CPU。

然后再加：

```text
CUDA / Vulkan。
```

---

# 144. 为什么先 CPU？

因为 CPU：

```text
少一个复杂 Backend
```

Call Stack 更清晰。

先理解：

```text
Graph
Tensor
Quant
KV
Sampler
```

再看 GPU。

---

# 145. 然后再到 ARM Linux

当 x86 理解后：

```text
RK3576 Ubuntu
```

编 CPU Backend。

对比：

```text
x86 SIMD
vs
ARM NEON / KleidiAI
```

非常有学习价值。

---

# 146. llama.cpp CLI 最少需要哪些参数？

最基本：

```bash
llama-cli \
    -m model.gguf \
    -p "Hello"
```

---

# 147. Chat Mode

对于 Instruct 模型：

```text
需要正确 Chat Template。
```

很多 GGUF：

```text
包含 tokenizer.chat_template Metadata。
```

Runtime 可以使用。

---

# 148. Chat Template 仍然不是 llama.cpp 自创协议

它来源于：

```text
模型训练时的对话格式。
```

llama.cpp：

```text
负责解释 / 应用。
```

---

# 149. Context

常见：

```text
-c
--ctx-size
```

控制：

```text
Context Capacity。
```

---

# 150. GPU Layers

常见：

```text
-ngl
--gpu-layers
```

控制 Offload。

官方 build 文档经常用：

```text
-ngl 99
```

表达：

> 尝试把绝大多数 / 全部支持的 Layer Offload 到 GPU。

---

# 151. `-ngl 99` 不是标准保证

不同：

```text
Model Layers
Architecture
Backend
```

数量不同。

更准确：

> 一个“足够大的值”，让 Runtime 尽可能 Offload 可 Offload Layers。

---

# 152. Flash Attention

CLI 可以启用：

```text
--flash-attn
```

当前不同 Backend：

```text
支持程度不同。
```

---

# 153. 为什么 Flash Attention 要看 Backend？

Flash Attention：

```text
需要专门 Kernel。
```

不是开启一个 Boolean：

```text
任何硬件都会神奇加速。
```

---

# 154. KV Cache Type

CLI：

```text
--cache-type-k
--cache-type-v
```

说明：

> llama.cpp 将 Weight Quantization 和 KV Quantization 分离。

---

# 155. Weight Q4 不代表 KV 也是 Q4

例如：

```text
Model:
Q4_K_M

KV:
F16
```

完全正常。

---

# 156. 为什么 K 和 V 可以不同类型？

K / V：

```text
数值敏感性
Kernel Support
Transpose Layout
```

可能不同。

Runtime 允许独立调优。

---

# 157. KV Offload

有些 Backend：

```text
KV Cache 可以放 Device。
```

也可以：

```text
禁用 KV Offload
```

让 KV 保留 CPU / host memory。

---

# 158. KV Offload 的 Tradeoff

Device KV：

```text
Attention 更快
减少 Host-Device Copy
```

但：

```text
占 GPU Memory。
```

---

# 159. Model Weight Offload 与 KV Offload 是两件事

```text
n_gpu_layers
```

主要影响：

```text
Model Layer / Weight Placement。
```

KV Offload：

```text
Memory State Placement。
```

---

# 160. Context Memory

当前 llama.cpp 新版源码已经不仅仅一个传统：

```text
llama-kv-cache
```

还出现：

```text
llama-memory
llama-memory-hybrid
llama-memory-recurrent
...
```

这是因为模型架构越来越多样。

---

# 161. 为什么出现 Memory Abstraction？

并非所有模型都是：

```text
标准 Transformer KV Cache。
```

还可能：

```text
Recurrent State
Sliding Window
Hybrid Architecture
State Space Model
```

所以 Runtime 需要更一般：

```text
Memory Module。
```

---

# 162. 这是现代 llama.cpp 源码阅读很重要的变化

不要只带着旧认知：

```text
“llama.cpp = llama-kv-cache.cpp”
```

当前项目已经在抽象：

```text
更通用 Model Memory。
```

---

# 163. `llama-context.cpp`

源码：

- [src/llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp)

重点阅读：

```text
constructor
sched_reserve
decode flow
graph allocation
output handling
memory update
```

---

# 164. Scheduler Reserve

初始化阶段：

```text
算 worst-case token count
```

然后：

```text
build graph
reserve buffer
```

避免：

```text
推理阶段不断重分配。
```

---

# 165. PP Graph Reserve

Prompt Processing：

```text
n_tokens = min(n_ctx, n_ubatch)
```

会建：

```text
较大的 graph。
```

---

# 166. TG Graph Reserve

Token Generation：

```text
n_tokens ≈ n_seqs
```

Graph 更接近：

```text
每 Sequence 一个新 Token。
```

---

# 167. 这和 06 章完全对应

```text
06:
Prefill vs Decode

07:
pp graph vs tg graph
```

---

# 168. Graph Split

Scheduler 可能把 Graph：

```text
切成多个 Backend Split。
```

例如：

```text
CPU Nodes
↓
copy
↓
GPU Nodes
```

---

# 169. Split 越多一定越好吗？

不是。

跨 Backend：

```text
Copy
Sync
```

会增加成本。

通常：

```text
尽量减少不必要的 device boundary。
```

---

# 170. `op_offload`

现代 Context / Scheduler 还支持：

```text
operator-level offload。
```

说明 Backend Placement 已经不仅是：

```text
按 Layer 粗粒度。
```

具体属于高级部分。

---

# 171. Model Graph

现代 LLM Graph 典型：

```text
Token Embedding
↓
RMSNorm
↓
Q/K/V Projection
↓
RoPE
↓
Attention
↓
Output Projection
↓
Residual
↓
RMSNorm
↓
Gate / Up
↓
Activation
↓
Down
↓
Residual
↓
Final Norm
↓
Output
```

---

# 172. 在 ggml 中它会变成

```text
GET_ROWS
RMS_NORM
MUL
MUL_MAT
ROPE
SOFT_MAX
...
```

构成 Graph。

---

# 173. 为什么 Runtime 不直接调用一个 `transformer()` Kernel？

因为：

```text
模型架构不断变化。
```

拆成：

```text
Primitive / Fused Ops
```

更灵活。

---

# 174. Fused Ops

当前 llama.cpp 会探测：

```text
Flash Attention
Gated Delta Net fused ops
其他特定结构 fused op
```

Backend 支持时：

```text
使用更高效路径。
```

---

# 175. Fused Op 的本质

原来：

```text
Op A
↓
write memory
↓
Op B
↓
write memory
↓
Op C
```

融合：

```text
A + B + C
一个 Kernel
```

减少：

```text
Launch
Memory Roundtrip
```

---

# 176. Backend Feature Detection

Runtime 不能假设：

```text
所有 Backend 都支持同一 Op。
```

所以会：

```text
probe / supports_op
```

再选择路径。

---

# 177. Model Compatibility

一个模型能不能在 llama.cpp 跑：

```text
不仅看 GGUF 能否生成。
```

还要：

```text
llama.cpp 是否实现对应 architecture graph。
```

---

# 178. 新模型支持通常需要什么？

概念上：

```text
Architecture Metadata Mapping
Tensor Name Mapping
Hyperparameter Loading
Tensor Loading
Graph Implementation
Tokenizer / Chat handling
```

---

# 179. 所以 `convert_hf_to_gguf.py` 支持 ≠ Runtime 一定完整支持

Conversion：

```text
能生成 Tensor Container。
```

Runtime：

```text
还必须知道怎么执行架构。
```

两端都需要。

---

# 180. 源码里的 `llama_arch`

用于：

```text
识别模型架构。
```

GGUF Metadata：

```text
general.architecture
```

会映射到内部：

```text
architecture enum / implementation。
```

---

# 181. 为什么 GGUF Metadata 不能乱改？

如果：

```text
architecture
head count
rope
context
```

错误：

```text
Runtime 会建错 Graph。
```

结果：

```text
加载失败
或输出错误。
```

---

# 182. Quantized Tensor 与 Graph Semantics 分离

Model Graph：

```text
仍然是 MatMul。
```

Weight：

```text
可以 Q4_K_M。
```

Backend：

```text
选择支持该类型的 MUL_MAT Kernel。
```

这是非常漂亮的抽象。

---

# 183. 同一个 Graph 可以换 Quant

例如：

```text
Qwen Architecture
```

Graph 不变。

模型文件：

```text
F16
Q8_0
Q4_K_M
```

都可以。

变化主要：

```text
Tensor Type
Buffer
Kernel Path
Memory
Performance。
```

---

# 184. 同一个 GGUF 可以换 Backend

模型：

```text
Q4_K_M.gguf
```

运行：

```text
CPU
CUDA
Vulkan
```

只要 Backend 支持对应 Op / Quant。

---

# 185. 这就是 llama.cpp 跨平台的核心

```text
Model
Graph
```

与：

```text
Hardware Kernel
```

分离。

---

# 186. 这和 AI Compiler 非常相似

```text
High-level Graph
↓
Backend
↓
Kernel
```

只是 llama.cpp：

```text
更偏专用 Runtime。
```

TVM：

```text
更偏通用 Compiler。
```

---

# 187. llama.cpp 是 AI Compiler 吗？

不完全。

它包含：

```text
Graph Build
Scheduling
Backend Dispatch
一些 Graph / Op Optimization
```

有 Compiler 特征。

但主要定位仍然：

> LLM Inference Runtime / Engine。

---

# 188. llama.cpp 是框架吗？

可以广义叫：

```text
Inference Framework。
```

但为了避免混淆，最好称：

```text
LLM Runtime / Inference Engine / Library。
```

---

# 189. 为什么 C/C++ 很重要？

部署环境：

```text
Android
Linux ARM
Desktop
Embedded Linux
```

不一定希望带：

```text
Python
PyTorch
巨大依赖。
```

C/C++：

```text
更轻
更容易集成
更直接控制内存和线程。
```

---

# 190. llama.cpp 对嵌入式开发者的价值

它把：

```text
AI 模型
```

和熟悉的：

```text
C/C++
CMake
Thread
Memory
SIMD
File Mapping
Backend
```

连接起来。

---

# 191. 这是很适合嵌入式 → Edge AI 的桥

如果已经熟悉：

```text
MCU / C
Linux / C++
```

llama.cpp 可以帮助把：

```text
LLM 理论
```

落到：

```text
真实系统软件。
```

---

# 192. 推荐源码阅读：第一阶段

只读：

```text
examples/simple/simple.cpp
include/llama.h
```

目标：

```text
知道外部 API。
```

---

# 193. 第一阶段完成标准

自己写一个：

```text
mini_llama_cli.cpp
```

只做：

```text
load
tokenize
decode
sample
print
```

不要复制完整 llama-cli。

---

# 194. 推荐源码阅读：第二阶段

```text
src/llama.cpp
src/llama-model-loader.cpp
```

目标：

```text
API → Model Loader。
```

---

# 195. 推荐源码阅读：第三阶段

```text
src/llama-context.cpp
```

目标：

```text
Context Init
Scheduler
Decode
Memory
Outputs。
```

---

# 196. 推荐源码阅读：第四阶段

```text
src/llama-graph.cpp
models/<architecture>.cpp
```

目标：

```text
Transformer → ggml Graph。
```

---

# 197. 推荐源码阅读：第五阶段

```text
ggml/include/ggml.h
ggml/src/ggml.c / ggml.cpp related
ggml/src/ggml-backend.cpp
```

目标：

```text
Tensor
Op
Graph
Backend。
```

---

# 198. 推荐源码阅读：第六阶段

选择一个 Backend：

```text
CPU
```

优先。

再看：

```text
CUDA / Vulkan
```

---

# 199. 不要一开始看 CUDA Kernel

如果还不清楚：

```text
Graph Node 是什么
Backend 怎么调
```

直接读 Kernel：

```text
没有上下文。
```

---

# 200. `llama-cli` 源码什么时候看？

在：

```text
simple.cpp
```

理解后。

CLI 更复杂，因为加入：

```text
chat
console
sampling config
session
grammar
multimodal
interactive
```

---

# 201. `llama-server` 什么时候看？

等理解：

```text
Context
Batch
Sequences
```

以后。

因为 Server 还加入：

```text
HTTP
Concurrent Request
Slot
Queue
Streaming
```

---

# 202. Sampling 源码什么时候看？

在：

```text
llama_decode
→ logits
```

理解后。

因为 Sampling：

```text
是 Forward 之后。
```

---

# 203. Quant Kernel 什么时候看？

等理解：

```text
ggml_type
MUL_MAT
CPU Backend
```

以后。

---

# 204. 一个关键源码调试技巧

在 `simple.cpp` 打断点：

```text
llama_model_load_from_file

llama_init_from_model

llama_decode

llama_sampler_sample
```

然后 Step Into。

---

# 205. 不要一次 Step Into 所有库函数

否则很快进入：

```text
STL
filesystem
thread
mmap
```

失去主线。

---

# 206. 建议用 Call Stack 学源码

每次进入：

```text
llama_decode
```

记录：

```text
调用了哪些核心层。
```

然后画：

```text
API
↓
Context
↓
Graph
↓
Scheduler
↓
Backend
```

---

# 207. Log Level

开发时：

```text
开启 Debug Log
```

观察：

```text
Model Metadata
Tensor Placement
Buffer Size
Backend
KV
Graph Split
```

很多内部状态已经有日志。

---

# 208. Load Log 很有价值

典型：

```text
architecture
vocab size
context length
embedding
layers
heads
quantization
tensor type count
buffer size
```

可以用来：

> 验证你对 Model 的理解。

---

# 209. 模型 Tensor Type Count

例如：

```text
f32 tensors
q8_0 tensors
q4_k tensors
```

说明：

> 一个 GGUF 中可以同时存在多种 Tensor Type。

---

# 210. 这再次说明 Q4_K_M 是 Mixed Quant

不是：

```text
所有 Tensor 一模一样 4bit。
```

---

# 211. Model Buffer Size

Runtime 常打印：

```text
CPU model buffer size
CUDA model buffer size
...
```

可以直接看到：

```text
Weight 被放到哪个 Backend。
```

---

# 212. Compute Buffer Size

Context 初始化时还可能打印：

```text
compute buffer size。
```

这不是 Weight。

它属于：

```text
Graph execution workspace。
```

---

# 213. KV Buffer Size

又是独立：

```text
runtime state memory。
```

---

# 214. 三类内存一定要分清

```text
Model Buffer
=
Weight

KV / Memory Buffer
=
Context State

Compute Buffer
=
Temporary Graph Workspace
```

---

# 215. OOM 时先看哪个？

都要看。

不能只看：

```text
GGUF file size。
```

---

# 216. Model File vs mmap RSS

使用 mmap：

```text
Virtual Address
```

与：

```text
Resident Memory
```

不一定完全相同。

所以 Linux 分析：

```text
RSS
PSS
page cache
```

也值得学习。

---

# 217. Linux Edge Device 内存分析

可以使用：

```text
free
top
htop
smem
/proc/<pid>/smaps
```

观察：

```text
模型加载
Prefill
Decode
```

不同阶段。

---

# 218. `mmap` 与 Swap

如果 RAM 不够：

```text
Page 被换出
```

LLM Decode：

```text
会灾难性变慢。
```

所以：

> “模型勉强能打开”不等于可用。

---

# 219. Threads

CPU 推理：

```text
threads
```

很重要。

但：

```text
越多不一定越快。
```

---

# 220. Why?

Decode：

```text
Memory Bandwidth Saturation
```

以后：

```text
增加 Thread
```

收益有限。

---

# 221. Prompt Threads 与 Decode Threads

某些参数 / Runtime 路径可以：

```text
对 Batch / Prompt Processing
```

使用不同线程策略。

因为：

```text
PP / TG Workload 不同。
```

---

# 222. NUMA

多 Socket 服务器：

```text
NUMA
```

会影响：

```text
Weight Placement
Memory Bandwidth
Thread Affinity。
```

本地单 SoC：

```text
一般简单很多。
```

---

# 223. GPU Memory

全 Offload：

```text
Weight
+
KV
+
Compute
```

都可能占 VRAM。

---

# 224. VRAM 不够时

可以：

```text
减少 GPU Layers
减少 Context
减少 KV Precision
换更小 Quant
```

---

# 225. Offload More Layers 一定更快？

通常有帮助。

但仍受：

```text
GPU capability
backend kernels
transfer
quant support
```

影响。

---

# 226. Shared Memory GPU

Apple / 部分 SoC：

```text
CPU / GPU shared memory
```

没有传统 PCIe VRAM Copy。

但仍有：

```text
Bandwidth / cache / allocation
```

问题。

---

# 227. ARM SoC llama.cpp 学习价值

可以真实观察：

```text
CPU thread scaling
memory bandwidth
quant type
context memory
thermal throttling。
```

这些非常接近：

```text
Edge AI 工程真实问题。
```

---

# 228. llama.cpp 不是部署终点

对于 RK3576：

```text
llama.cpp CPU
```

是很好的：

```text
Baseline。
```

然后：

```text
RKLLM NPU
```

做对比。

---

# 229. 最值得做的实验之一

同一个：

```text
Qwen Small Model
```

比较：

```text
llama.cpp Q4 CPU

vs

RKLLM NPU
```

记录：

```text
Model Size
Load Time
PP
TG
RAM
Power
CPU/NPU Usage
Temperature。
```

---

# 230. 这样能真正理解 NPU 的价值

不是：

```text
“NPU 有 6 TOPS，所以更强”
```

而是：

```text
同一个任务
真实 runtime 数据。
```

---

# 231. `llama-bench` 实验 1：Quant

准备：

```text
F16
Q8_0
Q6_K
Q5_K_M
Q4_K_M
```

测：

```text
pp512
tg128
```

---

# 232. 记录表

| Quant | Size | PP t/s | TG t/s | RAM |
|---|---:|---:|---:|---:|
| F16 | | | | |
| Q8_0 | | | | |
| Q6_K | | | | |
| Q5_K_M | | | | |
| Q4_K_M | | | | |

---

# 233. 观察重点

通常会看到：

```text
TG
```

对 Quant：

```text
非常敏感。
```

但：

```text
PP
```

趋势可能不同。

---

# 234. 实验 2：CPU Threads

```text
1
2
4
6
8
...
```

记录：

```text
PP
TG
```

---

# 235. 观察

找到：

```text
TG 开始饱和
```

的 Thread 数。

联系：

```text
Memory Bandwidth。
```

---

# 236. 实验 3：Context

配置：

```text
2K
4K
8K
16K
```

记录：

```text
KV memory
total memory
TG
```

---

# 237. 实验 4：KV Cache Quant

配置：

```text
F16
Q8
Q4
```

比较：

```text
KV memory
long context TG
quality。
```

---

# 238. 实验 5：GPU Offload

如果设备有支持 Backend：

```text
ngl:
0
10
20
99
```

记录：

```text
PP
TG
VRAM
CPU Usage。
```

---

# 239. 实验 6：Flash Attention

```text
FA off

FA on
```

测：

```text
short prompt
long prompt
```

观察：

```text
长 Prompt 收益。
```

---

# 240. 实验 7：Batch

```text
n_batch
n_ubatch
```

不同组合。

重点：

```text
PP
memory
```

---

# 241. 实验 8：源码 Trace

在：

```text
simple.cpp
```

加入日志：

```text
before tokenize
after tokenize

before context
after context

before decode
after decode

before sample
after sample
```

建立完整时序。

---

# 242. 实验 9：打印 Token

输出：

```text
Token ID
Token Piece
Position
```

真正理解：

```text
Generation Loop。
```

---

# 243. 实验 10：修改 Sampler

从：

```text
greedy
```

改：

```text
temperature + top-k + top-p
```

观察：

```text
输出差异。
```

---

# 244. 实验 11：GGUF Metadata

写小程序或使用工具：

```text
dump GGUF metadata。
```

观察：

```text
architecture
layer count
context
heads
tokenizer
chat_template
quantization。
```

---

# 245. 实验 12：Netron 对 llama.cpp 没那么直接

GGUF：

```text
不是 ONNX Graph。
```

它主要保存：

```text
Weight + Metadata。
```

Graph：

```text
由 llama.cpp 根据 Architecture 在 Runtime 构建。
```

这是非常重要的区别。

---

# 246. ONNX vs GGUF

ONNX：

```text
Graph
+
Operator
+
Weight
```

GGUF：

```text
Architecture Metadata
+
Weight
+
Tokenizer
```

具体执行 Graph：

```text
由 llama.cpp 代码生成。
```

---

# 247. 这是两种不同部署哲学

ONNX：

```text
Model carries graph.
```

llama.cpp：

```text
Runtime knows architecture,
file carries parameters + metadata.
```

---

# 248. 为什么 llama.cpp 可以这样做？

因为它服务的主要模型：

```text
架构家族有限
```

Runtime 内部：

```text
直接实现 Architecture。
```

---

# 249. 优点

```text
可高度专门优化
格式紧凑
Runtime 控制强
量化生态好。
```

---

# 250. 缺点

新 Architecture：

```text
需要 Runtime 增加代码支持。
```

不像：

```text
任意 ONNX Graph
```

直接由通用 Operator Runtime 执行。

---

# 251. llama.cpp vs ONNX Runtime

```text
llama.cpp:
Architecture-specific LLM Runtime

ONNX Runtime:
General Graph Runtime
```

---

# 252. llama.cpp vs PyTorch

```text
PyTorch:
Training + Dynamic Model Framework

llama.cpp:
Inference-only oriented Runtime
```

---

# 253. llama.cpp vs vLLM

```text
llama.cpp:
Local / CPU / heterogeneous
GGUF / quantization
Edge-friendly

vLLM:
GPU server
High concurrency
Paged KV / Scheduler
HF ecosystem
```

---

# 254. llama.cpp vs MLC-LLM

```text
llama.cpp:
Hand-written Runtime + Backend ecosystem

MLC-LLM:
ML Compilation driven deployment
TVM / compiler stack
```

---

# 255. llama.cpp vs RKLLM

```text
llama.cpp:
Open general Runtime
CPU/GPU backend

RKLLM:
Rockchip vendor LLM compiler/runtime
RKNPU
```

---

# 256. 本章最重要的源码关系图

```text
                   Application
                       │
                 llama-simple
                 llama-cli
                llama-server
                       │
                       ▼
                include/llama.h
                       │
                       ▼
                  src/llama.cpp
                       │
       ┌───────────────┼──────────────┐
       ▼               ▼              ▼
 model loader        context        sampler
       │               │
       ▼               ▼
 llama-model       llama-graph
                       │
                       ▼
                  memory / KV
                       │
                       ▼
                     ggml
              ┌────────┼─────────┐
              ▼        ▼         ▼
           Tensor    Graph     Backend
                                │
                ┌───────────────┼─────────────┐
                ▼               ▼             ▼
               CPU            CUDA          Vulkan ...
```

---

# 257. 本章第二张重要图：Model Loading

```text
model-Q4_K_M.gguf
        │
        ▼
gguf_init_from_file
        │
        ├── Metadata
        │
        ├── Tensor Info
        │
        └── Tensor Offsets
        ▼
llama_model_loader
        │
        ├── Detect Architecture
        ├── Load HParams
        ├── Load Vocab
        ├── Create Weight Tensors
        └── Assign Backend Buffers
        ▼
    llama_model
```

---

# 258. 第三张图：Inference

```text
Prompt
  ↓
Tokenizer
  ↓
llama_batch
  ↓
llama_decode
  ↓
Split to UBatch
  ↓
Build ggml Graph
  ↓
Backend Scheduler
  ↓
CPU / GPU Kernel
  ↓
Logits
  ↓
Sampler Chain
  ↓
Token
  ↓
token_to_piece
  ↓
Text
  ↓
next llama_decode
```

---

# 259. 第四张图：Quantization

```text
FP16 Tensor
     ↓
Block Quantization
     ↓
Quantized Codes
+
Scale / Metadata
     ↓
GGUF Tensor
     ↓
Quantized MUL_MAT Kernel
     ↓
Load Block
     ↓
Decode / Dot Product
     ↓
Output
```

---

# 260. 第五张图：RK3576 双路线

```text
                        Qwen
                          │
               ┌──────────┴──────────┐
               ▼                     ▼
          Hugging Face           Vendor Tool
               │                     │
               ▼                     ▼
          GGUF / Quant          RKLLM Convert
               │                     │
               ▼                     ▼
           llama.cpp              RKLLM
               │                     │
        ┌──────┴──────┐              ▼
        ▼             ▼             RKNPU
      ARM CPU       GPU*
        │
        └──────┬──────┘
               ▼
            RK3576

* GPU backend availability/performance depends on driver/backend support.
```

---

# 261. 这张图必须牢记

不要写成：

```text
GGUF
↓
RKNN Toolkit
↓
RKNPU
```

这是两个生态。

---

# 262. 为什么 GGUF 通常不能直接给 RKNN-Toolkit2？

GGUF：

```text
llama.cpp-specific / ggml-oriented LLM model container。
```

RKNN-Toolkit2：

```text
主要接收其支持的 Framework / interchange models，
并针对 RKNPU 编译。
```

LLM：

```text
还有单独 RKNN-LLM / RKLLM 工具链。
```

---

# 263. 未来理解 RKLLM 时可以比较

```text
GGUF
vs
RKLLM Model

llama_context
vs
RKLLM Runtime Context

ggml Backend
vs
RKNPU Runtime

Q4_K_M
vs
Rockchip Supported Quant
```

---

# 264. 为什么 llama.cpp 这一章非常关键？

因为它第一次让整条知识链真正落地：

```text
Transformer
↓
Tensor
↓
Quant
↓
Graph
↓
Memory
↓
Backend
↓
Kernel
↓
Hardware
```

---

# 265. 从 PyTorch 视角看 llama.cpp

PyTorch：

```python
logits = model(input_ids)
```

这一行背后：

```text
巨大 Framework
```

帮你做了所有事情。

llama.cpp：

> 把推理需要的这部分重新以轻量 C/C++ Runtime 实现。

---

# 266. 从 Runtime 视角看

```text
model(input_ids)
```

必须拆成：

```text
Token Embedding
Layer 0
Layer 1
...
Final Norm
Output
```

再变：

```text
ggml Op Graph。
```

---

# 267. 从 Hardware 视角看

Graph 中：

```text
MUL_MAT
```

最终：

```text
CPU quantized dot kernel
CUDA kernel
Metal kernel
Vulkan shader
...
```

---

# 268. 学完这章后应该具备什么能力？

不是：

```text
记住 50 个 CLI 参数。
```

而是：

```text
看到 llama.cpp log
能判断：
正在 load 什么
Buffer 在哪
KV 多大
Graph 在哪个 backend
PP/TG 为什么不同。
```

---

# 269. 推荐实际学习顺序

---

## Stage 1：会跑

```text
Build CPU
Download GGUF
llama-cli
llama-bench
```

目标：

```text
熟悉外部工具。
```

---

## Stage 2：懂 GGUF

学习：

- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [gguf.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h)
- [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)

目标：

```text
知道文件里是什么。
```

---

## Stage 3：懂 Quant

学习：

- [quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

自己做：

```text
F16 → Q8 → Q6 → Q5 → Q4。
```

目标：

```text
把 Quant 与 PP/TG 性能联系起来。
```

---

## Stage 4：读 simple.cpp

目标：

```text
抄一份最小 inference app。
```

---

## Stage 5：读 Public API

`include/llama.h`

目标：

```text
知道 model/context/batch/sampler 生命周期。
```

---

## Stage 6：读 Model Loader

目标：

```text
GGUF → llama_model。
```

---

## Stage 7：读 Context

目标：

```text
n_ctx
n_batch
n_ubatch
scheduler
pp/tg buffer
memory。
```

---

## Stage 8：读 Graph

目标：

```text
Transformer → ggml Op。
```

---

## Stage 9：读 ggml

目标：

```text
Tensor
Op
Graph
Backend。
```

---

## Stage 10：读 CPU Backend

目标：

```text
Quantized Matrix Operation 如何真正执行。
```

---

## Stage 11：再看 GPU Backend

根据设备选择：

```text
CUDA
Vulkan
Metal
SYCL
OpenCL。
```

---

## Stage 12：RK3576 Baseline

在 RK3576：

```text
ARM CPU llama.cpp
```

建立：

```text
PP / TG / RAM / Power Baseline。
```

再与后面 RKLLM 对比。

---

# 270. 本章必做实验清单

- [ ] CPU 编译 llama.cpp
- [ ] 运行 `llama-cli`
- [ ] 运行 `llama-bench`
- [ ] 看懂 `pp512`
- [ ] 看懂 `tg128`
- [ ] 下载一个 F16/BF16 GGUF 或从 HF 转换
- [ ] 使用 `llama-quantize` 做 Q8 / Q4
- [ ] 比较模型大小
- [ ] 比较 PP
- [ ] 比较 TG
- [ ] 阅读 `simple.cpp`
- [ ] 自己重写一个 mini CLI
- [ ] 阅读 `llama.h`
- [ ] 跟一次 `llama_model_load_from_file`
- [ ] 跟一次 `llama_decode`
- [ ] 打印 GGUF Metadata
- [ ] 观察 Model Buffer
- [ ] 观察 KV Buffer
- [ ] 观察 Compute Buffer
- [ ] 调整 `n_ctx`
- [ ] 调整 `n_batch / n_ubatch`
- [ ] 调整 CPU threads
- [ ] 有 GPU 时测试 `-ngl`
- [ ] 有支持 Backend 时测试 Flash Attention
- [ ] 在 ARM Linux 上跑一次
- [ ] 最终在 RK3576 建立 llama.cpp CPU Baseline

---

# 271. 常见误区

---

## 误区 1

```text
GGUF = llama.cpp
```

错误。

```text
GGUF = file format

llama.cpp = runtime
```

---

## 误区 2

```text
ggml = GGUF
```

错误。

```text
ggml = tensor/graph/backend library

GGUF = model container
```

---

## 误区 3

```text
Q4_K_M = GGUF
```

错误。

GGUF：

```text
container
```

Q4_K_M：

```text
quantization scheme
```

---

## 误区 4

```text
Q4_K_M = 每个 Weight 精确 4 bit
```

错误。

它是：

```text
mixed K-quant scheme。
```

---

## 误区 5

```text
llama.cpp = 只能跑 LLaMA
```

历史名称如此。

现代项目支持：

```text
大量 LLM / multimodal architecture。
```

具体支持持续更新。

---

## 误区 6

```text
llama.cpp = CPU only
```

错误。

当前有：

```text
CUDA
Metal
Vulkan
SYCL
OpenCL
CANN
...
```

多种 Backend。

---

## 误区 7

```text
llama.cpp 可以直接跑 Rockchip NPU
```

当前官方没有：

```text
RKNPU backend。
```

Rockchip NPU 使用：

```text
RKLLM / RKNN-LLM
```

等 Vendor Toolchain。

---

## 误区 8

```text
-ngl 99 = 99% GPU
```

不准确。

它表达：

```text
requested GPU layers。
```

实际 placement：

```text
看模型与 backend。
```

---

## 误区 9

```text
Model Q4 → KV 也是 Q4
```

错误。

KV Cache：

```text
单独配置。
```

---

## 误区 10

```text
llama_decode = tokenizer decode
```

完全错误。

`llama_decode`：

```text
执行 LLM forward。
```

`token_to_piece`：

```text
才是 token → text。
```

---

## 误区 11

```text
GGUF 里存完整 ggml graph
```

一般不应这样理解。

Graph：

```text
由 llama.cpp 根据 Architecture 构建。
```

---

## 误区 12

```text
量化只改变磁盘大小
```

错误。

还改变：

```text
Memory
Bandwidth
Kernel
TG speed
Quality。
```

---

## 误区 13

```text
所有 Backend 都支持所有 quant / op
```

错误。

应查：

```text
docs/ops.md
feature matrix。
```

---

# 272. 本章核心知识树

```text
llama.cpp
│
├── Model Format
│   └── GGUF
│       ├── Metadata
│       ├── Tensor Info
│       ├── Tensor Data
│       └── Alignment
│
├── Conversion
│   └── convert_hf_to_gguf.py
│
├── Quantization
│   ├── llama-quantize
│   ├── Q8_0
│   ├── Q6_K
│   ├── Q5_K_M
│   ├── Q4_K_M
│   └── Importance Matrix
│
├── Model
│   ├── llama_model
│   ├── architecture
│   ├── hparams
│   ├── vocab
│   └── weight tensors
│
├── Runtime
│   └── llama_context
│       ├── n_ctx
│       ├── n_batch
│       ├── n_ubatch
│       ├── scheduler
│       ├── memory
│       └── outputs
│
├── Inference
│   ├── llama_batch
│   ├── llama_decode
│   ├── PP Graph
│   ├── TG Graph
│   ├── KV / Memory
│   └── logits
│
├── Sampling
│   ├── sampler chain
│   ├── top-k
│   ├── top-p
│   ├── temperature
│   ├── grammar
│   └── token
│
├── ggml
│   ├── tensor
│   ├── op
│   ├── cgraph
│   ├── allocator
│   ├── buffer
│   └── backend scheduler
│
├── Backend
│   ├── CPU
│   ├── BLAS
│   ├── CUDA
│   ├── Metal
│   ├── Vulkan
│   ├── SYCL
│   ├── OpenCL
│   └── ...
│
└── Tools
    ├── llama-cli
    ├── llama-server
    ├── llama-simple
    ├── llama-bench
    └── llama-quantize
```

---

# 273. 本章最重要的思维模型

## 思维模型 1

```text
llama.cpp
=
LLM Runtime
```

---

## 思维模型 2

```text
ggml
=
Tensor + Graph + Backend Foundation
```

---

## 思维模型 3

```text
GGUF
=
Weights + Metadata + Tokenizer-oriented Model Container
```

---

## 思维模型 4

```text
Q4_K_M
=
Quantized Tensor Encoding Strategy
```

---

## 思维模型 5

```text
llama_model
=
Model / Weight

llama_context
=
Inference State
```

---

## 思维模型 6

```text
llama_decode
=
Batch
→
Build/Execute Graph
→
Update Memory
→
Produce Logits
```

---

## 思维模型 7

```text
Prompt Processing
=
PP Graph
=
Prefill

Token Generation
=
TG Graph
=
Decode
```

---

## 思维模型 8

```text
Model Architecture
+
Weight Tensor Type
+
Backend
=
Actual Kernel Path
```

---

## 思维模型 9

```text
Same GGUF
can run
on different backends

Same Architecture
can use
different quantizations
```

---

## 思维模型 10

```text
llama.cpp Route
≠
Rockchip NPU Route
```

---

# 274. 本章完成标准

如果下面大部分都可以自己解释，本章就可以结束。

## 生态

- [ ] 能解释 llama.cpp 是什么
- [ ] 能解释 ggml 是什么
- [ ] 能解释 GGUF 是什么
- [ ] 能解释 Q4_K_M 是什么
- [ ] 能画出四者关系
- [ ] 能解释 llama.cpp 与 PyTorch 的区别
- [ ] 能解释 llama.cpp 与 ONNX Runtime 的区别
- [ ] 能解释 llama.cpp 与 RKLLM 的区别

## GGUF

- [ ] 能解释 GGUF Header
- [ ] 能解释 Metadata
- [ ] 能解释 Tensor Info
- [ ] 能解释 Tensor Data
- [ ] 能解释 Alignment
- [ ] 能解释为什么 GGUF 适合 mmap
- [ ] 能解释 `general.architecture`
- [ ] 能解释 tokenizer metadata
- [ ] 能解释 chat template metadata

## Conversion / Quant

- [ ] 能使用 `convert_hf_to_gguf.py`
- [ ] 能使用 `llama-quantize`
- [ ] 能解释 conversion vs quantization
- [ ] 能解释为什么避免 re-quantization
- [ ] 能解释 BPW
- [ ] 能解释 K-quants
- [ ] 能解释 Q4_K_M 的 mixed 思想
- [ ] 能解释 importance matrix
- [ ] 能比较 F16 / Q8 / Q4 PP/TG

## Model / Context

- [ ] 能解释 `llama_model`
- [ ] 能解释 `llama_context`
- [ ] 能解释为什么二者分开
- [ ] 能解释 model params
- [ ] 能解释 context params
- [ ] 能解释 `n_ctx`
- [ ] 能解释 `n_batch`
- [ ] 能解释 `n_ubatch`

## Inference

- [ ] 能解释 `llama_batch`
- [ ] 能解释 `llama_decode`
- [ ] 能解释它不是 tokenizer decode
- [ ] 能解释 PP Graph
- [ ] 能解释 TG Graph
- [ ] 能解释为什么初始化时分别 reserve
- [ ] 能解释 KV / Memory 与 Context 的关系

## ggml

- [ ] 能解释 `ggml_tensor`
- [ ] 能解释 Tensor 同时可以是 Graph Node
- [ ] 能解释 Operator
- [ ] 能解释 `ggml_cgraph`
- [ ] 能解释 Build vs Compute
- [ ] 能解释 Backend
- [ ] 能解释 Backend Scheduler
- [ ] 看过 `docs/ops.md`

## Backend

- [ ] 能解释 CPU Backend
- [ ] 能解释 BLAS
- [ ] 能解释为什么 PP 更适合大 GEMM
- [ ] 能解释 CUDA / Metal / Vulkan 的位置
- [ ] 能解释 GPU Offload
- [ ] 能解释 `n_gpu_layers`
- [ ] 能解释 Op Coverage
- [ ] 知道 ARM NEON / KleidiAI 的位置

## Sampling

- [ ] 能解释 Sampler Chain
- [ ] 能解释 Greedy
- [ ] 能解释 Top-K
- [ ] 能解释 Top-P
- [ ] 能解释 Temperature
- [ ] 能解释 Grammar Sampling
- [ ] 能解释 logits → token

## Tools

- [ ] 能使用 llama-cli
- [ ] 能使用 llama-bench
- [ ] 能解释 pp512
- [ ] 能解释 tg128
- [ ] 能运行 llama-simple
- [ ] 能说明 llama-server 的位置

## Source Reading

- [ ] 完整读过 simple.cpp
- [ ] 看过 llama.h
- [ ] 跟过 model load
- [ ] 跟过 context init
- [ ] 跟过一次 llama_decode
- [ ] 看过 llama-model-loader.cpp
- [ ] 看过 llama-context.cpp
- [ ] 看过 llama-graph.cpp
- [ ] 看过 ggml-backend.cpp

## Edge / RK3576

- [ ] 能解释 llama.cpp 在 ARM Linux 上的价值
- [ ] 能解释为何不能直接等于 RKNPU
- [ ] 能画出 llama.cpp vs RKLLM 双路线
- [ ] 能在 RK3576 跑 CPU GGUF Baseline
- [ ] 能记录 PP / TG / RAM / 温度

---

# 275. 本章暂时不要求深入

暂时不要求：

- [ ] 手写所有 GGUF Parser
- [ ] 精通所有 Quant Format
- [ ] 手写 Q4_K CPU Kernel
- [ ] 精通 CUDA Backend
- [ ] 精通 Metal Shader
- [ ] 精通 Vulkan Shader
- [ ] 实现新 Architecture
- [ ] 实现新 Backend
- [ ] 实现 Backend Scheduler
- [ ] 精通多 GPU Split
- [ ] 精通 Speculative Decode
- [ ] 精通 llama-server Scheduler
- [ ] 精通 Multimodal
- [ ] 精通 RPC Backend

这些可以在系统方向继续深入。

---

# 276. 核心参考链接

## llama.cpp

- [Repository](https://github.com/ggml-org/llama.cpp)
- [README](https://github.com/ggml-org/llama.cpp/blob/master/README.md)
- [Build Guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [GGML Operations / Backend Support](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)
- [Feature Matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)

---

## 最小源码入口

- [simple.cpp](https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/simple.cpp)
- [llama.h](https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h)
- [src/llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama.cpp)

---

## Model

- [llama-model-loader.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model-loader.cpp)
- [llama-model.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)
- [llama-model.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.h)

---

## Context / Graph / Memory

- [llama-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp)
- [llama-context.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.h)
- [llama-graph.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp)
- [llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp)
- [llama-kv-cache.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.h)

---

## Sampling

- [llama-sampler.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp)

---

## GGUF

- [GGUF Specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp gguf.h](https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/gguf.h)
- [ggml gguf.h](https://github.com/ggml-org/ggml/blob/master/include/gguf.h)

---

## Conversion

- [convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)

---

## Quantization

- [Quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)

---

## ggml Backend

- [ggml-backend.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp)
- [GGML Ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md)

---

## Backend Docs

- [OpenCL Backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENCL.md)
- [SYCL Backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [Vulkan Source](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan)
- [CUDA Source](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-cuda)

---

## Benchmark

- [llama-bench Source](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/llama-bench.cpp)
- [Batched Bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/batched-bench/README.md)

---

# 277. 最推荐的源码学习路线

```text
simple.cpp
   ↓
llama.h
   ↓
llama.cpp
   ↓
model-loader
   ↓
model
   ↓
context
   ↓
graph
   ↓
memory / KV
   ↓
sampler
   ↓
ggml tensor
   ↓
ggml backend
   ↓
CPU kernel
   ↓
GPU backend
```

严格建议：

> 不要倒过来。

---

# 278. 与上一章的映射

```text
06_LLM推理原理
               07_llama.cpp

Prompt         → llama_tokenize
Prefill        → PP graph / llama_decode
KV Cache       → memory / kv modules
Decode         → TG graph / llama_decode
Logits         → output buffer
Sampling       → llama_sampler
Weight Quant   → ggml quant types
Hardware       → ggml backend
PP/TG t/s      → llama-bench
```

---

# 279. 与下一章的关系

下一步：

> **08_vLLM与MLC-LLM.md**

开始比较两种完全不同的 Runtime 思路。

```text
llama.cpp
=
手写 C/C++ Runtime
+
GGUF
+
CPU / Local / Heterogeneous

vLLM
=
GPU Serving Runtime
+
Scheduler
+
Paged KV
+
Continuous Batching

MLC-LLM
=
Compiler-driven Runtime
+
Cross-platform Deployment
```

这三者对比后：

> LLM Runtime 这一层的整体图就会真正完整。

---

# 280. 一句话总结

这一章真正要记住的是：

```text
llama.cpp
不是 GGUF
不是 Q4_K_M
不是模型

llama.cpp
=
运行模型的 Runtime
```

它向上读取：

```text
GGUF
```

GGUF 里面有：

```text
Metadata
+
Tokenizer
+
Weight Tensor
```

Tensor 可以：

```text
F16
Q8
Q4_K_M
...
```

llama.cpp 根据：

```text
Architecture
```

构建：

```text
ggml Graph
```

Graph 再经过：

```text
Backend Scheduler
```

落到：

```text
CPU
CUDA
Metal
Vulkan
...
```

最终由：

```text
Kernel
```

完成真实计算。

完整主线：

```text
HF Model
↓
GGUF
↓
Quantization
↓
llama_model
↓
llama_context
↓
llama_batch
↓
llama_decode
↓
ggml Graph
↓
Backend
↓
Kernel
↓
Hardware
↓
Logits
↓
Sampler
↓
Token
```

如果这一条链可以自己完整解释：

> 那么你已经不再只是“会用 llama.cpp 跑模型”，而是开始真正进入 **LLM Runtime / Edge AI Systems** 这一层。
