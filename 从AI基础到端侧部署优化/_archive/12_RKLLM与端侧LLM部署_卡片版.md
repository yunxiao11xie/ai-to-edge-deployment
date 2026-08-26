# 12_RKLLM与端侧LLM部署

> **所属路线**：AI 学习路线 · 第三部分  
> **定位**：把前面的 LLM 推理原理、量化、Runtime、AI Compiler 与 Rockchip NPU 工具链真正合流，理解 **Hugging Face LLM → RKLLM-Toolkit → 量化 / 编译 → `.rkllm` → RKLLM Runtime → RKNPU → 端侧流式生成** 的完整链路  
> **核心仓库**：[`airockchip/rknn-llm`](https://github.com/airockchip/rknn-llm)  
> **关联仓库**：[`airockchip/rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2)  
> **目标平台重点**：RK3576  
> **重点模型**：Qwen2 / Qwen2.5 / Qwen3 / Qwen3.5；本章特别以 **Qwen3.5-2B → RK3576** 作为主线案例  
> **当前官方版本参考**：RKNN-LLM / RKLLM v1.3.0  
> **学习边界**：本章重点解释 RKLLM 的编译、量化、Runtime、KV Cache、Prompt Cache、Sampling、LoRA、多模态和性能模型；下一章 `13_RK3576端侧AI实践.md` 再完成完整环境搭建、部署命令、Benchmark 和项目化落地。  
> **资料检查日期**：2026-08-25

---

# 0. 为什么需要单独学习 RKLLM？

上一章已经有：

```text
PyTorch / ONNX
↓
RKNN-Toolkit2
↓
.rknn
↓
RKNN Runtime
↓
RKNPU
```

那么一个自然的问题是：

> LLM 为什么不能直接按照普通 CNN / YOLO 模型那样，导出 ONNX → `.rknn` → `rknn_run()`？

核心原因是：

```text
LLM
≠
一次静态前向计算。
```

LLM 推理本质是：

```text
Prompt
↓
Tokenization
↓
Prefill
↓
KV Cache
↓
Decode Token 1
↓
Update KV
↓
Sampling
↓
Decode Token 2
↓
Update KV
↓
Sampling
↓
...
```

这不是：

```text
Input Tensor
↓
Model
↓
Output Tensor
```

这么简单。

因此 Rockchip 单独提供：

```text
RKLLM-Toolkit
+
RKLLM Runtime。
```

---

# 1. 本章最终要建立的完整链

```text
Hugging Face LLM
        │
        ▼
  RKLLM-Toolkit
        │
        ├── Model Parsing
        ├── Architecture Recognition
        ├── Weight Conversion
        ├── Quantization
        ├── Graph Optimization
        ├── Target Compilation
        ├── Context / Cache Configuration
        └── RKNPU Mapping
        │
        ▼
      .rkllm
        │
        ▼
   RKLLM Runtime
        │
        ├── Tokenizer
        ├── Chat Template
        ├── Prefill
        ├── KV Cache
        ├── Decode
        ├── Sampling
        ├── Prompt Cache
        ├── History
        ├── LoRA
        └── Streaming Callback
        │
        ▼
   RKNPU Driver
        │
        ▼
      RKNPU
```

---

# 2. 最先固定 8 个概念

```text
RKNN-LLM
RKLLM
RKLLM-Toolkit
.rkllm
RKLLM Runtime
librkllmrt.so
RKNN-Toolkit2
.rknn
```

---

# 3. RKNN-LLM 是什么？

官方仓库：

- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm)

它可以理解为：

> **Rockchip 面向 LLM / VLM 的 NPU 部署 SDK 与示例工程仓库。**

仓库中包含：

```text
RKLLM Toolkit usage
RKLLM Runtime
C/C++ API
LLM demo
Server demo
Multimodal demo
Benchmark
Scripts
SDK docs。
```

---

# 4. RKLLM 是什么？

RKLLM 可以指：

```text
Rockchip LLM software stack
```

也经常指：

```text
RKLLM model format / runtime route。
```

本笔记尽量明确：

```text
RKLLM-Toolkit
=
Host Compiler / Converter

.rkllm
=
LLM Deployment Artifact

RKLLM Runtime
=
Board-side Runtime。
```

---

# 5. `.rkllm` 是什么？

最重要定义：

> **`.rkllm` 是经过 RKLLM-Toolkit 面向特定 Rockchip NPU 平台转换、量化和优化后的 LLM 部署模型。**

它不是：

```text
GGUF
```

也不是：

```text
.rknn
```

更不是：

```text
ONNX。
```

---

# 6. `.rkllm` 和 `.rknn` 的区别

普通：

```text
CNN / Detection / Classification
↓
RKNN-Toolkit2
↓
.rknn
```

LLM：

```text
Qwen / Llama / Gemma ...
↓
RKLLM-Toolkit
↓
.rkllm
```

---

# 7. 为什么需要两种 Artifact？

普通模型运行模式：

```text
Input
↓
Forward
↓
Output。
```

LLM：

```text
Prompt
↓
Prefill
↓
Persistent KV State
↓
Autoregressive Decode Loop
↓
Sampling
↓
Streaming Output。
```

Runtime 语义不同。

---

# 8. RKLLM Runtime

官方定义：

> RKLLM Runtime 为 Rockchip NPU 平台提供 C/C++ LLM 编程接口。

核心动态库：

```text
librkllmrt.so
```

板端应用：

```text
Application
↓
rkllm.h
↓
librkllmrt.so
↓
RKNPU Driver
↓
NPU。
```

---

# 9. RKNN-Toolkit2 在这章还要不要用？

要，但角色不同。

对于纯文本 LLM：

```text
主要使用 RKLLM-Toolkit。
```

对于 VLM：

```text
Vision Encoder
↓
RKNN-Toolkit2
↓
.rknn

LLM Decoder
↓
RKLLM-Toolkit
↓
.rkllm。
```

---

# 10. 一个 Qwen-VL 软件栈

```text
Image
↓
Vision Encoder
↓
.rknn
↓
RKNN Runtime
↓
Image Embedding
        │
        ▼
Text Prompt + Image Embedding
        │
        ▼
      .rkllm
        │
        ▼
    RKLLM Runtime
        │
        ▼
       LLM
```

这也是为什么：

```text
RKNN
+
RKLLM
```

会同时出现在多模态工程里。

---

# 11. 当前官方平台支持

当前 `rknn-llm` README 列出的平台包括：

```text
RK3588 Series
RK3576 Series
RK3562 Series
RV1126B Series。
```

本路线：

```text
重点 RK3576。
```

---

# 12. 当前官方 Qwen 支持

截至 RKLLM v1.3.0：

```text
Qwen2
Qwen2.5
Qwen3
Qwen3.5
```

已在官方支持列表中。

v1.3.0 还新增：

```text
Qwen3.5
Gemma4
SmolLM3。
```

---

# 13. 这对本路线非常重要

因为我们的核心实践模型：

```text
Qwen3.5-2B
```

已经从：

```text
“社区尝试”
```

进入：

```text
官方支持列表。
```

所以当前不应该再使用旧教程中：

```text
custom_model=True
```

等临时适配路径作为首选。

---

# 14. 版本变化一定要关注

旧 Issue / 教程可能写：

```text
Qwen3.5 不支持
```

但当前 v1.3.0：

```text
已经官方加入支持。
```

这就是为什么：

> Vendor AI SDK 必须结合具体版本学习。

---

# 15. 当前 RKLLM v1.3.0 的关键变化

官方 CHANGELOG 当前包括：

```text
Qwen3.5 support
Gemma4 support
SmolLM3 support

multimodal input interface optimization
cache reuse optimization

multiple EOS token IDs
ignore_eos_token

tokenizer callback
embedding callback

RK3576 long-context decoding optimization

embedding input quantization optimization

OpenAI API server compatibility improvement

per-request max_new_tokens override
per-request sampling override。
```

---

# 16. LLM Runtime 为什么变化这么快？

因为 LLM Runtime 不只是：

```text
MatMul。
```

还需要支持：

```text
New Architecture
New Attention
New Cache Model
New Tokenizer
Thinking Mode
Tool Calling
VLM
LoRA
Prompt Cache
Long Context。
```

---

# 17. RKLLM 完整 Host → Board Workflow

建议记住：

```text
Hugging Face Model
↓
准备 Quantization Dataset
↓
RKLLM()
↓
load_huggingface()
↓
build()
↓
export_rkllm()
↓
model.rkllm
↓
Copy to RK3576
↓
librkllmrt.so
↓
rkllm_init()
↓
rkllm_run()
↓
Streaming Callback
↓
Text
```

---

# 18. Host 侧最小转换代码

官方 example 当前结构类似：

```python
from rkllm.api import RKLLM

llm = RKLLM()

ret = llm.load_huggingface(
    model="/path/to/model",
    device="cuda",
    dtype="float32"
)

ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype="w4a16",
    quantized_algorithm="grq",
    target_platform="rk3576",
    num_npu_core=1,
    dataset="./data_quant.json",
    max_context=4096
)

ret = llm.export_rkllm(
    "./model_w4a16_rk3576.rkllm"
)
```

参数需要：

```text
根据当前模型 / SDK / Platform 调整。
```

---

# 19. `RKLLM()`

和上一章：

```python
RKNN()
```

类似。

但它针对：

```text
LLM Model Conversion。
```

---

# 20. `load_huggingface()`

当前官方 example：

```python
llm.load_huggingface(
    model=modelpath,
    model_lora=None,
    device='cuda',
    dtype='float32',
    custom_config=None,
    load_weight=True
)
```

---

# 21. `device='cuda'` 是什么意思？

注意：

```text
这是 Host 转换阶段。
```

表示：

```text
RKLLM-Toolkit
```

可以在转换过程中使用：

```text
Host NVIDIA GPU
```

辅助处理。

它不意味着：

```text
最终模型跑 CUDA。
```

最终 Target：

```text
RK3576 RKNPU。
```

---

# 22. `dtype`

Host load 阶段：

```text
float32
float16
bfloat16
```

会影响：

```text
Host conversion memory
conversion behavior。
```

它不是简单等同：

```text
最终 .rkllm quant dtype。
```

---

# 23. 两种 Dtype 不要混淆

```text
load_huggingface(dtype=...)
=
Host Model Load Dtype

build(quantized_dtype=...)
=
Target Quantization Dtype。
```

---

# 24. GGUF 输入支持

当前 RKLLM CHANGELOG 也已经出现：

```text
GGUF model conversion
```

当前注明支持：

```text
q4_0
fp16。
```

所以 RKLLM Toolkit 可以在特定支持范围：

```text
load GGUF
↓
convert to RKLLM。
```

---

# 25. 但不要因此认为

```text
GGUF
=
RKLLM Runtime Native Model。
```

不是。

流程仍然是：

```text
GGUF
↓
RKLLM Toolkit
↓
RKLLM artifact。
```

---

# 26. `build()` 是核心 Compiler 阶段

官方 example：

```python
llm.build(
    do_quantization=True,
    optimization_level=...,
    quantized_dtype=...,
    quantized_algorithm=...,
    target_platform=...,
    num_npu_core=...,
    dataset=...,
    hybrid_rate=...,
    max_context=...
)
```

---

# 27. 可以怎么理解 `build()`？

```text
HF Architecture
↓
Parse Model
↓
Normalize Graph
↓
Recognize Transformer Blocks
↓
Quantize Weight / Activation
↓
Optimize Attention / MatMul / Norm
↓
Plan Context / KV Cache
↓
Map to RKNPU
↓
Generate RKLLM Artifact。
```

内部实现是 Vendor-specific：

```text
不要猜具体 IR / ISA。
```

---

# 28. `target_platform`

RK3576：

```python
target_platform="rk3576"
```

这和上一章：

```text
RKNN target_platform
```

完全同一个 Compiler 思想：

```text
Target-aware Compilation。
```

---

# 29. `num_npu_core`

告诉 Compiler：

```text
模型计划使用多少 NPU Core。
```

---

# 30. 为什么不是越多越好？

可能受：

```text
Model Architecture
Partition Strategy
DDR Bandwidth
Parallel Efficiency
Platform NPU topology。
```

影响。

所以：

```text
需要 Benchmark。
```

---

# 31. RK3576 与 RK3588 不应照抄配置

即使：

```text
都支持 RKLLM。
```

它们的：

```text
NPU Resources
CPU
DDR
Runtime optimization
```

不同。

---

# 32. `max_context`

非常重要。

例如：

```python
max_context=4096
```

告诉 Toolchain：

```text
目标最大 Context Length。
```

---

# 33. 为什么 max_context 影响部署？

因为：

```text
KV Cache Memory
```

随 Context 增长。

---

# 34. KV Cache Memory 近似

标准 Transformer：

```text
KV bytes
≈
2
×
num_layers
×
num_kv_heads
×
head_dim
×
context_length
×
bytes_per_element。
```

---

# 35. GQA 为什么重要？

如果：

```text
num_kv_heads
<
num_q_heads
```

KV Cache：

```text
显著减小。
```

---

# 36. Context 越长意味着

```text
KV Memory ↑
Decode Memory Traffic ↑
Attention Cost ↑
```

所以：

```text
max_context 不是越大越好。
```

---

# 37. 端侧应选择真实需要的 Context

例如：

```text
4096
```

可能比：

```text
32768
```

更适合产品。

除非确实需要：

```text
长文档 / Agent memory。
```

---

# 38. Quantization：RKLLM 的核心

当前官方 example 中：

```text
w8a8 / w8a8_gx
```

推荐：

```text
normal algorithm。
```

而：

```text
w4a16 / w4a16_gx
```

推荐：

```text
grq algorithm。
```

具体版本以当前 example 为准。

---

# 39. W8A8

概念：

```text
Weight 8-bit
Activation 8-bit。
```

---

# 40. W4A16

概念：

```text
Weight 4-bit
Activation / compute path 16-bit related。
```

---

# 41. 为什么端侧 LLM 特别喜欢 W4A16？

LLM Decode：

```text
每生成一个 Token
需要读取大量 Weight。
```

Weight：

```text
8 bit
↓
4 bit
```

理论上：

```text
Weight Memory Traffic
约减半。
```

---

# 42. 所以 W4A16 可能同时

```text
Model Memory ↓
DDR Bandwidth Pressure ↓
Decode Tokens/s ↑。
```

---

# 43. RK3576 官方 Benchmark 是很好的现实案例

当前官方 benchmark：

```text
Qwen3.5-2B
RK3576
seqlen = 128
new_tokens = 64
```

---

# 44. W4A16

当前官方数据：

```text
TTFT:
1661.01 ms

Decode:
11.03 tokens/s

Memory:
1236.12 MB。
```

---

# 45. W4A16_g128

```text
TTFT:
1852.42 ms

Decode:
10.10 tokens/s

Memory:
1280.15 MB。
```

---

# 46. W8A8

```text
TTFT:
1678.11 ms

Decode:
6.73 tokens/s

Memory:
2131.55 MB。
```

---

# 47. 这个结果说明什么？

至少在该：

```text
官方特定版本
特定 Shape
特定频率
特定转换参数
```

测试下：

```text
W4A16
```

不只是：

```text
更省内存。
```

还：

```text
Decode 明显更快。
```

---

# 48. 为什么？

最合理的系统解释是：

```text
Decode Weight Streaming
```

高度受：

```text
Memory Bandwidth
```

影响。

Weight 更小：

```text
每 Token 需要搬的数据 ↓。
```

---

# 49. 不要把 11.03 tokens/s 当固定性能

官方 benchmark 前提包括：

```text
CPU / NPU 最大频率
固定 seqlen / new_tokens
当前 SDK
特定 optimization_level。
```

真实项目还受：

```text
Prompt Length
Context
Temperature
CPU load
Thermal
DDR
OS
Background task
Runtime Version。
```

---

# 50. 官方 Benchmark 当前特别说明

benchmark 中写明：

```text
所有模型用 optimization_level=0
```

以获得其测试中的优化 Runtime 性能。

但：

```text
API export example
```

仍能看到：

```text
optimization_level=1。
```

---

# 51. 如何处理这种版本差异？

原则：

```text
Benchmark 场景
→ 按 benchmark.md 当前说明

普通转换
→ 按目标 SDK 当前官方 example / SDK 文档。
```

不要：

```text
把旧 example 的参数当永恒最佳实践。
```

---

# 52. `data_quant.json`

RKLLM 量化 Dataset 不同于 CNN 图像 Calibration。

官方 example 的格式类似：

```json
[
  {
    "input": "Human: 你好\nAssistant: ",
    "target": "你好！"
  }
]
```

---

# 53. 为什么 LLM Calibration 是文本？

因为 Compiler 需要观察：

```text
真实 LLM Activation Distribution。
```

数据由：

```text
Token Sequence
```

驱动。

---

# 54. Calibration Dataset 设计原则

应该覆盖：

```text
真实领域

常见 Prompt Length

中英文

System Prompt

典型输出类型。
```

---

# 55. 如果做技术助手

Dataset 可包含：

```text
代码
嵌入式
Linux
硬件
中文问答
英文术语。
```

---

# 56. 不要直接用完全不相关文本

例如产品是：

```text
工业设备助手。
```

Calibration 却全是：

```text
诗歌 / 新闻。
```

可能不够代表。

---

# 57. `grq`

当前 example 对：

```text
w4a16
```

推荐：

```text
grq。
```

可以把它先理解：

> Rockchip 针对 4-bit 权重量化提供的量化算法路径之一。

本阶段：

```text
会正确使用 + Benchmark + Accuracy Compare
```

即可。

---

# 58. Group Quantization

例如：

```text
w4a16_g128。
```

表示：

```text
带 Group Size 的 4-bit Weight Quant。
```

---

# 59. 为什么 Group Quant 可能提高 Accuracy？

不同 Weight group：

```text
各自有量化尺度。
```

比：

```text
大范围共用一个 Scale
```

更细致。

---

# 60. 但为什么可能变慢？

可能需要：

```text
更多 Scale Metadata
更复杂 Dequant
Kernel / Memory Access 变化。
```

---

# 61. 所以端侧选型永远是

```text
Accuracy
Memory
Tokens/s
TTFT
```

四者一起看。

---

# 62. Mixed Quantization

当前 CHANGELOG 也包含：

```text
mixed quantization algorithm。
```

思路：

```text
部分 group quant
+
部分 non-group quant。
```

目标：

```text
Accuracy / Performance tradeoff。
```

---

# 63. 量化不是一句“转 INT4”

真正要记录：

```text
quantized_dtype

quantized_algorithm

group size

hybrid / mixed ratio

calibration data

SDK version。
```

---

# 64. `export_rkllm()`

Compiler 完成：

```python
llm.export_rkllm(
    "./qwen3.5-2b-w4a16-rk3576.rkllm"
)
```

---

# 65. 到这里 Host 侧完成

板端不需要：

```text
Transformers
PyTorch
CUDA。
```

板端需要：

```text
.rkllm
librkllmrt.so
Application
RKNPU Driver。
```

---

# 66. RKLLM Runtime 的核心文件

头文件：

- [rkllm.h](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)

核心动态库：

```text
librkllmrt.so。
```

---

# 67. Runtime 最基本生命周期

```text
rkllm_createDefaultParam
↓
configure RKLLMParam
↓
rkllm_init
↓
prepare RKLLMInput
↓
prepare RKLLMInferParam
↓
rkllm_run
↓
callback streaming output
↓
rkllm_destroy。
```

---

# 68. 为什么 RKLLM 用 Callback？

LLM 输出不是：

```text
一次性一个 Tensor。
```

而是：

```text
Token
Token
Token
...
```

所以最自然：

```text
Streaming Callback。
```

---

# 69. `RKLLMParam`

官方当前结构包含：

```text
model_path

max_context_len

max_new_tokens

top_k

top_p

temperature

repeat_penalty

frequency_penalty

presence_penalty

mirostat

skip_special_token

ignore_eos_token

is_async

extend_param。
```

---

# 70. 这说明 Runtime 不只是 NPU Executor

它同时承担：

```text
LLM Generation Runtime。
```

---

# 71. `rkllm_createDefaultParam()`

当前 API 提供：

```cpp
RKLLMParam param =
    rkllm_createDefaultParam();
```

推荐：

```text
先用 Default
再修改必要字段。
```

---

# 72. 为什么不自己 memset 后全部填？

Vendor SDK：

```text
以后增加字段。
```

Default factory：

```text
兼容性通常更好。
```

---

# 73. `model_path`

```cpp
param.model_path =
    "/path/model.rkllm";
```

---

# 74. `max_context_len`

Runtime 实际允许使用的：

```text
最大 Context。
```

它不能：

```text
无视模型 Compile 时的限制无限增大。
```

---

# 75. `max_new_tokens`

每次最多：

```text
生成多少 Token。
```

---

# 76. 当前 v1.3.0

还支持：

```text
RKLLMInferParam
```

在单次 Request：

```text
覆盖 max_new_tokens。
```

---

# 77. 这有什么意义？

Server：

```text
不同请求
```

可以：

```text
不同 generation limit。
```

而不必：

```text
重新 init model。
```

---

# 78. Sampling 参数

```text
top_k
top_p
temperature
repeat_penalty
frequency_penalty
presence_penalty。
```

---

# 79. Temperature

低：

```text
更确定。
```

高：

```text
更随机。
```

---

# 80. `top_k`

只从：

```text
概率最大的 K 个 Token
```

中采样。

---

# 81. `top_p`

选择最小集合：

```text
累计概率 >= p。
```

---

# 82. Greedy

典型：

```text
top_k = 1。
```

---

# 83. Benchmark 时为什么最好 Greedy？

随机采样：

```text
Output Length / Token
```

可能变化。

性能实验：

```text
更难复现。
```

---

# 84. Repeat Penalty

降低：

```text
重复 Token。
```

---

# 85. Sampling 主要跑在哪里？

具体实现：

```text
由 RKLLM Runtime 决定。
```

不要未经官方说明断言：

```text
一定 CPU
```

或：

```text
一定 NPU。
```

---

# 86. `RKLLMExtendParam`

当前 Runtime header 中包含：

```text
n_batch
use_cross_attn
enabled_cpus_num / mask
embed_flash
...
```

---

# 87. `n_batch`

当前 header 注释：

```text
一个 forward 中并发处理的输入 sample 数。
```

默认：

```text
1。
```

---

# 88. 这和 vLLM 的 Continuous Batching 一样吗？

不是。

不要直接等同：

```text
RKLLM n_batch
=
vLLM scheduler batch。
```

这里只能说：

```text
RKLLM Runtime 提供 batched inference 相关参数。
```

具体调度机制：

```text
是 Vendor Runtime 实现。
```

---

# 89. `enabled_cpus_num / enabled_cpus_mask`

LLM Runtime 即使主要用：

```text
NPU
```

也仍需要：

```text
CPU。
```

---

# 90. CPU 可能负责什么？

可能包括：

```text
Runtime Control
Tokenizer
Sampling
Memory
Application
Pre/Post logic。
```

具体算子位置：

```text
不要未验证就断言。
```

---

# 91. 为什么官方 Server Demo 会设置 CPU mask？

为了：

```text
控制 Runtime 使用的 CPU Core。
```

改善：

```text
稳定性 / 性能。
```

---

# 92. 这再次说明：

```text
Edge LLM
=
NPU
+
CPU
+
DDR
+
Runtime。
```

不是：

```text
只有 NPU TOPS。
```

---

# 93. `RKLLMInputType`

当前 API 有四类：

```text
RKLLM_INPUT_PROMPT
RKLLM_INPUT_TOKEN
RKLLM_INPUT_EMBED
RKLLM_INPUT_MULTIMODAL。
```

---

# 94. Prompt Input

```text
直接输入文本。
```

Runtime：

```text
负责 tokenizer。
```

---

# 95. Token Input

```text
直接输入 token IDs。
```

---

# 96. 为什么 Token Input 有价值？

上层：

```text
自己管理 tokenizer。
```

可以：

```text
避免重复 tokenizer
做特殊 prompt control
接 Agent / protocol。
```

---

# 97. Embedding Input

直接：

```text
输入 embedding vector。
```

---

# 98. 为什么 Embedding Input 很重要？

VLM：

```text
Image Encoder
↓
Image Embedding
```

需要送进：

```text
LLM。
```

---

# 99. Multimodal Input

当前 Runtime 直接提供：

```text
RKLLM_INPUT_MULTIMODAL。
```

v1.3.0：

```text
还优化了 multimodal interface。
```

---

# 100. Tokenizer Callback

当前 v1.3.0 新增：

```text
tokenizer callback。
```

---

# 101. 为什么需要 Callback？

某些模型：

```text
没有 internal tokenizer
```

或：

```text
上层希望接管 tokenizer。
```

Runtime 可以：

```text
回调 Application。
```

---

# 102. Embedding Callback

类似：

```text
模型无 internal embedding
```

时：

```text
Application 提供 embedding。
```

---

# 103. 这说明 Runtime 越来越模块化

```text
Tokenizer
Embedding
LLM Core
```

可以：

```text
部分由 Application 管理。
```

---

# 104. `RKLLMInput`

当前结构包含：

```text
role

enable_thinking

input_type

prompt / token / embed / multimodal union。
```

---

# 105. `role`

当前 header 注释包括：

```text
"user"
"tool"
```

等输入角色。

---

# 106. 为什么 Tool Role 很重要？

Agent：

```text
LLM
↓
Tool Call
↓
Tool Result
↓
LLM。
```

Runtime 需要：

```text
正确构造 Chat Context。
```

---

# 107. `enable_thinking`

当前 API：

```text
可控制 Qwen3 thinking mode。
```

---

# 108. Thinking Mode 对 Edge 有什么影响？

如果模型输出：

```text
更多 reasoning tokens
```

总：

```text
Decode Time
Energy
Latency
```

会增加。

---

# 109. 所以产品不能只看单 Token/s

还要看：

```text
Total Generated Tokens。
```

---

# 110. 一个模型 12 tok/s

输出：

```text
100 Token
```

约：

```text
8.3s。
```

如果 Thinking：

```text
800 Token
```

即使速度一样：

```text
总响应时间大幅增加。
```

---

# 111. `RKLLMInferParam`

当前包括：

```text
mode

lora_params

prompt_cache_params

sampling_params

keep_history

max_new_tokens。
```

---

# 112. 为什么有 Per-request `sampling_params`？

Server：

```text
Request A:
temperature 0

Request B:
temperature 0.8
```

不需要：

```text
重新初始化模型。
```

---

# 113. `keep_history`

控制：

```text
是否保留对话历史。
```

---

# 114. LLM History 的本质是什么？

不是只保留：

```text
string。
```

Runtime 还需要：

```text
对应 KV Cache / Context State。
```

---

# 115. `keep_history=1`

多轮：

```text
User A
↓
Assistant A
↓
User B
```

可以：

```text
继续基于历史生成。
```

---

# 116. `keep_history=0`

更接近：

```text
独立请求。
```

---

# 117. KV Cache

这是端侧 LLM Runtime 最核心 State。

Prefill：

```text
每层生成 K/V。
```

Decode：

```text
新 Token
只追加新的 K/V。
```

---

# 118. 没有 KV Cache 会怎样？

每生成一个 Token：

```text
重新计算全部历史 Token。
```

复杂度和延迟：

```text
无法接受。
```

---

# 119. RKLLM 提供 `rkllm_clear_kv_cache`

当前 API 支持：

```text
清空全部 KV

保留 System Prompt

清指定 position range。
```

---

# 120. 为什么要 Clear KV？

例如：

```text
新会话
```

需要：

```text
清历史。
```

---

# 121. 为什么可以保留 System Prompt？

许多应用：

```text
System Prompt 固定。
```

可以：

```text
保留系统上下文
减少重复 Prefill。
```

---

# 122. Prompt Cache

RKLLM 当前还有：

```text
rkllm_load_prompt_cache
rkllm_release_prompt_cache。
```

并可在：

```text
RKLLMPromptCacheParam
```

中：

```text
save_prompt_cache
prompt_cache_path。
```

---

# 123. Prompt Cache 和 KV Cache 有什么区别？

KV Cache：

```text
当前 Runtime Context 中的
Attention State。
```

Prompt Cache：

```text
把可复用 Prompt 计算结果
保存 / 加载。
```

---

# 124. 类比 vLLM Prefix Cache

概念上：

```text
都是减少重复 Prefill。
```

但：

```text
实现机制不同。
```

不要把它们视为同一数据格式。

---

# 125. Prompt Cache 最适合

```text
固定 System Prompt

固定 Knowledge Prefix

设备指令

Agent Role Prompt。
```

---

# 126. 例如设备助手

所有请求：

```text
System:
你是某工业控制器的维护助手...
```

这一大段：

```text
完全重复。
```

Prompt Cache：

```text
可以减少 TTFT。
```

---

# 127. Cache 对 Decode Tokens/s 有帮助吗？

主要：

```text
减少 Prefix Prefill。
```

并不代表：

```text
每个 Decode Token 的 Weight MatMul
变快。
```

---

# 128. `rkllm_clear_kv_cache()` 支持 Range

当前 API 允许：

```text
[start_pos, end_pos)
```

选择性清理。

---

# 129. 这体现：

```text
LLM Runtime
```

已经提供更细粒度：

```text
Context Management。
```

---

# 130. Prefill

输入：

```text
Prompt Tokens:
T 个。
```

模型：

```text
一次 / 分块处理。
```

输出：

```text
首个 next-token logits
+
T 个位置对应 KV。
```

---

# 131. Prefill 的主要性能指标

```text
TTFT

Prefill Time

Prompt Tokens/s。
```

---

# 132. RKLLM Runtime 当前 `RKLLMPerfStat`

直接提供：

```text
prefill_time_ms

prefill_tokens

generate_time_ms

generate_tokens

memory_usage_mb。
```

---

# 133. 这是非常有价值的 API

可以直接算：

```text
Prefill TPS
=
prefill_tokens
/
prefill_time。
```

---

# 134. Decode TPS

```text
generate_tokens
/
generate_time。
```

---

# 135. TTFT 和 Prefill Time 完全一样吗？

不一定严格等同。

端到端 TTFT 还可能包括：

```text
Tokenization
Scheduling
Runtime overhead
First sampling
Callback。
```

Benchmark 表中的：

```text
TTFT
```

应按其官方定义使用。

---

# 136. Decode

每一步：

```text
1 / small number new token
↓
Weight MatMul
↓
Read KV
↓
Attention
↓
Logits
↓
Sample
↓
Append KV。
```

---

# 137. RK3576 Decode 为什么 W4A16 可能特别占优？

Decode：

```text
每 Token 重复读取 Model Weight。
```

如果：

```text
Weight 4-bit
```

DDR 流量：

```text
大幅降低。
```

---

# 138. 这就是前面 Roofline 的现实落地

```text
Decode
≈
Memory-Bandwidth Sensitive。
```

---

# 139. Long Context Decode

Context 越长：

```text
KV Cache 越大
```

每个 Token：

```text
Attention 需要读取更多 KV。
```

---

# 140. 所以 Decode TPS 会随 Context 增长而下降

程度取决：

```text
Architecture
GQA / MQA
Attention Type
Runtime optimization
Memory bandwidth。
```

---

# 141. v1.3.0 为什么专门写 RK3576 long-context decoding optimization？

因为这正是：

```text
Edge LLM 的关键痛点。
```

---

# 142. Attention Architecture 变化

Qwen3.5 等新模型：

```text
不一定完全是传统 LLaMA-style Full Attention。
```

Vendor Runtime：

```text
必须针对新 architecture 做支持。
```

---

# 143. 这就是为什么“HF 能跑”不代表 RKLLM 一定能跑

需要：

```text
RKLLM Toolkit
认识 Architecture。
```

---

# 144. Model Support 的真正含义

不是只有：

```text
Model Name
```

还包括：

```text
Architecture
Operator
Weight Layout
Attention
Tokenizer
Cache Semantics。
```

---

# 145. `rkllm_run()`

同步接口：

```cpp
rkllm_run(
    handle,
    &input,
    &infer_param,
    userdata
);
```

---

# 146. `rkllm_run_async()`

当前 API 也支持：

```text
异步推理。
```

---

# 147. 异步有什么价值？

GUI / Server：

```text
主线程
```

不需要：

```text
等待整个生成结束。
```

---

# 148. `rkllm_abort()`

可以：

```text
中止正在进行的生成。
```

---

# 149. 为什么对产品很重要？

用户：

```text
点击 Stop。
```

不能：

```text
让 NPU 继续生成 1000 Token。
```

---

# 150. `rkllm_is_running()`

可以：

```text
查询当前任务状态。
```

---

# 151. Callback 状态

RKLLM Result Callback：

```text
会收到生成结果与运行状态。
```

---

# 152. Result 包含

当前 API：

```text
text

token_id

last_hidden_layer

logits

perf。
```

---

# 153. 为什么 `token_id` 有价值？

可以：

```text
自定义 stream protocol
统计 Token
调试 tokenizer
构造 Agent control。
```

---

# 154. 为什么可以取 `logits`？

应用可以：

```text
自己做 Sampling

做分类 / scoring

分析概率。
```

---

# 155. Last Hidden Layer

可以用于：

```text
Embedding-like downstream task

Reward / classifier

Custom head。
```

---

# 156. 但取完整 Logits / Hidden State 很占内存

当前 Callback 注释甚至专门提供：

```text
释放 output buffer
```

相关行为。

原因：

```text
大 Vocabulary Logits
```

非常大。

---

# 157. LoRA

RKLLM Runtime 当前提供：

```text
rkllm_load_lora()
```

---

# 158. `RKLLMLoraAdapter`

包括：

```text
lora_adapter_path
lora_adapter_name
scale。
```

---

# 159. 为什么 LoRA Runtime Load 很有价值？

Base Model：

```text
只保存一份。
```

多个业务：

```text
Adapter A
Adapter B
Adapter C。
```

---

# 160. 端侧产品场景

例如：

```text
设备维护助手
客服助手
英语助手
```

可以：

```text
共享 Base
+
不同 Adapter。
```

---

# 161. LoRA 文件当前 example 中会看到 GGUF 路径示例

不要因此认为：

```text
Base model 也是 llama.cpp GGUF。
```

LoRA Adapter 的文件格式和 Base Runtime：

```text
是两个概念。
```

---

# 162. Chat Template

LLM 并不是把：

```text
"你好"
```

直接 Tokenize 就结束。

Instruct Model 实际需要：

```text
System
User
Assistant
Special Token。
```

---

# 163. Qwen 类模型常见形式

概念：

```text
<|im_start|>system
...
<|im_end|>

<|im_start|>user
...
<|im_end|>

<|im_start|>assistant
```

---

# 164. Template 错会发生什么？

模型：

```text
仍然能输出。
```

但可能：

```text
回复质量异常

不断续写 User

不停止

Thinking Token 异常。
```

---

# 165. RKLLM 提供 Chat Template API

官方 Server Demo 当前使用：

```text
rkllm_set_chat_template。
```

---

# 166. 当前版本还支持外部 Chat Template

说明：

```text
Template 也是 Runtime Compatibility 的一部分。
```

---

# 167. EOS

LLM 生成停止：

```text
EOS token。
```

---

# 168. v1.3.0 新增

```text
multiple EOS token IDs
```

以及：

```text
ignore_eos_token。
```

---

# 169. 为什么多个 EOS？

现代 Chat Model：

```text
可能有多个终止 / turn-end special token。
```

---

# 170. `ignore_eos_token`

如果开启：

```text
模型遇到 EOS 仍继续生成。
```

通常：

```text
Benchmark / 特殊控制
```

才会使用。

产品默认：

```text
慎用。
```

---

# 171. Qwen Thinking Mode

Qwen3：

```text
Thinking
```

和：

```text
Non-thinking
```

可能对：

```text
Output Token Count
Quality
Latency
```

影响巨大。

---

# 172. Benchmark 时必须记录

```text
enable_thinking

max_new_tokens

EOS behavior。
```

否则两个结果：

```text
不可比较。
```

---

# 173. 多模态

官方当前多模态 demo：

```text
Vision + Projector
→ RKNN

LLM
→ RKLLM。
```

---

# 174. Qwen3.5 多模态

当前 README 已给出：

```text
Qwen3.5-0.8B
```

多模态 quickstart 示例。

说明当前：

```text
Qwen3.5 VLM route
```

已经进入官方示例。

---

# 175. 为什么 Vision Encoder 还是 `.rknn`？

因为 Vision Encoder 更像：

```text
普通 Tensor Graph。
```

---

# 176. 为什么 Decoder 是 `.rkllm`？

因为 Decoder：

```text
需要 Autoregressive State。
```

---

# 177. VLM 的真正数据接口

```text
Image
↓
Vision Encoder
↓
Embedding / feature token
↓
LLM Input
↓
Text generation。
```

---

# 178. 当前 RKLLMInput 直接支持 Multimodal

这让：

```text
Image / Video + Prompt
```

集成更加统一。

---

# 179. Cross Attention

当前 RKLLM header 还定义：

```text
RKLLMCrossAttnParam。
```

用于：

```text
Decoder Cross Attention。
```

---

# 180. 其中包括

```text
encoder K cache

encoder V cache

encoder mask

encoder position。
```

---

# 181. 这说明什么？

RKLLM Runtime：

```text
不仅覆盖纯 Decoder-only LLM。
```

还在扩展：

```text
encoder-decoder / multimodal cache interaction。
```

---

# 182. Server Demo

官方仓库提供：

```text
rkllm_server_demo。
```

---

# 183. 为什么值得读？

它把：

```text
C Runtime
```

封装到：

```text
Python / Flask / Gradio / HTTP。
```

---

# 184. 当前 v1.3.0

还特别写：

```text
improved compatibility with OpenAI API interfaces。
```

---

# 185. 这使上层架构变成

```text
Agent / RAG / App
↓
OpenAI-compatible API
↓
RKLLM Server
↓
RK3576。
```

---

# 186. 与 vLLM 的关系

vLLM：

```text
Data Center GPU Serving。
```

RKLLM Server：

```text
Edge NPU Local Serving。
```

---

# 187. API 可能相似

但底层：

```text
完全不同。
```

---

# 188. 一个 Edge RAG 架构

```text
User
↓
Local App
↓
Retriever
↓
Local Vector DB
↓
Context
↓
RKLLM Server
↓
Qwen3.5-2B
↓
Answer。
```

---

# 189. 一个 Edge Agent 架构

```text
User
↓
Agent
↓
RKLLM
↓
Tool Call
↓
Local Device API
↓
Tool Result
↓
RKLLM
↓
Answer。
```

---

# 190. 端侧 Agent 最大优势

```text
Privacy

Offline

Low cloud cost

Low network dependency

Device control。
```

---

# 191. 最大限制

```text
Model capability

Memory

Decode speed

Context length

Power

Tool ecosystem。
```

---

# 192. 为什么 Qwen3.5-2B 很适合当前 RK3576 学习？

它位于一个比较合理平衡：

```text
不是太小
→ 能看到 LLM 能力

不是太大
→ W4A16 memory 可控。
```

---

# 193. 当前官方 RK3576 Benchmark

Qwen3.5：

```text
0.8B
2B
4B
```

都有结果。

---

# 194. 0.8B W4A16

当前官方：

```text
18.79 tokens/s
689.50 MB
TTFT 1369.31 ms。
```

---

# 195. 2B W4A16

```text
11.03 tokens/s
1236.12 MB
TTFT 1661.01 ms。
```

---

# 196. 4B W4A16

```text
5.02 tokens/s
2420.34 MB
TTFT 3976.41 ms。
```

---

# 197. 这三组数据非常适合做 Model Size Tradeoff

```text
0.8B:
速度优先

2B:
平衡

4B:
能力优先
但 latency 明显增加。
```

---

# 198. 不要只看 Weight Memory

实际：

```text
Runtime Memory
```

还包括：

```text
KV Cache

Workspace

Runtime

Tokenizer

Application。
```

---

# 199. Qwen3.5-2B 官方 Benchmark Memory

```text
~1.24 GB
```

只是该：

```text
测试配置
```

下的观测。

Context 拉长：

```text
Memory 还会增长。
```

---

# 200. 这就是为什么板子 RAM 很重要

假设：

```text
4GB RAM。
```

还要给：

```text
Linux
Desktop
App
Camera
Vector DB
```

留空间。

---

# 201. 8GB / 16GB 更适合做 LLM 实验

但：

```text
最终产品仍要按真实 memory budget 设计。
```

---

# 202. Benchmark 指标应该怎么拆？

至少：

```text
Model Load Time

TTFT

Prefill Tokens/s

Decode Tokens/s

Peak Memory

Total Response Time

Power

Temperature。
```

---

# 203. 为什么只有 Token/s 不够？

用户体验：

```text
按下 Enter
↓
多久看到第一个字？
```

这是：

```text
TTFT。
```

---

# 204. 长 Prompt

即使 Decode：

```text
11 tok/s。
```

如果 Prefill：

```text
5s。
```

体验仍然：

```text
很慢。
```

---

# 205. Agent 场景更看重 Prefill

因为每一步 Tool Loop：

```text
System Prompt
History
Tool Schema
Tool Result
```

可能非常长。

---

# 206. Prompt Cache 在 Agent 中非常重要

固定：

```text
System Prompt
Tool Schema
```

可以：

```text
Cache。
```

---

# 207. RAG 也一样

如果重复：

```text
固定 instruction prefix
```

可考虑：

```text
Prompt Cache。
```

---

# 208. Context Management

端侧不能无脑：

```text
一直 append History。
```

---

# 209. 原因

```text
KV Memory ↑

Attention Cost ↑

Decode TPS ↓

TTFT ↑。
```

---

# 210. 常见策略

```text
Sliding History

Summarize Old Turns

Keep System Prompt

Keep Latest N Turns

External Memory / RAG。
```

---

# 211. 这比单纯增大 max_context 更实用

端侧：

```text
Context Engineering
```

本身是：

```text
系统优化。
```

---

# 212. Prompt Cache vs History

Prompt Cache：

```text
复用固定 prefix。
```

History：

```text
保存当前 conversation state。
```

---

# 213. 两者可以同时使用

```text
System Prompt Cache
+
Conversation KV。
```

---

# 214. Long-context 性能测试

建议至少测：

```text
Prompt:
128
512
1024
2048
4096

Output:
128。
```

记录：

```text
TTFT
Decode tok/s
Memory。
```

---

# 215. 为什么 Decode 也要随 Context 测？

Attention：

```text
每个新 Token
要看历史 KV。
```

Context：

```text
越长
KV read 越多。
```

---

# 216. Quantization Benchmark

Qwen3.5-2B：

```text
W4A16

W4A16_g128

W8A8。
```

---

# 217. 不仅记录速度

还要：

```text
回答质量

Long-context quality

中文能力

代码能力

Instruction following。
```

---

# 218. 量化 Quality Evaluation

最好建立：

```text
20~100 个固定 Prompt。
```

---

# 219. Prompt 分类

```text
中文知识

数学

代码

嵌入式

长文本

格式化输出

Tool Call

安全拒答。
```

---

# 220. 为什么固定 Prompt 很重要？

每次：

```text
Toolkit Upgrade
Quantization Change
```

可以：

```text
Regression Test。
```

---

# 221. 不要只肉眼觉得“好像还行”

可以记录：

```text
Exact Match

JSON Valid Rate

Task Score

Human Preference。
```

---

# 222. Qwen3.5-2B RK3576 推荐实验路线

---

## Step 1：先跑官方预转换模型

目的：

```text
验证 Board Runtime。
```

不要第一步：

```text
自己转换。
```

---

# 223. Step 2：验证 RKLLM Runtime

跑：

```text
官方 llm_demo。
```

记录：

```text
Model Load
TTFT
Token/s
Memory。
```

---

# 224. Step 3：自己转换 W4A16

```text
HF Qwen3.5-2B
↓
RKLLM Toolkit v1.3.0
↓
w4a16
↓
rk3576
↓
.rkllm。
```

---

# 225. Step 4：比较官方 / 自转

如果：

```text
性能差很多。
```

检查：

```text
Toolkit version
optimization_level
quant algorithm
max_context
num_npu_core
frequency
Runtime version。
```

---

# 226. Step 5：W8A8

同模型：

```text
重新 build。
```

比较：

```text
Memory
Decode TPS
Quality。
```

---

# 227. Step 6：g128

测试：

```text
w4a16_g128。
```

观察：

```text
Accuracy / Performance。
```

---

# 228. Step 7：Context Sweep

```text
128
512
1024
2048
4096。
```

---

# 229. Step 8：Prompt Cache

固定 System Prompt：

```text
第一次：
build cache

第二次：
load cache。
```

比较：

```text
TTFT。
```

---

# 230. Step 9：History

```text
keep_history=0
vs
1。
```

理解：

```text
conversation state。
```

---

# 231. Step 10：Async / Abort

实现：

```text
stream generation

Stop button。
```

---

# 232. Step 11：Server

跑：

```text
rkllm_server_demo。
```

让 PC：

```text
OpenAI Client
↓
RK3576。
```

---

# 233. Step 12：Agent

做一个最小：

```text
calculator tool

system info tool

GPIO mock tool。
```

---

# 234. Step 13：RAG

本地：

```text
Markdown docs
↓
Embedding
↓
Retriever
↓
RKLLM。
```

---

# 235. Step 14：Sustained Benchmark

连续：

```text
30 分钟。
```

记录：

```text
temperature
frequency
tokens/s drift
memory
thermal。
```

---

# 236. 推荐最小 C++ Runtime 结构

```text
rkllm_minimal/
│
├── CMakeLists.txt
├── model/
│   └── qwen3.5-2b.rkllm
├── lib/
│   └── librkllmrt.so
├── include/
│   └── rkllm.h
└── src/
    └── main.cpp
```

---

# 237. `main.cpp` 心智结构

```text
signal handler
↓
create default param
↓
model path
↓
context / generation param
↓
callback
↓
rkllm_init
↓
input
↓
infer param
↓
rkllm_run
↓
stream callback
↓
rkllm_destroy。
```

---

# 238. Callback 里第一版只做什么？

```text
printf(result->text)
```

先跑通。

不要第一版就加：

```text
JSON
SSE
Agent
WebSocket
GUI。
```

---

# 239. 第二版

加入：

```text
token count
perf
state。
```

---

# 240. 第三版

加入：

```text
HTTP / SSE。
```

---

# 241. Server 端流式响应

```text
RKLLM Callback
↓
Queue
↓
SSE Chunk
↓
Client。
```

---

# 242. 为什么不要直接在 Callback 做重活？

Callback：

```text
属于 Runtime 输出路径。
```

如果：

```text
阻塞太久。
```

可能：

```text
影响生成。
```

更稳妥：

```text
轻量 push queue。
```

---

# 243. Async Runtime

```text
rkllm_run_async
```

更适合：

```text
Server / UI。
```

---

# 244. Thread Safety

不要未经文档验证就假设：

```text
一个 handle
可以被多个线程同时 run。
```

---

# 245. 更稳妥

```text
一个 handle
一个 active generation
```

结合：

```text
Request Queue。
```

具体并发能力：

```text
以当前 SDK 文档为准。
```

---

# 246. 多用户 Serving

RK3576 不是：

```text
vLLM 多 GPU Server。
```

---

# 247. 端侧更现实

```text
1 active generation

small queue

local clients。
```

---

# 248. 为什么？

```text
Memory
KV Cache
NPU throughput
```

有限。

---

# 249. 如果做多用户

重点考虑：

```text
Queue

Abort

Timeout

Context isolation

KV cleanup

Memory upper bound。
```

---

# 250. `rkllm_clear_kv_cache`

Server 中必须正确管理：

```text
User A
```

和：

```text
User B
```

的 Context。

---

# 251. 不允许不同用户串 Context

否则：

```text
隐私
正确性
```

都有问题。

---

# 252. LoRA 多租户

可以：

```text
Base model
+
different adapter。
```

但仍需：

```text
Runtime memory / switching cost benchmark。
```

---

# 253. 模型冷启动

LLM Model：

```text
GB 级。
```

`rkllm_init()`：

```text
需要加载 / 初始化。
```

---

# 254. 所以产品应该

```text
常驻 Model。
```

不要每个请求：

```text
init
run
destroy。
```

---

# 255. 冷启动指标

记录：

```text
Model File Read

rkllm_init time

First Request TTFT。
```

---

# 256. Page Cache

Linux 第二次启动：

```text
可能更快。
```

因为：

```text
Model File 已在 Page Cache。
```

---

# 257. Benchmark 要区分

```text
Cold Start

Warm Start。
```

---

# 258. DDR 是端侧 LLM 的核心资源

LLM：

```text
Model Weight
KV Cache
Runtime Workspace
OS
Application
```

都在：

```text
共享 DDR。
```

---

# 259. RK3576 是 SoC

CPU / GPU / NPU：

```text
共享 Memory System。
```

所以：

```text
CPU Heavy Task
```

也可能影响：

```text
NPU LLM。
```

---

# 260. Benchmark 时

尽量：

```text
关闭不必要后台任务

固定频率

记录 DDR / CPU load。
```

---

# 261. 官方仓库当前提供

```text
fix_freq_rk3576.sh
```

等脚本。

---

# 262. 官方 benchmark 方法还建议

```text
RKLLM_LOG_LEVEL=1
```

用于：

```text
输出性能与内存信息。
```

---

# 263. 还提供

```text
eval_perf_watch_cpu.sh

eval_perf_watch_npu.sh。
```

---

# 264. 为什么同时看 CPU / NPU？

LLM Runtime：

```text
不是 NPU-only。
```

---

# 265. 一个典型性能表

建议：

| Context | Quant | TTFT | Prefill tok/s | Decode tok/s | Peak RAM | CPU% | NPU% |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | W4A16 | | | | | | |
| 512 | W4A16 | | | | | | |
| 2048 | W4A16 | | | | | | |
| 4096 | W4A16 | | | | | | |

---

# 266. 不要只 Benchmark 一个 Prompt

因为 Prompt：

```text
长度
语言
Thinking
```

都影响：

```text
执行路径 / output length。
```

---

# 267. 建议固定

```text
Prompt token count

Generated token count。
```

---

# 268. `ignore_eos_token`

做纯性能测试时：

```text
可以保证生成固定 token 数。
```

但需要确认：

```text
不会影响目标测试定义。
```

---

# 269. 真实产品测试

应该：

```text
正常 EOS。
```

---

# 270. RKLLM 与 llama.cpp：最重要对比

---

## llama.cpp

```text
HF
↓
GGUF
↓
llama.cpp
↓
ggml
↓
CPU / GPU。
```

---

## RKLLM

```text
HF
↓
RKLLM-Toolkit
↓
.rkllm
↓
RKLLM Runtime
↓
RKNPU。
```

---

# 271. llama.cpp 的优势

```text
Open
Cross-platform
GGUF ecosystem
Easy model experimentation
CPU friendly
Source visible。
```

---

# 272. RKLLM 的优势

```text
Use RKNPU
Vendor hardware optimization
Edge SoC power efficiency
RK3576-specific optimization。
```

---

# 273. 为什么应该两条都学？

llama.cpp：

```text
帮助理解 LLM Runtime。
```

RKLLM：

```text
帮助掌握真实 Vendor NPU Deployment。
```

---

# 274. 实际 RK3576 可以同时保留两个 Baseline

```text
Qwen3.5-2B GGUF
↓
llama.cpp CPU

Qwen3.5-2B W4A16
↓
RKLLM NPU。
```

---

# 275. 比较

```text
Model size

Load time

TTFT

Decode t/s

RAM

CPU load

NPU load

Power

Output quality。
```

---

# 276. 这是非常有价值的项目

因为最终能够回答：

> 同一个 LLM 在 ARM CPU Runtime 和 Vendor NPU Runtime 上，为什么性能、格式和优化方法完全不同？

---

# 277. RKLLM 与 vLLM

vLLM：

```text
Server
Many concurrent requests
GPU
Paged KV
Scheduler。
```

RKLLM：

```text
Edge SoC
Local LLM
Vendor NPU
Resource constrained。
```

---

# 278. 两者学习重点不同

vLLM：

```text
Serving System。
```

RKLLM：

```text
Hardware Deployment。
```

---

# 279. RKLLM 与 MLC-LLM

MLC：

```text
Open Compiler
Relax / TIR
Cross-platform GPU。
```

RKLLM：

```text
Vendor Compiler
RKNPU。
```

---

# 280. 共同点

```text
HF Model
↓
Compile
↓
Target-specific artifact
↓
Runtime。
```

---

# 281. 这就是 Compiler Mental Model 的复用

```text
MLC Model Library

.rknn

.rkllm
```

都可以从：

```text
Target-specific Deployment Artifact
```

角度理解。

---

# 282. RKLLM 和普通 RKNN 最核心区别

RKNN：

```text
Static-ish tensor graph runtime。
```

RKLLM：

```text
Stateful autoregressive runtime。
```

---

# 283. Stateful 包括

```text
KV Cache

History

Sampling State

Prompt Cache

Generation Loop。
```

---

# 284. 这也是为什么 API 完全不同

RKNN：

```text
rknn_inputs_set
rknn_run
rknn_outputs_get。
```

RKLLM：

```text
rkllm_run
+
stream callback
+
context / generation state。
```

---

# 285. Quantization 生态也不同

RKNN：

```text
CNN / general tensor quant。
```

RKLLM：

```text
LLM-specific W4A16 / W8A8 / group quant。
```

---

# 286. 不要把 `.rknn` W4A16 和 `.rkllm` W4A16 当成同一使用路径

即使名称：

```text
相似。
```

Toolchain：

```text
不同。
```

---

# 287. 一个端侧 LLM 系统的完整分层

```text
Application
│
├── UI
├── Agent
├── RAG
├── Tool
└── Session Manager
       │
       ▼
Generation API
       │
       ▼
RKLLM Runtime
│
├── Tokenizer
├── Chat Template
├── Prompt Cache
├── KV Cache
├── Prefill
├── Decode
├── Sampling
└── Callback
       │
       ▼
RKNPU Runtime / Driver
       │
       ▼
NPU
       │
       ▼
DDR / SoC
```

---

# 288. 性能优化必须分层

---

## Layer 1：Model

```text
0.8B / 2B / 4B

Architecture

GQA

Context。
```

---

## Layer 2：Quant

```text
W8A8
W4A16
Group Quant。
```

---

## Layer 3：Compiler

```text
Toolkit version
optimization level
num_npu_core
target。
```

---

## Layer 4：Runtime

```text
context
history
prompt cache
CPU affinity
async。
```

---

## Layer 5：System

```text
Frequency
DDR
Thermal
Background task
Linux。
```

---

## Layer 6：Application

```text
Prompt design
Output length
RAG context
Thinking mode。
```

---

# 289. 很多“LLM 太慢”其实是 Layer 6 问题

例如：

```text
System Prompt 4000 Token

RAG 每次塞 6000 Token

Thinking 输出 2000 Token。
```

即使：

```text
Kernel 已经很快。
```

体验仍然：

```text
慢。
```

---

# 290. Edge Prompt Engineering 与 Cloud 不同

Cloud：

```text
可接受大 Context。
```

Edge：

```text
Context 就是 Memory + Time。
```

---

# 291. 所以端侧 Prompt 要更节制

```text
System Prompt 精简

Tool Schema 精简

RAG Top-K 控制

History Summarize。
```

---

# 292. RAG Context Compression

不要：

```text
检索 10 个完整文档
全部塞进去。
```

可以：

```text
Chunk
Rerank
Top-K
Compress。
```

---

# 293. Agent Tool Schema

大量 JSON Schema：

```text
也会占 Prompt Token。
```

---

# 294. Edge Agent

工具：

```text
越少越好
描述越短越好。
```

---

# 295. 这就是端侧系统优化

不仅：

```text
NPU。
```

而是：

```text
Token Economy。
```

---

# 296. Token Economy

可以定义：

> 用尽可能少的 Token 完成同样任务。

---

# 297. 为什么 Token 是资源？

Input Token：

```text
增加 Prefill。
```

History Token：

```text
增加 KV。
```

Output Token：

```text
直接增加 Decode 时间。
```

---

# 298. 所以端侧 AI 的一个核心工程指标

```text
Task Completion / Token。
```

---

# 299. Error Debug：模型转换失败

按顺序看：

```text
Toolkit version

Official model support

Transformers version

Model config

Architecture

dtype

Quant dataset

target platform

Host RAM / VRAM。
```

---

# 300. 为什么 Host RAM 也很重要？

转换 2B / 4B：

```text
可能需要远高于最终模型大小的内存。
```

因为转换阶段：

```text
原 Weight
中间 Tensor
Quant buffer
Compiler data。
```

同时存在。

---

# 301. Host GPU 不是必须？

官方 example 支持：

```text
device='cpu'
```

或：

```text
'cuda'。
```

具体：

```text
取决模型和 Host 资源。
```

---

# 302. Windows 怎么办？

本路线建议：

```text
Windows Host
↓
Ubuntu VM / WSL / Linux machine
↓
RKLLM Toolkit。
```

---

# 303. 为什么建议 Linux Conversion Environment？

Vendor SDK：

```text
Linux 测试最充分
依赖链更一致。
```

---

# 304. 但实际环境版本必须跟官方 Package 对齐

尤其：

```text
Python
Transformers
Torch
CUDA
glibc。
```

---

# 305. 当前 README Python 支持

当前列出：

```text
Python 3.9
3.10
3.11
3.12。
```

---

# 306. 但某些模型有特殊要求

例如：

```text
RWKV
```

官方 README 有单独：

```text
Python / requirements
```

说明。

---

# 307. 所以不要一个环境装所有模型

更稳妥：

```text
venv / conda
per SDK version。
```

---

# 308. 推荐环境命名

```text
rkllm-1.3.0-py310
```

---

# 309. 把依赖 Freeze

```bash
pip freeze > requirements-lock.txt
```

---

# 310. 为什么？

半年后：

```text
Transformers 自动升级
```

可能：

```text
Model class 行为变化。
```

---

# 311. Error Debug：Board init failed

查：

```text
model target

runtime version

driver version

librkllmrt path

libomp

LD_LIBRARY_PATH

model integrity

RAM。
```

---

# 312. 当前官方 README 特别提示

某些平台：

```text
可能缺 libomp.so。
```

需要：

```text
从交叉编译工具链准备对应库。
```

---

# 313. Error Debug：能生成但乱码

查：

```text
Tokenizer

Chat Template

Special Token

UTF-8 streaming

Callback string concatenation。
```

---

# 314. Error Debug：回答质量很差

查：

```text
Quantization

Calibration data

Template

Thinking mode

Sampling

Model itself。
```

---

# 315. Error Debug：多轮突然忘记

查：

```text
keep_history

KV Cache

max_context

context truncation。
```

---

# 316. Error Debug：越来越慢

查：

```text
Context growing

KV cache

Long-context decode

Thermal

Background load。
```

---

# 317. Error Debug：内存越来越高

查：

```text
History

KV Cache

Output buffer

Logits / hidden state

Session cleanup

Prompt cache。
```

---

# 318. Error Debug：第二个用户收到第一个用户内容

这是严重：

```text
Session / KV Cache isolation
```

问题。

---

# 319. 每个 Session 必须明确 Context 生命周期

```text
Create
↓
Run
↓
Keep / Clear
↓
Destroy。
```

---

# 320. 多模态 Debug

拆成：

```text
Vision Encoder Output

Embedding Bridge

LLM Input

Prompt Special Tokens。
```

不要直接：

```text
只看最终描述错。
```

---

# 321. Vision Encoder 可以单独验证

```text
PyTorch vision
vs
RKNN vision output。
```

---

# 322. LLM 也可以单独验证

给固定：

```text
embedding
```

观察：

```text
generation。
```

---

# 323. 这就是分层 Debug 思维

```text
Vision
↓
Bridge
↓
LLM。
```

---

# 324. 本章推荐实验仓库

```text
rkllm-learning/
│
├── 01_official_quickstart/
│
├── 02_qwen35_2b_w4a16/
│   ├── convert.py
│   ├── data_quant.json
│   ├── model_info.md
│   └── benchmark.md
│
├── 03_qwen35_2b_w8a8/
│
├── 04_cpp_runtime/
│   ├── CMakeLists.txt
│   └── main.cpp
│
├── 05_prompt_cache/
│
├── 06_history/
│
├── 07_async_abort/
│
├── 08_server/
│
├── 09_agent/
│
├── 10_rag/
│
├── 11_lora/
│
├── 12_multimodal/
│
└── docs/
    ├── version_matrix.md
    ├── benchmark_matrix.md
    ├── prompt_regression.md
    └── architecture.md
```

---

# 325. `version_matrix.md`

记录：

```text
RKLLM Toolkit

RKLLM Runtime

RKNPU Driver

BSP

Kernel

Python

Torch

Transformers

Model revision。
```

---

# 326. `model_info.md`

记录：

```text
Model

Parameters

Architecture

Quant

Context

NPU Core

Artifact size

Calibration dataset。
```

---

# 327. `benchmark_matrix.md`

至少：

```text
TTFT

Prefill t/s

Decode t/s

Peak RAM

CPU%

NPU%

Temperature

Power。
```

---

# 328. `prompt_regression.md`

记录：

```text
固定问题
期望行为
W4A16
W8A8
Toolkit version。
```

---

# 329. 本章推荐源码阅读顺序

---

## 第一阶段：README

- [rknn-llm README](https://github.com/airockchip/rknn-llm/blob/main/README.md)

先理解：

```text
Software Stack

Platform

Supported Model

Quickstart

Benchmark。
```

---

## 第二阶段：CHANGELOG

- [CHANGELOG](https://github.com/airockchip/rknn-llm/blob/main/CHANGELOG.md)

原因：

```text
这个仓库版本演进很快。
```

---

## 第三阶段：Export Example

- [export_rkllm.py](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/export/export_rkllm.py)

重点：

```text
load_huggingface

build

quantized_dtype

quantized_algorithm

target_platform

num_npu_core

max_context。
```

---

## 第四阶段：API Demo

- [rkllm_api_demo README](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/Readme.md)

看：

```text
Conversion
Build
C++ Deploy
Run。
```

---

## 第五阶段：C++ Demo

- [llm_demo.cpp](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/deploy/src/llm_demo.cpp)

重点：

```text
RKLLMParam

callback

rkllm_init

RKLLMInput

RKLLMInferParam

rkllm_run

Prompt Cache

LoRA。
```

---

## 第六阶段：Header

- [rkllm.h](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)

这是：

```text
最重要的 Runtime Contract。
```

---

## 第七阶段：Server Demo

- [rkllm_server_demo](https://github.com/airockchip/rknn-llm/tree/main/examples/rkllm_server_demo)

理解：

```text
Runtime
↓
HTTP API
↓
Streaming。
```

---

## 第八阶段：Multimodal

- [multimodal_model_demo](https://github.com/airockchip/rknn-llm/tree/main/examples/multimodal_model_demo)

重点：

```text
Vision `.rknn`
+
LLM `.rkllm`。
```

---

## 第九阶段：Benchmark

- [benchmark.md](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md)

学习：

```text
Model
Quant
TTFT
Decode TPS
Memory
Platform。
```

---

# 330. 本章必做实验

- [ ] Clone `rknn-llm`
- [ ] 确认当前 SDK 版本
- [ ] 阅读 CHANGELOG
- [ ] 跑官方预转换 `.rkllm`
- [ ] 在 RK3576 跑 `llm_demo`
- [ ] 打印 RKLLM performance log
- [ ] 记录 TTFT
- [ ] 记录 Decode t/s
- [ ] 记录 Memory
- [ ] 自己转换一个小 Qwen
- [ ] 自己转换 Qwen3.5-2B W4A16
- [ ] 比较官方模型
- [ ] 转 W8A8
- [ ] 比较 W4A16 vs W8A8
- [ ] 测 W4A16_g128
- [ ] 建 Calibration Dataset
- [ ] 测不同 max_context
- [ ] 读 `rkllm.h`
- [ ] 写最小 C++ Runtime
- [ ] 实现 Streaming Callback
- [ ] 使用 `RKLLMPerfStat`
- [ ] 测 `keep_history`
- [ ] 测 `rkllm_clear_kv_cache`
- [ ] 测 Prompt Cache
- [ ] 测 Async
- [ ] 测 Abort
- [ ] 使用 Server Demo
- [ ] 从 PC 用 OpenAI-style client 访问 RK3576
- [ ] 做 Context Sweep
- [ ] 做 30min sustained benchmark
- [ ] 保存完整 version matrix

---

# 331. 进阶实验

- [ ] Token Input
- [ ] Embedding Input
- [ ] Tokenizer Callback
- [ ] Embedding Callback
- [ ] LoRA Adapter
- [ ] Tool Role
- [ ] Agent
- [ ] RAG
- [ ] Vision `.rknn` + LLM `.rkllm`
- [ ] Multimodal Input
- [ ] Thinking Mode
- [ ] Multiple EOS
- [ ] Per-request Sampling
- [ ] Per-request max_new_tokens

---

# 332. 常见误区

---

## 误区 1

```text
RKLLM = RKNN。
```

错误。

```text
RKNN:
General NN

RKLLM:
LLM-specific。
```

---

## 误区 2

```text
.rkllm = .rknn。
```

错误。

---

## 误区 3

```text
Qwen3.5-2B
先导 ONNX 再转普通 .rknn。
```

不应作为当前官方 LLM 主路线。

正确：

```text
HF
↓
RKLLM Toolkit
↓
.rkllm。
```

---

## 误区 4

```text
RKLLM-Toolkit
=
板端 Runtime。
```

错误。

它主要：

```text
Host conversion / compiler。
```

---

## 误区 5

```text
librkllmrt.so
=
模型。
```

错误。

它是：

```text
Runtime。
```

---

## 误区 6

```text
W4A16
=
Q4_K_M。
```

错误。

---

## 误区 7

```text
Weight 4-bit
=
全部计算都是 INT4。
```

错误。

W4A16 本身已经说明：

```text
Weight / Activation path
不是同一精度。
```

---

## 误区 8

```text
4-bit 一定比 8-bit 精度差很多。
```

不一定。

要：

```text
实际 Benchmark。
```

---

## 误区 9

```text
4-bit 一定比 8-bit 慢。
```

RK3576 官方 Qwen3.5 benchmark：

```text
反而 W4A16 Decode 更快。
```

---

## 误区 10

```text
max_context 越大越好。
```

错误。

它消耗：

```text
Memory + Attention Time。
```

---

## 误区 11

```text
Prompt Cache = KV Cache。
```

不完全。

---

## 误区 12

```text
keep_history
只是保存字符串。
```

不完整。

LLM Runtime：

```text
需要管理 Context / Cache。
```

---

## 误区 13

```text
NPU 负责 LLM 的一切。
```

错误。

还有：

```text
CPU
DDR
Runtime
Tokenizer
Application。
```

---

## 误区 14

```text
NPU TOPS
可以直接推算 Token/s。
```

错误。

---

## 误区 15

```text
Tokens/s
是唯一用户体验指标。
```

错误。

还有：

```text
TTFT
Output length
Total latency。
```

---

## 误区 16

```text
Qwen3.5 旧 Issue 说不支持
所以现在也不能部署。
```

错误。

当前 v1.3.0：

```text
已经官方支持。
```

---

## 误区 17

```text
官方 benchmark
=
我的板子一定同样速度。
```

错误。

---

## 误区 18

```text
VLM 全部都转 `.rkllm`。
```

不一定。

官方多模态路线：

```text
Vision → `.rknn`

LLM → `.rkllm`。
```

---

## 误区 19

```text
RKLLM Server = vLLM。
```

错误。

API 层可以类似，

系统定位：

```text
完全不同。
```

---

## 误区 20

```text
端侧 Agent
只需要把云 Agent 代码搬过来。
```

不够。

必须考虑：

```text
Token budget
Context
Latency
Memory
Power。
```

---

# 333. 本章知识树

```text
RKLLM
│
├── Host Toolchain
│   └── RKLLM-Toolkit
│       ├── load_huggingface
│       ├── load_gguf
│       ├── build
│       ├── quantization
│       ├── target compilation
│       ├── context config
│       └── export_rkllm
│
├── Artifact
│   └── .rkllm
│
├── Quantization
│   ├── W8A8
│   ├── W8A8_gx
│   ├── W4A16
│   ├── W4A16_g128 / gx
│   ├── normal
│   ├── grq
│   └── mixed quant
│
├── Runtime
│   ├── RKLLMParam
│   ├── rkllm_init
│   ├── RKLLMInput
│   ├── RKLLMInferParam
│   ├── rkllm_run
│   ├── rkllm_run_async
│   ├── rkllm_abort
│   └── rkllm_destroy
│
├── Input
│   ├── Prompt
│   ├── Token
│   ├── Embedding
│   └── Multimodal
│
├── Generation
│   ├── Prefill
│   ├── Decode
│   ├── KV Cache
│   ├── Sampling
│   ├── EOS
│   ├── Thinking
│   └── Streaming
│
├── Sampling
│   ├── top_k
│   ├── top_p
│   ├── temperature
│   ├── repeat_penalty
│   ├── frequency_penalty
│   ├── presence_penalty
│   └── mirostat
│
├── Context
│   ├── max_context_len
│   ├── keep_history
│   ├── clear_kv_cache
│   └── Prompt Cache
│
├── Extension
│   ├── LoRA
│   ├── Tokenizer Callback
│   ├── Embedding Callback
│   ├── Cross Attention
│   ├── Tool Role
│   └── OpenAI-compatible Server
│
├── Performance
│   ├── TTFT
│   ├── Prefill TPS
│   ├── Decode TPS
│   ├── Memory
│   ├── CPU utilization
│   ├── NPU utilization
│   ├── Thermal
│   └── Power
│
├── VLM
│   ├── Vision Encoder
│   │   └── .rknn
│   ├── Embedding
│   └── LLM
│       └── .rkllm
│
└── Hardware
    ├── CPU
    ├── DDR
    ├── RKNPU Driver
    └── RKNPU
```

---

# 334. 本章最重要的 15 个思维模型

## 1

```text
RKLLM
=
LLM-specific Vendor Compiler
+
Runtime。
```

---

## 2

```text
.rkllm
=
RKNPU-targeted LLM Deployment Artifact。
```

---

## 3

```text
LLM
不是一次 Forward

而是：
Prefill
+
Stateful Decode Loop。
```

---

## 4

```text
KV Cache
=
Autoregressive Runtime 的核心状态。
```

---

## 5

```text
Prompt Cache
=
减少重复 Prefix Prefill。
```

---

## 6

```text
W4A16
不仅省 RAM

还可能减少 Decode Weight Traffic。
```

---

## 7

```text
Decode Token/s
受：
Weight Size
DDR
KV
Architecture
Runtime
共同影响。
```

---

## 8

```text
TTFT
≠
Decode Token/s。
```

---

## 9

```text
Long Context
=
更多 Memory
+
更多 Attention IO。
```

---

## 10

```text
Edge LLM
=
NPU + CPU + DDR + Runtime。
```

---

## 11

```text
端侧性能优化
不止是 Kernel。

还包括：
Prompt Token 数。
```

---

## 12

```text
VLM
=
.rknn Vision
+
.rkllm LLM
```

是官方典型路线。

---

## 13

```text
RKLLM
和
llama.cpp

是两套完全不同生态。
```

---

## 14

```text
官方支持状态
必须看当前 SDK Version。
```

---

## 15

```text
端侧 LLM 最终目标
不是跑起来，

而是：
能力 / 延迟 / RAM / 功耗
达到平衡。
```

---

# 335. 本章完成标准

## 生态

- [ ] 能解释 RKNN-LLM
- [ ] 能解释 RKLLM
- [ ] 能解释 RKLLM-Toolkit
- [ ] 能解释 `.rkllm`
- [ ] 能解释 RKLLM Runtime
- [ ] 能解释 `.rkllm` vs `.rknn`
- [ ] 能解释 RKLLM vs llama.cpp

## Compiler

- [ ] 能解释 `load_huggingface`
- [ ] 能解释 Host device / dtype
- [ ] 能解释 `build`
- [ ] 能解释 target_platform
- [ ] 能解释 num_npu_core
- [ ] 能解释 max_context
- [ ] 能解释 export_rkllm
- [ ] 能解释为什么 Build 是 Vendor Compiler

## Quantization

- [ ] 能解释 W8A8
- [ ] 能解释 W4A16
- [ ] 能解释 group quant
- [ ] 能解释 normal / grq 的位置
- [ ] 能解释 calibration data
- [ ] 能解释 W4A16 为什么可能 Decode 更快
- [ ] 能解释为什么 W4A16 ≠ Q4_K_M
- [ ] 能完成 W4A16 vs W8A8 benchmark

## Runtime

- [ ] 能使用 `rkllm_createDefaultParam`
- [ ] 能解释 RKLLMParam
- [ ] 能使用 rkllm_init
- [ ] 能构造 RKLLMInput
- [ ] 能构造 RKLLMInferParam
- [ ] 能使用 rkllm_run
- [ ] 能实现 Callback
- [ ] 能使用 rkllm_destroy
- [ ] 知道 async / abort / is_running

## Input

- [ ] 能解释 Prompt Input
- [ ] 能解释 Token Input
- [ ] 能解释 Embed Input
- [ ] 能解释 Multimodal Input
- [ ] 能解释 role
- [ ] 能解释 thinking mode

## LLM Inference

- [ ] 能解释 Prefill
- [ ] 能解释 Decode
- [ ] 能解释 KV Cache
- [ ] 能解释 Sampling
- [ ] 能解释 EOS
- [ ] 能解释 Streaming
- [ ] 能解释 TTFT
- [ ] 能解释 Prefill TPS
- [ ] 能解释 Decode TPS

## Context

- [ ] 能解释 max_context
- [ ] 能估算 KV Cache 增长
- [ ] 能解释 keep_history
- [ ] 能使用 clear KV cache
- [ ] 能解释 Prompt Cache
- [ ] 能完成 Long-context benchmark

## Sampling

- [ ] 能解释 top_k
- [ ] 能解释 top_p
- [ ] 能解释 temperature
- [ ] 能解释 repeat penalty
- [ ] 能解释为什么 benchmark 常用 greedy
- [ ] 能使用 per-request sampling param

## LoRA / Extension

- [ ] 能解释 Runtime LoRA
- [ ] 知道 tokenizer callback
- [ ] 知道 embedding callback
- [ ] 能解释 cross attention
- [ ] 能解释 tool role

## Server

- [ ] 能运行 rkllm_server_demo
- [ ] 能从 PC 调 RK3576 API
- [ ] 能实现 Streaming
- [ ] 能实现 Stop
- [ ] 能管理 Session
- [ ] 能避免 KV 串用户

## Qwen3.5-2B

- [ ] 知道当前 v1.3.0 已官方支持
- [ ] 能跑官方预转换模型
- [ ] 能自己转换 W4A16
- [ ] 能自己转换 W8A8
- [ ] 能比较官方 benchmark
- [ ] 能测 128/512/1024/2048/4096 context
- [ ] 能建立固定 Prompt Regression Set
- [ ] 能记录完整环境版本

## System

- [ ] 能解释 CPU/NPU/DDR 协作
- [ ] 能锁频 Benchmark
- [ ] 能看 CPU utilization
- [ ] 能看 NPU utilization
- [ ] 能测 sustained performance
- [ ] 能记录 thermal
- [ ] 能记录 cold/warm startup

---

# 336. 暂时不要求深入

暂时不要求：

- [ ] 逆向 `.rkllm` 文件格式
- [ ] 手写 RKNPU LLM Kernel
- [ ] 修改 RKLLM Runtime
- [ ] 自己实现 Qwen3.5 Backend
- [ ] 精通所有 RKLLM quant algorithm
- [ ] 自己实现 tokenizer callback production framework
- [ ] 精通 multimodal video path
- [ ] 精通 cross-attention cache layout
- [ ] 多用户高并发 Serving
- [ ] 实现 vLLM 级 Scheduler
- [ ] 自己写 RKNPU Driver
- [ ] 追求云 GPU 级吞吐

先做到：

```text
会转换
会部署
会测性能
会理解 Cache
会管理 Context
会做 Streaming
会解释为什么 W4A16 快
会把 LLM 稳定跑在 RK3576。
```

---

# 337. 核心参考链接

## 主仓库

- [RKNN-LLM Repository](https://github.com/airockchip/rknn-llm)
- [README](https://github.com/airockchip/rknn-llm/blob/main/README.md)
- [CHANGELOG](https://github.com/airockchip/rknn-llm/blob/main/CHANGELOG.md)
- [Benchmark](https://github.com/airockchip/rknn-llm/blob/main/benchmark.md)

---

## Conversion

- [RKLLM Export Example](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/export/export_rkllm.py)
- [RKLLM API Demo README](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/Readme.md)

重点：

```text
load_huggingface
build
export_rkllm
W8A8
W4A16
quant algorithm
max_context。
```

---

## C/C++ Runtime

- [llm_demo.cpp](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/deploy/src/llm_demo.cpp)
- [rkllm.h](https://github.com/airockchip/rknn-llm/blob/main/rkllm-runtime/Linux/librkllm_api/include/rkllm.h)
- [CMakeLists.txt](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_api_demo/deploy/CMakeLists.txt)

---

## Server

- [rkllm_server_demo](https://github.com/airockchip/rknn-llm/tree/main/examples/rkllm_server_demo)
- [Flask Server](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_server_demo/rkllm_server/flask_server.py)
- [Gradio Server](https://github.com/airockchip/rknn-llm/blob/main/examples/rkllm_server_demo/rkllm_server/gradio_server.py)

---

## Multimodal

- [Multimodal Demo](https://github.com/airockchip/rknn-llm/tree/main/examples/multimodal_model_demo)
- [Multimodal README](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/README.md)
- [Vision RKNN Export](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_vision_rknn.py)
- [LLM Export](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_rkllm.py)

---

# 338. 与前面章节的连接

## 03_Transformer与LLM

```text
Attention
KV Cache
Tokenizer
Autoregressive Generation
```

在本章：

```text
全部变成 Runtime API。
```

---

## 06_LLM推理原理

```text
Prefill
Decode
TTFT
Token/s
KV Cache
Memory-bound。
```

在本章：

```text
变成 RK3576 实际 Benchmark。
```

---

## 07_llama.cpp

```text
GGUF
↓
llama.cpp

vs

.rkllm
↓
RKLLM Runtime。
```

---

## 08_vLLM与MLC-LLM

```text
vLLM:
Serving-centric

MLC:
Compiler-centric

RKLLM:
Vendor Hardware Deployment-centric。
```

---

## 09_AI_Compiler与TVM

```text
Model
↓
Compiler
↓
Target Artifact
↓
Runtime
```

在这里：

```text
HF
↓
RKLLM-Toolkit
↓
.rkllm
↓
RKLLM Runtime。
```

---

## 10_GPU_Kernel与FlashAttention

```text
Decode Memory-Bound
```

在这里实际表现为：

```text
W4A16
可能显著提高 RK3576 Decode TPS。
```

---

## 11_RKNN与Rockchip_NPU

```text
General NPU:
.rknn

LLM NPU:
.rkllm。
```

---

# 339. 最终总图

```text
                          Hugging Face
                               │
                               ▼
                         Qwen3.5-2B
                               │
                               ▼
                         RKLLM-Toolkit
                               │
             ┌─────────────────┼─────────────────┐
             ▼                 ▼                 ▼
        Architecture       Quantization       Context
         Recognition       W4A16/W8A8      max_context
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ▼
                       Target = RK3576
                               │
                               ▼
                         NPU Compilation
                               │
                               ▼
                            .rkllm
                               │
                               ▼
                         RKLLM Runtime
                               │
             ┌─────────────────┼──────────────────┐
             ▼                 ▼                  ▼
         Tokenizer          KV Cache           Sampling
             │                 │                  │
             ▼                 ▼                  ▼
          Prefill           Decode            Top-k/Top-p
             │                 │                  │
             └─────────────────┼──────────────────┘
                               ▼
                       Streaming Callback
                               │
                               ▼
                           Application
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
               Chat           RAG          Agent
                               │
                               ▼
                            RK3576
```

---

# 340. 一句话总结

这章真正需要理解的不是：

```text
“Qwen3.5 怎么转成 .rkllm”
```

这一条命令。

而是：

```text
RKLLM-Toolkit
=
面向 RKNPU 的 LLM Vendor Compiler

.rkllm
=
面向目标 SoC 的 LLM Deployment Artifact

RKLLM Runtime
=
管理 Tokenizer、Prefill、KV Cache、Decode、
Sampling、Prompt Cache、History 与 Streaming
的 LLM Runtime

RKNPU
=
真正执行主要 Tensor Workload 的硬件。
```

完整链路：

```text
Hugging Face
↓
RKLLM-Toolkit
↓
Architecture / Quantization / Compile
↓
.rkllm
↓
RKLLM Runtime
↓
Prefill
↓
KV Cache
↓
Decode
↓
Sampling
↓
Streaming
↓
Application
↓
RK3576
```

对于当前重点的 Qwen3.5-2B：

```text
Qwen3.5-2B
↓
W4A16
↓
RK3576
```

已经有官方 v1.3.0 支持与官方 benchmark，因此它非常适合作为这整套学习路线最终的端侧 LLM 主案例。

下一章：

> **13_RK3576端侧AI实践.md**

就不再继续扩展理论，而会把整套路线真正落到板子上：

```text
Windows
↓
Ubuntu VM
↓
RKLLM Toolkit
↓
Qwen3.5-2B W4A16
↓
RK3576 Ubuntu
↓
RKLLM Runtime
↓
Benchmark
↓
C++ Service
↓
RAG / Agent
```

最终完成一个：

> **可复现、可测量、可解释、可写进简历的 RK3576 端侧 LLM 工程。**
