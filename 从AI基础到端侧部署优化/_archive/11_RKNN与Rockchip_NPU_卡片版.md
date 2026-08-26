# 11_RKNN与Rockchip_NPU

> **所属路线**：AI 学习路线 · 第三部分  
> **定位**：把前面 AI Compiler、GPU Kernel、量化与 Runtime 的知识正式迁移到 Rockchip NPU，建立从 **PyTorch / ONNX → RKNN-Toolkit2 → Graph Optimization / Quantization → `.rknn` → RKNN Runtime → RKNPU Driver → NPU** 的完整认知  
> **核心仓库**：[`airockchip/rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2) + [`airockchip/rknn_model_zoo`](https://github.com/airockchip/rknn_model_zoo)  
> **相关 LLM 仓库**：[`airockchip/rknn-llm`](https://github.com/airockchip/rknn-llm)  
> **目标平台重点**：RK3576  
> **当前参考版本**：RKNN-Toolkit2 v2.3.2（以官方仓库当前 README / CHANGELOG 为准）  
> **学习边界**：本章重点是通用 RKNN / RKNPU 模型部署与 NPU Compiler 思维；LLM/VLM 的 `RKLLM / RKNN-LLM` 将在下一章单独深入。  
> **资料检查日期**：2026-08-25

---

# 0. 本章最重要的问题

前面已经建立：

```text
PyTorch
↓
ONNX / Export Graph
↓
AI Compiler
↓
Graph IR
↓
Operator Optimization
↓
Quantization
↓
Lowering
↓
Kernel / Hardware
```

现在来到 Rockchip NPU：

```text
PyTorch / ONNX
↓
RKNN-Toolkit2
↓
???
↓
.rknn
↓
RKNN Runtime
↓
RKNPU
```

这里最关键的问题是：

> **这个 `???` 到底是什么？**

本章需要把它拆开为：

```text
Model Import
↓
Graph Analysis
↓
Operator Conversion
↓
Graph Optimization
↓
Layout / Dtype Transformation
↓
Quantization
↓
Target-aware Compilation
↓
NPU Operator Mapping
↓
Memory Planning / Runtime Metadata
↓
RKNN Deployment Artifact
```

因此最终不应该只会：

```python
rknn.build()
```

而应该知道：

> `rknn.build()` 背后本质上是在调用一套面向 RKNPU 的 Vendor AI Compiler。

---

# 1. 先把最容易混淆的 7 个概念彻底分开

```text
RKNN
.rknn
RKNN-Toolkit2
RKNN-Toolkit-Lite2
RKNN Runtime
RKNPU Driver
RKLLM / RKNN-LLM
```

这是整个 Rockchip AI 生态的基础。

---

# 2. RKNN 是什么？

“RKNN”在不同上下文里可能被泛指为：

```text
Rockchip Neural Network
```

生态 / SDK / Model Format。

为了避免混乱，本笔记尽量这样使用：

```text
RKNN Software Stack
=
整套 Rockchip NPU AI 软件栈
```

而具体文件：

```text
xxx.rknn
```

称：

```text
RKNN Model / RKNN Deployment Artifact。
```

---

# 3. `.rknn` 是什么？

最重要定义：

> **`.rknn` 是经过 Rockchip RKNN-Toolkit2 面向指定 Rockchip NPU 平台转换、优化、量化/精度处理并编译后的部署模型文件。**

它不是：

```text
PyTorch checkpoint
```

也不是：

```text
普通 ONNX graph
```

更不是：

```text
Python Runtime。
```

---

# 4. `.rknn` 为什么和 `.onnx` 不一样？

ONNX：

```text
通用交换格式
```

强调：

```text
Graph
Operator
Weight
跨 Framework / Runtime。
```

`.rknn`：

```text
Target-specific deployment artifact
```

强调：

```text
面向 Rockchip RKNPU 执行。
```

---

# 5. 不要过度猜测 `.rknn` 内部格式

它是：

```text
Vendor-defined binary artifact。
```

公开 API 允许：

```text
Load
Query
Execute。
```

但不要未经官方文档确认就断言其中一定逐字节包含：

```text
某种特定 command stream
某种固定 SRAM map
某种固定 ISA encoding。
```

更稳妥的理解是：

> 它包含 Runtime 在目标 RKNPU 上加载和执行模型所需的 Vendor-specific 编译结果、模型数据与元信息。

---

# 6. RKNN-Toolkit2 是什么？

官方定义：

> 用于在 PC / Host 侧完成模型转换、推理、性能评估和面向 Rockchip NPU 部署的软件开发工具包。

最典型工作：

```text
ONNX / PyTorch / TF ...
↓
RKNN-Toolkit2
↓
.rknn
```

---

# 7. RKNN-Toolkit2 运行在哪里？

主要是：

```text
开发 PC / Host。
```

典型环境：

```text
Ubuntu / Linux
Python
```

现在部分版本也提供：

```text
ARM64 Host package。
```

但最核心的开发模式仍然是：

```text
Host Convert
↓
Target Deploy。
```

---

# 8. RKNN-Toolkit-Lite2 是什么？

官方定位：

> 为 Rockchip NPU 平台提供 Python 推理接口。

即：

```text
板端 Python
```

通常：

```python
from rknnlite.api import RKNNLite
```

用于：

```text
load .rknn
init_runtime
inference
release。
```

---

# 9. RKNN Runtime 是什么？

官方定义：

> 为 Rockchip NPU 平台提供 C/C++ 编程接口的 Runtime。

典型库：

```text
librknnrt.so
```

应用：

```text
C / C++
↓
RKNN API
↓
RKNN Runtime
↓
RKNPU Driver
↓
NPU。
```

---

# 10. RKNPU Driver 是什么？

这是：

```text
Kernel Driver。
```

负责：

```text
Userspace Runtime
↓
Kernel
↓
NPU Hardware
```

的底层交互。

---

# 11. RKLLM / RKNN-LLM 是什么？

这是：

```text
大语言模型 / VLM
专用 Toolchain / Runtime。
```

官方 `rknn-toolkit2` README 明确指出：

> 如果部署 LLM，应使用单独的 RKNN-LLM SDK。

所以：

```text
普通 CNN / Detection / Vision Model
→ RKNN-Toolkit2

LLM / VLM
→ RKNN-LLM / RKLLM route
```

---

# 12. 一张总关系图

```text
                    Rockchip AI Software Stack
                              │
            ┌─────────────────┴──────────────────┐
            ▼                                    ▼
      General AI Model                       LLM / VLM
            │                                    │
            ▼                                    ▼
     RKNN-Toolkit2                         RKNN-LLM / RKLLM
            │                                    │
            ▼                                    ▼
          .rknn                         LLM-specific artifact
            │                                    │
      ┌─────┴─────┐                              │
      ▼           ▼                              ▼
 Lite2 Python   RKNN Runtime C/C++          RKLLM Runtime
      │           │                              │
      └─────┬─────┘                              │
            ▼                                    ▼
        RKNPU Driver                         RKNPU Driver
            │                                    │
            └─────────────┬──────────────────────┘
                          ▼
                         NPU
```

---

# 13. 官方核心仓库

## RKNN-Toolkit2

- [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2)

建议重点看：

```text
README.md
CHANGELOG.md
rknn-toolkit2/examples
rknn-toolkit-lite2/examples
rknpu2/runtime
rknpu2/examples
doc
```

---

# 14. RKNN Model Zoo

- [airockchip/rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo)

这是非常重要的工程仓库。

它提供：

```text
Model Export
↓
RKNN Conversion
↓
Python Inference
↓
C/C++ Inference
↓
Linux / Android Build
↓
Benchmark。
```

比只看 API 文档更适合学习：

> “真实项目怎样落地”。

---

# 15. 当前支持平台

当前官方 RKNN-Toolkit2 README 支持列表包含：

```text
RK3588 Series
RK3576 Series
RK3566 / RK3568
RK3562
RV1103 / RV1106
RV1103B / RV1106B
RV1126B
RK2118
```

具体兼容性：

```text
始终以当前 Toolkit Version
+
Runtime Version
+
Driver Version
```

为准。

---

# 16. RK3576 在哪里？

本学习路线重点：

```text
RK3576
```

因为它同时具备：

```text
ARM CPU
GPU
RKNPU
Linux
```

很适合做：

```text
CPU / GPU / NPU
异构部署对比。
```

---

# 17. 最值得建立的三条 RK3576 路线

```text
路线 1
llama.cpp
↓
ARM CPU / possible GPU backend
```

```text
路线 2
MLC / Vulkan / OpenCL
↓
GPU
```

```text
路线 3
RKNN / RKLLM
↓
RKNPU
```

---

# 18. 不要把三条线混起来

```text
GGUF
≠
.rknn
```

```text
llama.cpp
≠
RKNN Runtime
```

```text
Vulkan GPU
≠
RKNPU
```

```text
RKNN-Toolkit2
≠
RKNN-LLM
```

---

# 19. RKNN 完整 Workflow

最推荐记住：

```text
PyTorch
↓
Export ONNX
↓
ONNX Runtime Validation
↓
RKNN()
↓
rknn.config()
↓
rknn.load_onnx()
↓
rknn.build()
↓
Accuracy Validation
↓
rknn.export_rknn()
↓
.rknn
↓
Board Runtime
↓
RKNNLite / C API
↓
RKNPU
```

---

# 20. 一个最小 Toolkit2 转换程序

官方 example 的典型结构：

```python
from rknn.api import RKNN

rknn = RKNN(verbose=True)

rknn.config(
    target_platform='rk3576'
)

ret = rknn.load_onnx(
    model='model.onnx'
)

ret = rknn.build(
    do_quantization=False
)

ret = rknn.export_rknn(
    'model.rknn'
)

rknn.release()
```

---

# 21. 这五个 API 是第一阶段必须掌握的

```text
RKNN()

config()

load_onnx()

build()

export_rknn()
```

---

# 22. `RKNN(verbose=True)`

建议开发阶段：

```python
RKNN(verbose=True)
```

因为转换日志非常重要。

要学会看：

```text
Graph Optimize
Operator
Quantization
Warning
Fallback
Dtype
Tensor Shape
Performance。
```

---

# 23. `rknn.config()`

它不是简单：

```text
配置日志。
```

它会影响：

```text
Target
Input preprocessing
Quantization
Optimization
Dynamic shape
Performance debug
Memory evaluation
```

等编译 / 运行行为。

---

# 24. 最重要参数之一：`target_platform`

对于 RK3576：

```python
rknn.config(
    target_platform='rk3576'
)
```

为什么必须指定？

因为：

```text
不同 Rockchip NPU
```

支持：

```text
Operator
Dtype
Layout
Optimization
Quantization
```

能力不同。

---

# 25. Target-aware Compilation 再次出现

上一章 TVM：

```text
Target = cuda / llvm / vulkan。
```

这里：

```text
target_platform = rk3576。
```

本质完全相同：

> Compiler 必须知道最终硬件。

---

# 26. `mean_values` / `std_values`

典型：

```python
rknn.config(
    mean_values=[[0, 0, 0]],
    std_values=[[255, 255, 255]],
    target_platform='rk3576'
)
```

---

# 27. 这意味着什么？

将：

```text
Input Preprocess
```

部分融合 / 配置进：

```text
RKNN execution path。
```

例如概念：

```text
x'
=
(x - mean) / std。
```

---

# 28. 为什么要重视预处理？

最常见部署错误之一：

```text
PyTorch:
RGB
0~1
NCHW

RKNN:
BGR
0~255
NHWC
```

模型本身完全正确，

结果仍然：

```text
完全错误。
```

---

# 29. 因此转换前必须建立 Preprocess Contract

至少写清：

```text
Color:
RGB / BGR

Shape:
NCHW / NHWC

Dtype:
float32 / uint8

Range:
0~1 / 0~255

Mean

Std

Resize

Letterbox。
```

---

# 30. `load_onnx()`

推荐部署路径：

```text
PyTorch
↓
ONNX
↓
RKNN。
```

不是因为 RKNN 只支持 ONNX，

而是 ONNX：

```text
便于检查 Graph
便于 ONNX Runtime 验证
便于定位 Operator 问题。
```

---

# 31. PyTorch → ONNX 的价值

你可以先验证：

```text
PyTorch Output
≈
ONNX Runtime Output。
```

如果已经不同：

```text
问题不是 RKNN。
```

---

# 32. 三阶段正确性验证

一定建立：

```text
Stage A:
PyTorch

Stage B:
ONNX Runtime

Stage C:
RKNN
```

比较：

```text
A vs B
B vs C。
```

---

# 33. 不要直接 PyTorch → RKNN 后只看最终结果

否则错误时：

```text
不知道问题来自：
Export？
ONNX？
RKNN Convert？
Quant？
Preprocess？
Postprocess？
```

---

# 34. `rknn.build()`

这是整套 Toolchain 最核心 API。

表面：

```python
rknn.build(...)
```

内部可以抽象理解成：

```text
Imported Graph
↓
Validate
↓
Canonicalize
↓
Graph Optimization
↓
Operator Fusion
↓
Layout / Dtype Transformation
↓
Quantization
↓
Target Mapping
↓
NPU Compilation
↓
Runtime Artifact Construction。
```

---

# 35. `rknn.build()` ≈ Vendor Compiler Driver

这和：

```text
clang source.c
```

很像。

一个命令：

```text
背后调用几十个 Compiler Pass。
```

---

# 36. 为什么 build 会失败？

常见原因：

```text
Unsupported Op
Unsupported Shape
Unsupported Dtype
Dynamic Shape Constraint
Invalid Quantization
Layout Constraint
Model Too Large
Tool / Driver Compatibility
Graph Pattern Unsupported。
```

---

# 37. Unsupported Operator

这是 NPU 部署最常见问题之一。

例如某模型包含：

```text
Rare Scatter
Complex Gather
Custom Attention Pattern
Dynamic Control Flow。
```

可能：

```text
无法直接映射到 NPU。
```

---

# 38. 为什么 CPU 可以但 NPU 不行？

CPU：

```text
通用 ISA。
```

NPU：

```text
专用 Tensor Accelerator。
```

支持：

```text
有限 Operator / Pattern / Dtype / Shape。
```

---

# 39. NPU 的强项

通常是：

```text
Conv
MatMul
Activation
Pooling
Normalization
常见 Tensor Operator
```

具体：

```text
以当前 RKNN Operator Support 文档为准。
```

---

# 40. 为什么不要靠“感觉”判断支持？

同一个：

```text
MatMul
```

可能：

```text
某 Shape 支持
某 Dtype 支持
某 Broadcast 不支持。
```

所以：

```text
Operator Name 相同
≠
Hardware Mapping 一定相同。
```

---

# 41. Graph Optimization

当前 RKNN Toolkit CHANGELOG 会持续出现：

```text
LayerNorm optimization
Transpose optimization
MatMul optimization
Operator fusion
Transformer optimization
Graph optimization。
```

说明：

> RKNN-Toolkit2 本质上一直在迭代 Compiler Pass。

---

# 42. 为什么新 Toolkit 可能同一个模型更快？

模型：

```text
完全没变。
```

但 Compiler：

```text
Fusion 更好
Operator Mapping 更好
Layout 更少
Tiling 更合理
NPU Implementation 更新。
```

于是：

```text
性能提升。
```

---

# 43. 这就是 Compiler 版本的重要性

不要认为：

```text
SDK 版本只影响 API。
```

它还可能改变：

```text
生成的 .rknn。
```

---

# 44. Toolkit / Runtime / Driver Version 三角关系

部署必须记录：

```text
RKNN-Toolkit2 Version

RKNN Runtime Version

RKNPU Driver Version。
```

---

# 45. 为什么版本匹配很重要？

Host Compiler：

```text
生成 Artifact。
```

Board Runtime：

```text
解析 / 执行 Artifact。
```

Kernel Driver：

```text
和 Hardware 交互。
```

任意一层：

```text
差异过大
```

都可能：

```text
Load Fail
Runtime Error
Unsupported Feature。
```

---

# 46. 推荐每个项目都记录

```text
Board:
RK3576

OS:
Ubuntu ...

Kernel:
...

RKNPU Driver:
...

RKNN Runtime:
...

Toolkit2:
...

Python:
...

Model:
...
```

---

# 47. `export_rknn()`

编译成功后：

```python
rknn.export_rknn(
    'model.rknn'
)
```

---

# 48. Export 之后的 `.rknn`

这时：

```text
已经不需要 PyTorch。
```

板端只需要：

```text
Runtime
+
Model
+
Application。
```

---

# 49. 这就是 Deployment Runtime 思维

```text
Training Framework
```

与：

```text
Inference Runtime
```

彻底分离。

---

# 50. PC Simulator / Connected Board Debug

RKNN-Toolkit2 不只负责转换。

它还可以：

```text
在 Host 调用 Runtime
```

进行：

```text
Simulator-style inference
```

或者：

```text
连接真实 Board。
```

---

# 51. `rknn.init_runtime()`

开发 PC 上：

```text
可用于初始化测试 Runtime。
```

如果指定：

```text
target / device_id
```

可进行：

```text
连板运行。
```

具体参数：

```text
以当前 Toolkit API 文档和 example 为准。
```

---

# 52. `rknn_server`

PC 连板调试时：

```text
Board 上运行后台代理。
```

作用：

```text
PC RKNN-Toolkit2
↓ USB / transport
rknn_server
↓
Board RKNN Runtime
↓
NPU。
```

---

# 53. `rknn_server` 不是产品运行时必需架构

它主要服务：

```text
开发 / 调试 / 连板。
```

最终产品通常：

```text
Application
↓
librknnrt.so
↓
NPU。
```

---

# 54. 连板问题常见原因

```text
rknn_server 未启动

rknn_server 版本不匹配

librknnrt.so 版本不匹配

USB / ADB / transport

权限。
```

---

# 55. 第二部分：Quantization

NPU 部署中：

```text
Quantization
```

比 CPU/GPU 通用部署更重要。

---

# 56. 为什么 NPU 喜欢 INT8？

因为专用硬件通常对：

```text
INT8 MAC
```

具有：

```text
高吞吐
低功耗
低 Memory。
```

---

# 57. Quantization 的两个对象

```text
Weight
```

和：

```text
Activation。
```

---

# 58. 常见 PTQ

```text
FP32 / FP16 Model
↓
Calibration Dataset
↓
Estimate Activation Range
↓
Generate Quantization Params
↓
INT8 Model。
```

---

# 59. `do_quantization=True`

典型：

```python
rknn.build(
    do_quantization=True,
    dataset='./dataset.txt'
)
```

---

# 60. `dataset.txt`

通常列出：

```text
Calibration Input Samples。
```

它不是：

```text
训练数据标签文件。
```

核心作用：

> 让 Compiler 观察代表性输入下 Tensor 的数值分布。

---

# 61. 为什么 Calibration Data 很重要？

如果数据不代表真实分布：

```text
Activation Range
```

估计不好。

导致：

```text
Clipping
Scale 不合理
精度下降。
```

---

# 62. Calibration Dataset 应满足

```text
真实场景
覆盖主要输入分布
数量适中
预处理一致。
```

---

# 63. 不建议：

```text
随便拿 3 张图。
```

也不需要：

```text
整个训练集。
```

关键：

```text
Representative。
```

---

# 64. Affine Quantization

RKNN Runtime API 中可以看到：

```text
RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
```

Tensor Attr 还包含：

```text
scale
zero point。
```

---

# 65. 经典公式

```text
q
=
round(x / scale)
+
zero_point
```

反量化：

```text
x
≈
(q - zero_point) × scale。
```

---

# 66. 为什么有 Zero Point？

让：

```text
非对称数值范围
```

更好映射到：

```text
UINT8 / INT8。
```

---

# 67. Per-tensor vs Per-channel

Weight：

```text
整个 Tensor 一个 Scale
```

或：

```text
每个 Channel 一个 Scale。
```

---

# 68. Per-channel 通常

```text
精度更好
```

但：

```text
实现与 Metadata 更复杂。
```

---

# 69. Quantization Algorithm

RKNN example 中提供：

```text
MMSE
```

等量化算法实验。

目标：

```text
更合理选择 Scale
减少量化误差。
```

---

# 70. Hybrid Quantization

官方 example：

```text
hybrid_quant
```

流程：

```text
Step 1
↓
分析量化敏感层
↓
修改 quantization config
↓
Step 2
↓
重新 build。
```

---

# 71. 为什么 Hybrid Quant？

不是每一层：

```text
都适合 INT8。
```

有些：

```text
非常敏感。
```

可以：

```text
保留更高精度。
```

---

# 72. 混合精度本质

```text
Performance / Memory

vs

Accuracy。
```

---

# 73. 自动混合精度

当前 RKNN-Toolkit2 v2.3.2 CHANGELOG 已加入：

```text
automatic mixed precision
```

说明：

> 工具链在尝试自动决定部分高精度路径。

---

# 74. W4A16

当前官方 CHANGELOG 中：

```text
RK3576
```

已加入：

```text
W4A16 symmetric quantization
```

---

# 75. W4A16 是什么？

概念：

```text
Weight:
4-bit

Activation / Compute:
16-bit related path。
```

具体格式：

```text
以 RKNN 当前文档为准。
```

不要把它直接等同：

```text
GGUF Q4_K_M
```

或：

```text
MLC q4f16_1。
```

---

# 76. 三种“4-bit”必须分开

```text
llama.cpp:
Q4_K_M

MLC:
q4f16_1

RKNN:
W4A16
```

它们：

```text
名字相似
生态不同
数据布局不同
Kernel Contract 不同。
```

---

# 77. Quantization 的真实收益

```text
Model Size ↓

DDR Traffic ↓

NPU Compute Efficiency ↑

Power ↓
```

但可能：

```text
Accuracy ↓。
```

---

# 78. 所以 Quantization 不是只看“能 build”

还要验证：

```text
Numerical Error

Task Accuracy

Performance。
```

---

# 79. Accuracy Analysis

RKNN-Toolkit2 提供：

```text
accuracy_analysis
```

用于：

```text
分析不同层的量化 / Simulator Error。
```

---

# 80. 为什么 Layer-wise Accuracy 很重要？

最终：

```text
mAP 下降 10%
```

只告诉你：

```text
坏了。
```

但不知道：

```text
哪一层开始坏。
```

---

# 81. Layer-wise 对比

理想：

```text
Original FP
↓
Layer 0
≈ RKNN

Layer 1
≈

Layer 2
Error suddenly ↑
```

就能定位：

```text
Sensitive Layer。
```

---

# 82. Cosine Similarity

常见：

```text
Cosine Similarity
```

比较：

```text
Reference Tensor
vs
RKNN Tensor。
```

---

# 83. 但 Cosine 高不代表任务精度一定高

最终还是要：

```text
完整 Dataset Metric。
```

例如：

```text
Classification Top1
Detection mAP
Segmentation mIoU。
```

---

# 84. QAT

官方 Toolkit2 也提供：

```text
QAT model example。
```

QAT：

```text
Quantization Aware Training。
```

训练时：

```text
模拟 Quant Error。
```

---

# 85. PTQ vs QAT

PTQ：

```text
训练完
再量化。
```

QAT：

```text
训练时就考虑量化。
```

---

# 86. QAT 什么时候值得？

如果：

```text
PTQ 精度下降过大
```

并且：

```text
有训练数据 / 训练能力。
```

---

# 87. 第三部分：Tensor Shape / Layout

NPU 对：

```text
Tensor Layout
```

通常比 CPU 更敏感。

---

# 88. RKNN Runtime Tensor Format

API 中可以看到：

```text
RKNN_TENSOR_NCHW

RKNN_TENSOR_NHWC

RKNN_TENSOR_NC1HWC2
```

---

# 89. NCHW

```text
[N, C, H, W]
```

PyTorch：

```text
经常使用。
```

---

# 90. NHWC

```text
[N, H, W, C]
```

很多 NPU / Mobile Runtime：

```text
更加常见。
```

---

# 91. NC1HWC2

这是：

```text
NPU native / blocked layout
```

的一类表示。

它说明：

> NPU 内部最适合的物理 Layout 不一定等于 Framework 中的 NCHW / NHWC。

---

# 92. 为什么会有 Blocked Layout？

硬件 Tensor Unit：

```text
一次并行处理多个 Channel。
```

所以数据：

```text
按 Hardware Vector Width 分块
```

更高效。

---

# 93. Layout Transform 的代价

如果：

```text
NCHW
↓
NHWC
↓
NPU Native
↓
NCHW
```

中间转换太多：

```text
会增加 Memory Traffic。
```

---

# 94. 所以优化不只是 Operator FLOPs

还要看：

```text
Layout Transform
Copy
Transpose。
```

---

# 95. 这与 GPU Coalescing 本质一致

GPU：

```text
Layout
影响 Memory Coalescing。
```

NPU：

```text
Layout
影响 Tensor Unit / DMA。
```

---

# 96. `rknn_tensor_attr`

Runtime C API 中非常重要。

它包含：

```text
name

dims

n_dims

n_elems

size

fmt

type

qnt_type

zp

scale

w_stride

size_with_stride
...
```

---

# 97. 为什么一定要打印 Tensor Attr？

因为部署错误很多来自：

```text
你以为：
NCHW float32

模型实际：
NHWC uint8。
```

---

# 98. 推荐每次加载模型都打印

```text
Input count

Output count

Input name

Shape

Format

Type

Quantization

Scale / Zero Point

Size / Stride。
```

---

# 99. Stride

NPU Memory：

```text
实际每行存储宽度
```

可能大于：

```text
逻辑 Width。
```

即：

```text
Padding / Alignment。
```

---

# 100. `size` vs `size_with_stride`

零拷贝时尤其重要。

不要只按：

```text
logical tensor elements
```

分配。

应该：

```text
按 API 返回的实际 buffer requirement。
```

---

# 101. 为什么硬件喜欢 Alignment？

因为：

```text
DMA
Bus
Vector Load
Tensor Unit
```

通常：

```text
对齐访问效率更高。
```

---

# 102. 第四部分：Dynamic Shape

传统 NPU：

```text
非常喜欢固定 Shape。
```

因为 Compiler 可以：

```text
提前规划 Memory
提前选择 Kernel
提前 Tile。
```

---

# 103. 但实际模型需要 Dynamic Shape

例如：

```text
160×160
224×224
256×256。
```

Toolkit2 提供：

```text
dynamic_input
```

相关能力。

---

# 104. 动态 Shape 不是完全无限动态

通常：

```text
编译时给出若干候选 Shape
```

或：

```text
受约束的 Dynamic Shape。
```

具体支持：

```text
以当前 API 文档为准。
```

---

# 105. 为什么 NPU Dynamic Shape 更难？

不同 Shape：

```text
Tile
Memory Plan
Operator implementation
```

可能不同。

---

# 106. Runtime 动态 Shape

板端：

```text
选择当前输入 Shape
```

然后：

```text
在模型支持的范围内运行。
```

---

# 107. C API 当前也提供

```text
rknn_set_input_shapes(...)
```

用于：

```text
dynamic shape RKNN model。
```

---

# 108. 第五部分：Flash Attention / Transformer 支持

当前 RKNN-Toolkit2 CHANGELOG 中出现：

```text
SDPA
Transformer optimization
MatMul optimization
Flash Attention
SoftmaxMask
Norm
einsum
```

等。

---

# 109. 为什么这值得关注？

早期 NPU 更偏：

```text
CNN。
```

现代 NPU Toolchain 开始强化：

```text
Transformer。
```

---

# 110. 但是要注意术语

Rockchip Toolkit 中：

```text
Flash Attention support
```

不应该理解为：

```text
把 Dao-AILab CUDA source
直接跑在 RKNPU。
```

---

# 111. 更准确理解

它表示：

> Rockchip NPU Toolchain 为 Attention 相关 Graph Pattern / Operator 提供了面向 RKNPU 的专用优化或实现能力。

---

# 112. 同样的数学思想

可能仍然：

```text
减少中间 Tensor
减少 DDR IO
做 Tile
融合 Softmax / Attention。
```

但实现：

```text
由 Rockchip Compiler / NPU Runtime 决定。
```

---

# 113. 这正是上一章知识迁移的地方

GPU：

```text
Shared Memory
Register
Tensor Core。
```

NPU：

```text
On-chip Buffer / SRAM
Tensor Engine
DMA。
```

底层结构不同，

但优化原则仍然：

```text
少搬数据
多复用
充分使用矩阵单元。
```

---

# 114. 第六部分：板端 Python Runtime

安装：

```text
RKNN-Toolkit-Lite2。
```

典型：

```python
from rknnlite.api import RKNNLite

rknn_lite = RKNNLite()

rknn_lite.load_rknn(
    'model.rknn'
)

rknn_lite.init_runtime()

outputs = rknn_lite.inference(
    inputs=[input_data]
)

rknn_lite.release()
```

---

# 115. 为什么 Lite2 很适合第一阶段？

因为：

```text
Python
```

能快速验证：

```text
Model Load
Input
Inference
Output。
```

---

# 116. 但最终产品不一定用 Python

嵌入式项目：

```text
C/C++
```

更常见。

所以：

```text
Lite2 用于快速验证
C API 用于正式集成。
```

---

# 117. RK3576 的 Core Mask

官方 Lite2 example 对：

```text
RK3576 / RK3588
```

支持通过：

```text
core_mask
```

选择 NPU Core / 自动模式。

例如概念：

```python
rknn_lite.init_runtime(
    core_mask=RKNNLite.NPU_CORE_0
)
```

实际：

```text
哪些 core 组合适用于当前芯片
```

应以当前 Runtime/API 为准。

---

# 118. `NPU_CORE_ALL`

当前 Lite2 在 RK3576 支持更新中：

```text
NPU_CORE_ALL
```

表示：

```text
由 Runtime 根据平台选择多个 NPU Core。
```

---

# 119. 多 NPU Core 一定线性加速吗？

不一定。

因为：

```text
Graph Partition
Memory Bandwidth
Model Shape
Core Scheduling
Parallel Efficiency。
```

都会影响。

---

# 120. 正确方法

对实际模型测：

```text
AUTO

CORE_0

CORE_ALL
```

或者当前平台支持的配置。

---

# 121. 第七部分：C/C++ RKNN Runtime

这是嵌入式工程最重要部分之一。

核心头文件：

- [rknn_api.h](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Android/librknn_api/include/rknn_api.h)

---

# 122. 最基本生命周期

```text
Load .rknn file
↓
rknn_init
↓
rknn_query
↓
rknn_inputs_set
↓
rknn_run
↓
rknn_outputs_get
↓
postprocess
↓
rknn_outputs_release
↓
rknn_destroy
```

---

# 123. `rknn_init`

```cpp
rknn_context ctx;

ret = rknn_init(
    &ctx,
    model_data,
    model_size,
    0,
    NULL
);
```

作用：

```text
Load / initialize model
↓
Create Runtime Context。
```

---

# 124. `rknn_context`

可以理解：

```text
一个 RKNN Runtime Model Instance / Execution Context Handle。
```

不要和：

```text
LLM Context Window
```

混淆。

---

# 125. `rknn_query`

用于：

```text
查询 Model / Tensor / Runtime 信息。
```

---

# 126. 最先查

```text
Input / Output Count。
```

例如：

```text
RKNN_QUERY_IN_OUT_NUM。
```

---

# 127. 再查每个 Tensor Attr

```text
RKNN_QUERY_INPUT_ATTR

RKNN_QUERY_OUTPUT_ATTR。
```

---

# 128. 为什么 C++ 集成第一件事就是 Query？

模型更新后：

```text
Shape / Type / Layout
```

可能变化。

不要：

```text
全写死。
```

---

# 129. `rknn_inputs_set`

传统 API：

```text
告诉 Runtime：
输入 Buffer 在哪里
Shape / dtype / fmt 是什么。
```

---

# 130. `rknn_run`

真正：

```text
提交推理执行。
```

---

# 131. `rknn_outputs_get`

等待：

```text
Inference Complete
```

并获得：

```text
Output。
```

---

# 132. `rknn_outputs_release`

释放：

```text
Runtime 分配 / 返回的输出资源。
```

---

# 133. `rknn_destroy`

释放：

```text
Context。
```

---

# 134. 最小 C API 心智图

```text
Model File
↓
rknn_init
↓
Context
↓
Query IO
↓
Set Input
↓
Run NPU
↓
Get Output
↓
Postprocess
↓
Destroy
```

---

# 135. 第八部分：Zero Copy

这是 RKNN 性能优化非常重要的主题。

传统：

```text
Camera / CPU Buffer
↓
copy
↓
RKNN Input Buffer
↓
NPU
↓
copy Output
↓
CPU。
```

---

# 136. Copy 的问题

如果输入：

```text
1080p / 4K
```

或者：

```text
高 FPS。
```

内存 Copy：

```text
可能非常贵。
```

---

# 137. Zero Copy 思想

目标：

```text
Producer Buffer
↓
Directly usable by RKNN Runtime
```

尽可能避免：

```text
额外 memcpy。
```

---

# 138. RKNN API 提供

```text
rknn_create_mem

rknn_create_mem2

rknn_create_mem_from_phys

rknn_create_mem_from_fd

rknn_set_io_mem。
```

---

# 139. `rknn_create_mem`

Runtime：

```text
分配可供 NPU 使用的 Tensor Memory。
```

返回：

```text
rknn_tensor_mem*。
```

---

# 140. `rknn_set_io_mem`

将：

```text
Tensor Attr
+
Tensor Memory
```

绑定到：

```text
Model Input / Output。
```

---

# 141. 为什么这比 `rknn_inputs_set` 更值得做工程优化？

因为：

```text
Memory Ownership / Layout
```

可以更明确控制。

减少：

```text
Runtime 内部 Copy / Conversion。
```

---

# 142. Zero Copy 不代表“完全没有任何数据搬运”

真正意思：

```text
减少 CPU-visible unnecessary copy。
```

NPU 内部：

```text
仍可能通过 DMA
从 DDR 搬到 on-chip memory。
```

---

# 143. 这是非常重要的区别

```text
Zero Copy
≠
Zero Memory Movement。
```

---

# 144. Camera → RGA → NPU

Rockchip 上很典型的 pipeline：

```text
Camera
↓
DMA Buffer
↓
RGA Resize / Color Convert
↓
DMABUF
↓
RKNN
↓
NPU。
```

目标：

```text
减少 CPU memcpy
减少 CPU preprocess。
```

---

# 145. `rknn_create_mem_from_fd`

非常有价值。

如果上游：

```text
Camera / DRM / RGA
```

提供：

```text
DMA-BUF fd。
```

可以尝试：

```text
导入 RKNN Tensor Memory。
```

---

# 146. 这就是嵌入式异构 Pipeline

```text
Camera
↓
ISP
↓
RGA
↓
NPU
↓
CPU Postprocess
```

不是：

```text
所有东西都 memcpy 到 OpenCV Mat。
```

---

# 147. Stride 再次重要

Zero Copy Buffer：

```text
可能有 Padding。
```

因此必须看：

```text
w_stride
size_with_stride。
```

---

# 148. 如果忽略 Stride

可能：

```text
图像错位
Output 异常
越界
性能下降。
```

---

# 149. 第九部分：Input Preprocessing

很多模型部署性能：

```text
NPU inference 很快
```

但总 Pipeline：

```text
仍然慢。
```

---

# 150. 原因可能是

```text
JPEG Decode

Resize

Letterbox

BGR → RGB

NHWC ↔ NCHW

Normalize

Memcpy。
```

---

# 151. 所以 End-to-End Latency

```text
T_total
=
T_decode
+
T_preprocess
+
T_copy_in
+
T_npu
+
T_copy_out
+
T_postprocess。
```

---

# 152. 不要只报：

```text
NPU inference = 5ms。
```

如果：

```text
preprocess = 12ms
postprocess = 8ms。
```

最终：

```text
25ms。
```

---

# 153. 这也是为什么 Model Zoo C++ 很值得读

它往往包含：

```text
RGA
Image conversion
Postprocess
NPU Runtime
```

完整工程。

---

# 154. 第十部分：Performance Evaluation

RKNN Toolkit / Model Zoo 提供多种性能观察方式。

---

# 155. `perf_debug`

一些 example：

```text
rknn.config(
    perf_debug=True
)
```

用于：

```text
获得更详细的 Layer / Performance information。
```

具体输出：

```text
以版本为准。
```

---

# 156. `eval_mem`

用于：

```text
Memory Evaluation。
```

---

# 157. 为什么 Layer Performance 很重要？

如果模型：

```text
50ms
```

需要知道：

```text
是哪 3 个 Layer 占 40ms？
```

---

# 158. 最常见瓶颈

```text
MatMul / Conv 太大

Transpose

Resize

Unsupported CPU Op

Layout Conversion

Postprocess

Memory Copy。
```

---

# 159. CPU Fallback

如果某 Operator：

```text
无法在 NPU 执行
```

可能：

```text
转 CPU implementation。
```

是否允许 / 如何表现：

```text
依当前 Toolkit / Operator 而定。
```

---

# 160. 为什么 Fallback 很危险？

例如：

```text
NPU Layer
↓
Tensor to DDR
↓
CPU Op
↓
Tensor convert
↓
NPU Layer。
```

开销可能：

```text
远大于单个 Op。
```

---

# 161. 所以不要只看“NPU 利用率”

更要看：

```text
整个 Graph 是否完整落到 NPU
是否频繁切换。
```

---

# 162. 模型支持 ≠ 每个 Op 都高效

有时：

```text
能 build
```

但：

```text
某个 Op 走低效 path。
```

需要：

```text
Layer Perf。
```

---

# 163. RKNN Model Zoo Benchmark

Model Zoo 当前提供：

```text
多个模型
多个 Rockchip 平台
FP16 / INT8
FPS。
```

很适合：

```text
建立初始性能预期。
```

---

# 164. 但不能直接拿 Zoo FPS 当你的产品性能

因为：

```text
Input Shape

Toolkit Version

Model Variant

Pre/Postprocess

CPU Frequency

NPU Core

Thermal

DDR

OS。
```

都会不同。

---

# 165. 正确用法

把 Model Zoo 当：

```text
Reference Baseline。
```

然后：

```text
自己 Benchmark。
```

---

# 166. 第十一部分：NPU 为什么快？

不要只用：

```text
TOPS
```

解释。

---

# 167. NPU 的本质

可以抽象成：

```text
Tensor Compute Engine
+
On-chip Memory
+
DMA
+
Task Scheduler
+
Specialized Dataflow。
```

---

# 168. 为什么比 CPU 更适合神经网络？

CPU：

```text
非常通用。
```

NPU：

```text
牺牲通用性
换 Tensor Workload 效率。
```

---

# 169. 大量 MAC

神经网络核心：

```text
Multiply-Accumulate。
```

NPU：

```text
提供大量并行 MAC / Matrix Unit。
```

---

# 170. Low Precision

INT8：

```text
数据更小
算术更简单
单位面积更多 MAC。
```

---

# 171. On-chip Memory

如果每个 MAC 都：

```text
去 DDR 读数据。
```

NPU 也不会快。

所以 NPU 必须：

```text
Tile
↓
On-chip Buffer
↓
Reuse。
```

---

# 172. 这和 GPU Shared Memory 完全同一个基本原理

GPU：

```text
HBM
↓
Shared
↓
Tensor Core。
```

NPU：

```text
DDR
↓
On-chip SRAM / local buffer
↓
Matrix Engine。
```

---

# 173. DMA

NPU 常通过：

```text
DMA Engine
```

搬：

```text
Input / Weight / Output Tile。
```

---

# 174. Compute 与 DMA 重叠

理想：

```text
Compute Tile N
```

同时：

```text
DMA Tile N+1。
```

这就是：

```text
Pipeline / Double Buffer。
```

---

# 175. 为什么 Compiler 很重要？

因为用户没有直接写：

```text
RKNPU Kernel。
```

Compiler 必须决定：

```text
Tile

Layout

Dtype

Fusion

Memory

DMA

Operator Mapping。
```

---

# 176. 所以 NPU 性能高度依赖 Compiler

同一个 Hardware：

```text
不同 Toolkit Version
```

可以：

```text
性能不同。
```

---

# 177. 第十二部分：Graph Fusion

例子：

```text
Conv
↓
BatchNorm
↓
ReLU。
```

Compiler 可以：

```text
Fuse。
```

---

# 178. 为什么？

减少：

```text
Intermediate Tensor
```

落：

```text
DDR。
```

---

# 179. Transformer Fusion

例如：

```text
MatMul
↓
Scale
↓
Mask
↓
Softmax。
```

如果能识别：

```text
Attention Pattern
```

可以：

```text
走专用 implementation。
```

---

# 180. Pattern Recognition 非常重要

两个 mathematically-equivalent Graph：

```text
Graph A
```

和：

```text
Graph B
```

可能：

```text
Compiler 只识别 A。
```

于是性能：

```text
差很多。
```

---

# 181. 这就是“改模型 Graph 以适配 NPU”

不是：

```text
改变模型意义。
```

而是：

```text
写成 Compiler 更容易识别的 Pattern。
```

---

# 182. 常见 Model Surgery

```text
Replace unsupported op

Fold constants

Replace dynamic op

Simplify graph

Fix input shape

Change Resize

Change activation

Fuse preprocess

Avoid unnecessary transpose。
```

---

# 183. ONNX Simplification

有时：

```text
onnxsim
```

或手工 Graph Surgery：

```text
有帮助。
```

但必须：

```text
重新验证 Accuracy。
```

---

# 184. 第十三部分：Custom Operator

RKNN-Toolkit2 example 提供：

```text
custom_op。
```

说明：

```text
Toolchain 存在扩展机制。
```

---

# 185. 但 Custom Op 不等于自动在 NPU 上有硬件实现

需要区分：

```text
Model Converter 能认识这个 Op
```

和：

```text
RKNPU 有高效 Kernel。
```

---

# 186. Custom CPU / GPU Op

某些版本支持：

```text
custom operator
```

在：

```text
CPU / GPU
```

执行。

这更像：

```text
Fallback Extension。
```

---

# 187. 为什么这不是最佳性能路径？

可能发生：

```text
NPU
↓
External Op
↓
NPU。
```

增加：

```text
Copy / Sync。
```

---

# 188. 第十四部分：MatMul API

RKNN Runtime 还提供：

```text
rknn_matmul_api.h
```

用于：

```text
直接调用 NPU MatMul 能力。
```

---

# 189. 为什么单独有 MatMul API？

因为：

```text
Transformer
LLM
Linear Algebra
```

大量核心是：

```text
MatMul。
```

用户有时：

```text
不需要完整 static graph
```

而需要：

```text
直接执行矩阵乘。
```

---

# 190. MatMul API 会暴露更多 Layout 概念

例如：

```text
normal layout

native layout
```

说明：

> NPU 高性能矩阵乘会有自己的硬件友好数据布局。

---

# 191. 这和 CUTLASS CuTe Layout 的思想高度一致

GPU：

```text
Tensor Core-friendly Layout。
```

NPU：

```text
RKNPU native layout。
```

---

# 192. 只是 API 开放程度不同

CUDA：

```text
更多硬件细节对程序员开放。
```

RKNPU：

```text
更多由 Vendor Runtime / Compiler 封装。
```

---

# 193. 第十五部分：Flash Attention 与 RKNN

当前 Toolkit2 CHANGELOG 已出现：

```text
Flash Attention
```

且某版本注明：

```text
Only RK3562 and RK3576。
```

---

# 194. 对 RK3576 的意义

说明：

```text
RKNPU Toolchain
```

不仅适合：

```text
CNN
```

也在增强：

```text
Transformer Operator。
```

---

# 195. 但不要因此得出

```text
RKNN-Toolkit2 可以直接跑任意 LLM。
```

错误。

---

# 196. 为什么？

LLM 还需要：

```text
Tokenizer
KV Cache
Prefill
Decode
Sampling
Dynamic autoregressive loop
Long Context
Weight-specific quantization。
```

---

# 197. 所以 Rockchip 单独提供 RKNN-LLM

这会在：

```text
12_RKLLM与端侧LLM部署.md
```

继续。

---

# 198. 第十六部分：一个 YOLO 完整部署链

典型：

```text
PyTorch YOLO
↓
Export ONNX
↓
ONNX Runtime Verify
↓
RKNN config
↓
load_onnx
↓
INT8 build
↓
export .rknn
↓
RKNNLite test
↓
C++ Runtime
↓
RGA preprocess
↓
NPU
↓
YOLO postprocess。
```

---

# 199. 为什么 YOLO 是很好的 RKNN 入门模型？

它包含：

```text
Conv
Activation
Detection Head
Multi-output
Quantization
Postprocess。
```

足够工程化。

---

# 200. 但第一模型更建议 MobileNet

学习顺序：

```text
MobileNet
↓
ResNet
↓
YOLO
↓
Segmentation
↓
Transformer。
```

---

# 201. 为什么？

MobileNet：

```text
简单
Output 清晰
便于定位 pipeline。
```

---

# 202. 第十七部分：推荐 RKNN 学习路线

---

## Stage 1：跑官方 example

目标：

```text
不改模型
先跑通。
```

推荐：

```text
MobileNet / ResNet18。
```

---

## Stage 2：自己转 ONNX

```text
PyTorch
↓
ONNX。
```

---

## Stage 3：ONNX Runtime Verify

确保：

```text
ONNX 正确。
```

---

## Stage 4：FP16 / Non-quant RKNN

先：

```text
do_quantization=False。
```

---

## Stage 5：板端验证

用：

```text
RKNNLite。
```

---

## Stage 6：INT8 PTQ

加：

```text
Calibration Dataset。
```

---

## Stage 7：Accuracy Analysis

定位：

```text
敏感 Layer。
```

---

## Stage 8：Hybrid Quant / Mixed Precision

处理：

```text
INT8 精度下降。
```

---

## Stage 9：C Runtime

自己写：

```text
minimal_rknn.cpp。
```

---

## Stage 10：Zero Copy

用：

```text
rknn_create_mem
rknn_set_io_mem。
```

---

## Stage 11：RGA

把：

```text
Resize / Color Convert
```

从 CPU 移走。

---

## Stage 12：Benchmark

记录：

```text
Preprocess
NPU
Postprocess
Total。
```

---

## Stage 13：Dynamic Shape

实验：

```text
多 Input Shape。
```

---

## Stage 14：Transformer Operator

尝试：

```text
小 Attention / Transformer model。
```

理解：

```text
Toolchain support boundary。
```

---

# 203. 推荐实验 1：MobileNet FP

流程：

```text
ONNX
↓
RKNN FP
↓
Board。
```

记录：

```text
Output Top5
Latency
Memory。
```

---

# 204. 实验 2：MobileNet INT8

同一模型：

```text
INT8。
```

比较：

```text
Model Size
Latency
Accuracy。
```

---

# 205. 实验 3：Calibration Set

用：

```text
10
50
100
500
```

张代表数据。

观察：

```text
Accuracy。
```

---

# 206. 实验 4：MMSE Quant

比较：

```text
normal
vs
mmse。
```

---

# 207. 实验 5：Hybrid Quant

对精度敏感模型：

```text
Step1
↓
Find sensitive layer
↓
Hybrid config
↓
Step2。
```

---

# 208. 实验 6：Input Layout

比较：

```text
NCHW
NHWC。
```

观察：

```text
输入处理
Runtime
结果。
```

---

# 209. 实验 7：Zero Copy

比较：

```text
rknn_inputs_set

vs

rknn_create_mem + rknn_set_io_mem。
```

---

# 210. 测量

不要只测：

```text
rknn_run。
```

要测：

```text
copy-in
run
copy-out
total。
```

---

# 211. 实验 8：RGA Preprocess

```text
OpenCV CPU resize
```

vs：

```text
RGA resize。
```

---

# 212. 实验 9：YOLO

部署：

```text
YOLOv5 / YOLOv8
```

使用：

```text
Model Zoo
```

作为 reference。

---

# 213. 实验 10：Layer Performance

开启：

```text
perf_debug。
```

找到：

```text
Top 5 expensive layers。
```

---

# 214. 实验 11：Memory

开启：

```text
eval_mem。
```

记录：

```text
Weight
Internal Memory
IO。
```

---

# 215. 实验 12：Core Mask

RK3576：

```text
AUTO
CORE_0
ALL
```

在当前 SDK 支持范围内比较。

---

# 216. 实验 13：Dynamic Shape

转换：

```text
160
224
256
```

三种输入。

板端：

```text
分别运行。
```

---

# 217. 实验 14：Toolkit Version

如果有条件：

```text
2 个 Toolkit Version
```

同一 ONNX：

```text
重新 build。
```

比较：

```text
.rknn size
accuracy
latency。
```

---

# 218. 这个实验会非常直观说明

```text
Compiler Version
=
Performance Variable。
```

---

# 219. 第十八部分：调试方法

出现问题时按层排查。

---

# 220. Level 1：PyTorch

确认：

```text
原模型正常。
```

---

# 221. Level 2：ONNX

确认：

```text
ONNX Runtime Output 正常。
```

---

# 222. Level 3：RKNN FP

```text
do_quantization=False。
```

如果已经错：

```text
不是 INT8 问题。
```

---

# 223. Level 4：RKNN INT8

如果 FP 正确，INT8 错：

```text
重点看 Quantization。
```

---

# 224. Level 5：Simulator vs Board

如果 Simulator 正确，Board 错：

重点：

```text
Toolkit / Runtime / Driver
Input Memory
Layout
C API。
```

---

# 225. Level 6：Python vs C++

如果 Lite2 正确，C++ 错：

重点：

```text
Input Attr
Output Attr
dtype
fmt
stride
memory lifetime
postprocess。
```

---

# 226. 这是一条非常强的排错链

```text
PyTorch
↓
ONNX
↓
RKNN FP
↓
RKNN INT8
↓
RKNNLite
↓
C Runtime。
```

---

# 227. 不要跨层猜

例如：

```text
C++ 输出错
```

不要立刻：

```text
重新训练模型。
```

---

# 228. 常见错误 1：颜色顺序

```text
BGR vs RGB。
```

---

# 229. 常见错误 2：Input Range

```text
0~255
vs
0~1。
```

---

# 230. 常见错误 3：Normalize 重复

Model config：

```text
已经 mean/std。
```

应用又：

```text
手工 normalize。
```

等于：

```text
两次归一化。
```

---

# 231. 常见错误 4：Layout

```text
NCHW
vs
NHWC。
```

---

# 232. 常见错误 5：Quantized Output

模型输出：

```text
INT8。
```

应用：

```text
直接当 float。
```

---

# 233. 需要：

```text
scale / zp
```

反量化：

```text
f = (q - zp) * scale。
```

---

# 234. 常见错误 6：Stride

Buffer：

```text
按 logical width 写
```

但实际：

```text
w_stride > width。
```

---

# 235. 常见错误 7：Postprocess

YOLO：

```text
Anchor / sigmoid / DFL / scale / NMS
```

不同版本：

```text
完全不同。
```

---

# 236. 常见错误 8：Toolkit / Runtime mismatch

编译：

```text
新版 Toolkit。
```

板端：

```text
老 Runtime。
```

---

# 237. 常见错误 9：rknn_server

连板：

```text
server 不运行 / 版本异常。
```

---

# 238. 常见错误 10：模型 Graph 变了

新模型：

```text
Output Node
```

和旧代码：

```text
不匹配。
```

---

# 239. 第十九部分：性能优化顺序

不要一开始：

```text
调 NPU Core。
```

推荐顺序：

```text
1. Correctness

2. INT8 / W4A16 if appropriate

3. Remove Unsupported / CPU Ops

4. Reduce Layout Transform

5. Optimize Input Pipeline

6. Zero Copy

7. RGA

8. Core Scheduling

9. End-to-end Profiling。
```

---

# 240. 为什么 Correctness 第一？

错误模型：

```text
1000 FPS
```

没有意义。

---

# 241. 为什么 Unsupported Op 第二优先？

一个：

```text
CPU fallback
```

可能毁掉整个 NPU pipeline。

---

# 242. 为什么 Layout Transform 很重要？

它往往：

```text
没有 FLOPs
```

但：

```text
非常吃 Memory Bandwidth。
```

---

# 243. 为什么 Zero Copy 靠后？

先确保：

```text
模型和 Tensor Attr
都正确。
```

否则：

```text
Zero Copy Debug 更复杂。
```

---

# 244. 第二十部分：End-to-End 系统优化

实际产品：

```text
Camera
↓
Preprocess
↓
NPU
↓
Postprocess
↓
Application。
```

---

# 245. 目标不是：

```text
NPU FPS 最大。
```

而是：

```text
Product Latency
Power
Accuracy
Stability。
```

---

# 246. Camera Pipeline

理想：

```text
V4L2 / Camera
↓
DMA-BUF
↓
RGA
↓
RKNN
↓
NPU
↓
Output
```

---

# 247. 为什么比 OpenCV Pipeline 好？

OpenCV 可能：

```text
CPU Decode
CPU Resize
CPU Color
Memcpy
```

占用：

```text
CPU + DDR。
```

---

# 248. 异构 SoC 的正确思维

```text
CPU:
Control / Postprocess / Application

RGA:
Image Process

NPU:
Neural Network

GPU:
Graphics / optional compute

ISP:
Camera pipeline。
```

---

# 249. 不要让 CPU 干所有事

RK3576 的价值：

```text
异构计算。
```

---

# 250. 第二十一部分：与 AI Compiler 的映射

上一章：

```text
Frontend
↓
Relax
↓
Graph Pass
↓
TensorIR
↓
Schedule
↓
CodeGen。
```

RKNN 黑盒可抽象：

```text
Model Import
↓
Vendor Graph IR
↓
Graph Pass
↓
Quantization
↓
Operator Lowering
↓
RKNPU Mapping
↓
Vendor Code Generation
↓
.rknn。
```

---

# 251. 这不是说 RKNN 内部使用 TVM

不要误解。

这里只是：

```text
Compiler 抽象结构相似。
```

---

# 252. `load_onnx`

对应：

```text
Frontend。
```

---

# 253. `build`

对应：

```text
Optimization
+
Quantization
+
Lowering
+
Target Compilation。
```

---

# 254. `.rknn`

对应：

```text
Compiled Artifact。
```

---

# 255. `librknnrt.so`

对应：

```text
Runtime。
```

---

# 256. `RKNPU Driver`

对应：

```text
Device Driver。
```

---

# 257. NPU

对应：

```text
Target Hardware。
```

---

# 258. 第二十二部分：与 GPU Kernel 的映射

GPU：

```text
HBM
↓
Shared
↓
Register
↓
Tensor Core。
```

NPU：

```text
DDR
↓
On-chip Buffer
↓
Tensor Data Path
↓
MAC / Matrix Engine。
```

---

# 259. GPU Programmer 自己控制更多

```text
Tile
Thread
Shared
Warp。
```

---

# 260. NPU 中更多由 Compiler 控制

所以开发者主要影响：

```text
Graph
Shape
Layout
Dtype
Quantization
Fusion Pattern。
```

---

# 261. 这就是 NPU 优化思维的区别

GPU：

```text
写好 Kernel。
```

NPU：

```text
给 Compiler 一个
容易映射到硬件的 Graph。
```

---

# 262. 第二十三部分：与 llama.cpp 的区别

llama.cpp：

```text
GGUF
↓
ggml
↓
CPU / GPU Backend
↓
Kernel。
```

RKNN：

```text
ONNX / Model
↓
RKNN Compiler
↓
.rknn
↓
NPU Runtime。
```

---

# 263. llama.cpp 更开放

可以：

```text
看到 Graph
看到 Kernel
修改 Backend。
```

---

# 264. RKNN 更封装

用户主要看到：

```text
Compiler API
Runtime API
Profiling Result。
```

---

# 265. 所以学 llama.cpp / TVM 的价值

帮助理解：

```text
RKNN 黑盒里
大概发生什么。
```

---

# 266. 第二十四部分：与 MLC-LLM 的区别

MLC：

```text
Open Compiler Stack
Relax / TIR
CUDA / Vulkan / Metal。
```

RKNN：

```text
Vendor NPU Compiler
Target = RKNPU。
```

---

# 267. 两者共同点

```text
Model Conversion
Target Compilation
Runtime Artifact
Device Runtime。
```

---

# 268. 第二十五部分：与 ONNX Runtime EP 的类比

ONNX Runtime：

```text
Graph
↓
Execution Provider。
```

RKNN：

```text
Graph
↓
RKNPU Compiler / Runtime。
```

---

# 269. 如果某 Operator 不支持

ORT：

```text
可能落 CPU EP。
```

RKNN：

```text
可能 CPU path / conversion fail
depending on operator/toolchain。
```

---

# 270. 所以“Operator Coverage”是 NPU 生态生命线

硬件理论 TOPS 再高，

如果：

```text
模型大量 Op 不支持
```

实际性能：

```text
很差。
```

---

# 271. 第二十六部分：为什么 Transformer 对 NPU Toolchain 挑战更大？

CNN：

```text
Conv
ReLU
Pooling
```

Shape：

```text
相对固定。
```

---

# 272. Transformer

大量：

```text
MatMul
Softmax
LayerNorm
Transpose
Reshape
Gather
Attention
Dynamic Sequence。
```

---

# 273. 并且 LLM

还有：

```text
KV Cache
Autoregressive Decode
Dynamic Context
Sampling。
```

---

# 274. 所以普通 RKNN 与 RKLLM 分开很合理

普通：

```text
Static / Semi-static Tensor Graph。
```

LLM：

```text
Runtime State Machine。
```

---

# 275. 下一章要重点看

```text
RKNN-LLM
为什么需要单独 Runtime？
```

---

# 276. 第二十七部分：RKNN Model Zoo 的正确用法

不要只：

```text
copy example command。
```

而应该拆成：

```text
Model Conversion

Python Reference

C Runtime

Preprocess

Postprocess

Build Script。
```

---

# 277. 推荐先读目录结构

例如：

```text
examples/yolov5/
│
├── model
├── python
├── cpp
└── README。
```

---

# 278. 找出四条线

```text
1. Input preprocessing

2. RKNN inference

3. Output decoding

4. Platform build。
```

---

# 279. 对自己模型迁移

不是：

```text
整份代码复制。
```

而是：

```text
保留 Runtime Skeleton
替换：
Preprocess
Tensor Contract
Postprocess。
```

---

# 280. 第二十八部分：最小 C++ 工程应该长什么样？

```text
rknn_minimal/
│
├── CMakeLists.txt
├── model.rknn
├── include/
│   └── rknn_api.h
├── lib/
│   └── librknnrt.so
└── src/
    └── main.cpp
```

---

# 281. `main.cpp`

逻辑：

```text
read model
↓
rknn_init
↓
query input/output
↓
prepare tensor
↓
rknn_inputs_set
↓
rknn_run
↓
rknn_outputs_get
↓
print first values
↓
release。
```

---

# 282. 第一版不要加

```text
OpenCV
RGA
Camera
Thread Pool
NMS
GUI。
```

---

# 283. 为什么？

先验证：

```text
Runtime。
```

然后再：

```text
逐层集成。
```

---

# 284. 第二十九部分：多线程 / 多 Context

生产环境可能：

```text
多个 Model
多个 Stream
多个 Thread。
```

---

# 285. 需要关注

```text
Context lifecycle
Thread safety
NPU core scheduling
Memory
Synchronization。
```

---

# 286. 不要假设

```text
创建更多 Context
=
吞吐线性增加。
```

最终仍受：

```text
NPU Core
DDR
Runtime Scheduler。
```

---

# 287. 多 Camera

典型：

```text
Camera 0
Camera 1
Camera 2
```

可能需要：

```text
Preprocess Pipeline
Queue
NPU Scheduling
Postprocess Thread。
```

---

# 288. 第三十部分：性能记录模板

建议每个模型保存：

```text
model_name

source_format

input_shape

dtype

quant_mode

rknn_toolkit_version

runtime_version

driver_version

board

npu_core_mode

model_size

preprocess_ms

npu_ms

postprocess_ms

total_ms

fps

accuracy。
```

---

# 289. 不要只写：

```text
NPU 20 FPS。
```

没有：

```text
版本 / Shape / Quant
```

几乎无法复现。

---

# 290. 第三十一部分：功耗 / 温度

端侧：

```text
Performance
```

不只是：

```text
Peak FPS。
```

---

# 291. 长时间运行

可能：

```text
Thermal Throttling。
```

---

# 292. Benchmark 至少分

```text
Cold Start

Warm Inference

Sustained 5min / 10min。
```

---

# 293. 记录

```text
CPU Frequency

NPU Frequency

DDR Frequency

Temperature

Power。
```

具体工具：

```text
依据 Board BSP。
```

---

# 294. 第三十二部分：模型启动时间

产品还需要：

```text
Model Load Time。
```

---

# 295. 冷启动

```text
Read .rknn
↓
rknn_init
↓
Memory init
↓
Ready。
```

---

# 296. 这对什么重要？

```text
Battery Device

On-demand AI

Multiple Model Switch。
```

---

# 297. 第三十三部分：Memory Planning

即使 NPU 很快，

模型可能：

```text
DDR 不够。
```

---

# 298. 内存通常包括

```text
Model Weight

Internal Tensor

Input / Output

Runtime Workspace

Application Buffer

Camera Buffer。
```

---

# 299. Quantization 也改善 Memory

INT8：

```text
Weight Size ↓
Activation Size ↓
```

具体是否全部以低精度保存：

```text
取决于 compiled graph。
```

---

# 300. 第三十四部分：一个完整问题定位案例

假设：

```text
ONNX 正确
RKNN INT8 mAP 下降 20%。
```

---

# 301. Step 1

跑：

```text
RKNN FP。
```

---

# 302. 如果 FP 也错

查：

```text
Operator Conversion

Preprocess

Output Layout

Postprocess。
```

---

# 303. 如果 FP 正确

问题集中：

```text
Quantization。
```

---

# 304. Step 2

用：

```text
accuracy_analysis。
```

找到：

```text
第一个误差大的 Layer。
```

---

# 305. Step 3

尝试：

```text
更代表性的 calibration data

MMSE

per-channel strategy

hybrid quant

mixed precision。
```

---

# 306. Step 4

重新跑：

```text
完整 mAP。
```

---

# 307. 这比：

```text
盲目换模型
```

有效得多。

---

# 308. 第三十五部分：另一个性能案例

假设：

```text
NPU layer perf 总计 8ms

Application 总 latency 30ms。
```

---

# 309. 说明：

```text
问题不在 NPU。
```

查：

```text
OpenCV resize
Color Convert
Memcpy
NMS
Camera Buffer。
```

---

# 310. 可能优化：

```text
RGA
Zero Copy
Quantized Output Postprocess
Thread Pipeline。
```

---

# 311. 这就是“系统优化”与“算子优化”的区别

```text
Kernel 快
≠
产品快。
```

---

# 312. 第三十六部分：源码 / API 阅读顺序

推荐：

```text
README
↓
CHANGELOG
↓
Toolkit2 simple example
↓
Model Zoo simple model
↓
RKNNLite example
↓
rknn_api.h
↓
C API demo
↓
Zero Copy demo
↓
Dynamic Shape
↓
Hybrid Quant
↓
Model Zoo YOLO
↓
RKNN-LLM。
```

---

# 313. 第一阶段不要读什么？

不要：

```text
一开始研究二进制 .rknn 格式逆向

直接看 Kernel Driver 源码

直接研究所有 Operator。
```

---

# 314. 为什么？

你的主线应该先是：

```text
Model
↓
Compiler
↓
Runtime
↓
Hardware。
```

---

# 315. 第三十七部分：推荐实践仓库

```text
rknn-learning/
│
├── 01_mobilenet_fp/
│   ├── export_onnx.py
│   ├── convert_rknn.py
│   └── verify.py
│
├── 02_mobilenet_int8/
│   ├── dataset.txt
│   ├── quant.py
│   └── accuracy.py
│
├── 03_rknnlite/
│   └── inference.py
│
├── 04_c_api/
│   ├── CMakeLists.txt
│   └── main.cpp
│
├── 05_zero_copy/
│   └── main.cpp
│
├── 06_rga_preprocess/
│   └── ...
│
├── 07_yolo/
│   └── ...
│
├── 08_dynamic_shape/
│   └── ...
│
├── 09_perf_debug/
│   └── ...
│
└── notes/
    ├── model_contract.md
    ├── benchmark.md
    └── version_matrix.md
```

---

# 316. `model_contract.md`

每个模型记录：

```text
Input Name

Input Shape

Input Dtype

Input Format

Color

Mean / Std

Resize

Output Name

Output Shape

Output Dtype

Quant Scale / ZP

Postprocess。
```

---

# 317. `version_matrix.md`

记录：

```text
Toolkit2

Lite2

Runtime

Driver

BSP。
```

---

# 318. `benchmark.md`

记录：

```text
FP / INT8

preprocess

inference

postprocess

total

memory

accuracy。
```

---

# 319. 第三十八部分：本章必做实验

- [ ] Clone `rknn-toolkit2`
- [ ] Clone `rknn_model_zoo`
- [ ] 跑通一个官方 MobileNet example
- [ ] 自己导出一个 PyTorch ONNX
- [ ] 用 ONNX Runtime 验证
- [ ] 转 FP RKNN
- [ ] 板端 Lite2 验证
- [ ] 转 INT8 RKNN
- [ ] 准备 calibration dataset
- [ ] 比较 FP vs INT8 accuracy
- [ ] 使用 `accuracy_analysis`
- [ ] 尝试 MMSE
- [ ] 尝试 Hybrid Quant
- [ ] 读取 `.rknn` input/output attrs
- [ ] 打印 dtype / fmt / scale / zp
- [ ] 写最小 C API demo
- [ ] 使用 `rknn_inputs_set`
- [ ] 使用 `rknn_run`
- [ ] 使用 `rknn_outputs_get`
- [ ] 使用 `rknn_create_mem`
- [ ] 使用 `rknn_set_io_mem`
- [ ] 理解 stride
- [ ] 测 Zero Copy
- [ ] 测 NPU core mode
- [ ] 开 `perf_debug`
- [ ] 开 `eval_mem`
- [ ] 部署一个 YOLO
- [ ] 测 preprocess / NPU / postprocess
- [ ] 尝试 RGA preprocess
- [ ] 测 sustained performance
- [ ] 保存完整版本信息

---

# 320. 第三十九部分：常见误区

---

## 误区 1

```text
RKNN = .rknn。
```

不准确。

RKNN 可以泛指：

```text
生态 / SDK。
```

`.rknn`：

```text
具体 Deployment Artifact。
```

---

## 误区 2

```text
RKNN-Toolkit2
=
板端 Runtime。
```

错误。

它主要是：

```text
Host-side conversion / development SDK。
```

---

## 误区 3

```text
RKNN-Toolkit-Lite2
=
模型转换工具。
```

错误。

它主要：

```text
板端 Python Inference。
```

---

## 误区 4

```text
librknnrt.so
=
Compiler。
```

错误。

它是：

```text
Runtime。
```

---

## 误区 5

```text
.rknn
=
ONNX 改了扩展名。
```

错误。

它经过：

```text
Target-specific compilation。
```

---

## 误区 6

```text
模型 build 成功
=
全部跑 NPU。
```

不能这样假设。

应检查：

```text
Compiler log
Layer perf
Operator mapping。
```

---

## 误区 7

```text
NPU TOPS 高
=
任何模型都快。
```

错误。

还取决：

```text
Operator coverage
Quant
Layout
Memory
Compiler。
```

---

## 误区 8

```text
INT8
=
一定精度差很多。
```

错误。

很多模型：

```text
良好 calibration
```

可保持很高精度。

---

## 误区 9

```text
Calibration dataset
=
训练集越大越好。
```

错误。

关键：

```text
Representative。
```

---

## 误区 10

```text
Q4_K_M
=
RKNN W4A16。
```

错误。

---

## 误区 11

```text
Zero Copy
=
NPU 不搬数据。
```

错误。

只表示：

```text
减少不必要 CPU Copy。
```

---

## 误区 12

```text
NHWC / NCHW
只影响 API 写法。
```

错误。

它可能：

```text
直接影响 Layout Transform 和性能。
```

---

## 误区 13

```text
Toolkit 的 Flash Attention
=
Dao-AILab CUDA FlashAttention source。
```

错误。

是：

```text
不同 Backend / Hardware 的实现。
```

---

## 误区 14

```text
RKNN-Toolkit2
可以直接替代 RKLLM。
```

错误。

LLM：

```text
有独立 Toolchain。
```

---

## 误区 15

```text
C++ 比 Python 结果不一致
=
NPU 问题。
```

通常先查：

```text
Input attr
dtype
layout
stride
quant output
postprocess。
```

---

# 321. 第四十部分：知识树

```text
Rockchip NPU
│
├── Host Toolchain
│   └── RKNN-Toolkit2
│       ├── config
│       ├── load model
│       ├── graph optimization
│       ├── quantization
│       ├── target compilation
│       ├── accuracy analysis
│       ├── performance analysis
│       └── export .rknn
│
├── Model Artifact
│   └── .rknn
│       ├── target-specific
│       ├── compiled
│       ├── quantized / FP
│       └── runtime-loadable
│
├── Board Python
│   └── RKNN-Toolkit-Lite2
│       ├── load_rknn
│       ├── init_runtime
│       ├── inference
│       └── release
│
├── Board C/C++
│   └── RKNN Runtime
│       ├── rknn_init
│       ├── rknn_query
│       ├── rknn_inputs_set
│       ├── rknn_run
│       ├── rknn_outputs_get
│       ├── rknn_outputs_release
│       └── rknn_destroy
│
├── Memory API
│   ├── rknn_create_mem
│   ├── create_mem_from_fd
│   ├── set_io_mem
│   ├── stride
│   └── zero copy
│
├── Tensor
│   ├── NCHW
│   ├── NHWC
│   ├── NC1HWC2
│   ├── FP16
│   ├── INT8
│   ├── Scale
│   └── Zero Point
│
├── Quantization
│   ├── PTQ
│   ├── Calibration
│   ├── MMSE
│   ├── Hybrid Quant
│   ├── QAT
│   ├── Mixed Precision
│   └── W4A16
│
├── Compiler
│   ├── Graph Optimization
│   ├── Fusion
│   ├── Layout
│   ├── Operator Mapping
│   ├── Dynamic Shape
│   ├── Transformer Optimization
│   └── Target Compilation
│
├── Performance
│   ├── perf_debug
│   ├── eval_mem
│   ├── Layer Perf
│   ├── Zero Copy
│   ├── RGA
│   ├── Core Mask
│   └── End-to-End
│
├── Driver
│   └── RKNPU Kernel Driver
│
└── Hardware
    └── RKNPU
        ├── Tensor Compute
        ├── On-chip Memory
        ├── DMA
        └── DDR
```

---

# 322. 本章最重要的 12 个思维模型

## 1

```text
RKNN-Toolkit2
=
Vendor AI Compiler SDK。
```

---

## 2

```text
.rknn
=
Target-specific Deployment Artifact。
```

---

## 3

```text
RKNN Runtime
=
Board-side Execution Runtime。
```

---

## 4

```text
RKNNLite
=
Board-side Python API。
```

---

## 5

```text
RKNPU Driver
=
Runtime 与 Hardware 之间的 Kernel Layer。
```

---

## 6

```text
NPU 性能
=
Graph Mapping
+
Quantization
+
Data Movement
+
Tensor Engine Utilization。
```

---

## 7

```text
能 Build
≠
高性能。
```

---

## 8

```text
INT8
不仅省模型
还可能直接映射高吞吐 NPU Datapath。
```

---

## 9

```text
Zero Copy
=
减少 CPU-side redundant copy

不是：
完全没有 Memory Movement。
```

---

## 10

```text
GPU Kernel 优化：
程序员显式控制较多

NPU 优化：
Compiler 自动控制较多。
```

---

## 11

```text
对 NPU 最重要的输入是：
Graph + Shape + Layout + Dtype。
```

---

## 12

```text
General RKNN
≠
RKLLM。
```

---

# 323. 本章完成标准

如果下面大部分都可以自己解释，本章就可以结束。

## 生态

- [ ] 能解释 RKNN
- [ ] 能解释 `.rknn`
- [ ] 能解释 RKNN-Toolkit2
- [ ] 能解释 RKNN-Toolkit-Lite2
- [ ] 能解释 RKNN Runtime
- [ ] 能解释 RKNPU Driver
- [ ] 能解释 RKNN-LLM
- [ ] 能画出整个软件栈

## Compiler

- [ ] 能解释 `rknn.config`
- [ ] 能解释 `load_onnx`
- [ ] 能解释 `build`
- [ ] 能解释 `export_rknn`
- [ ] 能解释为什么 build 是 Compiler 阶段
- [ ] 能解释 Target-aware Compilation
- [ ] 能解释 Operator Mapping
- [ ] 能解释 Fusion
- [ ] 能解释 Layout Transformation
- [ ] 能解释 Dynamic Shape

## Quantization

- [ ] 能解释 PTQ
- [ ] 能解释 Calibration
- [ ] 能解释 Scale
- [ ] 能解释 Zero Point
- [ ] 能解释 Affine Quant
- [ ] 能解释 Per-channel
- [ ] 能解释 MMSE
- [ ] 能解释 Hybrid Quant
- [ ] 能解释 QAT
- [ ] 能解释 W4A16
- [ ] 能解释为什么 Q4_K_M ≠ W4A16

## Tensor Contract

- [ ] 能解释 NCHW
- [ ] 能解释 NHWC
- [ ] 能解释 NC1HWC2
- [ ] 能解释 Stride
- [ ] 能解释 `size_with_stride`
- [ ] 能打印 `rknn_tensor_attr`
- [ ] 能解释 quantized output 的 scale / zp

## Runtime

- [ ] 能解释 `rknn_context`
- [ ] 能使用 `rknn_init`
- [ ] 能使用 `rknn_query`
- [ ] 能使用 `rknn_inputs_set`
- [ ] 能使用 `rknn_run`
- [ ] 能使用 `rknn_outputs_get`
- [ ] 能使用 `rknn_outputs_release`
- [ ] 能使用 `rknn_destroy`

## Zero Copy

- [ ] 能解释 Zero Copy
- [ ] 能使用 `rknn_create_mem`
- [ ] 能使用 `rknn_set_io_mem`
- [ ] 知道 `create_mem_from_fd`
- [ ] 能解释 DMA-BUF
- [ ] 能解释 Zero Copy ≠ Zero Data Movement
- [ ] 能处理 stride

## Performance

- [ ] 能使用 / 理解 `perf_debug`
- [ ] 能使用 / 理解 `eval_mem`
- [ ] 能区分 NPU latency 和 End-to-End latency
- [ ] 能查 CPU fallback / low-efficiency op
- [ ] 能测 core mask
- [ ] 能测 sustained performance
- [ ] 能记录版本信息

## Correctness

- [ ] 能建立 PyTorch baseline
- [ ] 能建立 ONNX Runtime baseline
- [ ] 能比较 RKNN FP
- [ ] 能比较 RKNN INT8
- [ ] 能使用 accuracy analysis
- [ ] 能定位 preprocess 问题
- [ ] 能定位 postprocess 问题
- [ ] 能定位 quantization 问题

## RK3576

- [ ] 能在 RK3576 跑一个 `.rknn`
- [ ] 能用 Lite2
- [ ] 能用 C API
- [ ] 能完成 INT8 model
- [ ] 能做 Zero Copy 基础实验
- [ ] 能做 RGA + NPU pipeline
- [ ] 能建立 Benchmark 表

## Compiler Mental Model

- [ ] 能解释 TVM 与 RKNN 的共同抽象
- [ ] 能解释为什么 `.rknn` 是 compiled artifact
- [ ] 能解释 GPU Kernel 优化思维如何迁移到 NPU
- [ ] 能解释为什么 NPU 更依赖 Vendor Compiler
- [ ] 能解释为什么 Graph Pattern 会影响 NPU 性能

---

# 324. 暂时不要求深入

暂时不要求：

- [ ] 逆向 `.rknn` 二进制格式
- [ ] 手写 RKNPU Machine Code
- [ ] 精通 RKNPU Kernel Driver
- [ ] 精通所有 RKNN Operator
- [ ] 自己实现 NPU Compiler
- [ ] 修改 RKNPU Firmware
- [ ] 精通所有 Custom Op
- [ ] 精通所有量化算法
- [ ] 自己实现 RKNN Runtime
- [ ] 精通所有 Rockchip SoC
- [ ] 把任意 Transformer 强行塞进普通 RKNN

先做到：

```text
会转换
会验证
会部署
会调试
会量化
会 Profile
会解释 Compiler / Runtime / NPU 边界。
```

---

# 325. 核心参考链接

## RKNN-Toolkit2

- [RKNN-Toolkit2 Repository](https://github.com/airockchip/rknn-toolkit2)
- [README](https://github.com/airockchip/rknn-toolkit2/blob/master/README.md)
- [CHANGELOG](https://github.com/airockchip/rknn-toolkit2/blob/master/CHANGELOG.md)
- [Toolkit2 Examples](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples)
- [Toolkit-Lite2 Examples](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/examples)
- [RKNPU2 Runtime](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime)
- [RKNPU2 Examples](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/examples)

---

## C API

- [rknn_api.h](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Android/librknn_api/include/rknn_api.h)
- [rknn_matmul_api.h](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/runtime/Android/librknn_api/include/rknn_matmul_api.h)

重点搜索：

```text
rknn_init
rknn_query
rknn_inputs_set
rknn_run
rknn_outputs_get
rknn_outputs_release
rknn_create_mem
rknn_create_mem_from_fd
rknn_set_io_mem
rknn_set_core_mask
rknn_set_input_shapes。
```

---

## Zero Copy

- [rknn_create_mem_demo.cpp](https://github.com/airockchip/rknn-toolkit2/blob/master/rknpu2/examples/rknn_api_demo/src/rknn_create_mem_demo.cpp)

重点：

```text
rknn_create_mem
↓
rknn_set_io_mem
↓
rknn_run。
```

---

## Dynamic Shape

- [Toolkit2 Dynamic Shape Example](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/dynamic_shape)
- [Lite2 Dynamic Shape Example](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/examples/dynamic_shape)

---

## Quantization

- [Hybrid Quant Example](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/hybrid_quant)
- [MMSE Quant Example](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/quantize_algorithm_mmse)
- [PyTorch QAT Example](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/pytorch/resnet18_qat)

---

## Custom Operator

- [Custom Op Examples](https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit2/examples/functions/custom_op)

---

## Connected Board Debug

- [rknn_server Proxy Guide](https://github.com/airockchip/rknn-toolkit2/blob/master/doc/rknn_server_proxy.md)

---

## Model Zoo

- [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
- [Model Zoo README](https://github.com/airockchip/rknn_model_zoo/blob/main/README.md)

学习重点：

```text
MobileNet
ResNet
YOLO
Segmentation
Transformer-related models。
```

---

## LLM

- [RKNN-LLM Repository](https://github.com/airockchip/rknn-llm)

下一章专门展开。

---

# 326. 与前面章节的映射

## 与 04_PyTorch与模型框架

```text
PyTorch
↓
ONNX
↓
RKNN Frontend。
```

---

## 与 06_LLM推理原理

```text
Quantization
Memory
MatMul
Attention
```

在 RKNPU 上：

```text
由 Vendor Compiler / Runtime
实现。
```

---

## 与 07_llama.cpp

```text
llama.cpp
=
Open Runtime
GGUF
CPU/GPU

RKNN
=
Vendor Compiler + Runtime
.rknn
RKNPU。
```

---

## 与 08_MLC-LLM

```text
MLC:
Model
↓
Relax
↓
Compile
↓
Model Library

RKNN:
Model
↓
Vendor Compiler
↓
.rknn。
```

---

## 与 09_AI_Compiler与TVM

```text
TVM terminology        RKNN view

Frontend           →   load_onnx
Target             →   rk3576
Graph Pass         →   build optimization
Quant              →   do_quantization
Lowering           →   NPU operator mapping
Artifact           →   .rknn
Runtime            →   librknnrt.so
Driver             →   RKNPU driver
Hardware           →   NPU。
```

---

## 与 10_GPU_Kernel与FlashAttention

```text
GPU:
Tiling
Shared Memory
Tensor Core
Data Movement

NPU:
Tiling
On-chip Buffer
Matrix Engine
DMA。
```

---

# 327. 本章最重要的最终大图

```text
                    PyTorch Model
                          │
                          ▼
                    ONNX Export
                          │
                          ▼
                 ONNX Runtime Verify
                          │
                          ▼
                    RKNN-Toolkit2
                          │
          ┌───────────────┼─────────────────┐
          ▼               ▼                 ▼
       Frontend       Graph Optimize     Quantization
          │               │                 │
          └───────────────┼─────────────────┘
                          ▼
                  Target = RK3576
                          │
                          ▼
                 NPU Operator Mapping
                          │
                          ▼
                   Target Compilation
                          │
                          ▼
                       .rknn
                          │
                 ┌────────┴────────┐
                 ▼                 ▼
          RKNN-Toolkit-Lite2    RKNN Runtime
             Python                C/C++
                 │                 │
                 └────────┬────────┘
                          ▼
                    RKNPU Driver
                          │
                          ▼
                       RKNPU
                          │
                          ▼
                 Tensor Compute Engine
```

---

# 328. 最终再记住这条系统链

```text
Model Correctness
↓
Graph Compatibility
↓
Quantization Accuracy
↓
NPU Mapping
↓
Runtime Integration
↓
Memory / Zero Copy
↓
Pre/Postprocess
↓
End-to-End Performance
```

---

# 329. 一句话总结

这一章最重要的不是：

```text
会把 ONNX 转成 .rknn。
```

而是理解：

```text
RKNN-Toolkit2
=
Rockchip Vendor AI Compiler

.rknn
=
面向 RKNPU 的部署 Artifact

RKNN Runtime
=
板端执行层

RKNPU Driver
=
内核驱动层

RKNPU
=
真正执行 Tensor Workload 的硬件。
```

最终整条链：

```text
PyTorch
↓
ONNX
↓
RKNN-Toolkit2
↓
Graph Optimization
↓
Quantization
↓
Operator Mapping
↓
Target Compilation
↓
.rknn
↓
RKNN Runtime
↓
RKNPU Driver
↓
NPU
```

而真正的端侧工程性能还要继续看：

```text
Camera
↓
RGA
↓
Zero Copy
↓
NPU
↓
Postprocess
↓
Application。
```

当你可以解释：

```text
为什么模型需要编译？

为什么 `.rknn` 不是 ONNX？

为什么 INT8 会影响 NPU 性能？

为什么 Layout 会影响性能？

为什么某个 unsupported op 会拖慢整个 Graph？

为什么 Zero Copy 不等于零数据搬运？

为什么同一块 RK3576，
换 Toolkit 版本性能可能变化？
```

那么就已经真正从：

```text
“会用 RKNN SDK”
```

进入：

```text
“理解 Rockchip NPU Compiler + Runtime + Hardware”
```

这一层。

下一章：

> **12_RKLLM与端侧LLM部署.md**

将进一步把本章的：

```text
Vendor NPU Compiler / Runtime
```

扩展到 LLM 特有问题：

```text
Tokenizer
↓
Prefill
↓
KV Cache
↓
Decode
↓
Sampling
↓
LLM Quantization
↓
RKLLM Runtime
↓
RKNPU。
```
