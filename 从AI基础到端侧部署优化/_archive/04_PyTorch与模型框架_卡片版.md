# 04_PyTorch与模型框架

> **所属路线**：AI 学习路线 · 第一部分  
> **定位**：理解主流 AI 框架如何定义模型、组织数据、完成训练、保存权重、导出计算图，并最终进入跨平台 Runtime 与端侧部署  
> **核心框架**：PyTorch  
> **辅助框架与模型表示**：TensorFlow / Keras、LiteRT（TensorFlow Lite 后继）、ONNX、ONNX Runtime、ExecuTorch  
> **学习边界**：本章重点解决“Framework → Model → Graph → Export → Runtime”的工程链路；量化算法、AI Compiler、GPU Kernel、RKNN/RKLLM 在后续章节深入。  
> **资料检查日期**：2026-08-25

---

# 0. 本章学习目标

完成本章以后，需要能够回答：

1. PyTorch、TensorFlow、Keras、ONNX、ONNX Runtime、LiteRT 到底分别是什么？
2. “训练框架”“模型文件格式”“中间表示 IR”“推理 Runtime”为什么不能混为一谈？
3. PyTorch 的 Tensor、`nn.Module`、Parameter、Autograd、Optimizer、Dataset/DataLoader 分别处于什么位置？
4. `state_dict`、`.pt/.pth`、`.safetensors`、`.onnx`、`.tflite` 各自保存的到底是什么？
5. PyTorch Eager Mode 和 Graph / Compile 模式有什么区别？
6. `torch.compile()` 在优化什么？
7. `torch.export()` 为什么对部署比简单保存 Python 模型更重要？
8. TensorFlow 中 Eager、Graph、`tf.function`、Keras 分别是什么关系？
9. Keras 3 为什么已经不再等于 TensorFlow 专属前端？
10. TensorFlow Lite 与当前 LiteRT 是什么关系？
11. `.tflite` 文件是什么？
12. ONNX 为什么被称为“开放的模型中间表示”？
13. ONNX 的 Model、Graph、Node、Initializer、Tensor、Operator、Opset 分别是什么？
14. ONNX Runtime 为什么不是 ONNX 本身？
15. Execution Provider 是什么？
16. 为什么 ONNX Runtime 能把一部分 Operator 放 GPU，另一部分回退 CPU？
17. 为什么“模型能成功导出 ONNX”不等于“目标硬件一定能跑”？
18. 为什么动态 Shape、Unsupported Operator、Custom Operator 经常导致模型转换失败？
19. 为什么模型转换前后需要做 Numerical Validation？
20. 为什么预处理 / 后处理不一致会让“同一个模型”输出完全不同？
21. PyTorch → ONNX → ONNX Runtime 是什么链？
22. PyTorch → `torch.export` → ExecuTorch 是什么链？
23. TensorFlow / Keras → LiteRT 是什么链？
24. PyTorch → LiteRT 为什么现在也可以成为端侧路线？
25. PyTorch / ONNX → RKNN-Toolkit2 → `.rknn` 又处于什么位置？
26. 为什么理解这些框架，是后续学习 TVM、RKNN、AI Compiler 的前提？

本章最终需要建立的核心工程视图：

```text
        Training Framework
              │
              │
      PyTorch / TensorFlow
              │
              ▼
       Model Definition
              │
              ▼
          Training
              │
              ▼
       Learned Weights
              │
              ▼
      Export / Capture Graph
              │
      ┌───────┼────────┐
      ▼       ▼        ▼
    ONNX   torch.export  LiteRT
      │       │        │
      ▼       ▼        ▼
   Runtime  ExecuTorch LiteRT Runtime
      │       │        │
      └───────┼────────┘
              ▼
      CPU / GPU / NPU
```

---

# 1. 首先必须分清：框架、格式、Runtime、Compiler

这是本章最重要的一件事。

很多初学者会把：

```text
PyTorch
ONNX
ONNX Runtime
TensorFlow Lite
RKNN
GGUF
```

都叫作：

> “模型格式”或者“AI 框架”。

这是不准确的。

先建立四层。

---

# 2. Training Framework

主要负责：

```text
Model Definition
Data Pipeline
Forward
Autograd
Loss
Optimizer
Training
Distributed Training
```

典型：

```text
PyTorch
TensorFlow
JAX
```

它们解决的是：

> **如何表达和训练一个 AI 模型。**

---

# 3. Model Representation / File Format

主要负责：

> 把 Architecture / Graph / Weight / Metadata 以某种格式保存。

典型：

```text
PyTorch state_dict
.safetensors
ONNX
.tflite
GGUF
.rknn
```

但是这些格式的抽象层不同。

例如：

```text
state_dict
```

主要是：

```text
Weight Tensor
```

而：

```text
ONNX
```

包含：

```text
Graph
Operator
Weight
Input / Output
Shape / Type
Metadata
```

所以二者不能简单等价。

---

# 4. Runtime

Runtime 负责：

> **真正执行已经准备好的模型。**

例如：

```text
PyTorch Runtime
ONNX Runtime
LiteRT Runtime
ExecuTorch Runtime
RKNN Runtime
llama.cpp
```

Runtime 会负责：

```text
Load Model
Allocate Tensor
Schedule Operator
Select Kernel
Execute
Manage Memory
Return Output
```

---

# 5. Compiler / Converter

Compiler / Converter 负责：

```text
Framework Model
      ↓
Graph Capture
      ↓
Operator Conversion
      ↓
Graph Optimization
      ↓
Hardware Mapping
      ↓
Target Model
```

例如：

```text
torch.export
ONNX Exporter
LiteRT Converter
RKNN-Toolkit2
TVM
```

后面会发现：

> Runtime、Compiler、Model Format 是完全不同的角色。

---

# 6. 一张完整的软件栈图

```text
Python Application
       ↓
Training Framework
       ↓
PyTorch / TensorFlow
       ↓
Model Architecture
       ↓
Weights
       ↓
Graph Capture / Export
       ↓
Intermediate Representation
       ↓
Graph Optimization
       ↓
Operator Mapping
       ↓
Runtime
       ↓
Kernel
       ↓
CPU / GPU / NPU
```

这张图会贯穿后面所有章节。

---

# 7. 本章核心 GitHub 仓库

## PyTorch

GitHub：

- [pytorch/pytorch](https://github.com/pytorch/pytorch)

官方文档：

- [PyTorch Docs](https://docs.pytorch.org/)
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
- [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [`torch.export`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/api_reference.html)

---

## TensorFlow

GitHub：

- [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)

官方：

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

---

## Keras

GitHub：

- [keras-team/keras](https://github.com/keras-team/keras)

官方：

- [Keras](https://keras.io/)

需要注意：

> Keras 3 已经是 multi-backend API，可以使用 TensorFlow、JAX、PyTorch 后端。

因此现在：

```text
Keras
≠
TensorFlow 专属
```

---

## LiteRT

当前官方 GitHub：

- [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT)

官方文档：

- [Google LiteRT](https://developers.google.com/edge/litert)
- [LiteRT Samples](https://github.com/google-ai-edge/litert-samples)

LiteRT 是：

> TensorFlow Lite 的后继端侧 Runtime / Toolchain。

当前仍然广泛使用：

```text
.tflite
```

作为模型文件。

---

## ONNX

GitHub：

- [onnx/onnx](https://github.com/onnx/onnx)

官方：

- [ONNX Documentation](https://onnx.ai/)
- [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html)
- [ONNX IR Specification](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX Operators](https://onnx.ai/onnx/operators/)

---

## ONNX Runtime

GitHub：

- [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)

官方：

- [ONNX Runtime](https://onnxruntime.ai/)
- [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)

---

## ExecuTorch

GitHub：

- [pytorch/executorch](https://github.com/pytorch/executorch)

官方：

- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)

它负责：

> PyTorch 面向 mobile / embedded / edge 的轻量化部署。

---

# 8. 本章为什么以 PyTorch 为主？

因为在现代 AI / LLM 生态中：

```text
Research
Open-source LLM
Hugging Face
Fine-tuning
AI System Research
```

大量工作都以 PyTorch 为核心。

我们的路线中：

```text
MiniMind
LLMs-from-scratch
Transformers
Qwen
LLM Training
```

也主要是 PyTorch。

因此：

> **PyTorch 需要真正掌握。**

而：

```text
TensorFlow / Keras
```

需要做到：

> 看得懂、会基本训练、理解模型表示和端侧导出。

---

# 9. PyTorch 的核心对象

可以把 PyTorch 先分成：

```text
PyTorch
│
├── Tensor
├── Autograd
├── nn.Module
├── Dataset / DataLoader
├── Optimizer
├── Device
├── Serialization
├── Compile
└── Export
```

---

# 10. Tensor

PyTorch 的计算基础：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0])
```

Tensor 最重要的属性：

```text
shape
dtype
device
requires_grad
```

例如：

```python
print(x.shape)
print(x.dtype)
print(x.device)
```

---

# 11. Tensor = Data + Metadata

一个 Tensor 可以先理解成：

```text
Data Buffer
+
Shape
+
Stride
+
Dtype
+
Device
```

例如：

```text
shape:
[32, 3, 224, 224]

dtype:
float32

device:
cuda:0
```

这套信息后面到了：

```text
ONNX Tensor
RKNN Tensor
GPU Tensor
NPU Tensor
```

仍然存在。

---

# 12. Dtype

常见：

```text
float32
float16
bfloat16
int64
int32
int8
bool
```

在 Training：

```text
FP32
FP16
BF16
```

非常常见。

在 Inference：

```text
FP32
FP16
INT8
INT4
```

更常见。

---

# 13. Device

Tensor 的 Device：

```text
CPU
CUDA GPU
MPS
...
```

例如：

```python
device = torch.device("cuda")

x = x.to(device)
model = model.to(device)
```

核心：

> Tensor 真正存储和执行的位置发生了变化。

---

# 14. Tensor Shape 是模型工程的第一语言

对于任何模型：

```text
先追 Shape
```

MLP：

```text
[B, D]
```

CNN：

```text
[B, C, H, W]
```

Transformer：

```text
[B, T, D]
```

Deployment：

```text
Input Shape
Output Shape
Dynamic Axis
Layout
```

都是核心问题。

---

# 15. Stride

Stride 描述：

> Tensor 每个维度在底层内存中的跨步。

例如：

```text
Tensor Shape
≠
Memory Layout
```

这会在后面：

```text
contiguous
transpose
NCHW
NHWC
Kernel Optimization
```

中变得非常重要。

---

# 16. `contiguous()`

例如：

```python
y = x.transpose(1, 2)
```

可能只改变：

```text
View + Stride
```

并没有真正重新排列数据。

某些 Kernel 需要连续内存：

```python
y = y.contiguous()
```

这个概念后面会直接进入性能优化。

---

# 17. Autograd

上一章已经通过 micrograd 理解：

```text
Computational Graph
↓
Backward
↓
Gradient
```

PyTorch：

```python
x = torch.tensor(
    2.0,
    requires_grad=True
)

y = x ** 2

y.backward()

print(x.grad)
```

输出：

```text
4
```

因为：

```text
dy/dx = 2x
```

---

# 18. PyTorch Autograd 的工程实现

官方 Autograd 文档：

- [Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

概念上：

```text
Forward Operation
       ↓
Build Dynamic Graph
       ↓
grad_fn
       ↓
Backward
       ↓
Chain Rule
       ↓
Parameter.grad
```

也就是上一章 micrograd 的工业级版本。

---

# 19. `requires_grad`

如果：

```python
x.requires_grad = True
```

PyTorch 会记录：

> 对 x 后续进行的、与梯度相关的 Operation。

模型 Parameter：

```text
默认 requires_grad=True
```

---

# 20. `no_grad` 与 `inference_mode`

Inference 时不需要：

```text
Backward Graph
```

常见：

```python
with torch.no_grad():
    y = model(x)
```

PyTorch 还提供：

```python
torch.inference_mode()
```

用于更彻底的 inference 场景优化。

核心思想：

```text
Training
需要 Graph

Inference
通常不需要 Gradient Graph
```

---

# 21. `nn.Module`

PyTorch 所有模型组件最重要的抽象：

```python
class MyModel(nn.Module):
```

Module 可以包含：

```text
Parameter
Buffer
Submodule
Forward
Hooks
Device State
Training / Eval State
```

官方：

- [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)

---

# 22. 最小 Module

```python
import torch.nn as nn

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
```

---

# 23. `__init__()` 和 `forward()`

`__init__`：

```text
定义模型有哪些组件
```

`forward`：

```text
定义 Tensor 如何流过模型
```

例如：

```text
x
↓
fc1
↓
relu
↓
fc2
↓
output
```

---

# 24. `model(x)` 为什么会调用 forward？

正常不直接：

```python
model.forward(x)
```

而：

```python
model(x)
```

因为 `nn.Module.__call__` 还会处理：

```text
Hooks
Autograd
Compile-related logic
```

然后再调用：

```text
forward()
```

---

# 25. Parameter

模型 Weight：

```python
self.weight
```

通常是：

```text
nn.Parameter
```

Parameter 是特殊 Tensor：

> 被 Module 注册后，会被自动识别为可训练参数。

---

# 26. `model.parameters()`

```python
for p in model.parameters():
    print(p.shape)
```

Optimizer：

```python
optimizer = torch.optim.Adam(
    model.parameters()
)
```

就是通过这一机制找到：

```text
所有可训练 Parameter
```

---

# 27. Buffer

不是所有模型状态都需要 Gradient。

例如 BatchNorm：

```text
running_mean
running_var
```

这类数据：

```text
是模型状态
但不是 trainable parameter
```

PyTorch 使用：

```text
Buffer
```

管理。

---

# 28. `state_dict`

```python
model.state_dict()
```

包含：

```text
Parameters
+
Persistent Buffers
```

因此：

```text
state_dict
```

更准确的理解是：

> 模型的可序列化状态。

---

# 29. `state_dict` 不等于完整 Python 模型

假设：

```python
class Net(nn.Module):
    ...
```

保存：

```python
torch.save(
    model.state_dict(),
    "model.pth"
)
```

文件并没有完整表达：

```text
Python class 的全部执行逻辑
```

加载时通常仍然需要：

```python
model = Net()
model.load_state_dict(...)
```

所以：

```text
Architecture Code
+
state_dict
=
可恢复 PyTorch Model
```

---

# 30. `.pt` / `.pth`

它们只是：

> 常用文件后缀。

并没有严格规定：

```text
.pt 一定是什么
.pth 一定是什么
```

里面可能是：

```text
state_dict
checkpoint
整个 Python Object
```

所以不要只看扩展名判断文件语义。

---

# 31. Checkpoint

Training 时常保存：

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "loss": ...
}
```

这样：

```text
Training
```

可以中断后恢复。

因此：

```text
Checkpoint
```

通常比纯模型 Weight 包含更多训练状态。

---

# 32. Safetensors

现代 Hugging Face 生态中经常：

```text
.safetensors
```

主要用于：

> 安全、高效地保存 Tensor。

它重点保存：

```text
Weight Tensor
```

而不是 Python 执行逻辑。

所以：

```text
config.json
+
model.safetensors
+
model code / known architecture
```

才能共同恢复模型。

---

# 33. Data Pipeline

PyTorch：

```text
Dataset
+
DataLoader
```

负责：

```text
Load
Transform
Shuffle
Batch
Parallel Data Loading
```

---

# 34. Dataset

经典接口：

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...
```

核心：

```text
index
→
one sample
```

---

# 35. DataLoader

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

负责：

```text
Batch
Shuffle
Multi-process loading
```

---

# 36. Dataset 和 Model 是两个独立系统

必须建立：

```text
Data Pipeline
       ↓
Batch Tensor
       ↓
Model
```

模型本身一般不负责：

```text
从硬盘读图片
```

这是 Data Pipeline 的工作。

---

# 37. Optimizer

例如：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)
```

主要维护：

```text
Parameter
Gradient
Optimizer State
```

然后：

```python
optimizer.step()
```

更新 Weight。

---

# 38. 一个完整 PyTorch Training Loop

```python
for epoch in range(epochs):

    model.train()

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = loss_fn(logits, y)

        loss.backward()

        optimizer.step()
```

这一段已经在上一章理解。

本章要增加工程视角：

```text
Python
 ↓
PyTorch Dispatcher
 ↓
ATen Operator
 ↓
CPU / CUDA Kernel
```

---

# 39. PyTorch Eager Mode

PyTorch 最经典的模式：

```text
一条 Python 语句
↓
马上执行对应 Operation
```

例如：

```python
a = x @ w
b = torch.relu(a)
c = b + residual
```

基本按照 Python 控制流立即执行。

这叫：

```text
Eager Execution
```

---

# 40. Eager Mode 的优点

```text
直观
Pythonic
容易 Debug
动态控制流方便
Research 开发效率高
```

这是 PyTorch 成功的重要原因。

---

# 41. Eager Mode 的问题

如果每个 Operator：

```text
都由 Python 单独调度
```

可能产生：

```text
Python Overhead
Kernel Launch Overhead
难以跨 Operator 优化
```

于是现代 PyTorch 引入：

```text
torch.compile
```

---

# 42. `torch.compile()`

官方：

- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch 2.x](https://pytorch.org/get-started/pytorch-2-x/)

基本：

```python
compiled_model = torch.compile(model)
```

目标：

> 在保持 PyTorch 编程体验的同时，捕获可优化区域并生成更高效实现。

---

# 43. PyTorch 2 Compile Stack

PyTorch 2 系列核心技术常见：

```text
TorchDynamo
↓
Graph Capture

AOTAutograd
↓
Ahead-of-time Autograd

PrimTorch
↓
Operator Decomposition

TorchInductor
↓
Compiler Backend
```

最终：

```text
CPU / GPU code
```

---

# 44. `torch.compile` 不等于模型导出

这是很重要的区别。

`torch.compile`：

```text
主要优化当前 PyTorch 程序执行
```

而：

```text
torch.export
```

主要目标：

> 获取可以 Ahead-of-Time 使用的规范化 Tensor Graph。

---

# 45. `torch.export`

官方：

- [`torch.export` API](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/api_reference.html)

概念：

```text
nn.Module
+
Example Inputs
       ↓
torch.export
       ↓
ExportedProgram
       ↓
Normalized Tensor Graph
```

其中会尽量：

```text
消除 Python 控制流
规范化 Operator
记录 Shape Constraint
```

这对：

```text
Compiler
Edge Runtime
Backend
```

更加友好。

---

# 46. 为什么部署喜欢 Graph？

Python Model：

```text
可以动态 if
for
class
list
dict
runtime object
```

硬件 Compiler 更喜欢：

```text
明确的 Operator Graph
明确 Input
明确 Output
明确 Shape Constraint
```

因此：

```text
Python Program
      ↓
Graph Capture
      ↓
Deployment
```

是一条非常核心的路线。

---

# 47. Eager vs Graph

Eager：

```text
Python
↓
Operator
↓
Execute
```

Graph：

```text
Python / Model
↓
Capture
↓
Graph
↓
Analyze Whole Graph
↓
Optimize
↓
Execute
```

Graph 的优势：

```text
Operator Fusion
Constant Folding
Dead Code Elimination
Memory Planning
Hardware Partitioning
Code Generation
```

---

# 48. Graph 是后续 AI Compiler 的入口

后面的：

```text
ONNX
TVM
RKNN
TensorRT
ExecuTorch
```

都极度依赖：

```text
Graph
Operator
Tensor
Shape
```

因此：

> 从 `nn.Module` 走向 Graph，是模型工程到 AI System 的分界点。

---

# 49. TensorFlow 的核心思想

TensorFlow：

- [tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)

历史上 TensorFlow 最典型的思想就是：

```text
Tensor
+
Computation Graph
```

TensorFlow 2 默认也支持：

```text
Eager Execution
```

同时通过：

```text
tf.function
```

将 Python Function 转换为可优化 Graph。

---

# 50. TensorFlow Eager

TensorFlow 2：

```python
import tensorflow as tf

x = tf.constant([1.0, 2.0])

y = x * 2
```

默认直接执行。

所以今天：

```text
PyTorch
TensorFlow
```

在“开发体验”上已经没有早期那么极端的差异。

---

# 51. `tf.function`

概念：

```python
@tf.function
def f(x):
    return x * 2
```

TensorFlow 会尝试：

```text
Trace Python Function
↓
Build Graph
↓
Optimize
↓
Execute
```

这和：

```text
Eager → Graph
```

思想一致。

---

# 52. Keras

Keras：

```text
High-level Neural Network API
```

帮助开发者用更统一的方式：

```text
Build
Train
Evaluate
Save
```

模型。

---

# 53. Keras 3 的重要变化

过去：

```text
Keras
≈
TensorFlow high-level API
```

现在 Keras 3：

```text
Keras API
      ↓
TensorFlow Backend
or
JAX Backend
or
PyTorch Backend
```

所以：

> Keras 已经更像一个 multi-backend Model API。

---

# 54. Keras Sequential

适合简单线性堆叠：

```python
model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10)
])
```

适合：

```text
Layer 1
↓
Layer 2
↓
Layer 3
```

---

# 55. Keras Functional API

复杂 Graph：

```text
Multiple Inputs
Skip Connection
Multiple Outputs
Branch
```

适合 Functional API。

概念：

```python
inputs = keras.Input(...)

x = layers.Dense(...)(inputs)

outputs = layers.Dense(...)(x)

model = keras.Model(inputs, outputs)
```

它本身就非常 Graph-oriented。

---

# 56. Keras Subclassing

如果需要高度自定义：

```python
class MyModel(keras.Model):
    ...
```

和：

```text
PyTorch nn.Module
```

在思想上很接近。

---

# 57. TensorFlow / Keras Model Save

需要区分：

```text
Training-friendly model archive
```

和：

```text
Deployment artifact
```

Keras 3 自身有：

```text
.keras
```

模型格式。

TensorFlow 生态还长期使用：

```text
SavedModel
```

作为图和函数导出的重要格式。

---

# 58. SavedModel

概念上可以包含：

```text
Graph / Function
Variables
Assets
Signatures
```

因此比：

```text
纯 Weight
```

更适合部署 / Serving。

---

# 59. 为什么 TensorFlow 现在不是本路线主框架？

不是因为 TensorFlow 没用。

而是本路线后续：

```text
LLM
Hugging Face
MiniMind
Qwen
LoRA
AI Systems
```

更偏 PyTorch。

TensorFlow 本章目标：

```text
理解框架思想
能看懂 Keras
能基本训练
理解 SavedModel / LiteRT 导出
```

即可。

---

# 60. TensorFlow Lite 现在应该怎么理解？

过去：

```text
TensorFlow
↓
TFLite Converter
↓
.tflite
↓
TFLite Interpreter
↓
Mobile / Embedded
```

这条链仍然大量存在。

但是现在 Google 已将整体品牌和技术路线升级为：

```text
LiteRT
```

---

# 61. LiteRT

官方：

- [LiteRT](https://developers.google.com/edge/litert)
- [google-ai-edge/LiteRT](https://github.com/google-ai-edge/LiteRT)

当前官方描述：

> LiteRT 是 Google 面向 edge platform 的高性能 on-device ML / GenAI framework，并建立在 TensorFlow Lite 的成熟基础上。

所以：

```text
TensorFlow Lite
→
LiteRT
```

应该理解成：

> 生态延续与升级，而不是完全推倒重来。

---

# 62. `.tflite`

LiteRT 目前仍然支持：

```text
.tflite
```

作为非常重要的部署模型格式。

概念：

```text
Framework Model
       ↓
Conversion
       ↓
FlatBuffer-based model
       ↓
.tflite
       ↓
LiteRT Runtime
```

---

# 63. LiteRT 已经不只支持 TensorFlow

当前官方路线支持：

```text
TensorFlow
PyTorch
JAX
```

模型转换。

因此不要再形成：

```text
TFLite
=
只有 TensorFlow 模型才能用
```

这种旧认知。

---

# 64. LiteRT 当前两类重要 API 思想

当前生态中可以重点认识：

```text
Interpreter API
```

以及新的：

```text
CompiledModel API
```

---

# 65. Interpreter API

经典 TensorFlow Lite 风格：

```text
Load .tflite
↓
Allocate Tensors
↓
Set Input
↓
Invoke
↓
Get Output
```

这就是传统轻量 Runtime 模式。

---

# 66. CompiledModel API

当前 LiteRT V2 强调：

```text
Automatic Accelerator Selection
Async Execution
NPU Distribution
Efficient I/O Buffer Handling
```

也就是：

> Runtime 进一步承担硬件选择、编译和调度职责。

这和后面：

```text
Execution Provider
Delegate
Backend
```

思路会越来越接近。

---

# 67. LiteRT Hardware

当前 LiteRT 面向：

```text
CPU
GPU
NPU
Embedded / IoT
Android
Desktop
Web
```

所以它本质是一个：

> Cross-platform Edge AI Runtime / Toolchain。

---

# 68. LiteRT 与 RKNN 的类比

LiteRT：

```text
General Model
↓
Convert / Optimize
↓
.tflite
↓
LiteRT Runtime
↓
CPU / GPU / Supported NPU
```

RKNN：

```text
ONNX / Framework Model
↓
RKNN-Toolkit2
↓
.rknn
↓
RKNN Runtime
↓
Rockchip NPU
```

从抽象层次上：

> 二者都在解决“Framework Model → Edge Hardware”的问题。

---

# 69. ONNX 是什么？

ONNX：

```text
Open Neural Network Exchange
```

官方：

- [onnx/onnx](https://github.com/onnx/onnx)
- [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html)
- [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html)

最重要的理解：

> **ONNX 是开放的模型表示规范，而不是一个训练框架。**

---

# 70. ONNX 为什么存在？

假设：

```text
PyTorch Model
```

只能 PyTorch 读取。

目标设备：

```text
NVIDIA TensorRT
Intel OpenVINO
ONNX Runtime
RKNN Converter
```

都需要理解 PyTorch 内部所有 Python 语义，会非常麻烦。

所以加入一个中间层：

```text
PyTorch
      ↓
     ONNX
      ↓
Different Runtime / Compiler
```

---

# 71. ONNX 类似一种 AI IR

可以粗略类比：

```text
C/C++
↓
Compiler IR
↓
Machine Code
```

AI：

```text
PyTorch / TensorFlow
↓
ONNX
↓
Runtime / Compiler
↓
CPU / GPU / NPU
```

严格来说：

> ONNX 不是所有 AI Compiler 唯一的 IR。

但作为开放交换格式非常重要。

---

# 72. ONNX Model 的核心组成

官方 IR：

```text
Model
│
├── Graph
├── Opset Import
├── IR Version
├── Metadata
└── Functions
```

Graph：

```text
Graph
│
├── Input
├── Output
├── Node
├── Initializer
└── Value Info
```

---

# 73. ModelProto

ONNX 顶层：

```text
ModelProto
```

负责：

```text
Graph
IR Version
Opset
Producer
Metadata
```

---

# 74. GraphProto

Graph：

> 实际计算数据流。

例如：

```text
Input
 ↓
MatMul
 ↓
Add
 ↓
ReLU
 ↓
Output
```

---

# 75. Node

ONNX 中：

```text
Node
```

通常表示一次：

```text
Operator Call
```

例如：

```text
Conv
MatMul
Add
Relu
Softmax
Reshape
```

---

# 76. Operator

Operator 是：

> 已定义语义的计算操作。

例如：

```text
MatMul
```

官方会定义：

```text
Input
Output
Type
Attributes
Version
Semantics
```

完整列表：

- [ONNX Operators](https://onnx.ai/onnx/operators/)

---

# 77. Node 和 Operator 区别

可以理解：

```text
Operator
=
函数定义 / 规范

Node
=
Graph 中某一次函数调用
```

例如：

```text
Operator:
Conv

Graph:
conv1 Node
conv2 Node
conv3 Node
```

三个 Node 都调用：

```text
Conv Operator
```

---

# 78. Initializer

Initializer 常用于：

```text
Weight
Bias
Constant Tensor
```

例如 Linear：

```text
X
+
W
+
B
```

其中：

```text
W
B
```

通常就是 Graph Initializer。

---

# 79. 一个 Linear 在 ONNX 中是什么？

PyTorch：

```python
nn.Linear(...)
```

ONNX 可能表示成：

```text
MatMul
+
Add
```

或某些情况下：

```text
Gemm
```

所以：

> Framework Layer 不一定一一对应一个 ONNX Node。

这非常重要。

---

# 80. Layer ≠ Operator

PyTorch：

```text
nn.MultiheadAttention
```

可能导出成：

```text
MatMul
Add
Reshape
Transpose
Softmax
MatMul
...
```

所以：

```text
High-level Layer
↓
Many Primitive Operators
```

---

# 81. Input / Output

ONNX Graph 会明确：

```text
Input Name
Input Type
Input Shape

Output Name
Output Type
Output Shape
```

这对于 Runtime 非常关键。

---

# 82. Dynamic Shape

例如：

```text
Batch Size
```

可能不是固定：

```text
1
```

而是：

```text
N
```

Sequence Length：

```text
T
```

也可能动态变化。

这就是：

```text
Dynamic Shape
```

---

# 83. 为什么 Dynamic Shape 让部署更复杂？

固定：

```text
[1,3,224,224]
```

Compiler 可以提前：

```text
Memory Plan
Kernel Selection
Optimization
```

动态：

```text
[B,3,H,W]
```

就需要考虑：

```text
不同 Shape
不同 Workspace
不同 Kernel
不同 Memory
```

所以 NPU 常常更喜欢：

```text
Static Shape
```

---

# 84. Opset

ONNX 里非常关键：

```text
Opset Version
```

它不是：

```text
ONNX 软件版本
```

而是：

> 模型所依赖 Operator 语义集合的版本。

---

# 85. 为什么 Opset 很重要？

例如：

```text
Resize
Slice
Softmax
```

不同 Opset 版本：

```text
Input
Attribute
Behavior
```

可能有变化。

所以某个 Runtime：

```text
只支持到某些 Opset / Operator Version
```

导出版本过新：

```text
可能无法加载
```

---

# 86. ONNX IR Version vs Opset Version

必须分清：

```text
IR Version
=
ONNX 模型结构格式版本

Opset Version
=
Operator 规范集合版本
```

它们是：

```text
两套独立版本系统
```

---

# 87. ONNX Checker

导出后：

```python
import onnx

model = onnx.load("model.onnx")

onnx.checker.check_model(model)
```

可以进行结构合法性检查。

但：

> Checker 通过也不代表目标 Runtime 一定支持全部 Operator。

---

# 88. Shape Inference

ONNX 可以尝试：

```text
推导中间 Tensor Shape
```

对于：

```text
Debug
Compiler
Visualization
```

很有价值。

---

# 89. Netron

非常推荐：

- [Netron](https://github.com/lutzroeder/netron)

它可以可视化：

```text
ONNX
TFLite
TorchScript
很多模型格式
```

看到：

```text
Graph
Operators
Tensor Shape
Weight
```

对于后面的：

```text
RKNN
TVM
```

学习非常有帮助。

---

# 90. PyTorch 导出 ONNX

当前 PyTorch 生态支持将模型导出：

```text
PyTorch
↓
ONNX
```

概念：

```python
torch.onnx.export(
    model,
    example_input,
    "model.onnx"
)
```

具体参数随 PyTorch 版本变化，应以当前官方文档为准。

---

# 91. 为什么需要 Example Input？

Exporter 需要：

```text
实际 Tensor Shape
dtype
执行路径
```

帮助捕获模型。

特别是动态 Python Model：

```text
不同输入
可能走不同分支
```

Graph Exporter 必须明确：

> 哪些行为可以被转换成静态 / 受约束 Graph。

---

# 92. Export 的本质

不是：

```text
把 .pth 改扩展名成 .onnx
```

而是：

```text
Python Model Semantics
       ↓
Graph Capture
       ↓
Operator Mapping
       ↓
Weight Serialization
       ↓
ONNX Model
```

所以一定可能失败。

---

# 93. Unsupported Operator

例如 PyTorch 模型使用：

```text
Custom CUDA Op
特殊 Python Logic
新 Operator
```

ONNX Standard 中没有对应语义：

```text
Exporter
无法转换
```

于是：

```text
Export Failed
```

或者需要：

```text
Custom Operator
```

---

# 94. Control Flow

Python：

```python
if x.sum() > 0:
    ...
else:
    ...
```

对于 Graph Export：

```text
可能很难静态表达
```

现代 Export 工具已经支持越来越多 Control Flow，但仍然需要：

```text
明确可捕获的图语义
```

---

# 95. ONNX Runtime 是什么？

ONNX：

```text
Model Specification
```

ONNX Runtime：

```text
Execution Engine
```

二者关系：

```text
model.onnx
      ↓
ONNX Runtime
      ↓
Kernel
      ↓
Hardware
```

---

# 96. ONNX Runtime InferenceSession

核心：

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx"
)

output = session.run(
    None,
    {
        "input": input_array
    }
)
```

核心类：

```text
InferenceSession
```

---

# 97. ONNX Runtime 内部视角

```text
ONNX Graph
    ↓
Graph Optimization
    ↓
Execution Provider Partition
    ↓
Operator Kernel Selection
    ↓
Memory Planning
    ↓
Execute
```

---

# 98. Execution Provider

这是 ONNX Runtime 最重要的概念之一。

Execution Provider：

> 给某种 Hardware / Accelerator 提供 Operator Kernel 和执行能力。

例如：

```text
CPUExecutionProvider
CUDAExecutionProvider
TensorRTExecutionProvider
OpenVINOExecutionProvider
CoreMLExecutionProvider
QNNExecutionProvider
```

具体支持列表以当前 ORT 文档为准。

---

# 99. Execution Provider 不等于 GPU Driver

它更像：

```text
ONNX Runtime
    ↓
Hardware Backend Adapter
    ↓
Optimized Kernels / SDK
    ↓
Hardware
```

---

# 100. EP Partition

假设 Graph：

```text
Conv
↓
CustomOp
↓
Relu
```

CUDA EP：

```text
支持 Conv
支持 Relu
不支持 CustomOp
```

那么可能：

```text
Conv → GPU
CustomOp → CPU
Relu → GPU
```

---

# 101. CPU Fallback

Fallback 可以让模型：

```text
能跑
```

但可能：

```text
GPU ↔ CPU Copy
```

非常频繁。

结果：

> 性能反而极差。

所以：

```text
“GPU EP 已启用”
```

不等于：

```text
“整个模型都在 GPU 上跑”
```

---

# 102. 这是以后 NPU 部署极其常见的问题

例如：

```text
100 个 Operator
```

其中：

```text
95 个 NPU
5 个 CPU
```

如果 5 个 CPU Op 导致：

```text
频繁 Tensor Copy
同步
Layout Conversion
```

可能严重拖慢模型。

所以：

> Operator Coverage 比“理论 TOPS”重要得多。

---

# 103. Graph Optimization

Runtime 可以做：

```text
Constant Folding
Operator Fusion
Redundant Node Elimination
Layout Optimization
```

例如：

```text
Conv
+
BatchNorm
```

可能融合。

或者：

```text
MatMul
+
Add
```

优化成更合适的 Kernel。

---

# 104. Runtime 和 Compiler 的边界不是绝对的

现代 Runtime：

```text
ONNX Runtime
LiteRT
TensorRT
```

往往内部也做：

```text
Graph Compilation
Fusion
Kernel Selection
Hardware Partition
```

所以：

> Runtime 经常包含 Compiler 成分。

后面 TVM 会更系统地讨论。

---

# 105. ONNX Runtime 的 I/O Binding

如果模型跑在 GPU：

默认：

```text
NumPy CPU Input
↓
Copy to GPU
↓
Inference
↓
Copy Output to CPU
```

可能产生额外 Copy。

ORT 提供：

```text
IOBinding
```

可以让 Tensor 更直接地留在 Device。

这反映一个重要性能原则：

> **数据搬运也是推理成本。**

---

# 106. 框架性能经常不是纯 Compute 问题

模型执行时间可能包括：

```text
Python Overhead
Tensor Allocation
Memory Copy
Layout Conversion
Kernel Launch
Synchronization
Compute
```

所以后面优化：

```text
不仅看 FLOPs
```

---

# 107. ExecuTorch

ExecuTorch：

- [pytorch/executorch](https://github.com/pytorch/executorch)

它解决：

```text
PyTorch Model
       ↓
Export
       ↓
Edge Representation
       ↓
Delegate / Backend
       ↓
Lightweight Runtime
       ↓
Mobile / Embedded
```

---

# 108. ExecuTorch 为什么重要？

因为：

```text
PyTorch
```

本身是很大的训练 / 开发框架。

嵌入式设备不希望部署完整 PyTorch Python 环境。

所以：

```text
Training Framework
↓
Export
↓
Small Edge Runtime
```

是很自然的路线。

---

# 109. `torch.export` 与 ExecuTorch

抽象链：

```text
nn.Module
   ↓
torch.export
   ↓
ExportedProgram
   ↓
ExecuTorch Lowering
   ↓
.pte
   ↓
ExecuTorch Runtime
```

所以：

```text
torch.export
```

正在成为 PyTorch AOT / Edge 生态的重要桥梁。

---

# 110. `.pte`

ExecuTorch 的部署产物之一：

```text
.pte
```

不要混淆：

```text
.pt
.pth
.pte
```

它们语义完全不同。

---

# 111. 模型文件格式总表

| 格式 | 主要角色 |
|---|---|
| `.pt/.pth` | PyTorch 常用序列化 / checkpoint 后缀 |
| `.safetensors` | Tensor Weight 存储 |
| `.keras` | Keras 模型归档 |
| SavedModel | TensorFlow graph/functions + variables 等 |
| `.onnx` | 开放计算图 / IR 模型格式 |
| `.tflite` | LiteRT / TFLite 端侧部署格式 |
| `.pte` | ExecuTorch 部署程序 |
| `.gguf` | llama.cpp 生态 LLM 模型格式 |
| `.rknn` | Rockchip NPU 编译部署模型 |

这张表需要牢记：

> **文件后缀背后的抽象层完全不同。**

---

# 112. 一个“模型”究竟包含什么？

讨论 Model 时最好拆成：

```text
Architecture
+
Weights
+
Tokenizer / Preprocess
+
Metadata
+
Runtime Assumptions
```

对于 CV：

```text
Architecture
+
Weights
+
Resize
+
Normalize
+
Input Layout
+
Class Labels
```

对于 LLM：

```text
Architecture
+
Weights
+
Tokenizer
+
Chat Template
+
Generation Config
```

---

# 113. 为什么预处理很重要？

假设训练时：

```text
RGB
224×224
float32
/255
mean/std normalize
```

部署时：

```text
BGR
256×256
uint8
未 normalize
```

即使 Weight 完全一样：

```text
Output 也会完全不同
```

---

# 114. Input Layout

常见：

```text
NCHW
```

PyTorch 常见：

```text
[B,C,H,W]
```

TensorFlow / LiteRT 常见：

```text
NHWC
```

即：

```text
[B,H,W,C]
```

如果 Layout 弄错：

```text
模型无法正确工作
```

---

# 115. Layout Conversion

Runtime 可能需要：

```text
NCHW
↓
NHWC
```

转换。

这可能增加：

```text
Memory Copy
Latency
```

硬件优化时：

> Layout 是非常重要的问题。

---

# 116. Dtype mismatch

模型输入期望：

```text
float32
```

但你传：

```text
uint8
```

或者量化模型期望：

```text
int8 + scale / zero point
```

你却直接传 float：

```text
结果都会错
```

---

# 117. Quantized Tensor

INT8 模型往往需要：

```text
real_value
≈
scale × (quantized_value - zero_point)
```

所以：

```text
Tensor Data
+
Quantization Parameters
```

共同定义真实数值。

这个留到量化章节深入。

---

# 118. Post-processing

例如 Classification：

```text
Logits
↓
Softmax
↓
Argmax
↓
Class
```

Detection：

```text
Model Output
↓
Decode Box
↓
Confidence Filter
↓
NMS
↓
Final Detection
```

部署模型正确但后处理错误：

> 仍然会得到“模型坏了”的假象。

---

# 119. Numerical Validation

模型转换后必须验证：

```text
Framework Output
vs
Exported Runtime Output
```

例如：

```text
PyTorch
vs
ONNX Runtime
```

使用相同：

```text
Input
Preprocess
dtype
shape
```

然后：

```python
np.max(np.abs(
    pytorch_output - onnx_output
))
```

---

# 120. 为什么会有 Numerical Difference？

可能来自：

```text
Floating Point Error
Different Kernel
Different Operator Implementation
Graph Fusion
Quantization
Different Precision
Different Resize Semantics
```

小差异通常正常。

巨大差异：

```text
需要 Debug
```

---

# 121. 模型导出建议的 Debug 顺序

```text
1. 固定 Input

2. Framework Inference

3. Export

4. Exported Runtime Inference

5. 比较最终 Output

6. 若错误：
   比较 Intermediate Tensor

7. 找到第一个明显偏差 Operator
```

这是非常通用的模型转换 Debug 方法。

---

# 122. 为什么 Netron 很重要？

导出：

```text
model.onnx
```

打开 Netron。

观察：

```text
Input Shape
Output Shape
Operator
Initializer
Dynamic Dimension
Layout
```

很多问题会直接暴露。

---

# 123. PyTorch → ONNX → ORT 最小实验

PyTorch：

```python
model.eval()

x = torch.randn(1, 10)
```

导出 ONNX。

然后：

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

onnx_output = session.run(
    None,
    {"input": x.numpy()}
)
```

最后比较：

```text
PyTorch Output
vs
ORT Output
```

---

# 124. 为什么这一实验必须做？

因为它会把：

```text
Module
→
Graph
→
File
→
Runtime
```

第一次完整串起来。

这是从：

> “会训练模型”

进入：

> “会部署模型”

的关键一步。

---

# 125. TensorFlow / Keras → LiteRT

经典流程：

```text
Keras Model
    ↓
Export / Conversion
    ↓
LiteRT Model
    ↓
.tflite
    ↓
LiteRT Interpreter / CompiledModel
```

再运行：

```text
CPU
GPU
NPU
```

---

# 126. PyTorch → LiteRT

当前 LiteRT 官方也提供：

```text
PyTorch Model
↓
LiteRT Torch Conversion
↓
.tflite
```

所以：

> PyTorch 训练并不意味着一定只能走 ONNX。

Edge 生态正在越来越 multi-framework。

---

# 127. Framework Interoperability

今天模型部署更像：

```text
            PyTorch
          /    |    \
        ONNX LiteRT Export
         |      |     |
        ORT  LiteRT ExecuTorch
```

甚至：

```text
ONNX
↓
RKNN-Toolkit2
↓
.rknn
```

所以：

> 训练框架和最终 Runtime 已经逐渐解耦。

---

# 128. 为什么仍然存在转换痛点？

理论上：

```text
Framework
↓
IR
↓
Any Hardware
```

非常美好。

现实：

```text
Operator Version
Dynamic Shape
Custom Op
Numerical Semantics
Layout
Control Flow
Quantization
Hardware Support
```

都会制造兼容性问题。

---

# 129. Operator Coverage

这是部署里最现实的问题。

假设模型需要：

```text
100 种 Operator
```

目标 NPU 只支持：

```text
80 种
```

剩下：

```text
无法编译
```

或者：

```text
Fallback CPU
```

所以模型部署之前需要：

```text
Operator Support Analysis
```

---

# 130. Custom Operator

如果模型使用：

```text
特殊算法
```

标准 ONNX / LiteRT 不支持。

可以：

```text
注册 Custom Op
```

但这意味着：

```text
需要自己实现 Kernel
```

这已经开始进入第三部分。

---

# 131. Static Graph 为什么更容易部署？

Static：

```text
Input Shape 已知
Operator 已知
Graph 已知
```

Compiler 可以：

```text
Memory Planning
Fusion
Scheduling
Pre-allocate Buffer
Select Kernel
```

所以：

> Edge NPU 通常更偏爱 Static Graph。

---

# 132. Dynamic Framework 为什么更适合 Research？

PyTorch Python：

```text
if
for
list
class
debugger
print
```

开发体验极佳。

所以：

```text
Research / Train
→
Dynamic Framework
```

而：

```text
Deployment
→
Static / Captured Graph
```

是非常常见的分工。

---

# 133. Training Graph vs Inference Graph

Training：

```text
Forward
+
Backward
+
Optimizer
```

Inference：

```text
Forward Only
```

因此导出部署时：

```text
通常只保留 Forward Graph
```

从而可以删除：

```text
Gradient
Optimizer State
Training-only Op
```

---

# 134. Dropout 在 Inference

Training：

```text
Dropout ON
```

Inference：

```text
Dropout OFF
```

如果忘记：

```python
model.eval()
```

导出 / 推理行为可能不正确。

---

# 135. BatchNorm 在 Inference

Training：

```text
使用 Batch Statistics
更新 Running Statistics
```

Inference：

```text
使用 Running Mean / Var
```

所以：

```text
train()
eval()
```

不仅是语义标签。

---

# 136. 常见模型导出 Checklist

在 Export 前：

```text
[ ] model.eval()

[ ] 固定随机种子（如果需要）

[ ] 准备真实 Example Input

[ ] 确认 dtype

[ ] 确认 input layout

[ ] 确认 dynamic / static shape

[ ] 清理 training-only logic

[ ] 避免非必要 Python side effect

[ ] 明确 output semantics

[ ] 保存原 Framework baseline output
```

---

# 137. Export 后 Checklist

```text
[ ] Model Checker

[ ] Netron 打开

[ ] Input / Output Name

[ ] Shape

[ ] Opset

[ ] Operator List

[ ] Runtime Load

[ ] Same-input Output Comparison

[ ] Performance Benchmark
```

---

# 138. Benchmark 必须区分什么？

```text
Warmup

Latency

Throughput

Batch Size

Input Shape

Precision

Device

Number of Threads

Copy Time

Preprocess Time

Postprocess Time
```

否则两个 Benchmark：

> 没有可比性。

---

# 139. Latency vs Throughput

Latency：

```text
一个 Request 要多久
```

Throughput：

```text
单位时间处理多少 Request / Sample
```

Edge AI 经常更关注：

```text
Latency
Power
Memory
```

Server AI 更可能同时重视：

```text
Throughput
```

---

# 140. Batch Size 对 Runtime 很重要

Batch 大：

```text
并行度 ↑
Throughput ↑
Memory ↑
Latency 可能 ↑
```

Batch 1：

```text
低延迟
```

更接近很多：

```text
Embedded / Real-time
```

场景。

---

# 141. CPU Thread

CPU Runtime 还需要考虑：

```text
Thread Count
Affinity
SIMD
Cache
Memory Bandwidth
```

所以同一个 ONNX Model：

```text
不同 CPU Runtime 配置
```

性能可能差很多。

---

# 142. PyTorch Dispatcher：往下一层看

当写：

```python
torch.matmul(x, w)
```

并不是 Python 自己做矩阵乘。

大致：

```text
Python API
↓
Dispatcher
↓
ATen Operator
↓
CPU / CUDA Backend
↓
Kernel
```

这是 PyTorch 向下连接硬件的核心思想之一。

---

# 143. ATen

ATen：

> PyTorch 的核心 Tensor / Operator Library 之一。

很多 PyTorch Operator：

```text
add
matmul
conv
softmax
```

最终对应：

```text
ATen Operator
```

---

# 144. Backend

同一个高层：

```text
MatMul
```

在不同 Device：

```text
CPU
CUDA
MPS
...
```

会选择：

```text
不同 Backend Kernel
```

这也是：

```text
Operator Semantics
和
Kernel Implementation
```

分离的典型例子。

---

# 145. Operator vs Kernel 再明确一次

Operator：

```text
“做什么”
```

例如：

```text
MatMul
```

Kernel：

```text
“在特定硬件上怎么高效做”
```

例如：

```text
CPU AVX MatMul Kernel
CUDA Tensor Core GEMM
NPU MatMul Kernel
```

---

# 146. Framework vs Backend

Framework：

```text
PyTorch
```

定义：

```text
Tensor
Model
Training
```

Backend：

```text
CUDA
CPU
MPS
...
```

负责：

```text
Operator Execution
```

所以：

```text
PyTorch ≠ CUDA
```

---

# 147. TensorFlow 与 XLA

TensorFlow 生态中还有：

```text
XLA
```

用于：

```text
Graph Compilation
Optimization
Target Code
```

本章只需要知道位置：

```text
TensorFlow Program
↓
Graph
↓
XLA
↓
Hardware
```

详细 Compiler 思维在 TVM 章节统一展开。

---

# 148. PyTorch Compile 与 AI Compiler 的关系

`torch.compile` 已经在做：

```text
Graph Capture
Operator Decomposition
Fusion
Code Generation
```

所以它本身已经带有强烈：

```text
AI Compiler
```

属性。

后面 TVM 会把这些概念抽象得更完整。

---

# 149. ONNX 与 AI Compiler 的关系

ONNX 更像：

```text
Input IR / Interchange Format
```

Compiler：

```text
ONNX
↓
Parse
↓
Graph IR
↓
Optimize
↓
Lower
↓
Codegen
```

例如：

```text
TVM
RKNN Compiler
TensorRT
```

都可能消费类似 Graph。

---

# 150. PyTorch → RKNN

对于很多视觉模型：

```text
PyTorch
↓
ONNX
↓
RKNN-Toolkit2
↓
Quantize / Compile
↓
.rknn
↓
RKNN Runtime
↓
Rockchip NPU
```

这条链中：

```text
PyTorch
=
Training Framework

ONNX
=
Interchange Graph

RKNN-Toolkit2
=
Vendor Converter / Compiler Toolchain

.rknn
=
Compiled Deployment Model

RKNN Runtime
=
Device Runtime
```

这几个角色必须完全区分。

---

# 151. TensorFlow → RKNN

某些模型也可以：

```text
TensorFlow / TFLite
↓
RKNN-Toolkit2
↓
.rknn
```

具体支持格式 / Operator 要看对应 Toolkit 版本。

因此：

> ONNX 很常用，但不是所有部署路线唯一入口。

---

# 152. 为什么 ONNX 对 RKNN 特别重要？

因为大量 PyTorch 模型：

```text
先导出 ONNX
```

再进入：

```text
RKNN Toolkit
```

所以：

```text
ONNX Graph
Operator
Shape
Opset
```

是后续排错最常接触的内容。

---

# 153. 为什么不应该只会“复制转换命令”？

如果转换报：

```text
Unsupported Op
Shape Error
Quantization Error
Layout Error
```

只有理解：

```text
Graph
Operator
Tensor
Runtime
```

才能定位。

否则会变成：

```text
不停换版本
不停复制命令
```

但不知道为什么成功 / 失败。

---

# 154. 本章建议实际学习顺序

---

## Stage 1：PyTorch Tensor

重点：

```text
Tensor
Shape
Dtype
Device
Stride
Contiguous
Broadcast
```

实验：

```text
创建 Tensor
reshape
transpose
permute
matmul
to(cuda)
```

目标：

> 任何 Tensor 操作先想 Shape 和 Memory Layout。

---

## Stage 2：PyTorch Module

学习：

```text
nn.Module
Parameter
Buffer
state_dict
train/eval
```

自己写：

```text
MLP
CNN
Tiny Transformer Block
```

目标：

> 能完全读懂一个基础 PyTorch Model Class。

---

## Stage 3：Data / Training

学习：

```text
Dataset
DataLoader
Loss
Optimizer
Training Loop
Checkpoint
```

目标：

> 自己写一个完整可恢复训练工程。

---

## Stage 4：Autograd 工程化

结合：

- [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)

重点：

```text
requires_grad
grad_fn
no_grad
inference_mode
```

目标：

> 把上一章 micrograd 与真实 PyTorch 对应起来。

---

## Stage 5：PyTorch Compile / Export

学习：

- [`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [`torch.export`](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/api_reference.html)

重点区分：

```text
Optimize Execution
vs
Export Graph
```

---

## Stage 6：ONNX

学习：

- [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html)
- [ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX Operators](https://onnx.ai/onnx/operators/)

理解：

```text
Model
Graph
Node
Operator
Initializer
Input/Output
Shape
Opset
```

---

## Stage 7：ONNX Runtime

学习：

- [ONNX Runtime](https://github.com/microsoft/onnxruntime)
- [Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

重点：

```text
InferenceSession
Execution Provider
Graph Optimization
CPU Fallback
IOBinding
```

---

## Stage 8：TensorFlow / Keras

只需要：

```text
Tensor
Keras Model
Sequential
Functional
Subclass
tf.function
SavedModel / Model Save
```

目标：

> 能看懂主流 TensorFlow / Keras Model。

---

## Stage 9：LiteRT

学习：

- [LiteRT Official](https://developers.google.com/edge/litert)
- [LiteRT GitHub](https://github.com/google-ai-edge/LiteRT)

理解：

```text
TensorFlow Lite → LiteRT

.tflite
Interpreter
CompiledModel
CPU / GPU / NPU
Conversion
Quantization
```

---

## Stage 10：ExecuTorch

只需要建立：

```text
PyTorch
↓
torch.export
↓
ExecuTorch
↓
.pte
↓
Edge Runtime
```

的位置认识。

---

# 155. 必做实验 1：PyTorch Model + Checkpoint

写一个：

```text
MLP / CNN
```

完成：

```text
Train
Save state_dict
Restart Python
Create Model
Load state_dict
Inference
```

验证：

```text
Before Save Output
≈
After Load Output
```

---

# 156. 实验 2：完整 Training Checkpoint

保存：

```text
Model
Optimizer
Epoch
Loss
```

重新启动：

```text
Resume Training
```

目标：

> 分清 Model Weight 和 Training State。

---

# 157. 实验 3：PyTorch → ONNX

模型：

```text
Simple MLP
```

完成：

```text
PyTorch
↓
Export ONNX
↓
onnx.checker
↓
Netron
```

观察：

```text
nn.Linear
```

最终在 ONNX 中如何表示。

---

# 158. 实验 4：PyTorch vs ONNX Runtime

同一 Input：

```text
PyTorch Output
```

和：

```text
ORT Output
```

比较：

```text
max absolute error
mean absolute error
```

目标：

> 建立转换后的 Numerical Validation 习惯。

---

# 159. 实验 5：观察 ONNX Graph

使用 Netron：

```text
CNN
```

重点找：

```text
Conv
Relu
MaxPool
Flatten / Reshape
Gemm / MatMul
```

并和：

```text
PyTorch model
```

逐层对应。

---

# 160. 实验 6：Dynamic Shape

导出两个模型：

```text
Static Batch = 1

Dynamic Batch
```

观察 ONNX Input Shape 差异。

理解：

> Dynamic Shape 为什么会增加 Runtime / Compiler 难度。

---

# 161. 实验 7：Execution Provider

如果有 NVIDIA GPU：

```text
CPUExecutionProvider

CUDAExecutionProvider
```

对同一个 ONNX Model Benchmark。

观察：

```text
Latency
Warmup
Device Copy
```

---

# 162. 实验 8：制造 CPU Fallback

选择一个目标 EP 不完全支持的 Model / Op。

观察：

```text
Provider Assignment
```

体会：

> “用了 GPU/NPU Runtime”不等于“所有节点都在 Accelerator”。

---

# 163. 实验 9：Keras Model

写一个：

```text
Sequential MLP / CNN
```

完成：

```text
compile
fit
evaluate
predict
save
```

目标不是精通 Keras。

目标：

> 能把 PyTorch 概念映射到 Keras。

---

# 164. 实验 10：LiteRT

使用一个简单 Keras 或 PyTorch Model：

```text
Convert
↓
.tflite
↓
LiteRT Runtime
↓
Inference
```

比较：

```text
Original Framework
vs
LiteRT
```

输出。

---

# 165. PyTorch vs TensorFlow / Keras 对照

| 概念 | PyTorch | Keras / TensorFlow |
|---|---|---|
| Tensor | `torch.Tensor` | `tf.Tensor` / Keras tensor |
| Model Base | `nn.Module` | `keras.Model` |
| Dense | `nn.Linear` | `layers.Dense` |
| Conv | `nn.Conv2d` | `layers.Conv2D` |
| Dataset | `Dataset/DataLoader` | `tf.data.Dataset` / Keras data APIs |
| Autograd | `torch.autograd` | `GradientTape` |
| Train Mode | `model.train()` | Training flag / framework-managed |
| Eval | `model.eval()` | inference / evaluate |
| Save Weight | `state_dict` | model/weight saving APIs |
| Compile Graph | `torch.compile` | `tf.function` / XLA |
| Export | `torch.export` / ONNX | SavedModel / LiteRT conversion |

表的目的是：

> 找共同抽象，而不是背 API。

---

# 166. PyTorch 与 ONNX 对照

| PyTorch | ONNX |
|---|---|
| `nn.Module` | Graph / subgraph |
| `nn.Linear` | Gemm or MatMul + Add |
| `nn.Conv2d` | Conv |
| `nn.ReLU` | Relu |
| Parameter | Initializer Tensor |
| Tensor | Tensor Value |
| forward | Computation Graph |
| Python control flow | Graph control-flow ops / captured semantics |
| `.pth` | `.onnx` |

---

# 167. ONNX 与 Runtime 对照

ONNX：

```text
定义：
“要算什么”
```

Runtime：

```text
决定：
“怎么在硬件上算”
```

例如：

```text
ONNX MatMul
```

可能由：

```text
CPU Kernel
CUDA cuBLAS
TensorRT
QNN
OpenVINO
```

实现。

---

# 168. LiteRT 与 ONNX 的区别

ONNX：

```text
开放模型交换 / IR 规范
```

LiteRT：

```text
Edge Runtime + Conversion / Optimization Toolchain
```

虽然：

```text
.tflite
```

也包含 Graph。

但二者生态定位不同。

---

# 169. ExecuTorch 与 LiteRT 的区别

ExecuTorch：

```text
PyTorch-native Edge Deployment
```

LiteRT：

```text
Google AI Edge Cross-framework Runtime
```

二者都在竞争 / 覆盖：

```text
On-device AI
```

---

# 170. ONNX Runtime 与 LiteRT 的区别

ONNX Runtime：

```text
围绕 ONNX
Cross-platform
Server + Desktop + Edge
Execution Provider 生态
```

LiteRT：

```text
更偏 On-device / Edge
Mobile / Embedded
GPU / NPU
.tflite / LiteRT ecosystem
```

---

# 171. 为什么不存在“一个格式统治所有硬件”？

因为硬件不同：

```text
CPU
GPU
NPU
DSP
MCU
```

对：

```text
Operator
Precision
Layout
Memory
Dynamic Shape
Control Flow
```

支持不同。

所以：

```text
Universal Model
```

通常还需要：

```text
Target-specific Compilation
```

---

# 172. 端侧 AI 的基本现实

```text
Train Once
Deploy Anywhere
```

是理想。

现实：

```text
Train
↓
Export
↓
Convert
↓
Modify Unsupported Ops
↓
Quantize
↓
Compile
↓
Validate
↓
Benchmark
↓
Optimize
```

这是 Edge AI Engineer 真正做的事情。

---

# 173. 模型转换失败的常见原因

```text
Unsupported Operator
Unsupported Opset
Dynamic Shape
Control Flow
Custom Layer
Tensor Layout
Dtype
Quantization
Model Version
Converter Bug
```

---

# 174. Runtime Load 成功但结果错误

常见原因：

```text
Input Resize 错

RGB/BGR 错

Normalize 错

NCHW/NHWC 错

dtype 错

Quantization scale 错

Output decode 错

Tokenizer 错
```

所以：

> 不要第一反应怀疑模型 Weight。

---

# 175. Runtime 能跑但速度很慢

常见：

```text
CPU Fallback
频繁 Device Copy
Dynamic Shape
低效 Layout
未使用 FP16 / INT8
Batch 不合适
线程配置不合理
Operator 没融合
Kernel 实现差
```

---

# 176. Model Size 不是 Runtime Memory

模型文件：

```text
100 MB
```

运行内存可能：

```text
300 MB
```

因为还需要：

```text
Activation
Workspace
Runtime Buffer
Input / Output
Allocator
Cache
```

LLM 还会有：

```text
KV Cache
```

---

# 177. Runtime Peak Memory

真正端侧部署要关心：

```text
Peak RSS
GPU Memory
NPU Memory
CMA
ION / DMA Buffer
```

而不是只看：

```text
model.onnx size
```

---

# 178. Graph Optimization 的三个层次

可以粗略：

```text
High-level Graph

Operator-level

Kernel-level
```

例如：

```text
Conv + BN Fusion
```

属于 Graph / Operator。

```text
Tiled GEMM
```

属于 Kernel。

---

# 179. Framework 层与 Hardware 层之间还有很多层

```text
PyTorch
↓
Graph
↓
IR
↓
Graph Optimizer
↓
Backend
↓
Operator Library
↓
Kernel
↓
Driver
↓
Hardware
```

本章主要解决前半部分。

---

# 180. 为什么这一章非常关键？

因为后面：

```text
llama.cpp
vLLM
MLC-LLM
TVM
RKNN
```

其实都会重复出现：

```text
Model
Graph
Tensor
Operator
Backend
Runtime
```

如果这几个概念不牢：

> 后面会一直混淆“模型文件”“Runtime”“Compiler”。

---

# 181. 本章核心学习树

```text
Model Framework
│
├── PyTorch
│   ├── Tensor
│   │   ├── Shape
│   │   ├── Dtype
│   │   ├── Device
│   │   └── Stride
│   │
│   ├── Autograd
│   │   ├── requires_grad
│   │   ├── grad_fn
│   │   └── backward
│   │
│   ├── nn.Module
│   │   ├── Parameter
│   │   ├── Buffer
│   │   ├── state_dict
│   │   └── train / eval
│   │
│   ├── Dataset / DataLoader
│   ├── Optimizer
│   ├── Serialization
│   ├── torch.compile
│   └── torch.export
│
├── TensorFlow / Keras
│   ├── Tensor
│   ├── Eager
│   ├── tf.function
│   ├── Sequential
│   ├── Functional
│   ├── Subclassing
│   └── SavedModel / Keras Save
│
├── LiteRT
│   ├── TensorFlow Lite Legacy
│   ├── .tflite
│   ├── Conversion
│   ├── Quantization
│   ├── Interpreter API
│   ├── CompiledModel API
│   └── CPU / GPU / NPU
│
├── ONNX
│   ├── ModelProto
│   ├── GraphProto
│   ├── Node
│   ├── Operator
│   ├── Initializer
│   ├── Tensor
│   ├── Shape
│   ├── IR Version
│   └── Opset
│
├── ONNX Runtime
│   ├── InferenceSession
│   ├── Graph Optimization
│   ├── Execution Provider
│   ├── CPU Fallback
│   ├── IOBinding
│   └── Kernel
│
└── ExecuTorch
    ├── torch.export
    ├── Edge Lowering
    ├── .pte
    └── Lightweight Runtime
```

---

# 182. 本章最重要的思维模型

## 思维模型 1

```text
Framework
≠
Model Format
≠
Runtime
≠
Compiler
```

---

## 思维模型 2

```text
PyTorch
主要解决：
Model Development + Training
```

---

## 思维模型 3

```text
ONNX
主要解决：
Portable Computation Graph Representation
```

---

## 思维模型 4

```text
ONNX Runtime
主要解决：
Execute ONNX Graph
```

---

## 思维模型 5

```text
LiteRT
主要解决：
On-device Model Conversion + Runtime + Acceleration
```

---

## 思维模型 6

```text
Deployment
不是保存 Weight

而是：

Model
↓
Capture / Export
↓
Graph
↓
Convert / Compile
↓
Runtime
↓
Hardware
```

---

## 思维模型 7

```text
Layer
≠
Operator
```

一个高层 Layer：

```text
可能被拆成多个 Operator
```

---

## 思维模型 8

```text
Operator
=
What to compute

Kernel
=
How to compute efficiently
on specific hardware
```

---

## 思维模型 9

```text
“模型能跑”
≠
“模型高性能运行”
```

---

## 思维模型 10

```text
Correct Deployment
=
Correct Graph
+
Correct Weight
+
Correct Preprocess
+
Correct Dtype
+
Correct Layout
+
Correct Runtime
+
Correct Postprocess
```

---

# 183. 本章完成标准

如果下面大部分都能自己解释，本章就可以结束。

## PyTorch

- [ ] 能熟练创建和操作 Tensor
- [ ] 能解释 Shape / Dtype / Device
- [ ] 知道 Stride / Contiguous 是什么
- [ ] 能自己写 `nn.Module`
- [ ] 能解释 Parameter
- [ ] 能解释 Buffer
- [ ] 能解释 `model.parameters()`
- [ ] 能解释 `state_dict`
- [ ] 能区分 Weight 和 Checkpoint
- [ ] 能写 Dataset / DataLoader
- [ ] 能写完整 Training Loop
- [ ] 能解释 `train()` / `eval()`
- [ ] 能解释 `no_grad` / `inference_mode`
- [ ] 能解释 Eager Execution
- [ ] 知道 `torch.compile` 解决什么
- [ ] 知道 `torch.export` 解决什么
- [ ] 能解释 `compile` 与 `export` 的区别

## Model Serialization

- [ ] 能解释 `.pt/.pth`
- [ ] 能解释 `.safetensors`
- [ ] 能解释 `.onnx`
- [ ] 能解释 `.tflite`
- [ ] 能解释 `.pte`
- [ ] 知道 Architecture 与 Weight 是两个概念
- [ ] 知道模型文件大小不等于 Runtime 内存

## TensorFlow / Keras

- [ ] 知道 TensorFlow 2 默认支持 Eager
- [ ] 知道 `tf.function`
- [ ] 能看懂 Keras Sequential
- [ ] 能看懂 Functional API
- [ ] 知道 Keras 3 是 multi-backend
- [ ] 知道 SavedModel 的基本位置
- [ ] 能完成一个最小 Keras Training

## LiteRT

- [ ] 知道 TensorFlow Lite 与 LiteRT 的关系
- [ ] 知道 `.tflite`
- [ ] 能解释 Interpreter Runtime
- [ ] 知道 CompiledModel API 的定位
- [ ] 知道 LiteRT 支持 TensorFlow / PyTorch / JAX 路线
- [ ] 知道 LiteRT 面向 CPU / GPU / NPU / Edge
- [ ] 跑过一次 `.tflite` Inference

## ONNX

- [ ] 能解释 ONNX 为什么存在
- [ ] 能解释 ModelProto
- [ ] 能解释 GraphProto
- [ ] 能解释 Node
- [ ] 能解释 Operator
- [ ] 能解释 Initializer
- [ ] 能解释 Input / Output
- [ ] 能解释 Opset
- [ ] 能解释 IR Version
- [ ] 能解释 Dynamic Shape
- [ ] 能解释 Custom Operator
- [ ] 能用 Netron 查看 ONNX Graph

## ONNX Runtime

- [ ] 能创建 `InferenceSession`
- [ ] 能运行 ONNX Model
- [ ] 能解释 Execution Provider
- [ ] 能解释 CPU Fallback
- [ ] 能解释 Graph Optimization
- [ ] 知道 IOBinding 在解决什么
- [ ] 能完成 PyTorch vs ORT 数值比较
- [ ] 最好做过 CPU vs GPU EP Benchmark

## Deployment Engineering

- [ ] 能画出 PyTorch → ONNX → ORT
- [ ] 能画出 PyTorch → torch.export → ExecuTorch
- [ ] 能画出 Keras → LiteRT
- [ ] 能画出 PyTorch → LiteRT
- [ ] 能画出 PyTorch → ONNX → RKNN
- [ ] 能解释 Unsupported Operator
- [ ] 能解释 Static / Dynamic Shape
- [ ] 能解释 Layout NCHW / NHWC
- [ ] 能解释 Dtype mismatch
- [ ] 能解释 Preprocess mismatch
- [ ] 能解释 Numerical Validation
- [ ] 能解释为什么 CPU Fallback 可能很慢

---

# 184. 本章不要求深入

暂时不要求：

- [ ] 阅读完整 PyTorch C++ 源码
- [ ] 手写 Autograd Engine
- [ ] 精通 TorchInductor
- [ ] 精通 XLA
- [ ] 精通 ONNX protobuf 源码
- [ ] 自己实现 ONNX Runtime Execution Provider
- [ ] 自己实现 LiteRT Delegate
- [ ] 自己实现 Custom Kernel
- [ ] 精通量化
- [ ] 精通 TVM
- [ ] 精通 TensorRT
- [ ] 精通 RKNN
- [ ] 精通 GPU Kernel

这些后面逐层进入。

---

# 185. 核心参考链接

## PyTorch

- [PyTorch Repository](https://github.com/pytorch/pytorch)
- [PyTorch Docs](https://docs.pytorch.org/)
- [PyTorch Tutorials](https://docs.pytorch.org/tutorials/)
- [nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)
- [Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [PyTorch 2.x Compile Stack](https://pytorch.org/get-started/pytorch-2-x/)
- [torch.export API](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/export/api_reference.html)

建议优先看：

```text
Tensor
nn.Module
Autograd
DataLoader
state_dict
torch.compile
torch.export
```

---

## TensorFlow

- [TensorFlow Repository](https://github.com/tensorflow/tensorflow)
- [TensorFlow Official](https://www.tensorflow.org/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)

---

## Keras

- [Keras Repository](https://github.com/keras-team/keras)
- [Keras Official](https://keras.io/)

---

## LiteRT

- [LiteRT Repository](https://github.com/google-ai-edge/LiteRT)
- [LiteRT Official](https://developers.google.com/edge/litert)
- [LiteRT Samples](https://github.com/google-ai-edge/litert-samples)

重点：

```text
Model Conversion
.tflite
Optimization
Interpreter API
CompiledModel API
CPU / GPU / NPU
```

---

## ONNX

- [ONNX Repository](https://github.com/onnx/onnx)
- [ONNX Introduction](https://onnx.ai/onnx/intro/)
- [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html)
- [ONNX IR Specification](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX Operators](https://onnx.ai/onnx/operators/)
- [ONNX Python API](https://onnx.ai/onnx/intro/python)

---

## ONNX Runtime

- [ONNX Runtime Repository](https://github.com/microsoft/onnxruntime)
- [ONNX Runtime Official](https://onnxruntime.ai/)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [Execution Providers](https://onnxruntime.ai/docs/execution-providers/)

---

## ExecuTorch

- [ExecuTorch Repository](https://github.com/pytorch/executorch)
- [ExecuTorch Documentation](https://docs.pytorch.org/executorch/)

---

## Model Visualization

- [Netron](https://github.com/lutzroeder/netron)

---

# 186. 推荐学习组合

```text
PyTorch
=
真正掌握

TensorFlow / Keras
=
会读 + 会基本训练

ONNX
=
真正理解 Graph / Operator / IR

ONNX Runtime
=
真正完成一次跨框架推理

LiteRT
=
理解现代端侧 Runtime

ExecuTorch
=
理解 PyTorch Native Edge Path
```

不要平均用力。

---

# 187. 本章和前面三章的关系

前面：

```text
01
机器学习是什么

02
神经网络如何训练

03
Transformer / LLM 如何工作
```

本章：

```text
04
这些模型在真实软件框架里
如何表示、训练、保存、导出和运行
```

所以它其实是：

> 理论 AI → AI Systems

的第一座桥。

---

# 188. 本章和后续 Runtime 的关系

本章已经建立：

```text
Model
Graph
Operator
Runtime
Backend
```

接下来第二部分进入：

```text
AI Agent
llama.cpp
vLLM
MLC-LLM
```

其中：

```text
llama.cpp
```

会让我们看到：

> 不通过 PyTorch，也可以直接实现一套 LLM Runtime。

---

# 189. 本章和 AI Compiler 的关系

当模型已经变成：

```text
Graph
```

下一步自然会问：

```text
Graph 可以优化什么？
Operator 可以融合吗？
如何 Lower 到 Hardware？
如何选择 Kernel？
如何做 Code Generation？
```

这就是：

```text
AI Compiler
```

---

# 190. 本章和 RKNN 的关系

最终 Rockchip 路线：

```text
PyTorch
↓
ONNX
↓
RKNN-Toolkit2
↓
Graph / Operator Analysis
↓
Quantization
↓
NPU Compilation
↓
.rknn
↓
RKNN Runtime
↓
RK3576
```

本章实际已经学完：

```text
前两步最核心的抽象。
```

---

# 191. 一句话总结

本章真正需要记住的不是几十个 API。

而是这一条链：

```text
Python Model
      ↓
Training Framework
      ↓
Parameters / Weights
      ↓
Graph Capture / Export
      ↓
Model IR / Deployment Format
      ↓
Runtime
      ↓
Backend
      ↓
Operator Kernel
      ↓
Hardware
```

具体到 PyTorch：

```text
PyTorch nn.Module
      ↓
Training
      ↓
state_dict / safetensors
      ↓
torch.export / ONNX
      ↓
ExecuTorch / ONNX Runtime / LiteRT / RKNN Toolchain
      ↓
CPU / GPU / NPU
```

当这一层真正理解以后：

```text
ONNX
.tflite
.pte
.rknn
GGUF
```

就不再只是不同的“文件后缀”。

而会变成：

> **不同 AI 软件栈在不同抽象层，为了把模型从开发框架送到底层硬件而设计出来的工程产物。**
