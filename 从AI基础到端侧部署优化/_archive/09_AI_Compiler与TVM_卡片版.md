# 09_AI_Compiler与TVM

> **所属路线**：AI 学习路线 · 第三部分  
> **定位**：从 LLM Runtime 继续向下进入 AI Compiler，理解模型如何从 Framework / Graph 经过 IR、Graph Optimization、Tensor Program、Schedule、Lowering、Code Generation，最终变成 CPU/GPU/NPU 上执行的代码  
> **核心仓库**：[`apache/tvm`](https://github.com/apache/tvm)  
> **核心抽象**：Relax + TensorIR（当前主线进一步拆分为 `tirx` + `s_tir`）  
> **关键扩展**：TVMScript、DLight、MetaSchedule、BYOC、Relax VM、LLVM/CUDA/Vulkan/OpenCL/Metal CodeGen  
> **学习边界**：本章解决 AI Compiler 的总体架构和 TVM 核心机制；FlashAttention、Triton、CUTLASS、GPU Kernel 在下一阶段深入，RKNN/RKLLM 则用于理解 Vendor NPU Compiler / Runtime。  
> **学习方式**：不要把本章学成“TVM API 教程”，而要始终追问：**一个 MatMul 是如何从模型里的一个算子，最终变成硬件上的线程、访存和指令的？**  
> **版本说明**：本章由 09 / 10 两版《AI_Compiler与TVM》笔记合并而成（2026-08-25），以 09 版为基线并吸收 10 版独有内容。  
> **资料检查日期**：2026-08-25

---

# 0. 本章最重要的问题

前面已经知道：

```text
PyTorch Model
↓
Runtime
↓
MatMul / Attention / RMSNorm
↓
CPU / GPU / NPU
```

但是这里还有一个巨大的黑盒：

```text
MatMul
↓
???
↓
GPU Kernel
```

AI Compiler 就是在解决这个：

```text
???
```

完整一点：

```text
PyTorch / ONNX / LLM Model
            ↓
        High-level Graph
            ↓
             IR
            ↓
       Graph Optimization
            ↓
      Tensor-level Program
            ↓
          Schedule
            ↓
         Lowering
            ↓
      Target-specific IR
            ↓
         CodeGen
            ↓
    CPU / GPU / NPU Kernel
            ↓
          Runtime
```

本章最终要能回答：

> **一个 PyTorch 中的 `torch.matmul(x, w)`，到底经过了什么，最终才变成 CPU SIMD 指令、CUDA Kernel，或者 NPU 上的矩阵计算任务？**

---

# 1. 本章学习目标

完成本章以后，需要能够回答：

1. AI Compiler 到底是什么？
2. 为什么 PyTorch Model 不能直接等价于 GPU/NPU 上执行的 Machine Code？
3. AI Compiler 和传统 C/C++ Compiler 有什么相同点？
4. AI Compiler 和传统 Compiler 最大的不同在哪里？
5. Frontend、IR、Optimization、Lowering、CodeGen、Runtime 分别是什么？
6. Graph IR 和 Tensor IR 为什么需要分成不同层？
7. ONNX 和 TVM Relax 有什么区别？
8. Relax 为什么被称为 Graph-level IR？
9. TensorIR 为什么被称为 Primitive Tensor Function IR？
10. 当前 TVM 主线中的 `tirx` 和 `s_tir` 分别是什么？
11. 为什么很多旧 TVM 教程还在使用 `Relay`、`tvm.tir`？
12. Relay 与 Relax 是什么关系？
13. `IRModule` 是什么？
14. `Relax Function` 和 `PrimFunc` 有什么区别？
15. `R.call_tir()` 为什么非常关键？
16. 为什么 Relax 和 TensorIR 可以存在于同一个 IRModule 中？
17. 什么叫 Cross-level Optimization？
18. 什么是 Legalization？
19. `LegalizeOps` 在做什么？
20. Graph Fusion 是什么？
21. Constant Folding、Dead Code Elimination、Layout Transform 是什么？
22. 为什么 AI Compiler 要做 Shape Analysis？
23. Static Shape 和 Dynamic Shape 对 Compiler 有什么影响？
24. Symbolic Shape 为什么重要？
25. Tensor Program 是什么？
26. 一个 MatMul 在 TensorIR 中如何表示成 Loop Nest？
27. Schedule 是什么？
28. 为什么“计算结果相同”的两个 Loop Program 性能可能差 100 倍？
29. `split` 是什么？
30. `reorder` 是什么？
31. `fuse` 是什么？
32. `parallel` 是什么？
33. `vectorize` 是什么？
34. `unroll` 是什么？
35. GPU 中 `bind(threadIdx.x / blockIdx.x)` 是什么？
36. Tiling 为什么几乎是高性能 Kernel 的核心？
37. `cache_read` / `cache_write` 为什么能改善 Memory Locality？
38. Shared Memory Tiling 是什么？
39. Register Blocking 是什么？
40. Tensorization 是什么？
41. Tensor Core / AMX / NPU Matrix Unit 怎么通过 Tensor Intrinsic 接入？
42. DLight 是什么？
43. MetaSchedule 是什么？
44. Auto-tuning 为什么本质上是在搜索 Schedule Design Space？
45. Manual Schedule、DLight、MetaSchedule 三者怎么选？
46. Target 是什么？
47. 为什么 Target 不能只写成“GPU”？
48. LLVM CodeGen 和 CUDA CodeGen 有什么不同？
49. Source CodeGen 与 LLVM CodeGen 两类后端是什么？
50. Host Code 和 Device Code 为什么需要拆开？
51. Relax VM 是什么？
52. Relax VM 为什么自己并不执行 MatMul？
53. PackedFunc / Runtime Module 是什么？
54. AOT 和 VM 有什么区别？
55. RPC 在 TVM 中解决什么？
56. BYOC 是什么？
57. 为什么 BYOC 对 Vendor NPU 非常重要？
58. External Library Dispatch 与“全部自己生成 Kernel”有什么区别？
59. cuBLAS / cuDNN / CUTLASS 为什么可以被 Compiler 当成外部实现？
60. TVM 和 MLC-LLM是什么关系？
61. TVM 和 llama.cpp 有什么关系？
62. TVM 和 vLLM 有什么关系？
63. TVM 和 RKNN-Toolkit2 在抽象层上有哪些相似之处？
64. RKNN 为什么可以理解成 Vendor AI Compiler + Runtime Toolchain？
65. 为什么理解 TVM 后，再看 `.rknn` 编译过程会清晰很多？
66. 如何正确阅读 TVM 这样的大型 Compiler 仓库？
67. 为什么不应该一开始就扎进 CodeGen C++？
68. 如何用一个 MatMul 从 Relax 一路追到 GPU Kernel？
69. AI Compiler 工程师到底在优化什么？
70. 端侧 AI 工程师为什么值得掌握 Compiler 思维？

本章最终要建立完整链路：

```text
PyTorch / ONNX / MLC Model
            ↓
          Frontend
            ↓
           Relax
     Graph-level IR
            ↓
 Graph Transformation
            ↓
       Legalize / Fuse
            ↓
         TensorIR
     Primitive Functions
            ↓
          Schedule
            ↓
 Split / Reorder / Tile
 Cache / Vectorize / Bind
            ↓
         Lowering
            ↓
          Target
            ↓
         CodeGen
      ┌─────┼──────┐
      ↓     ↓      ↓
    LLVM   CUDA   Vulkan
      ↓     ↓      ↓
     CPU   GPU    GPU
```

然后进一步：

```text
Compiled Kernels
      ↓
Runtime Module
      ↓
Relax VM / AOT Runtime
      ↓
Application
```

---

# 2. 核心仓库：Apache TVM

GitHub：

- [apache/tvm](https://github.com/apache/tvm)

官方文档：

- [Apache TVM Documentation](https://tvm.apache.org/docs/)
- [TVM Architecture](https://github.com/apache/tvm/tree/main/docs/arch)
- [Relax Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/relax)
- [TensorIR Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/tensor_ir)

TVM 当前官方对自己的核心定位是：

> Open Machine Learning Compiler Framework。

当前设计重点：

```text
Python-first Transformation
+
Universal Deployment
+
Relax Graph-level IR
+
TensorIR Tensor-level IR
+
Cross-level Optimization
```

---

# 3. 为什么学习 TVM？

因为它把前面大量零散概念：

```text
Graph
Operator
Tensor
Shape
Backend
Kernel
Runtime
Target
Quantization
```

统一到：

```text
Compiler
```

这一个系统里。

---

# 4. 前面其实已经一直在接触 Compiler

PyTorch：

```text
torch.compile
```

MLC-LLM：

```text
Relax IR
Compiler Pass
Target
Codegen
```

RKNN：

```text
Model Conversion
Graph Optimization
Quantization
NPU Compilation
```

甚至 llama.cpp：

```text
Graph Build
Backend Scheduling
Kernel Selection
```

都带有 Compiler 思维。

TVM：

> 把这些概念系统化。

---

# 5. AI Compiler 是什么？

一个实用定义：

> **AI Compiler 是将高层机器学习模型或 Tensor Program 转换、优化并 Lower 成特定硬件可高效执行程序的编译系统。**

输入可能是：

```text
PyTorch
ONNX
TensorFlow
TFLite
Custom Model IR
```

输出可能是：

```text
CPU Machine Code
CUDA Kernel
PTX
OpenCL C
SPIR-V
Metal Shader
Vendor Runtime Module
```

---

# 6. 传统 Compiler

经典：

```text
C/C++
↓
AST
↓
IR
↓
Optimization
↓
LLVM IR
↓
Machine Code
```

---

# 7. AI Compiler

可以粗略：

```text
PyTorch / ONNX
↓
Graph IR
↓
Graph Optimization
↓
Tensor IR
↓
Schedule
↓
Lowering
↓
Target IR / Source
↓
Machine Code / GPU Kernel
```

---

# 8. 为什么 AI Compiler 更复杂？

因为 AI Workload 有几个突出特点：

```text
巨大 Tensor
大量矩阵计算
高度规则的 Loop
复杂 Memory Hierarchy
多种 Accelerator
低精度计算
Dynamic Shape
Specialized Matrix Unit
```

---

# 9. 硬件最终需要什么？

CPU：

```text
Load
Store
SIMD
FMA
Branch
Thread
```

GPU：

```text
blockIdx
threadIdx
global memory
shared memory
register
warp
mma
```

NPU：

```text
DMA
Tensor Layout
On-chip SRAM
Matrix Unit
Command Stream
```

高层模型抽象和硬件执行抽象差距巨大。

因此：

```text
Model Operator
↓
Compiler
↓
Hardware Program
```

是必需的。

---

# 10. AI Compiler 最核心的两个问题

第一：

```text
算什么？
```

第二：

```text
怎么在这块硬件上算最快？
```

---

# 11. “算什么”

例如：

```text
C = A @ B
```

数学含义确定。

---

# 12. “怎么计算”

同一个 MatMul：

```text
三重 for 循环

Tiled MatMul

SIMD MatMul

Multi-thread MatMul

CUDA Shared-memory MatMul

Tensor Core MatMul
```

都得到：

```text
C = A @ B
```

但性能：

```text
可以完全不同。
```

---

# 13. Algorithm 与 Schedule 分离

TVM 中非常重要的思想：

```text
Computation
≠
Schedule
```

Computation：

```text
定义结果是什么。
```

Schedule：

```text
定义如何执行。
```

---

# 14. 一个 MatMul

数学：

```text
C[i,j]
=
Σ A[i,k] × B[k,j]
```

最朴素：

```python
for i:
    for j:
        C[i, j] = 0
        for k:
            C[i, j] += A[i, k] * B[k, j]
```

---

# 15. 为什么朴素 MatMul 慢？

不是因为：

```text
数学不对。
```

而是：

```text
Cache Locality 差
SIMD 利用低
Parallelism 不够
Memory Reuse 差
GPU Mapping 不合理
```

---

# 16. Compiler 优化的核心

不是：

```text
改变答案。
```

而是：

> 在保持程序语义不变的前提下改变执行结构。

---

# 17. TVM 总体结构

可以先画：

```text
                  Apache TVM
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
      Relax        TensorIR       Runtime
        │             │             │
 Graph-level      Tensor-level      VM/AOT
 Optimization     Optimization      Module
        │             │             │
        └───────┬─────┘             │
                ▼                   │
              Target                │
                │                   │
                ▼                   │
              CodeGen               │
                └─────────┬─────────┘
                          ▼
                       Hardware
```

---

# 18. 当前 TVM 最重要的设计原则

当前官方 README 特别强调：

```text
Cross-level Design
```

即：

```text
Relax
+
TensorIR
```

可以一起存在：

```text
同一个 IRModule。
```

这样 Graph 层和 Tensor Kernel 层：

```text
不再完全割裂。
```

---

# 19. 为什么需要两层 IR？

假设模型：

```text
Conv
↓
ReLU
↓
MatMul
↓
Add
```

Graph 层需要知道：

```text
Operator Dependency
Tensor Shape
Dataflow
Function Boundary
```

---

# 20. Kernel 层需要知道：

```text
for loop
memory buffer
thread binding
tile
vectorize
shared memory
```

---

# 21. 如果只有 Graph IR

可以：

```text
Fuse Conv + ReLU
```

但很难表达：

```text
把 i loop split 成 32×8。
```

---

# 22. 如果只有 Loop IR

可以：

```text
Tile MatMul。
```

但很难理解：

```text
整个 Model Graph 哪两个 Operator 可以 Fuse。
```

---

# 23. 所以：

```text
Graph IR
+
Tensor IR
```

两层都需要。

---

# 24. Relax

官方定义：

> Relax 是 TVM 中用于表示和优化模型 Computational Graph 的高层抽象。

核心：

```text
Function
Dataflow
Tensor
Shape
Call
Operator
Control Flow
```

---

# 25. 一个最小 Relax 程序

概念：

```python
@R.function
def main(
    x: R.Tensor((1, 784), "float32"),
    w: R.Tensor((784, 256), "float32"),
):
    with R.dataflow():
        lv0 = R.matmul(x, w)
        gv = R.nn.relu(lv0)
        R.output(gv)

    return gv
```

这里描述：

```text
x
↓
MatMul
↓
ReLU
↓
Output
```

---

# 26. Relax 不关心 MatMul 内部三层 Loop

这一层只需要知道：

```text
MatMul
```

是什么 Operator。

所以：

```text
Relax = Graph-level abstraction。
```

---

# 27. Relax 的几个关键特点

当前官方强调：

```text
First-class Symbolic Shape
Multi-level Abstraction
Composable Transformation
```

---

# 28. Symbolic Shape

例如：

```text
x:
[n, 784]
```

其中：

```text
n
```

不是固定数字。

而是：

```text
Symbolic Dimension。
```

---

# 29. 为什么 Symbolic Shape 重要？

现代模型：

```text
Batch
Sequence Length
Image Size
```

可能动态变化。

Compiler 不能总假设：

```text
所有 Shape 固定。
```

---

# 30. Symbolic Shape 的价值

Compiler 可以知道：

```text
n 是动态
```

同时保留：

```text
x.shape[1] = 784
```

这类约束。

---

# 31. Dynamic Shape 不等于“Compiler 什么都不知道”

Compiler 可以知道：

```text
n > 0
n <= 4096
某两个维度相等
```

这叫：

```text
Shape Constraint。
```

---

# 32. Static Shape 与 Dynamic Shape

Static Shape：

```text
[1, 4096]
```

Compiler：

```text
知道每个维度。
```

更容易：

```text
AOT specialization
Memory planning
Kernel tuning。
```

Dynamic Shape：

```text
[n, 4096]
```

Compiler：

```text
不知道 n 的精确值。
```

需要：

```text
Symbolic Analysis
Dynamic Loop
Shape Constraint
Specialization。
```

---

# 33. LLM 的 Decode 是特殊 Shape

Decode：

```text
T = 1
```

Prefill：

```text
T >> 1。
```

所以 Runtime / Compiler 经常：

```text
为两种 Shape
准备不同执行路径。
```

---

# 34. Relax Dataflow

```text
R.dataflow()
```

描述：

```text
一段以 Tensor Dataflow 为核心的区域。
```

Compiler 可以在其中：

```text
分析依赖
做 Fusion
消除临时值。
```

---

# 35. IRModule

TVM 最重要的数据结构之一：

```text
IRModule。
```

它不是：

```text
单个函数。
```

而是：

> 一组相关 IR Function 的模块容器。

---

# 36. 一个 IRModule 可以同时包含

```text
Relax Function
+
TensorIR PrimFunc
+
External Function
```

---

# 37. 这是 Cross-level Optimization 的基础

例如：

```text
Relax main
```

调用：

```text
TIR matmul
TIR relu
```

都在同一个：

```text
IRModule。
```

---

# 38. PrimFunc

TensorIR 中核心：

```text
PrimFunc。
```

它表示：

> 一个 Primitive Tensor Function。

例如：

```text
MatMul Kernel。
```

---

# 39. Relax Function vs PrimFunc

Relax：

```text
Model / Graph level
```

PrimFunc：

```text
Loop / Buffer / Kernel level
```

---

# 40. `R.call_tir`

这是连接两层的关键。

概念：

```python
lv = R.call_tir(
    cls.matmul,
    (x, w),
    out_sinfo=...
)
```

意义：

```text
Relax Graph
↓
调用一个 TensorIR PrimFunc。
```

---

# 41. Cross-level IR

一个 Module：

```text
@R.function
main(...)
    ↓
call_tir(matmul)

@T.prim_func
matmul(...)
```

Graph 与 Kernel：

```text
同时存在。
```

---

# 42. 为什么 Cross-level 很强？

Compiler 可以：

```text
在 Graph 层识别 Fusion
```

又：

```text
在 TensorIR 层优化 Loop。
```

甚至：

```text
Graph 信息指导 Kernel Schedule。
```

---

# 43. 这就是现代 TVM 的核心方向

当前官方 README 已明确：

> 当前设计聚焦于 TensorIR + Relax 的 cross-level design。

因此学习 TVM：

```text
不要只学老 Relay。
```

---

# 44. Relay 是什么？

Relay：

```text
TVM 历史上非常重要的 Graph-level IR。
```

大量经典教程：

```text
PyTorch → Relay → TVM
```

---

# 45. 现在为什么主要学 Relax？

当前 TVM 主线：

```text
Relax
```

已经成为高层模型 Graph 表示和优化重点。

所以：

```text
新学习路线：
Relax 优先。
```

---

# 46. 旧资料怎么办？

看到：

```text
relay.frontend
relay.build
tvm.tir
```

不要认为完全没价值。

其中：

```text
Graph Compilation
Schedule
TIR
CodeGen
```

思想仍然重要。

但 API：

```text
需要对照当前文档。
```

---

# 47. 当前 TensorIR 的一个重要变化

2026 当前主线：

```text
原来的 tir
```

正在进一步拆成：

```text
tirx
+
s_tir。
```

---

# 48. `tirx`

当前官方定义：

```text
Core TensorIR IR + Lowering。
```

包含：

```text
PrimFunc
Buffer
SBlock
Expression
Statement
Analysis
Lowering Pass
```

---

# 49. `s_tir`

Schedulable TIR：

```text
Schedule Primitives
Tensor Intrinsics
MetaSchedule
DLight
```

即：

> “如何调度 / 优化 TensorIR”这一层。

---

# 50. 简化理解

```text
tirx
=
What the low-level tensor program is

s_tir
=
How to transform / schedule it
```

---

# 51. 为什么拆开？

可以理解为：

```text
IR Core
```

和：

```text
Schedule / Tuning System
```

职责更加清晰。

---

# 52. TVMScript 当前写法

当前 TensorIR TVMScript：

```python
from tvm.script import tirx as T
```

例如：

```python
@T.prim_func
def add(...):
    ...
```

---

# 53. 注意旧教程

旧代码常见：

```python
from tvm.script import tir as T
```

这是版本差异。

所以：

> 学习概念时看旧资料没问题；真正跑代码时以当前 checkout 对应文档为准。

---

# 54. TensorIR

TensorIR：

> 描述 Tensor Kernel 的 Loop、Buffer、Memory、Thread Binding 等低层结构。

---

# 55. 一个 Vector Add

数学：

```text
C[i] = A[i] + B[i]
```

TensorIR 概念：

```python
for i in range(N):
    C[i] = A[i] + B[i]
```

---

# 56. 一个 MatMul

```python
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j]
```

这就是：

```text
Loop Nest。
```

---

# 57. Spatial Axis

`i`：

```text
不同 i 对应不同输出元素。
```

称：

```text
Spatial Axis。
```

---

# 58. Reduction Axis

`k`：

```text
多个 k 累加到同一个 C[i,j]。
```

称：

```text
Reduction Axis。
```

---

# 59. 为什么要区分 Spatial / Reduction？

因为：

```text
Parallelization
Reorder
Tiling
Tensorization
```

对它们的约束不同。

---

# 60. Buffer

TensorIR 不只看到：

```text
Tensor Shape。
```

而是：

```text
Buffer
Memory Scope
Index
Stride
Access Region。
```

---

# 61. 为什么 Buffer 很关键？

高性能计算本质之一：

```text
让数据待在更快的 Memory。
```

比如 GPU：

```text
Global Memory
↓
Shared Memory
↓
Register
```

---

# 62. TensorIR Block

Block：

```text
一个具有明确读写区域和迭代轴的计算区域。
```

例如：

```text
matmul block。
```

Compiler 可以知道：

```text
Reads:
A[i,k]
B[k,j]

Writes:
C[i,j]
```

---

# 63. Dependency Analysis

由于有：

```text
Block Read / Write Region
```

Compiler 可以分析：

```text
Producer
Consumer
Data Dependency。
```

---

# 64. Schedule

Schedule：

> 在保持程序语义等价的前提下，对 TensorIR 执行结构进行变换。

---

# 65. 最核心的理解

```text
Computation:
C = A @ B
```

不变。

Schedule：

```text
Loop 怎样组织
数据放哪里
线程怎么映射
```

可以变化。

---

# 66. Schedule Design Space

例如 MatMul：

```text
tile size = 8?
16?
32?

vector width = 4?
8?

threads = 128?
256?

shared memory?
register block?
```

组合：

```text
非常多。
```

---

# 67. 这就是优化搜索空间

```text
Design Space。
```

Manual Schedule：

```text
人找一个点。
```

Auto-tuning：

```text
系统搜索很多点。
```

---

# 68. `split`

假设：

```text
for i in 0..1024
```

Split：

```text
i_outer
i_inner
```

例如：

```text
1024
=
32 × 32。
```

---

# 69. 为什么 split？

为了：

```text
Tiling
Thread Mapping
Vectorization
Cache。
```

---

# 70. `reorder`

原：

```text
i
j
k
```

可以重排：

```text
i_outer
j_outer
k_outer
i_inner
j_inner
k_inner
```

---

# 71. 为什么 reorder？

改变：

```text
Memory Access Order
Reuse
Parallelism。
```

---

# 72. `fuse`

两个 Loop：

```text
i
j
```

合成：

```text
fused = i * N + j。
```

---

# 73. 为什么 fuse？

适合：

```text
Flatten iteration space
Parallelize
GPU thread mapping。
```

---

# 74. `parallel`

告诉 CPU：

```text
不同 Iteration
可以多线程并行。
```

---

# 75. `vectorize`

把：

```text
多个 Scalar Iteration
```

变：

```text
SIMD vector instruction。
```

例如：

```text
8×FP32
```

一次处理。

---

# 76. CPU SIMD

常见：

```text
AVX
AVX2
AVX512
NEON
SVE
```

Compiler 需要把 Loop 组织成：

```text
适合 vectorize 的形式。
```

---

# 77. 为什么 ARM Edge 也需要 Compiler Schedule？

RK3576 CPU：

```text
ARM Cortex
```

如果只写 Scalar：

```text
性能浪费。
```

Compiler 需要：

```text
Vectorize → NEON / SVE-friendly code。
```

---

# 78. `unroll`

例如：

```text
for k in range(4)
```

展开：

```text
body(0)
body(1)
body(2)
body(3)
```

减少：

```text
Loop Control Overhead。
```

但：

```text
Code Size ↑。
```

---

# 79. GPU `bind`

GPU：

```text
blockIdx.x
threadIdx.x
```

需要将 Loop：

```text
绑定到 GPU Execution Hierarchy。
```

---

# 80. GPU Grid

```text
Grid
↓
Block
↓
Thread
```

Schedule 负责：

```text
Loop
→
Block / Thread。
```

---

# 81. Tiling

Tiling：

```text
把大问题切成小块。
```

MatMul：

```text
A
B
```

不直接一次访问全部。

而：

```text
Tile A
Tile B
↓
Compute Tile C。
```

---

# 82. 为什么 Tiling 重要？

因为高速 Memory：

```text
容量小。
```

需要：

```text
把正在计算的小 Tile
放进 Cache / Shared Memory。
```

---

# 83. Cache Blocking

CPU：

```text
L1 / L2 / L3
```

GPU：

```text
Shared Memory / Register。
```

Tiling 目标：

```text
提高 Data Reuse。
```

---

# 84. Data Reuse

MatMul 中：

```text
A[i,k]
```

会被多个：

```text
j
```

重复使用。

如果每次都从 DRAM 读：

```text
浪费带宽。
```

---

# 85. Arithmetic Intensity 再次出现

06 章：

```text
Operations / Bytes Moved。
```

Tiling：

```text
提高 Data Reuse
↓
减少 Bytes Moved per FLOP
↓
Arithmetic Intensity ↑。
```

---

# 86. Compiler 与推理性能终于连起来

06：

```text
Decode 为什么 Memory-Bound？
```

本章：

```text
Compiler 如何减少 Memory Traffic？
```

---

# 87. Shared Memory Tiling

GPU：

```text
Global Memory
↓
Load Tile
↓
Shared Memory
↓
many MAC
```

减少：

```text
Global Memory Traffic。
```

---

# 88. `cache_read`

Schedule 可以：

> 插入一个读缓存阶段。

例如：

```text
A global
↓
A_shared
↓
Compute。
```

---

# 89. `cache_write`

把：

```text
中间结果
```

先写：

```text
Local / Shared Buffer。
```

然后再：

```text
write back。
```

---

# 90. Register Blocking

更小 Tile：

```text
放 Register。
```

让：

```text
一个线程重复利用。
```

---

# 91. 为什么 Register 最快？

它最靠近：

```text
Compute Unit。
```

但容量：

```text
非常有限。
```

---

# 92. Register Pressure

使用太多 Register：

```text
每个 Thread 占资源过多
```

导致：

```text
Occupancy ↓。
```

所以：

```text
不是 Register 越多越好。
```

---

# 93. Schedule 是典型 Tradeoff 问题

```text
更大 Tile
→
Reuse ↑

但：
Shared Memory ↑
Register ↑
Occupancy ↓。
```

这也是为什么自动调优有价值。

人很难：

```text
根据公式直接找到完美参数。
```

---

# 94. 为什么 MatMul 复杂？

高性能 GEMM 常有多层 Tile：

```text
Global Tile
↓
Shared Tile
↓
Warp Tile
↓
Register Tile
```

---

# 95. 所以后面 CUTLASS 会很自然

CUTLASS：

```text
就是高度工程化的 GEMM tiling / data movement / tensor core system。
```

TVM Schedule：

```text
让 Compiler 也能表达类似优化。
```

---

# 96. `compute_at`

经典 Schedule 思想：

> 控制 Producer 在 Consumer Loop 的哪个位置计算。

影响：

```text
Intermediate Tensor Lifetime
Memory Reuse
Fusion。
```

---

# 97. `reverse_compute_at`

从 Consumer 角度：

```text
调整写回 / consumer compute placement。
```

本章知道位置即可。

---

# 98. Tensorization

Tensorization：

> 将一段匹配特定计算模式的 Loop Program 替换成硬件专用 Intrinsic。

---

# 99. 例如：

普通：

```text
for m
 for n
  for k
   C += A*B
```

匹配：

```text
16×16×16 Matrix Multiply。
```

然后：

```text
替换为 Tensor Core MMA 指令。
```

---

# 100. Tensor Intrinsic

Compiler 需要知道：

```text
这段计算模式
```

对应：

```text
wmma.mma
AMX tile dot
某 NPU matrix instruction。
```

---

# 101. Tensorization 为什么是 AI Compiler 的关键？

AI Hardware：

```text
最强的算力
```

通常不在普通 Scalar ALU。

而在：

```text
Tensor Core
Matrix Engine
NPU MAC Array。
```

Compiler 必须：

```text
把 Tensor Program 匹配到这些硬件单元。
```

---

# 102. CPU 也有类似思想

例如：

```text
Intel AMX
ARM SME / SVE
```

也是：

```text
特殊矩阵 / 向量能力。
```

---

# 103. Tensorization 是理解 NPU Compiler 的关键入口

NPU：

```text
Matrix MAC Array
```

就是专门计算：

```text
Tensor Block。
```

可以把 NPU Compiler 的重要工作之一理解成：

```text
Tensor Program
↓
Tile
↓
Layout
↓
Hardware Matrix Primitive。
```

---

# 104. Legalization

Relax：

```text
R.matmul
R.nn.relu
```

是高层 Operator。

真正执行前：

```text
需要 lower / legalize。
```

---

# 105. `LegalizeOps`

官方 Relax 教程常先做：

```text
LegalizeOps。
```

概念：

```text
High-level Relax Op
↓
call_tir
↓
TensorIR implementation。
```

---

# 106. 例如 MatMul

```text
R.matmul
```

经过 Legalization：

```text
生成 / 绑定
TensorIR PrimFunc。
```

---

# 107. 为什么叫 Legalize？

因为：

```text
高层 Operator
```

要变成：

```text
Compiler 后续能够具体执行 / Lower 的合法低层形式。
```

---

# 108. Graph Optimization

Relax 层可以做：

```text
Operator Fusion
Constant Folding
Dead Code Elimination
Layout Rewrite
Function Inlining
Shape Optimization
Pattern Rewrite
```

---

# 109. Operator Fusion

例如：

```text
MatMul
↓
Bias Add
↓
ReLU
```

如果分开：

```text
MatMul writes DRAM
Add reads/writes DRAM
ReLU reads/writes DRAM。
```

---

# 110. Fuse 后

```text
MatMul + Bias + ReLU
```

可以：

```text
减少 intermediate memory traffic
减少 kernel launch。
```

---

# 111. Graph Fusion vs Kernel Fusion

两者相关但不完全一样。

Graph Fusion：

```text
高层决定哪些 Op 组成一个 Region。
```

Kernel Fusion：

```text
最终生成一个 / 更少的底层 Kernel。
```

---

# 112. Constant Folding

如果：

```text
C = 2 + 3
```

Compile Time：

```text
直接变 5。
```

无需 Runtime 计算。

---

# 113. Weight-related Constant Folding

某些：

```text
固定 reshape
固定 transpose
常量 scale
```

可以：

```text
在编译阶段提前处理。
```

---

# 114. Dead Code Elimination

某个 Tensor：

```text
计算了
但最终没人用。
```

Compiler：

```text
删掉。
```

---

# 115. Layout Transformation

例如：

```text
NCHW
→
NHWC
```

或者：

```text
Blocked Layout。
```

目的是：

```text
匹配 Hardware / Kernel。
```

---

# 116. Layout 不只是模型接口问题

在 AI Compiler 中：

```text
Layout
```

本身就是：

```text
Optimization Variable。
```

---

# 117. Dynamic Shape

LLM：

```text
Sequence Length T
```

天然动态。

Compiler 不能简单：

```text
为 T=128 写死所有代码。
```

---

# 118. 可能策略

```text
Symbolic Shape
Shape Range
Specialization
Multiple Kernels
Dynamic Loop
Padding。
```

---

# 119. Specialization

如果常见：

```text
T=1
```

Decode。

可以：

```text
专门编一个 Decode Kernel。
```

Prefill：

```text
T=512
```

另一套。

---

# 120. 这就是 MLC 的意义之一

前面看到：

```text
batch_prefill
batch_decode
batch_verify
```

可以分别编译优化。

---

# 121. Cross-level Optimization 示例

Graph 知道：

```text
这个 MatMul 后接 GELU。
```

TensorIR 知道：

```text
MatMul Loop。
```

Compiler 可以：

```text
将 GELU 融合进 MatMul epilogue。
```

---

# 122. 这比只做 Graph 或只做 Kernel 更强

因为：

```text
Graph Semantic
+
Kernel Structure
```

都可用。

---

# 123. DLight

TVM 中：

```text
DLight
```

可以理解为：

> 一组自动应用的、面向常见 Tensor Workload 的高性能 Schedule Rule。

---

# 124. 为什么需要 DLight？

很多 Operator：

```text
MatMul
GEMV
Reduction
Elementwise
```

存在成熟优化模板。

不必：

```text
每次从零手写 Schedule。
```

---

# 125. DLight 与 Manual Schedule

Manual：

```text
你自己 split / reorder / bind。
```

DLight：

```text
系统根据 Pattern
应用预定义规则。
```

---

# 126. DLight 适合

```text
快速得到较好性能
默认 GPU Schedule
Compiler Pipeline。
```

---

# 127. MetaSchedule

MetaSchedule：

> TVM 自动搜索高性能 Schedule 的系统。

---

# 128. 核心思想

```text
TensorIR
↓
Design Space
↓
Generate Candidates
↓
Compile
↓
Benchmark
↓
Cost Model
↓
Search
↓
Best Schedule
```

---

# 129. Auto-tuning 为什么必须真实测？

GPU：

```text
理论上看起来合理
```

的 Tile Size：

```text
实际不一定最快。
```

因为：

```text
Occupancy
Register Pressure
Shared Memory
Cache
Instruction Mix
```

非常复杂。

---

# 130. Cost Model

MetaSchedule 不希望：

```text
穷举所有组合。
```

所以：

```text
学习哪些候选更有希望。
```

---

# 131. Search Space

由：

```text
Schedule Rule
```

生成。

例如：

```text
Tile sizes
Thread binding
Vectorization
Unroll。
```

---

# 132. Manual vs DLight vs MetaSchedule

```text
Manual
=
人直接设计 Schedule

DLight
=
规则驱动的自动 Schedule

MetaSchedule
=
搜索 + 测量 + Cost Model
```

---

# 133. 推荐学习顺序

```text
Manual
↓
DLight
↓
MetaSchedule。
```

否则：

```text
直接 AutoTune
```

很难知道系统在搜索什么。

---

# 134. Target

TVM Target：

> 描述代码将运行在哪种硬件和后端上。

例如：

```text
llvm

cuda

vulkan

opencl

metal
```

---

# 135. Target 不只是一个字符串

Target 还可以包含：

```text
Architecture
Features
Thread Limits
Memory Properties
Device Attributes。
```

---

# 136. 例如 NVIDIA A100

概念：

```text
target:
cuda
arch:
sm_80
```

Compiler 可以知道：

```text
Tensor Core Generation
Shared Memory Capacity
Warp properties。
```

---

# 137. 为什么 Target-aware Optimization？

同一个：

```text
MatMul
```

在：

```text
x86 CPU
ARM CPU
NVIDIA GPU
Apple GPU
```

最佳 Schedule：

```text
完全不同。
```

---

# 138. ARM Target

CPU Target：

```text
llvm
```

同时可以指定：

```text
ARM architecture
NEON feature
```

最终：

```text
LLVM
↓
AArch64 Machine Code。
```

---

# 139. Lowering

Lowering：

> 从高层抽象逐步变成更低层、更接近硬件的程序表示。

---

# 140. 一个概念链

```text
Relax MatMul
↓
TensorIR MatMul
↓
Scheduled TensorIR
↓
Flatten Buffer
↓
Lower Intrinsics
↓
Thread / Memory Lowering
↓
Target Code
```

---

# 141. 为什么要多阶段 Lowering？

一次直接：

```text
PyTorch
→
PTX
```

几乎无法维护。

中间 IR：

```text
每层负责一种抽象。
```

---

# 142. TIR Lowering Pass

例如：

```text
Buffer Flattening
Intrinsic Lowering
Thread Lowering
Vector Lowering
Storage Rewrite
```

具体当前 pipeline 会随版本变化。

---

# 143. Code Generation

当前 TVM CodeGen 官方架构明确：

```text
Code Generation
=
TIR PrimFunc
→
Target Executable Code。
```

---

# 144. 当前 `tvm.compile()`

现代 TVM 主线中：

```text
tvm.compile()
```

可以统一触发：

```text
Relax Phase
+
TIR Phase。
```

旧教程中：

```text
relax.build()
```

仍然大量出现。

---

# 145. 当前 CodeGen 两阶段

概念：

```text
Relax
↓
Graph optimization / VM codegen

TIR PrimFunc
↓
TIR pipeline
↓
Target codegen。
```

---

# 146. Host / Device Split

GPU Program：

```text
Host CPU Code
+
Device GPU Kernel。
```

Compiler 要拆开。

---

# 147. Host Code

负责：

```text
Allocate
Launch Kernel
Pass Arguments
Control Flow。
```

---

# 148. Device Code

负责：

```text
真正并行计算。
```

例如：

```text
CUDA kernel。
```

---

# 149. CodeGen Families

当前 TVM CodeGen 文档把后端粗分：

```text
LLVM Family

Source Family。
```

---

# 150. LLVM Family

例如：

```text
x86
ARM
ROCm 部分路径
NVPTX。
```

概念：

```text
TIR
↓
LLVM IR
↓
Machine Code / ISA。
```

---

# 151. CPU LLVM

```text
TIR
↓
LLVM IR
↓
x86 / ARM Machine Code。
```

---

# 152. Source CodeGen

某些 GPU Backend：

```text
生成 Source / Target IR
```

再交外部 Compiler。

例如：

```text
CUDA C
OpenCL C
Metal
WGSL。
```

---

# 153. CUDA

概念：

```text
TIR
↓
CUDA CodeGen
↓
CUDA C / PTX pipeline
↓
GPU Code。
```

---

# 154. Vulkan

```text
TIR
↓
SPIR-V。
```

---

# 155. Metal

```text
TIR
↓
Metal Shading Language。
```

---

# 156. WebGPU

```text
TIR
↓
WGSL。
```

---

# 157. 为什么 TVM 能跨平台？

高层：

```text
Relax
TensorIR
```

与：

```text
Target CodeGen
```

分离。

---

# 158. 但跨平台不意味着同一 Schedule 通用

CPU 最佳 Schedule：

```text
不等于 GPU。
```

所以：

```text
Target-specific Scheduling
```

仍然必要。

---

# 159. Runtime Module

编译产物在 TVM 中常包装成：

```text
runtime.Module。
```

它可以包含：

```text
Host Code
Device Code
External Library
Metadata。
```

---

# 160. Module Import

例如：

```text
Host Module
```

可以：

```text
import Device Module。
```

形成完整可执行 artifact。

---

# 161. PackedFunc

TVM Runtime 的核心统一调用接口之一：

```text
PackedFunc。
```

用于：

```text
Python
C++
Runtime Module
External Backend
```

之间统一调用。

---

# 162. 为什么 PackedFunc 很重要？

Compiler 生成：

```text
各种 Function。
```

Runtime 需要：

```text
统一 ABI / 调用机制。
```

---

# 163. Relax VM

Relax VM：

> 执行已编译 Relax Program 的 Runtime Control Engine。

---

# 164. Relax VM 自己算 MatMul 吗？

不是。

当前官方架构说明：

```text
VM 自己几乎只做 Control / Dispatch。
```

实际数学：

```text
交给编译好的 TIR Kernel
或 External Library。
```

---

# 165. VM Bytecode

VM：

```text
Call
Ret
Goto
If
```

这类少量指令：

```text
编排执行。
```

---

# 166. Relax VM Pipeline

```text
IRModule
├── Relax
└── TIR
↓
tvm.compile
↓
Relax Bytecode
+
Compiled TIR Kernels
+
Constants
↓
VMExecutable
↓
VirtualMachine。
```

---

# 167. 所以：

```text
Relax
=
What to execute at graph/control level

TIR Kernel
=
Actual tensor math

VM
=
Orchestration at runtime。
```

---

# 168. 这和 vLLM Scheduler 不一样

Relax VM：

```text
执行一个编译程序内部的 control flow。
```

vLLM Scheduler：

```text
调度多个动态 Request。
```

都叫 Runtime Control：

```text
但抽象层不同。
```

---

# 169. AOT

Ahead-of-Time：

```text
Compile on Host
↓
Artifact
↓
Deploy Runtime。
```

---

# 170. 为什么端侧喜欢 AOT？

设备：

```text
不需要完整 Compiler。
```

只带：

```text
小 Runtime
+
Compiled Module。
```

---

# 171. MLC 就非常依赖这个思想

```text
PC
↓
Compile Model Library
↓
Android / iOS / Web。
```

---

# 172. RPC

TVM RPC：

> 可以把编译产物发送到远程设备并运行 / benchmark。

---

# 173. 为什么 RPC 对 Compiler 很重要？

Auto-tuning：

```text
Compiler 在 PC。
```

目标设备：

```text
ARM board。
```

可以：

```text
编译候选
↓
发到 Board
↓
真实 Benchmark
↓
返回速度。
```

---

# 174. 这对 RK3576 学习也非常有启发

即使未来不是用 TVM 编 RKNPU：

```text
Host Compiler
↓
Remote Device
↓
Benchmark
```

是端侧 Compiler 的经典 Workflow。

---

# 175. BYOC

BYOC：

```text
Bring Your Own Codegen。
```

目的：

> 将一部分 Relax Graph 交给外部 Backend / Vendor Compiler。

---

# 176. 为什么需要 BYOC？

TVM 不可能：

```text
自己实现所有硬件的所有 Kernel。
```

很多硬件已有：

```text
cuBLAS
cuDNN
CUTLASS
Vendor NPU SDK。
```

---

# 177. BYOC 流程

当前官方架构可以概括：

```text
Relax IRModule
↓
Pattern Match
↓
FuseOpsByPattern
↓
Annotate Composite / Codegen
↓
RunCodegen
↓
External Backend Compiles Subgraph
↓
Remaining Graph continues TVM compilation
↓
Link
↓
Deployable Artifact。
```

---

# 178. Pattern

例如：

```text
MatMul + Bias + ReLU
```

某外部 Backend：

```text
有一个 fused implementation。
```

Compiler 可以：

```text
识别整个 Pattern。
```

---

# 179. Partition

Graph：

```text
A
↓
B
↓
C
↓
D
```

可能：

```text
A/B → TVM
C/D → External Backend。
```

---

# 180. External CodeGen

External Backend：

```text
编译被划出的 Subgraph。
```

返回：

```text
Runtime Module。
```

---

# 181. 为什么 BYOC 和 NPU 很像？

Vendor NPU：

```text
只支持部分 Operator / Pattern。
```

可以：

```text
Supported Subgraph → NPU

Unsupported → CPU。
```

---

# 182. 这和 ONNX Runtime EP 也很像

前面学过：

```text
Execution Provider Graph Partition。
```

BYOC：

```text
也是“把子图交给特定 Backend”。
```

---

# 183. 但它们不是同一个实现

只是：

```text
系统设计思想类似。
```

---

# 184. TVM 当前外部库示例

官方 BYOC 文档当前列举：

```text
cuBLAS
CUTLASS
cuDNN
DNNL
```

等 Backend。

---

# 185. 为什么调用 cuBLAS 可能比自己生成 MatMul 更好？

Vendor Library：

```text
多年手工优化
大量架构特化
```

所以 Compiler 不应该：

```text
为了“纯编译”拒绝成熟 Library。
```

---

# 186. Compiler 的目标

不是：

```text
所有 Kernel 都自己写。
```

而是：

> 选择最优执行实现。

---

# 187. Library Dispatch

例如 MatMul：

```text
small GEMV
→ custom generated kernel

large FP16 GEMM
→ cuBLAS

specific fused GEMM
→ CUTLASS。
```

---

# 188. 这就是现代 AI Compiler 的现实

```text
Generated Code
+
External Library
+
Vendor Runtime
```

混合。

---

# 189. AI Compiler vs Runtime

Compiler：

```text
分析 / 变换 / 生成。
```

Runtime：

```text
加载 / 执行 / 调度。
```

---

# 190. 但边界会融合

JIT Runtime：

```text
运行时编译。
```

Runtime 可能：

```text
根据 Shape 选择 Kernel。
```

所以：

```text
不是绝对分割。
```

---

# 191. TVM vs PyTorch

PyTorch：

```text
Model Development
Training
Eager Runtime。
```

TVM：

```text
Compile / Optimize / Deploy。
```

---

# 192. 当前 PyTorch → TVM 的重要路径

现代路径之一：

```text
PyTorch
↓
torch.export
↓
ExportedProgram
↓
TVM Relax Frontend
↓
Relax IR。
```

---

# 193. 当前 TVM 有 PyTorch Relax Frontend

典型接口：

```text
tvm.relax.frontend.torch.from_exported_program
```

---

# 194. 为什么 `torch.export` 很适合 Compiler？

它比任意 Python：

```text
更接近可静态分析的 Tensor Graph。
```

---

# 195. ONNX → TVM

另一条：

```text
PyTorch
↓
ONNX
↓
TVM ONNX Frontend
↓
Relax。
```

---

# 196. 两条路线比较

```text
PyTorch → torch.export → Relax

PyTorch → ONNX → Relax
```

都合理。

---

# 197. 为什么不用 ONNX 统一所有东西？

因为：

```text
Framework-native Export
```

可能保留：

```text
更丰富 Shape / Operator 信息。
```

而 ONNX：

```text
更通用。
```

---

# 198. Frontend Compatibility 永远是现实问题

某 Operator：

```text
PyTorch 有
```

TVM Frontend：

```text
还没 Converter。
```

就可能：

```text
Import Failed。
```

---

# 199. 所以 AI Compiler 不只优化 Kernel

还大量做：

```text
Frontend Operator Coverage
Shape Semantics
Dynamic Control Flow
Dtype
Layout。
```

---

# 200. TVM vs ONNX Runtime

ONNX Runtime：

```text
Graph Runtime
+
Execution Provider。
```

TVM：

```text
Compiler Framework
+
Graph IR
+
Tensor IR
+
Code Generation。
```

---

# 201. TVM 更适合回答：

```text
“这个 Kernel
到底怎么生成？”
```

---

# 202. ONNX Runtime 更适合回答：

```text
“这个 ONNX Graph
如何高效执行？”
```

---

# 203. TVM vs llama.cpp

llama.cpp：

```text
LLM-specific Runtime
Architecture implemented in C++
ggml backend。
```

TVM：

```text
General ML Compiler
IR-driven
Code generation。
```

---

# 204. 共同点

```text
Tensor
Graph
Backend
Kernel
Target。
```

---

# 205. 最大区别

llama.cpp：

```text
Runtime-oriented
手写 model architecture / backend。
```

TVM：

```text
Compiler-oriented
Program transformation / codegen。
```

---

# 206. TVM vs vLLM

vLLM：

```text
Request Scheduler
KV Cache
Continuous Batching
Serving。
```

TVM：

```text
Model / Kernel Compilation。
```

---

# 207. 两者可以合作吗？

当然。

一个系统完全可能：

```text
vLLM Scheduler
↓
Compiled Model / Kernel
↓
TVM / Triton / CUTLASS。
```

---

# 208. TVM vs MLC-LLM

关系最直接。

```text
MLC-LLM
=
LLM vertical stack

TVM
=
underlying compiler infrastructure。
```

---

# 209. MLC 中已经见过：

```text
Relax
IRModule
Compiler Pass
Target
TIR。
```

本章只是把：

```text
底层原理展开。
```

---

# 210. TVM vs RKNN-Toolkit2

在抽象层上：

```text
都有：
Model Import
Graph Analysis
Operator Conversion
Optimization
Quantization
Target Compilation
Runtime Artifact。
```

---

# 211. 但是：

TVM：

```text
Open General Compiler Framework。
```

RKNN-Toolkit2：

```text
Rockchip Vendor Toolchain
Target = RKNPU。
```

---

# 212. `.rknn` 可以类比什么？

概念上：

```text
Compiler 生成的 Target-specific Deployment Artifact。
```

类似：

```text
TVM compiled module
```

但：

```text
内部格式完全不同。
```

---

# 213. `.rknn` 里面可能有什么？

因为：

```text
Target
=
Rockchip RKNPU。
```

其中可能包含：

```text
Vendor-specific Graph
Quant Parameters
Memory Plan
Operator Mapping
Hardware Metadata。
```

所以 `.rknn` 不能像 ONNX 一样通用。

---

# 214. 为什么学 TVM 后 RKNN 会清晰？

当 RKNN 报：

```text
Unsupported Op
Layout Error
Quantization Error
Dynamic Shape Error
```

你会知道问题处于：

```text
Frontend
Graph IR
Lowering
Operator Mapping
Target Backend
```

哪个阶段。

---

# 215. Vendor NPU Compiler

可以抽象：

```text
ONNX
↓
Frontend
↓
Vendor Graph IR
↓
Graph Optimization
↓
Quantization
↓
Operator Lowering
↓
NPU Mapping
↓
Memory Planning
↓
Command / Model Generation
↓
Runtime Artifact。
```

---

# 216. 这和 TVM 主线高度类似

只是：

```text
Target 更专用。
```

---

# 217. 为什么 NPU Compiler 更严格？

CPU/GPU：

```text
通用计算能力较强。
```

NPU：

```text
Operator Set
Dtype
Shape
Layout
Memory
```

限制更强。

---

# 218. 所以 NPU Compile Error 很常见

不是：

```text
SDK 太差。
```

很多时候是：

> 硬件本身只支持有限执行模型。

---

# 219. AI Compiler 必须做 Hardware Mapping

例如：

```text
MatMul
```

要映射到：

```text
NPU Matrix Unit。
```

---

# 220. Unsupported Op

例如：

```text
某特殊 Scatter。
```

NPU 无硬件 / 编译支持。

可能：

```text
CPU Fallback
```

或：

```text
Compile Fail。
```

---

# 221. RKNN Quantization Failure 怎么理解？

可能处于：

```text
Calibration
Dtype Conversion
Scale Propagation
Operator Constraint
Accuracy Constraint。
```

---

# 222. Graph Partition

因此：

```text
Supported
→ NPU

Unsupported
→ CPU。
```

---

# 223. Fallback 的成本

```text
NPU
↓
DDR / Copy
↓
CPU
↓
DDR
↓
NPU。
```

可能：

```text
性能很差。
```

---

# 224. Fallback 为什么格外贵？

因为：

```text
Tensor Transfer
Layout Conversion
Synchronization
Memory Copy
```

可能远比：

```text
那个 Op 本身
```

还贵。

---

# 225. Compiler 性能优化的四个层次

```text
Graph Level
Tensor Level
Kernel Level
Hardware Level。
```

---

# 226. Graph Level

```text
Fusion
Constant Folding
Layout
Partition
Quantization Propagation。
```

---

# 227. Tensor Level

```text
Loop
Tile
Cache
Vectorization
Thread Mapping。
```

---

# 228. Kernel Level

```text
Instruction
Register
Shared Memory
Tensor Core
Warp。
```

---

# 229. Hardware Level

```text
Cache
Bandwidth
Compute Unit
SRAM
DMA
ISA。
```

---

# 230. Quantization Compiler

量化加入以后：

```text
FP32 / FP16
↓
INT8 / INT4
```

Compiler 还需要处理：

```text
Scale
Zero Point
Dtype Propagation
Quantize / Dequantize
Kernel Selection。
```

---

# 231. 这和 RKNN 直接相关

RKNN：

```text
Calibration Dataset
↓
Quantization
↓
Target NPU Kernel。
```

就是 Compiler Pipeline 的一部分。

---

# 232. 一个 MatMul 完整旅程

---

# 233. Step 1：PyTorch

```python
y = torch.matmul(x, w)
```

---

# 234. Step 2：Export

```text
torch.export
```

得到：

```text
Tensor Graph。
```

---

# 235. Step 3：TVM Frontend

导入：

```text
Relax。
```

---

# 236. Step 4：Relax

```text
R.matmul(x, w)
```

---

# 237. Step 5：Legalize

```text
R.matmul
↓
TensorIR PrimFunc。
```

---

# 238. Step 6：Primitive Loop

```text
for i
 for j
  for k
   C[i,j] += A[i,k] * B[k,j]
```

---

# 239. Step 7：Schedule

CPU：

```text
split
reorder
parallel
vectorize
cache。
```

GPU：

```text
tile
bind block
bind thread
shared memory
register。
```

---

# 240. Step 8：Tensorize

如果 Target：

```text
Tensor Core。
```

匹配：

```text
MMA intrinsic。
```

---

# 241. Step 9：Lower

```text
高层 TensorIR
↓
低层 Target-friendly TIR。
```

---

# 242. Step 10：CodeGen

CPU：

```text
LLVM
↓
Machine Code。
```

GPU：

```text
CUDA / PTX
or
SPIR-V。
```

---

# 243. Step 11：Runtime

```text
Load Module
↓
Allocate Tensor
↓
Call Kernel
↓
Return Output。
```

---

# 244. 这就是 AI Compiler 全链路

```text
Model
↓
Graph
↓
Tensor Program
↓
Schedule
↓
Code
↓
Hardware。
```

---

# 245. 一个非常重要的问题：为什么 Compiler 不能只优化 FLOPs？

两个实现：

```text
FLOPs 完全相同。
```

但：

```text
Memory Access
```

可能完全不同。

---

# 246. 这就是 06 章的 Memory-Bound 再次出现

Compiler 不只是：

```text
减少计算。
```

还要：

```text
减少 Data Movement。
```

---

# 247. Data Movement

GPU：

```text
HBM
→
L2
→
Shared Memory
→
Register。
```

---

# 248. 速度差异往往来自

```text
数据在哪里？
什么时候搬？
搬多少次？
能不能复用？
```

---

# 249. 所以 Tiling 是核心

因为：

```text
Tiling
=
控制 Data Reuse。
```

---

# 250. FlashAttention 和 TVM 的关系

FlashAttention：

```text
Attention-specific IO-aware Algorithm + Kernel。
```

TVM：

```text
Compiler Infrastructure。
```

理论上可以：

```text
用 TensorIR / Schedule
表达类似 Tiling / Fusion。
```

---

# 251. Triton 与 TVM 的关系

Triton：

```text
GPU Kernel Programming Language / Compiler。
```

TVM：

```text
End-to-end ML Compiler。
```

---

# 252. Triton 更靠近：

```text
Kernel Layer。
```

TVM 更跨：

```text
Graph → Kernel → Runtime。
```

---

# 253. CUTLASS 与 TVM

CUTLASS：

```text
高性能 CUDA Linear Algebra Templates / Kernels。
```

TVM 可以：

```text
自己生成 Kernel
```

或：

```text
BYOC / dispatch CUTLASS。
```

---

# 254. 所以下一章为什么要学 GPU Kernel？

因为 TVM 已经告诉我们：

```text
Schedule
↓
Kernel。
```

下一步需要知道：

```text
一个好 Kernel 到底长什么样。
```

---

# 255. MetaSchedule 为什么和 Benchmark 紧密相关？

Compiler 无法只看代码：

```text
完美预测性能。
```

因此：

```text
Compile Candidate
↓
Run on Hardware
↓
Measure。
```

---

# 256. 真实硬件是最终裁判

同样：

```text
Tile=16
Tile=32
```

哪个更快：

```text
要测。
```

---

# 257. 这和工程中的 Benchmark 思维完全一致

llama.cpp：

```text
llama-bench。
```

vLLM：

```text
serving benchmark。
```

TVM：

```text
kernel auto-tuning benchmark。
```

---

# 258. Compiler Cost Model

目标：

```text
少测一些
但尽量找到好 Schedule。
```

---

# 259. Search Strategy

例如：

```text
Evolutionary Search
Random Search
Cost-model-guided Search。
```

具体版本会演进。

本章重点理解：

```text
搜索思想。
```

---

# 260. DLight vs MetaSchedule 的工程选择

开发初期：

```text
DLight
```

快速得到：

```text
不错的默认性能。
```

---

# 261. 极致性能

对关键 Kernel：

```text
MetaSchedule
+
Benchmark。
```

---

# 262. 已知最优实现

如果已有：

```text
cuBLAS / CUTLASS。
```

直接：

```text
External Library Dispatch
```

可能更合理。

---

# 263. 这形成三条优化路线

```text
Generate
Rule-based Schedule

Search
Auto-tuning

Reuse
External Library。
```

---

# 264. Compiler 工程的现实

没有一种：

```text
永远最佳。
```

高性能系统：

```text
三者混合。
```

---

# 265. Source Code Map

TVM 很大。

不要：

```text
从 src/target/codegen.cc
```

直接开始。

推荐分层。

---

# 266. 第一阶段：文档

先看：

- [TVM README](https://github.com/apache/tvm)
- [Architecture Index](https://github.com/apache/tvm/blob/main/docs/arch/index.rst)
- [Relax Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/relax)
- [TensorIR Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/tensor_ir)

---

# 267. 第二阶段：TVMScript

推荐：

- [TVMScript Architecture](https://github.com/apache/tvm/blob/main/docs/arch/tvmscript.rst)

学会：

```text
看 IR。
```

---

# 268. 为什么先学读 IR？

Compiler Debug：

```text
最重要技能之一
```

就是：

> 看每个 Pass 前后 IR 发生了什么。

---

# 269. 第三阶段：Relax

重点：

```text
python/tvm/relax
src/relax
docs/deep_dive/relax。
```

---

# 270. Relax 学习重点

```text
Function
Dataflow
StructInfo
Symbolic Shape
Operator
call_tir
transform pass。
```

---

# 271. 第四阶段：Legalize

追：

```text
R.matmul
```

如何：

```text
变成 PrimFunc。
```

---

# 272. 第五阶段：TensorIR

当前：

```text
src/tirx
src/s_tir。
```

---

# 273. TensorIR 学习重点

```text
PrimFunc
Buffer
Block
Loop
Schedule。
```

---

# 274. 第六阶段：Schedule

当前源码中：

```text
src/s_tir/schedule。
```

可以看到 Schedule Primitive：

```text
Split
Fuse
Reorder
Parallel
Vectorize
Bind
Unroll
CacheRead
CacheWrite。
```

---

# 275. 第七阶段：DLight

找：

```text
dlight
```

看：

```text
MatMul / GEMV Schedule Rule。
```

---

# 276. 第八阶段：MetaSchedule

看：

```text
Design Space
Schedule Rule
Runner
Builder
Database
Cost Model。
```

---

# 277. 第九阶段：CodeGen

最后：

- [Code Generation Architecture](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst)

再看：

```text
src/target
src/target/source
src/target/llvm。
```

---

# 278. 第十阶段：Runtime

看：

```text
src/runtime
```

以及：

- [Relax VM Architecture](https://github.com/apache/tvm/blob/main/docs/arch/relax_vm.rst)

---

# 279. 第十一阶段：BYOC

推荐：

- [External Library Dispatch / BYOC](https://github.com/apache/tvm/blob/main/docs/arch/external_library_dispatch.rst)

---

# 280. 第十二阶段：MLC 对照

回去看：

```text
mlc-llm compiler_pass
```

会突然：

```text
容易很多。
```

---

# 281. 本章必做实验 1：打印 Relax IR

用一个简单：

```text
MLP。
```

创建：

```text
Relax Module。
```

然后：

```text
mod.show()
```

观察：

```text
Tensor Shape
Dataflow
R.matmul
R.nn.relu。
```

---

# 282. 实验 2：LegalizeOps

对：

```text
R.matmul
```

执行：

```text
LegalizeOps。
```

观察：

```text
前：
Relax Operator

后：
call_tir + PrimFunc。
```

---

# 283. 这是本章最重要实验之一

因为它第一次直观看到：

```text
Graph
↓
Tensor Program。
```

---

# 284. 实验 3：手写 Vector Add PrimFunc

使用当前：

```text
TVMScript / tirx。
```

自己写：

```text
C[i] = A[i] + B[i]。
```

---

# 285. 实验 4：手写 MatMul PrimFunc

实现：

```text
i
j
k
```

三层 Loop。

先：

```text
完全不优化。
```

---

# 286. 实验 5：Loop Split

把：

```text
i
j
```

Split。

观察：

```text
IR 变化。
```

---

# 287. 实验 6：Reorder

改变：

```text
Loop Order。
```

Benchmark。

观察：

```text
即使 FLOPs 相同
性能也变化。
```

---

# 288. 实验 7：CPU Parallel

对外层：

```text
parallel。
```

测：

```text
1 core
vs
multi-core。
```

---

# 289. 实验 8：Vectorize

对内层：

```text
vectorize。
```

比较：

```text
LLVM generated performance。
```

---

# 290. 实验 9：Tiling

做：

```text
blocked matmul。
```

比较：

```text
naive
vs
tiled。
```

---

# 291. 实验 10：GPU Bind

如果有 NVIDIA：

```text
blockIdx.x
threadIdx.x。
```

生成 CUDA。

---

# 292. 实验 11：Shared Memory

加入：

```text
cache_read(shared)。
```

观察：

```text
TIR
GPU Memory Scope。
```

---

# 293. 实验 12：Register Tile

尝试：

```text
local buffer
```

把更小的 Tile 留在 Register，观察 IR 与性能变化。

---

# 294. 实验 13：打印生成代码

CPU：

```text
LLVM IR / assembly。
```

GPU：

```text
CUDA source / PTX
depending pipeline。
```

---

# 295. 实验 14：DLight

对一个 MatMul：

```text
应用 DLight GPU Schedule。
```

比较：

```text
Manual。
```

---

# 296. 实验 15：MetaSchedule

让：

```text
AutoTune
```

搜索 MatMul。

记录：

```text
Trial
Best schedule
Latency。
```

---

# 297. 实验 16：PyTorch → Relax

```text
torch.export
↓
from_exported_program
↓
Relax IR。
```

然后：

```text
compile + run。
```

---

# 298. 实验 17：ONNX → Relax

用同一个 MLP：

```text
PyTorch
↓
ONNX
↓
TVM Relax。
```

比较：

```text
两条 Frontend。
```

---

# 299. 实验 18：CPU vs CUDA Target

同一个 TensorIR：

```text
Target LLVM

vs

Target CUDA。
```

观察：

```text
Schedule / CodeGen 差异。
```

---

# 300. 实验 19：RPC

如果有开发板：

```text
Host PC
↓
compile ARM artifact
↓
RPC / remote
↓
board benchmark。
```

---

# 301. 实验 20：LLVM ARM Cross Compile

将一个简单模型：

```text
Target = ARM
```

Cross Compile，观察 AArch64 生成结果。

---

# 302. 实验 21：RK3576 CPU 部署

在：

```text
RK3576 Ubuntu
```

运行：

```text
TVM-generated ARM CPU Module。
```

记录：

```text
Latency
CPU Usage
Memory。
```

它和 RKNPU 无关，但可以建立 General Compiler → ARM CPU 的 Baseline。

---

# 303. 实验 22：BYOC 思维实验

无需立刻写完整 Backend。

选择：

```text
MatMul
```

思考：

```text
如果目标是 Vendor NPU，
需要哪些：
Pattern
Partition
CodeGen
Runtime API。
```

---

# 304. 实验 23：连接 MLC-LLM

重新打印：

```text
MLC Llama Relax IR。
```

尝试找：

```text
Prefill
Decode
MatMul
PagedKV。
```

---

# 305. 推荐学习优先级

| 内容 | 优先级 |
|---|---|
| AI Compiler 总体链路 | ★★★★★ |
| Relax | ★★★★★ |
| IRModule | ★★★★★ |
| Relax vs TensorIR | ★★★★★ |
| PrimFunc | ★★★★★ |
| Loop / Block / Buffer | ★★★★★ |
| Schedule | ★★★★★ |
| Split / Reorder / Fuse | ★★★★★ |
| Tiling | ★★★★★ |
| Cache Read / Write | ★★★★★ |
| Parallel / Vectorize | ★★★★★ |
| GPU Thread Binding | ★★★★★ |
| LegalizeOps | ★★★★★ |
| Cross-level Optimization | ★★★★★ |
| Target | ★★★★★ |
| Lowering | ★★★★★ |
| CodeGen | ★★★★★ |
| DLight | ★★★★ |
| MetaSchedule | ★★★★ |
| Tensorize | ★★★★★ |
| BYOC | ★★★★★ |
| Relax VM | ★★★★ |
| RPC | ★★★ |
| Disco Distributed Runtime | ★★ |

---

# 306. 为什么 BYOC 优先级很高？

因为最终：

```text
Rockchip NPU
```

就是非常典型的：

```text
External Accelerator。
```

虽然 RKNN 不是 TVM BYOC 实现，

但：

```text
Partition
Supported Op
External Compiler
Runtime Call。
```

思维高度一致。

---

# 307. 为什么 Tensorize 优先级很高？

因为 AI Accelerator 的核心价值：

```text
Specialized Matrix Instruction。
```

Compiler 只有把 Loop：

```text
Tensorize
```

才能真正用上。

---

# 308. 为什么 MetaSchedule 不排最前？

如果不懂：

```text
split
tile
cache
bind
```

就不知道：

```text
AutoTune 在搜什么。
```

---

# 309. 当前 TVM 版本特别提醒

截至 2026-08：

```text
TVM main
```

正处于：

```text
tirx / s_tir
```

迁移后的新主线。

近期 Release Candidate 也持续推进：

```text
TIRx。
```

---

# 310. 所以搜到旧教程时

可能看到：

```text
Relay
AutoTVM
AutoScheduler
tvm.tir.Schedule
TE Schedule。
```

这些属于：

```text
TVM 历史路线。
```

---

# 311. 哪些旧概念仍值得学？

```text
Tiling
Schedule
Tensorization
Auto-tuning
Graph Optimization。
```

永远值得。

---

# 312. 哪些 API 不应死记？

```text
旧 Relay Build
旧 TE Schedule
旧 AutoTVM Config。
```

除非：

```text
维护旧项目。
```

---

# 313. 现代学习路线

建议：

```text
Relax
↓
TensorIR
↓
tirx / s_tir
↓
DLight
↓
MetaSchedule
↓
CodeGen
↓
BYOC。
```

---

# 314. TVMScript 为什么非常重要？

Compiler IR 如果只显示：

```text
C++ Object。
```

很难读。

TVMScript：

```text
让 IR 看起来像 Python。
```

---

# 315. 但 TVMScript 不是普通 Python

它是：

> IR 的可读 / 可写语法表示。

例如：

```text
T.prim_func
R.function。
```

最终构造：

```text
IR Node。
```

---

# 316. 这和 Triton Python 也不同

Triton：

```text
Python-like Kernel Programming Language。
```

TVMScript：

```text
TVM IR Representation DSL。
```

---

# 317. Compiler Debug 的基本方法

```text
Input Model
↓
Print IR
↓
Pass
↓
Print IR
↓
Pass
↓
Print IR。
```

找：

```text
哪一步发生错误。
```

---

# 318. 模型 Compile 结果不对

不要：

```text
直接怀疑 CodeGen。
```

先：

```text
Frontend Output
Relax IR
Legalization
TIR
```

逐层验证。

---

# 319. Numerical Validation

和 ONNX 转换一样：

```text
PyTorch baseline
vs
TVM output。
```

---

# 320. 数值对比指标

不要：

```text
只看最终分类结果。
```

最好：

```text
compare tensor
max abs error
mean error
relative error。
```

---

# 321. Intermediate Validation

如果最终差异大：

```text
比较中间 Tensor。
```

定位：

```text
第一个产生明显错误的 Op。
```

---

# 322. Compiler Performance Debug

先分：

```text
Graph Problem
Kernel Problem
Runtime Problem。
```

---

# 323. Graph Problem

例如：

```text
Fusion 没发生
Layout 多次转换
CPU/GPU partition 太碎。
```

---

# 324. Kernel Problem

例如：

```text
Tile 差
Occupancy 低
Shared Memory 使用差
Vectorization 差。
```

---

# 325. Runtime Problem

例如：

```text
Memory Copy
Launch Overhead
Synchronization。
```

---

# 326. Compile Time vs Runtime

有些优化：

```text
Compile 要 1 小时。
```

但模型：

```text
运行几百万次。
```

可能值得。

---

# 327. Edge Device 场景

通常：

```text
Host Compile
↓
Device Run。
```

可以接受：

```text
较长编译时间
换设备低延迟。
```

---

# 328. 这就是 AOT 的价值

```text
把复杂工作提前做。
```

---

# 329. TVM 的 Python-first

当前 TVM 另一个重点：

```text
Compiler Transformation
尽量可在 Python 定制。
```

这降低：

```text
修改 Compiler Pipeline
```

门槛。

---

# 330. 为什么 Compiler 核心仍然大量 C++？

```text
IR Node
Analysis
Runtime
CodeGen
Performance-critical infrastructure。
```

仍需要：

```text
C++。
```

---

# 331. FFI

Python：

```text
调用 C++ Compiler Core。
```

需要：

```text
FFI。
```

TVM 现在也持续推进：

```text
tvm-ffi
```

基础设施。

---

# 332. 学源码时不用一开始钻 FFI

先：

```text
理解 Python API。
```

然后：

```text
找到 FFI key
↓
C++ implementation。
```

---

# 333. Source Reading 方法

例如：

```python
sch.split(...)
```

先找到：

```text
Python Schedule API。
```

再追：

```text
FFI。
```

然后：

```text
src/s_tir/schedule。
```

---

# 334. 这样源码不会失控

否则：

```text
C++ Template
ObjectRef
FFI
Reflection
```

会非常难。

---

# 335. 推荐学习项目 1：Tiny Compiler Notebook

做一个 Notebook：

```text
MatMul
```

每步保存：

```text
00_relax.py
01_legalized.py
02_tir_naive.py
03_tiled.py
04_vectorized.py
05_cuda.py
06_generated_code.txt。
```

---

# 336. 这个项目非常重要

因为最后得到：

```text
一条肉眼可见的 Compiler Pipeline。
```

---

# 337. 推荐项目 2：CPU GEMM Optimization

目标：

```text
Naive MatMul
↓
Tiled
↓
Parallel
↓
Vectorized
```

每步：

```text
Benchmark。
```

---

# 338. 推荐项目 3：GPU MatMul

```text
Naive GPU
↓
Thread Bind
↓
Shared Memory
↓
Register Blocking
↓
Tensorization。
```

---

# 339. 推荐项目 4：PyTorch MLP Deployment

```text
PyTorch
↓
torch.export
↓
Relax
↓
LLVM
↓
CPU Runtime。
```

---

# 340. 推荐项目 5：PyTorch CNN → ARM

```text
PyTorch
↓
Relax
↓
ARM LLVM
↓
Cross Compile
↓
RK3576 CPU。
```

这不使用 NPU。

目的：

> 学习通用 Compiler 对 ARM 的部署。

---

# 341. 推荐项目 6：TVM vs RKNN 思维对照

同一个：

```text
MobileNet / Tiny CNN。
```

路线 A：

```text
ONNX
↓
TVM
↓
ARM CPU。
```

路线 B：

```text
ONNX
↓
RKNN-Toolkit2
↓
RKNPU。
```

---

# 342. 比较

```text
Frontend
Operator Support
Quantization
Compile Artifact
Runtime
Performance
Debug Method。
```

---

# 343. 这个实验的价值

不是：

```text
谁快。
```

而是：

> 看 General Compiler 和 Vendor NPU Compiler 的差异。

---

# 344. 推荐项目 7：MLC 源码映射

找到：

```text
MLC Compiler Pass
```

对应：

```text
TVM Relax Pass。
```

---

# 345. 直到能够回答

```text
MLC-LLM
为什么能从 Python Llama Model
生成 CUDA / Vulkan / Metal。
```

---

# 346. 本章核心参考链接

## Apache TVM

- [Apache TVM Repository](https://github.com/apache/tvm)
- [TVM Official Documentation](https://tvm.apache.org/docs/)
- [Architecture Index](https://github.com/apache/tvm/blob/main/docs/arch/index.rst)

---

## Relax

- [Relax Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/relax)
- [Relax Abstraction](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/abstraction.rst)
- [Understand Relax](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/learning.rst)
- [Relax Creation Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/tutorials/relax_creation.py)
- [Relax Transformation Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/tutorials/relax_transformation.py)

重点：

```text
R.function
R.dataflow
StructInfo
Symbolic Shape
R.call_tir
LegalizeOps
Pass。
```

---

## TensorIR

- [TensorIR Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/tensor_ir)
- [TensorIR Index](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/index.rst)
- [TensorIR Learning](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/learning.rst)
- [TensorIR Creation Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/tir_creation.py)
- [TensorIR Transformation Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/tir_transformation.py)

重点：

```text
tirx
s_tir
PrimFunc
Buffer
Block
Loop
Schedule。
```

---

## Schedule

当前源码重点：

- [src/s_tir/schedule](https://github.com/apache/tvm/tree/main/src/s_tir/schedule)
- [schedule.cc](https://github.com/apache/tvm/blob/main/src/s_tir/schedule/schedule.cc)

重点 Primitive：

```text
split
fuse
reorder
parallel
vectorize
bind
unroll
cache_read
cache_write。
```

---

## DLight

- [DLight GPU Scheduling Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/dlight_gpu_scheduling.py)

重点：

```text
Rule-based GPU scheduling。
```

---

## MetaSchedule

- [MetaSchedule Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/meta_schedule.py)

重点：

```text
Design Space
Builder
Runner
Cost Model
Database
Search。
```

---

## TVMScript

- [TVMScript Architecture](https://github.com/apache/tvm/blob/main/docs/arch/tvmscript.rst)

---

## CodeGen

- [Code Generation Architecture](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst)

重点：

```text
Target
TIR Pipeline
Host / Device Split
LLVM
CUDA
ROCm
Metal
OpenCL
Vulkan
WebGPU。
```

---

## Relax VM

- [Relax VM Architecture](https://github.com/apache/tvm/blob/main/docs/arch/relax_vm.rst)

重点：

```text
VM Bytecode
Call
Runtime Module
TIR Kernel Dispatch。
```

---

## BYOC

- [External Library Dispatch / BYOC](https://github.com/apache/tvm/blob/main/docs/arch/external_library_dispatch.rst)

重点：

```text
FuseOpsByPattern
RunCodegen
External Module
cuBLAS
CUTLASS
cuDNN。
```

---

## PyTorch Frontend

源码可搜索：

```text
python/tvm/relax/frontend/torch
```

重点：

```text
torch.export.ExportedProgram
→
Relax。
```

---

## ONNX Frontend

源码：

```text
python/tvm/relax/frontend/onnx
```

重点：

```text
ONNX Graph
→
Relax。
```

---

# 347. 推荐源码阅读顺序

严格建议：

```text
README
↓
Architecture Docs
↓
TVMScript
↓
Relax Tutorial
↓
Relax IR
↓
LegalizeOps
↓
TensorIR Tutorial
↓
Schedule
↓
DLight
↓
MetaSchedule
↓
CodeGen
↓
Runtime
↓
BYOC。
```

不要：

```text
CodeGen
↓
LLVM
↓
CUDA
```

倒着学。

---

# 348. 与整套学习路线的关系

前面：

```text
01  AI / ML
02  Deep Learning
03  Transformer / LLM
04  PyTorch
05  Agent / RAG
06  LLM Inference
07  llama.cpp
08  vLLM / MLC-LLM
```

已经一路从：

```text
模型
```

进入：

```text
Runtime。
```

本章正式开始继续向下：

```text
Runtime
↓
Graph
↓
Compiler
↓
Tensor Program
↓
Kernel
↓
Hardware。
```

---

# 349. 与前面 04 的映射

```text
04_PyTorch与模型框架

PyTorch Model
↓
torch.export
↓
Graph

09_AI_Compiler与TVM

Graph
↓
Relax
↓
TensorIR
↓
Code。
```

---

# 350. 与 06 LLM 推理的映射

```text
06：
MatMul / Attention
为什么慢？

09：
Compiler
怎样改变执行程序
让它更快？
```

---

# 351. 与 07 llama.cpp 的映射

llama.cpp：

```text
ggml Graph
↓
Backend
↓
Kernel。
```

TVM：

```text
Relax
↓
TensorIR
↓
CodeGen。
```

---

# 352. 两者的共同核心

```text
Graph
↓
Operator
↓
Kernel
↓
Hardware。
```

---

# 353. 与 08 MLC 的映射

```text
08:
看到了 Relax / TIR

09:
解释 Relax / TIR 到底是什么。
```

---

# 354. 与下一阶段 GPU Kernel 的关系

本章最后：

```text
Scheduled TensorIR
↓
GPU Kernel。
```

下一章继续：

```text
GPU Thread
Warp
Shared Memory
Register
GEMM
FlashAttention
Triton
CUTLASS。
```

---

# 355. 与 RKNN 的关系

后面：

```text
RKNN-Toolkit2
```

再看：

```text
Model
↓
Graph
↓
Quant
↓
NPU Compile
↓
.rknn。
```

会发现：

```text
Compiler Pipeline
```

已经非常熟悉。

---

# 356. 常见误区

---

## 误区 1

```text
TVM = 一个模型 Runtime。
```

不准确。

核心是：

```text
ML Compiler Framework。
```

---

## 误区 2

```text
Relax = TensorIR。
```

错误。

```text
Relax:
Graph Level

TensorIR:
Tensor Program Level。
```

---

## 误区 3

```text
TensorIR = Machine Code。
```

错误。

它仍是：

```text
IR。
```

后面还要：

```text
Lower + CodeGen。
```

---

## 误区 4

```text
Schedule 改变数学算法。
```

通常不是。

它主要：

```text
改变执行方式
保持语义。
```

---

## 误区 5

```text
Tiling = 减少 FLOPs。
```

通常：

```text
FLOPs 基本不变。
```

主要改善：

```text
Memory Reuse。
```

---

## 误区 6

```text
GPU 核心越多 Compiler 就自动快。
```

如果：

```text
Thread Mapping 差
Memory IO 差
```

仍然慢。

---

## 误区 7

```text
AutoTune 不需要懂 Schedule。
```

错误。

不懂：

```text
Search Space
```

就很难 Debug。

---

## 误区 8

```text
TVM 现在还是 Relay 为主。
```

对于新学习路线：

```text
Relax 优先。
```

---

## 误区 9

```text
所有教程中的 tvm.tir API
都和当前 main 一样。
```

错误。

当前主线已经：

```text
tirx + s_tir。
```

---

## 误区 10

```text
BYOC = CPU Fallback。
```

不是。

BYOC 是：

```text
把匹配 Subgraph
交给 External Codegen。
```

---

## 误区 11

```text
Compiler 一定自己生成所有 Kernel。
```

错误。

可能：

```text
调用 cuBLAS
CUTLASS
cuDNN
Vendor Backend。
```

---

## 误区 12

```text
TVM 能编 CUDA
所以可以自动编所有 NPU。
```

错误。

NPU 需要：

```text
Target Backend
CodeGen / Runtime Integration
Operator Mapping。
```

---

## 误区 13

```text
理解 TVM = 会写几个 API。
```

不够。

真正要掌握：

```text
IR
Transformation
Schedule
Lowering
CodeGen。
```

---

# 357. 本章知识树

```text
AI Compiler
│
├── Frontend
│   ├── PyTorch
│   ├── torch.export
│   ├── ONNX
│   └── TFLite
│
├── High-level IR
│   └── Relax
│       ├── Function
│       ├── Dataflow
│       ├── Tensor
│       ├── StructInfo
│       ├── Symbolic Shape
│       ├── Operator
│       └── call_tir
│
├── IRModule
│   ├── Relax Function
│   ├── PrimFunc
│   └── External Function
│
├── Graph Optimization
│   ├── LegalizeOps
│   ├── Fusion
│   ├── Constant Folding
│   ├── DCE
│   ├── Layout
│   └── Pattern Rewrite
│
├── TensorIR
│   ├── tirx
│   │   ├── PrimFunc
│   │   ├── Buffer
│   │   ├── Block
│   │   ├── Loop
│   │   └── Lowering
│   │
│   └── s_tir
│       ├── Schedule
│       ├── Tensor Intrinsic
│       ├── DLight
│       └── MetaSchedule
│
├── Schedule
│   ├── split
│   ├── fuse
│   ├── reorder
│   ├── parallel
│   ├── vectorize
│   ├── unroll
│   ├── bind
│   ├── cache_read
│   ├── cache_write
│   └── tensorize
│
├── Lowering
│   ├── Buffer Lowering
│   ├── Intrinsic Lowering
│   ├── Thread Lowering
│   └── Host/Device Split
│
├── Target
│   ├── LLVM
│   ├── CUDA
│   ├── ROCm
│   ├── Vulkan
│   ├── OpenCL
│   ├── Metal
│   └── WebGPU
│
├── CodeGen
│   ├── LLVM IR
│   ├── Machine Code
│   ├── CUDA
│   ├── PTX
│   ├── SPIR-V
│   └── WGSL
│
├── Runtime
│   ├── Module
│   ├── PackedFunc
│   ├── Relax VM
│   ├── AOT
│   └── RPC
│
└── External Backend
    └── BYOC
        ├── Pattern
        ├── Partition
        ├── RunCodegen
        ├── cuBLAS
        ├── CUTLASS
        └── Vendor Accelerator
```

---

# 358. 本章最重要的思维模型

## 思维模型 1

```text
AI Compiler
=
Model
→
IR
→
Transform
→
Lower
→
Code。
```

---

## 思维模型 2

```text
Relax
=
Graph Level

TensorIR
=
Kernel / Loop Level。
```

---

## 思维模型 3

```text
Computation
=
What

Schedule
=
How。
```

---

## 思维模型 4

```text
Same FLOPs
≠
Same Performance。
```

---

## 思维模型 5

```text
Performance
高度取决于：
Data Movement
Memory Locality
Parallelism。
```

---

## 思维模型 6

```text
Tiling
=
把 Working Set
搬进更快 Memory
并重复利用。
```

---

## 思维模型 7

```text
Tensorization
=
Loop Pattern
→
Hardware Matrix Instruction。
```

---

## 思维模型 8

```text
Target
决定：
Schedule
Lowering
CodeGen。
```

---

## 思维模型 9

```text
Compiler
不一定自己实现所有 Kernel。

可以：
Generate
Search
Dispatch Library。
```

---

## 思维模型 10

```text
General AI Compiler
和
Vendor NPU Compiler

核心抽象
高度相似。
```

---

# 359. 本章完成标准

如果下面大部分都能自己解释，本章就可以结束。

## AI Compiler 基础

- [ ] 能解释 AI Compiler
- [ ] 能画出 Frontend → IR → Lowering → CodeGen
- [ ] 能解释传统 Compiler 与 AI Compiler 的关系
- [ ] 能解释为什么 AI Compiler 特别关注 Tensor / Memory / Accelerator
- [ ] 能解释 Compiler vs Runtime

## Relax

- [ ] 能解释 Relax
- [ ] 能解释 Graph-level IR
- [ ] 能解释 `R.function`
- [ ] 能解释 `R.dataflow`
- [ ] 能解释 StructInfo
- [ ] 能解释 Symbolic Shape
- [ ] 能解释 Dynamic Shape
- [ ] 能解释 `R.call_tir`

## IRModule

- [ ] 能解释 IRModule
- [ ] 能解释为什么 Relax + TensorIR 可以共存
- [ ] 能解释 Cross-level Optimization
- [ ] 能看懂一个简单 TVMScript Module

## TensorIR

- [ ] 能解释 PrimFunc
- [ ] 能解释 Buffer
- [ ] 能解释 Block
- [ ] 能解释 Spatial Axis
- [ ] 能解释 Reduction Axis
- [ ] 能解释 Loop Nest
- [ ] 能解释 TensorIR 为什么比 Graph 更低层
- [ ] 知道当前 `tirx / s_tir` 拆分

## Graph Transform

- [ ] 能解释 LegalizeOps
- [ ] 能解释 Operator Fusion
- [ ] 能解释 Constant Folding
- [ ] 能解释 Dead Code Elimination
- [ ] 能解释 Layout Optimization
- [ ] 能解释 Graph Partition

## Schedule

- [ ] 能解释 Schedule
- [ ] 能解释 computation vs schedule
- [ ] 能解释 split
- [ ] 能解释 fuse
- [ ] 能解释 reorder
- [ ] 能解释 parallel
- [ ] 能解释 vectorize
- [ ] 能解释 unroll
- [ ] 能解释 bind
- [ ] 能解释 cache_read
- [ ] 能解释 cache_write
- [ ] 能解释 Tiling
- [ ] 能解释 Register Blocking

## GPU Mapping

- [ ] 能解释 blockIdx / threadIdx
- [ ] 能解释 Shared Memory
- [ ] 能解释 Register
- [ ] 能解释为什么 GPU MatMul 要 Tile
- [ ] 能解释 Data Reuse
- [ ] 能解释 Memory Coalescing 的基本位置

## Tensorization

- [ ] 能解释 Tensor Intrinsic
- [ ] 能解释 Tensorize
- [ ] 能解释 Tensor Core 与 Loop Pattern 的关系
- [ ] 能解释为什么 NPU Compiler 也需要类似映射

## Automation

- [ ] 能解释 DLight
- [ ] 能解释 MetaSchedule
- [ ] 能解释 Design Space
- [ ] 能解释 Cost Model
- [ ] 能解释 Builder / Runner
- [ ] 能解释为什么必须 Benchmark

## Lowering / CodeGen

- [ ] 能解释 Lowering
- [ ] 能解释 Target
- [ ] 能解释 Host / Device Split
- [ ] 能解释 LLVM CodeGen
- [ ] 能解释 CUDA CodeGen
- [ ] 能解释 Vulkan/SPIR-V
- [ ] 能解释 Source CodeGen 与 LLVM 路线的差异

## Runtime

- [ ] 能解释 runtime.Module
- [ ] 能解释 PackedFunc
- [ ] 能解释 Relax VM
- [ ] 能解释为什么 VM 不直接做 MatMul
- [ ] 能解释 AOT
- [ ] 能解释 RPC

## BYOC

- [ ] 能解释 BYOC
- [ ] 能解释 Pattern Matching
- [ ] 能解释 Subgraph Partition
- [ ] 能解释 External CodeGen
- [ ] 能解释 External Runtime Module
- [ ] 能解释 BYOC 与 NPU Backend 的关系
- [ ] 能解释 BYOC 与 ONNX Runtime EP 的相似思路

## 工程实践

- [ ] 能创建一个 Relax MLP
- [ ] 能执行 LegalizeOps 并看 IR 差异
- [ ] 能写 Vector Add PrimFunc
- [ ] 能写 Naive MatMul PrimFunc
- [ ] 能做至少 3 种 Schedule 变换
- [ ] 能 Benchmark schedule 前后性能
- [ ] 能针对 CPU 编译
- [ ] 最好能针对 CUDA/Vulkan 编译
- [ ] 能把 PyTorch ExportedProgram 导入 Relax
- [ ] 能解释 TVM 与 MLC-LLM 的关系
- [ ] 能解释 TVM 与 RKNN 的抽象关系

---

# 360. 本章暂时不要求深入

暂时不要求：

- [ ] 自己实现 Relax Parser
- [ ] 自己实现 TensorIR AST
- [ ] 精通所有 Relax Pass
- [ ] 精通所有 Schedule Primitive
- [ ] 自己写 MetaSchedule Cost Model
- [ ] 精通 Polyhedral Compilation
- [ ] 自己实现 LLVM Backend
- [ ] 自己实现 CUDA CodeGen
- [ ] 自己实现完整 BYOC Backend
- [ ] 精通 Disco
- [ ] 精通分布式 TVM
- [ ] 精通所有 Hardware Target
- [ ] 自己实现 RKNPU Backend

先建立：

```text
完整 Compiler Mental Model。
```

---

# 361. 本章最推荐的学习方式

不要：

```text
连续看 100 页 TVM 文档。
```

而要：

```text
一个 MatMul
↓
每学一个概念
就修改一次 IR
↓
Benchmark。
```

---

# 362. 建议建立实验仓库

```text
tvm-learning/
│
├── 01_relax_basic.py
├── 02_legalize.py
├── 03_tir_vector_add.py
├── 04_tir_matmul_naive.py
├── 05_schedule_split.py
├── 06_schedule_tile.py
├── 07_cpu_parallel.py
├── 08_cpu_vectorize.py
├── 09_cuda_bind.py
├── 10_shared_memory.py
├── 11_dlight.py
├── 12_meta_schedule.py
├── 13_torch_export_relax.py
├── 14_onnx_relax.py
└── 15_rk3576_cpu.py
```

---

# 363. 每个实验统一记录

```text
Input Shape

Target

Original IR

Transformed IR

Generated Code

Latency

Correctness

Memory。
```

---

# 364. 这样最后你真正拥有的是

```text
一套 Compiler 学习资产。
```

而不是：

```text
看过 TVM。
```

---

# 365. 与最终 RK3576 主线连接

最终：

```text
PyTorch
↓
ONNX
↓
RKNN-Toolkit2
↓
.rknn
↓
RKNPU
```

虽然没有直接经过 TVM，

但你会用 TVM 学到的语言理解它：

```text
Frontend
↓
Graph IR
↓
Optimization
↓
Quantization
↓
Lowering
↓
Target Mapping
↓
Compiled Artifact
↓
Runtime。
```

---

# 366. 这是本章最重要的价值

TVM 不只是：

```text
一个以后可能直接用的工具。
```

更重要：

> **它提供了一套理解所有 AI 编译器的共同语言。**

---

# 367. 最终总图

```text
                    AI Model
                       │
                       ▼
                  Framework
             PyTorch / ONNX
                       │
                       ▼
                    Frontend
                       │
                       ▼
                     Relax
                Graph-level IR
                       │
              ┌────────┴────────┐
              ▼                 ▼
      Graph Optimization    Shape Analysis
              │                 │
              └────────┬────────┘
                       ▼
                 LegalizeOps
                       │
                       ▼
                  TensorIR
                       │
              ┌────────┴────────┐
              ▼                 ▼
            tirx              s_tir
       Core Tensor IR        Schedule
              │                 │
              │          ┌──────┼──────┐
              │          ▼      ▼      ▼
              │        DLight  Manual MetaSchedule
              │          │      │      │
              └──────────┴──────┴──────┘
                       │
                       ▼
                  Lowering
                       │
                       ▼
                    Target
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        LLVM          CUDA        Vulkan
          │            │            │
          ▼            ▼            ▼
       ARM/x86       NVIDIA       SPIR-V
          │            │            │
          └────────────┼────────────┘
                       ▼
                  Runtime Module
                       │
                       ▼
                 Relax VM / AOT
                       │
                       ▼
                    Hardware
```

---

# 368. 最终主线

```text
PyTorch Model
      ↓
Graph Capture
      ↓
Relax
      ↓
Graph Optimization
      ↓
TensorIR
      ↓
Schedule
      ↓
Lowering
      ↓
Target
      ↓
CodeGen
      ↓
Kernel
      ↓
CPU / GPU
```

对应 Vendor NPU：

```text
ONNX
↓
Vendor Graph IR
↓
Operator Optimization
↓
Quantization
↓
Hardware Mapping
↓
NPU Compiler
↓
.rknn / vendor artifact
↓
Runtime
↓
NPU
```

---

# 369. 一句话总结

这一章真正需要记住的不是：

```text
TVM API。
```

而是：

```text
Model
↓
Graph IR
↓
Tensor IR
↓
Schedule
↓
Lowering
↓
CodeGen
↓
Runtime
↓
Hardware。
```

其中：

```text
Relax
=
模型层“算什么”

TensorIR
=
Kernel 层“具体怎么算”

Schedule
=
“怎样把同一个计算安排得更快”

Target
=
“最终要跑在哪”

CodeGen
=
“把低层程序翻译成目标代码”
```

当这一条链真正理解以后，

下一章：

> **10_GPU_Kernel与FlashAttention.md**

就可以继续从：

```text
Schedule
↓
GPU Kernel
```

往下钻。

你会开始真正研究：

```text
Thread
Warp
Block
Register
Shared Memory
Global Memory
Coalescing
Occupancy
GEMM Tiling
Tensor Core
FlashAttention
Triton
CUTLASS
```

也就是：

> **Compiler 最终到底在试图生成什么样的“好 Kernel”。**
