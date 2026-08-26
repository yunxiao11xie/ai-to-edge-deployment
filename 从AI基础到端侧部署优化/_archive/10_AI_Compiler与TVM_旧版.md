# 10_AI_Compiler与TVM

> **所属路线**：AI 学习路线 · 第三部分  
> **定位**：从 LLM Runtime 继续向下进入 AI Compiler，建立从 **模型图 → 中间表示 IR → Tensor Program → Schedule → Lowering → CodeGen → Runtime → CPU/GPU/NPU** 的完整编译器认知  
> **核心仓库**：[`apache/tvm`](https://github.com/apache/tvm)  
> **核心抽象**：Relax + TensorIR（当前主线进一步拆分为 `tirx` + `s_tir`）  
> **关键扩展**：TVMScript、LegalizeOps、DLight、MetaSchedule、BYOC、Relax VM、LLVM/CUDA/Vulkan/OpenCL/Metal CodeGen  
> **学习方式**：不要把本章学成“TVM API 教程”，而要始终追问：**一个 MatMul 是如何从模型里的一个算子，最终变成硬件上的线程、访存和指令的？**  
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

完成本章后，应能解释：

1. AI Compiler 是什么？
2. 为什么 AI Model 需要 Compiler？
3. AI Compiler 与 GCC / Clang / LLVM 的关系是什么？
4. Frontend 是什么？
5. IR 是什么？
6. 为什么 Compiler 不能只使用一种 IR？
7. Graph IR 与 Tensor IR 有什么区别？
8. Relax 是什么？
9. TensorIR 是什么？
10. 当前 TVM 中 `tirx` 与 `s_tir` 分别是什么？
11. `IRModule` 是什么？
12. Relax Function 与 PrimFunc 有什么区别？
13. `R.call_tir` 为什么重要？
14. Symbolic Shape 是什么？
15. Dynamic Shape 为什么是 LLM Compiler 的核心问题之一？
16. `LegalizeOps` 做了什么？
17. Graph Fusion 是什么？
18. Pattern Rewrite 是什么？
19. Constant Folding 是什么？
20. Dead Code Elimination 是什么？
21. Layout Transformation 为什么影响性能？
22. Loop Nest 是什么？
23. Spatial Axis 与 Reduction Axis 是什么？
24. Schedule 是什么？
25. Computation 与 Schedule 为什么要分开？
26. `split` 是什么？
27. `reorder` 是什么？
28. `fuse` 是什么？
29. `parallel` 是什么？
30. `vectorize` 是什么？
31. `unroll` 是什么？
32. `bind` 是什么？
33. Tiling 为什么如此重要？
34. `cache_read` / `cache_write` 在优化什么？
35. CPU Cache Blocking 与 GPU Shared Memory Tiling 有什么共同本质？
36. Register Blocking 是什么？
37. Tensorization 是什么？
38. Tensor Core 为什么需要 Tensor Intrinsic？
39. DLight 是什么？
40. MetaSchedule 是什么？
41. Auto-tuning 到底在搜索什么？
42. Target 是什么？
43. `llvm`、`cuda`、`vulkan`、`opencl`、`metal` 是什么层？
44. Lowering 是什么？
45. CodeGen 是什么？
46. Host Code 与 Device Code 为什么要拆开？
47. Relax VM 是什么？
48. Runtime Module 是什么？
49. PackedFunc 是什么？
50. AOT 与 VM 有什么区别？
51. RPC 在 TVM 中有什么价值？
52. BYOC 是什么？
53. 为什么 BYOC 对 NPU Compiler 特别重要？
54. TVM 为什么可以调用 cuBLAS / cuDNN / CUTLASS，而不是所有 Kernel 自己生成？
55. TVM 与 MLC-LLM 的关系是什么？
56. TVM 与 llama.cpp 有什么区别？
57. TVM 与 vLLM 有什么区别？
58. TVM 与 ONNX Runtime 有什么区别？
59. TVM 与 RKNN-Toolkit2 有哪些相似的编译器抽象？
60. `.rknn` 为什么可以理解成 Vendor Compiler 的 Target-specific Artifact？
61. 为什么理解 TVM 后，RKNN 的“算子不支持、量化失败、Graph Partition、Fallback”会更好理解？
62. 一个 MatMul 如何从 Relax 一路走到 TensorIR？
63. 一个 MatMul 如何通过 Tiling / Cache / Vectorization / Tensorization 变快？
64. Compiler 工程师到底在优化 FLOPs，还是优化 Data Movement？
65. 为什么同样 FLOPs 的两个 Kernel，速度可能差几十倍？
66. 为什么 AI Compiler 最终必须理解硬件？
67. 为什么 Edge AI 工程师也值得掌握 Compiler Mental Model？

---

# 2. 核心仓库：Apache TVM

GitHub：

- [apache/tvm](https://github.com/apache/tvm)

建议重点收藏：

- [TVM Architecture](https://github.com/apache/tvm/tree/main/docs/arch)
- [Relax Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/relax)
- [TensorIR Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/tensor_ir)
- [TVMScript Architecture](https://github.com/apache/tvm/blob/main/docs/arch/tvmscript.rst)
- [Code Generation Architecture](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst)
- [Relax VM Architecture](https://github.com/apache/tvm/blob/main/docs/arch/relax_vm.rst)
- [External Library Dispatch / BYOC](https://github.com/apache/tvm/blob/main/docs/arch/external_library_dispatch.rst)

当前 TVM 的整体核心可以压缩为：

```text
IRModule
│
├── Relax
│   └── Graph-level Model Program
│
└── TensorIR
    ├── tirx
    │   └── Core Tensor IR + Lowering
    │
    └── s_tir
        └── Schedule + Tensor Intrinsic
            + DLight + MetaSchedule
```

---

# 3. AI Compiler 到底是什么？

先从传统编译器开始。

C 代码：

```c
for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
}
```

传统 Compiler：

```text
C Source
↓
AST
↓
Compiler IR
↓
Optimization
↓
Machine IR
↓
Machine Code
```

例如：

```text
Clang
↓
LLVM IR
↓
x86 / ARM Machine Code
```

AI Compiler 本质一样：

```text
High-level Program
↓
IR
↓
Transformation
↓
Lowering
↓
Target Code
```

只是输入程序变成了：

```text
Tensor Program / Neural Network
```

---

# 4. AI Compiler 的输入是什么？

可能是：

```text
PyTorch
TensorFlow
JAX
ONNX
TFLite
MLC Model Definition
自定义 Graph IR
```

这些高层表示里：

```text
Linear
Conv
MatMul
Attention
Softmax
RMSNorm
```

只是：

```text
“数学上要算什么”
```

---

# 5. 硬件最终需要什么？

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

两者抽象差距巨大。

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

# 6. AI Compiler 与传统 Compiler 的共同点

都有：

```text
Frontend
IR
Optimization
Lowering
CodeGen
Runtime / ABI
```

---

# 7. AI Compiler 的特殊之处

AI Workload 特别强调：

```text
Tensor Shape
Tensor Layout
Reduction
Memory Hierarchy
Massive Parallelism
Low Precision
Matrix Accelerator
Dynamic Shape
Operator Fusion
Kernel Selection
```

所以 AI Compiler 往往会拥有：

```text
Graph IR
+
Tensor IR
```

两个非常重要的层次。

---

# 8. 第一条主线：Graph IR

假设模型：

```text
Input
↓
Linear
↓
GELU
↓
Linear
↓
Output
```

Graph IR 关心：

```text
Operator 是谁？
依赖关系是什么？
Tensor Shape 是什么？
哪个 Op 可以 Fusion？
哪个 Tensor 是 Constant？
```

---

# 9. 第二条主线：Tensor IR

拿其中：

```text
Linear
```

实际上是：

```text
MatMul
+
Bias
```

MatMul：

```text
C[i,j]
=
Σ A[i,k] × B[k,j]
```

Tensor IR 关心：

```text
Loop 怎么排？
Tile 多大？
哪个 Loop 并行？
哪个 Loop Vectorize？
数据放 Cache 还是 Shared Memory？
线程如何映射 GPU？
```

---

# 10. 为什么需要两层？

因为：

```text
Graph Optimization
```

和：

```text
Kernel Optimization
```

不是同一个问题。

Graph：

```text
MatMul → Add → ReLU
```

可以决定：

```text
三个 Op 是否 Fusion
```

Kernel：

```text
融合后的 Loop
```

则决定：

```text
如何 Tile / Vectorize / Bind
```

---

# 11. TVM 的两层核心抽象

当前 TVM：

```text
Relax
=
High-level Graph IR

TensorIR
=
Low-level Tensor Program IR
```

而且：

```text
两者可以同时存在于 IRModule。
```

---

# 12. IR 是什么？

IR：

```text
Intermediate Representation
```

中文：

```text
中间表示
```

可以理解：

> Compiler 内部使用的一种“比用户代码低层、比机器代码高层”的程序表示。

---

# 13. 为什么不能直接 PyTorch → CUDA？

理论上不是完全不可能。

但工程上会非常难：

```text
PyTorch Python
↓
???
↓
PTX
```

中间没有稳定抽象：

```text
无法分层优化
无法调试
无法复用 Pass
难以支持多硬件
```

所以 Compiler 需要：

```text
多个 IR 层级。
```

---

# 14. IRModule

当前 TVM 中：

```text
IRModule
```

是整个 Compiler Stack 最重要的数据结构之一。

它可以保存：

```text
多个 Function。
```

---

# 15. IRModule 中两种核心 Function

```text
Relax Function

tirx::PrimFunc
```

---

# 16. Relax Function

表示：

```text
High-level Model / Subgraph
```

可以拥有：

```text
Tensor
Tuple
Control Flow
Symbolic Shape
Function Call
```

---

# 17. PrimFunc

表示：

```text
Primitive Tensor Function
```

更加接近：

```text
Loop
Buffer
Load
Store
Thread
Vector
Tensor Instruction
```

---

# 18. 一张核心图

```text
IRModule
│
├── @R.function
│      main()
│      │
│      ├── call_tir(matmul)
│      └── call_tir(relu)
│
├── @T.prim_func
│      matmul()
│
└── @T.prim_func
       relu()
```

---

# 19. 这就是 Cross-level IR

模型层：

```text
Relax
```

与 Kernel 层：

```text
TensorIR
```

可以互相连接。

---

# 20. `R.call_tir`

非常重要。

概念：

```python
y = R.call_tir(
    matmul,
    (x, w),
    out_ty=...
)
```

意思：

```text
在 Relax Graph 中
调用一个 TensorIR PrimFunc。
```

---

# 21. 为什么这很强？

因为 Compiler 同时知道：

```text
高层 Graph Context
```

和：

```text
底层 Loop Structure
```

所以可以：

```text
Cross-level Optimization。
```

---

# 22. Relax 是什么？

Relax：

> TVM 用于模型计算图表示、分析与优化的高层 IR。

一个简单例子：

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

它表达：

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

# 23. Relax 不关心什么？

它不需要马上知道：

```text
MatMul 内部到底有几层 for。
```

这一层只是：

```text
Graph Semantics。
```

---

# 24. Relax 的核心能力

当前官方重点：

```text
First-class Symbolic Shape
Multi-level Abstraction
Composable Transformations
```

---

# 25. Symbolic Shape

例如：

```text
Input:
[n, 4096]
```

其中：

```text
n
```

不是编译时固定数字。

---

# 26. 为什么 Symbolic Shape 重要？

LLM：

```text
Batch
Sequence Length
```

本来就是动态。

例如：

```text
T = 128
T = 512
T = 8192
```

Runtime 都可能遇到。

---

# 27. Static Shape

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

---

# 28. Dynamic Shape

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
Specialization
```

---

# 29. LLM 的 Decode 是特殊 Shape

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

# 30. Relax Dataflow

```python
with R.dataflow():
```

描述：

```text
纯 Dataflow Region。
```

Compiler 可以：

```text
分析依赖
做 Fusion
消除临时结果。
```

---

# 31. Graph Optimization

Relax 层典型 Pass：

```text
Constant Folding
Dead Code Elimination
Operator Fusion
Pattern Rewrite
Layout Rewrite
Library Dispatch
```

---

# 32. Constant Folding

例如：

```text
2 + 3
```

编译阶段直接：

```text
5
```

---

# 33. AI 模型里的 Constant Folding

例如：

```text
固定 Reshape Shape
固定 Transpose
固定 Scale
Constant Weight Transform
```

有些都能提前完成。

---

# 34. Dead Code Elimination

```text
某个结果
没有任何 Consumer。
```

Compiler：

```text
删除这段计算。
```

---

# 35. Operator Fusion

原：

```text
MatMul
↓
Bias Add
↓
GELU
```

可能变成：

```text
Fused MatMul + Bias + GELU。
```

---

# 36. 为什么 Fusion 快？

未融合：

```text
Kernel 1
↓
写 DRAM
↓
Kernel 2
↓
读 DRAM
↓
写 DRAM
↓
Kernel 3
```

融合：

```text
Kernel 1
↓
中间值尽量留在 Register / Cache
↓
直接继续计算。
```

减少：

```text
Memory Traffic
Kernel Launch。
```

---

# 37. Graph Fusion 和 Kernel Fusion

Graph Fusion：

```text
决定“哪些 Op 放一起”。
```

Kernel Fusion：

```text
真正生成一个联合 Kernel。
```

---

# 38. Layout Optimization

Tensor：

```text
Shape 一样
```

但 Layout：

```text
可以不同。
```

例如 CNN：

```text
NCHW
NHWC
```

---

# 39. 为什么 Layout 会影响性能？

因为：

```text
Memory Contiguity
Vector Access
Cache
GPU Coalescing
Tensor Core Layout
```

都受影响。

---

# 40. Legalization

Relax：

```text
R.matmul
```

只是高层 Operator。

最终需要：

```text
具体 Tensor Program。
```

---

# 41. `LegalizeOps`

TVM Relax 中一个核心 Pass：

```text
LegalizeOps
```

作用：

```text
High-level Relax Op
↓
TensorIR implementation
↓
R.call_tir。
```

---

# 42. MatMul Legalize

原：

```text
R.matmul(A, B)
```

可能变：

```text
Relax:
call_tir(matmul, A, B)

TensorIR:
@T.prim_func
matmul(...)
```

---

# 43. 这是本章第一条重要跨层链

```text
Graph Operator
↓
Legalization
↓
Tensor Program。
```

---

# 44. TensorIR 是什么？

TensorIR：

> TVM 中描述 Primitive Tensor Computation 的低层 IR。

它显式表达：

```text
Loop
Buffer
Block
Memory Access
Thread Binding
Vector Operation
Tensor Intrinsic
```

---

# 45. 当前 TVM：TensorIR 已进一步拆分

2026 当前主线：

```text
TensorIR
├── tirx
└── s_tir
```

---

# 46. `tirx`

核心：

```text
IR Definition
+
Lowering Infrastructure。
```

包括：

```text
PrimFunc
Buffer
SBlock
Expressions
Statements
Analysis
Lowering Pass。
```

---

# 47. `s_tir`

全称：

```text
Schedulable TIR。
```

包含：

```text
Schedule Primitive
Tensor Intrinsic
DLight
MetaSchedule。
```

---

# 48. 最简单的区分

```text
tirx
=
Tensor Program 本身是什么

s_tir
=
怎么变换 / 调优 Tensor Program。
```

---

# 49. TVMScript 当前写法

新主线常见：

```python
from tvm.script import tirx as T
```

然后：

```python
@T.prim_func
def matmul(...):
    ...
```

---

# 50. 为什么很多旧教程写 `tvm.script.tir`？

因为：

```text
历史 API。
```

TensorIR 概念没变。

但：

```text
源码目录和 API
已经演进。
```

---

# 51. Relay 又是什么？

Relay：

```text
TVM 历史上的 Graph-level IR。
```

旧教程常见：

```text
PyTorch
↓
Relay
↓
TIR
↓
LLVM / CUDA。
```

---

# 52. 当前新学习路线

建议：

```text
Relax
↓
TensorIR
```

而不是把：

```text
Relay
```

作为主线。

---

# 53. 旧 Relay 教程还有价值吗？

有。

可以学习：

```text
Graph Compiler
Operator Fusion
Frontend
Target
Runtime
```

等思想。

但：

```text
API 不应照搬。
```

---

# 54. 一个 TensorIR Vector Add

逻辑：

```text
C[i] = A[i] + B[i]
```

朴素程序：

```python
for i in range(N):
    C[i] = A[i] + B[i]
```

---

# 55. TensorIR 不只是 Loop

它还知道：

```text
A / B / C 是 Buffer
读哪些区域
写哪些区域
Loop Axis 类型
```

---

# 56. Buffer

Buffer：

```text
Tensor 对应的低层内存视图。
```

包含：

```text
Shape
Dtype
Stride
Memory Scope
Pointer-related information。
```

---

# 57. Block

TensorIR 中：

```text
Block
```

用来描述一段计算及其：

```text
Iteration Domain
Read Region
Write Region
Init
Reduction。
```

---

# 58. Spatial Axis

例如：

```text
i
j
```

不同：

```text
i,j
```

对应不同输出。

---

# 59. Reduction Axis

MatMul：

```text
k
```

多个 `k`：

```text
累加到同一个 C[i,j]。
```

---

# 60. 一个 MatMul

```text
C[i,j]
=
Σ_k A[i,k] × B[k,j]
```

朴素 Loop：

```python
for i in range(M):
    for j in range(N):
        for k in range(K):
            C[i,j] += A[i,k] * B[k,j]
```

---

# 61. 为什么这个 MatMul 可能很慢？

不是 FLOPs 错。

而是：

```text
Memory Locality
Cache Reuse
Parallelism
SIMD
GPU Mapping
```

都没优化。

---

# 62. Schedule 是什么？

Schedule：

> 保持计算语义不变，但改变 Tensor Program 的执行结构。

---

# 63. 最重要的区分

```text
Computation
=
算什么

Schedule
=
怎么计算。
```

---

# 64. 一个数学表达可以有无数 Schedule

MatMul：

```text
Naive
Tiled
Parallel
Vectorized
Shared-memory
Tensor-core。
```

结果：

```text
都一样。
```

速度：

```text
完全不同。
```

---

# 65. `split`

原 Loop：

```text
i = 0..1023
```

拆成：

```text
io
ii
```

例如：

```text
1024 = 32 × 32。
```

---

# 66. Split 最主要用途

```text
Tiling
Thread Mapping
Vectorization
Cache Blocking。
```

---

# 67. `reorder`

原：

```text
i
j
k
```

可以变：

```text
io
jo
ko
ii
ji
ki。
```

---

# 68. Reorder 为什么重要？

改变：

```text
Memory Access Sequence
Reuse Distance
Parallelism
Vectorization Condition。
```

---

# 69. `fuse`

把：

```text
i
j
```

合成：

```text
fused loop。
```

适合：

```text
Flatten Work
Parallel
Thread Bind。
```

---

# 70. `parallel`

CPU：

```text
把不同迭代
分给多个线程 / Core。
```

---

# 71. `vectorize`

将：

```text
Scalar Loop
```

变：

```text
SIMD Vector Operation。
```

---

# 72. CPU SIMD

例如：

```text
x86:
AVX2
AVX512

ARM:
NEON
SVE。
```

---

# 73. 为什么 ARM Edge 也需要 Compiler Schedule？

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

# 74. `unroll`

原：

```text
for k in 0..3
```

可以展开：

```text
body(k=0)
body(k=1)
body(k=2)
body(k=3)
```

---

# 75. Unroll 优点

```text
减少 Loop Control
暴露更多 Instruction-level Parallelism。
```

---

# 76. Unroll 缺点

```text
Code Size ↑
Register Pressure ↑。
```

---

# 77. GPU Schedule

GPU 不只是：

```text
parallel()
```

还需要：

```text
Loop
↓
blockIdx.x
threadIdx.x。
```

---

# 78. `bind`

例如：

```text
outer loop
→
blockIdx.x

inner loop
→
threadIdx.x。
```

---

# 79. GPU Execution Hierarchy

```text
Grid
↓
Block
↓
Warp
↓
Thread。
```

---

# 80. 为什么 Compiler 必须理解这个层级？

因为：

```text
GPU Kernel 性能
```

取决于：

```text
多少 Block
多少 Thread
如何共享数据
如何访问 Global Memory。
```

---

# 81. Tiling 是核心中的核心

MatMul：

```text
M × N × K
```

大矩阵。

硬件最快 Memory：

```text
容量有限。
```

所以必须：

```text
切 Tile。
```

---

# 82. 一个 GPU MatMul Tiling

概念：

```text
Global A/B
↓
Tile A/B
↓
Shared Memory
↓
Warp Tile
↓
Register Tile
↓
FMA / MMA。
```

---

# 83. 为什么 Tile 会快？

因为：

```text
同一块 A/B 数据
```

从慢 Memory 搬一次后：

```text
重复使用很多次。
```

---

# 84. Arithmetic Intensity 再次出现

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

# 85. Compiler 与推理性能终于连起来

06：

```text
Decode 为什么 Memory-Bound？
```

09/10：

```text
Compiler 如何减少 Memory Traffic？
```

---

# 86. `cache_read`

概念：

```text
Global Buffer
↓
Local Cache Buffer
↓
Consumer。
```

GPU：

```text
global
↓
shared。
```

---

# 87. `cache_write`

结果先：

```text
写 Local / Shared
```

再：

```text
write back。
```

---

# 88. CPU Cache Blocking 与 GPU Shared Memory Tiling

本质都一样：

> **把 Working Set 搬进更快但更小的 Memory，尽可能重复使用。**

---

# 89. Register Blocking

更小 Tile：

```text
直接留在 Register。
```

让一个线程：

```text
重复执行很多 FMA。
```

---

# 90. 为什么 Register 最快？

它最靠近：

```text
Compute Unit。
```

但容量：

```text
非常有限。
```

---

# 91. Register Pressure

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

# 92. Schedule 是典型 Tradeoff 问题

```text
更大 Tile
→
Reuse ↑

但：
Shared Memory ↑
Register ↑
Occupancy ↓。
```

---

# 93. 这就是为什么自动调优有价值

人很难：

```text
根据公式直接找到完美参数。
```

---

# 94. Tensorization

Tensorization：

> 将一段符合某种计算 Pattern 的 Loop 替换为硬件提供的 Tensor / Matrix Intrinsic。

---

# 95. NVIDIA Tensor Core

例如：

```text
16×16×16 Matrix Multiply。
```

硬件有：

```text
MMA instruction。
```

Compiler 要：

```text
识别 Loop Pattern
↓
匹配 Tensor Intrinsic
↓
Tensorize。
```

---

# 96. CPU 也有类似思想

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

# 97. NPU 更明显

NPU：

```text
Matrix MAC Array
```

就是专门计算：

```text
Tensor Block。
```

Compiler 必须：

```text
把高层 MatMul
匹配硬件 Tensor Unit。
```

---

# 98. 所以 Tensorization 是理解 NPU Compiler 的关键入口

你可以把 NPU Compiler 的重要工作之一理解成：

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

# 99. DLight

DLight：

> TVM 中面向常见 Workload 的自动 Schedule Rule 系统。

---

# 100. 为什么 DLight 出现？

很多常见 Operator：

```text
MatMul
GEMV
Reduction
Elementwise
```

已经有成熟优化经验。

没有必要：

```text
每个模型手写一次 Schedule。
```

---

# 101. DLight 的思想

```text
Inspect TensorIR
↓
Recognize Pattern
↓
Apply Scheduling Rule
↓
Generate Good Kernel。
```

---

# 102. DLight 与 Manual Schedule

Manual：

```text
工程师直接写：
split
reorder
bind
cache_read。
```

DLight：

```text
规则自动完成。
```

---

# 103. MetaSchedule

MetaSchedule：

> 自动搜索高性能 Schedule 的 Auto-tuning 系统。

---

# 104. MetaSchedule 的完整思维

```text
Tensor Program
↓
Design Space
↓
Schedule Candidates
↓
Build
↓
Run on Hardware
↓
Measure Latency
↓
Cost Model
↓
Search
↓
Best Schedule。
```

---

# 105. Design Space

例如：

```text
Tile M:
8 / 16 / 32 / 64

Tile N:
8 / 16 / 32 / 64

Threads:
64 / 128 / 256

Unroll:
0 / 4 / 8

Vector Width:
1 / 4 / 8
```

组合数量：

```text
非常大。
```

---

# 106. 为什么不穷举？

因为：

```text
Build + Benchmark
```

非常昂贵。

所以：

```text
Search Strategy
+
Cost Model。
```

---

# 107. Cost Model

根据已有测量：

```text
预测哪些 Candidate
更可能快。
```

减少：

```text
无效测量。
```

---

# 108. Manual / DLight / MetaSchedule

非常建议记住：

```text
Manual Schedule
=
人设计

DLight
=
规则驱动

MetaSchedule
=
搜索驱动。
```

---

# 109. 学习顺序

```text
Manual
↓
DLight
↓
MetaSchedule。
```

不要反过来。

---

# 110. 为什么？

如果不会：

```text
split
tile
cache
bind
```

就不知道：

```text
AutoTune 到底在调什么。
```

---

# 111. Target

Compiler 必须知道：

```text
要编给谁？
```

这就是：

```text
Target。
```

---

# 112. 常见 Target

```text
llvm
cuda
rocm
vulkan
opencl
metal
webgpu
```

---

# 113. Target 不只是“CPU / GPU”

还可能包含：

```text
Architecture
ISA Feature
Compute Capability
Memory Properties
Thread Limits。
```

---

# 114. 例如 CUDA

```text
cuda
+
sm_80
```

Compiler 可以知道：

```text
目标是 Ampere。
```

---

# 115. 为什么硬件架构信息重要？

因为：

```text
Tensor Core 类型
Shared Memory
Warp
Instruction
```

代际不同。

---

# 116. ARM Target

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

# 117. Lowering

Lowering：

> 将高层 IR 逐步变成更低层、更加接近目标机器的 IR。

---

# 118. 一条 Lowering 链

```text
Relax Operator
↓
call_tir
↓
TensorIR PrimFunc
↓
Scheduled TensorIR
↓
Lowered TIR
↓
Target-specific IR
↓
Code。
```

---

# 119. 为什么不能一次 Lower？

Compiler 工程需要：

```text
每个阶段负责一种抽象。
```

否则：

```text
难扩展
难调试
难做 Pass。
```

---

# 120. 典型 Low-level Transformation

例如：

```text
Buffer Flattening
Thread Lowering
Intrinsic Lowering
Storage Rewrite
Vector Lowering
Host/Device Split。
```

---

# 121. CodeGen

CodeGen：

```text
Code Generation。
```

把：

```text
Lowered TIR
```

变成：

```text
Target Executable Code。
```

---

# 122. CPU CodeGen

```text
TensorIR
↓
LLVM IR
↓
x86 / ARM。
```

---

# 123. CUDA CodeGen

概念：

```text
GPU TensorIR
↓
CUDA / target code
↓
PTX / binary
↓
NVIDIA GPU。
```

---

# 124. Vulkan

```text
TensorIR
↓
SPIR-V
↓
Vulkan Driver
↓
GPU。
```

---

# 125. Metal

```text
TensorIR
↓
Metal Shader
↓
Apple GPU。
```

---

# 126. OpenCL

```text
TensorIR
↓
OpenCL Kernel
↓
Device Compiler
↓
GPU / Accelerator。
```

---

# 127. WebGPU

```text
TensorIR
↓
WGSL / WebGPU-oriented code
↓
Browser GPU。
```

---

# 128. 为什么 TVM 可以跨平台？

因为：

```text
High-level IR
```

与：

```text
Target CodeGen
```

分离。

---

# 129. 但是注意

```text
同一份 Relax
```

可以跨平台。

不代表：

```text
同一个 Schedule
```

跨平台最优。

---

# 130. CPU Schedule vs GPU Schedule

CPU：

```text
parallel
vectorize
cache blocking。
```

GPU：

```text
block/thread binding
shared memory
warp tile。
```

---

# 131. Target-aware Optimization

Compiler 应根据：

```text
Target
```

选择：

```text
不同 Schedule。
```

---

# 132. Host Code 与 Device Code

GPU Program 实际有两个部分。

Host：

```text
CPU 上运行。
```

负责：

```text
Memory
Kernel Launch
Control。
```

Device：

```text
GPU Kernel。
```

负责：

```text
Parallel Tensor Compute。
```

---

# 133. 为什么需要 Host / Device Split？

代码中：

```text
一些部分只能 CPU 做
一些部分应该 GPU 做。
```

Compiler 要拆开：

```text
Host Module
+
Device Module。
```

---

# 134. Runtime Module

TVM 编译产物通常包装为：

```text
runtime.Module。
```

里面可能：

```text
CPU Code
GPU Kernel
External Library
Metadata。
```

---

# 135. PackedFunc

TVM Runtime 的统一函数调用抽象之一：

```text
PackedFunc。
```

---

# 136. 为什么需要 PackedFunc？

Compiler 产物来自：

```text
Python
C++
LLVM
CUDA
External Runtime。
```

需要：

```text
统一 Function ABI。
```

---

# 137. Relax VM

Relax VM：

> 用来执行编译后的 Relax Program 的 Virtual Machine Runtime。

---

# 138. VM 是不是解释执行 MatMul？

不是。

VM 主要：

```text
Control / Dispatch。
```

真正：

```text
MatMul
```

由：

```text
编译好的 TIR Kernel
```

执行。

---

# 139. Relax VM Pipeline

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

# 140. VM Instruction

更像：

```text
Call
Ret
Goto
If
```

这类控制指令。

---

# 141. 所以：

```text
Relax VM
=
程序执行控制

TIR Kernel
=
Tensor Math。
```

---

# 142. AOT

Ahead-of-Time：

```text
Host Compiler
↓
Compiled Artifact
↓
Target Device
↓
Runtime。
```

---

# 143. 为什么端侧喜欢 AOT？

设备上：

```text
无需带完整 Compiler。
```

只需要：

```text
Runtime
+
Compiled Binary。
```

---

# 144. 这和 MLC-LLM 完全对应

MLC：

```text
Host
↓
compile model library
↓
Android / iOS / Web。
```

---

# 145. RPC

TVM 提供 RPC：

```text
Host
↔
Remote Device。
```

---

# 146. 为什么 Compiler 需要 RPC？

Auto-tuning：

```text
Host 生成 Candidate
↓
Target Device 实测
↓
返回 Latency。
```

---

# 147. 例如 RK3576 CPU

```text
Windows/Ubuntu PC
↓
Cross Compile ARM Kernel
↓
RPC / copy
↓
RK3576
↓
Benchmark。
```

---

# 148. 真实硬件是最终裁判

一个 Schedule：

```text
理论很好
```

并不代表：

```text
真实设备最快。
```

---

# 149. BYOC

BYOC：

```text
Bring Your Own Codegen。
```

---

# 150. 解决什么？

TVM 不可能：

```text
为每个 NPU
自己实现所有 CodeGen。
```

Vendor 已有：

```text
Compiler
Runtime
Kernel Library。
```

---

# 151. BYOC 思想

```text
Relax Graph
↓
Find Supported Pattern
↓
Fuse Subgraph
↓
Partition
↓
External CodeGen
↓
Vendor Module
↓
Runtime。
```

---

# 152. Pattern Match

例如：

```text
Conv
↓
Bias
↓
ReLU。
```

Vendor Accelerator：

```text
有 fused implementation。
```

Compiler 可以：

```text
把整个 Pattern 交过去。
```

---

# 153. Subgraph Partition

Graph：

```text
A → B → C → D
```

可能：

```text
A/B:
TVM

C/D:
External Accelerator。
```

---

# 154. 为什么这和 NPU 极其相关？

NPU：

```text
只支持特定 Operator / Layout / Dtype。
```

所以：

```text
Supported Subgraph
→ NPU

Unsupported
→ CPU。
```

---

# 155. CPU Fallback

如果某个 Op：

```text
NPU 不支持。
```

可能：

```text
NPU
↓
CPU
↓
NPU。
```

---

# 156. 为什么 Fallback 可能很慢？

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

# 157. ONNX Runtime EP 与 BYOC

之前学过：

```text
Execution Provider。
```

核心也是：

```text
Graph Partition
↓
Supported Nodes
↓
Backend。
```

---

# 158. 思想相似

```text
ONNX Runtime EP

TVM BYOC

Vendor NPU Graph Partition
```

本质都有：

```text
Backend Capability Analysis。
```

---

# 159. External Library Dispatch

并不是只有 NPU。

GPU：

```text
cuBLAS
cuDNN
CUTLASS
```

都可以作为：

```text
External high-performance implementation。
```

---

# 160. 为什么 Compiler 不必自己生成所有 MatMul？

因为：

```text
cuBLAS
```

可能已经：

```text
高度针对 GPU 优化。
```

直接调用：

```text
可能比自动生成 Kernel 更好。
```

---

# 161. Compiler 的目标不是“全自己做”

而是：

> **选出最快、正确、适配当前 Target 的实现。**

---

# 162. 三种 Kernel 获得方式

```text
1. Compiler Generate

2. AutoTune / Search

3. External Library Dispatch。
```

---

# 163. 实际系统经常三者混合

例如：

```text
Elementwise:
generated

MatMul:
cuBLAS / CUTLASS

Special fused pattern:
custom generated。
```

---

# 164. TVM 与 PyTorch

PyTorch：

```text
Model Development
Training
Eager / Compile Runtime。
```

TVM：

```text
Model Optimization
Compilation
Deployment。
```

---

# 165. PyTorch → TVM

现代一条重要路线：

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

# 166. 为什么 `torch.export` 适合 Compiler？

因为：

```text
它把 Python Program
变成更加规范、可分析的 Tensor Graph。
```

---

# 167. ONNX → TVM

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

# 168. 两条路线

```text
PyTorch → torch.export → Relax

PyTorch → ONNX → Relax。
```

---

# 169. Frontend 的职责

```text
External Model
↓
Read Operators
↓
Map Attributes
↓
Map Shape / Dtype
↓
Construct Relax。
```

---

# 170. Unsupported Operator

如果：

```text
PyTorch Op
```

TVM Frontend：

```text
没有 Converter。
```

会：

```text
Import Fail。
```

---

# 171. 这也是 AI Compiler 工程的一大块

不是所有工作都是：

```text
GPU Kernel。
```

还有大量：

```text
Frontend Operator Coverage
Shape Semantics
Dynamic Control Flow。
```

---

# 172. TVM 与 ONNX Runtime

ONNX Runtime：

```text
Graph Execution Runtime
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
Schedule
+
CodeGen。
```

---

# 173. TVM 更适合研究：

```text
Operator
最终如何变 Kernel。
```

---

# 174. llama.cpp 与 TVM

llama.cpp：

```text
LLM-specific Runtime
ggml Graph
Backend
Hand-optimized Kernel。
```

TVM：

```text
General ML Compiler
IR Transform
Generated Code。
```

---

# 175. 共同点

```text
Tensor
Graph
Backend
Kernel
Hardware。
```

---

# 176. 最大区别

```text
llama.cpp:
Runtime-centric

TVM:
Compiler-centric。
```

---

# 177. vLLM 与 TVM

vLLM：

```text
Request Scheduling
Continuous Batching
KV Management
Serving。
```

TVM：

```text
Model / Tensor Compilation。
```

---

# 178. 一个完整系统可以同时有二者

```text
vLLM Scheduler
↓
Compiled Model
↓
TVM / Triton / CUTLASS Kernel
↓
GPU。
```

---

# 179. MLC-LLM 与 TVM

这一对最直接。

```text
MLC-LLM
=
基于 TVM 的 LLM Compiler + Runtime Stack。
```

---

# 180. 前一章看到：

```text
MLC Model
↓
Relax
↓
Compiler Pass
↓
TIR
↓
Target。
```

本章：

```text
把这些概念真正拆开。
```

---

# 181. TVM 与 RKNN-Toolkit2

抽象层对比：

```text
TVM
│
├── Frontend
├── Graph IR
├── Graph Optimization
├── TensorIR
├── Target
├── CodeGen
└── Runtime

RKNN Toolchain
│
├── Model Import
├── Graph Analysis
├── Quantization
├── Operator Mapping
├── NPU Optimization
├── NPU Compilation
└── RKNN Runtime
```

---

# 182. 它们不是同一个 Compiler

但：

```text
Compiler Mental Model
```

高度相似。

---

# 183. `.rknn` 怎么理解？

不要只把它当：

```text
模型文件。
```

更应该理解：

> **已经经过 Rockchip Compiler 面向 RKNPU 做过转换、优化和硬件映射的部署 Artifact。**

---

# 184. 为什么 `.rknn` 不能像 ONNX 一样通用？

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

---

# 185. 这和 TVM Target-specific Artifact 很相似

概念：

```text
General IR
↓
Target Compilation
↓
Device Artifact。
```

---

# 186. RKNN Unsupported Op 怎么理解？

Compiler Pipeline：

```text
Frontend
↓
Graph
↓
Operator Legalization
↓
Target Mapping。
```

失败可能是：

```text
这个 Op
无法映射到 RKNPU。
```

---

# 187. RKNN Quantization Failure

可能处于：

```text
Calibration
Dtype Conversion
Scale Propagation
Operator Constraint
Accuracy Constraint。
```

---

# 188. Graph Partition

如果：

```text
某部分支持 NPU
某部分不支持。
```

可以：

```text
Partition。
```

---

# 189. 为什么学 TVM 后看 Vendor Compiler 更容易？

你已经拥有语言：

```text
Frontend
IR
Pass
Legalization
Lowering
Target
CodeGen
Runtime。
```

---

# 190. MatMul：完整编译旅程

现在用一个 MatMul 贯穿整章。

PyTorch：

```python
y = torch.matmul(x, w)
```

---

# 191. Step 1：Graph Capture

```text
torch.export
```

把：

```text
Python Program
```

转为：

```text
Tensor Graph。
```

---

# 192. Step 2：Frontend

TVM：

```text
ExportedProgram
↓
Relax。
```

---

# 193. Step 3：Relax

IR：

```text
R.matmul(x, w)。
```

---

# 194. Step 4：Graph Optimization

可能：

```text
MatMul + Bias + Activation
```

做：

```text
Fusion。
```

---

# 195. Step 5：Legalize

```text
R.matmul
↓
call_tir
↓
TensorIR MatMul PrimFunc。
```

---

# 196. Step 6：Naive Tensor Program

```text
for i
    for j
        for k
            C[i,j] += A[i,k] * B[k,j]。
```

---

# 197. Step 7：Schedule

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
bind
shared memory
register blocking。
```

---

# 198. Step 8：Tensorize

如果 Target 支持：

```text
Tensor Core / Matrix Unit。
```

---

# 199. Step 9：Lowering

```text
High-level TensorIR
↓
Lowered Target-oriented TensorIR。
```

---

# 200. Step 10：CodeGen

CPU：

```text
LLVM
↓
ARM/x86。
```

GPU：

```text
CUDA / SPIR-V / Metal。
```

---

# 201. Step 11：Runtime

```text
Load Module
↓
Allocate Buffer
↓
Launch Kernel
↓
Get C。
```

---

# 202. 这就是整章最重要的一条链

```text
torch.matmul
↓
Relax MatMul
↓
TensorIR
↓
Schedule
↓
Lower
↓
CodeGen
↓
Machine / GPU Code。
```

---

# 203. 为什么同样 MatMul 会差几十倍？

FLOPs：

```text
一样。
```

但：

```text
Memory Traffic
Cache Hit
Vector Width
Threads
Occupancy
Tensor Core Usage
```

可能完全不同。

---

# 204. AI Compiler 真正优化的不是只有 FLOPs

更重要：

```text
Data Movement
+
Parallelism
+
Hardware Utilization。
```

---

# 205. Data Movement 为什么重要？

GPU：

```text
HBM
↓
L2
↓
Shared Memory
↓
Register。
```

每一层：

```text
速度 / 容量不同。
```

---

# 206. 好 Kernel 的核心问题

```text
数据什么时候搬？

搬到哪里？

搬多少？

一块数据能复用几次？
```

---

# 207. 这也是 FlashAttention 的核心思想

FlashAttention：

```text
不改变 Attention 数学结果
```

但重构：

```text
Tiling
Memory Movement
Softmax Execution。
```

---

# 208. 所以 TVM → GPU Kernel 是自然过渡

Compiler 最终想生成：

```text
更好的 Kernel。
```

下一章就研究：

```text
Kernel 为什么好。
```

---

# 209. Triton 与 TVM

Triton：

```text
GPU Kernel Language / Compiler。
```

TVM：

```text
End-to-end ML Compiler。
```

---

# 210. Triton 更靠近：

```text
TensorIR / Kernel CodeGen
```

这一层。

---

# 211. CUTLASS 与 TVM

CUTLASS：

```text
CUDA GEMM / Tensor Core
高性能模板库。
```

TVM：

```text
可以生成自己的 Kernel
```

也可以：

```text
External Dispatch CUTLASS。
```

---

# 212. FlashAttention 与 TVM

FlashAttention：

```text
Attention-specific Algorithm + Kernel。
```

TVM：

```text
Compiler Infrastructure。
```

二者可以在：

```text
Tiling
Fusion
Memory
Schedule
```

上产生非常强的联系。

---

# 213. 推荐源码阅读顺序

TVM 很大。

不要：

```text
一开始读 CodeGen C++。
```

推荐：

```text
README
↓
Architecture
↓
TVMScript
↓
Relax
↓
LegalizeOps
↓
TensorIR
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

---

# 214. 第一阶段：Architecture

重点：

- [Architecture Index](https://github.com/apache/tvm/blob/main/docs/arch/index.rst)

需要找到：

```text
IRModule
Relax Function
PrimFunc
Transformation
Target
Runtime。
```

---

# 215. 第二阶段：TVMScript

重点：

- [TVMScript Architecture](https://github.com/apache/tvm/blob/main/docs/arch/tvmscript.rst)

目标：

> 会“读 IR”。

---

# 216. 为什么先学 IR 可视化？

Compiler Debug 最重要的方式之一：

```text
Pass Before
↓
Print IR
↓
Pass
↓
Print IR After。
```

---

# 217. 第三阶段：Relax

重点：

- [Relax Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/relax)
- [Understand Relax](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/learning.rst)
- [Relax Transformation](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/tutorials/relax_transformation.py)

---

# 218. Relax 必须掌握

```text
R.function
R.dataflow
R.Tensor
StructInfo
Symbolic Shape
R.call_tir
LegalizeOps
Fusion。
```

---

# 219. 第四阶段：TensorIR

重点：

- [TensorIR Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/tensor_ir)

---

# 220. TensorIR 必须掌握

```text
PrimFunc
Buffer
Block / SBlock
Loop
Spatial Axis
Reduction Axis
Schedule。
```

---

# 221. 第五阶段：Schedule

当前源码：

- [src/s_tir/schedule](https://github.com/apache/tvm/tree/main/src/s_tir/schedule)

重点：

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

# 222. 第六阶段：DLight

推荐：

- [DLight GPU Scheduling Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/dlight_gpu_scheduling.py)

---

# 223. 第七阶段：MetaSchedule

推荐：

- [MetaSchedule Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/meta_schedule.py)

---

# 224. 第八阶段：CodeGen

推荐：

- [Code Generation Architecture](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst)

然后再看：

```text
src/target
```

---

# 225. 第九阶段：Relax VM

- [Relax VM Architecture](https://github.com/apache/tvm/blob/main/docs/arch/relax_vm.rst)

---

# 226. 第十阶段：BYOC

- [External Library Dispatch](https://github.com/apache/tvm/blob/main/docs/arch/external_library_dispatch.rst)

---

# 227. Compiler 源码的正确读法

假设想理解：

```text
sch.split()
```

不要直接：

```text
全文搜索 C++。
```

而是：

```text
Python API
↓
FFI
↓
C++ Schedule Primitive
↓
IR Mutation。
```

---

# 228. 为什么需要这种阅读方式？

TVM 底层包含：

```text
ObjectRef
FFI
Reflection
IR Node
C++ Template
```

如果没有 API 主线：

```text
很容易迷路。
```

---

# 229. 本章必做实验 1：创建 Relax Graph

创建：

```text
MatMul + ReLU。
```

打印：

```text
Relax IR。
```

---

# 230. 实验 2：LegalizeOps

执行：

```text
LegalizeOps。
```

观察：

```text
Before:
R.matmul

After:
R.call_tir
+
PrimFunc。
```

---

# 231. 这个实验最重要

因为你第一次真正看到：

```text
Graph IR
↓
Tensor IR。
```

---

# 232. 实验 3：Vector Add

手写：

```text
PrimFunc。
```

---

# 233. 实验 4：Naive MatMul

手写：

```text
M/N/K
三层 Loop。
```

---

# 234. 实验 5：Split

对：

```text
i / j
```

做：

```text
split。
```

打印：

```text
IR Before / After。
```

---

# 235. 实验 6：Reorder

调整 Loop 顺序。

Benchmark。

---

# 236. 实验 7：CPU Parallel

外层：

```text
parallel。
```

看：

```text
多核收益。
```

---

# 237. 实验 8：Vectorize

内层：

```text
vectorize。
```

查看：

```text
LLVM 生成效果。
```

---

# 238. 实验 9：Tiled MatMul

比较：

```text
Naive
vs
Tiled。
```

记录：

```text
Latency。
```

---

# 239. 实验 10：GPU Thread Binding

如果有 NVIDIA GPU：

```text
blockIdx.x
threadIdx.x。
```

---

# 240. 实验 11：Shared Memory

```text
cache_read
→ shared。
```

---

# 241. 实验 12：Register Tile

尝试：

```text
local buffer。
```

---

# 242. 实验 13：Generated Code

打印：

```text
CPU LLVM IR

or

GPU source / PTX。
```

---

# 243. 实验 14：DLight

同一个 MatMul：

```text
Manual
vs
DLight。
```

---

# 244. 实验 15：MetaSchedule

搜索：

```text
不同 Schedule。
```

记录：

```text
Best Latency。
```

---

# 245. 实验 16：PyTorch → Relax

```text
PyTorch Model
↓
torch.export
↓
TVM Relax
↓
Compile
↓
Run。
```

---

# 246. 实验 17：ONNX → Relax

同一个模型：

```text
ONNX
↓
Relax。
```

比较 Frontend。

---

# 247. 实验 18：LLVM ARM

将一个简单模型：

```text
Target = ARM。
```

Cross Compile。

---

# 248. 实验 19：RK3576 CPU

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

---

# 249. 为什么这个实验有价值？

它和：

```text
RKNPU
```

无关。

但可以建立：

```text
General Compiler → ARM CPU
```

Baseline。

---

# 250. 实验 20：与 RKNN 对照

同一个：

```text
Small CNN / MLP。
```

路线 A：

```text
PyTorch
↓
ONNX
↓
TVM
↓
ARM CPU。
```

路线 B：

```text
PyTorch
↓
ONNX
↓
RKNN-Toolkit2
↓
RKNPU。
```

---

# 251. 对比什么？

```text
Model Import
Supported Ops
Quantization
Compile Time
Runtime API
Latency
Memory
Debug Experience。
```

---

# 252. 这个实验的目的

不是：

```text
证明 NPU 一定更快。
```

而是：

> 理解 General Compiler 与 Vendor Accelerator Compiler 的差别。

---

# 253. 本章推荐建立一个实验仓库

```text
tvm-learning/
│
├── 01_relax_basic.py
├── 02_relax_legalize.py
├── 03_tir_vector_add.py
├── 04_tir_matmul_naive.py
├── 05_schedule_split.py
├── 06_schedule_reorder.py
├── 07_cpu_parallel.py
├── 08_cpu_vectorize.py
├── 09_tiled_matmul.py
├── 10_cuda_bind.py
├── 11_shared_memory.py
├── 12_dlight.py
├── 13_meta_schedule.py
├── 14_torch_export_relax.py
├── 15_onnx_relax.py
└── 16_rk3576_cpu.py
```

---

# 254. 每个实验固定记录

```text
Input Shape

Target

Original IR

Transformed IR

Generated Code

Latency

Correctness

Peak Memory。
```

---

# 255. 为什么要保存 IR？

因为最终：

```text
你可以肉眼看到 Compiler
到底做了什么。
```

---

# 256. 性能 Debug 三层法

遇到：

```text
模型慢。
```

不要直接调 Kernel。

先分：

```text
Graph Problem

Tensor / Kernel Problem

Runtime Problem。
```

---

# 257. Graph Problem

例如：

```text
Fusion 没发生
多余 Layout Transform
Subgraph Partition 太碎
Constant 没 Fold。
```

---

# 258. Kernel Problem

例如：

```text
Tile 不合理
Memory Coalescing 差
Occupancy 低
Vectorization 不好
Tensor Core 没用上。
```

---

# 259. Runtime Problem

例如：

```text
Kernel Launch
Memory Copy
Synchronization
Allocator
Device Transfer。
```

---

# 260. Compiler Correctness Debug

如果输出错误：

```text
PyTorch baseline
↓
Frontend output
↓
Relax
↓
Legalized IR
↓
TensorIR
↓
Compiled result。
```

逐层对比。

---

# 261. Numerical Validation

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

# 262. Quantization Compiler

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

# 263. 这和 RKNN 直接相关

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

# 264. Compiler 与 Kernel 的关系

Compiler 最终要产生：

```text
Kernel。
```

Kernel 真正决定：

```text
每个 Thread
读什么
写什么
怎么算。
```

---

# 265. 所以下一步为什么学 GPU Kernel？

因为：

```text
Schedule
```

最终只是抽象。

下一步要理解：

```text
什么 Schedule
在 GPU 上为什么快。
```

---

# 266. 下一阶段核心概念

```text
Thread
Warp
Block
Grid

Register
Shared Memory
Global Memory

Memory Coalescing
Occupancy

GEMM Tiling

Tensor Core

FlashAttention

Triton

CUTLASS。
```

---

# 267. 本章知识树

```text
AI Compiler
│
├── Frontend
│   ├── PyTorch
│   ├── torch.export
│   ├── ONNX
│   └── Custom Model
│
├── IRModule
│   ├── Relax Function
│   └── TensorIR PrimFunc
│
├── Relax
│   ├── Tensor
│   ├── Dataflow
│   ├── StructInfo
│   ├── Symbolic Shape
│   ├── Operator
│   ├── call_tir
│   └── Control Flow
│
├── Graph Pass
│   ├── LegalizeOps
│   ├── Fusion
│   ├── Constant Folding
│   ├── DCE
│   ├── Pattern Rewrite
│   └── Layout
│
├── TensorIR
│   ├── tirx
│   │   ├── PrimFunc
│   │   ├── Buffer
│   │   ├── SBlock
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
    ├── BYOC
    ├── Pattern Match
    ├── Graph Partition
    ├── cuBLAS
    ├── cuDNN
    ├── CUTLASS
    └── Vendor NPU
```

---

# 268. 本章最重要的十个思维模型

## 1

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

## 2

```text
Relax
=
Graph Level。
```

---

## 3

```text
TensorIR
=
Tensor Program / Kernel Level。
```

---

## 4

```text
Computation
=
What

Schedule
=
How。
```

---

## 5

```text
Same FLOPs
≠
Same Performance。
```

---

## 6

```text
Performance
=
Compute
+
Data Movement
+
Parallelism
+
Hardware Utilization。
```

---

## 7

```text
Tiling
=
Data Reuse。
```

---

## 8

```text
Tensorization
=
Loop Pattern
→
Hardware Tensor Instruction。
```

---

## 9

```text
Compiler
不一定自己生成所有 Kernel。
```

可以：

```text
Generate
Search
Library Dispatch。
```

---

## 10

```text
TVM
提供的是理解所有 AI Compiler
的一套共同语言。
```

---

# 269. 本章完成标准

## Compiler 基础

- [ ] 能解释 AI Compiler
- [ ] 能画 Frontend → IR → Lowering → CodeGen
- [ ] 能解释 Compiler vs Runtime
- [ ] 能解释为什么 AI Compiler 需要 Graph IR 和 Tensor IR

## Relax

- [ ] 能解释 Relax
- [ ] 能解释 Relax Function
- [ ] 能解释 Dataflow
- [ ] 能解释 StructInfo
- [ ] 能解释 Symbolic Shape
- [ ] 能解释 Dynamic Shape
- [ ] 能解释 `R.call_tir`

## IRModule

- [ ] 能解释 IRModule
- [ ] 能解释 Relax Function 与 PrimFunc 共存
- [ ] 能解释 Cross-level Optimization

## TensorIR

- [ ] 能解释 PrimFunc
- [ ] 能解释 Buffer
- [ ] 能解释 Block / SBlock
- [ ] 能解释 Spatial Axis
- [ ] 能解释 Reduction Axis
- [ ] 能解释 Loop Nest
- [ ] 能解释 `tirx`
- [ ] 能解释 `s_tir`

## Graph Optimization

- [ ] 能解释 LegalizeOps
- [ ] 能解释 Fusion
- [ ] 能解释 Constant Folding
- [ ] 能解释 Dead Code Elimination
- [ ] 能解释 Layout Transformation
- [ ] 能解释 Pattern Rewrite

## Schedule

- [ ] 能解释 Schedule
- [ ] 能解释 computation vs schedule
- [ ] 能解释 split
- [ ] 能解释 reorder
- [ ] 能解释 fuse
- [ ] 能解释 parallel
- [ ] 能解释 vectorize
- [ ] 能解释 unroll
- [ ] 能解释 bind
- [ ] 能解释 cache_read
- [ ] 能解释 cache_write
- [ ] 能解释 Tiling
- [ ] 能解释 Register Blocking

## Tensorization

- [ ] 能解释 Tensor Intrinsic
- [ ] 能解释 Tensorize
- [ ] 能解释 Tensor Core / Matrix Engine 如何接入
- [ ] 能解释为什么 NPU Compiler 也需要类似机制

## Auto Optimization

- [ ] 能解释 DLight
- [ ] 能解释 MetaSchedule
- [ ] 能解释 Design Space
- [ ] 能解释 Cost Model
- [ ] 能解释为什么真实 Benchmark 很重要

## Target / CodeGen

- [ ] 能解释 Target
- [ ] 能解释 Lowering
- [ ] 能解释 LLVM
- [ ] 能解释 CUDA CodeGen
- [ ] 能解释 Vulkan / SPIR-V
- [ ] 能解释 Host / Device Split
- [ ] 能解释 Runtime Module

## Runtime

- [ ] 能解释 PackedFunc
- [ ] 能解释 Relax VM
- [ ] 能解释 VM 为什么不直接做 MatMul
- [ ] 能解释 AOT
- [ ] 能解释 RPC

## BYOC

- [ ] 能解释 BYOC
- [ ] 能解释 Pattern Matching
- [ ] 能解释 Graph Partition
- [ ] 能解释 External CodeGen
- [ ] 能解释 Vendor NPU 为什么适合 BYOC 思维
- [ ] 能解释 BYOC 与 ONNX Runtime EP 的共同思想

## 实践

- [ ] 能打印一个 Relax IR
- [ ] 能执行一次 LegalizeOps
- [ ] 能看到 `R.matmul → call_tir`
- [ ] 能手写 Vector Add PrimFunc
- [ ] 能手写 Naive MatMul PrimFunc
- [ ] 能进行 split/reorder
- [ ] 能做 CPU Parallel
- [ ] 能做 Vectorize
- [ ] 能完成一次 Tiled MatMul
- [ ] 最好能做 GPU Thread Binding
- [ ] 能查看 Generated Code
- [ ] 能完成 PyTorch → Relax
- [ ] 能解释 TVM 与 MLC-LLM
- [ ] 能解释 TVM 与 RKNN

---

# 270. 暂时不要求深入

当前阶段暂时不要求：

- [ ] 自己实现 Relax Parser
- [ ] 自己实现 IR AST
- [ ] 精通所有 Relax Pass
- [ ] 精通所有 s_tir Primitive
- [ ] 自己实现 MetaSchedule Cost Model
- [ ] 精通 Polyhedral Compiler
- [ ] 自己实现 LLVM Backend
- [ ] 自己实现 CUDA CodeGen
- [ ] 自己实现完整 BYOC Backend
- [ ] 自己实现 RKNPU Target
- [ ] 精通所有 TVM Runtime
- [ ] 精通分布式 TVM

本章目标是：

```text
建立 Compiler Mental Model
+
能够看懂 IR
+
能够做基础 Schedule。
```

---

# 271. 核心参考链接

## TVM 主仓库

- [apache/tvm](https://github.com/apache/tvm)

---

## Architecture

- [TVM Architecture Index](https://github.com/apache/tvm/blob/main/docs/arch/index.rst)

重点：

```text
IRModule
Relax
tirx
s_tir
Target
Runtime。
```

---

## Relax

- [Relax Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/relax)
- [Relax Abstraction](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/abstraction.rst)
- [Understand Relax](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/learning.rst)
- [Relax Transformation](https://github.com/apache/tvm/blob/main/docs/deep_dive/relax/tutorials/relax_transformation.py)

---

## TensorIR

- [TensorIR Deep Dive](https://github.com/apache/tvm/tree/main/docs/deep_dive/tensor_ir)
- [TensorIR Index](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/index.rst)
- [TensorIR Learning](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/learning.rst)
- [TensorIR Transformation](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/tir_transformation.py)

---

## Schedule

- [s_tir Schedule Source](https://github.com/apache/tvm/tree/main/src/s_tir/schedule)

---

## DLight

- [DLight GPU Scheduling](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/dlight_gpu_scheduling.py)

---

## MetaSchedule

- [MetaSchedule Tutorial](https://github.com/apache/tvm/blob/main/docs/deep_dive/tensor_ir/tutorials/meta_schedule.py)

---

## TVMScript

- [TVMScript Architecture](https://github.com/apache/tvm/blob/main/docs/arch/tvmscript.rst)

---

## CodeGen

- [Code Generation Architecture](https://github.com/apache/tvm/blob/main/docs/arch/codegen.rst)

---

## Relax VM

- [Relax VM Architecture](https://github.com/apache/tvm/blob/main/docs/arch/relax_vm.rst)

---

## BYOC

- [External Library Dispatch / BYOC](https://github.com/apache/tvm/blob/main/docs/arch/external_library_dispatch.rst)

---

# 272. 与整套学习路线的关系

前面：

```text
01
AI / ML

02
Deep Learning

03
Transformer / LLM

04
PyTorch

05
Agent / RAG

06
LLM Inference

07
llama.cpp

08
vLLM / MLC-LLM
```

已经一路从：

```text
模型
```

进入：

```text
Runtime。
```

现在：

```text
10 AI Compiler / TVM
```

正式开始继续向下：

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

# 273. 与后续 GPU Kernel 的关系

下一阶段：

```text
TensorIR Schedule
↓
GPU Kernel
```

继续深入：

```text
Thread
Warp
Block
Grid
Shared Memory
Register
Global Memory
Coalescing
Occupancy
GEMM
Tensor Core
FlashAttention
Triton
CUTLASS。
```

---

# 274. 与 RKNN 的关系

再往后：

```text
TVM
=
General AI Compiler

RKNN-Toolkit2
=
Vendor NPU Compiler Toolchain。
```

你会发现：

```text
模型转换失败
算子不支持
量化异常
Fallback
NPU Compile
```

不再是零散 SDK 问题。

而是：

```text
Compiler Pipeline 中的具体问题。
```

---

# 275. 最终总图

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

# 276. 一句话总结

这章最重要的不是：

```text
记住 TVM API。
```

而是彻底建立：

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
Target
↓
CodeGen
↓
Kernel
↓
Hardware
```

这条链。

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
如何让同一个计算更快

Target
=
到底为哪块硬件优化

CodeGen
=
把 IR 真正翻译成目标代码
```

当这一条链能独立解释以后，下一步学习：

> **GPU Kernel、GEMM、Triton、FlashAttention、CUTLASS**

就不再是零散 CUDA 技巧。

而是在回答：

> **AI Compiler 最终到底要生成怎样的 Kernel，才能把硬件算力真正发挥出来？**
