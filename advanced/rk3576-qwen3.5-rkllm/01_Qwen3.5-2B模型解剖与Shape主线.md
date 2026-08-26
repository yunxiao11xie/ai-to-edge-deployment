---
tags: [Qwen3.5, 模型结构, Shape, GatedDeltaNet, GQA, RK3576]
---

# Qwen3.5-2B 模型解剖与 Shape 主线

> [!important] 本章结论
> Qwen3.5-2B 不是一个“24 层标准 Decoder-only Self-Attention 模型”。官方 Config 显示它使用**混合层结构**：3 个线性注意力层后接 1 个全注意力层，重复 6 次。部署、缓存和性能分析必须基于这个真实结构。

## 1. 为什么部署前必须先读 Config

面对一个模型，第一步不应是复制转换脚本，而是回答：

```text
模型类是什么？
输入不只是文本吗？
有多少层？
每层是哪种类型？
Hidden / Head / FFN / Vocabulary 多大？
是否使用 GQA？
是否包含 Vision Encoder？
Tokenizer 与 Chat Template 在哪里？
```

这些信息决定：

- Toolkit 是否支持该架构；
- 哪些 Tensor 需要转换；
- Prefill/Decode 的 Shape；
- 缓存和工作区大小；
- `.rkllm` 为什么比理想 4bit 权重估算更大；
- 多模态时哪些部分走 RKNN、哪些走 RKLLM。

## 2. 官方 Config 的关键字段

Qwen3.5-2B 当前官方配置的核心字段如下：

| 字段 | 值 | 部署意义 |
|---|---:|---|
| Architecture | `Qwen3_5ForConditionalGeneration` | 顶层是多模态生成模型，不只是纯文本类 |
| Language Parameters | 2B 级 | 决定权重容量的主量级 |
| `hidden_size` | 2048 | 主 Hidden State Shape 的 D |
| `num_hidden_layers` | 24 | 文本主干层数 |
| `intermediate_size` | 6144 | FFN 中间维度 |
| `vocab_size` | 248320 | Embedding/LM Head 很大 |
| `tie_word_embeddings` | true | 输入 Embedding 与输出 LM Head 共享权重 |
| `num_attention_heads` | 8 | 全注意力层的 Q Head 数 |
| `num_key_value_heads` | 2 | 全注意力层使用 GQA |
| `head_dim` | 256 | 8×256=2048，与 Hidden Size 对齐 |
| `full_attention_interval` | 4 | 每 4 层出现一次全注意力 |
| `max_position_embeddings` | 262144 | 模型原生配置上限，不等于 4GB 板端可用长度 |
| Vision Depth | 24 | 模型还包含视觉编码器 |
| Vision Hidden | 1024 | 视觉主干内部维度 |
| Vision Output Hidden | 2048 | 视觉特征最终对齐语言 Hidden Size |

> [!warning] 配置上限不是部署建议
> `max_position_embeddings=262144` 只说明模型配置支持的原生上下文范围。RK3576 4GB 环境必须根据 Runtime 支持、状态内存、工作区、延迟和系统余量重新选择 2048/4096 等实际值。

## 3. 24 层的真实排列

`layer_types` 显示如下规律：

```text
Layer 00  Linear Attention
Layer 01  Linear Attention
Layer 02  Linear Attention
Layer 03  Full Attention

Layer 04  Linear Attention
Layer 05  Linear Attention
Layer 06  Linear Attention
Layer 07  Full Attention

...重复...

Layer 20  Linear Attention
Layer 21  Linear Attention
Layer 22  Linear Attention
Layer 23  Full Attention
```

因此：

```text
24 Layers
├── 18 Linear-Attention / Gated DeltaNet Layers
└──  6 Full-Attention / Gated Attention Layers
```

这会直接改变缓存分析：

- 6 个全注意力层维护随序列增长的 K/V；
- 18 个线性注意力层使用其自身的递推状态/卷积状态；
- 不能使用 `24×T×KV Heads×Head Dim` 估算整个模型的标准 KV Cache；
- RKLLM 内部对这些状态的具体布局属于 Runtime 实现，必须通过日志、头文件和实测验证。

## 4. 文本输入到输出的 Shape 主线

固定符号：

```text
B = Batch，端侧交互通常为 1
T = 当前输入 Token 数
D = Hidden Size = 2048
V = Vocabulary Size = 248320
Dff = FFN Intermediate = 6144
Hq = Query Heads = 8
Hkv = KV Heads = 2
Dh = Head Dim = 256
```

主数据流：

```text
Input IDs       [B, T]
    ↓ Embedding Lookup
Hidden          [B, T, 2048]
    ↓ 24 Hybrid Layers
Hidden          [B, T, 2048]
    ↓ Final Norm
Hidden          [B, T, 2048]
    ↓ Tied LM Head
Logits          [B, T, 248320]
```

在 Decode 中通常只需要最后位置的 Logits：

```text
Last Hidden [B, 1, 2048]
    ×
LM Head Weight [2048, 248320]
    ↓
Next-token Logits [B, 1, 248320]
```

248320 的大词表意味着：

- Embedding 参数量约为 `248320×2048≈508.6M`；
- 如果不共享 LM Head，输出层还会再增加约 508.6M 参数；
- Weight Tying 避免重复保存这块巨大权重；
- 每个 Decode Step 的最终词表投影与 Sampling 也不是免费的。

## 5. 全注意力层的 Shape

对全注意力层：

```text
X   [B,T,2048]
Q   [B,8,T,256]
K   [B,2,T,256]
V   [B,2,T,256]
```

因为 `Hq=8`、`Hkv=2`，每 4 个 Query Head 共享一组 K/V，这是 GQA。

概念计算：

```text
Q = XWq
K = XWk
V = XWv
Q,K 应用部分 RoPE
Score = QKᵀ / sqrt(Dh)
Score += Causal Mask
Prob = Softmax(Score)
Context = Prob × V
Output = Context × Wo
```

Prefill 时：

```text
QKᵀ → [B,8,T,T]
```

Decode 时，新 Query 长度为 1：

```text
Q_new     [B,8,1,256]
K_cache   [B,2,T,256]
Score     [B,8,1,T]
```

Runtime 会处理 GQA 的 Head 映射，不能简单把 K/V 物理复制 4 份后再认为那就是实际缓存布局。

## 6. 线性注意力层为什么不同

标准全注意力显式比较当前 Query 与历史 Key，历史长度 T 会进入计算和 KV Cache。

线性注意力/Gated DeltaNet 采用递推状态更新的思路，目标之一是避免构造完整的 `T×T` 注意力矩阵。可以先建立如下抽象：

```text
Full Attention:
历史信息主要以 K/V Sequence Cache 保存
状态规模随 T 线性增长

Linear Attention / DeltaNet:
历史信息压缩进固定或近固定大小的递推状态
每步更新 State，再由当前输入读取 State
```

但不要从通用论文直接推断 RKLLM 的内部 Buffer。Qwen3.5 Config 还包含：

- 16 个线性 QK Head；
- 16 个线性 V Head；
- 线性 Head Dim 128；
- 卷积 Kernel Dim 4；
- `mamba_ssm_dtype=float32`。

这些字段说明线性层存在不同于全注意力 KV Cache 的状态与数值精度要求。实际内存必须通过 Runtime 的 Perf Stat、进程映射和上下文 Sweep 测量。

## 7. FFN 的参数与计算

每层 FFN 的中间维度为 6144。若采用常见 gated FFN 结构，可抽象为：

```text
gate = SiLU(XW_gate)
up   = XW_up
out  = (gate ⊙ up)W_down
```

对应 Shape：

```text
X       [B,T,2048]
Gate    [B,T,6144]
Up      [B,T,6144]
Output  [B,T,2048]
```

仅三个大矩阵的参数量级约为：

```text
2048×6144×2 + 6144×2048
= 37,748,736 parameters / layer
```

24 层仅 FFN 大矩阵就接近 0.906B 参数。这个估算解释了为什么：

- LLM 不只是 Attention 贵；
- Decode 每步仍要读取大量 FFN 权重；
- Weight-only 量化对 Decode 带宽非常重要。

这是结构级估算，不等于精确参数统计；精确值应从模型 Tensor 清单求和。

## 8. 视觉编码器的真实位置

Qwen3.5-2B 顶层是 `ForConditionalGeneration`，官方 Config 含 Vision Encoder：

```text
Image / Video
    ↓ Processor
Pixel Tensor + Grid Metadata
    ↓ Vision Encoder (Depth 24, Hidden 1024)
Visual Features
    ↓ Merge / Projection
Visual Embeddings [N_visual, 2048]
    ↓ 与 Text Embeddings 拼接
Language Backbone
```

这解释了当前实战为什么只完成语言部分：RKLLM-Toolkit 导出语言主干为 `.rkllm`；真正多模态还要把视觉部分导出/转换为 `.rknn`，并通过 Embedding/Multimodal Input 接入 RKLLM。

## 9. 模型目录里分别有什么

一个 Hugging Face 模型目录至少要区分：

| 文件 | 作用 | 出错表现 |
|---|---|---|
| `config.json` | 模型结构 | 架构识别失败、Shape 不一致 |
| `*.safetensors` | 权重 Tensor | 缺文件、Hash 不一致、加载失败 |
| `tokenizer.json` 等 | 文本↔Token | 输出乱码、特殊 Token 错误 |
| `chat_template` | Role 格式 | 模型不按对话方式回答 |
| Processor Config | 图像/视频预处理 | 视觉 Token 和输入 Shape 错误 |
| Generation Config | 默认采样参数 | 输出风格与长度变化 |

“模型下载完成”必须通过文件完整性、Git Revision/Commit、Hash 和一次 Hugging Face 基线推理确认。

## 10. 实验

### 实验 1：Config 审计

从 `config.json` 输出：

```text
Architecture
Hidden Size
Layer Count
Layer Types
Vocabulary
Full Attention Interval
Q/KV Heads
Head Dim
FFN Intermediate
Vision Config
```

验收：能指出 6 个 Full Attention Layer 的索引。

### 实验 2：Tensor 清单与参数统计

遍历 Safetensors Header，按模块统计：

```text
Embedding
Linear Attention
Full Attention
FFN
Norm
Vision Encoder
```

验收：总和与模型规模一致，并解释最大 Tensor 来自哪里。

### 实验 3：Shape Trace

在 Host Hugging Face Baseline 中给关键模块注册 Hook，记录短文本下的 Shape；视觉模式单独记录 Pixel、Grid 和 Visual Embedding Shape。

## 11. 常见误区

| 误区 | 正确认识 |
|---|---|
| Qwen3.5-2B 就是小号 LLaMA | 它使用混合线性/全注意力结构并带 Vision Encoder |
| 24 层都需要标准 KV Cache | 只有全注意力层按标准 K/V Sequence 理解，线性层有其他状态 |
| 262K Context 在 4GB 板上也能直接开 | 配置能力不等于设备容量与 Runtime 支持 |
| 2B 参数全部都能按 4bit 计算文件大小 | Artifact 还有高精度 Tensor、元数据、布局和工具链表示 |
| 模型权重就是完整模型 | Tokenizer、Config、Chat Template、Processor 同样属于模型契约 |

## 12. 验收清单

- [ ] 能画出 18 个 Linear Attention + 6 个 Full Attention 的层次结构；
- [ ] 能写出全注意力层 Prefill/Decode 的 Q/K/V Shape；
- [ ] 能解释 GQA 为什么减少 K/V Cache；
- [ ] 能估算 Embedding 和每层 FFN 的参数量级；
- [ ] 能解释 Vision Output 为什么是 2048；
- [ ] 不再对所有 24 层套用标准 KV Cache 公式；
- [ ] 能区分官方事实、结构估算和 RKLLM 内部未知实现。

## 13. 参考

- [Qwen3.5-2B Model Card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5-2B Config](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)
- [Rockchip RKLLM Qwen3.5 Multimodal Export Example](https://github.com/airockchip/rknn-llm/tree/main/examples/multimodal_model_demo)
- [04_Prefill_Decode_混合注意力与缓存](./04_Prefill_Decode_混合注意力与缓存.md)
- [08_IMX415_RKNN_RKLLM多模态链路](./08_IMX415_RKNN_RKLLM多模态链路.md)
