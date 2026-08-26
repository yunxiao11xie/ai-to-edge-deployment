# 03_Transformer与LLM

> **所属路线**：AI 学习路线 · 第一部分  
> **定位**：从 Language Modeling、Attention、Transformer，一直到 GPT / LLM 的预训练、指令微调与自回归推理  
> **核心仓库**：`rasbt/LLMs-from-scratch` + `jingyaogong/minimind` + `karpathy/nn-zero-to-hero` + `d2l-ai/d2l-en`  
> **主要框架**：PyTorch  
> **学习边界**：本章重点解决“LLM 是什么、如何训练、如何推理”；Agent、RAG、llama.cpp、vLLM、MLC-LLM 放到后续文档。  
> **资料检查日期**：2026-08-25

---

# 0. 本章学习目标

完成本章以后，需要能够回答：

1. Language Model 到底在学习什么？
2. 为什么 GPT 本质上可以理解成一个 `Next Token Predictor`？
3. Token、Token ID、Vocabulary、Tokenizer 分别是什么？
4. BPE 为什么比简单字符级分词更实用？
5. Embedding 为什么可以把离散 Token 变成连续向量？
6. 为什么 RNN 之后会出现 Attention？
7. Query、Key、Value 到底是什么？
8. Self-Attention 的矩阵计算过程是什么？
9. 为什么 Attention 要除以 `sqrt(d_k)`？
10. Causal Mask 在 GPT 中解决什么问题？
11. Multi-Head Attention 为什么比单头 Attention 更有表达能力？
12. Transformer Block 由哪些核心组件组成？
13. Residual Connection、LayerNorm / RMSNorm、FFN 分别做什么？
14. Encoder-only、Encoder-Decoder、Decoder-only Transformer 有什么区别？
15. GPT 为什么使用 Decoder-only + Causal Attention？
16. Positional Encoding / RoPE 为什么必不可少？
17. 一个现代 LLM 的数据流是什么？
18. Pretraining、SFT、LoRA、DPO、PPO、GRPO 分别在训练链中的什么位置？
19. Base Model 和 Instruct / Chat Model 有什么区别？
20. Chat Template 是什么？
21. LLM 推理为什么分成 Prefill 和 Decode？
22. KV Cache 到底缓存了什么？
23. 为什么 Decode 往往受 Memory Bandwidth 影响？
24. Temperature、Top-K、Top-P 如何影响生成？
25. Context Length 为什么会影响内存和速度？
26. GPT-2 风格模型和 Qwen3 这类现代 LLM 在结构上有哪些关键差异？
27. 为什么理解这一章，是后续 `llama.cpp / vLLM / RKLLM / AI Compiler` 的前提？

本章最终要建立一条完整链：

```text
Raw Text
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embedding
   ↓
Transformer Blocks
   ↓
Logits
   ↓
Sampling
   ↓
Next Token
   ↓
Append Token
   ↓
Repeat
```

以及训练链：

```text
Large Text Corpus
      ↓
Next Token Prediction
      ↓
Pretrained Base Model
      ↓
Instruction Data
      ↓
SFT
      ↓
Preference / Alignment
      ↓
Chat / Instruct Model
```

---

# 1. 本章核心 GitHub 仓库

本章同样不只看一个仓库。

四个仓库承担不同职责：

```text
LLMs-from-scratch
│
├── Text → Token
├── Attention
├── GPT
├── Pretraining
├── Instruction Finetuning
├── LoRA
└── KV Cache 等 Bonus
```

```text
MiniMind
│
├── 现代 Decoder-only LLM
├── RMSNorm
├── RoPE
├── GQA
├── SwiGLU / SiLU FFN
├── MoE
├── Pretrain
├── SFT
├── LoRA
├── DPO
├── PPO / GRPO / CISPO
└── Tool / Agentic RL
```

```text
NN Zero to Hero
│
├── Language Modeling
├── Autoregressive Generation
├── GPT from Scratch
└── Tokenizer / BPE
```

```text
D2L
│
├── Attention 理论
├── Bahdanau Attention
├── Multi-Head Attention
└── Transformer 体系化解释
```

推荐方式：

```text
先用 LLMs-from-scratch 建立 GPT 主线
            ↓
用 Zero-to-Hero 再从头手写一次
            ↓
用 D2L 补 Attention 理论
            ↓
用 MiniMind 对齐现代 Qwen 类结构
```

---

# 2. LLMs-from-scratch：本章第一主线

GitHub：

- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

这个仓库非常适合作为本章主教材，因为它的章节本身就是一条完整的 LLM 构建路线：

```text
Chapter 1
Understanding LLMs
        ↓
Chapter 2
Working with Text Data
        ↓
Chapter 3
Coding Attention Mechanisms
        ↓
Chapter 4
Implementing GPT from Scratch
        ↓
Chapter 5
Pretraining on Unlabeled Data
        ↓
Chapter 6
Finetuning for Classification
        ↓
Chapter 7
Finetuning to Follow Instructions
```

仓库主入口：

- [Repository README](https://github.com/rasbt/LLMs-from-scratch)
- [Chapter 2: Working with Text Data](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02)
- [Chapter 3: Coding Attention Mechanisms](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03)
- [Chapter 4: Implementing GPT](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04)
- [Chapter 5: Pretraining](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05)
- [Chapter 7: Instruction Finetuning](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07)

---

# 3. MiniMind：本章第二主线

GitHub：

- [jingyaogong/minimind](https://github.com/jingyaogong/minimind)

MiniMind 的价值在于：

> 它不只是在讲 GPT-2，而是在一个足够小的模型里，把现代开源 LLM 的结构与完整训练链路做出来。

重点入口：

- [MiniMind README](https://github.com/jingyaogong/minimind/blob/master/README.md)
- [model/model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)
- [trainer/train_pretrain.py](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py)
- [trainer/train_full_sft.py](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py)
- [trainer](https://github.com/jingyaogong/minimind/tree/master/trainer)

当前 MiniMind-3 主线结构对齐现代 Qwen3 / Qwen3-MoE 生态思想，因此非常适合从：

```text
教学 GPT
```

过渡到：

```text
现代 LLM
```

---

# 4. Zero to Hero：GPT 与 Tokenizer

GitHub：

- [karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)

本章重点：

### Lecture 2

Language Modeling：

- [Course README](https://github.com/karpathy/nn-zero-to-hero/blob/master/README.md)

### Lecture 7

Let's build GPT：

- [NN Zero to Hero README - GPT](https://github.com/karpathy/nn-zero-to-hero/blob/master/README.md)

### Lecture 8

Let's build the GPT Tokenizer：

- [Tokenizer Lecture Entry](https://github.com/karpathy/nn-zero-to-hero/blob/master/README.md)
- [karpathy/minbpe](https://github.com/karpathy/minbpe)

这一组材料的最大价值：

> 把“LLM 是下一个 Token 预测器”从一句话，真正变成可以跑起来的代码。

---

# 5. D2L：Attention 与 Transformer 理论补充

GitHub：

- [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en)

重点章节：

- [Attention Mechanisms and Transformers](https://github.com/d2l-ai/d2l-en/tree/master/chapter_attention-mechanisms-and-transformers)

D2L 更适合作为：

```text
Attention 理论
+
Transformer 结构
+
数学解释
```

的补充，而不是本章唯一主线。

---

# 6. 什么是 Language Model？

Language Model：

> 对一个 Token 序列的概率分布进行建模。

给定：

```text
今天天气很
```

模型需要预测：

```text
好
冷
热
不错
...
```

每个候选 Token 都有一个概率。

例如：

```text
好      0.31
冷      0.19
热      0.12
不错    0.08
...
```

Language Model 的核心任务可以写成：

```text
P(x_1, x_2, ..., x_T)
```

使用链式分解：

```text
P(x_1, ..., x_T)
=
P(x_1)
P(x_2 | x_1)
P(x_3 | x_1, x_2)
...
P(x_T | x_<T)
```

所以：

> **自回归 Language Model 就是在不断学习“根据前面的 Token，预测下一个 Token”。**

---

# 7. GPT 本质上是什么？

GPT：

```text
Generative
Pre-trained
Transformer
```

三个词分别代表：

```text
Generative
→
能够生成新的 Token 序列

Pre-trained
→
先在大规模文本上做预训练

Transformer
→
模型主体采用 Transformer 架构
```

核心任务：

```text
Context Tokens
      ↓
GPT
      ↓
Probability of Next Token
```

所以可以把 GPT 先理解成：

> 一个巨大的、基于 Transformer 的 Next Token Predictor。

---

# 8. 从 Character-level Language Model 开始最容易理解

例如训练文本：

```text
hello
```

字符级 Token：

```text
h e l l o
```

训练样本可以形成：

```text
h       → e

h e     → l

h e l   → l

h e l l → o
```

或者更统一地：

```text
Input:
h e l l

Target:
e l l o
```

也就是：

> Target 是 Input 整体向右移动一位。

---

# 9. LLMs-from-scratch 中的数据构造

推荐直接阅读：

- [GPTDatasetV1 / gpt.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py)

它的基本思路：

```text
Token IDs:

[t0, t1, t2, t3, t4]
```

生成：

```text
Input:
[t0, t1, t2, t3]

Target:
[t1, t2, t3, t4]
```

这就是：

> Next Token Prediction Training Data。

---

# 10. 为什么一个序列能产生多个监督信号？

假设：

```text
Input:

I love machine learning
```

实际上模型在一次 Forward 中同时学习：

```text
I
→
love

I love
→
machine

I love machine
→
learning
```

所以一个长度：

```text
T
```

的序列，可以提供大约：

```text
T
```

个 token-level prediction targets。

这也是自监督预训练的重要基础。

---

# 11. Self-Supervised Learning

LLM 预训练通常不需要人工逐句标注：

```text
问题 → 正确答案
```

因为文本本身就可以产生 Target。

例如：

```text
原文：
The cat sits on the mat.
```

自动构造：

```text
Input:
The cat sits on the

Target:
cat sits on the mat
```

因此：

> 文本自身就是监督信号。

这叫：

```text
Self-Supervised Learning
```

---

# 12. Tokenizer：LLM 与文本世界之间的接口

计算机不能直接处理：

```text
“你好”
```

神经网络真正处理的是：

```text
Integer IDs
```

所以：

```text
Text
  ↓
Tokenizer
  ↓
Token
  ↓
Token ID
```

例如：

```text
"Hello world"
```

可能变成：

```text
["Hello", " world"]
```

然后：

```text
[15496, 995]
```

具体结果取决于 Tokenizer。

---

# 13. Token 不等于 Word

Token 可以是：

```text
一个字
一个字符
一个单词
单词的一部分
空格 + 单词
Byte 序列
特殊符号
```

所以：

```text
Token
≠
Word
```

这是后面理解：

```text
Context Length
Tokens/s
Tokenizer Compatibility
```

的基础。

---

# 14. Vocabulary

Tokenizer 有一个：

```text
Vocabulary
```

也就是：

```text
Token ↔ Token ID
```

的映射集合。

例如：

```text
0 → <pad>
1 → <bos>
2 → <eos>
...
```

Vocabulary Size：

```text
V
```

后面会直接决定：

```text
Embedding Matrix
```

和：

```text
LM Head
```

的维度。

---

# 15. 为什么 Vocabulary Size 会影响模型参数量？

Embedding：

```text
[V, D]
```

LM Head：

```text
[D, V]
```

如果：

```text
V = 150000
D = 4096
```

这两个矩阵都会很大。

因此：

> Tokenizer 并不是无关紧要的文本预处理工具，它直接影响模型大小、训练效率和推理行为。

MiniMind README 也特别强调：

> 对小模型而言，Vocabulary Size 对 Embedding / Output Layer 参数占比尤其敏感。

参考：

- [MiniMind Tokenizer 说明](https://github.com/jingyaogong/minimind/blob/master/README.md)

---

# 16. Character-level Tokenizer

最简单：

```text
a
b
c
...
你
好
...
```

每个字符一个 Token。

优点：

```text
实现简单
几乎没有 OOV
```

缺点：

```text
Sequence 太长
语义粒度太细
```

因此现代 LLM 通常不直接使用纯字符级 Tokenizer。

---

# 17. Word-level Tokenizer

可以直接：

```text
machine
learning
is
great
```

作为 Token。

问题：

```text
Vocabulary 极大
新词处理困难
拼写变化多
多语言更复杂
```

所以现代模型通常使用：

> Subword Tokenization。

---

# 18. BPE：Byte Pair Encoding

Karpathy 的：

- [minbpe](https://github.com/karpathy/minbpe)

非常适合把 BPE 从头做一遍。

核心思想：

```text
先从较小单位开始
      ↓
统计最频繁的相邻 Pair
      ↓
Merge
      ↓
重复
```

例如：

```text
l o w
l o w e r
```

如果：

```text
l + o
```

经常出现：

```text
lo
```

就可能被合并成新 Token。

不断 Merge：

```text
字符 / Byte
    ↓
常见 Subword
    ↓
更紧凑 Sequence
```

---

# 19. Tokenizer 的两个基本操作

任何 Tokenizer 最核心：

```text
encode()
```

以及：

```text
decode()
```

即：

```text
Text
 ↓ encode
Token IDs
 ↓ decode
Text
```

---

# 20. Special Tokens

常见：

```text
<BOS>
<EOS>
<PAD>
<UNK>
```

现代 Chat / Tool 模型还有：

```text
<|system|>
<|user|>
<|assistant|>
<think>
<tool_call>
<tool_response>
```

具体 Token 名称随模型不同。

MiniMind 当前 Tokenizer 也加入：

```text
Tool Calling
Thinking
```

相关模板标记。

---

# 21. Tokenizer 和 Model 必须匹配

这是极其重要的一点。

模型 Weight 是在特定：

```text
Vocabulary
Token ID Mapping
Tokenizer Rules
```

上训练出来的。

如果换一个 Tokenizer：

```text
Token ID 1234
```

可能代表完全不同的文本片段。

所以：

> 不能随便给一个训练好的 LLM 换 Tokenizer。

---

# 22. Embedding：从 ID 到向量

Token ID 只是整数：

```text
123
456
789
```

它本身没有连续数学意义。

所以需要：

```text
Embedding Layer
```

将：

```text
Token ID
```

映射成：

```text
D-dimensional Vector
```

例如：

```text
Token 123
   ↓
[0.12, -0.45, ..., 0.88]
```

---

# 23. Embedding Matrix

假设：

```text
Vocabulary Size = V
Embedding Dim = D
```

Embedding Matrix：

```text
[V, D]
```

每个 Token ID：

```text
i
```

相当于：

> 取 Embedding Matrix 的第 i 行。

PyTorch：

```python
embedding = nn.Embedding(vocab_size, emb_dim)
```

---

# 24. Embedding 也是 Parameter

Embedding 不是固定词典。

它里面的向量：

```text
也是模型参数
```

会随着 Training：

```text
Backprop
```

不断更新。

最后不同 Token 的向量会形成某种：

```text
语义 / 统计结构
```

---

# 25. Position：为什么 Token 顺序不能丢？

Self-Attention 本身如果没有位置信息，会很难区分：

```text
狗咬人
```

和：

```text
人咬狗
```

因为 Token 集合可能一样。

所以需要加入：

```text
Position Information
```

---

# 26. Absolute Positional Embedding

早期 GPT 风格：

```text
Token Embedding
+
Position Embedding
```

例如：

```text
position 0
position 1
position 2
...
```

都有一个可学习向量。

然后：

```text
x
=
token_embedding
+
position_embedding
```

---

# 27. 为什么现代 LLM 经常使用 RoPE？

现代 Decoder-only LLM 中常见：

```text
RoPE
Rotary Positional Embedding
```

它不是简单把位置向量加到 Token Embedding。

而是：

> 在 Q / K 的向量空间中加入与位置相关的旋转。

现代模型例如 MiniMind 的核心代码：

- [model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

其中可以直接看到：

```text
precompute_freqs_cis
apply_rotary_pos_emb
```

---

# 28. RoPE 需要先理解什么？

本章先理解三点：

1. Self-Attention 自身缺少顺序信息。
2. RoPE 将 Position 注入 Q/K。
3. Attention Score 因此能够感知相对位置关系。

不用在第一遍就手推所有复数旋转公式。

---

# 29. 从 RNN 到 Attention

上一章 RNN：

```text
x1
 ↓
h1
 ↓
h2
 ↓
h3
 ↓
h4
```

存在：

```text
Sequential Dependency
```

以及：

```text
Long-range information bottleneck
```

Attention 的新思路：

> 当前位置不要只依赖一个压缩后的 Hidden State，而是直接选择性地读取其他位置的信息。

---

# 30. Attention 的直觉

一句话：

```text
The animal didn't cross the street because it was too tired.
```

当模型处理：

```text
it
```

时：

```text
it
```

应该重点关注：

```text
animal
```

而不是所有 Token 等权重处理。

Attention 就是在计算：

> 当前 Token 应该“关注”哪些 Token，以及关注多少。

---

# 31. Query、Key、Value

最容易类比数据库检索。

对于每个 Token：

```text
Query
=
我想找什么？

Key
=
我这里有什么信息？

Value
=
真正要读取的信息
```

Attention：

```text
Query
  ↓
与所有 Key 比较
  ↓
得到 Similarity / Score
  ↓
Softmax
  ↓
Attention Weight
  ↓
对 Value 做加权求和
```

---

# 32. Q、K、V 从哪里来？

假设输入：

```text
X
[B, T, D]
```

通过三个不同 Linear Projection：

```text
Q = X W_Q

K = X W_K

V = X W_V
```

所以：

> Q、K、V 不是三份不同输入，而是同一个 Hidden State 经过不同 Weight Matrix 得到的三种表示。

---

# 33. Self-Attention

如果：

```text
Q
K
V
```

都来自同一个 Sequence：

```text
X
```

就叫：

```text
Self-Attention
```

因为：

> Sequence 在“看自己”。

---

# 34. Scaled Dot-Product Attention

核心公式：

```text
Attention(Q, K, V)
=
softmax(QK^T / sqrt(d_k)) V
```

必须做到：

> 不只是会背公式，而是知道每一步 Tensor Shape。

---

# 35. Attention Shape 推导

设：

```text
B = Batch
T = Sequence Length
D = Hidden Dimension
d = Head Dimension
```

那么：

```text
Q:
[B, T, d]

K:
[B, T, d]

V:
[B, T, d]
```

计算：

```text
Q @ K^T
```

得到：

```text
[B, T, T]
```

这个：

```text
T × T
```

矩阵就是：

> 每个 Token 对每个 Token 的 Attention Score。

---

# 36. 为什么是 T × T？

因为：

```text
第 1 个 Token
```

要和：

```text
1...T
```

所有 Key 比较。

第 2 个也是。

所以：

```text
T Queries
×
T Keys
=
T² Scores
```

这直接带来：

> 标准 Self-Attention 在 Sequence Length 上的二次复杂度。

---

# 37. Attention 为什么长上下文昂贵？

因为 Score Matrix：

```text
[T, T]
```

当：

```text
T
```

翻倍：

```text
T²
```

大约变 4 倍。

这就是为什么：

```text
Long Context
```

在 Attention 中非常昂贵。

后面会看到：

```text
FlashAttention
Sliding Window Attention
Sparse Attention
```

等方案。

---

# 38. Dot Product 在衡量什么？

```text
q · k
```

可以粗略理解为：

> Query 与 Key 的匹配程度。

值越大：

```text
更相关
```

值越小：

```text
更不相关
```

然后 Softmax 将这些 Score 变成：

```text
Attention Weight
```

---

# 39. 为什么除以 sqrt(d_k)？

如果：

```text
d_k
```

很大，Dot Product 的数值尺度可能变大。

Softmax 输入过大时：

```text
分布过于尖锐
```

梯度也容易不稳定。

所以：

```text
QK^T
```

除以：

```text
sqrt(d_k)
```

用于控制尺度。

---

# 40. Softmax

Attention Score：

```text
[2.1, 0.3, -1.2, ...]
```

经过 Softmax：

```text
[0.72, 0.12, 0.03, ...]
```

满足：

```text
每个值 >= 0

sum = 1
```

所以可以把它解释成：

```text
Attention Distribution
```

---

# 41. Weighted Sum of Values

得到 Attention Weight 后：

```text
Attention Weight
        ×
Value
```

最后加权求和：

```text
Context Vector
```

于是：

> 当前 Token 的新表示，融合了它所关注位置的信息。

---

# 42. Attention 的完整数据流

```text
X
│
├── Linear → Q
├── Linear → K
└── Linear → V

Q @ K^T
    ↓
Scale
    ↓
Mask
    ↓
Softmax
    ↓
Attention Weights
    ↓
@ V
    ↓
Context
```

这条链要能自己画出来。

---

# 43. Causal Attention

GPT 是：

```text
Autoregressive Language Model
```

预测：

```text
Token t
```

时不能偷看：

```text
Token t+1
Token t+2
...
```

所以需要：

```text
Causal Mask
```

---

# 44. Causal Mask 长什么样？

例如 Sequence Length = 4：

```text
Token 1:
可以看 1

Token 2:
可以看 1,2

Token 3:
可以看 1,2,3

Token 4:
可以看 1,2,3,4
```

Mask：

```text
[ 0   -∞  -∞  -∞
  0    0  -∞  -∞
  0    0   0  -∞
  0    0   0   0 ]
```

加到 Attention Scores 后：

```text
Softmax(-∞)
≈
0
```

所以未来 Token 的权重为 0。

---

# 45. 为什么 Training 时可以一次算完整 Sequence？

虽然 GPT 是一个一个 Token 生成的。

Training 时已经知道整段文本：

```text
[t0,t1,t2,t3,t4]
```

借助 Causal Mask：

```text
每个位置仍然只能看过去
```

但整个：

```text
T positions
```

可以并行计算。

所以：

> GPT Training 可以对 Sequence Dimension 高度并行，而 RNN 通常需要逐时间步依赖。

---

# 46. Multi-Head Attention

单头：

```text
1 个 Q/K/V 子空间
```

多头：

```text
Head 1
Head 2
Head 3
...
Head H
```

每个 Head 可以学习不同关系：

```text
局部依赖
语法关系
长距离依赖
实体关联
...
```

然后：

```text
Concat
 ↓
Output Projection
```

---

# 47. Multi-Head Shape

设：

```text
D = 768
H = 12 heads
```

那么：

```text
head_dim
=
768 / 12
=
64
```

输入：

```text
[B,T,768]
```

reshape：

```text
[B,T,12,64]
```

通常 transpose：

```text
[B,12,T,64]
```

之后每个 Head 独立 Attention。

---

# 48. MHA 的本质不是运行 12 个完全独立模型

工程上通常：

```text
一次大的 Linear
```

同时得到：

```text
Q for all heads
```

再：

```text
reshape
```

成 Head 维度。

所以：

> Multi-Head 是张量布局上的分组与并行，不是 Python for-loop 跑 12 次模型。

---

# 49. GQA：Grouped Query Attention

现代 LLM 中常见：

```text
GQA
```

传统 MHA：

```text
num_q_heads
=
num_kv_heads
```

GQA：

```text
num_q_heads
>
num_kv_heads
```

例如 MiniMind 默认：

```text
num_attention_heads = 8
num_key_value_heads = 4
```

参考：

- [MiniMind model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

---

# 50. GQA 为什么重要？

Decode 时：

```text
K
V
```

需要缓存。

如果减少 KV Heads：

```text
KV Cache
```

就会明显减小。

所以 GQA 是：

> 模型能力、推理速度与 KV Cache 内存之间的一种折中。

这个概念会直接进入后面的：

```text
llama.cpp
vLLM
RKLLM
```

---

# 51. MQA

更激进：

```text
所有 Query Heads
共享 1 组 K/V
```

叫：

```text
Multi-Query Attention
MQA
```

关系：

```text
MHA
num_q = num_kv

GQA
num_q > num_kv > 1

MQA
num_kv = 1
```

---

# 52. Transformer Block

一个典型 Decoder-only Transformer Block：

```text
Input
  │
  ├───────────── Residual ─────────────┐
  ↓                                    │
Norm                                   │
  ↓                                    │
Self-Attention                         │
  ↓                                    │
Add ←──────────────────────────────────┘
  │
  ├───────────── Residual ─────────────┐
  ↓                                    │
Norm                                   │
  ↓                                    │
FFN / MLP                              │
  ↓                                    │
Add ←──────────────────────────────────┘
  ↓
Output
```

现代模型通常是：

```text
Pre-Norm
```

结构。

---

# 53. Residual Connection

形式：

```text
y = x + F(x)
```

作用之一：

> 给深层网络提供更直接的信息和 Gradient 路径。

没有 Residual：

```text
x
↓
F1
↓
F2
↓
F3
...
```

有 Residual：

```text
x
├────────→ +
└→ F(x) ──┘
```

这对训练非常深的 Transformer 至关重要。

---

# 54. LayerNorm

Transformer 中常见：

```text
LayerNorm
```

它通常沿：

```text
Hidden Dimension
```

对单个 Token 的向量做归一化。

与 BatchNorm 不同：

> LayerNorm 不依赖整个 Batch 的统计量。

所以更适合：

```text
Sequence Model
```

---

# 55. RMSNorm

现代 LLM 常用：

```text
RMSNorm
```

MiniMind 直接实现：

- [RMSNorm source](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

核心思想：

```text
x
↓
根据 RMS 归一化
↓
乘可学习 Weight
```

相比 LayerNorm：

```text
不做 Mean Centering
```

结构更简单。

---

# 56. FFN 是 Transformer 中另一大计算块

Attention 后通常还有：

```text
Feed Forward Network
FFN
```

经典形式：

```text
D
 ↓
4D
 ↓
Activation
 ↓
D
```

例如：

```text
768
 ↓
3072
 ↓
GELU
 ↓
768
```

---

# 57. 为什么 FFN 很重要？

Transformer 不只是 Attention。

一个 Block 主要由：

```text
Attention
+
FFN
```

组成。

Attention：

```text
Token 之间交换信息
```

FFN：

```text
对每个 Token 的 Channel / Feature 做非线性变换
```

两者作用不同。

---

# 58. 现代 LLM 中的 SwiGLU 风格 FFN

MiniMind 中可以看到：

```text
gate_proj
up_proj
down_proj
SiLU
```

源代码：

- [MiniMind FeedForward](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

结构大致：

```text
x
├→ gate_proj → SiLU ─┐
│                    ×
└→ up_proj ──────────┘
         ↓
     down_proj
```

这是现代 LLaMA / Qwen 类模型中很典型的 FFN 设计。

---

# 59. Transformer Block 的核心 Operator

一个现代 Decoder Block 可以大致拆成：

```text
RMSNorm
Linear(Q)
Linear(K)
Linear(V)
RoPE
MatMul(Q,K)
Softmax
MatMul(Attn,V)
Linear(O)
Add

RMSNorm
Linear(Gate)
SiLU
Linear(Up)
Mul
Linear(Down)
Add
```

后面进入：

```text
AI Compiler
NPU
GPU Kernel
```

看到的就是这些 Operator。

---

# 60. Transformer Architecture 的三种主要类型

---

## Encoder-only

典型：

```text
BERT
```

特点：

```text
双向 Attention
```

适合：

```text
Understanding
Classification
Embedding
```

---

## Encoder-Decoder

典型：

```text
Original Transformer
T5
```

结构：

```text
Encoder
+
Decoder
```

适合：

```text
Translation
Sequence-to-Sequence
```

---

## Decoder-only

典型：

```text
GPT
LLaMA
Qwen
Mistral
DeepSeek
```

特点：

```text
Causal Self-Attention
Autoregressive Generation
```

目前通用 LLM 最重要的一条路线。

---

# 61. GPT 为什么使用 Decoder-only？

因为任务本身：

```text
根据过去 Token
预测下一个 Token
```

天然符合：

```text
Causal Decoder
```

所以不需要单独 Encoder。

---

# 62. GPT Model 的完整结构

经典 GPT：

```text
Token IDs
    ↓
Token Embedding
    +
Position Embedding
    ↓
Dropout
    ↓
Transformer Block × N
    ↓
Final Norm
    ↓
LM Head
    ↓
Logits
```

LLMs-from-scratch Chapter 4：

- [Chapter 4](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04)
- [gpt.py](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py)

---

# 63. LM Head

最后 Hidden State：

```text
[B,T,D]
```

需要变成：

```text
[B,T,V]
```

其中：

```text
V = Vocabulary Size
```

于是：

```text
Linear(D → V)
```

这就是：

```text
LM Head
```

输出：

```text
Logits for every vocabulary token
```

---

# 64. Logits

Logits：

> Softmax 之前的原始分数。

例如：

```text
Vocabulary:
50000 tokens
```

最后一个位置：

```text
[50000]
```

每个值代表一个 Token 的 raw score。

---

# 65. Training Loss

训练时：

```text
Logits
[B,T,V]
```

Target：

```text
[B,T]
```

然后：

```text
CrossEntropyLoss
```

计算：

> 每个位置对正确 Next Token 的预测误差。

---

# 66. 为什么最后不需要手动 Softmax 再 CrossEntropy？

PyTorch 的：

```python
nn.CrossEntropyLoss()
```

通常直接接：

```text
Logits
```

内部会以数值更稳定的方式完成相关计算。

所以：

```text
Training:
Logits → CrossEntropy
```

不应先手动：

```text
Softmax
```

再传进去。

---

# 67. Weight Tying

现代 LLM 常见：

```text
Input Embedding Weight
```

和：

```text
Output LM Head Weight
```

共享。

MiniMind 默认：

```text
tie_word_embeddings = True
```

参考：

- [MiniMind model source](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

这样可以：

```text
减少参数
```

并利用输入 / 输出 Token Space 的相关性。

---

# 68. 什么是 Pretraining？

Pretraining：

> 在大规模通用文本上训练 Next Token Prediction。

数据：

```text
Web
Books
Code
Papers
Wiki
...
```

处理：

```text
Clean
Deduplicate
Filter
Tokenize
```

然后：

```text
Token Sequences
       ↓
Next Token Prediction
       ↓
Base Model
```

---

# 69. Base Model

Pretraining 之后的模型：

```text
Base Model
```

主要学到：

```text
语言结构
统计规律
世界知识
代码模式
推理模式的一部分
```

但它未必天然擅长：

```text
“用户问一句
助手规整回答一句”
```

所以还需要：

```text
Instruction Tuning
```

---

# 70. LLMs-from-scratch Pretraining

核心：

- [Chapter 5](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05)

它会将：

```text
GPT architecture
+
Text Dataset
+
Next Token Loss
+
Optimizer
```

组合成完整 Training Loop。

本章建议：

> 至少自己完整跑一次小 GPT 的 Pretraining。

不是为了训练有用模型。

而是为了彻底理解：

```text
Data
→
Tokens
→
Batch
→
Logits
→
Loss
→
Backward
→
Update
```

---

# 71. Perplexity

语言模型常见评价：

```text
Perplexity
PPL
```

它和 Cross Entropy 紧密相关。

直觉：

> 模型面对下一个 Token 有多“困惑”。

越低：

```text
通常越好
```

但：

> 不同 Tokenizer 下的 PPL 不适合简单横向比较。

因为 Token 切分粒度不同。

---

# 72. 为什么数据比模型结构同样重要？

如果数据：

```text
重复
低质量
错误
垃圾文本
污染评测集
```

模型也会学到这些问题。

所以现代 LLM Pipeline：

```text
Raw Data
 ↓
Cleaning
 ↓
Deduplication
 ↓
Quality Filtering
 ↓
Tokenization
 ↓
Training
```

数据工程是核心组成部分。

---

# 73. Scaling：为什么模型会越来越大？

能力一般同时依赖：

```text
Model Size
Data Size
Compute
```

这和第一章学到的 AI 三要素一致：

```text
Algorithm
Data
Compute
```

LLM 只是把这三个量推到了极大规模。

---

# 74. Parameter Count

例如：

```text
2B
7B
32B
70B
```

B：

```text
Billion Parameters
```

参数主要来自：

```text
Embedding
Attention Linear
FFN Linear
LM Head
```

其中大模型中：

> Transformer Blocks 往往占主要参数。

---

# 75. 为什么 FFN 参数很多？

假设：

```text
Hidden D = 4096
Intermediate = 11008
```

几个巨大的 Linear：

```text
4096 × 11008
```

参数非常多。

所以：

> LLM 中不只是 Attention 很贵，FFN / MLP 同样是非常重要的计算与参数来源。

---

# 76. MoE：Mixture of Experts

现代部分模型使用：

```text
MoE
```

核心：

> 不让每个 Token 都经过同一个 FFN，而是先 Router，再只激活少数 Expert。

```text
Token
 ↓
Router
 ↓
Top-K Experts
 ↓
Expert FFN
 ↓
Weighted Combine
```

MiniMind 也包含：

```text
Dense
+
MoE
```

实现：

- [MiniMind MOEFeedForward](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

---

# 77. MoE 的核心价值

可以拥有：

```text
很大的总 Parameter Count
```

但每个 Token 只激活：

```text
部分 Parameters
```

所以：

```text
Total Parameters
≠
Active Parameters per Token
```

这也是理解：

```text
DeepSeek MoE
Qwen MoE
```

等模型规格的基础。

---

# 78. MoE 也带来新问题

例如：

```text
Load Balance
Expert Routing
Communication
Memory
Deployment Complexity
```

所以 MoE：

> 不是“免费变大”。

尤其端侧部署中，MoE 的：

```text
Weight Memory
```

仍然可能非常大。

---

# 79. SFT：Supervised Fine-Tuning

Pretrained Base Model：

```text
擅长续写
```

我们希望：

```text
听懂 Instruction
按照 Assistant 风格回答
```

于是准备：

```text
Instruction
Input
Response
```

数据进行监督训练。

这叫：

```text
SFT
```

---

# 80. LLMs-from-scratch 的 Instruction Finetuning

重点：

- [Chapter 7](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07)
- [Instruction Finetuning Code](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/gpt_instruction_finetuning.py)

它非常适合看：

```text
Instruction Data
 ↓
Formatting
 ↓
Tokenization
 ↓
Dataset
 ↓
Collate
 ↓
Labels / Ignore Index
 ↓
Training
```

---

# 81. 为什么 SFT 仍然是 Next Token Prediction？

假设：

```text
User:
What is 2+2?

Assistant:
4
```

格式化后：

```text
<System>...
<User>What is 2+2?
<Assistant>4
```

模型仍然学习：

```text
Token t
→
Token t+1
```

区别主要在：

```text
Training Data Distribution
```

变成了：

> Instruction / Response 格式。

---

# 82. Chat Template

不同模型对对话有固定序列格式：

```text
System
User
Assistant
```

以及特殊 Token。

这叫：

```text
Chat Template
```

例如概念上：

```text
<BOS>
<system>
You are helpful.
</system>
<user>
你好
</user>
<assistant>
```

然后模型开始生成。

---

# 83. 为什么 Chat Template 不能随便改？

SFT 时模型是在特定：

```text
Token Pattern
```

上训练的。

如果推理时格式完全不同：

```text
模型可能无法正确判断角色边界
```

所以：

> Tokenizer + Chat Template + Model Weight 是一个整体。

这对后面部署 Qwen：

```text
llama.cpp
RKLLM
```

极其重要。

---

# 84. Base Model vs Instruct Model

Base：

```text
Prompt:
中国的首都是

可能：
北京。北京位于...
```

它更像：

```text
Text Completion
```

Instruct：

```text
User:
中国首都是哪里？

Assistant:
北京。
```

它经过：

```text
SFT / Alignment
```

更符合人机对话行为。

---

# 85. Full Fine-Tuning

Full SFT：

```text
所有 Model Parameters
```

都参与更新。

优点：

```text
能力上限高
```

缺点：

```text
显存高
计算成本高
保存完整 Weight
```

MiniMind：

- [train_full_sft.py](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py)

---

# 86. LoRA

LoRA：

```text
Low-Rank Adaptation
```

核心思想：

> 冻结大部分原始 Weight，只训练小规模低秩增量矩阵。

假设：

```text
W
```

不直接更新。

而：

```text
W'
=
W + ΔW
```

其中：

```text
ΔW
=
B A
```

且 Rank：

```text
r << D
```

---

# 87. 为什么 LoRA 省资源？

Full Fine-Tuning：

```text
更新几十亿 Parameter
```

LoRA：

```text
只更新很少的 Adapter Parameter
```

所以：

```text
Gradient Memory ↓
Optimizer State ↓
Checkpoint Size ↓
```

非常适合个人 GPU。

---

# 88. LLMs-from-scratch LoRA

仓库有专门的：

```text
Appendix E
Parameter-efficient Finetuning with LoRA
```

入口：

- [LLMs-from-scratch Repository](https://github.com/rasbt/LLMs-from-scratch)

建议在 SFT 之后理解 LoRA。

---

# 89. QLoRA 是什么？

在 LoRA 基础上：

> Base Model 使用低比特量化存储，再训练 LoRA Adapter。

概念：

```text
Quantized Base Weight
+
Trainable LoRA
```

可以进一步降低显存。

注意：

> QLoRA 的“量化训练”与后面 llama.cpp 的纯推理量化，目标和技术细节并不完全一样。

---

# 90. Preference Alignment

SFT 后：

```text
模型会回答
```

但未必：

```text
回答偏好最好
安全性最好
格式最好
帮助性最好
```

所以还可能使用：

```text
Preference Data
```

例如：

```text
Prompt
+
Chosen Response
+
Rejected Response
```

进行 Alignment。

---

# 91. RLHF

经典：

```text
Human Preference Data
      ↓
Reward Model
      ↓
Reinforcement Learning
      ↓
Aligned Model
```

常见算法历史上包括：

```text
PPO
```

但实际现代训练流程已经有很多不同方案。

---

# 92. DPO

Direct Preference Optimization：

> 不显式训练独立 Reward Model，也可以直接利用 Preference Pair 优化策略。

LLMs-from-scratch Bonus：

- [DPO from scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/04_preference-tuning-with-dpo)

MiniMind 也包含 DPO 流程。

---

# 93. PPO / GRPO / CISPO

MiniMind 当前还包含：

```text
PPO
GRPO
CISPO
```

以及 Agentic RL。

本章不要求把这些强化学习算法全部推导。

只需要建立训练阶段位置：

```text
Pretrain
 ↓
SFT
 ↓
Preference / RLAIF / RL
```

---

# 94. 训练路线应该怎么理解？

不是所有模型都严格：

```text
Pretrain → SFT → RLHF
```

现代 LLM 训练 Pipeline 会变化。

学习阶段先建立：

```text
Pretraining
=
获得基础语言模型能力

SFT
=
学习指令 / 对话行为

Preference / RL
=
进一步对齐偏好、推理或特定行为
```

即可。

---

# 95. 推理：训练完成以后发生什么？

用户输入：

```text
你好，请介绍一下 Transformer
```

流程：

```text
Text
 ↓
Tokenizer
 ↓
Input IDs
 ↓
Embedding
 ↓
Transformer
 ↓
Logits
 ↓
Sampling
 ↓
Next Token
 ↓
Append
 ↓
Transformer
 ↓
Next Token
 ...
```

这就是：

```text
Autoregressive Generation
```

---

# 96. 为什么生成必须一个 Token 一个 Token？

因为：

```text
Token t+1
```

依赖：

```text
Token <= t
```

而：

```text
Token t+2
```

又依赖刚刚生成的：

```text
Token t+1
```

所以 Generation 是：

```text
Sequential
```

的。

这和 Training 的 Sequence Parallelism 是一个巨大区别。

---

# 97. Training vs Generation

Training：

```text
整段真实 Token 已知
+
Causal Mask
→
一次并行计算 T 个位置
```

Generation：

```text
未来 Token 不知道
→
必须生成 1 个
→
加入 Context
→
再生成下一个
```

这个区别直接导致：

```text
Prefill
vs
Decode
```

---

# 98. Prefill

用户输入 Prompt：

```text
T = 1000 tokens
```

模型先对整段 Prompt 做：

```text
Forward
```

并构建：

```text
KV Cache
```

这阶段：

```text
Prefill
```

通常具有：

```text
大 Matrix Multiplication
较高并行度
```

往往更偏：

```text
Compute-bound
```

具体取决于模型和硬件。

---

# 99. Decode

Prefill 后开始：

```text
一次生成 1 Token
```

每次：

```text
new token
 ↓
Transformer
 ↓
next token
```

这阶段：

```text
Decode
```

Decode 的 Batch / Token Dimension 通常更小。

---

# 100. 为什么 Decode 经常更受内存带宽限制？

每生成一个 Token：

```text
大量 Model Weights
```

仍需要被读取用于计算。

但是本次有效计算规模相对 Prefill 小。

所以经常呈现：

```text
Arithmetic Intensity 较低
```

于是：

> Memory Bandwidth 很可能成为关键瓶颈。

这就是为什么：

```text
INT4
Q4
```

不仅省内存，还可能明显提高 Decode 速度。

---

# 101. KV Cache 是什么？

对于每一层 Attention：

```text
过去 Token 的 K
过去 Token 的 V
```

在后续 Decode 中会重复使用。

如果每次都重新计算过去全部 Token：

```text
非常浪费
```

所以缓存：

```text
K Cache
V Cache
```

这就是：

```text
KV Cache
```

---

# 102. 为什么不缓存 Q？

当前新 Token 的：

```text
Q
```

只在当前 Query 中使用。

下一步生成新 Token：

```text
会有新的 Q
```

而过去 Token 的：

```text
K / V
```

要不断被新 Query 使用。

所以缓存的是：

```text
K
V
```

---

# 103. Without KV Cache

第 1 步：

```text
重新算 Token 1...T
```

第 2 步：

```text
重新算 Token 1...T+1
```

第 3 步：

```text
重新算 Token 1...T+2
```

极其浪费。

---

# 104. With KV Cache

Prefill：

```text
Token 1...T
 ↓
K/V Cache
```

Decode Step：

```text
只对 New Token
计算新的 Q/K/V
```

然后：

```text
new Q
×
all cached K
```

Attention：

```text
all cached V
```

继续生成。

---

# 105. KV Cache 为什么占内存？

粗略依赖：

```text
Layers
×
Sequence Length
×
KV Heads
×
Head Dim
×
2(K+V)
×
Bytes per element
```

所以：

```text
Context 越长
KV Cache 越大
```

---

# 106. GQA 和 KV Cache 再联系一次

传统 MHA：

```text
很多 KV Heads
```

GQA：

```text
减少 KV Heads
```

所以：

```text
KV Cache Size ↓
Memory Bandwidth ↓
```

这就是为什么现代推理模型非常重视 GQA。

---

# 107. Context Length

Context Length：

> 模型一次能够处理的 Token 数量上限或有效窗口。

例如：

```text
4K
8K
32K
128K
...
```

注意：

```text
Token
≠
Character
```

所以：

```text
32K Context
```

并不等于：

```text
32000 个汉字
```

---

# 108. 长上下文的成本

Context 增大：

```text
Attention Work ↑
KV Cache ↑
Memory ↑
Prefill Latency ↑
```

所以：

> Long Context 从来不是“免费加一个配置值”。

---

# 109. RoPE Scaling / YaRN

当模型希望超过原始训练 Context：

```text
Position Encoding
```

也需要处理外推问题。

MiniMind 当前包含：

```text
YaRN
```

相关 RoPE Scaling 配置。

参考：

- [model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

本章只需要知道：

> 长上下文不仅是 KV Cache 问题，也涉及位置编码是否能够外推。

---

# 110. Sampling：模型怎么从 Logits 选 Token？

模型给出：

```text
Logits
```

可以：

```text
Greedy
```

直接取最大值。

也可以：

```text
Sampling
```

按概率随机选择。

---

# 111. Greedy Decoding

```text
argmax(logits)
```

每次都取概率最大 Token。

优点：

```text
确定
简单
```

缺点：

```text
可能重复
表达僵硬
容易陷入局部模式
```

---

# 112. Temperature

通常：

```text
logits / temperature
```

再 Softmax。

Temperature 低：

```text
分布更尖锐
更确定
```

Temperature 高：

```text
分布更平
更随机
```

---

# 113. Top-K

只保留：

```text
概率最高的 K 个 Token
```

其他：

```text
过滤
```

再从 K 个中采样。

---

# 114. Top-P / Nucleus Sampling

不是固定 K 个。

而是从高到低累积概率：

```text
直到总概率 >= p
```

例如：

```text
p = 0.9
```

候选集合大小会动态变化。

---

# 115. Sampling 不是模型能力本身

同一个 Model：

```text
不同 Temperature
不同 Top-P
```

可能生成风格差异很大。

所以评估模型时：

> 必须区分 Model Weight 和 Decoding Strategy。

---

# 116. EOS

生成过程中如果模型输出：

```text
EOS Token
```

通常代表：

```text
生成结束
```

所以：

```text
Tokenizer Special Tokens
```

直接参与推理控制。

---

# 117. BOS

某些模型输入前：

```text
BOS
```

代表：

```text
Beginning of Sequence
```

不同模型是否需要、如何使用不完全相同。

因此：

> 不要把所有模型 Tokenizer 行为想当然地统一处理。

---

# 118. Padding

Batch 中不同 Sequence 长度不同。

可能需要：

```text
PAD Token
```

补齐 Shape。

同时需要：

```text
Attention Mask
```

避免模型关注 Padding。

---

# 119. Causal Mask vs Padding Mask

Causal Mask：

```text
不能看未来
```

Padding Mask：

```text
不能看 PAD
```

两个作用完全不同。

在真实模型中可能组合使用。

---

# 120. Attention Mask

概念上：

```text
Allowed
→ 0

Blocked
→ -∞
```

在 Softmax 后：

```text
Blocked Position
≈ 0 weight
```

---

# 121. GPT-2 风格模型 vs 现代 LLM

建议先通过 `LLMs-from-scratch` 理解 GPT-2 风格：

```text
Learned Position Embedding
LayerNorm
MHA
GELU FFN
Decoder-only
```

再通过 MiniMind 理解现代结构：

```text
RoPE
RMSNorm
GQA
SiLU / SwiGLU-like FFN
KV Cache
MoE optional
Long Context Scaling
```

这两个阶段不要颠倒。

---

# 122. 为什么不要一上来读 Qwen3 完整源码？

因为一个现代 LLM 工程仓库同时会包含：

```text
Tensor Parallel
FlashAttention
Cache
Generation API
Quantization
Distributed Training
MoE Routing
Checkpoint Conversion
Serving
```

初学者很容易把：

```text
Transformer 原理
```

和：

```text
工程框架细节
```

混在一起。

更好的方式：

```text
Tiny GPT
→
Modern Tiny LLM
→
Real Qwen
```

---

# 123. MiniMind Model Source 推荐阅读顺序

文件：

- [model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

不要从第一行到最后一行硬啃。

建议顺序：

```text
1. MiniMindConfig

2. RMSNorm

3. Attention

4. FeedForward

5. MiniMindBlock

6. MiniMindModel

7. MiniMindForCausalLM

8. MOEFeedForward
```

---

# 124. 先看 Config

Config 会告诉你模型的骨架：

```text
hidden_size
num_hidden_layers
vocab_size
num_attention_heads
num_key_value_heads
head_dim
intermediate_size
max_position_embeddings
rope_theta
```

看到一个陌生 LLM：

> 第一件事往往就是先看 Config。

---

# 125. Model Config 和 Weight 是不同的

Config：

```text
描述 Architecture
```

Weight：

```text
描述训练后 Parameter Value
```

Tokenizer：

```text
描述文本 ↔ Token
```

所以一个可用 LLM 通常至少涉及：

```text
Architecture / Config
+
Weights
+
Tokenizer
+
Generation / Chat Config
```

---

# 126. 一个 LLM 文件夹里到底有什么？

在 Hugging Face 风格模型中常见：

```text
config.json

tokenizer.json
tokenizer_config.json

model.safetensors
或多个 shard

generation_config.json

chat_template
```

这些并不是重复文件。

每类负责不同部分。

---

# 127. Safetensors / PyTorch Checkpoint

Weight 可以存：

```text
.pth
.bin
.safetensors
```

这些主要是：

```text
Parameter Tensor
```

的存储格式。

而后面：

```text
GGUF
.rknn
RKLLM model
```

会加入更多部署相关信息或经过编译 / 量化转换。

---

# 128. LLM 推理的 Shape

输入：

```text
input_ids
[B,T]
```

Embedding：

```text
[B,T,D]
```

Attention Q：

```text
[B,H,T,d]
```

Attention Score：

```text
[B,H,T,T]
```

Hidden：

```text
[B,T,D]
```

Logits：

```text
[B,T,V]
```

这是本章必须形成的 Shape 主线。

---

# 129. Decode 时 Shape 有什么变化？

Prefill：

```text
T = prompt length
```

例如：

```text
[B,1024,D]
```

Decode：

```text
new token length = 1
```

当前输入常接近：

```text
[B,1,D]
```

但 KV Cache 中仍有：

```text
past T
```

所以 Attention Query：

```text
1
```

需要读取：

```text
T cached Keys / Values
```

---

# 130. Prefill 与 Decode 的 GEMM 形态不同

Prefill：

```text
M 较大
```

可以形成比较大的矩阵乘。

Decode：

```text
M 往往很小
```

接近：

```text
Matrix × Vector
```

或小 GEMM。

这就是为什么：

> 同一个模型在 Prefill 和 Decode 上，硬件利用率与瓶颈可能完全不同。

后面 `llama.cpp` 文档会继续深入。

---

# 131. 为什么理解 MatMul 对 LLM 很重要？

Attention Linear：

```text
Q = X Wq
K = X Wk
V = X Wv
```

FFN：

```text
X W1
X W2
X W3
```

LM Head：

```text
X W_vocab
```

大量核心运算：

```text
Matrix Multiplication
```

所以：

```text
LLM
↓
MatMul / GEMM
↓
Kernel
↓
Hardware
```

是后半部分学习路线的主轴。

---

# 132. FLOPs 不是唯一性能指标

LLM 推理性能还依赖：

```text
Memory Bandwidth
Memory Capacity
Cache
Tensor Layout
Quantization
Kernel Efficiency
Batch Size
Sequence Length
KV Cache
```

所以：

> 不能只看 NPU TOPS 或 GPU TFLOPS 判断实际 Tokens/s。

---

# 133. 为什么参数量直接影响推理内存？

假设：

```text
2B Parameters
```

FP16：

```text
约 2 Bytes / Parameter
```

仅 Weight：

```text
约 4 GB
```

INT8：

```text
约 2 GB
```

INT4：

```text
理论裸 Weight 约 1 GB
```

实际还要加：

```text
Scale
Metadata
Alignment
KV Cache
Runtime Buffer
```

因此会更大。

---

# 134. Quantization 在这一章只建立位置

本章先理解：

```text
FP32
FP16
BF16
INT8
INT4
```

以及：

```text
降低 Weight Precision
→
减少 Weight Memory
→
减少 Memory Traffic
```

详细：

```text
Q4_K_M
GGUF
Weight-only Quant
Activation Quant
```

留到 Runtime / Quantization 文档。

---

# 135. LLM 与 VLM 的关系

VLM：

```text
Vision Language Model
```

可以粗略理解：

```text
Image
 ↓
Vision Encoder
 ↓
Visual Tokens / Features
 ↓
Projection
 ↓
LLM
 ↓
Text Output
```

所以 LLM 往往仍是多模态模型中的核心语言 Backbone。

本章不展开 Vision Encoder。

---

# 136. LLM 与 Agent 的关系

LLM：

```text
模型
```

Agent：

```text
LLM
+
Tool
+
Memory
+
Workflow
+
Environment
```

所以：

> Agent 不是一种新的 Transformer 模型结构。

而是：

> 建立在 LLM 之上的系统层。

下一阶段会专门讨论。

---

# 137. 一个现代 Chat LLM 的完整链

```text
User Text
   ↓
Chat Template
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Embedding
   ↓
Transformer × N
   │
   ├── RMSNorm
   ├── GQA Attention
   ├── RoPE
   ├── Residual
   ├── RMSNorm
   └── SwiGLU FFN
   ↓
Final RMSNorm
   ↓
LM Head
   ↓
Logits
   ↓
Sampling
   ↓
Next Token
   ↓
KV Cache Update
   ↓
Repeat
   ↓
Decode Text
```

这张图必须最终可以不看笔记自己画出来。

---

# 138. 一个 LLM 的训练阶段图

```text
Raw Text
   ↓
Cleaning / Dedup
   ↓
Tokenizer
   ↓
Token Dataset
   ↓
Pretraining
   ↓
Base Model
   ↓
Instruction Dataset
   ↓
SFT
   ↓
Instruct Model
   ↓
Preference / RL Data
   ↓
DPO / PPO / GRPO ...
   ↓
Aligned Chat Model
```

---

# 139. 本章建议实际学习顺序

---

## Stage 1：Language Model

使用：

- [Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
- [LLMs-from-scratch Chapter 2](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02)

理解：

```text
Token Sequence
Next Token Prediction
Autoregressive
Input / Target Shift
```

目标：

> 能自己写一个 Character-level Bigram Language Model。

---

## Stage 2：Tokenizer

学习：

- [karpathy/minbpe](https://github.com/karpathy/minbpe)
- [LLMs-from-scratch ch02](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02)

理解：

```text
Vocabulary
Encode
Decode
BPE
Special Tokens
```

目标：

> 自己实现一个极简 BPE，至少跑通 merge → encode → decode。

---

## Stage 3：Attention

学习：

- [LLMs-from-scratch Chapter 3](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03)
- [D2L Attention](https://github.com/d2l-ai/d2l-en/tree/master/chapter_attention-mechanisms-and-transformers)

重点：

```text
Q
K
V
Scaled Dot Product
Softmax
Causal Mask
Multi-Head
```

目标：

> 手写一个 Self-Attention Layer。

---

## Stage 4：GPT

学习：

- [LLMs-from-scratch Chapter 4](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04)
- [Karpathy GPT Lecture](https://github.com/karpathy/nn-zero-to-hero/blob/master/README.md)

自己搭：

```text
Embedding
Position
Attention
FFN
Norm
Residual
LM Head
```

目标：

> 自己实现一个 Tiny GPT。

---

## Stage 5：Pretraining

学习：

- [LLMs-from-scratch Chapter 5](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05)

跑：

```text
Small Corpus
→
GPT Training
→
Loss Curve
→
Text Generation
```

目标：

> 完整经历一次 LLM Pretraining。

---

## Stage 6：Instruction Finetuning

学习：

- [LLMs-from-scratch Chapter 7](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07)

理解：

```text
Instruction Data
Formatting
Chat Style
SFT
Evaluation
```

目标：

> 把 Base Model 变成一个最简单的 Instruction Model。

---

## Stage 7：LoRA

理解：

```text
Freeze Base
+
Low-rank Adapter
```

跑一次 LoRA Fine-Tuning。

---

## Stage 8：现代 LLM 结构

开始读：

- [MiniMind model_minimind.py](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)

按顺序理解：

```text
RMSNorm
RoPE
GQA
SwiGLU-style FFN
KV Cache
MoE
```

目标：

> 从 GPT-2 思维迁移到 Qwen / LLaMA 类现代 LLM。

---

## Stage 9：完整 MiniMind Pipeline

重点：

- [train_pretrain.py](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py)
- [train_full_sft.py](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py)
- [MiniMind README](https://github.com/jingyaogong/minimind/blob/master/README.md)

跑：

```text
Pretrain
→
SFT
→
Inference
```

有资源再尝试：

```text
LoRA
DPO
GRPO
```

---

# 140. 本章必须完成的实验 1：Bigram Language Model

输入：

```text
tiny text corpus
```

实现：

```text
Vocabulary
Encode
Embedding / Count Table
Next Token Probability
Sampling
```

目标：

> 真正理解 Language Modeling。

---

# 141. 实验 2：BPE Tokenizer

自己实现或跟着：

- [karpathy/minbpe](https://github.com/karpathy/minbpe)

完成：

```text
Text
↓
Bytes / Base Tokens
↓
Pair Statistics
↓
Merge Rules
↓
Vocabulary
↓
Encode / Decode
```

然后观察：

```text
中文
英文
代码
数字
特殊符号
```

如何被拆分。

---

# 142. 实验 3：Single-head Self-Attention

不调用：

```python
nn.MultiheadAttention
```

自己写：

```python
Q = x @ Wq
K = x @ Wk
V = x @ Wv

scores = Q @ K.transpose(-2, -1)
scores = scores / sqrt(d)

scores = scores + causal_mask

weights = softmax(scores)

out = weights @ V
```

重点打印：

```text
Q shape
K shape
V shape
Score shape
Weight shape
Output shape
```

---

# 143. 实验 4：Multi-Head Causal Attention

实现：

```text
[B,T,D]
 ↓
Q/K/V
 ↓
[B,H,T,d]
 ↓
Attention
 ↓
Concat
 ↓
[B,T,D]
```

目标：

> Shape 必须完全清楚。

---

# 144. 实验 5：Tiny GPT

实现：

```text
Tokenizer
Embedding
Position Embedding
Transformer Block × N
LM Head
```

数据：

```text
Tiny Shakespeare
中文小文本
代码片段
```

训练到：

```text
能够生成形式上合理的文本
```

即可。

不要追求真正 ChatGPT 能力。

---

# 145. 实验 6：Pretraining Loss

记录：

```text
Train Loss
Validation Loss
```

观察：

```text
Epoch
Step
Learning Rate
```

对 Loss 的影响。

同时尝试：

```text
Context Length
Batch Size
Hidden Size
Layers
```

小范围变化。

---

# 146. 实验 7：Sampling 对比

固定同一个 Prompt。

分别：

```text
Greedy

Temperature = 0.2

Temperature = 0.8

Temperature = 1.2

Top-K

Top-P
```

观察生成差异。

---

# 147. 实验 8：KV Cache

至少做一个简化版本：

```text
Generation without KV Cache
```

和：

```text
Generation with KV Cache
```

比较：

```text
每 Token 推理时间
```

目标：

> 亲眼看到 Cache 为什么重要。

LLMs-from-scratch 仓库也提供 KV Cache Bonus 材料。

---

# 148. 实验 9：MiniMind Model 阅读

不要第一遍训练。

先直接：

```python
print(model)
```

然后逐层：

```text
Embedding
Block
Attention
FFN
Norm
LM Head
```

统计：

```text
Parameter Count
```

打印：

```text
每个 Tensor Shape
```

---

# 149. 实验 10：MiniMind Pretrain → SFT

使用仓库提供的 Mini 数据。

至少跑通：

```text
Pretraining
↓
得到 Pretrain Weight
↓
SFT
↓
得到 Full SFT Weight
↓
Chat / Eval
```

目标：

> 亲手完成一次从 Base 到 Chat 的最小链路。

---

# 150. 常见初学误区

---

## 误区 1：LLM = Transformer

不完全。

Transformer 是：

```text
Architecture
```

LLM 是：

```text
Architecture
+
Tokenizer
+
Weights
+
Training Data
+
Training Recipe
+
Inference / Generation
```

组成的完整系统。

---

## 误区 2：Token = Word

错误。

Token 可以是：

```text
Subword
Byte
Character
Special Symbol
```

---

## 误区 3：Attention 就是“模型知道什么重要”

太模糊。

工程上必须落实到：

```text
QK^T
↓
Scale
↓
Mask
↓
Softmax
↓
V
```

---

## 误区 4：Transformer 只有 Attention

错误。

一个 Block 还有大量：

```text
FFN
Norm
Residual
Linear
```

而且 FFN 参数和计算量非常重要。

---

## 误区 5：LLM 推理就是一次 Forward

不完全。

Chat Generation 是：

```text
Prefill
+
Decode × N
```

---

## 误区 6：有 KV Cache 就不用算 Attention

错误。

KV Cache 只是避免：

```text
重复计算过去 Token 的 K/V
```

当前 Query 仍要和：

```text
历史 K
```

做 Attention。

---

## 误区 7：Context Length 只影响最大输入长度

错误。

还会影响：

```text
KV Cache
Prefill Latency
Attention Work
Memory
```

---

## 误区 8：SFT 给模型“增加全部知识”

SFT 更重要的作用通常是：

```text
改变行为分布
学习指令格式
学习任务形式
```

而不是简单等同于“灌知识”。

---

## 误区 9：LoRA 等于 Quantization

完全不同。

LoRA：

```text
Parameter-efficient Fine-Tuning
```

Quantization：

```text
降低数值精度
```

二者可以同时使用。

---

## 误区 10：GGUF 是一种 LLM Architecture

不是。

GGUF 是后续：

```text
llama.cpp
```

生态中的模型文件格式。

模型 Architecture 可能是：

```text
LLaMA
Qwen
Gemma
...
```

---

# 151. 必须掌握的核心 Shape

```text
Input IDs
[B,T]

Embedding
[B,T,D]

Q
[B,H,T,d]

K
[B,H_kv,T,d]

V
[B,H_kv,T,d]

Attention Score
[B,H,T,T]

Attention Output
[B,T,D]

FFN Hidden
[B,T,D_ff]

Block Output
[B,T,D]

Logits
[B,T,Vocab]
```

看到模型代码：

> 第一件事仍然是追 Shape。

---

# 152. 必须掌握的四个维度符号

建议固定：

```text
B
=
Batch Size

T
=
Sequence Length

D
=
Hidden Size

V
=
Vocabulary Size
```

Attention 再加：

```text
H
=
Number of Heads

d
=
Head Dimension
```

所以：

```text
D = H × d
```

传统 MHA 中通常如此。

---

# 153. 从公式映射到 PyTorch

Self-Attention：

```text
Q = XWq
K = XWk
V = XWv
```

PyTorch：

```python
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)
```

---

# 154. 从 Attention 映射到 Kernel

模型层：

```text
Attention
```

算子层：

```text
Linear
MatMul
Softmax
Mask
MatMul
Linear
```

Kernel 层：

```text
GEMM
Softmax Kernel
FlashAttention Kernel
```

硬件：

```text
GPU / NPU
```

这就是后面第三部分的桥梁。

---

# 155. FlashAttention 为什么会出现？

标准 Attention 中会产生大型：

```text
T × T
```

中间矩阵。

核心问题不仅是：

```text
计算量
```

还有：

```text
HBM / DRAM 数据搬运
```

FlashAttention 的重点：

> 通过更好的 Tiling 与 IO-aware 算法减少高代价内存访问。

具体留到：

```text
10_GPU_Kernel与FlashAttention.md
```

---

# 156. 为什么 llama.cpp 需要理解本章？

llama.cpp 里面会出现：

```text
Token
Tokenizer
Embedding
RoPE
Attention
KV Cache
RMSNorm
FFN
Sampling
```

如果这章不清楚：

> llama.cpp 只能变成“会执行命令”，无法理解 Runtime。

---

# 157. 为什么 vLLM 需要理解本章？

vLLM 的核心问题包括：

```text
KV Cache
Batching
Sequence
Prefill
Decode
Scheduler
PagedAttention
```

它们全部建立在：

> LLM 自回归推理机制

之上。

---

# 158. 为什么 RKLLM 需要理解本章？

部署 Qwen 到 RK3576：

```text
不是只把文件转换一下
```

你还需要理解：

```text
Model Architecture
Context Length
KV Cache
Prefill
Decode
Quantization
Supported Operator
Runtime Memory
```

否则出了性能问题无法定位。

---

# 159. 本章学习树

```text
Language Model
│
├── Autoregressive Modeling
│   ├── Input / Target Shift
│   └── Next Token Prediction
│
├── Tokenizer
│   ├── Token
│   ├── Token ID
│   ├── Vocabulary
│   ├── BPE
│   ├── Encode / Decode
│   └── Special Tokens
│
├── Embedding
│   ├── Token Embedding
│   └── Position Information
│
├── Attention
│   ├── Q
│   ├── K
│   ├── V
│   ├── Scaled Dot Product
│   ├── Causal Mask
│   ├── Softmax
│   ├── MHA
│   └── GQA
│
├── Transformer Block
│   ├── Attention
│   ├── Residual
│   ├── LayerNorm / RMSNorm
│   ├── FFN
│   ├── GELU / SiLU
│   └── RoPE
│
├── GPT / Decoder-only
│   ├── Embedding
│   ├── Transformer × N
│   ├── LM Head
│   └── Logits
│
├── Training
│   ├── Pretraining
│   ├── SFT
│   ├── LoRA
│   ├── DPO
│   ├── PPO
│   └── GRPO
│
└── Inference
    ├── Chat Template
    ├── Prefill
    ├── Decode
    ├── KV Cache
    ├── Context Length
    ├── Sampling
    ├── Temperature
    ├── Top-K
    └── Top-P
```

---

# 160. 本章最重要的思维模型

## 思维模型 1

```text
LLM
=
Next Token Predictor
```

虽然真实能力很复杂，但训练核心可以先这样理解。

---

## 思维模型 2

```text
Text
不能直接进入 Neural Network

必须：

Text
↓
Tokenizer
↓
Token IDs
↓
Embedding
```

---

## 思维模型 3

```text
Self-Attention
=
Token-to-Token Information Routing
```

---

## 思维模型 4

```text
Attention
=
QK^T
→
Scale
→
Mask
→
Softmax
→
V
```

---

## 思维模型 5

```text
Transformer Block
=
Attention
+
FFN
+
Norm
+
Residual
```

---

## 思维模型 6

```text
GPT
=
Decoder-only Transformer
+
Causal Language Modeling
```

---

## 思维模型 7

```text
Base Model
=
Pretrained Language Model

Chat Model
=
Base
+
Instruction / Alignment Training
```

---

## 思维模型 8

```text
Training
=
整段并行 Next Token Prediction

Generation
=
逐 Token Autoregressive Decode
```

---

## 思维模型 9

```text
KV Cache
=
缓存过去 Token 的 K / V
避免重复计算
```

---

## 思维模型 10

```text
LLM Performance
=
Compute
+
Memory Bandwidth
+
KV Cache
+
Kernel
+
Quantization
+
Scheduler
+
Hardware
```

这会直接进入第二、三部分。

---

# 161. 本章完成标准

如果以下问题基本都能自己解释，本章就可以结束。

## Language Model

- [ ] 能解释什么是 Language Model
- [ ] 能解释 Autoregressive
- [ ] 能解释 Next Token Prediction
- [ ] 能解释 Input / Target 为什么错一位
- [ ] 能解释 Self-Supervised Pretraining

## Tokenizer

- [ ] 能解释 Token 和 Word 的区别
- [ ] 能解释 Vocabulary
- [ ] 能解释 Token ID
- [ ] 能解释 BPE 基本思想
- [ ] 能解释 `encode()` / `decode()`
- [ ] 能解释 Special Token
- [ ] 能说明为什么 Tokenizer 必须与 Weight 匹配
- [ ] 跑过一次 `minbpe` 或自己写过最小 BPE

## Embedding / Position

- [ ] 能解释 `nn.Embedding`
- [ ] 能解释 Embedding Matrix Shape
- [ ] 能解释为什么需要 Position
- [ ] 知道 Absolute Position 和 RoPE 的基本区别
- [ ] 知道 RoPE 主要作用于 Q/K

## Attention

- [ ] 能解释 Q/K/V
- [ ] 能解释 Self-Attention
- [ ] 能写出 Attention 核心公式
- [ ] 能推导 Attention Tensor Shape
- [ ] 能解释为什么是 `T × T`
- [ ] 能解释 `sqrt(d_k)`
- [ ] 能解释 Softmax
- [ ] 能解释 Causal Mask
- [ ] 能解释 MHA
- [ ] 能解释 GQA
- [ ] 知道 MQA 的位置

## Transformer

- [ ] 能画出 Transformer Block
- [ ] 能解释 Residual
- [ ] 能解释 LayerNorm / RMSNorm
- [ ] 能解释 FFN
- [ ] 能解释 SwiGLU 风格 FFN
- [ ] 能区分 Encoder-only / Encoder-Decoder / Decoder-only
- [ ] 能解释 GPT 为什么是 Decoder-only

## GPT / LLM

- [ ] 能画出 GPT 完整数据流
- [ ] 能解释 LM Head
- [ ] 能解释 Logits
- [ ] 能解释 Cross Entropy
- [ ] 能解释 Weight Tying
- [ ] 能解释 Parameter Count 大致来自哪里
- [ ] 能解释 Dense vs MoE
- [ ] 能解释 Total Parameter vs Active Parameter

## Training

- [ ] 能解释 Pretraining
- [ ] 能解释 Base Model
- [ ] 能解释 SFT
- [ ] 能解释 Chat Template
- [ ] 能区分 Base / Instruct
- [ ] 能解释 Full Fine-Tuning
- [ ] 能解释 LoRA
- [ ] 知道 QLoRA 的基本思想
- [ ] 知道 DPO / PPO / GRPO 在 Training Pipeline 的位置
- [ ] 跑过一次 Tiny GPT Pretraining
- [ ] 最好跑过一次 MiniMind Pretrain → SFT

## Inference

- [ ] 能解释 Autoregressive Generation
- [ ] 能解释 Prefill
- [ ] 能解释 Decode
- [ ] 能解释 KV Cache
- [ ] 能解释为什么不缓存 Q
- [ ] 能解释 Context Length 对 KV Cache 的影响
- [ ] 能解释 Greedy
- [ ] 能解释 Temperature
- [ ] 能解释 Top-K
- [ ] 能解释 Top-P
- [ ] 能解释 EOS

## Runtime / Hardware Bridge

- [ ] 能说明 Linear 为什么最终对应 MatMul
- [ ] 能把 Attention 拆成 Operator
- [ ] 能解释为什么 Prefill 与 Decode 性能特征不同
- [ ] 能解释为什么 Decode 容易受 Memory Bandwidth 影响
- [ ] 能解释为什么量化有可能提升 Decode 速度
- [ ] 能解释为什么 GQA 有利于 KV Cache
- [ ] 能说明 FlashAttention 大致在优化什么
- [ ] 能说明为什么后续需要 llama.cpp / vLLM / RKLLM

---

# 162. 本章不要求完成的内容

不要求：

- [ ] 手推 RoPE 全部复数数学
- [ ] 精通所有 Position Encoding
- [ ] 精通所有 Attention Variant
- [ ] 精通 MoE Router 数学
- [ ] 精通 PPO / GRPO 推导
- [ ] 从零训练十亿参数模型
- [ ] 精通分布式训练
- [ ] 精通 Tensor Parallel
- [ ] 精通 Pipeline Parallel
- [ ] 精通 FlashAttention Kernel
- [ ] 精通 CUDA
- [ ] 精通量化算法
- [ ] 精通 llama.cpp
- [ ] 精通 vLLM
- [ ] 精通 RKLLM

这些属于后续章节。

---

# 163. 核心参考链接

## LLMs-from-scratch

- [Repository](https://github.com/rasbt/LLMs-from-scratch)
- [Chapter 2 - Text Data](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02)
- [Chapter 3 - Attention](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03)
- [Chapter 3 README](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch03/README.md)
- [Chapter 4 - GPT](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04)
- [GPT Source](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch04/01_main-chapter-code/gpt.py)
- [Chapter 5 - Pretraining](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05)
- [Chapter 7 - Instruction Finetuning](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07)
- [Instruction Finetuning Script](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/gpt_instruction_finetuning.py)
- [DPO Bonus](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch07/04_preference-tuning-with-dpo)

---

## MiniMind

- [Repository](https://github.com/jingyaogong/minimind)
- [README](https://github.com/jingyaogong/minimind/blob/master/README.md)
- [Model Source](https://github.com/jingyaogong/minimind/blob/master/model/model_minimind.py)
- [Pretraining Script](https://github.com/jingyaogong/minimind/blob/master/trainer/train_pretrain.py)
- [Full SFT Script](https://github.com/jingyaogong/minimind/blob/master/trainer/train_full_sft.py)
- [Trainer Directory](https://github.com/jingyaogong/minimind/tree/master/trainer)
- [Dataset Directory](https://github.com/jingyaogong/minimind/tree/master/dataset)

重点读：

```text
model/model_minimind.py
trainer/train_pretrain.py
trainer/train_full_sft.py
```

---

## Karpathy

- [Neural Networks: Zero to Hero](https://github.com/karpathy/nn-zero-to-hero)
- [Course README](https://github.com/karpathy/nn-zero-to-hero/blob/master/README.md)
- [minbpe](https://github.com/karpathy/minbpe)
- [nanoGPT](https://github.com/karpathy/nanoGPT)

本章重点：

```text
Lecture 2
Language Modeling

Lecture 7
GPT from Scratch

Lecture 8
GPT Tokenizer
```

---

## D2L

- [D2L Repository](https://github.com/d2l-ai/d2l-en)
- [Attention Mechanisms and Transformers](https://github.com/d2l-ai/d2l-en/tree/master/chapter_attention-mechanisms-and-transformers)

---

# 164. 推荐仓库组合方式

推荐：

```text
LLMs-from-scratch
=
主教材

Zero to Hero
=
亲手从零建立 Language Model / GPT / Tokenizer

D2L
=
补齐 Attention 理论与 Transformer 体系

MiniMind
=
从教学 GPT 迁移到现代 Qwen 类 LLM
```

这四者结合后，基本可以把：

```text
Transformer / LLM
```

从“会调 API”变成：

> 能够阅读结构、理解训练、理解推理，并继续往 Runtime / Hardware 下钻。

---

# 165. 从本章进入第二部分

到这里，我们已经完成第一部分最核心的理论链：

```text
AI
 ↓
Machine Learning
 ↓
Deep Learning
 ↓
Neural Network
 ↓
Attention
 ↓
Transformer
 ↓
LLM
```

接下来的问题不再只是：

> LLM 是怎么构成的？

而是：

> **怎样把 LLM 变成真正可用的应用和高性能推理系统？**

于是学习重点开始转向：

```text
LLM
│
├── Application
│   ├── RAG
│   ├── Tool Calling
│   └── Agent
│
└── Runtime
    ├── llama.cpp
    ├── vLLM
    └── MLC-LLM
```

---

# 166. 一句话总结

这一章真正需要理解的主线是：

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embedding
 ↓
Transformer
 ↓
Attention + FFN
 ↓
Logits
 ↓
Next Token
```

Transformer 最核心：

```text
Q
K
V
↓
Attention
↓
Token-to-Token Information Routing
```

GPT 最核心：

```text
Causal Transformer
+
Next Token Prediction
```

LLM Training 最核心：

```text
Pretraining
→
Base Model
→
SFT
→
Chat / Instruct Model
→
Preference / RL Alignment
```

LLM Inference 最核心：

```text
Prefill
→
KV Cache
→
Decode
→
Sampling
→
Next Token
→
Repeat
```

当这几条链全部清楚以后，下一阶段：

```text
AI Agent
llama.cpp
vLLM
MLC-LLM
```

就不再是彼此孤立的框架，而会变成：

> **围绕同一个 LLM 模型，在应用层、Runtime 层和硬件层做不同事情的一整套系统。**
