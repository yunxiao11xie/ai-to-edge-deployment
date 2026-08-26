---
tags:
  - RK3576
  - RKLLM
  - 性能优化
  - Roofline
aliases:
  - RK3576性能分析
---

# RK3576 性能模型：GEMM、GEMV 与 Roofline

> 本文不把“6 TOPS”当成应用速度。目标是建立一套可以解释 **TTFT、Prefill tok/s、Decode tok/s 为什么变化** 的性能模型，并用实验判定瓶颈。

## 1. 首先分清四类时间

一次请求的用户等待时间不是一个数字：

\[
T_{total}=T_{template}+T_{tokenize}+T_{prefill}+T_{decode}+T_{render}
\]

| 指标 | 含义 | 用户感知 |
|---|---|---|
| TTFT | 请求进入到首个 token 可见 | “多久开始回答” |
| Prefill tok/s | 批量处理输入 token 的速度 | 长提示词首字等待 |
| Decode tok/s | 自回归逐 token 生成速度 | 回答是否流畅 |
| E2E latency | 从提交到回答结束 | 整体完成时间 |

必须说明计时边界。只引用 Runtime 日志的 Prefill/Generate，不能等价成 Qt 应用的端到端体验。

## 2. Qwen3.5-2B 中矩阵运算的 Shape

设序列长度为 `T`，隐藏维度 `D=2048`，FFN 中间维度 `Dff=6144`。

### 2.1 Prefill 更接近 GEMM

线性层的核心形式为：

\[
[T,D]\times[D,K]\rightarrow[T,K]
\]

当 `T` 较大时，同一块权重可被多个 token 复用，容易形成矩阵—矩阵乘法（GEMM）。计算单元利用率和算术强度通常比 Decode 更高。

以 FFN 为例：

- Gate/Up 投影：`[T,2048] × [2048,6144]`；
- Down 投影：`[T,6144] × [6144,2048]`；
- 最后的 LM Head：`[T,2048] × [2048,248320]`，但推理实现可能只对所需位置做输出投影或使用专门优化，不能仅凭公式推断实际耗时。

### 2.2 Decode 更接近 GEMV/小 GEMM

每步通常只有一个新 token：

\[
[1,D]\times[D,K]\rightarrow[1,K]
\]

这是矩阵—向量乘法（GEMV）或极小批次 GEMM。每生成一个 token，都要重新读取大量权重，权重复用率很低。因此 Decode 经常是**带宽、访存和调度受限**，不是理论 TOPS 受限。

这解释了当前实测中 Prefill 约 `26.37 tok/s`，Decode 约 `6.14 tok/s`：两者执行的是同一模型，却处于完全不同的工作区间。

## 3. Roofline：算力上限和带宽上限取较小者

Roofline 的核心关系是：

\[
P_{attainable}\leq\min(P_{peak}, BW\times AI)
\]

- `P_peak`：特定精度、算子和 Shape 下的可达计算峰值；
- `BW`：端到端有效内存带宽，而不是只看 DRAM 标称带宽；
- `AI`：算术强度，即每搬运一个字节完成多少运算。

Decode 中 `AI` 较低；即使 NPU 标称 TOPS 很高，数据喂不满计算阵列，TOPS 也无法直接转化为 tok/s。

### 3.1 一个有用但粗糙的 Decode 上界

若每生成一个 token 至少要流过 `W_effective` 字节的有效权重，则有：

\[
tokens/s\leq\frac{BW_{effective}}{W_{effective}}
\]

这个式子适合做数量级检查，不能作为精确预测，原因包括：

- 权重可能分块、压缩、缓存或重排；
- 还有激活、状态、KV Cache 和中间缓冲区流量；
- NPU、DDR、CPU 之间可能发生额外拷贝；
- Runtime 可能把多个算子融合，也可能产生同步与启动开销；
- `.rkllm` 文件大小不等于每 token 的真实 DRAM 流量。

## 4. “6 TOPS”为什么不能直接换算 tok/s

芯片宣传值通常要求特定数据类型、稠密度、频率和理想 Shape。应用还受以下条件影响：

1. W4A16 中并非所有算子都是 4 bit 运算；
2. Softmax、归一化、采样、Tokenizer 等不一定运行在 NPU；
3. 小矩阵、动态 Shape、同步和数据搬运会降低阵列利用率；
4. Qwen3.5 混合了线性注意力和全注意力，执行路径不是单一 GEMM；
5. 温度、功耗和 DVFS 会改变持续频率；
6. Ubuntu Desktop、Qt、摄像头和后台服务与推理共享 CPU、内存及带宽。

因此性能结论必须落到“模型版本 + 量化 + 上下文 + 频率 + 温度 + Runtime”的完整条件上。

## 5. 性能瓶颈的分层归因

| 层次 | 典型信号 | 验证办法 |
|---|---|---|
| 模型/算法 | 长提示词 TTFT 随长度明显增长 | 做 Prompt 长度扫描 |
| 算子/Shape | Prefill 快、Decode 明显慢 | 比较 Prefill 与 Decode；改变批量或上下文 |
| 内存带宽 | NPU 利用率不持续、Decode 对后台内存流量敏感 | 关闭桌面/相机并做 A/B |
| 容量/换页 | 延迟突刺、swap in/out、系统卡顿 | 监测 `vmstat` 与 PSS |
| Runtime 调度 | 短请求固定开销占比高 | 扫描不同生成长度 |
| CPU 前后处理 | NPU 空闲但单核忙 | 分离 Tokenizer、采样与 UI 计时 |
| 热/频率 | 前几轮快、持续运行后慢 | 同时记录温度和频率 |

不要一看到“速度慢”就改量化。先确定是哪一层慢。

## 6. 一套可复现的基准方法

### 6.1 固定条件

- 固定 `.rkllm` 文件及 SHA-256；
- 固定 Toolkit、Runtime、头文件和驱动版本；
- 固定 `max_context_len`、`max_new_tokens`、采样参数；
- 固定提示词 token 数，而不只固定中文字数；
- 固定是否保存历史、是否使用 Prompt Cache；
- 记录散热方式、环境温度、系统负载和桌面是否运行；
- 预热 2～3 次，再正式测试至少 10 次。

### 6.2 测试矩阵

| 变量 | 建议取值 | 要回答的问题 |
|---|---|---|
| Prompt tokens | 32/128/512/1024/2048 | Prefill 是否近似线性 |
| Generate tokens | 32/64/128/256 | Decode 是否稳定，有无热降频 |
| Context limit | 1024/2048/4096 | 预分配是否影响内存和初始化 |
| NPU cores | 1/2/工具允许的其他值 | 多核收益和带宽竞争 |
| Quant | W4A16/W8A8 | 容量、质量和速度交换 |
| 系统场景 | 纯终端/Qt/相机+Qt | 产品负载对推理的影响 |

### 6.3 统计方法

平均值会隐藏偶发卡顿，至少报告：

- p50、p95 TTFT；
- Prefill 和 Decode 的均值、标准差；
- 峰值 PSS 或 Runtime 明确定义的 Peak Memory；
- 首轮与持续 10 分钟后的速度；
- 错误率、超时率和中止成功率。

## 7. 当前 W4A16 基线该怎样解释

已跑通基线：Prompt 20 tokens，实际生成 55 tokens，三次平均：

| 指标 | 当前值 |
|---|---:|
| Prefill 时间 | 762.27 ms |
| Prefill 速度 | 26.37 tok/s |
| Decode 时间 | 8955.12 ms |
| Decode 速度 | 6.14 tok/s |
| 一次日志中的 Peak Memory | 1501 MB |

这组数据能证明“闭环可运行”，但还不能证明：

- 不同上下文下是否稳定；
- Qt、IMX415、RKNN 同开后的速度；
- 1501 MB 是进程 RSS、PSS、Runtime 工作集还是工具内部统计；
- W4A16 是否优于 W8A8；
- 长时间运行是否热降频。

这些正是下一轮实验要补齐的内容。

## 8. 性能报告模板

```yaml
model: Qwen3.5-2B-text
artifact_sha256: TODO
quant: W4A16_GRQ
runtime: 1.3.0
driver: 0.9.7
prompt_tokens: 20
generated_tokens: 55
context_limit: 2048
npu_cores: 2
warmup_runs: 3
measured_runs: 10
ttft_ms_p50: TODO
ttft_ms_p95: TODO
prefill_tps_mean: TODO
decode_tps_mean: TODO
pss_peak_mb: TODO
temperature_peak_c: TODO
swap_in_out: TODO
desktop_camera_state: TODO
```

## 9. 验收标准

- 能解释 GEMM 与 GEMV 为什么对应不同性能区间；
- 不再用 TOPS 直接推算 tok/s；
- 能用 Roofline 判断“算力受限”还是“带宽受限”；
- 基准包含版本、Shape、温度、内存和统计分位数；
- 每项优化都有 A/B 对照，并能指出优化影响了哪一层。

## 参考

- [01_Qwen3.5-2B模型解剖与Shape主线](./01_Qwen3.5-2B模型解剖与Shape主线.md)
- [04_Prefill_Decode_混合注意力与缓存](./04_Prefill_Decode_混合注意力与缓存.md)
- [10_Qwen3.5-2B_RK3576完整可复现实战](./10_Qwen3.5-2B_RK3576完整可复现实战.md)

