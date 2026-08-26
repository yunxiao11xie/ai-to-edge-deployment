---
tags:
  - RK3576
  - IMX415
  - RKNN
  - RKLLM
  - 多模态
aliases:
  - RK3576本地多模态链路
---

# IMX415 + RKNN + RKLLM 多模态链路

> 真正的本地多模态不是“把 JPEG 发给 LLM”。它是一条跨越摄像头驱动、像素预处理、视觉编码器、跨模态投影、语言模型和 Qt 显示的异构流水线。

## 1. 产品目标与工程边界

本机条件：RK3576、4GB RAM、32GB 存储、10 寸屏、IMX415、键鼠、Ubuntu Desktop。目标是本地设备助手：

- 看到实时画面；
- 用户点击拍照或选定区域；
- 设备在本地理解画面并回答问题；
- Qt 显示流式答案、性能状态与错误提示；
- 后续可以接语音、传感器、RAG 和局域网 API。

第一版不要追求“每帧都跑大模型”。合理交互是：高帧率预览 + 用户触发/低频视觉推理 + 流式文本生成。

## 2. 完整数据流

```text
IMX415 Sensor
   ↓ MIPI CSI-2 / ISP / Media Controller
V4L2 Capture（优先 NV12/YUV）
   ├──────────────→ 显示支路 → Qt 10 寸屏预览
   │
   └→ 选帧/裁剪 → RGA/CPU resize + color convert
                    ↓ 视觉模型规定的 tensor
                 RKNN Vision Encoder（NPU）
                    ↓ image embeddings / features
                 跨模态位置与投影处理
                    ↓ multimodal input
                 RKLLM Language Model（NPU）
                    ↓ token callback
                 Qt 文本流式显示
```

链路中每一条箭头都需要约定 Shape、dtype、layout、颜色空间、所有权和生命周期。

## 3. IMX415 到用户空间：先把摄像头当独立子系统

先单独验收：

1. Media graph、Sensor、ISP 和视频节点枚举正常；
2. 分辨率、帧率、曝光和白平衡可控；
3. 连续预览 30 分钟无丢帧、绿屏和缓冲泄漏；
4. 保存一帧后用离线工具确认颜色与步长正确；
5. 摄像头开关 100 次不会耗尽 dma-buf/CMA。

不要在摄像头链路尚不稳定时引入 RKNN 和 RKLLM，否则任何绿屏、颜色偏差或崩溃都难以定位。

## 4. 显示支路与推理支路必须分开

IMX415 可输出高分辨率画面，但视觉编码器通常需要较小且具有固定规则的输入。推荐：

```text
Capture buffer
   ├─ Preview queue：只保留最新帧，允许丢帧
   └─ Inference queue：用户触发时锁定一帧，容量 1～2
```

- 预览队列不能无限堆积；旧帧没有价值；
- 推理线程不能占住 V4L2 缓冲导致采集停滞；
- Qt 显示与 NPU 输入尽量避免各复制一份 4K RGB；
- 使用 RGA/硬件路径前，先用 CPU 参考实现验证像素正确性；
- 明确 stride，不要假设 `stride == width × channels`。

## 5. Qwen3.5-2B 的视觉结构不是黑盒图片接口

模型配置给出的视觉部分关键参数为：

| 参数 | 值 | 含义 |
|---|---:|---|
| vision depth | 24 | 视觉 Transformer 层数 |
| hidden size | 1024 | 视觉主干宽度 |
| intermediate size | 4096 | 视觉 FFN 宽度 |
| attention heads | 16 | 视觉注意力头数 |
| patch size | 16 | 空间 patch 尺寸 |
| temporal patch size | 2 | 时间维 patch 规则 |
| spatial merge size | 2 | 空间 token 合并 |
| out hidden size | 2048 | 输出对齐语言隐藏维度 |

官方 Qwen3.5 视觉导出示例使用 `pixel` 和 `grid_thw` 作为输入。`grid_thw` 描述时间、高度、宽度方向的网格，意味着预处理不仅是 resize；还要满足模型处理器的 patch、merge、动态分辨率和位置编码约定。

## 6. `.rknn` 与 `.rkllm` 的职责

典型拆分为：

| 制品 | 执行内容 | 输出/输入边界 |
|---|---|---|
| Vision `.rknn` | 视觉编码器或导出的视觉子图 | 像素 tensor → 视觉 embedding |
| Language `.rkllm` | Qwen3.5 语言模型部分 | 文本/embedding/多模态输入 → token |

RKLLM C API 支持 Prompt、Token、Embedding 和 Multimodal 等输入类型，但具体结构、字段和顺序必须参考同版本官方多模态 Demo。不能把视觉 embedding 当普通 UTF-8 字符串传递。

跨模态边界需要核对：

- embedding dtype；
- embedding Shape 和有效长度；
- image token 在 Chat Template 中的位置；
- `grid_thw` 与输出 token 数的对应关系；
- 图像、embedding 和请求对象在异步运行期间是否仍然有效；
- Vision 与 LLM 是否能同时占用 NPU，还是必须串行调度。

## 7. 预处理必须逐项可验证

建立如下记录：

```yaml
capture_format: NV12
capture_size: [3840, 2160]
crop_policy: center_crop_or_letterbox_TODO
processor_revision: TODO
vision_input_dtype: float32_TODO_verify
vision_input_layout: TODO
normalization: TODO_from_processor
grid_thw: TODO_record_each_request
```

调试时保存：

1. 摄像头原始帧；
2. 裁剪/缩放后可视化图；
3. 输入 tensor 的 min/max/mean 和前若干值；
4. `grid_thw`；
5. PC 参考视觉输出与 RKNN 输出的相似度或误差。

若文字识别错误、颜色理解反了或目标位置偏移，优先查颜色空间、resize/crop、归一化和 token 排列，不要先怪量化。

## 8. 线程与资源所有权

推荐最小线程结构：

```text
GUI Thread
  ├─ 显示最新预览帧
  ├─ 收取 token event
  └─ 发出“拍照/提问/停止”命令

Capture Thread
  └─ V4L2 dequeue/queue，发布只读帧句柄

Vision Worker
  └─ 选帧 → 预处理 → RKNN → embedding

LLM Worker
  └─ 串行拥有 RKLLM handle，流式回调入队
```

关键规则：

- Qt 控件只在 GUI 线程更新；
- Capture buffer 通过明确的引用计数或拷贝策略管理；
- 队列有容量和丢弃策略；
- Vision/LLM 共用 NPU 时，由统一调度器决定顺序；
- 用户停止请求后，旧请求的 token 不得写入新对话；
- 每个请求携带 request_id，所有事件必须核对 ID。

## 9. 4GB 内存下的调度策略

多模态版的主要风险不是单个模型，而是同时驻留：

- Ubuntu Desktop + Qt；
- 4K 摄像头多缓冲；
- RKNN 视觉模型、工作区与输出 embedding；
- RKLLM 模型、状态和上下文；
- 图像缩放中间副本；
- 网页/API/日志等辅助模块。

建议先实测两种模式：

### 模式 A：视觉和语言同时常驻

体验好，但峰值高。适合容量确认后使用。

### 模式 B：分阶段加载/运行

视觉编码完成后释放临时缓冲，必要时卸载视觉模型，再运行 LLM。首问延迟更大，但适合 4GB 容量兜底。

不能只根据模型文件大小选模式，要用 [06_4GB内存容量工程与测量方法](./06_4GB内存容量工程与测量方法.md) 的 PSS、CMA 和全功能峰值决定。

## 10. 分阶段实施路线

### Stage 0：纯文本基线

已完成 Qwen3.5-2B W4A16 的 RKLLM 推理，固定版本和性能数据。

### Stage 1：相机预览

只实现 IMX415 → V4L2 → Qt。验收帧率、颜色、稳定性和内存。

### Stage 2：离线视觉编码器

先用固定测试图片跑 PC 参考和 RKNN，不接相机，不接 LLM。验证输入 tensor 与输出 embedding。

### Stage 3：摄像头接视觉模型

点击拍照后选帧、预处理、推理；验证和固定图片结果一致。

### Stage 4：视觉 embedding 接 RKLLM

使用官方同版本 Multimodal Demo 作为金标准，先得到单图问答结果。

### Stage 5：Qt 产品闭环

加入流式输出、停止生成、错误恢复、性能面板和容量保护。

### Stage 6：连续感知

只在上述稳定后增加目标检测、事件触发、OCR 或周期性理解，避免让 VLM 无意义地处理每一帧。

## 11. 多模态验收集

准备固定的 20～50 张图片：

- 室内物品、人物、室外场景；
- 中文/英文印刷文字；
- 小目标、低照度、逆光、模糊；
- 相同场景不同裁剪与分辨率；
- 无法回答或图中不存在目标的负样本。

每张图片记录：原图哈希、处理器版本、输入尺寸、`grid_thw`、Vision 输出摘要、Prompt、回答、TTFT、Decode、峰值内存。评估不仅看“回答像不像”，还看物体存在性、计数、OCR、方位和拒答。

## 12. 验收标准

- 能画出从 IMX415 到 token 的完整数据流；
- 显示支路与推理支路解耦且队列有界；
- 预处理有 PC 参考与 tensor 统计证据；
- 明确 `.rknn` 和 `.rkllm` 的接口契约；
- 多模态全功能场景在 4GB 上无 OOM 和持续换页；
- Qt UI 不被 NPU 推理或 Callback 阻塞。

## 官方参考

- [RKLLM Multimodal Demo](https://github.com/airockchip/rknn-llm/tree/main/examples/multimodal_model_demo)
- [Qwen3.5 视觉导出示例](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_vision.py)
- [Qwen3.5-2B 配置](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json)

