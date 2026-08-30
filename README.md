# 从 AI 基础到端侧部署优化

> 一套面向嵌入式与端侧 AI 工程师的系统化学习教程：从机器学习、Transformer 与 LLM 出发，逐步深入推理 Runtime、AI Compiler、GPU/NPU 优化，并最终落地到 RK3576。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![Version: v0.2](https://img.shields.io/badge/version-v0.2-blue.svg)](./从AI基础到端侧部署优化/AI学习路线图_v0.2.md) [![GitHub Stars](https://img.shields.io/github/stars/yunxiao11xie/ai-to-edge-deployment?style=flat&logo=github&label=stars)](https://github.com/yunxiao11xie/ai-to-edge-deployment/stargazers)

## 项目定位

这套教程围绕一个核心问题展开：

> 一个 Transformer / LLM，从 PyTorch 中的模型定义开始，究竟经历了什么，最终才变成 ARM CPU、GPU 或 Rockchip NPU 上执行的一条条计算指令？

内容按照“模型是什么 → 模型如何运行 → 模型为什么快或慢 → 如何部署到端侧硬件”的路径组织，适合具备一定 Python、Linux、C/C++ 或嵌入式基础，希望系统学习端侧 AI 的开发者。

## 学习路线

完整导航请从 [AI 学习路线图 v0.2](./从AI基础到端侧部署优化/AI学习路线图_v0.2.md) 开始。

### 第一部分：AI 与模型基础

1. [AI 基础与机器学习](./从AI基础到端侧部署优化/01_AI基础与机器学习.md)
   - [公开实验室](./labs/01_ai_basics_ml/README.md) · [完整实验](./labs/01_ai_basics_ml/完整实验.ipynb) · [练习](./labs/01_ai_basics_ml/练习.ipynb) · [真实任务](./labs/01_ai_basics_ml/真实任务_设备故障预警.md)
2. [深度学习与神经网络](./从AI基础到端侧部署优化/02_深度学习与神经网络.md)
3. [PyTorch 与模型框架](./从AI基础到端侧部署优化/03_PyTorch与模型框架.md)
4. [Transformer 与 LLM](./从AI基础到端侧部署优化/04_Transformer与LLM.md)

### 第二部分：LLM 应用与推理 Runtime

5. [LLM 推理原理](./从AI基础到端侧部署优化/05_LLM推理原理.md)
6. [AI Agent 与 RAG](./从AI基础到端侧部署优化/06_AI_Agent与RAG.md)
7. [llama.cpp](./从AI基础到端侧部署优化/07_llama.cpp.md)
8. [vLLM 与 MLC-LLM](./从AI基础到端侧部署优化/08_vLLM与MLC-LLM.md)

### 第三部分：Compiler、硬件优化与端侧部署

9. [AI Compiler 与 TVM](./从AI基础到端侧部署优化/09_AI_Compiler与TVM.md)
10. [GPU Kernel 与 FlashAttention](./从AI基础到端侧部署优化/10_GPU_Kernel与FlashAttention.md)
11. [RKNN 与 Rockchip NPU](./从AI基础到端侧部署优化/11_RKNN与Rockchip_NPU.md)
12. [RKLLM 与端侧 LLM 部署](./从AI基础到端侧部署优化/12_RKLLM与端侧LLM部署.md)
13. [RK3576 端侧 AI 实践](./从AI基础到端侧部署优化/13_RK3576端侧AI实践.md)

### 实战项目

- [RK3576 本地多模态设备助手：项目总纲与 Linux 学习路线](./从AI基础到端侧部署优化/RK3576本地多模态设备助手/00_项目总纲与Linux系统学习路线.md)

### 进阶专题：RK3576 端侧大模型部署

- [Qwen3.5-2B + RKLLM：从模型转换到可复现实战](./advanced/rk3576-qwen3.5-rkllm/README.md)

## 阅读建议

- 初学者建议从第 1 章开始顺序阅读。
- 熟悉 AI 模型、重点关注部署的读者，可以从第 5 章开始。
- 从事 Rockchip 平台开发的读者，可以重点阅读第 11～13 章及实战项目。
- `_archive` 保存早期卡片版与历史稿件，不属于当前教程主线。


## 第 1 章公开实验室

第 1 章已完整开放一套可运行的学习样板。它不只提供阅读材料，而是把一次学习拆成五个连续环节：

```text
阅读讲义 → 运行完整实验 → 独立练习 → 机器验收 → 完成真实任务
```

| 环节     | 内容                                           | 直接入口                                                                    |
| -------- | ---------------------------------------------- | --------------------------------------------------------------------------- |
| 阅读讲义 | 建立 AI、机器学习、训练、推理与评估的知识框架  | [第 1 章 AI 基础与机器学习](./从AI基础到端侧部署优化/01_AI基础与机器学习.md) |
| 完整实验 | 从头运行回归、分类、聚类与 Q-learning 参考实现 | [完整实验 Notebook](./labs/01_ai_basics_ml/完整实验.ipynb)                   |
| 独立练习 | 补全四项核心任务，观察错误并解释结果           | [练习 Notebook](./labs/01_ai_basics_ml/练习.ipynb)                           |
| 机器验收 | 检查真实模型变量、预测结果、指标和学习策略     | [实验室与验收说明](./labs/01_ai_basics_ml/README.md)                         |
| 真实任务 | 构建设备故障预警原型，提交可复算的预测证据     | [真实任务说明](./labs/01_ai_basics_ml/真实任务_设备故障预警.md)              |

### 快速开始

克隆仓库后进入实验目录：

```bash
cd labs/01_ai_basics_ml
python -m venv .venv
```

激活虚拟环境：

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

安装依赖并启动实验：

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

建议先打开 `完整实验.ipynb`，运行并理解完整流程；随后关闭参考实现，独立完成 `练习.ipynb`。练习通过后，再进入真实任务。详细要求、验收方式和文件说明见[第 1 章实验室导航](./labs/01_ai_basics_ml/README.md)。

> GitHub 页面可以预览 Notebook，但要修改代码、运行单元格和生成验收结果，需要先把仓库克隆到本地或下载后在 Jupyter 中打开。

## 项目状态

当前为持续完善中的早期版本。AI 框架、推理工具链和芯片 SDK 更新较快，请结合各章节标注的版本与日期阅读。欢迎通过 Issue 提交勘误、建议和实践反馈。

## 参与贡献

欢迎提交 Issue 或 Pull Request，包括但不限于：

- 修正技术错误、失效链接和命令；
- 补充可复现实验、性能数据与硬件环境；
- 改进章节结构、图表和示例代码；
- 分享不同端侧平台上的迁移经验。

提交贡献时，请注明测试环境、软件版本和硬件平台。

## 许可证

本项目采用 [MIT License](./LICENSE)。引用或使用第三方项目、文档与代码时，其版权仍归原作者所有，并遵循对应项目的许可证。
