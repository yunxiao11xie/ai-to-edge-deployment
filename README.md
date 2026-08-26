# 从 AI 基础到端侧部署优化

> 一套面向嵌入式与端侧 AI 工程师的系统化学习教程：从机器学习、Transformer 与 LLM 出发，逐步深入推理 Runtime、AI Compiler、GPU/NPU 优化，并最终落地到 RK3576。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## 项目定位

这套教程围绕一个核心问题展开：

> 一个 Transformer / LLM，从 PyTorch 中的模型定义开始，究竟经历了什么，最终才变成 ARM CPU、GPU 或 Rockchip NPU 上执行的一条条计算指令？

内容按照“模型是什么 → 模型如何运行 → 模型为什么快或慢 → 如何部署到端侧硬件”的路径组织，适合具备一定 Python、Linux、C/C++ 或嵌入式基础，希望系统学习端侧 AI 的开发者。

## 学习路线

完整导航请从 [AI 学习路线图 v0.2](./从AI基础到端侧部署优化/AI学习路线图_v0.2.md) 开始。

### 第一部分：AI 与模型基础

1. [AI 基础与机器学习](./从AI基础到端侧部署优化/01_AI基础与机器学习.md)
2. [深度学习与神经网络](./从AI基础到端侧部署优化/02_深度学习与神经网络.md)
3. [Transformer 与 LLM](./从AI基础到端侧部署优化/03_Transformer与LLM.md)
4. [PyTorch 与模型框架](./从AI基础到端侧部署优化/04_PyTorch与模型框架.md)

### 第二部分：LLM 应用与推理 Runtime

5. [AI Agent 与 RAG](./从AI基础到端侧部署优化/05_AI_Agent与RAG.md)
6. [LLM 推理原理](./从AI基础到端侧部署优化/06_LLM推理原理.md)
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
- 熟悉 AI 模型、重点关注部署的读者，可以从第 6 章开始。
- 从事 Rockchip 平台开发的读者，可以重点阅读第 11～13 章及实战项目。
- `_archive` 保存早期卡片版与历史稿件，不属于当前教程主线。

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
