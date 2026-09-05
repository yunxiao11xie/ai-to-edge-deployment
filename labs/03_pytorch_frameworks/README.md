# 第 3 章实验室：PyTorch 与模型框架

> 本目录把“框架—格式—Runtime”变成一条可运行、可验证、可交付的工程链路。

## 学习入口

| 环节 | 作用 | 入口 |
|---|---|---|
| 阅读讲义 | 建立软件栈和工件边界 | [第 3 章讲义](../../从AI基础到端侧部署优化/03_PyTorch与模型框架.md) |
| 完整实验 | 跑通 Checkpoint、ONNX 与 ORT | [完整实验](./完整实验.ipynb) |
| 独立练习 | 补全保存、导出和验证步骤 | [练习](./练习.ipynb) |
| 验收 | 检查工件存在、图结构和数值一致性 | [`validate.py`](./validate.py) |
| 真实任务 | 交付可由第三方审计的模型包 | [任务说明](./真实任务_模型交付包审计.md) |

```text
建立角色地图 → 运行参考链路 → 独立生成工件 → 机器验收
      → 打包模型与 Manifest → 第三方复算
```

## 环境与运行

Python 3.10+，默认 CPU，不要求 GPU。完整实验约 90～120 分钟，练习约 120～180 分钟，真实任务约 3～5 小时。

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m jupyter lab
```

练习通过后生成 `artifacts/exercise_result.json`。真实任务生成 `artifacts/delivery_manifest.json`、Checkpoint 与 ONNX 文件后运行：

```bash
python validate.py --challenge artifacts/delivery_manifest.json
```

练习验收需要保留 Notebook 中的 `model`、`sample` 以及工件路径和输出变量。验收器会重新加载 Checkpoint、核对当前模型输出、检查 ONNX 接口并实际运行 ORT。

真实任务按 `delivery_model.py` 的固定结构重建模型，使用 `weights_only=True` 严格加载 Checkpoint，再分别执行 PyTorch 和 ORT。验收器同时检查 Manifest 与 ONNX 中的名称、float32、特征维、同名动态 Batch 和 Opset 17，核对固定输入及两份输出，并实际测试 Batch=1 和 Batch=7。跨 Runtime 最大绝对误差须 ≤ `1e-5`，申报输出及误差与重算结果的差异须 ≤ `1e-6`；工程结论至少 100 个字符，内容由人工检查。

导出统一使用 `dynamic_axes`、`opset_version=17` 和 `dynamo=False`。权重、ONNX 与 Manifest 必须一起提交，具体要求见[任务说明](./真实任务_模型交付包审计.md)。

## 文件结构

```text
03_pytorch_frameworks/
├─ README.md
├─ 完整实验.ipynb
├─ 练习.ipynb
├─ validate.py
├─ delivery_model.py        # 真实任务的固定模型结构
├─ challenge_starter.py
├─ 真实任务_模型交付包审计.md
├─ requirements.txt
├─ tests/test_chapter03_validate.py
└─ artifacts/
```
