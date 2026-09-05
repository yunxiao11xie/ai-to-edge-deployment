# 第 2 章实验室：深度学习与神经网络

> 本目录把第 2 章组织成“讲义 + 完整实验 + 练习 + 验收 + 真实任务”的学习闭环。

## 学习入口

| 环节 | 作用 | 入口 |
|---|---|---|
| 阅读讲义 | 建立神经网络、梯度、训练循环、CNN 的知识框架 | [第 2 章讲义](../../从AI基础到端侧部署优化/02_深度学习与神经网络.md) |
| 完整实验 | 观察 Autograd、MLP 与 CNN 的参考实现 | [完整实验](./完整实验.ipynb) |
| 独立练习 | 补全关键训练步骤并解释 Shape | [练习](./练习.ipynb) |
| 验收 | 检查梯度、模型质量和 CNN 输出 | [`validate.py`](./validate.py) |
| 真实任务 | 完成轻量 IMU 动作识别原型 | [任务说明](./真实任务_IMU动作识别.md) |

```text
阅读核心概念 → 运行完整实验 → 关闭参考答案 → 完成 TODO
      → 生成练习证据 → 完成真实任务 → 验收提交物
```

完整实验解释一条正确路径；练习改变数据和待完成步骤；真实任务进一步加入类别不均衡、参数量和端侧取舍，不能靠复制输出通过。

## 环境

| 项目 | 要求 |
|---|---|
| Python | 3.10+ |
| GPU | 不需要，默认 CPU |
| 网络 | 安装依赖后不需要 |
| 预计时间 | 完整实验 90～120 分钟；练习 120～180 分钟；真实任务 3～5 小时 |

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m jupyter lab
```

## 验收

练习 Notebook 最后一格检查：梯度是否正确；MLP 是否完成训练且达到基础准确率；CNN 输出 Shape 与数值是否合法。通过后生成 `artifacts/exercise_result.json`。

真实任务完成后运行：

```bash
python validate.py --challenge artifacts/challenge_submission.json
```

真实任务必须同时提交 `challenge_submission.json` 与 `challenge_model.pth`。验收器按 JSON 中的结构描述重建 MLP、严格加载权重，从固定训练集重算标准化统计量，再对固定测试集执行 CPU 推理。逐样本预测、Accuracy、Macro-F1、混淆矩阵和参数量都必须与实际模型一致。

质量线保持 Accuracy ≥ 0.88、Macro-F1 ≥ 0.86、参数量 1～5,000；参数量由加载后的模型核算。工程结论至少 80 个字符，内容质量由人工检查。旧版只有 JSON 的提交需要按更新后的起始代码重新生成。

## 文件结构

```text
02_deep_learning_nn/
├─ README.md
├─ 完整实验.ipynb
├─ 练习.ipynb
├─ validate.py
├─ task_data.py
├─ imu_model.py             # 可配置 MLP 的重建与权重加载
├─ challenge_starter.py
├─ 真实任务_IMU动作识别.md
├─ requirements.txt
├─ tests/test_chapter02_validate.py
└─ artifacts/
```
