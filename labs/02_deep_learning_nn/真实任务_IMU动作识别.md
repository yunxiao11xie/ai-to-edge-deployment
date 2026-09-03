# 真实任务：端侧 IMU 动作识别原型

## 背景与目标

边缘设备从六轴 IMU 窗口提取 6 个 RMS 特征，需要识别 `静止(0)`、`行走(1)`、`剧烈运动(2)`。请训练一个轻量 MLP，并提交可复算的测试证据。

## 固定规则

1. 使用 `task_data.py`；测试比例 `0.2`、`random_state=42`、`stratify=y`。
2. 只能用训练集计算均值和标准差，禁止数据泄漏。
3. 固定 `torch.manual_seed(42)`，在 CPU 训练。
4. 模型必须至少包含一个隐藏层和非线性激活。
5. 不得手工修改标签或预测；必须提交测试集编号和逐样本预测。

## 最低质量线

| 项目 | 要求 |
|---|---:|
| Accuracy | ≥ 0.88 |
| Macro-F1 | ≥ 0.86 |
| 参数量 | 1～5,000 |

运行 `python challenge_starter.py`，完成其中三个 TODO，生成 `artifacts/challenge_submission.json`，然后运行：

```bash
python validate.py --challenge artifacts/challenge_submission.json
```

提交物还需包含至少 80 个中文字符的工程结论，说明最容易混淆的类别、参数量与模型能力的关系，以及端侧部署下一步应测量的延迟和内存证据。

<details><summary>提示 1：训练不收敛</summary>先确认标签是 long 类型，输出层不要加 Softmax，CrossEntropyLoss 会处理 logits。</details>
<details><summary>提示 2：基础网络</summary>尝试 Linear(6, 16) → ReLU → Linear(16, 3)，Adam 学习率 0.01。</details>

## 迁移问题

如果 IMU 的安装方向改变，模型文件和代码都没有变化，为什么准确率仍可能下降？上线前应补充什么验证？
