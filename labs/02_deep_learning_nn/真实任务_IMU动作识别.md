# 真实任务：端侧 IMU 动作识别原型

## 背景与目标

边缘设备从六轴 IMU 窗口提取 6 个 RMS 特征，需要识别 `静止(0)`、`行走(1)`、`剧烈运动(2)`。请训练一个轻量 MLP，并提交可复算的测试证据。

## 固定规则

1. 使用 `task_data.py`；测试比例 `0.2`、`random_state=42`、`stratify=y`。
2. 只能用训练集计算均值和标准差，禁止数据泄漏。
3. 固定 `torch.manual_seed(42)`，在 CPU 训练。
4. 使用 `imu_model.py` 的 `build_imu_model(model_spec)` 创建 MLP，至少一个隐藏层，输入为 6 维、输出为 3 类。`hidden_sizes` 可配置 1～8 个隐藏层，`activation` 支持 `relu`、`tanh`、`sigmoid`，由你选择宽度和激活函数。
5. 不得手工修改标签或预测；必须提交测试集编号和逐样本预测。

## 最低质量线

| 项目 | 要求 |
|---|---:|
| Accuracy | ≥ 0.88 |
| Macro-F1 | ≥ 0.86 |
| 参数量 | 从实际加载的模型核算，1～5,000，且与申报值一致 |
| 预测一致性 | 逐样本预测与 Checkpoint 重新推理结果完全一致 |

运行 `python challenge_starter.py`，完成模型创建、训练、预测和工程结论四项 TODO。脚本会保存两份交付物：

- `artifacts/challenge_model.pth`：模型的 `state_dict`；
- `artifacts/challenge_submission.json`：包含 `model_spec`、`checkpoint` 文件名、测试集编号、预测、指标、混淆矩阵、参数量和工程结论。

两份文件必须放在同一目录并一起提交。验收器使用 `weights_only=True` 在 CPU 加载权重，按结构描述重建模型，严格检查权重键与 Shape；随后从固定训练集重新计算均值与标准差（`std + 1e-7`），对固定测试集重新推理，核对逐样本预测并重算质量指标。只提交 JSON 或只填写一个参数量不能通过。

旧版仅含 JSON 的提交需要用更新后的起始代码重新生成模型权重和结构描述。该检查能验证提交物的一致性，但不能证明训练过程从未接触测试数据。

生成交付物后运行：

```bash
python validate.py --challenge artifacts/challenge_submission.json
```

提交物还需包含至少 80 个字符的中文工程结论，说明最容易混淆的类别、参数量与模型能力的关系，以及端侧部署下一步应测量的延迟和内存证据。机器检查长度，内容质量由人工检查。

<details><summary>提示 1：训练不收敛</summary>先确认标签是 long 类型，输出层不要加 Softmax，CrossEntropyLoss 会处理 logits。</details>
<details><summary>提示 2：基础网络</summary>尝试 Linear(6, 16) → ReLU → Linear(16, 3)，Adam 学习率 0.01。</details>

## 迁移问题

如果 IMU 的安装方向改变，模型文件和代码都没有变化，为什么准确率仍可能下降？上线前应补充什么验证？
