"""第 2 章真实任务起始代码。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from task_data import load_imu_activity_data
from imu_model import build_imu_model

RANDOM_STATE = 42

def main() -> None:
    torch.manual_seed(RANDOM_STATE)
    features, targets, names = load_imu_activity_data()
    indices = np.arange(len(targets))
    train_i, test_i = train_test_split(indices, test_size=0.2, random_state=RANDOM_STATE, stratify=targets)
    mean, std = features[train_i].mean(0), features[train_i].std(0) + 1e-7
    x_train = torch.tensor((features[train_i] - mean) / std)
    y_train = torch.tensor(targets[train_i], dtype=torch.long)
    x_test = torch.tensor((features[test_i] - mean) / std)
    # TODO 1：选择隐藏层宽度和激活函数，使用 build_imu_model(model_spec) 创建 MLP。
    # 可使用多个隐藏层，例如 [16, 8]；activation 支持 relu/tanh/sigmoid。
    model_spec = {"hidden_sizes": [16], "activation": "relu"}
    model = None
    # TODO 2：使用 CrossEntropyLoss 和优化器完成训练。
    # TODO 3：令 predictions 为 x_test 的类别预测 NumPy 数组。
    predictions = None
    if model is None or predictions is None:
        print("请完成模型、训练和预测 TODO。")
        return
    predictions = np.asarray(predictions)
    truth = targets[test_i]
    metrics = {"accuracy": accuracy_score(truth, predictions), "macro_f1": f1_score(truth, predictions, average="macro")}
    parameter_count = sum(p.numel() for p in model.parameters())
    conclusion = ""  # TODO 4：至少 80 个中文字符的误差与端侧取舍分析。
    path = Path("artifacts/challenge_submission.json"); path.parent.mkdir(exist_ok=True)
    checkpoint = path.parent / "challenge_model.pth"
    torch.save(model.state_dict(), checkpoint)
    submission = {"model_spec": model_spec, "checkpoint": checkpoint.name,
                  "feature_names": names, "test_indices": test_i.tolist(), "predictions": predictions.tolist(),
                  "metrics": metrics, "confusion_matrix": confusion_matrix(truth, predictions, labels=[0,1,2]).tolist(),
                  "parameter_count": parameter_count, "conclusion": conclusion}
    path.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print("已生成", path)

if __name__ == "__main__": main()
