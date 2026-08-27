"""真实任务起始代码：端侧设备故障预警原型。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from task_data import load_device_health_data


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


RANDOM_STATE = 42
TEST_SIZE = 0.2


def main() -> None:
    features, targets, feature_names = load_device_health_data()
    all_indices = np.arange(len(targets))

    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    x_train = features[train_indices]
    x_test = features[test_indices]
    y_train = targets[train_indices]
    y_test = targets[test_indices]

    print("特征：", feature_names)
    print("训练集：", x_train.shape, "测试集：", x_test.shape)
    print("训练集正常/故障：", np.bincount(y_train))
    print("测试集正常/故障：", np.bincount(y_test))

    # TODO 1：建立分类模型。建议先从 StandardScaler + LogisticRegression 开始。
    model = None

    # TODO 2：训练模型，并对 x_test 生成 0/1 预测。
    predictions = None

    if model is None or predictions is None:
        print("\n📝 请先完成模型创建、训练和预测。")
        return

    predictions = np.asarray(predictions, dtype=np.int64)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_fault": precision_score(y_test, predictions, pos_label=1, zero_division=0),
        "recall_fault": recall_score(y_test, predictions, pos_label=1, zero_division=0),
        "f1_fault": f1_score(y_test, predictions, pos_label=1, zero_division=0),
    }
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()

    # TODO 3：用至少 80 个中文字符解释误报、漏报和端侧部署时的模型取舍。
    conclusion = ""

    submission = {
        "student": "请填写你的名字或代号",
        "model_name": type(model).__name__,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "test_indices": test_indices.tolist(),
        "predictions": predictions.tolist(),
        "metrics": metrics,
        "confusion_matrix": matrix,
        "conclusion": conclusion,
    }

    output_path = Path("artifacts/challenge_submission.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n已生成提交物：", output_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("混淆矩阵：", matrix)


if __name__ == "__main__":
    main()
