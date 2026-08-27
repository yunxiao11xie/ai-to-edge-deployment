import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LAB_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_DIR))

from task_data import load_device_health_data  # noqa: E402
from validate import build_reference_namespace, validate_challenge, validate_exercise  # noqa: E402


def test_reference_exercise_passes() -> None:
    report = validate_exercise(build_reference_namespace())
    assert report["passed"], report


def test_incomplete_exercise_fails_cleanly() -> None:
    report = validate_exercise({})
    assert not report["passed"]
    assert len(report["checks"]) == 4


def test_challenge_metrics_are_recomputed(tmp_path: Path) -> None:
    features, targets, _ = load_device_health_data()
    indices = np.arange(len(targets))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=targets
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    model.fit(features[train_indices], targets[train_indices])
    predictions = model.predict(features[test_indices])
    truth = targets[test_indices]

    submission = {
        "student": "validator-test",
        "model_name": "LogisticRegression",
        "random_state": 42,
        "test_size": 0.2,
        "test_indices": test_indices.tolist(),
        "predictions": predictions.tolist(),
        "metrics": {
            "accuracy": accuracy_score(truth, predictions),
            "precision_fault": precision_score(truth, predictions, pos_label=1, zero_division=0),
            "recall_fault": recall_score(truth, predictions, pos_label=1, zero_division=0),
            "f1_fault": f1_score(truth, predictions, pos_label=1, zero_division=0),
        },
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "conclusion": "这是用于验收器测试的工程结论。故障预警不能只看准确率，还要检查漏报和误报。端侧部署时应结合模型效果、资源消耗、数据漂移和现场风险继续验证，不能把一次离线结果直接当作生产结论。",
    }
    path = tmp_path / "submission.json"
    path.write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")

    report = validate_challenge(path)
    assert report["passed"], report


def test_fake_reported_metric_is_rejected(tmp_path: Path) -> None:
    features, targets, _ = load_device_health_data()
    indices = np.arange(len(targets))
    _, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=targets
    )
    predictions = targets[test_indices]
    submission = {
        "test_indices": test_indices.tolist(),
        "predictions": predictions.tolist(),
        "metrics": {
            "accuracy": 0.5,
            "precision_fault": 1.0,
            "recall_fault": 1.0,
            "f1_fault": 1.0,
        },
        "confusion_matrix": confusion_matrix(targets[test_indices], predictions, labels=[0, 1]).tolist(),
        "conclusion": "这个提交故意伪造准确率，用于确认验收器确实会根据逐样本预测重新计算所有指标，而不是相信提交文件里的数字。只有能够被原始预测重新计算的结果，才可以被视为可信的实验和工程证据。",
    }
    path = tmp_path / "fake.json"
    path.write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")

    report = validate_challenge(path)
    assert not report["passed"]
    metric_check = next(item for item in report["checks"] if item["name"] == "指标可复算")
    assert not metric_check["passed"]
