import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 隔离第 1、2 章同名 task_data 的模块缓存，支持全仓库一起测试。
data = load_module("chapter02_task_data", LAB / "task_data.py")
with patch.dict(sys.modules, {"task_data": data}):
    validator = load_module("chapter02_validate", LAB / "validate.py")
from imu_model import build_imu_model


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def submission(tmp_path):
    torch.manual_seed(42)
    features, targets, _ = data.load_imu_activity_data()
    train, test = train_test_split(np.arange(len(targets)), test_size=.2, random_state=42, stratify=targets)
    mean, std = features[train].mean(0), features[train].std(0) + 1e-7
    inputs = torch.tensor((features[train] - mean) / std)
    labels = torch.tensor(targets[train], dtype=torch.long)
    spec = {"hidden_sizes": [16], "activation": "relu"}
    model = build_imu_model(spec)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(150):
        optimizer.zero_grad()
        torch.nn.functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor((features[test] - mean) / std)).argmax(1).numpy()
    torch.save(model.state_dict(), tmp_path / "model.pth")
    truth = targets[test]
    sub = {"model_spec": spec, "checkpoint": "model.pth", "test_indices": test.tolist(),
           "predictions": pred.tolist(),
           "metrics": {"accuracy": accuracy_score(truth, pred), "macro_f1": f1_score(truth, pred, average="macro")},
           "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
           "parameter_count": sum(p.numel() for p in model.parameters()),
           "conclusion": "模型按固定训练集进行标准化并学习三类动作。混淆矩阵用于定位容易混淆的动作，参数量由实际权重核算；部署前仍需在目标设备测量延迟、峰值内存、功耗和传感器漂移，并使用真实采集数据复验。"}
    return tmp_path / "submission.json", sub


def validate(submission):
    path, sub = submission
    path.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    return validator.validate_challenge(path)


def test_challenge_reloads_model_and_recomputes_metrics(submission):
    report = validate(submission)
    assert report["passed"], report
    assert report["recomputed_metrics"]["parameter_count"] == 163
    assert report["recomputed_metrics"]["accuracy"] >= .88


@pytest.mark.parametrize("key,value", [
    ("parameter_count", 1), ("parameter_count", True),
    ("predictions", 1), ("predictions", [[0]] * 120), ("predictions", [0.9] * 120),
    ("test_indices", [0] * 120),
    ("metrics", {"accuracy": "invalid", "macro_f1": 1}),
    ("metrics", {"accuracy": float("nan"), "macro_f1": 1}),
    ("confusion_matrix", [[120]]), ("conclusion", "太短"),
    ("model_spec", {"hidden_sizes": [], "activation": "relu"}),
    ("model_spec", {"hidden_sizes": [1000], "activation": "relu"}),
    ("model_spec", {"hidden_sizes": [16], "activation": "identity"}),
    ("model_spec", {"hidden_sizes": [32], "activation": "relu"}),
    ("checkpoint", "missing.pth"),
])
def test_invalid_submission_fails_cleanly(submission, key, value):
    submission[1][key] = value
    assert not validate(submission)["passed"]


def test_consistent_forged_predictions_and_metrics_are_rejected(submission):
    _, targets, _ = data.load_imu_activity_data()
    sub = submission[1]
    truth = targets[sub["test_indices"]]
    pred = np.asarray(sub["predictions"])
    pred[0] = (pred[0] + 1) % 3
    sub["predictions"] = pred.tolist()
    sub["metrics"] = {"accuracy": accuracy_score(truth, pred), "macro_f1": f1_score(truth, pred, average="macro")}
    sub["confusion_matrix"] = confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist()
    report = validate(submission)
    assert not report["passed"]
    assert not next(x for x in report["checks"] if x["name"] == "模型预测复算")["passed"]


@pytest.mark.parametrize("kind", ["garbage", "empty_state", "nonfinite"])
def test_invalid_checkpoint_is_rejected(submission, kind):
    checkpoint = submission[0].parent / "model.pth"
    if kind == "garbage":
        checkpoint.write_text("not a checkpoint", encoding="utf-8")
    elif kind == "empty_state":
        torch.save({}, checkpoint)
    else:
        state = torch.load(checkpoint, weights_only=True)
        state["0.weight"][0, 0] = float("nan")
        torch.save(state, checkpoint)
    assert not validate(submission)["passed"]


def test_exercise_accepts_valid_evidence():
    report = validator.validate_exercise({"gradient_value": 6.0, "mlp_predictions": [0, 1, 1, 0],
                                         "mlp_targets": [0, 1, 1, 0], "cnn_output": np.zeros((8, 10))})
    assert report["passed"], report


def test_incomplete_exercise_fails_cleanly():
    assert not validator.validate_exercise({})["passed"]


@pytest.mark.parametrize("content", ["{", "[]"])
def test_invalid_json_fails_cleanly(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    assert not validator.validate_challenge(path)["passed"]
