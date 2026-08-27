"""第 1 章练习与真实任务的机器验收器。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_diabetes, load_iris, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from task_data import load_device_health_data


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str


def _first(namespace: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        value = namespace.get(name)
        if value is not None:
            return value
    return None


def _safe_check(name: str, function: Any) -> CheckResult:
    try:
        passed, message = function()
        return CheckResult(name=name, passed=bool(passed), message=str(message))
    except Exception as exc:  # 学习阶段需要返回可理解的失败信息
        return CheckResult(name=name, passed=False, message=f"检查时出现 {type(exc).__name__}: {exc}")


def validate_exercise(namespace: Mapping[str, Any]) -> dict[str, Any]:
    """检查练习 Notebook 当前命名空间中的四项核心成果。"""

    def check_regression() -> tuple[bool, str]:
        predictions = _first(namespace, "reg_y_pred")
        targets = _first(namespace, "reg_y_test")
        model = _first(namespace, "reg_model")
        if model is None or predictions is None or targets is None:
            return False, "缺少 reg_model、reg_y_pred 或 reg_y_test。"
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        if predictions.shape != targets.shape:
            return False, f"预测形状 {predictions.shape} 与标签形状 {targets.shape} 不一致。"
        score = r2_score(targets, predictions)
        passed = np.isfinite(predictions).all() and score >= 0.35
        return passed, f"R²={score:.3f}，要求预测均为有限数且 R²≥0.35。"

    def check_classification() -> tuple[bool, str]:
        predictions = _first(namespace, "clf_y_pred", "iris_y_pred")
        targets = _first(namespace, "clf_y_test", "iris_y_test")
        model = _first(namespace, "clf_model", "iris_model")
        if model is None or predictions is None or targets is None:
            return False, "缺少分类模型、预测结果或测试标签。"
        predictions = np.asarray(predictions)
        targets = np.asarray(targets)
        if predictions.shape != targets.shape:
            return False, f"预测形状 {predictions.shape} 与标签形状 {targets.shape} 不一致。"
        score = accuracy_score(targets, predictions)
        return score >= 0.85, f"Accuracy={score:.3f}，要求 Accuracy≥0.85。"

    def check_clustering() -> tuple[bool, str]:
        features = _first(namespace, "cluster_X")
        labels = _first(namespace, "cluster_labels")
        model = _first(namespace, "cluster_model")
        if model is None or features is None or labels is None:
            return False, "缺少 cluster_model、cluster_X 或 cluster_labels。"
        features = np.asarray(features)
        labels = np.asarray(labels)
        if len(features) != len(labels):
            return False, "每个样本必须恰好对应一个簇编号。"
        unique_clusters = np.unique(labels)
        if len(unique_clusters) != 3:
            return False, f"当前得到 {len(unique_clusters)} 个簇，要求得到 3 个簇。"
        score = silhouette_score(features, labels)
        return score >= 0.50, f"轮廓系数={score:.3f}，要求轮廓系数≥0.50。"

    def check_q_learning() -> tuple[bool, str]:
        q_table = _first(namespace, "rl_q_table", "q_table")
        successes = _first(namespace, "rl_successes", "success_history")
        if q_table is None or successes is None:
            return False, "缺少 Q 表或训练成功记录。"
        q_table = np.asarray(q_table)
        successes = np.asarray(successes)
        if q_table.shape != (6, 2):
            return False, f"Q 表形状为 {q_table.shape}，要求为 (6, 2)。"

        state = 0
        path = [state]
        for _ in range(10):
            action = int(np.argmax(q_table[state]))
            state = max(0, state - 1) if action == 0 else min(5, state + 1)
            path.append(state)
            if state == 5:
                break

        recent_success = float(successes[-100:].mean()) if len(successes) >= 100 else float(successes.mean())
        passed = path[-1] == 5 and recent_success >= 0.80
        return passed, f"贪心路径={'→'.join(map(str, path))}，最近成功率={recent_success:.3f}。"

    checks = [
        _safe_check("回归", check_regression),
        _safe_check("分类", check_classification),
        _safe_check("聚类", check_clustering),
        _safe_check("强化学习", check_q_learning),
    ]
    return {
        "chapter": "01_ai_basics_ml",
        "passed": all(item.passed for item in checks),
        "checks": [asdict(item) for item in checks],
    }


def print_report(report: Mapping[str, Any]) -> None:
    print("\n========== 第 1 章统一验收 ==========")
    for item in report["checks"]:
        icon = "✅" if item["passed"] else "❌"
        print(f"{icon} {item['name']}：{item['message']}")
    print("-------------------------------------")
    print("✅ 全部通过" if report["passed"] else "⏸️ 尚未全部通过，请根据上面的证据继续修改。")


def save_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def validate_challenge(submission_path: str | Path) -> dict[str, Any]:
    """根据真实预测重新计算指标，验收设备故障预警任务。"""

    path = Path(submission_path)
    submission = json.loads(path.read_text(encoding="utf-8"))
    _, targets, _ = load_device_health_data()
    all_indices = np.arange(len(targets))
    _, expected_test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )

    submitted_indices = np.asarray(submission.get("test_indices", []), dtype=np.int64)
    predictions = np.asarray(submission.get("predictions", []), dtype=np.int64)
    conclusion = str(submission.get("conclusion", "")).strip()
    reported = submission.get("metrics", {})

    def check_split() -> tuple[bool, str]:
        passed = np.array_equal(submitted_indices, expected_test_indices)
        return passed, "测试集编号与 random_state=42、test_size=0.2、stratify=y 的固定划分一致。" if passed else "测试集编号不匹配，请按任务要求重新划分。"

    def check_predictions() -> tuple[bool, str]:
        valid_values = set(np.unique(predictions).tolist()).issubset({0, 1})
        passed = len(predictions) == len(expected_test_indices) and valid_values
        return passed, f"收到 {len(predictions)} 个预测，要求 {len(expected_test_indices)} 个 0/1 预测。"

    checks = [
        _safe_check("固定数据划分", check_split),
        _safe_check("预测格式", check_predictions),
    ]

    if checks[0].passed and checks[1].passed:
        truth = targets[submitted_indices]
        accuracy = accuracy_score(truth, predictions)
        precision = precision_score(truth, predictions, pos_label=1, zero_division=0)
        recall = recall_score(truth, predictions, pos_label=1, zero_division=0)
        f1 = f1_score(truth, predictions, pos_label=1, zero_division=0)
        matrix = confusion_matrix(truth, predictions, labels=[0, 1]).tolist()

        def check_metrics() -> tuple[bool, str]:
            expected = {
                "accuracy": accuracy,
                "precision_fault": precision,
                "recall_fault": recall,
                "f1_fault": f1,
            }
            for key, value in expected.items():
                reported_value = reported.get(key)
                if reported_value is None or not math.isclose(float(reported_value), value, abs_tol=1e-6):
                    return False, f"{key} 报告值与根据 predictions 重算的结果不一致。"
            return True, "提交的四项指标都能由真实预测重新计算得到。"

        def check_quality() -> tuple[bool, str]:
            passed = accuracy >= 0.78 and recall >= 0.70 and f1 >= 0.68
            return passed, f"Accuracy={accuracy:.3f}，故障 Recall={recall:.3f}，故障 F1={f1:.3f}。"

        def check_matrix() -> tuple[bool, str]:
            passed = submission.get("confusion_matrix") == matrix
            return passed, f"重算混淆矩阵为 {matrix}。"

        checks.extend(
            [
                _safe_check("指标可复算", check_metrics),
                _safe_check("模型质量", check_quality),
                _safe_check("混淆矩阵", check_matrix),
            ]
        )
    else:
        accuracy = precision = recall = f1 = None
        matrix = None

    checks.append(
        CheckResult(
            name="工程结论",
            passed=len(conclusion) >= 80,
            message=f"结论长度 {len(conclusion)} 个字符，要求至少 80 个字符并解释误报、漏报与部署取舍。",
        )
    )

    return {
        "chapter": "01_ai_basics_ml",
        "task": "device_health_warning",
        "passed": all(item.passed for item in checks),
        "checks": [asdict(item) for item in checks],
        "recomputed_metrics": {
            "accuracy": accuracy,
            "precision_fault": precision,
            "recall_fault": recall,
            "f1_fault": f1,
            "confusion_matrix": matrix,
        },
    }


def build_reference_namespace() -> dict[str, Any]:
    """生成一组参考结果，用于确认验收器自身没有损坏。"""

    namespace: dict[str, Any] = {}

    reg_x, reg_y = load_diabetes(return_X_y=True)
    reg_x_train, reg_x_test, reg_y_train, reg_y_test = train_test_split(
        reg_x, reg_y, test_size=0.2, random_state=42
    )
    reg_model = LinearRegression().fit(reg_x_train, reg_y_train)
    namespace.update(
        reg_model=reg_model,
        reg_y_pred=reg_model.predict(reg_x_test),
        reg_y_test=reg_y_test,
    )

    iris = load_iris()
    clf_x_train, clf_x_test, clf_y_train, clf_y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target
    )
    clf_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf_model.fit(clf_x_train, clf_y_train)
    namespace.update(
        clf_model=clf_model,
        clf_y_pred=clf_model.predict(clf_x_test),
        clf_y_test=clf_y_test,
    )

    cluster_x, reference_cluster_labels = make_blobs(
        n_samples=300, centers=3, cluster_std=0.85, random_state=42
    )
    namespace.update(
        # 自检关注验收规则本身；完整 Notebook 会真实运行 K-Means。
        cluster_model="reference-cluster-model",
        cluster_X=cluster_x,
        cluster_labels=reference_cluster_labels,
    )

    q_table = np.array(
        [[0.0, 7.4], [0.0, 7.9], [0.0, 8.4], [0.0, 9.0], [0.0, 10.0], [0.0, 0.0]]
    )
    namespace.update(rl_q_table=q_table, rl_successes=np.ones(500, dtype=np.int64))
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser(description="第 1 章机器学习实验验收器")
    parser.add_argument("--challenge", type=Path, help="真实任务提交 JSON 路径")
    parser.add_argument("--self-test", action="store_true", help="使用内置参考结果检查验收器")
    args = parser.parse_args()

    if args.challenge:
        report = validate_challenge(args.challenge)
    elif args.self_test:
        report = validate_exercise(build_reference_namespace())
    else:
        parser.print_help()
        return 2

    print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
