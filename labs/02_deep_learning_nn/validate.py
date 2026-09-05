"""第 2 章练习与真实任务验收器。"""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from task_data import load_imu_activity_data
from imu_model import load_imu_model

@dataclass(frozen=True)
class CheckResult:
    name: str; passed: bool; message: str

def _safe(name, fn):
    try:
        ok, message = fn(); return CheckResult(name, bool(ok), str(message))
    except Exception as exc:
        return CheckResult(name, False, f"检查时出现 {type(exc).__name__}: {exc}")

def validate_exercise(ns: Mapping[str, Any]) -> dict[str, Any]:
    def gradient():
        value = ns.get("gradient_value")
        ok = value is not None and math.isclose(float(value), 6.0, abs_tol=1e-5)
        return ok, f"复合函数在 x=2 处的梯度={value}，要求为 6。"
    def mlp():
        pred, truth = ns.get("mlp_predictions"), ns.get("mlp_targets")
        if pred is None or truth is None: return False, "缺少 mlp_predictions 或 mlp_targets。"
        score = accuracy_score(np.asarray(truth), np.asarray(pred))
        return score >= .85, f"MLP Accuracy={score:.3f}，要求 ≥0.85。"
    def cnn():
        output = np.asarray(ns.get("cnn_output")) if ns.get("cnn_output") is not None else None
        ok = output is not None and output.shape == (8, 10) and np.isfinite(output).all()
        return ok, f"CNN 输出 Shape={None if output is None else output.shape}，要求 (8, 10) 且数值有限。"
    checks=[_safe("Autograd",gradient),_safe("MLP",mlp),_safe("CNN Shape",cnn)]
    return {"chapter":"02_deep_learning_nn","passed":all(x.passed for x in checks),"checks":[asdict(x) for x in checks]}

def validate_challenge(path: str|Path) -> dict[str, Any]:
    path = Path(path)
    checks = []
    recomputed = {}

    def report():
        return {"chapter": "02_deep_learning_nn", "task": "imu_activity",
                "passed": all(x.passed for x in checks),
                "checks": [asdict(x) for x in checks], "recomputed_metrics": recomputed}

    try:
        sub = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(sub, dict):
            raise ValueError("提交文件必须是 JSON 对象。")
    except (OSError, ValueError) as exc:
        checks.append(CheckResult("提交文件", False, str(exc)))
        return report()

    features, targets, _ = load_imu_activity_data()
    train, expected = train_test_split(np.arange(len(targets)), test_size=.2,
                                       random_state=42, stratify=targets)

    def split_check():
        indices = np.asarray(sub.get("test_indices", []))
        ok = indices.dtype.kind in "iu" and np.array_equal(indices, expected)
        return ok, "测试集编号必须是固定分层划分产生的整数序列，顺序也须一致。"

    def predictions_check():
        pred = np.asarray(sub.get("predictions", []))
        ok = (pred.shape == expected.shape and pred.dtype.kind in "iu"
              and np.isin(pred, [0, 1, 2]).all())
        return ok, f"预测必须是 {len(expected)} 个整数类别 0/1/2。"

    checks.extend([_safe("固定数据划分", split_check), _safe("预测格式", predictions_check)])
    model = None

    def checkpoint_check():
        nonlocal model
        name = sub.get("checkpoint")
        if not isinstance(name, str) or Path(name).name != name:
            return False, "checkpoint 必须是与提交 JSON 同目录的文件名。"
        model = load_imu_model(sub.get("model_spec"), path.parent / name)
        return True, "已按结构描述严格加载 state_dict，权重均为有限数。"

    checks.append(_safe("模型结构与权重", checkpoint_check))
    if checks[-1].passed:
        count = sum(p.numel() for p in model.parameters())
        reported_count = sub.get("parameter_count")
        recomputed["parameter_count"] = count
        checks.append(CheckResult("参数量", type(reported_count) is int
                                  and reported_count == count and 1 <= count <= 5000,
                                  f"权重对应参数量={count}，申报值={reported_count}，上限 5,000。"))

        def inference_check():
            # 预处理统计量由固定训练集重算，不接受提交者提供的测试标签或统计量。
            mean = features[train].mean(0)
            std = features[train].std(0) + 1e-7
            inputs = torch.tensor((features[expected] - mean) / std)
            with torch.no_grad():
                logits = model(inputs).numpy()
            if logits.shape != (len(expected), 3) or not np.isfinite(logits).all():
                return False, "实际模型输出必须为有限的 [120, 3] logits。"
            actual = logits.argmax(axis=1)
            truth = targets[expected]
            recomputed.update(accuracy=float(accuracy_score(truth, actual)),
                              macro_f1=float(f1_score(truth, actual, average="macro")),
                              confusion_matrix=confusion_matrix(truth, actual, labels=[0, 1, 2]).tolist())
            return np.array_equal(np.asarray(sub.get("predictions", [])), actual), "逐样本预测必须与重新加载权重后的 CPU 推理一致。"

        checks.append(_safe("模型预测复算", inference_check))
        if "accuracy" in recomputed:
            def metrics_check():
                reported = sub.get("metrics", {})
                ok = all(math.isclose(float(reported.get(k, -1)), recomputed[k],
                                      rel_tol=0, abs_tol=1e-6) for k in ("accuracy", "macro_f1"))
                return ok, "Accuracy 与 Macro-F1 必须与模型实际推理的重算结果一致。"

            checks.append(_safe("指标可复算", metrics_check))
            acc, macro = recomputed["accuracy"], recomputed["macro_f1"]
            checks.append(CheckResult("模型质量", acc >= .88 and macro >= .86,
                                      f"Accuracy={acc:.3f}（≥0.88），Macro-F1={macro:.3f}（≥0.86）。"))
            checks.append(CheckResult("混淆矩阵", sub.get("confusion_matrix") == recomputed["confusion_matrix"],
                                      f"重算矩阵为 {recomputed['confusion_matrix']}。"))
    conclusion = sub.get("conclusion", "")
    checks.append(CheckResult("工程结论", isinstance(conclusion, str) and len(conclusion.strip()) >= 80,
                              "须提交至少 80 个字符的工程结论；内容质量仍需人工检查。"))
    return report()

def save_report(report,path):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); return target

def print_report(report):
    for x in report["checks"]: print(("✅" if x["passed"] else "❌"),x["name"],x["message"])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--challenge",type=Path); args=parser.parse_args()
    if not args.challenge: parser.print_help(); return 2
    report=validate_challenge(args.challenge); print_report(report); return 0 if report["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
