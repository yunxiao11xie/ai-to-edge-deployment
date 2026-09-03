"""第 2 章练习与真实任务验收器。"""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from task_data import load_imu_activity_data

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
    sub=json.loads(Path(path).read_text(encoding="utf-8")); _, targets, _=load_imu_activity_data()
    indices=np.arange(len(targets)); _, expected=train_test_split(indices,test_size=.2,random_state=42,stratify=targets)
    submitted=np.asarray(sub.get("test_indices",[]),dtype=np.int64); pred=np.asarray(sub.get("predictions",[]),dtype=np.int64)
    checks=[]
    checks.append(CheckResult("固定数据划分",np.array_equal(submitted,expected),"测试集编号必须来自固定分层划分。"))
    checks.append(CheckResult("预测格式",len(pred)==len(expected) and set(np.unique(pred)).issubset({0,1,2}),f"收到 {len(pred)} 个预测。"))
    if checks[0].passed and checks[1].passed:
        truth=targets[submitted]; acc=accuracy_score(truth,pred); macro=f1_score(truth,pred,average="macro"); matrix=confusion_matrix(truth,pred,labels=[0,1,2]).tolist()
        reported=sub.get("metrics",{})
        metrics_ok=all(math.isclose(float(reported.get(k,-1)),v,abs_tol=1e-6) for k,v in {"accuracy":acc,"macro_f1":macro}.items())
        checks += [CheckResult("指标可复算",metrics_ok,"Accuracy 与 Macro-F1 必须能由预测复算。"),
                   CheckResult("模型质量",acc>=.88 and macro>=.86,f"Accuracy={acc:.3f}，Macro-F1={macro:.3f}。"),
                   CheckResult("混淆矩阵",sub.get("confusion_matrix")==matrix,f"重算矩阵为 {matrix}。")]
    count=sub.get("parameter_count",0); checks.append(CheckResult("参数量",isinstance(count,int) and 1<=count<=5000,f"参数量={count}，要求 1～5000。"))
    conclusion=str(sub.get("conclusion","")).strip(); checks.append(CheckResult("工程结论",len(conclusion)>=80,f"结论长度 {len(conclusion)}，要求 ≥80。"))
    return {"chapter":"02_deep_learning_nn","task":"imu_activity","passed":all(x.passed for x in checks),"checks":[asdict(x) for x in checks]}

def save_report(report,path):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); return target

def print_report(report):
    for x in report["checks"]: print(("✅" if x["passed"] else "❌"),x["name"],x["message"])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--challenge",type=Path); args=parser.parse_args()
    if not args.challenge: parser.print_help(); return 2
    report=validate_challenge(args.challenge); print_report(report); return 0 if report["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
