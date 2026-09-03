"""第 3 章工件与数值一致性验收器。"""
from __future__ import annotations
import argparse, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import onnx
import onnxruntime as ort

@dataclass(frozen=True)
class CheckResult:
    name: str; passed: bool; message: str

def _safe(name, fn):
    try:
        ok,msg=fn(); return CheckResult(name,bool(ok),str(msg))
    except Exception as exc: return CheckResult(name,False,f"检查时出现 {type(exc).__name__}: {exc}")

def validate_exercise(ns: Mapping[str,Any]) -> dict[str,Any]:
    def checkpoint():
        path=Path(str(ns.get("checkpoint_path",""))); return path.is_file() and path.stat().st_size>0,f"Checkpoint={path}。"
    def onnx_model():
        path=Path(str(ns.get("onnx_path",""))); model=onnx.load(path); onnx.checker.check_model(model)
        return len(model.graph.node)>0,f"ONNX 节点数={len(model.graph.node)}。"
    def numerical():
        a=np.asarray(ns.get("pytorch_output")); b=np.asarray(ns.get("ort_output")); error=float(np.max(np.abs(a-b)))
        return a.shape==b.shape and error<=1e-5,f"最大绝对误差={error:.3e}，要求 ≤1e-5。"
    checks=[_safe("Checkpoint",checkpoint),_safe("ONNX Graph",onnx_model),_safe("数值一致性",numerical)]
    return {"chapter":"03_pytorch_frameworks","passed":all(x.passed for x in checks),"checks":[asdict(x) for x in checks]}

def validate_challenge(manifest_path: str|Path) -> dict[str,Any]:
    path=Path(manifest_path); sub=json.loads(path.read_text(encoding="utf-8")); base=path.parent
    checkpoint=base/str(sub.get("checkpoint","")); onnx_path=base/str(sub.get("onnx_model","")); checks=[]
    checks.append(CheckResult("Checkpoint 工件",checkpoint.is_file() and checkpoint.stat().st_size>0,f"Checkpoint={checkpoint.name}。"))
    def graph_check():
        model=onnx.load(onnx_path); onnx.checker.check_model(model); dim=model.graph.input[0].type.tensor_type.shape.dim[0]
        dynamic=bool(dim.dim_param) and not dim.HasField("dim_value")
        return dynamic and len(model.graph.node)>0,f"节点数={len(model.graph.node)}，Dynamic Batch={dynamic}。"
    checks.append(_safe("ONNX 工件与动态图",graph_check))
    def inference_check():
        sample=np.asarray(sub.get("sample_input"),dtype=np.float32); reference=np.asarray(sub.get("torch_output"),dtype=np.float32)
        session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"]); actual=session.run([str(sub.get("output_name","output"))],{str(sub.get("input_name","input")):sample})[0]
        reported=np.asarray(sub.get("ort_output"),dtype=np.float32); error=float(np.max(np.abs(reference-actual)))
        honest=np.allclose(reported,actual,atol=1e-6,rtol=1e-6) and math.isclose(float(sub.get("max_abs_error",-1)),error,abs_tol=1e-6)
        shape_ok=sample.shape==(4,8) and actual.shape==(4,3)
        return shape_ok and honest and error<=1e-5,f"输入={sample.shape}，输出={actual.shape}，重算误差={error:.3e}。"
    checks.append(_safe("Runtime 数值复算",inference_check))
    conclusion=str(sub.get("conclusion","")).strip(); checks.append(CheckResult("工程结论",len(conclusion)>=100,f"结论长度 {len(conclusion)}，要求 ≥100。"))
    return {"chapter":"03_pytorch_frameworks","task":"model_delivery_audit","passed":all(x.passed for x in checks),"checks":[asdict(x) for x in checks]}

def save_report(report,path):
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); return target

def print_report(report):
    for x in report["checks"]: print(("✅" if x["passed"] else "❌"),x["name"],x["message"])

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--challenge",type=Path); args=parser.parse_args()
    if not args.challenge: parser.print_help(); return 2
    report=validate_challenge(args.challenge); print_report(report); return 0 if report["passed"] else 1

if __name__=="__main__": raise SystemExit(main())
