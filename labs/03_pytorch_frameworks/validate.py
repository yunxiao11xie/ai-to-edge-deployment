"""第 3 章工件与数值一致性验收器。"""
from __future__ import annotations
import argparse, copy, json, math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import onnx
import onnxruntime as ort
import torch
from delivery_model import DeliveryMLP

@dataclass(frozen=True)
class CheckResult:
    name: str; passed: bool; message: str

def _safe(name, fn):
    try:
        ok,msg=fn(); return CheckResult(name,bool(ok),str(msg))
    except Exception as exc: return CheckResult(name,False,f"检查时出现 {type(exc).__name__}: {exc}")


def _load_checkpoint(model, path):
    restored = copy.deepcopy(model).cpu().eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    if not all(torch.isfinite(value).all().item() for value in restored.state_dict().values()):
        raise ValueError("Checkpoint 包含非有限权重。")
    return restored


def _same_output(reported, actual):
    value = np.asarray(reported, dtype=np.float32)
    return (value.shape == actual.shape and np.isfinite(value).all()
            and np.allclose(value, actual, atol=1e-6, rtol=0))


def _run_pair(model, session, sample):
    with torch.no_grad():
        reference = model(torch.from_numpy(sample)).numpy()
    actual = session.run(["output"], {"input": sample})[0]
    if reference.shape != actual.shape or reference.size == 0:
        raise ValueError("PyTorch 和 ORT 输出 Shape 不一致或为空。")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("PyTorch 和 ORT 输出必须全部为有限数。")
    error = float(np.max(np.abs(reference - actual)))
    return reference, actual, error


def _graph_contract(path, input_features, output_features):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1 or not model.graph.node:
        return False, "要求单输入、单输出且图非空。"
    if next((x.version for x in model.opset_import if x.domain == ""), None) != 17:
        return False, "要求 ONNX 默认域 Opset 17。"
    for value, name, features in [(model.graph.input[0], "input", input_features),
                                  (model.graph.output[0], "output", output_features)]:
        tensor = value.type.tensor_type
        dims = tensor.shape.dim
        if (value.name != name or tensor.elem_type != onnx.TensorProto.FLOAT or len(dims) != 2
                or not dims[0].dim_param or dims[0].HasField("dim_value")
                or dims[1].dim_value != features):
            return False, f"{name} 必须为 float32 [动态 Batch, {features}]。"
    input_batch = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    output_batch = model.graph.output[0].type.tensor_type.shape.dim[0].dim_param
    if input_batch != output_batch:
        return False, "输入和输出必须使用同一个动态 Batch 符号。"
    return True, "输入/输出名称、float32、动态 Batch、特征维和 Opset 17 均符合契约。"


def validate_exercise(ns: Mapping[str,Any]) -> dict[str,Any]:
    restored = None
    def checkpoint():
        nonlocal restored
        restored = _load_checkpoint(ns["model"], ns["checkpoint_path"])
        sample = ns["sample"].detach().cpu()
        with torch.no_grad():
            expected = copy.deepcopy(ns["model"]).cpu().eval()(sample).numpy()
            actual = restored(sample).numpy()
        return _same_output(actual, expected), "重载权重的输出必须与当前实验模型一致。"
    def onnx_model():
        sample = ns["sample"].detach().cpu()
        with torch.no_grad():
            output = restored(sample)
        return _graph_contract(ns["onnx_path"], sample.shape[1], output.shape[1])
    def numerical():
        sample = ns["sample"].detach().cpu().numpy()
        session = ort.InferenceSession(str(ns["onnx_path"]), providers=["CPUExecutionProvider"])
        reference, actual, error = _run_pair(restored, session, sample)
        honest = _same_output(ns.get("pytorch_output"), reference) and _same_output(ns.get("ort_output"), actual)
        return honest and error <= 1e-5, f"重载权重与 ORT 的最大绝对误差={error:.3e}；报告输出须与实际输出一致。"
    checks=[_safe("Checkpoint",checkpoint),_safe("ONNX Graph",onnx_model),_safe("数值一致性",numerical)]
    return {"chapter":"03_pytorch_frameworks","passed":all(x.passed for x in checks),"checks":[asdict(x) for x in checks]}

def validate_challenge(manifest_path: str|Path) -> dict[str,Any]:
    path = Path(manifest_path)
    checks = []
    recomputed = {}

    def report():
        return {"chapter": "03_pytorch_frameworks", "task": "model_delivery_audit",
                "passed": all(x.passed for x in checks), "checks": [asdict(x) for x in checks],
                "recomputed_metrics": recomputed}

    try:
        sub = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(sub, dict):
            raise ValueError("Manifest 必须是 JSON 对象。")
        for key in ("checkpoint", "onnx_model"):
            name = sub.get(key)
            if not isinstance(name, str) or not name or Path(name).name != name:
                raise ValueError(f"{key} 必须是与 Manifest 同目录的文件名。")
    except (OSError, ValueError) as exc:
        checks.append(CheckResult("提交文件", False, str(exc)))
        return report()

    checkpoint = path.parent / sub["checkpoint"]
    onnx_path = path.parent / sub["onnx_model"]
    model = None

    def checkpoint_check():
        nonlocal model
        model = _load_checkpoint(DeliveryMLP(), checkpoint)
        recomputed["parameter_count"] = sum(p.numel() for p in model.parameters())
        return True, "按固定 8→16→3 MLP 严格加载 state_dict，权重均为有限数。"

    checks.append(_safe("Checkpoint 工件", checkpoint_check))
    def graph_check():
        return _graph_contract(onnx_path, 8, 3)
    checks.append(_safe("ONNX 工件与动态图",graph_check))

    sample = torch.linspace(-1, 1, 32, dtype=torch.float32).reshape(4, 8).numpy()

    def manifest_check():
        expected = {"input_name": "input", "output_name": "output", "input_shape": [4, 8],
                    "input_dtype": "float32", "opset": 17, "runtime": "onnxruntime"}
        for key, value in expected.items():
            if type(sub.get(key)) is not type(value) or sub[key] != value:
                return False, f"Manifest 的 {key} 必须为 {value}。"
        submitted_sample = np.asarray(sub.get("sample_input"), dtype=np.float32)
        return np.array_equal(submitted_sample, sample), "sample_input 必须等于固定的 linspace(-1, 1, 32).reshape(4, 8)。"

    checks.append(_safe("Manifest 输入契约", manifest_check))
    def inference_check():
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        reference, actual, error = _run_pair(model, session, sample)
        recomputed["max_abs_error"] = error
        honest = (_same_output(sub.get("torch_output"), reference)
                  and _same_output(sub.get("ort_output"), actual)
                  and math.isclose(float(sub.get("max_abs_error", -1)), error, rel_tol=0, abs_tol=1e-6))
        if not honest or error > 1e-5 or actual.shape != (4, 3):
            return False, f"固定输入重算误差={error:.3e}；两份报告输出必须与实际推理一致。"
        # 不只检查符号维，实际运行未用于导出的 Batch 和不同输入。
        batch_errors = {}
        for batch in (1, 7):
            probe = np.random.default_rng(42 + batch).normal(size=(batch, 8)).astype(np.float32)
            _, output, probe_error = _run_pair(model, session, probe)
            batch_errors[str(batch)] = probe_error
            if output.shape != (batch, 3) or probe_error > 1e-5:
                return False, f"Batch={batch} 的输出或误差不符合要求：{probe_error:.3e}。"
        recomputed["dynamic_batch_errors"] = batch_errors
        return True, f"固定输入及 Batch=1/7 均通过权重→PyTorch→ORT 复算；误差={error:.3e}。"
    checks.append(_safe("Runtime 数值复算",inference_check))
    conclusion = sub.get("conclusion", "")
    checks.append(CheckResult("工程结论", isinstance(conclusion, str) and len(conclusion.strip()) >= 100,
                              "须提交至少 100 个字符的工程结论；内容质量仍需人工检查。"))
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
