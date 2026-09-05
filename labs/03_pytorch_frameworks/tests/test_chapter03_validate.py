import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
spec = importlib.util.spec_from_file_location("chapter03_validate", LAB / "validate.py")
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)
from delivery_model import DeliveryMLP


@pytest.fixture
def delivery(tmp_path):
    torch.manual_seed(42)
    model = DeliveryMLP().eval()
    sample = torch.linspace(-1, 1, 32).reshape(4, 8)
    checkpoint, graph = tmp_path / "model.pth", tmp_path / "model.onnx"
    torch.save(model.state_dict(), checkpoint)
    torch.onnx.export(model, sample, graph, input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        reference = model(sample).numpy()
    actual = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"]).run(["output"], {"input": sample.numpy()})[0]
    sub = {"checkpoint": checkpoint.name, "onnx_model": graph.name,
           "input_name": "input", "output_name": "output", "input_shape": [4, 8],
           "input_dtype": "float32", "opset": 17, "runtime": "onnxruntime",
           "sample_input": sample.numpy().tolist(), "torch_output": reference.tolist(),
           "ort_output": actual.tolist(), "max_abs_error": float(np.max(np.abs(reference - actual))),
           "conclusion": "模型包明确记录输入名称、形状和数据类型，并同时交付权重与 ONNX 图。上线前需要固定框架、Opset 和 Runtime 版本，排查目标执行提供器发生 CPU Fallback 的情况；数值误差必须在约定容差内，且仍需在真实目标设备测量延迟、内存、功耗、预热和数据搬运成本。"}
    ns = {"model": model, "sample": sample, "checkpoint_path": checkpoint, "onnx_path": graph,
          "pytorch_output": reference, "ort_output": actual}
    return tmp_path / "manifest.json", sub, ns


def validate(delivery):
    path, sub, _ = delivery
    path.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
    return validator.validate_challenge(path)


def test_exercise_and_delivery_pass(delivery):
    exercise = validator.validate_exercise(delivery[2])
    assert exercise["passed"], exercise
    report = validate(delivery)
    assert report["passed"], report
    assert set(report["recomputed_metrics"]["dynamic_batch_errors"]) == {"1", "7"}


@pytest.mark.parametrize("key,value", [
    ("sample_input", [[0.] * 8] * 4), ("torch_output", [[0.] * 3] * 4),
    ("torch_output", [0., 0., 0.]), ("ort_output", [[float("nan")] * 3] * 4),
    ("max_abs_error", 1.), ("max_abs_error", "invalid"),
    ("input_name", "wrong"), ("input_shape", [1, 8]), ("input_dtype", "float64"),
    ("opset", 18), ("runtime", "other"), ("checkpoint", "missing.pth"),
    ("onnx_model", "missing.onnx"), ("conclusion", "太短"),
])
def test_invalid_manifest_is_rejected(delivery, key, value):
    delivery[1][key] = value
    assert not validate(delivery)["passed"]


@pytest.mark.parametrize("kind", ["garbage", "empty_state", "nonfinite", "different_weights"])
def test_bad_or_mismatched_checkpoint_is_rejected(delivery, kind):
    checkpoint = delivery[2]["checkpoint_path"]
    if kind == "garbage":
        checkpoint.write_text("not a checkpoint", encoding="utf-8")
    elif kind == "empty_state":
        torch.save({}, checkpoint)
    else:
        state = torch.load(checkpoint, weights_only=True)
        state["net.2.bias"] += 1
        if kind == "nonfinite":
            state["net.0.weight"][0, 0] = float("inf")
        torch.save(state, checkpoint)
    assert not validate(delivery)["passed"]
    assert not validator.validate_exercise(delivery[2])["passed"]


@pytest.mark.parametrize("change", ["static_input", "static_output", "different_batch_symbol", "opset", "wrong_output_width"])
def test_graph_contract_is_checked_from_artifact(delivery, change):
    path = delivery[2]["onnx_path"]
    graph = onnx.load(path)
    if change == "static_input":
        graph.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 4
    elif change == "static_output":
        graph.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 4
    elif change == "different_batch_symbol":
        graph.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "other_batch"
    elif change == "opset":
        graph.opset_import[0].version = 16
    else:
        graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 4
    onnx.save(graph, path)
    assert not validate(delivery)["passed"]


def test_actual_dynamic_batch_execution_is_required(delivery):
    # 声明动态维，但内部强制 reshape 为 [4, 3]：只看图的符号维无法识别。
    path = delivery[2]["onnx_path"]
    graph = onnx.load(path)
    for node in graph.graph.node:
        for i, name in enumerate(node.output):
            if name == "output":
                node.output[i] = "before_reshape"
    shape = onnx.numpy_helper.from_array(np.array([4, 3], dtype=np.int64), name="fixed_shape")
    graph.graph.initializer.append(shape)
    graph.graph.node.append(onnx.helper.make_node("Reshape", ["before_reshape", "fixed_shape"], ["output"]))
    onnx.save(graph, path)
    report = validate(delivery)
    assert not report["passed"]
    checks = {x["name"]: x["passed"] for x in report["checks"]}
    assert checks["ONNX 工件与动态图"]
    assert not checks["Runtime 数值复算"]


def test_exercise_does_not_accept_fabricated_arrays(delivery):
    ns = delivery[2]
    ns["pytorch_output"] = ns["ort_output"] = np.zeros((4, 3))
    assert not validator.validate_exercise(ns)["passed"]


def test_incomplete_exercise_fails_cleanly():
    assert not validator.validate_exercise({})["passed"]


@pytest.mark.parametrize("content", ["{", "[]"])
def test_invalid_json_fails_cleanly(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    assert not validator.validate_challenge(path)["passed"]
