import importlib.util, json, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
LAB=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(LAB))
spec=importlib.util.spec_from_file_location("chapter03_validate",LAB/"validate.py")
validator=importlib.util.module_from_spec(spec); sys.modules[spec.name]=validator; spec.loader.exec_module(validator)
validate_challenge, validate_exercise=validator.validate_challenge, validator.validate_exercise

class Tiny(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,3))
    def forward(self,x): return self.net(x)

def make_artifacts(folder):
    torch.manual_seed(42); model=Tiny().eval(); sample=torch.linspace(-1,1,32).reshape(4,8); checkpoint=folder/"model.pth"; graph=folder/"model.onnx"
    torch.save(model.state_dict(),checkpoint)
    torch.onnx.export(model,sample,graph,input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"batch"},"output":{0:"batch"}},opset_version=17)
    with torch.no_grad(): ref=model(sample).numpy()
    actual=ort.InferenceSession(str(graph),providers=["CPUExecutionProvider"]).run(["output"],{"input":sample.numpy()})[0]
    return checkpoint,graph,sample.numpy(),ref,actual

def test_exercise_and_delivery_pass(tmp_path):
    checkpoint,graph,sample,ref,actual=make_artifacts(tmp_path)
    assert validate_exercise({"checkpoint_path":checkpoint,"onnx_path":graph,"pytorch_output":ref,"ort_output":actual})["passed"]
    sub={"checkpoint":checkpoint.name,"onnx_model":graph.name,"input_name":"input","output_name":"output","sample_input":sample.tolist(),"torch_output":ref.tolist(),"ort_output":actual.tolist(),"max_abs_error":float(np.max(np.abs(ref-actual))),"conclusion":"模型包明确记录输入名称、形状和数据类型，并同时交付权重与 ONNX 图。上线前需要固定框架、Opset 和 Runtime 版本，排查目标执行提供器发生 CPU Fallback 的情况；数值误差必须在约定容差内，且仍需在真实目标设备测量延迟、内存、功耗、预热和数据搬运成本。"}
    manifest=tmp_path/"manifest.json"; manifest.write_text(json.dumps(sub,ensure_ascii=False),encoding="utf-8")
    assert validate_challenge(manifest)["passed"]

def test_incomplete_exercise_fails_cleanly(): assert not validate_exercise({})["passed"]
