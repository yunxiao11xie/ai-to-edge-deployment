"""第 3 章真实模型交付任务起始代码。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

class DeliveryMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,3))
    def forward(self,x): return self.net(x)

def main():
    torch.manual_seed(42); output_dir=Path("artifacts"); output_dir.mkdir(exist_ok=True)
    model=DeliveryMLP().eval(); sample=torch.linspace(-1,1,32).reshape(4,8)
    checkpoint=output_dir/"challenge_model.pth"; onnx_path=output_dir/"challenge_model.onnx"
    # TODO 1：保存 model.state_dict()。
    # TODO 2：导出 ONNX，输入/输出命名为 input/output，并设置 Dynamic Batch，opset_version=17。
    # TODO 3：分别取得 PyTorch 和 ONNX Runtime 输出。
    torch_output=None; ort_output=None
    if torch_output is None or ort_output is None or not checkpoint.exists() or not onnx_path.exists():
        print("请完成保存、导出和推理 TODO。"); return
    torch_output=np.asarray(torch_output); ort_output=np.asarray(ort_output); max_error=float(np.max(np.abs(torch_output-ort_output)))
    manifest={"checkpoint":checkpoint.name,"onnx_model":onnx_path.name,"input_name":"input","output_name":"output",
              "input_shape":[4,8],"input_dtype":"float32","opset":17,"runtime":"onnxruntime",
              "sample_input":sample.numpy().tolist(),"torch_output":torch_output.tolist(),"ort_output":ort_output.tolist(),
              "max_abs_error":max_error,"conclusion":""} # TODO 4：至少 100 个中文字符。
    path=output_dir/"delivery_manifest.json"; path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); print("已生成",path)

if __name__=="__main__": main()
