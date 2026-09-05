"""用可审计的结构描述重建 IMU MLP，不加载提交者的 Python 代码。"""

import torch
from torch import nn


def build_imu_model(spec: dict) -> nn.Sequential:
    """支持一个或多个隐藏层，每个隐藏层使用同一种非线性激活。"""
    if not isinstance(spec, dict):
        raise ValueError("model_spec 必须包含 hidden_sizes 和 activation。")
    hidden = spec.get("hidden_sizes")
    activations = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    activation = spec.get("activation")
    if (not isinstance(hidden, list) or not 1 <= len(hidden) <= 8
            or any(type(width) is not int or width <= 0 for width in hidden)):
        raise ValueError("hidden_sizes 必须是含 1～8 个正整数的列表。")
    if not isinstance(activation, str) or activation not in activations:
        raise ValueError("activation 必须为 relu、tanh 或 sigmoid。")
    widths = [6, *hidden, 3]
    count = sum((a + 1) * b for a, b in zip(widths, widths[1:]))
    if count > 5000:
        raise ValueError(f"结构参数量为 {count}，超过 5,000 上限。")
    layers = []
    for index, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if index < len(hidden):
            layers.append(activations[activation]())
    return nn.Sequential(*layers)


def load_imu_model(spec: dict, checkpoint) -> nn.Sequential:
    model = build_imu_model(spec)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError("Checkpoint 包含非有限权重。")
    return model.eval()
