"""第 1 章真实任务使用的确定性模拟传感器数据。"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification


FEATURE_NAMES = (
    "temperature_c",
    "vibration_rms",
    "motor_current_a",
    "pressure_bar",
    "load_ratio",
    "running_hours",
)


def load_device_health_data() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """返回可重复生成的设备状态数据、标签和特征名称。

    标签 0 表示正常，1 表示需要检查。数据仅用于教学，不代表真实设备规律。
    """

    raw_x, labels = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_repeated=0,
        n_clusters_per_class=1,
        weights=[0.78, 0.22],
        class_sep=1.60,
        flip_y=0.025,
        random_state=42,
    )

    offsets = np.array([62.0, 1.4, 4.5, 1.8, 0.65, 2500.0])
    scales = np.array([8.0, 0.45, 0.8, 0.25, 0.18, 1200.0])
    sensor_x = raw_x * scales + offsets

    return sensor_x.astype(np.float64), labels.astype(np.int64), FEATURE_NAMES


if __name__ == "__main__":
    features, targets, names = load_device_health_data()
    print("特征：", names)
    print("数据形状：", features.shape)
    print("正常/故障样本：", np.bincount(targets))
