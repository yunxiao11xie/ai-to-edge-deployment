"""第 2 章真实任务使用的确定性 IMU 特征数据。"""

from __future__ import annotations

import numpy as np


def load_imu_activity_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(2026)
    counts = [260, 210, 130]
    centers = np.array([
        [0.25, 0.30, 0.20, 0.18, 0.22, 0.15],
        [1.15, 0.95, 1.05, 0.75, 0.82, 0.70],
        [1.80, 1.55, 1.65, 1.35, 1.20, 1.45],
    ])
    parts = [rng.normal(center, 0.22 + index * 0.03, size=(count, 6))
             for index, (count, center) in enumerate(zip(counts, centers))]
    features = np.vstack(parts).astype(np.float32)
    targets = np.concatenate([np.full(count, index) for index, count in enumerate(counts)]).astype(np.int64)
    order = rng.permutation(len(targets))
    names = ["acc_x_rms", "acc_y_rms", "acc_z_rms", "gyro_x_rms", "gyro_y_rms", "gyro_z_rms"]
    return features[order], targets[order], names
