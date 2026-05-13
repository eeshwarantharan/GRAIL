from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class ImuModel:
    accel_noise_std: float = 0.05
    bias_walk_std: float = 1e-4
    bias_z: float = 0.0

    def measure_z_accel(self, true_acc_z: float, dt_s: float, rng: np.random.Generator) -> float:
        self.bias_z += rng.normal(0.0, self.bias_walk_std * math.sqrt(dt_s))
        return float(true_acc_z + self.bias_z + rng.normal(0.0, self.accel_noise_std))
