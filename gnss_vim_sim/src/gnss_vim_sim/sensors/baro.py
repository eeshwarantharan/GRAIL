from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class BaroModel:
    sigma_m: float
    bias_walk_std: float = 0.01
    bias_m: float = 0.0

    def measure(self, true_z_m: float, dt_s: float, proximity: float, rng: np.random.Generator) -> float:
        self.bias_m += rng.normal(0.0, self.bias_walk_std * math.sqrt(dt_s))
        local_pressure_error = proximity * rng.normal(0.0, 0.05)
        return float(true_z_m + self.bias_m + local_pressure_error + rng.normal(0.0, self.sigma_m))
