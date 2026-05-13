from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gnss_vim_sim.world.scene import MeshScene


@dataclass
class RangefinderModel:
    sigma_m: float

    def measure_altitude(self, scene: MeshScene, pos: np.ndarray, rng: np.random.Generator) -> float:
        distance_to_surface = scene.raycast_down(pos)
        return float(distance_to_surface + rng.normal(0.0, self.sigma_m))
