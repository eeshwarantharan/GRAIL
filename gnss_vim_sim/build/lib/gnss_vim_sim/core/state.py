from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleState:
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray

    @staticmethod
    def at(position: list[float]) -> "VehicleState":
        return VehicleState(
            pos=np.array(position, dtype=float),
            vel=np.zeros(3, dtype=float),
            acc=np.zeros(3, dtype=float),
        )


@dataclass
class EnergyLedger:
    total_j: float = 0.0
    vehicle_j: float = 0.0
    gnss_j: float = 0.0
    ml_j: float = 0.0
    range_j: float = 0.0
    range_pulses: int = 0

    def add_base(self, vehicle_w: float, dt_s: float) -> None:
        e = vehicle_w * dt_s
        self.vehicle_j += e
        self.total_j += e

    def add_sensor_power(self, *, gnss_w: float = 0.0, ml_w: float = 0.0, dt_s: float = 0.0) -> None:
        gnss_e = gnss_w * dt_s
        ml_e = ml_w * dt_s
        self.gnss_j += gnss_e
        self.ml_j += ml_e
        self.total_j += gnss_e + ml_e

    def add_range_pulse(self, power_w: float, duration_s: float) -> None:
        e = power_w * duration_s
        self.range_j += e
        self.total_j += e
        self.range_pulses += 1

    @property
    def total_wh(self) -> float:
        return self.total_j / 3600.0

    @property
    def marginal_sensing_wh(self) -> float:
        return (self.gnss_j + self.ml_j + self.range_j) / 3600.0
