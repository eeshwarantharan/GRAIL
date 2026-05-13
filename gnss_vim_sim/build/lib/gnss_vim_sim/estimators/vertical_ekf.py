from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class FusionParams:
    base_gnss_sigma_m: float
    baro_sigma_m: float
    range_sigma_m: float
    adaptive_alpha: float
    range_trigger_threshold: float
    chi2_gate_threshold: float


class VerticalEKF:
    """Three-state vertical EKF: z, vz, baro_bias."""

    def __init__(self, name: str, z0: float, params: FusionParams):
        self.name = name
        self.x = np.array([z0, 0.0, 0.0], dtype=float)
        self.P = np.eye(3, dtype=float) * 3.0
        self.params = params
        self.range_pulses = 0
        self.gnss_updates = 0
        self.gnss_rejections = 0
        self.last_innovation = 0.0
        self.last_nis = 0.0
        self.last_gnss_r = float("nan")

    @property
    def altitude(self) -> float:
        return float(self.x[0])

    def predict(self, accel_z: float, dt_s: float) -> None:
        f = np.array([[1.0, dt_s, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        b = np.array([0.5 * dt_s * dt_s, dt_s, 0.0])
        q = np.diag([2e-4, 1e-2, 1e-5])
        self.x = f @ self.x + b * accel_z
        self.P = f @ self.P @ f.T + q

    def update_baro(self, z_baro: float) -> None:
        self._update(z_baro, np.array([[1.0, 0.0, 1.0]]), self.params.baro_sigma_m**2)

    def update_range(self, z_range: float) -> None:
        self._update(z_range, np.array([[1.0, 0.0, 0.0]]), self.params.range_sigma_m**2)
        self.range_pulses += 1

    def update_gnss_fixed(self, z_gnss: float) -> None:
        r = self.params.base_gnss_sigma_m**2
        self.last_gnss_r = r
        self._update_gnss(z_gnss, r)

    def update_gnss_vdop(self, z_gnss: float, vdop: float) -> None:
        sigma = self.params.base_gnss_sigma_m * max(vdop, 0.1)
        r = sigma * sigma
        self.last_gnss_r = r
        self._update_gnss(z_gnss, r)

    def update_gnss_chi2(self, z_gnss: float, vdop: float) -> None:
        h = np.array([[1.0, 0.0, 0.0]])
        r = (self.params.base_gnss_sigma_m * max(vdop, 0.1)) ** 2
        innovation, s = self._innovation(z_gnss, h, r)
        nis = innovation * innovation / s
        self.last_innovation = innovation
        self.last_nis = nis
        self.last_gnss_r = r
        if nis <= self.params.chi2_gate_threshold:
            self._update(z_gnss, h, r)
            self.gnss_updates += 1
        else:
            self.gnss_rejections += 1

    def update_gnss_ml(self, z_gnss: float, vdop: float, risk: float) -> None:
        risk = float(np.clip(risk, 0.0, 1.0))
        sigma = self.params.base_gnss_sigma_m * max(vdop, 0.1)
        r = sigma * sigma * math.exp(self.params.adaptive_alpha * risk)
        self.last_gnss_r = r
        self._update_gnss(z_gnss, r)

    def _update_gnss(self, z_gnss: float, r: float) -> None:
        self._update(z_gnss, np.array([[1.0, 0.0, 0.0]]), r)
        self.gnss_updates += 1

    def _innovation(self, z_meas: float, h: np.ndarray, r: float) -> tuple[float, float]:
        innovation = float(z_meas - (h @ self.x)[0])
        s = float((h @ self.P @ h.T)[0, 0] + r)
        return innovation, s

    def _update(self, z_meas: float, h: np.ndarray, r: float) -> None:
        innovation, s = self._innovation(z_meas, h, r)
        k = (self.P @ h.T) / s
        self.x = self.x + k.flatten() * innovation
        self.P = (np.eye(3) - k @ h) @ self.P
        self.last_innovation = innovation
        self.last_nis = innovation * innovation / s
