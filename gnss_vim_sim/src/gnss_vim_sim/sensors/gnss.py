from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
import numpy as np


FEATURE_COLS = [
    "n_sats",
    "mean_cn0",
    "std_cn0",
    "min_cn0",
    "range_cn0",
    "canopy_cn0",
    "mean_elev",
    "min_elev",
    "max_elev",
    "std_elev",
    "elev_spread",
    "vdop",
    "mean_los_ratio",
    "min_los_ratio",
    "std_los_ratio",
    "phase_locked_frac",
    "mean_delay_ns",
    "max_delay_ns",
    "std_delay_ns",
    "mean_mp_error",
    "std_mp_error",
    "max_mp_error",
]


@dataclass
class GnssEpoch:
    z_m: float
    vdop: float
    features: dict[str, float]
    is_bad_truth: bool
    true_error_m: float
    lat_deg: float = float("nan")
    lon_deg: float = float("nan")
    alt_m: float = float("nan")
    h_acc_m: float = float("nan")
    v_acc_m: float = float("nan")
    n_sats: int = 0
    dropped: bool = False
    covariance_enu: tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))


@dataclass
class GazeboGnssConfig:
    horizontal_sigma_m: float = 1.4
    vertical_sigma_m: float = 2.2
    urban_vertical_bias_m: float = 7.0
    dropout_base: float = 0.01
    dropout_urban: float = 0.12
    bad_z_threshold_m: float = 3.0


class GazeboStyleGnss:
    """Pose-driven GNSS model similar to SITL/Gazebo, with urban degradation.

    It deliberately does not replay the training CSV. It derives signal-like
    features from true pose, environmental proximity, and stochastic degradation.
    """

    def __init__(self, cfg: GazeboGnssConfig):
        self.cfg = cfg
        self.vertical_bias_state = 0.0

    def measure(
        self,
        pos: np.ndarray,
        rng: np.random.Generator,
        frame=None,
        proximity: float = 0.0,
    ) -> GnssEpoch:
        urban = float(np.clip(max(proximity, (pos[2] - 2.0) / 10.0), 0.0, 1.0))
        dropout_p = float(np.clip(self.cfg.dropout_base + self.cfg.dropout_urban * urban, 0.0, 0.95))
        dropped = bool(rng.random() < dropout_p)

        n_sats = max(4, int(round(11 - 5.0 * urban + rng.normal(0.0, 1.0))))
        mean_cn0 = float(np.clip(43.0 - 12.0 * urban + rng.normal(0.0, 1.4), 18.0, 50.0))
        std_cn0 = float(np.clip(2.6 + 5.5 * urban + rng.normal(0.0, 0.5), 0.5, 12.0))
        vdop = float(np.clip(1.1 + 2.4 * urban + 7.0 / max(n_sats, 4) + rng.normal(0.0, 0.12), 1.0, 9.0))
        phase_locked = float(np.clip(0.98 - 0.42 * urban + rng.normal(0.0, 0.03), 0.2, 1.0))
        mean_los = float(np.clip(0.96 - 0.55 * urban + rng.normal(0.0, 0.04), 0.05, 1.0))
        delay_ns = float(np.clip(0.35 + 11.0 * urban + rng.normal(0.0, 0.8), 0.0, 35.0))
        mp_error = float(np.clip(0.15 + 3.5 * urban + rng.normal(0.0, 0.35), 0.0, 12.0))

        h_sigma = self.cfg.horizontal_sigma_m * (1.0 + 1.5 * urban) * vdop
        v_sigma = self.cfg.vertical_sigma_m * (1.0 + 2.5 * urban) * vdop
        bias_target = self.cfg.urban_vertical_bias_m * urban
        self.vertical_bias_state = 0.96 * self.vertical_bias_state + 0.04 * bias_target
        vertical_bias = self.vertical_bias_state + rng.normal(0.0, 0.35 * urban)

        east_noise = rng.normal(0.0, h_sigma)
        north_noise = rng.normal(0.0, h_sigma)
        z_noise = rng.normal(0.0, v_sigma)
        measured_enu = np.array([pos[0] + east_noise, pos[1] + north_noise, pos[2] + vertical_bias + z_noise])
        if dropped:
            measured_enu[:] = np.nan

        if frame is not None and not dropped:
            lat, lon, alt = frame.enu_to_lla(measured_enu)
        elif frame is not None:
            lat, lon, alt = float("nan"), float("nan"), float("nan")
        else:
            lat, lon, alt = float("nan"), float("nan"), float(measured_enu[2])

        features = {
            "n_sats": float(n_sats),
            "mean_cn0": mean_cn0,
            "std_cn0": std_cn0,
            "min_cn0": mean_cn0 - 1.8 * std_cn0,
            "range_cn0": 3.8 * std_cn0,
            "canopy_cn0": mean_cn0 - 1.5 * urban,
            "mean_elev": float(np.clip(54.0 - 18.0 * urban + rng.normal(0.0, 2.0), 12.0, 82.0)),
            "min_elev": float(np.clip(12.0 - 3.0 * urban + rng.normal(0.0, 1.0), 5.0, 35.0)),
            "max_elev": float(np.clip(78.0 - 5.0 * urban + rng.normal(0.0, 1.0), 45.0, 90.0)),
            "std_elev": float(np.clip(15.0 + 4.0 * urban + rng.normal(0.0, 1.0), 4.0, 35.0)),
            "elev_spread": float(np.clip(66.0 - 4.0 * urban + rng.normal(0.0, 2.0), 20.0, 85.0)),
            "vdop": vdop,
            "mean_los_ratio": mean_los,
            "min_los_ratio": float(np.clip(mean_los - 0.22 - 0.15 * urban, 0.0, 1.0)),
            "std_los_ratio": float(np.clip(0.04 + 0.14 * urban, 0.01, 0.35)),
            "phase_locked_frac": phase_locked,
            "mean_delay_ns": delay_ns,
            "max_delay_ns": delay_ns * (1.4 + 0.9 * urban),
            "std_delay_ns": float(np.clip(0.25 + 3.8 * urban + rng.normal(0.0, 0.3), 0.0, 15.0)),
            "mean_mp_error": mp_error,
            "std_mp_error": float(np.clip(0.15 + 1.4 * urban + rng.normal(0.0, 0.2), 0.0, 7.0)),
            "max_mp_error": float(np.clip(mp_error * (1.5 + urban), 0.0, 25.0)),
        }
        true_error_m = float("inf") if dropped else abs(float(measured_enu[2] - pos[2]))
        return GnssEpoch(
            z_m=float(measured_enu[2]) if not dropped else float("nan"),
            vdop=vdop,
            features=features,
            is_bad_truth=bool(dropped or true_error_m > self.cfg.bad_z_threshold_m),
            true_error_m=true_error_m,
            lat_deg=float(lat),
            lon_deg=float(lon),
            alt_m=float(alt),
            h_acc_m=float(h_sigma),
            v_acc_m=float(v_sigma),
            n_sats=n_sats,
            dropped=dropped,
            covariance_enu=(h_sigma * h_sigma, h_sigma * h_sigma, v_sigma * v_sigma),
        )


class FeatureReplayGnss:
    def __init__(
        self,
        feature_csv: Path | None,
        bad_z_threshold_m: float,
        sigma_m: float = 3.0,
        max_rows: int = 200_000,
    ):
        self.rows = self._load_rows(feature_csv, max_rows)
        self.bad_z_threshold_m = bad_z_threshold_m
        self.sigma_m = sigma_m

    @staticmethod
    def _load_rows(path: Path | None, max_rows: int) -> list[dict[str, float]]:
        if path is None or not path.exists():
            return []
        rows: list[dict[str, float]] = []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for i, raw in enumerate(reader):
                if i >= max_rows:
                    break
                row: dict[str, float] = {}
                for k, v in raw.items():
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        continue
                rows.append(row)
        return rows

    def measure(
        self,
        pos: np.ndarray,
        rng: np.random.Generator,
        frame=None,
        proximity: float = 0.0,
    ) -> GnssEpoch:
        if self.rows:
            row = self._nearest_feature_row(pos, rng)
            features = {col: float(row.get(col, 0.0)) for col in FEATURE_COLS}
            delay_bias_m = 299_792_458.0 * features.get("mean_delay_ns", 0.0) * 1e-9
            mp_bias_m = 0.5 * features.get("mean_mp_error", 0.0)
            z_m = float(pos[2] + delay_bias_m + mp_bias_m + rng.normal(0.0, self.sigma_m))
            if "estimated_z_m" in row and "true_z" in row:
                true_error_m = abs(float(row["estimated_z_m"] - row["true_z"]))
            else:
                true_error_m = abs(z_m - float(pos[2]))
            lat, lon, alt = frame.enu_to_lla(np.array([pos[0], pos[1], z_m])) if frame is not None else (float("nan"), float("nan"), z_m)
            return GnssEpoch(
                z_m=z_m,
                vdop=max(float(features.get("vdop", 2.0)), 0.1),
                features=features,
                is_bad_truth=true_error_m > self.bad_z_threshold_m,
                true_error_m=true_error_m,
                lat_deg=float(lat),
                lon_deg=float(lon),
                alt_m=float(alt),
                h_acc_m=self.sigma_m,
                v_acc_m=self.sigma_m,
                n_sats=int(features.get("n_sats", 0)),
                covariance_enu=(self.sigma_m**2, self.sigma_m**2, self.sigma_m**2),
            )
        return self._synthetic_epoch(pos, rng, frame)

    def _nearest_feature_row(self, pos: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
        sample_size = min(len(self.rows), 4000)
        idx = rng.choice(len(self.rows), size=sample_size, replace=False)
        best_i = int(idx[0])
        best_d = float("inf")
        for i in idx:
            row = self.rows[int(i)]
            dx = row.get("true_x", pos[0]) - pos[0]
            dy = row.get("true_y", pos[1]) - pos[1]
            dz = (row.get("true_z", pos[2]) - pos[2]) * 3.0
            d = dx * dx + dy * dy + dz * dz
            if d < best_d:
                best_i = int(i)
                best_d = d
        return self.rows[best_i]

    def _synthetic_epoch(self, pos: np.ndarray, rng: np.random.Generator, frame=None) -> GnssEpoch:
        urban_factor = float(np.clip((pos[2] - 1.0) / 10.0, 0.0, 1.0))
        vdop = 1.5 + 2.5 * urban_factor + rng.normal(0.0, 0.2)
        mean_cn0 = 42.0 - 10.0 * urban_factor + rng.normal(0.0, 1.5)
        n_sats = max(4, int(round(10 - 4 * urban_factor + rng.normal(0.0, 1.0))))
        delay_ns = max(0.0, 0.8 + 8.0 * urban_factor + rng.normal(0.0, 1.0))
        features = {
            "n_sats": float(n_sats),
            "mean_cn0": mean_cn0,
            "std_cn0": 3.0 + 3.0 * urban_factor,
            "min_cn0": mean_cn0 - 8.0,
            "range_cn0": 14.0 + 8.0 * urban_factor,
            "canopy_cn0": mean_cn0 - 2.0,
            "mean_elev": 50.0 - 12.0 * urban_factor,
            "min_elev": 15.0,
            "max_elev": 80.0,
            "std_elev": 15.0,
            "elev_spread": 65.0,
            "vdop": vdop,
            "mean_los_ratio": 1.0 - 0.35 * urban_factor,
            "min_los_ratio": 0.7 - 0.25 * urban_factor,
            "std_los_ratio": 0.05 + 0.08 * urban_factor,
            "phase_locked_frac": 1.0 - 0.25 * urban_factor,
            "mean_delay_ns": delay_ns,
            "max_delay_ns": delay_ns * 2.0,
            "std_delay_ns": 0.5 + 2.0 * urban_factor,
            "mean_mp_error": 0.2 + 2.0 * urban_factor,
            "std_mp_error": 0.2 + 1.0 * urban_factor,
            "max_mp_error": 0.5 + 5.0 * urban_factor,
        }
        bias = 299_792_458.0 * delay_ns * 1e-9
        z_m = float(pos[2] + bias + rng.normal(0.0, self.sigma_m))
        true_error_m = abs(z_m - float(pos[2]))
        lat, lon, alt = frame.enu_to_lla(np.array([pos[0], pos[1], z_m])) if frame is not None else (float("nan"), float("nan"), z_m)
        return GnssEpoch(
            z_m,
            max(vdop, 0.1),
            features,
            true_error_m > self.bad_z_threshold_m,
            true_error_m,
            lat_deg=float(lat),
            lon_deg=float(lon),
            alt_m=float(alt),
            h_acc_m=self.sigma_m,
            v_acc_m=self.sigma_m * max(vdop, 0.1),
            n_sats=int(features["n_sats"]),
            covariance_enu=(self.sigma_m**2, self.sigma_m**2, (self.sigma_m * max(vdop, 0.1)) ** 2),
        )


def build_gnss_source(sensor_cfg, feature_csv: Path | None = None, base_sigma_m: float = 3.0):
    source = sensor_cfg.gnss_source.lower()
    if source in {"gazebo", "sitl", "pose"}:
        return GazeboStyleGnss(
            GazeboGnssConfig(
                horizontal_sigma_m=sensor_cfg.gnss_horizontal_sigma_m,
                vertical_sigma_m=sensor_cfg.gnss_vertical_sigma_m,
                urban_vertical_bias_m=sensor_cfg.gnss_urban_vertical_bias_m,
                dropout_base=sensor_cfg.gnss_dropout_base,
                dropout_urban=sensor_cfg.gnss_dropout_urban,
                bad_z_threshold_m=sensor_cfg.bad_z_threshold_m,
            )
        )
    if source in {"feature_replay", "csv", "replay"}:
        return FeatureReplayGnss(
            feature_csv,
            sensor_cfg.bad_z_threshold_m,
            sigma_m=base_sigma_m,
            max_rows=sensor_cfg.max_gnss_feature_rows,
        )
    raise ValueError(f"Unknown gnss_source '{sensor_cfg.gnss_source}'")
