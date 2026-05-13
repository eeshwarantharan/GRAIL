from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class SceneConfig:
    frame: str
    origin_lat_deg: float
    origin_lon_deg: float
    origin_alt_m: float
    blend_file: str
    mesh_dir: str


@dataclass(frozen=True)
class MissionConfig:
    cruise_speed_mps: float
    waypoint_acceptance_m: float
    waypoints: list[dict]
    route_planner: str = "astar"
    planner_grid_m: float = 6.0
    planner_clearance_m: float = 10.0


@dataclass(frozen=True)
class SensorConfig:
    imu_rate_hz: float
    baro_rate_hz: float
    gnss_rate_hz: float
    range_rate_hz: float
    gnss_source: str = "gazebo"
    gnss_feature_csv: str | None = None
    gnss_l1_only: bool = True
    bad_z_threshold_m: float = 3.0
    max_gnss_feature_rows: int = 200_000
    gnss_horizontal_sigma_m: float = 1.4
    gnss_vertical_sigma_m: float = 2.2
    gnss_urban_vertical_bias_m: float = 7.0
    gnss_dropout_base: float = 0.01
    gnss_dropout_urban: float = 0.12


@dataclass(frozen=True)
class FusionConfig:
    base_gnss_sigma_m: float
    baro_sigma_m: float
    range_sigma_m: float
    adaptive_alpha: float
    range_trigger_threshold: float
    chi2_gate_threshold: float


@dataclass(frozen=True)
class EnergyConfig:
    base_vehicle_power_w: float
    gnss_power_w: float
    ml_power_w: float
    range_power_w: float
    range_pulse_duration_s: float


@dataclass(frozen=True)
class SimConfig:
    name: str
    seed: int
    duration_s: float
    dt_s: float
    scene: SceneConfig
    mission: MissionConfig
    sensors: SensorConfig
    fusion: FusionConfig
    energy: EnergyConfig
    config_dir: Path

    @staticmethod
    def load(path: str | Path) -> "SimConfig":
        cfg_path = Path(path).resolve()
        raw = json.loads(cfg_path.read_text())
        return SimConfig(
            name=raw["name"],
            seed=int(raw.get("seed", 42)),
            duration_s=float(raw["duration_s"]),
            dt_s=float(raw["dt_s"]),
            scene=SceneConfig(**raw["scene"]),
            mission=MissionConfig(**raw["mission"]),
            sensors=SensorConfig(**raw["sensors"]),
            fusion=FusionConfig(**raw["fusion"]),
            energy=EnergyConfig(**raw["energy"]),
            config_dir=cfg_path.parent,
        )

    def resolve(self, value: str | None) -> Path | None:
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.config_dir / path).resolve()
