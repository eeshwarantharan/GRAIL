from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gnss_vim_sim.core.config import SimConfig


class ConfigValidationError(ValueError):
    """Raised when a simulator config is structurally invalid."""


@dataclass(frozen=True)
class ValidationReport:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            body = "\n".join(f"- {msg}" for msg in self.errors)
            raise ConfigValidationError(f"Invalid simulator config:\n{body}")


def validate_config(cfg: SimConfig, *, require_mesh: bool = False) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []

    if cfg.dt_s <= 0:
        errors.append("dt_s must be > 0")
    if cfg.duration_s <= 0:
        errors.append("duration_s must be > 0")
    if cfg.mission.cruise_speed_mps <= 0:
        errors.append("mission.cruise_speed_mps must be > 0")
    if cfg.mission.waypoint_acceptance_m <= 0:
        errors.append("mission.waypoint_acceptance_m must be > 0")
    if len(cfg.mission.waypoints) < 2:
        errors.append("mission.waypoints must contain at least takeoff and landing points")

    for i, wp in enumerate(cfg.mission.waypoints):
        for key in ("name", "x", "y", "z"):
            if key not in wp:
                errors.append(f"mission.waypoints[{i}] is missing {key!r}")
        for key in ("x", "y", "z"):
            if key in wp:
                try:
                    float(wp[key])
                except (TypeError, ValueError):
                    errors.append(f"mission.waypoints[{i}].{key} must be numeric")

    if cfg.scene.frame.upper() != "ENU":
        errors.append("scene.frame must currently be 'ENU'")
    if not (-90.0 <= cfg.scene.origin_lat_deg <= 90.0):
        errors.append("scene.origin_lat_deg must be in [-90, 90]")
    if not (-180.0 <= cfg.scene.origin_lon_deg <= 180.0):
        errors.append("scene.origin_lon_deg must be in [-180, 180]")

    mesh_dir = cfg.resolve(cfg.scene.mesh_dir)
    if mesh_dir is None or not mesh_dir.exists():
        msg = f"scene.mesh_dir does not exist: {mesh_dir}"
        (errors if require_mesh else warnings).append(msg)
    elif not mesh_dir.is_dir():
        errors.append(f"scene.mesh_dir must be a directory: {mesh_dir}")
    else:
        ply_count = len(list(mesh_dir.glob("*.ply")))
        if ply_count == 0:
            msg = f"scene.mesh_dir contains no .ply files: {mesh_dir}"
            (errors if require_mesh else warnings).append(msg)

    blend = cfg.resolve(cfg.scene.blend_file)
    if cfg.scene.blend_file and blend is not None and not blend.exists():
        warnings.append(f"scene.blend_file is set but was not found: {blend}")

    if cfg.sensors.gnss_source not in {"gazebo", "feature_replay"}:
        errors.append("sensors.gnss_source must be one of: gazebo, feature_replay")
    if cfg.sensors.gnss_source == "feature_replay":
        csv_path = cfg.resolve(cfg.sensors.gnss_feature_csv)
        if csv_path is None or not csv_path.exists():
            errors.append(f"sensors.gnss_feature_csv is required for feature_replay and was not found: {csv_path}")

    for field, value in [
        ("sensors.imu_rate_hz", cfg.sensors.imu_rate_hz),
        ("sensors.baro_rate_hz", cfg.sensors.baro_rate_hz),
        ("sensors.gnss_rate_hz", cfg.sensors.gnss_rate_hz),
        ("sensors.range_rate_hz", cfg.sensors.range_rate_hz),
        ("fusion.base_gnss_sigma_m", cfg.fusion.base_gnss_sigma_m),
        ("fusion.baro_sigma_m", cfg.fusion.baro_sigma_m),
        ("fusion.range_sigma_m", cfg.fusion.range_sigma_m),
        ("energy.base_vehicle_power_w", cfg.energy.base_vehicle_power_w),
    ]:
        if value <= 0:
            errors.append(f"{field} must be > 0")

    return ValidationReport(errors=errors, warnings=warnings)


def format_report(report: ValidationReport) -> str:
    lines: list[str] = []
    if report.errors:
        lines.append("Errors:")
        lines.extend(f"- {msg}" for msg in report.errors)
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {msg}" for msg in report.warnings)
    return "\n".join(lines) if lines else "Config validation passed."
