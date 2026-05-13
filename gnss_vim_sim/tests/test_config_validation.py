import json
from pathlib import Path

import pytest

from gnss_vim_sim.core.config import SimConfig
from gnss_vim_sim.core.validation import ConfigValidationError, validate_config


def _write_config(tmp_path: Path, **overrides) -> Path:
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    (mesh_dir / "empty.ply").write_text("ply\nformat ascii 1.0\nend_header\n")
    cfg = {
        "name": "validation_test",
        "seed": 1,
        "duration_s": 2.0,
        "dt_s": 0.1,
        "scene": {
            "frame": "ENU",
            "origin_lat_deg": 0.0,
            "origin_lon_deg": 0.0,
            "origin_alt_m": 0.0,
            "blend_file": "",
            "mesh_dir": "mesh",
        },
        "mission": {
            "cruise_speed_mps": 2.0,
            "waypoint_acceptance_m": 1.0,
            "waypoints": [
                {"name": "takeoff", "x": 0, "y": 0, "z": 2},
                {"name": "land", "x": 5, "y": 0, "z": 2},
            ],
        },
        "sensors": {
            "imu_rate_hz": 50,
            "baro_rate_hz": 25,
            "gnss_rate_hz": 1,
            "range_rate_hz": 1,
            "gnss_source": "gazebo",
        },
        "fusion": {
            "base_gnss_sigma_m": 4.0,
            "baro_sigma_m": 0.3,
            "range_sigma_m": 0.05,
            "adaptive_alpha": 5.0,
            "range_trigger_threshold": 0.6,
            "chi2_gate_threshold": 9.0,
        },
        "energy": {
            "base_vehicle_power_w": 300,
            "gnss_power_w": 0.05,
            "ml_power_w": 0.5,
            "range_power_w": 8,
            "range_pulse_duration_s": 0.02,
        },
    }
    cfg.update(overrides)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    return path


def test_validate_config_accepts_minimal_valid_config(tmp_path):
    cfg = SimConfig.load(_write_config(tmp_path))
    report = validate_config(cfg, require_mesh=True)

    assert report.ok


def test_validate_config_reports_invalid_dt(tmp_path):
    cfg = SimConfig.load(_write_config(tmp_path, dt_s=0))
    report = validate_config(cfg)

    with pytest.raises(ConfigValidationError):
        report.raise_for_errors()
