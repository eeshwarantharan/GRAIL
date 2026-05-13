from __future__ import annotations

from pathlib import Path
import numpy as np

from gnss_vim_sim.core.config import SimConfig
from gnss_vim_sim.core.coordinates import LocalFrame
from gnss_vim_sim.core.state import EnergyLedger, VehicleState
from gnss_vim_sim.estimators.vertical_ekf import FusionParams, VerticalEKF
from gnss_vim_sim.io.logging import write_csv, write_json
from gnss_vim_sim.ml.runtime import RuntimeModel
from gnss_vim_sim.planning.mission import Mission, SimpleMultirotor, Waypoint
from gnss_vim_sim.planning.router import plan_safe_route
from gnss_vim_sim.sensors.baro import BaroModel
from gnss_vim_sim.sensors.gnss import GnssEpoch, build_gnss_source
from gnss_vim_sim.sensors.imu import ImuModel
from gnss_vim_sim.sensors.rangefinder import RangefinderModel
from gnss_vim_sim.sim.metrics import binary_metrics, estimator_metrics, improvement_metrics
from gnss_vim_sim.world.scene import MeshScene


ESTIMATOR_NAMES = ["fixed_gnss", "vdop_chi2", "always_range", "ml_integrity"]


class SimulationRunner:
    def __init__(self, cfg: SimConfig, model: RuntimeModel, out_dir: Path):
        self.cfg = cfg
        self.model = model
        self.out_dir = out_dir
        self.rng = np.random.default_rng(cfg.seed)
        self.scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
        self.frame = LocalFrame(
            cfg.scene.origin_lat_deg,
            cfg.scene.origin_lon_deg,
            cfg.scene.origin_alt_m,
        )

    def run(self) -> dict:
        requested_waypoints = [Waypoint.from_dict(w) for w in self.cfg.mission.waypoints]
        route = plan_safe_route(
            requested_waypoints,
            self.scene,
            planner=self.cfg.mission.route_planner,
            grid_m=self.cfg.mission.planner_grid_m,
            clearance_m=self.cfg.mission.planner_clearance_m,
        )
        waypoints = route.waypoints
        mission = Mission(
            waypoints,
            self.cfg.mission.cruise_speed_mps,
            self.cfg.mission.waypoint_acceptance_m,
        )
        vehicle = SimpleMultirotor(
            VehicleState.at(waypoints[0].position.tolist()),
            self.cfg.mission.cruise_speed_mps,
        )
        params = FusionParams(**self.cfg.fusion.__dict__)
        ekfs = {name: VerticalEKF(name, waypoints[0].position[2], params) for name in ESTIMATOR_NAMES}
        ledgers = {name: EnergyLedger() for name in ESTIMATOR_NAMES}

        imu = ImuModel()
        baro = BaroModel(self.cfg.fusion.baro_sigma_m)
        gnss = build_gnss_source(
            self.cfg.sensors,
            self.cfg.resolve(self.cfg.sensors.gnss_feature_csv),
            base_sigma_m=self.cfg.fusion.base_gnss_sigma_m,
        )
        rangefinder = RangefinderModel(self.cfg.fusion.range_sigma_m)

        rows: list[dict] = []
        t = 0.0
        baro_next = 0.0
        gnss_next = 0.0
        last_gnss: GnssEpoch | None = None
        last_risk = 0.0
        range_fired_this_epoch = {name: False for name in ESTIMATOR_NAMES}

        while t <= self.cfg.duration_s + 1e-9:
            target = mission.step_target(vehicle.state)
            state = vehicle.step(target, self.cfg.dt_s)
            proximity = self.scene.proximity_score(state.pos)
            az = imu.measure_z_accel(state.acc[2], self.cfg.dt_s, self.rng)

            for ekf in ekfs.values():
                ekf.predict(az, self.cfg.dt_s)
            for ledger in ledgers.values():
                ledger.add_base(self.cfg.energy.base_vehicle_power_w, self.cfg.dt_s)

            if t >= baro_next - 1e-12:
                baro_next += 1.0 / self.cfg.sensors.baro_rate_hz
                z_baro = baro.measure(state.pos[2], self.cfg.dt_s, proximity, self.rng)
                for ekf in ekfs.values():
                    ekf.update_baro(z_baro)
            else:
                z_baro = np.nan

            gnss_epoch = False
            z_gnss = np.nan
            gnss_bad_truth = False
            gnss_true_error_m = np.nan
            range_meas = np.nan
            range_fired_this_epoch = {name: False for name in ESTIMATOR_NAMES}

            if t >= gnss_next - 1e-12:
                gnss_next += 1.0 / self.cfg.sensors.gnss_rate_hz
                gnss_epoch = True
                last_gnss = gnss.measure(state.pos, self.rng, frame=self.frame, proximity=proximity)
                z_gnss = last_gnss.z_m
                gnss_bad_truth = last_gnss.is_bad_truth
                gnss_true_error_m = last_gnss.true_error_m
                last_risk = self.model.predict_score(
                    last_gnss.features,
                    context={"t": t, "position_enu": state.pos.tolist(), "sensor": "gnss"},
                )

                range_meas = rangefinder.measure_altitude(self.scene, state.pos, self.rng)

                if not last_gnss.dropped and np.isfinite(z_gnss):
                    ekfs["fixed_gnss"].update_gnss_fixed(z_gnss)
                    ekfs["vdop_chi2"].update_gnss_chi2(z_gnss, last_gnss.vdop)
                    ekfs["always_range"].update_gnss_vdop(z_gnss, last_gnss.vdop)
                    ekfs["ml_integrity"].update_gnss_ml(z_gnss, last_gnss.vdop, last_risk)
                else:
                    for ekf in ekfs.values():
                        ekf.gnss_rejections += 1
                ekfs["always_range"].update_range(range_meas)

                range_fired_this_epoch["always_range"] = True
                ledgers["always_range"].add_range_pulse(
                    self.cfg.energy.range_power_w, self.cfg.energy.range_pulse_duration_s
                )

                if last_risk >= self.cfg.fusion.range_trigger_threshold:
                    ekfs["ml_integrity"].update_range(range_meas)
                    range_fired_this_epoch["ml_integrity"] = True
                    ledgers["ml_integrity"].add_range_pulse(
                        self.cfg.energy.range_power_w, self.cfg.energy.range_pulse_duration_s
                    )

                for name, ledger in ledgers.items():
                    ledger.add_sensor_power(
                        gnss_w=self.cfg.energy.gnss_power_w,
                        ml_w=self.cfg.energy.ml_power_w if name == "ml_integrity" else 0.0,
                        dt_s=1.0 / self.cfg.sensors.gnss_rate_hz,
                    )

            true_lat, true_lon, true_alt = self.frame.enu_to_lla(state.pos)
            row = {
                "t": round(t, 4),
                "true_x": float(state.pos[0]),
                "true_y": float(state.pos[1]),
                "true_z": float(state.pos[2]),
                "true_lat": float(true_lat),
                "true_lon": float(true_lon),
                "true_alt": float(true_alt),
                "target_wp": mission.target.name,
                "route_planner": route.planner_used,
                "baro_z": float(z_baro) if np.isfinite(z_baro) else "",
                "gnss_epoch": gnss_epoch,
                "gnss_z": float(z_gnss) if np.isfinite(z_gnss) else "",
                "gnss_lat": float(last_gnss.lat_deg) if last_gnss and np.isfinite(last_gnss.lat_deg) else "",
                "gnss_lon": float(last_gnss.lon_deg) if last_gnss and np.isfinite(last_gnss.lon_deg) else "",
                "gnss_alt": float(last_gnss.alt_m) if last_gnss and np.isfinite(last_gnss.alt_m) else "",
                "gnss_vdop": float(last_gnss.vdop) if last_gnss else "",
                "gnss_h_acc_m": float(last_gnss.h_acc_m) if last_gnss else "",
                "gnss_v_acc_m": float(last_gnss.v_acc_m) if last_gnss else "",
                "gnss_n_sats": int(last_gnss.n_sats) if last_gnss else "",
                "gnss_dropped": bool(last_gnss.dropped) if last_gnss else False,
                "gnss_bad_truth": bool(gnss_bad_truth),
                "gnss_true_error_m": float(gnss_true_error_m) if np.isfinite(gnss_true_error_m) else "",
                "model_score": float(last_risk),
                "ml_risk": float(last_risk),
                "range_z": float(range_meas) if np.isfinite(range_meas) else "",
                "proximity_score": float(proximity),
            }
            for name, ekf in ekfs.items():
                row[f"{name}_z"] = ekf.altitude
                row[f"{name}_nis"] = ekf.last_nis
                row[f"{name}_gnss_r"] = ekf.last_gnss_r if np.isfinite(ekf.last_gnss_r) else ""
                row[f"{name}_range_fired"] = range_fired_this_epoch[name]
            rows.append(row)
            t += self.cfg.dt_s

        return self._write_outputs(rows, ekfs, ledgers)

    def _write_outputs(self, rows: list[dict], ekfs: dict[str, VerticalEKF], ledgers: dict[str, EnergyLedger]) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        write_csv(self.out_dir / "flight_log.csv", rows)
        self._write_route_artifacts(rows)

        summary = {
            "config": self.cfg.name,
            "gnss_source": self.cfg.sensors.gnss_source,
            "route_planner": {
                "mode": self.cfg.mission.route_planner,
                "used": rows[0].get("route_planner", "") if rows else "",
                "planned_waypoints": len({r.get("target_wp", "") for r in rows}),
            },
            "runtime_model": type(self.model).__name__,
            "integrity_model": type(self.model).__name__,
            "coordinate_frame": {
                "frame": self.cfg.scene.frame,
                "origin_lat_deg": self.cfg.scene.origin_lat_deg,
                "origin_lon_deg": self.cfg.scene.origin_lon_deg,
                "origin_alt_m": self.cfg.scene.origin_alt_m,
            },
            "scene": self.scene.stats().__dict__,
            "estimators": {},
            "model_metrics": binary_metrics(rows),
            "integrity": binary_metrics(rows),
            "policy_improvement": improvement_metrics(rows),
            "improvement": improvement_metrics(rows),
        }
        duration_min = max(self.cfg.duration_s / 60.0, 1e-12)
        for name in ESTIMATOR_NAMES:
            ledger = ledgers[name]
            summary["estimators"][name] = {
                **estimator_metrics(rows, f"{name}_z"),
                "gnss_updates": ekfs[name].gnss_updates,
                "gnss_rejections": ekfs[name].gnss_rejections,
                "range_pulses": ekfs[name].range_pulses,
                "range_activations_per_min": ekfs[name].range_pulses / duration_min,
                "total_energy_wh": ledger.total_wh,
                "marginal_sensing_wh": ledger.marginal_sensing_wh,
                "range_energy_wh": ledger.range_j / 3600.0,
            }
        write_json(self.out_dir / "summary.json", summary)
        return summary

    def _write_route_artifacts(self, rows: list[dict]) -> None:
        seen = set()
        route_rows = []
        for row in rows:
            name = row.get("target_wp", "")
            if not name or name in seen:
                continue
            seen.add(name)
            route_rows.append(
                {
                    "name": name,
                    "x": row["true_x"],
                    "y": row["true_y"],
                    "z": row["true_z"],
                    "lat": row["true_lat"],
                    "lon": row["true_lon"],
                    "alt": row["true_alt"],
                    "planner": row.get("route_planner", ""),
                }
            )
        write_csv(self.out_dir / "route_plan.csv", route_rows)
        write_json(
            self.out_dir / "route_plan.json",
            {
                "planner": rows[0].get("route_planner", "") if rows else "",
                "waypoints": route_rows,
            },
        )
