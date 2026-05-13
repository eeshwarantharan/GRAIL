from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from gnss_vim_sim.core.state import VehicleState


@dataclass(frozen=True)
class Waypoint:
    name: str
    position: np.ndarray

    @staticmethod
    def from_dict(raw: dict) -> "Waypoint":
        return Waypoint(
            name=str(raw.get("name", "wp")),
            position=np.array([raw["x"], raw["y"], raw["z"]], dtype=float),
        )


class Mission:
    def __init__(self, waypoints: list[Waypoint], cruise_speed_mps: float, acceptance_m: float):
        if len(waypoints) < 2:
            raise ValueError("mission requires at least two waypoints")
        self.waypoints = waypoints
        self.cruise_speed_mps = cruise_speed_mps
        self.acceptance_m = acceptance_m
        self.index = 0

    @property
    def target(self) -> Waypoint:
        return self.waypoints[min(self.index, len(self.waypoints) - 1)]

    def step_target(self, state: VehicleState) -> np.ndarray:
        if (
            np.linalg.norm(self.target.position - state.pos) < self.acceptance_m
            and self.index < len(self.waypoints) - 1
        ):
            self.index += 1
        return self.target.position


class SimpleMultirotor:
    """Compact point-mass controller for repeatable estimator tests."""

    def __init__(self, state: VehicleState, cruise_speed_mps: float):
        self.state = state
        self.cruise_speed_mps = cruise_speed_mps

    def step(self, target: np.ndarray, dt_s: float) -> VehicleState:
        error = target - self.state.pos
        distance = float(np.linalg.norm(error))
        desired_vel = np.zeros(3) if distance < 1e-6 else error / distance * self.cruise_speed_mps
        desired_vel = np.clip(desired_vel, -self.cruise_speed_mps, self.cruise_speed_mps)
        acc_cmd = (desired_vel - self.state.vel) / 0.8
        acc_cmd -= 0.15 * self.state.vel
        acc_cmd = np.clip(acc_cmd, -4.0, 4.0)
        self.state.pos = self.state.pos + self.state.vel * dt_s + 0.5 * acc_cmd * dt_s * dt_s
        self.state.vel = self.state.vel + acc_cmd * dt_s
        self.state.acc = acc_cmd
        return self.state
