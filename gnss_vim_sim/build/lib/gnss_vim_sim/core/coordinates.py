from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


EARTH_RADIUS_M = 6_378_137.0


@dataclass(frozen=True)
class LocalFrame:
    """Simple ENU frame for mission-scale simulation."""

    origin_lat_deg: float
    origin_lon_deg: float
    origin_alt_m: float = 0.0

    @property
    def lat_rad(self) -> float:
        return math.radians(self.origin_lat_deg)

    def lla_to_enu(self, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        d_lat = math.radians(lat_deg - self.origin_lat_deg)
        d_lon = math.radians(lon_deg - self.origin_lon_deg)
        east = d_lon * EARTH_RADIUS_M * math.cos(self.lat_rad)
        north = d_lat * EARTH_RADIUS_M
        up = alt_m - self.origin_alt_m
        return np.array([east, north, up], dtype=float)

    def enu_to_lla(self, enu: np.ndarray) -> tuple[float, float, float]:
        east, north, up = [float(v) for v in enu]
        lat = self.origin_lat_deg + math.degrees(north / EARTH_RADIUS_M)
        lon = self.origin_lon_deg + math.degrees(east / (EARTH_RADIUS_M * math.cos(self.lat_rad)))
        alt = self.origin_alt_m + up
        return lat, lon, alt
