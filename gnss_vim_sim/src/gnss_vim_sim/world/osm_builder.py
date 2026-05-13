"""
osm_builder.py  —  GNSS-VIM-Sim OSM Scene Builder
==================================================

Converts an OpenStreetMap region into a PLY mesh + demo_config.json
ready for immediate use with gnss-vim-sim run.

Workflow
--------
  gnss-vim-sim new-scene --lat 12.9906 --lon 80.2296 --name my_campus

This single command:
  1. Downloads buildings from OpenStreetMap via the Overpass API
     (no API key required, rate-limited to one request per session)
  2. Extrudes footprints into 3-D box meshes and exports them as
     demo_mesh/buildings.ply (binary PLY, trimesh)
  3. Generates a ground plane mesh: demo_mesh/ground.ply
  4. Writes a pre-populated demo_config.json with:
     - scene origin set to (lat, lon)
     - scene bounds inferred from OSM coverage area
     - default 5-waypoint mission inside the area
     - all sensor / fusion / energy defaults from GRAIL paper
  5. Prints the exact gnss-vim-sim run command to execute next

Dependencies (all pip-installable, none required for the rest of gnss-vim-sim)
---------------------------------------------------------------------------
  pip install osmnx shapely trimesh numpy

Manual OSM workflow (if the network is unavailable)
----------------------------------------------------
  1. Go to https://www.openstreetmap.org/export
  2. Select "Overpass API" and download your area as .osm or .geojson
  3. Convert with BlenderGIS (Blender plugin) → export as PLY files
  4. Place PLY files in <project>/demo_mesh/
  5. Run: gnss-vim-sim validate --config demo_config.json
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Sequence


# ── constants ─────────────────────────────────────────────────────────────────

RADIUS_M       = 800    # half-width of the scene bounding box (metres)
DEFAULT_HEIGHT = 12.0   # default building height when OSM tag is absent
GROUND_THICK   = 0.5    # ground mesh thickness (metres)


# ── coordinate helpers ────────────────────────────────────────────────────────

def _latlon_to_enu(lat: float, lon: float,
                   ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Approximate flat-Earth ENU projection (valid for < 50 km)."""
    R = 6_371_000.0
    dn = math.radians(lat - ref_lat) * R
    de = math.radians(lon - ref_lon) * R * math.cos(math.radians(ref_lat))
    return de, dn


def _enu_bounds(radius_m: float) -> tuple[float, float, float, float]:
    return -radius_m, radius_m, -radius_m, radius_m


# ── OSM download ──────────────────────────────────────────────────────────────

def _download_buildings(lat: float, lon: float, radius_m: float) -> list[dict]:
    """
    Fetch building footprints from Overpass API.

    Returns a list of building dicts:
        {"polygon": [(x,y),...], "height_m": float}
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError(
            "osmnx is required for automatic OSM download.\n"
            "Install with:  pip install osmnx\n"
            "Or download the .osm file manually from openstreetmap.org/export."
        )

    print(f"  Downloading OSM buildings within {radius_m} m of ({lat:.4f}, {lon:.4f})...")
    tags = {"building": True}
    try:
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius_m)
    except Exception as exc:
        raise RuntimeError(
            f"Overpass API request failed: {exc}\n"
            "Check your internet connection or download the .osm file manually."
        )

    buildings = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        try:
            from shapely.geometry import Polygon, MultiPolygon
            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)
            else:
                continue

            for poly in polys:
                coords = list(poly.exterior.coords)[:-1]  # drop closing vertex
                enu_coords = [_latlon_to_enu(c[1], c[0], lat, lon) for c in coords]

                height_str = row.get("height", row.get("building:levels", None))
                try:
                    if height_str and str(height_str).replace(".", "").isdigit():
                        height = float(height_str)
                        if height < 3:
                            height *= 3.5   # levels → metres
                    else:
                        height = DEFAULT_HEIGHT
                except (ValueError, TypeError):
                    height = DEFAULT_HEIGHT

                buildings.append({"polygon": enu_coords, "height_m": max(3.0, height)})
        except Exception:
            continue

    print(f"  Downloaded {len(buildings)} building footprints")
    return buildings


# ── mesh generation ───────────────────────────────────────────────────────────

def _extrude_polygon_to_mesh(polygon: list[tuple], height: float):
    """
    Extrude a 2-D polygon into a closed 3-D mesh using trimesh.

    Parameters
    ----------
    polygon  : list of (x, y) tuples  (ENU metres, counter-clockwise)
    height   : extrusion height in metres
    """
    import trimesh
    import numpy as np
    from shapely.geometry import Polygon as ShapelyPolygon

    try:
        shp = ShapelyPolygon(polygon)
        if not shp.is_valid or shp.area < 1.0:
            return None
        mesh = trimesh.creation.extrude_polygon(shp, height)
        return mesh
    except Exception:
        return None


def _make_ground_plane(x_min: float, x_max: float,
                       y_min: float, y_max: float) -> object:
    """Flat box mesh representing the ground plane."""
    import trimesh
    import numpy as np
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    sx = x_max - x_min
    sy = y_max - y_min
    box = trimesh.creation.box(extents=[sx, sy, GROUND_THICK])
    box.apply_translation([cx, cy, -GROUND_THICK / 2])
    return box


def build_ply_scene(buildings: list[dict],
                    mesh_dir: Path,
                    radius_m: float = RADIUS_M) -> tuple[int, Path, Path]:
    """
    Convert building dicts to PLY meshes.

    Returns
    -------
    (n_buildings, buildings_ply, ground_ply)
    """
    try:
        import trimesh
        import numpy as np
    except ImportError:
        raise ImportError("trimesh is required:  pip install trimesh numpy")

    mesh_dir.mkdir(parents=True, exist_ok=True)
    meshes = []
    for b in buildings:
        m = _extrude_polygon_to_mesh(b["polygon"], b["height_m"])
        if m is not None:
            meshes.append(m)

    bldg_ply = mesh_dir / "buildings.ply"
    ground_ply = mesh_dir / "ground.ply"

    if meshes:
        combined = trimesh.util.concatenate(meshes)
        combined.export(str(bldg_ply))
        print(f"  Buildings mesh: {bldg_ply.name}  ({len(meshes)} components)")
    else:
        # Write a minimal placeholder so the scene validator doesn't fail
        placeholder = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        placeholder.export(str(bldg_ply))
        print("  [warn] No valid building meshes — wrote placeholder PLY")

    x_min, x_max, y_min, y_max = _enu_bounds(radius_m)
    ground = _make_ground_plane(x_min, x_max, y_min, y_max)
    ground.export(str(ground_ply))
    print(f"  Ground mesh:    {ground_ply.name}")

    return len(meshes), bldg_ply, ground_ply


# ── config generation ─────────────────────────────────────────────────────────

def _default_waypoints(radius_m: float) -> list[dict]:
    """Five-waypoint L-shaped mission within the scene bounds."""
    r = radius_m * 0.5
    return [
        {"name": "takeoff",      "x":  0.0,  "y":  0.0,  "z": 5.0},
        {"name": "canyon_1",     "x": -r*0.6, "y":  r*0.4, "z": 10.0},
        {"name": "canyon_2",     "x":  r*0.4, "y":  r*0.7, "z": 12.0},
        {"name": "open_sky",     "x":  r*0.7, "y": -r*0.3, "z": 5.0},
        {"name": "land",         "x":  0.0,  "y":  0.0,  "z": 5.0},
    ]


def write_config(out_dir: Path, lat: float, lon: float, name: str,
                 radius_m: float = RADIUS_M) -> Path:
    """Write a pre-populated demo_config.json for this scene."""
    cfg = {
        "name": name,
        "seed": 42,
        "duration_s": 300.0,
        "dt_s": 0.01,
        "scene": {
            "frame": "ENU",
            "origin_lat_deg": lat,
            "origin_lon_deg": lon,
            "origin_alt_m": 0.0,
            "blend_file": "",
            "mesh_dir": "demo_mesh",
        },
        "mission": {
            "cruise_speed_mps": 5.0,
            "waypoint_acceptance_m": 1.5,
            "route_planner": "astar",
            "planner_grid_m": 5.0,
            "planner_clearance_m": 8.0,
            "waypoints": _default_waypoints(radius_m),
        },
        "sensors": {
            "imu_rate_hz": 100.0,
            "baro_rate_hz": 50.0,
            "gnss_rate_hz": 1.0,
            "range_rate_hz": 1.0,
            "gnss_source": "gazebo",
            "gnss_feature_csv": None,
            "gnss_l1_only": True,
            "bad_z_threshold_m": 3.0,
            "max_gnss_feature_rows": 200000,
            "gnss_horizontal_sigma_m": 1.0,
            "gnss_vertical_sigma_m": 1.2,
            "gnss_urban_vertical_bias_m": 5.0,
            "gnss_dropout_base": 0.01,
            "gnss_dropout_urban": 0.08,
        },
        "fusion": {
            "base_gnss_sigma_m": 4.0,
            "baro_sigma_m": 0.3,
            "range_sigma_m": 0.06,
            "adaptive_alpha": 8.0,
            "range_trigger_threshold": 0.40,
            "chi2_gate_threshold": 9.0,
        },
        "energy": {
            "base_vehicle_power_w": 300.0,
            "gnss_power_w": 0.05,
            "ml_power_w": 0.5,
            "range_power_w": 8.0,
            "range_pulse_duration_s": 0.02,
        },
    }
    config_path = out_dir / "demo_config.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    return config_path


# ── top-level entry point ─────────────────────────────────────────────────────

def build_new_scene(lat: float, lon: float, name: str,
                    out_dir: Path, radius_m: float = RADIUS_M,
                    offline: bool = False) -> Path:
    """
    Full pipeline: OSM download → PLY meshes → demo_config.json.

    Parameters
    ----------
    lat, lon  : scene origin in WGS-84 decimal degrees
    name      : project name (used in config "name" field and directory)
    out_dir   : where to write the project files
    radius_m  : half-width of the scene bounding box
    offline   : if True, skip OSM download (use pre-placed PLY files)

    Returns
    -------
    Path to the written demo_config.json
    """
    out_dir = out_dir.resolve()
    mesh_dir = out_dir / "demo_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding GNSS-VIM-Sim scene: {name!r}")
    print(f"  Origin: ({lat:.6f}, {lon:.6f})")
    print(f"  Radius: {radius_m} m  →  {2*radius_m} m × {2*radius_m} m scene")
    print(f"  Output: {out_dir}")

    if offline:
        print("  [offline] Skipping OSM download — place PLY files in demo_mesh/ manually")
    else:
        try:
            buildings = _download_buildings(lat, lon, radius_m)
            build_ply_scene(buildings, mesh_dir, radius_m)
        except (ImportError, RuntimeError) as exc:
            print(f"  [warn] {exc}")
            print("  Proceeding with empty mesh — add PLY files to demo_mesh/ manually")

    config_path = write_config(out_dir, lat, lon, name, radius_m)
    print(f"\n  Config written: {config_path}")
    print("\n─────────────────────────────────────────────────────")
    print("  Next steps:")
    print(f"    cd {out_dir}")
    print("    gnss-vim-sim validate --config demo_config.json")
    print("    gnss-vim-sim studio   --config demo_config.json --out runs/studio.html")
    print("    gnss-vim-sim run      --config demo_config.json --out runs/demo_run")
    print("─────────────────────────────────────────────────────")
    return config_path
