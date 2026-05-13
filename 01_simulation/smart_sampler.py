"""
smart_sampler.py
================
Geometry-aware 3D receiver position generator for the GRAIL fingerprint pipeline.

Strategy
--------
Three placement types capture the full diversity of urban multipath environments:

  wall_halo  — rows of points 2.5 m outside each building wall at 5 m spacing.
               These sit in the deepest multipath shadow and show the strongest
               signal-quality variation with height.
  corner     — one point 2.5 m diagonally outward from each building corner.
               Diffraction hotspots with highly distinctive per-floor fingerprints.
  open_space — jittered grid across non-building area.
               Captures the open-sky (low-VDOP) regime.

Every 2D point is replicated at seven height levels z ∈ {1,4,7,10,13,16,19} m,
matching the GRAIL floor convention (floor 0–6, 3 m storey spacing, 1 m receiver
offset above floor).

Usage
-----
  # Requires building PLY meshes exported from BlenderGIS:
  python smart_sampler.py --mesh-dir meshes/ --out sampling_points.csv

  # Visualise ground-floor layout:
  python smart_sampler.py --mesh-dir meshes/ --visualise

  # Custom scene bounds and z-levels:
  python smart_sampler.py --mesh-dir meshes/ --xmin -200 --xmax 200 \\
                           --ymin -200 --ymax 200 --z-levels 1,4,7,10

Dependencies
------------
  numpy, pandas, matplotlib (optional, for visualisation)
  trimesh, scipy (optional, for PLY-based building footprint extraction)
"""

from __future__ import annotations

import argparse
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Default configuration (IITM campus, v6 settings)
# ---------------------------------------------------------------------------

SCENE_LAT = 12.990628       # Campus origin latitude  (WGS-84)
SCENE_LON = 80.229689       # Campus origin longitude (WGS-84)
SCENE_ALT = 12.5            # Campus origin ellipsoidal altitude (m)

X_MIN, X_MAX = -387.0, 387.0   # OSM bounding box, East  (m from origin)
Y_MIN, Y_MAX = -335.0, 335.0   # OSM bounding box, North (m from origin)
SCENE_PADDING_M = 5.0           # Inset from bounding-box edges (avoids boundary artefacts)

Z_LEVELS = [1, 4, 7, 10, 13, 16, 19]   # Floor height levels (m)

WALL_OFFSET_M   = 2.5    # Distance of wall-halo points from building wall (m)
WALL_SPACING_M  = 5.0    # Along-wall spacing between wall-halo points (m)
CORNER_OFFSET_M = 2.5    # Diagonal offset from building corner vertex (m)
OPEN_GRID_M     = 15.0   # Open-space grid spacing (m)
OPEN_JITTER_M   = 5.0    # Random jitter radius for open-space grid (m)
DEDUP_GRID_M    = 3.0    # Deduplication grid size (m) — merges very close points

BUILDING_PLYS = [
    "map_osm_buildings_001.ply",
    "map_osm_buildings_002.ply",
]

FLOOR_MAP = {1: 0, 4: 1, 7: 2, 10: 3, 13: 4, 16: 5, 19: 6}


# ---------------------------------------------------------------------------
# Building footprint extraction
# ---------------------------------------------------------------------------

def load_building_footprints(mesh_dir: str | Path) -> list[np.ndarray]:
    """Return a list of (N,2) convex-hull polygons for each building component.

    Requires trimesh and scipy.  Falls back to an empty list (open-space-only
    sampling) when those packages are unavailable.
    """
    mesh_dir = Path(mesh_dir)
    try:
        import trimesh
        from scipy.spatial import ConvexHull
    except ImportError:
        print("[WARN] trimesh/scipy not installed — skipping building footprint extraction.")
        return []

    polygons: list[np.ndarray] = []
    MIN_AREA, MIN_VERTS = 12.0, 4

    for ply_name in BUILDING_PLYS:
        ply_path = mesh_dir / ply_name
        if not ply_path.exists():
            continue
        try:
            mesh = trimesh.load(str(ply_path), force="mesh", process=False)
        except Exception:
            continue

        components = mesh.split(only_watertight=False)
        if not hasattr(components, "__len__"):
            components = [components]

        for comp in components:
            if len(comp.vertices) < MIN_VERTS:
                continue
            xy = comp.vertices[:, :2]
            try:
                hull = ConvexHull(xy)
                poly = xy[hull.vertices]
            except Exception:
                poly = xy
            if len(poly) < 3 or abs(_polygon_area(poly)) < MIN_AREA:
                continue
            cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
            if X_MIN <= cx <= X_MAX and Y_MIN <= cy <= Y_MAX:
                polygons.append(np.asarray(poly, dtype=np.float64))

    print(f"[sampler] Loaded {len(polygons)} building footprints from {mesh_dir}")
    return polygons


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _point_in_any_polygon(px: float, py: float, polygons: list[np.ndarray]) -> bool:
    for poly in polygons:
        n, inside, j = len(poly), False, len(poly) - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > py) != (yj > py)) and px < (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi:
                inside = not inside
            j = i
        if inside:
            return True
    return False


# ---------------------------------------------------------------------------
# Point generators
# ---------------------------------------------------------------------------

def generate_wall_halo_points(polygons: list[np.ndarray]) -> list[tuple]:
    """Place points WALL_OFFSET_M outside each building wall at WALL_SPACING_M intervals."""
    pts: list[tuple] = []
    for poly in polygons:
        n = len(poly)
        cx, cy = float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
        for i in range(n):
            p1, p2 = poly[i], poly[(i + 1) % n]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            edge_len = math.hypot(dx, dy)
            if edge_len < WALL_SPACING_M:
                continue
            # Outward normal (pointing away from building centroid)
            nx, ny = -dy / edge_len, dx / edge_len
            midx, midy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            if (midx + nx - cx) ** 2 + (midy + ny - cy) ** 2 < (midx - cx) ** 2 + (midy - cy) ** 2:
                nx, ny = -nx, -ny
            num_pts = max(1, int(edge_len / WALL_SPACING_M))
            for k in range(num_pts):
                t = (k + 0.5) / num_pts
                mx, my = p1[0] + t * dx, p1[1] + t * dy
                pts.append((mx + nx * WALL_OFFSET_M, my + ny * WALL_OFFSET_M, "wall_halo"))
    return pts


def generate_corner_points(polygons: list[np.ndarray]) -> list[tuple]:
    """Place one point CORNER_OFFSET_M diagonally outward from each corner vertex."""
    pts: list[tuple] = []
    for poly in polygons:
        cx, cy = float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))
        for vx, vy in poly:
            dx, dy = vx - cx, vy - cy
            dist = math.hypot(dx, dy)
            if dist < 1e-6:
                continue
            pts.append((vx + (dx / dist) * CORNER_OFFSET_M, vy + (dy / dist) * CORNER_OFFSET_M, "corner"))
    return pts


def generate_open_space_points(
    polygons: list[np.ndarray], rng: np.random.Generator
) -> list[tuple]:
    """Jittered grid across non-building area within the scene bounding box."""
    pts: list[tuple] = []
    for x in np.arange(X_MIN, X_MAX, OPEN_GRID_M):
        for y in np.arange(Y_MIN, Y_MAX, OPEN_GRID_M):
            jx = float(rng.uniform(-OPEN_JITTER_M / 2, OPEN_JITTER_M / 2))
            jy = float(rng.uniform(-OPEN_JITTER_M / 2, OPEN_JITTER_M / 2))
            px, py = x + jx, y + jy
            if not _point_in_any_polygon(px, py, polygons):
                pts.append((px, py, "open_space"))
    return pts


def deduplicate_xy(pts_2d: list[tuple], grid_m: float = DEDUP_GRID_M) -> list[tuple]:
    """Remove spatially duplicate points by snapping to a coarse grid."""
    seen: set = set()
    dedup: list[tuple] = []
    for x, y, ptype in pts_2d:
        key = (round(x / grid_m), round(y / grid_m))
        if key not in seen:
            seen.add(key)
            dedup.append((x, y, ptype))
    return dedup


def expand_z_levels(pts_2d: list[tuple], z_levels: list[int] = Z_LEVELS) -> pd.DataFrame:
    """Replicate every 2D point at each floor height level."""
    rows = []
    for x, y, ptype in pts_2d:
        for z in z_levels:
            floor = FLOOR_MAP.get(int(z), int(round(z / 3)))
            rows.append({
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "z": float(z),
                "floor": floor,
                "point_type": ptype,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise(polygons: list[np.ndarray], pts_df: pd.DataFrame, out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[WARN] matplotlib not available — skipping visualisation.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(22, 11), gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    ax.set_facecolor("#d8e8d8")

    # Safe-zone boundary
    safe_rect = mpatches.Rectangle(
        (X_MIN + SCENE_PADDING_M, Y_MIN + SCENE_PADDING_M),
        (X_MAX - X_MIN - 2 * SCENE_PADDING_M),
        (Y_MAX - Y_MIN - 2 * SCENE_PADDING_M),
        linewidth=2, edgecolor="#FF8C00", facecolor="none",
        linestyle="--", zorder=3, label="Sampling safe zone",
    )
    ax.add_patch(safe_rect)

    for poly in polygons:
        closed = np.vstack([poly, poly[0]])
        ax.fill(closed[:, 0], closed[:, 1], color="#E0C9A6", edgecolor="#4A4A4A",
                linewidth=1.5, alpha=0.9, zorder=2)

    gf = pts_df[pts_df["z"] == 1]
    styles = {
        "wall_halo":  dict(c="#FF3366", s=15, edgecolor="white", linewidth=0.5, alpha=0.9, label="Wall halo"),
        "corner":     dict(c="#000080", s=25, edgecolor="white", linewidth=0.5, alpha=0.9, label="Corner"),
        "open_space": dict(c="#00CC66", s=10, alpha=0.6, label="Open space"),
    }
    for ptype, kw in styles.items():
        sub = gf[gf["point_type"] == ptype]
        if not sub.empty:
            ax.scatter(sub["x"], sub["y"], zorder=4, **kw)

    ax.set_xlim(X_MIN - 60, X_MAX + 60)
    ax.set_ylim(Y_MIN - 60, Y_MAX + 60)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Ground-floor sampling (z = 1 m)", fontsize=14, fontweight="bold")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.legend(loc="upper right", fontsize=10)
    stats = (
        f"Unique XY:   {len(gf):,}\n"
        f"Total 3-D:   {len(pts_df):,}\n"
        f"Z-levels:    {len(pts_df['z'].unique())}\n"
        f"Wall pad:    {WALL_OFFSET_M} m\n"
        f"Grid:        {OPEN_GRID_M} m"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=10, zorder=5)

    ax2 = axes[1]
    ptype_order = ["wall_halo", "corner", "open_space"]
    colors_bar  = ["#FF3366", "#000080", "#00CC66"]
    counts_df = pts_df.groupby(["z", "point_type"]).size().unstack(fill_value=0)
    counts_df = counts_df.reindex(columns=ptype_order, fill_value=0)
    bottoms = np.zeros(len(Z_LEVELS))
    for ptype, color in zip(ptype_order, colors_bar):
        vals = counts_df[ptype].reindex(Z_LEVELS, fill_value=0).values
        ax2.barh(Z_LEVELS, vals, height=2.0, left=bottoms, color=color,
                 alpha=0.8, label=ptype.replace("_", " ").title())
        bottoms += vals
    ax2.set_yticks(Z_LEVELS)
    ax2.set_xlabel("Receiver count", fontsize=11)
    ax2.set_ylabel("Height z-level (m)", fontsize=11)
    ax2.set_title("Receiver distribution per z-level", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, axis="x")

    plt.suptitle("GRAIL Smart Sampler — Receiver Placement", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"[sampler] Visualisation saved → {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GRAIL geometry-aware 3D GNSS receiver position generator."
    )
    ap.add_argument("--mesh-dir",  default="meshes",              help="Directory with building PLY files.")
    ap.add_argument("--out",       default="sampling_points.csv", help="Output CSV path.")
    ap.add_argument("--z-levels",  default=",".join(map(str, Z_LEVELS)),
                    help="Comma-separated floor heights in metres (default: %(default)s).")
    ap.add_argument("--xmin",      type=float, default=X_MIN)
    ap.add_argument("--xmax",      type=float, default=X_MAX)
    ap.add_argument("--ymin",      type=float, default=Y_MIN)
    ap.add_argument("--ymax",      type=float, default=Y_MAX)
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--visualise", action="store_true", help="Save a sampling-map PNG.")
    args = ap.parse_args()

    # Override module-level bounds if custom values provided
    global X_MIN, X_MAX, Y_MIN, Y_MAX
    X_MIN, X_MAX = args.xmin, args.xmax
    Y_MIN, Y_MAX = args.ymin, args.ymax
    z_levels = [int(z) for z in args.z_levels.split(",")]

    polygons = load_building_footprints(args.mesh_dir)

    rng = np.random.default_rng(args.seed)
    wall_pts   = generate_wall_halo_points(polygons)
    corner_pts = generate_corner_points(polygons)
    open_pts   = generate_open_space_points(polygons, rng)

    all_2d = wall_pts + corner_pts + open_pts

    # Strict padding cull: discard points too close to the scene boundary
    safe = (X_MIN + SCENE_PADDING_M, X_MAX - SCENE_PADDING_M,
            Y_MIN + SCENE_PADDING_M, Y_MAX - SCENE_PADDING_M)
    all_2d = [(x, y, t) for x, y, t in all_2d if safe[0] <= x <= safe[1] and safe[2] <= y <= safe[3]]
    all_2d = deduplicate_xy(all_2d)
    pts_df  = expand_z_levels(all_2d, z_levels)
    pts_df.to_csv(args.out, index=False)

    n_xy = len(pts_df["x"].drop_duplicates())
    print(f"[sampler] {n_xy:,} unique XY locations × {len(z_levels)} z-levels = {len(pts_df):,} 3-D points")
    print(f"[sampler] Output → {args.out}")

    if args.visualise:
        out_png = str(args.out).replace(".csv", "_map.png")
        visualise(polygons, pts_df, out_png)


if __name__ == "__main__":
    main()
