from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

from gnss_vim_sim.core.config import SimConfig
from gnss_vim_sim.core.coordinates import LocalFrame
from gnss_vim_sim.world.scene import MeshScene


def plan_mission_interactively(cfg: SimConfig, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("interactive mission planning requires matplotlib") from exc

    scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
    frame = LocalFrame(cfg.scene.origin_lat_deg, cfg.scene.origin_lon_deg, cfg.scene.origin_alt_m)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title("Click waypoints: first=takeoff, last=landing. Press Enter when done.")
    ax.set_xlabel("ENU east m")
    ax.set_ylabel("ENU north m")
    ax.grid(True, alpha=0.3)
    if len(scene.vertices):
        step = max(1, len(scene.vertices) // 10000)
        verts = scene.vertices[::step]
        ax.scatter(verts[:, 0], verts[:, 1], s=2, c="#888888", alpha=0.2, label="mesh")
    current = cfg.mission.waypoints
    if current:
        ax.plot([w["x"] for w in current], [w["y"] for w in current], "--", c="#888", label="current")
    ax.legend(loc="best")
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    if len(pts) < 2:
        raise RuntimeError("need at least takeoff and landing waypoints")

    waypoints = []
    for i, (x, y) in enumerate(pts):
        if i == 0:
            name, z = "takeoff", 1.0
        elif i == len(pts) - 1:
            name, z = "land", 1.0
        else:
            default_z = 7.0 if i % 2 else 10.0
            raw = input(f"Altitude for waypoint {i} at ENU ({x:.1f}, {y:.1f}) [{default_z:.1f} m]: ").strip()
            z = float(raw) if raw else default_z
            name = f"wp_{i:02d}"
        lat, lon, alt = frame.enu_to_lla([x, y, z])
        waypoints.append(
            {
                "name": name,
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "z": round(float(z), 3),
                "lat": round(float(lat), 9),
                "lon": round(float(lon), 9),
                "alt": round(float(alt), 3),
            }
        )

    raw_cfg = {
        "name": cfg.name,
        "seed": cfg.seed,
        "duration_s": cfg.duration_s,
        "dt_s": cfg.dt_s,
        "scene": asdict(cfg.scene),
        "mission": {
            "cruise_speed_mps": cfg.mission.cruise_speed_mps,
            "waypoint_acceptance_m": cfg.mission.waypoint_acceptance_m,
            "waypoints": waypoints,
        },
        "sensors": asdict(cfg.sensors),
        "fusion": asdict(cfg.fusion),
        "energy": asdict(cfg.energy),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw_cfg, indent=2))


def preview_mission(cfg: SimConfig, out_path: Path | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("mission preview requires matplotlib") from exc

    scene = MeshScene(cfg.resolve(cfg.scene.mesh_dir), cfg.resolve(cfg.scene.blend_file))
    frame = LocalFrame(cfg.scene.origin_lat_deg, cfg.scene.origin_lon_deg, cfg.scene.origin_alt_m)
    wps = cfg.mission.waypoints

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(121)
    ax.set_title("Top-down Mission")
    if len(scene.vertices):
        step = max(1, len(scene.vertices) // 10000)
        verts = scene.vertices[::step]
        ax.scatter(verts[:, 0], verts[:, 1], s=2, c="#888888", alpha=0.2)
    ax.plot([w["x"] for w in wps], [w["y"] for w in wps], "k-o")
    ax.scatter([wps[0]["x"]], [wps[0]["y"]], marker="*", s=180, c="green", label="takeoff")
    ax.scatter([wps[-1]["x"]], [wps[-1]["y"]], marker="X", s=130, c="black", label="landing")
    for w in wps:
        lat, lon, alt = frame.enu_to_lla([w["x"], w["y"], w["z"]])
        ax.annotate(f'{w["name"]}\n{w["z"]:.1f}m\n{lat:.6f},{lon:.6f}', (w["x"], w["y"]), fontsize=7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("ENU east m")
    ax.set_ylabel("ENU north m")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax3 = fig.add_subplot(122, projection="3d")
    ax3.plot([w["x"] for w in wps], [w["y"] for w in wps], [w["z"] for w in wps], "k-o")
    ax3.set_title("3D Mission Preview")
    ax3.set_xlabel("east m")
    ax3.set_ylabel("north m")
    ax3.set_zlabel("up m")
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    else:
        plt.show()
