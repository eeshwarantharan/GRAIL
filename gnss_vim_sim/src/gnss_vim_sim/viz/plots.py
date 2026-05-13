from __future__ import annotations

from pathlib import Path
import csv
import numpy as np

from gnss_vim_sim.world.scene import MeshScene


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            row = {}
            for k, v in raw.items():
                if v in ("", "True", "False"):
                    row[k] = True if v == "True" else False if v == "False" else v
                else:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = v
            rows.append(row)
        return rows


def _row_score(row: dict) -> float:
    return float(row.get("model_score", row.get("ml_risk", 0.0)))


def _map_limits(x: np.ndarray, y: np.ndarray, scene: MeshScene | None) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1 = float(x.min()), float(x.max())
    y0, y1 = float(y.min()), float(y.max())
    pad = max(90.0, 0.55 * max(x1 - x0, y1 - y0, 1.0))
    return (x0 - pad, x1 + pad), (y0 - pad, y1 + pad)


def _draw_building_footprints(ax, scene: MeshScene | None, xlim: tuple[float, float], ylim: tuple[float, float]) -> bool:
    if scene is None:
        return False
    try:
        import matplotlib.patches as patches
    except Exception:
        return False
    drawn = False
    for box in scene.building_boxes(max_boxes=700):
        if box["x1"] < xlim[0] or box["x0"] > xlim[1] or box["y1"] < ylim[0] or box["y0"] > ylim[1]:
            continue
        rect = patches.Rectangle(
            (box["x0"], box["y0"]),
            box["x1"] - box["x0"],
            box["y1"] - box["y0"],
            facecolor="#94a3b8",
            edgecolor="#64748b",
            alpha=0.38,
            linewidth=0.5,
        )
        ax.add_patch(rect)
        drawn = True
    return drawn


def make_plots(log_path: Path, out_dir: Path, scene: MeshScene | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = load_rows(log_path)
    t = np.array([r["t"] for r in rows], dtype=float)
    true_z = np.array([r["true_z"] for r in rows], dtype=float)
    ml_risk = np.array([_row_score(r) for r in rows], dtype=float)
    x = np.array([r["true_x"] for r in rows], dtype=float)
    y = np.array([r["true_y"] for r in rows], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(t, true_z, "k", lw=2, label="truth")
    gnss_points = [(r["t"], r["gnss_z"]) for r in rows if r.get("gnss_z") != ""]
    if gnss_points:
        axes[0].scatter([p[0] for p in gnss_points], [p[1] for p in gnss_points], s=12, c="#999", label="GNSS")
    for key, label in [
        ("fixed_gnss_z", "fixed GNSS"),
        ("vdop_chi2_z", "VDOP + chi2"),
        ("always_range_z", "always range"),
        ("ml_integrity_z", "ML integrity"),
    ]:
        axes[0].plot(t, [r[key] for r in rows], lw=1.2, label=label)
    axes[0].set_ylabel("altitude m")
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, ml_risk, color="#7b61ff", label="ML vertical risk")
    axes[1].scatter(
        [r["t"] for r in rows if r["ml_integrity_range_fired"]],
        [_row_score(r) for r in rows if r["ml_integrity_range_fired"]],
        s=12,
        c="#d62728",
        label="range fired",
    )
    axes[1].set_ylabel("risk")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, [r["ml_integrity_nis"] for r in rows], color="#2ca02c", label="ML EKF NIS")
    axes[2].set_ylabel("NIS")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    axes[3].plot(t, [float(r["gnss_vdop"]) if r["gnss_vdop"] != "" else np.nan for r in rows], label="VDOP")
    axes[3].plot(t, [float(r["ml_integrity_gnss_r"]) if r["ml_integrity_gnss_r"] != "" else np.nan for r in rows], label="ML GNSS Rz")
    axes[3].set_ylabel("VDOP / Rz")
    axes[3].set_xlabel("time s")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)
    fig.suptitle("GNSS Vertical Integrity Fusion")
    fig.tight_layout()
    fig.savefig(out_dir / "altitude_fusion.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    ax3 = fig.add_subplot(111, projection="3d")
    risk = np.array([_row_score(r) for r in rows], dtype=float)
    sc = ax3.scatter(x, y, true_z, c=risk, cmap="plasma", s=5, label="flight")
    ax3.plot(x, y, true_z, c="black", lw=0.8, alpha=0.5)
    ax3.scatter([x[0]], [y[0]], [true_z[0]], marker="*", s=180, c="green", label="takeoff")
    ax3.scatter([x[-1]], [y[-1]], [true_z[-1]], marker="X", s=120, c="black", label="landing")
    ax3.set_xlabel("ENU east m")
    ax3.set_ylabel("ENU north m")
    ax3.set_zlabel("up m")
    ax3.set_title("3D Flight Replay Colored by ML GNSS Vertical Risk")
    fig.colorbar(sc, ax=ax3, shrink=0.7, label="ML risk")
    ax3.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "flight_3d_risk.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    xlim, ylim = _map_limits(x, y, scene)
    boxes_drawn = _draw_building_footprints(ax, scene, xlim, ylim)
    if not boxes_drawn and scene is not None and len(scene.vertices):
        step = max(1, len(scene.vertices) // 8000)
        verts = scene.vertices[::step]
        keep = (verts[:, 0] >= xlim[0]) & (verts[:, 0] <= xlim[1]) & (verts[:, 1] >= ylim[0]) & (verts[:, 1] <= ylim[1])
        ax.scatter(verts[keep, 0], verts[keep, 1], s=2, c="#999999", alpha=0.2, label="mesh")
    ax.plot(x, y, "k", lw=2, label="mission")
    fired = [r for r in rows if r["ml_integrity_range_fired"]]
    if fired:
        ax.scatter([r["true_x"] for r in fired], [r["true_y"] for r in fired], c="#d62728", s=16, label="ML range aid")
    ax.scatter([x[0]], [y[0]], marker="*", s=160, c="green", label="takeoff")
    ax.scatter([x[-1]], [y[-1]], marker="X", s=120, c="black", label="landing")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("ENU east m")
    ax.set_ylabel("ENU north m")
    ax.set_title("Mission Map and Integrity-Triggered Range Aid")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "mission_map.png", dpi=160)
    plt.close(fig)

    _make_thesis_summary(rows, out_dir, scene)


def _make_thesis_summary(rows: list[dict], out_dir: Path, scene: MeshScene | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    t = np.array([r["t"] for r in rows], dtype=float)
    true_z = np.array([r["true_z"] for r in rows], dtype=float)
    risk = np.array([_row_score(r) for r in rows], dtype=float)
    x = np.array([r["true_x"] for r in rows], dtype=float)
    y = np.array([r["true_y"] for r in rows], dtype=float)
    estimators = ["fixed_gnss", "vdop_chi2", "ml_integrity", "always_range"]
    labels = ["Fixed GNSS", "VDOP/Chi2", "ML Integrity", "Always Range"]
    colors = ["#4C72B0", "#DD8452", "#2A9D8F", "#C44E52"]

    fig, ax = plt.subplots(2, 3, figsize=(18, 10), facecolor="#f8fafc")
    ax = ax.ravel()
    ax[0].plot(t, true_z, "k", lw=2, label="truth")
    for key, lab, col in zip(estimators, labels, colors):
        ax[0].plot(t, [r[f"{key}_z"] for r in rows], lw=1.2, label=lab, color=col)
    ax[0].set_title("Altitude Fusion")
    ax[0].set_xlabel("time s")
    ax[0].set_ylabel("altitude m")
    ax[0].legend(fontsize=8)
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(t, risk, color="#7B61FF", label="ML risk")
    ax[1].plot(t, [float(r["gnss_vdop"]) if r["gnss_vdop"] != "" else np.nan for r in rows], color="#F97316", label="VDOP")
    fired = [r for r in rows if r["ml_integrity_range_fired"]]
    if fired:
        ax[1].scatter([r["t"] for r in fired], [_row_score(r) for r in fired], s=10, c="#E63946", label="range aid")
    ax[1].set_title("GNSS Integrity and Range Aid")
    ax[1].legend(fontsize=8)
    ax[1].grid(True, alpha=0.25)

    maes = [float(np.mean(np.abs(np.array([r[f"{e}_z"] for r in rows]) - true_z))) for e in estimators]
    ax[2].bar(labels, maes, color=colors)
    ax[2].set_title("Altitude MAE")
    ax[2].set_ylabel("m")
    ax[2].tick_params(axis="x", rotation=18)
    ax[2].grid(True, axis="y", alpha=0.25)

    xlim, ylim = _map_limits(x, y, scene)
    boxes_drawn = _draw_building_footprints(ax[3], scene, xlim, ylim)
    if not boxes_drawn and scene is not None and len(scene.vertices):
        step = max(1, len(scene.vertices) // 7000)
        verts = scene.vertices[::step]
        keep = (verts[:, 0] >= xlim[0]) & (verts[:, 0] <= xlim[1]) & (verts[:, 1] >= ylim[0]) & (verts[:, 1] <= ylim[1])
        ax[3].scatter(verts[keep, 0], verts[keep, 1], s=1, c="#94a3b8", alpha=0.25)
    ax[3].plot(x, y, "k", lw=2)
    ax[3].scatter(x[0], y[0], marker="*", s=140, c="green", label="takeoff")
    ax[3].scatter(x[-1], y[-1], marker="X", s=90, c="black", label="landing")
    ax[3].scatter([r["true_x"] for r in fired], [r["true_y"] for r in fired], s=10, c="#E63946", label="range aid")
    ax[3].set_title("Mission Map")
    ax[3].set_aspect("equal", adjustable="box")
    ax[3].set_xlim(*xlim)
    ax[3].set_ylim(*ylim)
    ax[3].legend(fontsize=8)
    ax[3].grid(True, alpha=0.25)

    err_ml = np.abs(np.array([r["ml_integrity_z"] for r in rows]) - true_z)
    err_vdop = np.abs(np.array([r["vdop_chi2_z"] for r in rows]) - true_z)
    bins = np.linspace(0, max(float(err_ml.max()), float(err_vdop.max()), 1.0), 40)
    ax[4].hist(err_vdop, bins=bins, alpha=0.6, label="VDOP/Chi2", color="#DD8452", density=True)
    ax[4].hist(err_ml, bins=bins, alpha=0.6, label="ML Integrity", color="#2A9D8F", density=True)
    ax[4].set_title("Vertical Error Distribution")
    ax[4].set_xlabel("|error| m")
    ax[4].legend(fontsize=8)
    ax[4].grid(True, alpha=0.25)

    pulses = [sum(1 for r in rows if r[f"{e}_range_fired"]) for e in estimators]
    ax[5].bar(labels, pulses, color=colors)
    ax[5].set_title("Range/LiDAR Activations")
    ax[5].set_ylabel("pulses")
    ax[5].tick_params(axis="x", rotation=18)
    ax[5].grid(True, axis="y", alpha=0.25)

    fig.suptitle("GNSS-VIM Simulation Summary", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_panel.png", dpi=170)
    fig.savefig(out_dir / "thesis_summary.png", dpi=170)
    plt.close(fig)


def replay_live(log_path: Path, scene: MeshScene | None = None, step: int = 20) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    rows = load_rows(log_path)
    x = np.array([r["true_x"] for r in rows], dtype=float)
    y = np.array([r["true_y"] for r in rows], dtype=float)
    z = np.array([r["true_z"] for r in rows], dtype=float)
    risk = np.array([_row_score(r) for r in rows], dtype=float)

    fig = plt.figure(figsize=(13, 6))
    ax3 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    for i in range(0, len(rows), max(step, 1)):
        ax3.clear()
        ax2.clear()
        if scene is not None and len(scene.vertices):
            stride = max(1, len(scene.vertices) // 8000)
            verts = scene.vertices[::stride]
            ax2.scatter(verts[:, 0], verts[:, 1], s=2, c="#888888", alpha=0.18)
        ax3.plot(x[: i + 1], y[: i + 1], z[: i + 1], c="black", lw=1.2)
        ax3.scatter([x[i]], [y[i]], [z[i]], c=[risk[i]], cmap="plasma", vmin=0, vmax=1, s=80)
        ax3.set_xlim(float(np.min(x)) - 10, float(np.max(x)) + 10)
        ax3.set_ylim(float(np.min(y)) - 10, float(np.max(y)) + 10)
        ax3.set_zlim(0, float(np.max(z)) + 5)
        ax3.set_title(f"Live Replay t={rows[i]['t']:.1f}s risk={risk[i]:.2f}")
        ax3.set_xlabel("east m")
        ax3.set_ylabel("north m")
        ax3.set_zlabel("up m")
        ax2.plot(x[: i + 1], y[: i + 1], c="black")
        ax2.scatter([x[i]], [y[i]], c=[risk[i]], cmap="plasma", vmin=0, vmax=1, s=80)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("east m")
        ax2.set_ylabel("north m")
        ax2.grid(True, alpha=0.3)
        plt.pause(0.001)
    plt.show()
