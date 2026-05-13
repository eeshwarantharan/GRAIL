from __future__ import annotations

from pathlib import Path
import numpy as np

from gnss_vim_sim.viz.plots import load_rows
from gnss_vim_sim.world.scene import MeshScene


def _series(rows: list[dict], key: str, default=np.nan) -> np.ndarray:
    vals = []
    for row in rows:
        val = row.get(key, default)
        vals.append(default if val == "" else val)
    return np.array(vals, dtype=float)


def _series_first(rows: list[dict], keys: list[str], default=np.nan) -> np.ndarray:
    for key in keys:
        if rows and key in rows[0]:
            return _series(rows, key, default)
    return np.full(len(rows), default, dtype=float)


def _mesh_traces(scene: MeshScene):
    import plotly.graph_objects as go

    if scene.mesh is not None and hasattr(scene.mesh, "faces") and len(scene.vertices):
        vertices = np.asarray(scene.vertices)
        faces = np.asarray(scene.mesh.faces)
        if len(faces) > 35_000:
            faces = faces[np.linspace(0, len(faces) - 1, 35_000).astype(int)]
        return [
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#b8bdc7",
                opacity=0.42,
                flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.75, specular=0.08, roughness=0.85),
                lightposition=dict(x=150, y=-220, z=450),
                name="OSM/Blender mesh",
                hoverinfo="skip",
            )
        ]

    if len(scene.vertices):
        step = max(1, len(scene.vertices) // 12_000)
        verts = scene.vertices[::step]
        return [
            go.Scatter3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                mode="markers",
                marker=dict(size=2, color="rgba(120,126,138,0.40)"),
                name="mesh vertices",
                hoverinfo="skip",
            )
        ]
    return []


def _heading_vectors(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.maximum(np.hypot(dx, dy), 1e-6)
    return dx / norm, dy / norm


def make_dashboard(log_path: Path, out_html: Path, scene: MeshScene | None = None, frame_step: int = 25) -> bool:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        return False

    rows = load_rows(log_path)
    if not rows:
        return False

    scene = scene or MeshScene(None)
    t = _series(rows, "t")
    x = _series(rows, "true_x")
    y = _series(rows, "true_y")
    z = _series(rows, "true_z")
    risk = _series_first(rows, ["model_score", "ml_risk"], 0.0)
    gnss_z = _series(rows, "gnss_z")
    vdop = _series(rows, "gnss_vdop")
    ml_z = _series(rows, "ml_integrity_z")
    fixed_z = _series(rows, "fixed_gnss_z")
    vdop_z = _series(rows, "vdop_chi2_z")
    always_z = _series(rows, "always_range_z")
    ml_r = _series(rows, "ml_integrity_gnss_r")
    hdx, hdy = _heading_vectors(x, y)

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.58, 0.22, 0.20],
        subplot_titles=(
            "3D Flight View",
            "Top-Down Mission and Model Score",
            "Altitude / Position Sensor Fusion",
            "Sensor Quality and Model Score",
            "Adaptive Measurement Covariance",
        ),
        horizontal_spacing=0.07,
        vertical_spacing=0.11,
    )

    for trace in _mesh_traces(scene):
        fig.add_trace(trace, row=1, col=1)

    if len(scene.vertices):
        bounds = scene.vertices[:, :2]
        z0 = np.nanmin(scene.vertices[:, 2])
        fig.add_trace(
            go.Mesh3d(
                x=[bounds[:, 0].min(), bounds[:, 0].max(), bounds[:, 0].max(), bounds[:, 0].min()],
                y=[bounds[:, 1].min(), bounds[:, 1].min(), bounds[:, 1].max(), bounds[:, 1].max()],
                z=[z0, z0, z0, z0],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="#eef1f5",
                opacity=0.55,
                name="ground plane",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="#101828", width=7),
            name="planned flight path",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(size=4, color=risk, colorscale="Turbo", cmin=0, cmax=1, colorbar=dict(title="model score", x=1.02)),
            name="model-score trail",
            text=[f"t={ti:.1f}s<br>model score={ri:.2f}" for ti, ri in zip(t, risk)],
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    drone_marker_index = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[x[0]],
            y=[y[0]],
            z=[z[0]],
            mode="markers+text",
            marker=dict(size=11, color="#16a34a", symbol="diamond"),
            text=["UAV"],
            textposition="top center",
            name="industrial UAV",
        ),
        row=1,
        col=1,
    )
    drone_cone_index = len(fig.data)
    fig.add_trace(
        go.Cone(
            x=[x[0]],
            y=[y[0]],
            z=[z[0] + 0.9],
            u=[hdx[0]],
            v=[hdy[0]],
            w=[0.0],
            sizemode="absolute",
            sizeref=5.5,
            anchor="tail",
            colorscale=[[0, "#16a34a"], [1, "#16a34a"]],
            showscale=False,
            name="heading",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    if len(scene.vertices):
        step = max(1, len(scene.vertices) // 12_000)
        verts = scene.vertices[::step]
        fig.add_trace(
            go.Scatter(
                x=verts[:, 0],
                y=verts[:, 1],
                mode="markers",
                marker=dict(size=2, color="rgba(100,108,120,0.28)"),
                name="map underlay",
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="ground track", line=dict(color="#101828", width=3)), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=5, color=risk, colorscale="Turbo", cmin=0, cmax=1),
            name="model-score map",
        ),
        row=2,
        col=1,
    )
    map_drone_index = len(fig.data)
    fig.add_trace(
        go.Scatter(x=[x[0]], y=[y[0]], mode="markers", marker=dict(size=14, color="#16a34a", symbol="diamond"), name="UAV map"),
        row=2,
        col=1,
    )

    fig.add_trace(go.Scatter(x=t, y=z, mode="lines", name="true z", line=dict(color="#101828", width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=gnss_z, mode="markers", name="GNSS z", marker=dict(size=4, color="rgba(90,90,90,0.55)")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=fixed_z, mode="lines", name="fixed covariance", line=dict(color="#4C72B0")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=vdop_z, mode="lines", name="quality/chi2", line=dict(color="#DD8452")), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=ml_z, mode="lines", name="model-adaptive", line=dict(color="#2A9D8F", width=3)), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=always_z, mode="lines", name="always range", line=dict(color="#C44E52")), row=2, col=2)

    fig.add_trace(go.Scatter(x=t, y=risk, mode="lines", name="model score", line=dict(color="#7B61FF", width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=vdop, mode="lines", name="sensor quality / VDOP", line=dict(color="#F97316", width=2)), row=3, col=1)
    fired_t = [row["t"] for row in rows if row.get("ml_integrity_range_fired")]
    fired_r = [row["ml_risk"] for row in rows if row.get("ml_integrity_range_fired")]
    fig.add_trace(
        go.Scatter(x=fired_t, y=fired_r, mode="markers", name="range aid fired", marker=dict(size=7, color="#E63946")),
        row=3,
        col=1,
    )

    fig.add_trace(go.Scatter(x=t, y=ml_r, mode="lines", name="model-adaptive Rz", line=dict(color="#0EA5E9", width=2)), row=3, col=2)
    fig.add_trace(go.Scatter(x=t, y=np.full_like(t, 25.0), mode="lines", name="fixed Rz", line=dict(color="#94A3B8", dash="dash")), row=3, col=2)

    frame_indices = list(range(0, len(rows), max(1, frame_step)))
    if frame_indices[-1] != len(rows) - 1:
        frame_indices.append(len(rows) - 1)
    fig.frames = [
        go.Frame(
            name=str(i),
            data=[
                go.Scatter3d(x=[x[i]], y=[y[i]], z=[z[i]], mode="markers+text", marker=dict(size=11, color="#16a34a", symbol="diamond"), text=["UAV"], textposition="top center"),
                go.Cone(x=[x[i]], y=[y[i]], z=[z[i] + 0.9], u=[hdx[i]], v=[hdy[i]], w=[0.0], sizemode="absolute", sizeref=5.5, anchor="tail", colorscale=[[0, "#16a34a"], [1, "#16a34a"]], showscale=False),
                go.Scatter(x=[x[i]], y=[y[i]], mode="markers", marker=dict(size=14, color="#16a34a", symbol="diamond")),
            ],
            traces=[drone_marker_index, drone_cone_index, map_drone_index],
            layout=go.Layout(title_text=f"UAV Mesh Simulation Replay | t={t[i]:.1f}s | model score={risk[i]:.2f}"),
        )
        for i in frame_indices
    ]

    fig.update_layout(
        title="UAV Mesh Simulation Replay | Sensor, Model, Estimator, and Energy Evidence",
        height=1280,
        template="plotly_white",
        margin=dict(l=35, r=35, t=95, b=30),
        scene=dict(
            xaxis_title="ENU east m",
            yaxis_title="ENU north m",
            zaxis_title="up m",
            aspectmode="data",
            camera=dict(eye=dict(x=1.25, y=-1.55, z=0.82), center=dict(x=0, y=0, z=-0.08)),
            bgcolor="#f8fafc",
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.01,
                "y": 1.055,
                "buttons": [
                    {"label": "Play Flight", "method": "animate", "args": [None, {"frame": {"duration": 38, "redraw": True}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
                ],
            }
        ],
        sliders=[
            {
                "steps": [
                    {"method": "animate", "label": f"{t[i]:.0f}s", "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                    for i in frame_indices
                ],
                "x": 0.18,
                "y": 1.05,
                "len": 0.76,
            }
        ],
        legend=dict(orientation="h", yanchor="bottom", y=-0.025, xanchor="center", x=0.5),
    )
    fig.update_xaxes(title_text="ENU east m", row=2, col=1)
    fig.update_yaxes(title_text="ENU north m", row=2, col=1, scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text="time s", row=2, col=2)
    fig.update_yaxes(title_text="altitude m", row=2, col=2)
    fig.update_xaxes(title_text="time s", row=3, col=1)
    fig.update_yaxes(title_text="model score / sensor quality", row=3, col=1)
    fig.update_xaxes(title_text="time s", row=3, col=2)
    fig.update_yaxes(title_text="Rz covariance", row=3, col=2)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs=True, auto_play=False)
    return True
