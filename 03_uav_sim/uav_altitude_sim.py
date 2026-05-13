"""
uav_altitude_sim.py  —  GRAIL UAV Altitude Simulation
======================================================

Honest 4-EKF comparison for altitude-sensor fusion on a UAV flying
through an urban canyon scene (IITM campus by default).

EKF variants
------------
  A  PX4-standard chi-squared gate with LiDAR fallback (baseline)
  B  ML adaptive-R gate (GRAIL proposal) — proactive LiDAR trigger
  C  Always-on LiDAR (performance upper bound, energy lower bound)
  D  Blind GNSS only (no gate, no LiDAR — worst-case GNSS failure)

Key result (IITM 300-s mission)
  EKF-B canyon MAE  = 0.097 m  vs  EKF-C = 0.085 m  (12 mm penalty)
  EKF-B LiDAR fires 41.9 % less often than EKF-C  →  857 J saved

Usage
-----
  python uav_altitude_sim.py                          # interactive waypoint picker
  python uav_altitude_sim.py --no-interact            # default 8-waypoint mission
  python uav_altitude_sim.py --smoke                  # 30-second quick test
  python uav_altitude_sim.py --feat-csv path/to.csv   # custom feature dataset
  python uav_altitude_sim.py --model path/to.pkl      # custom ML model
  python uav_altitude_sim.py --duration 600 --no-live # longer silent run
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Default paths (relative to this script) ────────────────────────────────────
_HERE = Path(__file__).parent
_GRAIL = _HERE.parent

DEFAULT_FEAT_CSV = str(_GRAIL / "data" / "gnss_ml_features.csv")
DEFAULT_MODEL    = str(_GRAIL / "models" / "xgboost_classifier_v2.pkl")
DEFAULT_MESH_DIR = str(_GRAIL / "gnss_vim_sim" / "demo" / "meshes")
DEFAULT_OUT_LOG  = str(_HERE / "flight_log.csv")
DEFAULT_OUT_RES  = str(_HERE / "sim_results.png")
DEFAULT_OUT_TRAJ = str(_HERE / "sim_3d.png")

# ── Scene bounds (IITM OSM export in ENU metres) ────────────────────────────────
X_MIN, X_MAX = -387.0, 387.0
Y_MIN, Y_MAX = -335.0, 335.0

# ── Simulation timing ──────────────────────────────────────────────────────────
DT           = 0.01    # 100 Hz inner loop
GNSS_RATE    = 1.0     # Hz
BARO_RATE    = 50.0    # Hz (downsampled to 10 Hz in EKF)
SIM_DURATION = 300.0   # seconds

# ── Sensor noise (matched to real hardware) ────────────────────────────────────
IMU_ACC_NOISE  = 0.05   # m/s²  (MPU-6000 typical)
IMU_BIAS_WALK  = 1e-4   # m/s² / √s
BARO_NOISE     = 0.30   # m     (MS5611)
BARO_BLDG_ERR  = 2.0    # m     Bernoulli effect near wall, 1σ
GNSS_Z_NOISE   = 5.0    # m     1σ open-sky vertical
LIDAR_NOISE    = 0.03   # m     VL53L1X / Garmin LidarLite

# ── EKF noise matrices ─────────────────────────────────────────────────────────
Q_EKF       = np.diag([1e-4, 1e-2, 1e-5])  # process noise [z, vz, baro_bias]
R_BARO      = np.array([[0.09]])            # σ = 0.3 m
R_GNSS_BASE = np.array([[25.0]])            # σ = 5 m open-sky
R_LIDAR     = np.array([[0.001]])           # σ = 0.03 m
CHI2_THRESH = 9.0                           # 3σ chi-squared gate
R_ALPHA     = 8.0                           # ML R-inflation exponent α
ML_THRESH   = 0.40                          # LiDAR trigger threshold τ

# ── Sensor power (watts) ───────────────────────────────────────────────────────
W_GNSS  = 0.05   # GNSS receiver
W_LIDAR = 8.0    # small ToF LiDAR (Garmin v3HP: 0.5W; Velodyne: 30W)
W_ML    = 0.5    # Jetson Nano inference

# ── 22 GNSS fingerprint features ──────────────────────────────────────────────
FEAT_COLS = [
    "n_sats", "mean_cn0", "std_cn0", "min_cn0", "range_cn0", "canopy_cn0",
    "mean_elev", "min_elev", "max_elev", "std_elev", "elev_spread",
    "vdop", "mean_los_ratio", "min_los_ratio", "std_los_ratio",
    "phase_locked_frac", "mean_delay_ns", "max_delay_ns", "std_delay_ns",
    "mean_mp_error", "std_mp_error", "max_mp_error",
]

# ── Default IITM waypoints ─────────────────────────────────────────────────────
WAYPOINTS_IITM = [
    [   0,    0,  1.0],   # takeoff — open sky near centre
    [-150,  -80,  1.0],   # south-west open area
    [-100,   50,  7.0],   # near western building cluster
    [  50,  100, 10.0],   # dense cluster — urban canyon
    [ 150,   80, 10.0],   # near large north-east building
    [ 100,  -50,  1.0],   # return to open sky (east)
    [  50, -150,  1.0],   # open south
    [   0,    0,  1.0],   # land at origin
]

PALETTE = {
    "bg":   "#FAFAFA",
    "grid": "#EBEBEB",
    "text": "#1A1A2E",
    "bldg": "#D2B48C",
    "A":    "#4C72B0",  # chi-squared
    "B":    "#2A9D8F",  # ML-gated  (green)
    "C":    "#E63946",  # always-LiDAR
    "D":    "#F4A261",  # GNSS-only
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SCENE / PLY MESH
# ══════════════════════════════════════════════════════════════════════════════

def load_ply_scene(mesh_dir: str) -> tuple:
    """
    Load all PLY files from mesh_dir.

    Returns
    -------
    scene       : trimesh.Scene or None
    bldg_polys  : list of 2-D convex hull arrays (one per building component)
    """
    try:
        import trimesh
        from scipy.spatial import ConvexHull
    except ImportError:
        print("  [info] trimesh / scipy not installed — running without 3D mesh")
        return None, []

    if not os.path.isdir(mesh_dir):
        print(f"  [info] mesh_dir not found: {mesh_dir!r} — running without mesh")
        return None, []

    ply_files = sorted(f for f in os.listdir(mesh_dir) if f.endswith(".ply"))
    if not ply_files:
        print(f"  [info] No PLY files in {mesh_dir!r}")
        return None, []

    meshes, bldg_polys = [], []
    for fname in ply_files:
        path = os.path.join(mesh_dir, fname)
        try:
            m = trimesh.load(path, force="mesh", process=False)
            meshes.append(m)
        except Exception:
            pass

    scene = trimesh.util.concatenate(meshes) if meshes else None
    if scene:
        print(f"  PLY scene: {len(ply_files)} files, {len(scene.vertices):,} vertices")

    # Building footprints for 2-D map
    bldg_files = [f for f in ply_files if "building" in f.lower()]
    for fname in bldg_files:
        path = os.path.join(mesh_dir, fname)
        try:
            mesh = trimesh.load(path, force="mesh", process=False)
            components = mesh.split(only_watertight=False)
            if not hasattr(components, "__len__"):
                components = [components]
            for comp in components:
                if len(comp.vertices) < 4:
                    continue
                xy = comp.vertices[:, :2]
                cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
                if not (X_MIN <= cx <= X_MAX and Y_MIN <= cy <= Y_MAX):
                    continue
                try:
                    hull = ConvexHull(xy)
                    bldg_polys.append(xy[hull.vertices])
                except Exception:
                    pass
        except Exception:
            pass

    print(f"  Building footprints: {len(bldg_polys)}")
    return scene, bldg_polys


def lidar_range(scene, pos: np.ndarray) -> float:
    """Downward ray-cast. Returns AGL height in metres."""
    if scene is None:
        return max(0.1, float(pos[2]))
    try:
        locs, _, _ = scene.ray.intersects_location(
            np.array([pos]), np.array([[0., 0., -1.]]))
        if len(locs):
            return max(0.1, float(np.linalg.norm(locs - pos, axis=1).min()))
    except Exception:
        pass
    return max(0.1, float(pos[2]))


def wall_proximity(pos: np.ndarray, scene) -> float:
    """
    Fraction in [0, 1]: 1 = directly against building, 0 = open field.
    Used to scale barometric Bernoulli-effect error near walls.
    """
    if scene is None:
        return 0.0
    min_d = 999.0
    for az in range(0, 360, 45):
        rad = math.radians(az)
        ray = np.array([[math.cos(rad), math.sin(rad), 0.0]])
        try:
            locs, _, _ = scene.ray.intersects_location(np.array([pos]), ray)
            if len(locs):
                min_d = min(min_d, float(np.linalg.norm(locs[0] - pos)))
        except Exception:
            pass
    return float(np.clip(1.0 - min_d / 20.0, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ML MODEL & GNSS FEATURE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def load_assets(feat_csv: str, model_pkl: str) -> tuple:
    """
    Load the XGBoost classifier and the Sionna GNSS feature dataset.

    Falls back gracefully if files are missing:
      - model absent → VDOP heuristic
      - feat_csv absent → physics-based synthetic GNSS
    """
    model = None
    if os.path.exists(model_pkl):
        with open(model_pkl, "rb") as f:
            model = pickle.load(f)
        print(f"  ML model: {model_pkl}")
    else:
        print(f"  [warn] ML model not found at {model_pkl!r} — using VDOP heuristic")

    feat_df = kdtree = None
    if os.path.exists(feat_csv):
        feat_df = pd.read_csv(feat_csv)
        print(f"  GNSS features: {len(feat_df):,} records from {feat_csv!r}")
        if "true_z" in feat_df.columns:
            from scipy.spatial import cKDTree
            kdtree = cKDTree(feat_df[["true_z"]].values)
            print("  1-D KD-Tree built (altitude-based multipath lookup)")
        else:
            print("  [warn] 'true_z' column absent — using physics fallback")
    else:
        print(f"  [warn] Feature CSV not found at {feat_csv!r} — using physics fallback")

    return model, feat_df, kdtree


def lookup_gnss_features(feat_df, kdtree, pos_xyz: list, rng) -> dict:
    """
    Return a fingerprint feature dict for the UAV's current altitude.

    Uses 1-D altitude KD-Tree: query the 50 nearest records by Z, then
    pick one randomly to capture multipath diversity at that height.
    If the dataset is unavailable, returns a physics-based approximation.
    """
    if feat_df is None or kdtree is None:
        z = pos_xyz[2]
        vdop  = max(1.0, 2.0 + z * 0.15 + rng.normal(0, 0.3))
        delay = max(0.0, z * 0.9 + rng.normal(0, 1.5))
        return {
            "n_sats":       max(4, int(8 - z * 0.1)),
            "mean_cn0":     max(15.0, 40 - z * 0.6),
            "vdop":         vdop,
            "mean_delay_ns": delay,
            "mean_elev":    45.0,
        }

    _, idx = kdtree.query(np.array([pos_xyz[2]]), k=min(50, len(feat_df)))
    row = feat_df.iloc[rng.choice(np.atleast_1d(idx))]
    feats = {c: float(row[c]) for c in FEAT_COLS if c in feat_df.columns}

    # Add inter-epoch noise to prevent identical sequential readings
    feats["mean_cn0"]       = feats.get("mean_cn0", 35.0)       + rng.normal(0, 1.5)
    feats["vdop"]           = feats.get("vdop", 2.0)            + rng.normal(0, 0.2)
    feats["mean_delay_ns"]  = max(0, feats.get("mean_delay_ns", 0) + rng.normal(0, 0.5))
    return feats


def ml_probability(model, feats: dict) -> float:
    """
    P(|z_error| > 3 m) = probability that GNSS altitude is untrustworthy.

    When the trained model is absent, falls back to a VDOP/delay heuristic:
      p = clip((VDOP - 1.5) / 4 + delay_ns / 20, 0, 1)
    """
    if model is None:
        v = feats.get("vdop", 2.0)
        d = feats.get("mean_delay_ns", 0.0)
        return float(np.clip((v - 1.5) / 4.0 + d / 20.0, 0.0, 1.0))

    avail = [c for c in FEAT_COLS if c in feats]
    x = np.array([[feats.get(c, 0.0) for c in avail]], dtype=np.float32)
    try:
        return float(model.predict_proba(x)[0, 1])
    except Exception:
        return 0.5


def gnss_altitude_measurement(true_z: float, feats: dict, rng) -> tuple[float, float]:
    """
    Simulate the GNSS altitude measurement a receiver would report.

    Physics model:
        z_gnss = z_true + Δz_mp + ε_noise

    Multipath bias (projects delay spread onto vertical):
        Δz_mp = c × τ_rms × sin(ē)
    where ē is the mean satellite elevation angle.
    """
    tau_s   = feats.get("mean_delay_ns", 0.0) * 1e-9
    el_rad  = math.radians(max(5.0, feats.get("mean_elev", 45.0)))
    mp_bias = 3e8 * tau_s * math.sin(el_rad)
    noise   = rng.normal(0, GNSS_Z_NOISE)
    return true_z + mp_bias + noise, mp_bias


# ══════════════════════════════════════════════════════════════════════════════
# 3.  3-STATE ALTITUDE EKF
#     State: x = [z, v_z, b_baro]
#     Each of the four EKF variants inherits this class.
# ══════════════════════════════════════════════════════════════════════════════

class AltEKF:
    """
    3-state altitude EKF:  x = [z, v_z, baro_bias]^T

    Measurement models
    ------------------
    Baro   : z_baro ≈ z + baro_bias         H_baro = [1, 0, 1]
    GNSS   : z_gnss ≈ z                     H_z    = [1, 0, 0]
    LiDAR  : z_lidar≈ z                     H_z    = [1, 0, 0]

    This matches the altitude sub-state in PX4 EKF2 (docs §6.2).
    """

    H_BARO = np.array([[1., 0., 1.]])
    H_Z    = np.array([[1., 0., 0.]])

    def __init__(self, z0: float, name: str):
        self.name          = name
        self.x             = np.array([z0, 0.0, 0.0])
        self.P             = np.diag([4.0, 1.0, 0.25])
        self.lidar_pulses  = 0
        self.gnss_accepted = 0
        self.gnss_total    = 0

    # ── Predict ────────────────────────────────────────────────────────────────

    def predict(self, az: float, dt: float):
        F = np.array([[1., dt, 0.],
                      [0.,  1., 0.],
                      [0.,  0., 1.]])
        B = np.array([0.5 * dt**2, dt, 0.])
        self.x = F @ self.x + B * az
        self.P = F @ self.P @ F.T + Q_EKF * dt

    # ── Internal update ────────────────────────────────────────────────────────

    def _update(self, z_meas: float, H: np.ndarray, R: np.ndarray):
        nu  = z_meas - float((H @ self.x)[0])
        S   = float((H @ self.P @ H.T)[0, 0]) + float(R[0, 0])
        K   = (self.P @ H.T) / S
        self.x = self.x + K.flatten() * nu
        self.P = (np.eye(3) - K @ H) @ self.P
        return nu, S

    # ── Sensor updates ─────────────────────────────────────────────────────────

    def update_baro(self, z_baro: float):
        self._update(z_baro, self.H_BARO, R_BARO)

    def update_lidar(self, z_lidar: float):
        self._update(z_lidar, self.H_Z, R_LIDAR)
        self.lidar_pulses += 1

    # ── GNSS update (differs by EKF variant) ──────────────────────────────────

    def update_gnss_chi2(self, z_gnss: float, z_lidar: float) -> bool:
        """
        EKF-A (PX4 baseline): chi-squared innovation gate.
        Rejects GNSS if ν²/S ≥ χ²(3σ) = 9.0 and fires LiDAR as safety fallback.
        """
        self.gnss_total += 1
        nu = z_gnss - float((self.H_Z @ self.x)[0])
        S  = float((self.H_Z @ self.P @ self.H_Z.T)[0, 0]) + float(R_GNSS_BASE[0, 0])
        if nu**2 / S < CHI2_THRESH:
            self._update(z_gnss, self.H_Z, R_GNSS_BASE)
            self.gnss_accepted += 1
            return False
        else:
            self.update_lidar(z_lidar)
            return True

    def update_gnss_ml(self, z_gnss: float, z_lidar: float, ml_p: float) -> bool:
        """
        EKF-B (GRAIL): ML adaptive-R gate.
        R_gnss,k = R_base · exp(α · p_k)

        p_k → 1 : R → ∞, Kalman gain → 0  (GNSS silently down-weighted)
        p_k → 0 : R = R_base              (GNSS fully trusted)
        LiDAR fires only when ml_p > τ (proactive, not reactive).
        """
        self.gnss_total += 1
        R_adaptive = R_GNSS_BASE * math.exp(R_ALPHA * ml_p)
        self._update(z_gnss, self.H_Z, R_adaptive)
        if ml_p < ML_THRESH:
            self.gnss_accepted += 1
        lidar_fired = ml_p > ML_THRESH
        if lidar_fired:
            self.update_lidar(z_lidar)
        return lidar_fired

    def update_gnss_always_lidar(self, z_lidar: float) -> bool:
        """EKF-C: Ignore GNSS; always fuse LiDAR. Performance upper bound."""
        self.update_lidar(z_lidar)
        return True

    def update_gnss_blind(self, z_gnss: float) -> bool:
        """EKF-D: Accept all GNSS without gating. Worst-case GNSS failure."""
        self.gnss_total += 1
        self.gnss_accepted += 1
        self._update(z_gnss, self.H_Z, R_GNSS_BASE)
        return False

    @property
    def altitude(self) -> float:
        return float(self.x[0])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  POINT-MASS DRONE PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

class Drone:
    """
    Point-mass drone with cascaded P-controller.
    Sufficient for altitude-filter validation — not for full attitude dynamics.
    """

    def __init__(self, pos0, mass: float = 1.5, drag: float = 0.25, v_max: float = 5.0):
        self.pos   = np.array(pos0, dtype=float)
        self.vel   = np.zeros(3)
        self.mass  = mass
        self.drag  = drag
        self.v_max = v_max

    def step(self, target, dt: float, rng) -> float:
        err  = np.array(target, dtype=float) - self.pos
        dist = np.linalg.norm(err)
        if dist < 0.3:
            self.vel *= 0.8
            self.pos += self.vel * dt
            return float(self.pos[2])

        v_des = (err / dist) * min(self.v_max, dist * 1.5)
        accel = (v_des - self.vel) * 2.0 - self.drag * self.vel / self.mass
        accel += rng.normal(0, 0.01, size=3)

        self.vel += accel * dt
        self.pos += self.vel * dt + 0.5 * accel * dt**2
        return float(self.pos[2])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(
    scene,
    model,
    feat_df,
    kdtree,
    waypoints: list,
    duration: float = SIM_DURATION,
    live_plot: bool = True,
) -> tuple[pd.DataFrame, list, dict]:
    """
    Run the 4-EKF UAV altitude simulation.

    Returns
    -------
    df         : per-tick flight log DataFrame
    ekfs       : [EKF-A, EKF-B, EKF-C, EKF-D]
    sens_J     : sensor-only energy (Joules) per EKF variant
    """
    rng = np.random.default_rng(42)

    drone = Drone(waypoints[0])
    ekf_a = AltEKF(waypoints[0][2], "EKF-A")
    ekf_b = AltEKF(waypoints[0][2], "EKF-B")
    ekf_c = AltEKF(waypoints[0][2], "EKF-C")
    ekf_d = AltEKF(waypoints[0][2], "EKF-D")
    ekfs  = [ekf_a, ekf_b, ekf_c, ekf_d]

    sens_J = {e.name: 0.0 for e in ekfs}

    imu_bias = 0.0
    t = gnss_t = baro_t = 0.0
    wp_idx = 0

    cols = ["t", "x", "y", "true_z", "z_gnss", "mp_bias",
            "ml_p", "bprox",
            "EKF-A", "EKF-B", "EKF-C", "EKF-D",
            "lidar_A", "lidar_B", "lidar_C", "lidar_D",
            "gnss_err"]
    log = {c: [] for c in cols}

    if live_plot:
        plt.ion()
        fig_live, ax_live = plt.subplots(figsize=(12, 4))
        fig_live.canvas.manager.set_window_title("Live Altitude — UAV Sim")
        ax_live.set_xlabel("Time (s)")
        ax_live.set_ylabel("Altitude (m)")
        t_live, z_live_A, z_live_B, z_live_T = [], [], [], []

    print(f"\nRunning 4-EKF simulation: {duration:.0f} s, {len(waypoints)} waypoints")
    t0_wall = time.time()

    while t < duration:
        target = waypoints[min(wp_idx, len(waypoints) - 1)]
        if np.linalg.norm(np.array(target) - drone.pos) < 1.5 and wp_idx < len(waypoints) - 1:
            wp_idx += 1

        true_z = drone.step(target, DT, rng)

        # IMU
        imu_bias += rng.normal(0, IMU_BIAS_WALK * math.sqrt(DT))
        meas_az   = imu_bias + rng.normal(0, IMU_ACC_NOISE)

        for e in ekfs:
            e.predict(meas_az, DT)

        bprox = wall_proximity(drone.pos, scene) if scene else 0.0
        lidar_flags = [False, False, False, False]

        # Barometer (10 Hz in EKF, 50 Hz modelled)
        baro_t -= DT
        if baro_t <= 0:
            baro_t  = 1.0 / min(BARO_RATE, 10.0)
            z_baro  = true_z + rng.normal(0, BARO_NOISE) + bprox * rng.normal(0, BARO_BLDG_ERR)
            for e in ekfs:
                e.update_baro(z_baro)

        # GNSS + LiDAR (1 Hz)
        ml_p = 0.0
        z_gnss = gnss_err = mp_bias = float("nan")

        gnss_t -= DT
        if gnss_t <= 0:
            gnss_t = 1.0 / GNSS_RATE

            feats            = lookup_gnss_features(feat_df, kdtree, drone.pos.tolist(), rng)
            z_gnss, mp_bias  = gnss_altitude_measurement(true_z, feats, rng)
            gnss_err         = abs(z_gnss - true_z)
            ml_p             = ml_probability(model, feats)
            z_lidar          = lidar_range(scene, drone.pos) + rng.normal(0, LIDAR_NOISE)

            lidar_flags[0] = ekf_a.update_gnss_chi2(z_gnss, z_lidar)
            lidar_flags[1] = ekf_b.update_gnss_ml(z_gnss, z_lidar, ml_p)
            lidar_flags[2] = ekf_c.update_gnss_always_lidar(z_lidar)
            lidar_flags[3] = ekf_d.update_gnss_blind(z_gnss)

            # Sensor-only energy (Joules per 1-Hz GNSS epoch)
            sens_J["EKF-A"] += W_GNSS + (W_LIDAR if lidar_flags[0] else 0)
            sens_J["EKF-B"] += W_GNSS + W_ML + (W_LIDAR if lidar_flags[1] else 0)
            sens_J["EKF-C"] += W_GNSS + W_LIDAR
            sens_J["EKF-D"] += W_GNSS

        # Append tick to log
        log["t"].append(t)
        log["x"].append(float(drone.pos[0]))
        log["y"].append(float(drone.pos[1]))
        log["true_z"].append(true_z)
        log["z_gnss"].append(z_gnss)
        log["mp_bias"].append(mp_bias)
        log["ml_p"].append(ml_p)
        log["bprox"].append(bprox)
        log["gnss_err"].append(gnss_err)
        for e in ekfs:
            log[e.name].append(e.altitude)
        for i, e in enumerate(ekfs):
            log[f"lidar_{e.name[-1]}"].append(float(lidar_flags[i]))

        # Live plot every 50 ticks
        if live_plot and len(log["t"]) % 50 == 0:
            t_live.append(t)
            z_live_T.append(true_z)
            z_live_A.append(ekf_a.altitude)
            z_live_B.append(ekf_b.altitude)
            ax_live.cla()
            ax_live.plot(t_live, z_live_T, "k-", lw=2, label="True")
            ax_live.plot(t_live, z_live_A, c=PALETTE["A"], lw=1.2,
                         label=f"EKF-A (χ²) {abs(ekf_a.altitude-true_z):.2f} m err")
            ax_live.plot(t_live, z_live_B, c=PALETTE["B"], lw=1.5,
                         label=f"EKF-B (ML) {abs(ekf_b.altitude-true_z):.2f} m err")
            ax_live.set_xlabel("Time (s)")
            ax_live.set_ylabel("Altitude (m)")
            ax_live.legend(fontsize=8, loc="upper right")
            ax_live.grid(True, color=PALETTE["grid"])
            fig_live.canvas.draw()
            plt.pause(0.001)

        t += DT

    if live_plot:
        plt.ioff()
        plt.close(fig_live)

    wall = time.time() - t0_wall
    print(f"  Done in {wall:.1f} s real-time ({duration/wall:.0f}× speed-up)")
    return pd.DataFrame(log), ekfs, sens_J


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PUBLICATION-QUALITY PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _style_ax(ax, title: str, xl: str, yl: str):
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(True, color=PALETTE["grid"], lw=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_color("#DDD"); sp.set_linewidth(0.7)
    ax.set_title(title, fontsize=10, fontweight="bold", color=PALETTE["text"], pad=6)
    ax.set_xlabel(xl, fontsize=9, color="#555")
    ax.set_ylabel(yl, fontsize=9, color="#555")
    ax.tick_params(labelsize=8)


def plot_results(df: pd.DataFrame, ekfs: list, sens_J: dict,
                 waypoints: list, out_path: str):
    """6-panel results figure saved to out_path."""
    t      = df["t"].values
    true_z = df["true_z"].values
    canyon = true_z > 3.5

    fig = plt.figure(figsize=(20, 16), facecolor=PALETTE["bg"])
    fig.suptitle(
        "UAV Altitude Simulation — Honest 4-EKF Comparison\n"
        "ML-gated EKF-B vs chi² baseline vs Always-LiDAR vs Unguarded GNSS",
        fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.99,
    )
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

    # P1: Altitude timeline
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, true_z, "k-", lw=2.5, label="True altitude", zorder=5)
    styles = [("-", 3), ("-", 4), ("--", 2), (":", 1)]
    pal_keys = ["A", "B", "C", "D"]
    for e, pk, (ls, zo) in zip(ekfs, pal_keys, styles):
        ax1.plot(t, df[e.name], color=PALETTE[pk], lw=1.5, ls=ls, alpha=0.85,
                 label=e.name, zorder=zo)
    gnss_mask = ~np.isnan(df["z_gnss"].values)
    ax1.scatter(t[gnss_mask], df["z_gnss"].values[gnss_mask],
                s=12, c="#F4A261", alpha=0.4, label="GNSS z (raw)", zorder=2)
    in_canyon = False
    for i in range(len(t)):
        if canyon[i] and not in_canyon:
            c_start = t[i]; in_canyon = True
        elif not canyon[i] and in_canyon:
            ax1.axvspan(c_start, t[i], alpha=0.06, color="#E63946")
            in_canyon = False
    _style_ax(ax1, "Altitude: True vs 4 EKF variants  (red shading = canyon segment)",
              "Time (s)", "Altitude (m)")
    ax1.legend(fontsize=8, ncol=7, loc="upper right")

    # P2: ML gate probability
    ax2 = fig.add_subplot(gs[1, 0])
    ml_p_vals = df["ml_p"].values
    ax2.plot(t, ml_p_vals, c="#7F77DD", lw=1.2, label="p_k = P(GNSS untrustworthy)")
    ax2.fill_between(t, ml_p_vals, ML_THRESH, where=ml_p_vals > ML_THRESH,
                     alpha=0.35, color=PALETTE["C"], label="LiDAR fires (EKF-B)")
    ax2.axhline(ML_THRESH, color=PALETTE["C"], lw=1.3, linestyle="--",
                label=f"Threshold τ = {ML_THRESH}")
    ax2.set_ylim(-0.05, 1.1)
    gnss_rows = df[gnss_mask]
    c_p = gnss_rows.loc[gnss_rows["true_z"] > 3.5, "ml_p"].values
    o_p = gnss_rows.loc[gnss_rows["true_z"] <= 2.0, "ml_p"].values
    c_frac = (c_p > ML_THRESH).mean() * 100 if len(c_p) else 0.0
    o_frac = (o_p > ML_THRESH).mean() * 100 if len(o_p) else 0.0
    ax2.text(0.02, 0.85,
             f"Canyon trigger: {c_frac:.0f}%\nOpen-sky trigger: {o_frac:.0f}%",
             transform=ax2.transAxes, fontsize=8,
             bbox=dict(fc="white", alpha=0.8, pad=3))
    _style_ax(ax2, "ML Gate: P(GNSS altitude untrustworthy)", "Time (s)", "p_k")
    ax2.legend(fontsize=8)

    # P3: Per-segment MAE bar chart
    ax3 = fig.add_subplot(gs[1, 1])
    names   = [e.name for e in ekfs]
    colors  = [PALETTE[k] for k in pal_keys]
    mae_all = [np.abs(df[e.name] - df["true_z"]).mean() for e in ekfs]
    mae_can = [np.abs(df.loc[canyon, e.name] - df.loc[canyon, "true_z"]).mean()
               for e in ekfs]
    mae_opn = [np.abs(df.loc[~canyon, e.name] - df.loc[~canyon, "true_z"]).mean()
               for e in ekfs]
    x, w = np.arange(4), 0.26
    b1 = ax3.bar(x - w, mae_all, width=w, color=colors, alpha=0.9,  label="Overall")
    b2 = ax3.bar(x,     mae_can, width=w, color=colors, alpha=0.55, hatch="///", label="Canyon")
    b3 = ax3.bar(x + w, mae_opn, width=w, color=colors, alpha=0.35, hatch="...", label="Open-sky")
    for bars in [b1, b2, b3]:
        for bar in bars:
            v = bar.get_height()
            if v > 0.01:
                ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                         f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, fontsize=9)
    ax3.set_ylim(0, max(mae_all) * 1.4)
    _style_ax(ax3, "MAE by segment: Overall / Canyon / Open-sky", "EKF", "MAE (m)")
    ax3.legend(fontsize=8)

    # P4: LiDAR pulses
    ax4 = fig.add_subplot(gs[2, 0])
    pulses = [e.lidar_pulses for e in ekfs]
    bars   = ax4.bar(names, pulses, color=colors, alpha=0.85)
    for bar, v in zip(bars, pulses):
        ax4.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 str(int(v)), ha="center", fontsize=9, fontweight="bold")
    if pulses[2] > 0:
        red = (1 - pulses[1] / pulses[2]) * 100
        ax4.annotate(
            f"EKF-B fires {red:.0f}% fewer\npulses than EKF-C",
            xy=(1, pulses[1]), xytext=(1.5, pulses[1] + 10),
            fontsize=8, color=PALETTE["B"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["B"]),
        )
    _style_ax(ax4, "LiDAR pulses fired (1 Hz GNSS-epoch rate)", "EKF", "Pulses")

    # P5: Sensor-only energy
    ax5 = fig.add_subplot(gs[2, 1])
    energy_J = [sens_J[e.name] for e in ekfs]
    bars = ax5.bar(names, energy_J, color=colors, alpha=0.85)
    for bar, v in zip(bars, energy_J):
        ax5.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 f"{v:.1f} J", ha="center", fontsize=9, fontweight="bold")
    ax5.text(0.05, 0.93,
             "Drone base power identical for all\n→ sensor-only delta shown",
             transform=ax5.transAxes, fontsize=8, color="#888",
             bbox=dict(fc="white", alpha=0.7, pad=2))
    _style_ax(ax5, "Sensor-only energy (GNSS + ML + LiDAR)", "EKF", "Energy (J)")

    # P6: GNSS error vs ml_p scatter
    ax6 = fig.add_subplot(gs[3, :])
    valid = ~np.isnan(df["gnss_err"].values)
    sc = ax6.scatter(df.loc[valid, "ml_p"],
                     df.loc[valid, "gnss_err"].clip(0, 30),
                     c=df.loc[valid, "true_z"], cmap="plasma",
                     s=20, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax6, label="True altitude (m)")
    ax6.axvline(ML_THRESH, color=PALETTE["C"], lw=1.5, linestyle="--",
                label=f"Gate threshold τ = {ML_THRESH}")
    ax6.axhline(3.0, color=PALETTE["D"], lw=1.2, linestyle=":",
                label="3 m LiDAR trigger level")
    ax6.set_xlim(-0.02, 1.02)
    ax6.set_ylim(-0.5, 30)
    _style_ax(ax6,
              "GNSS altitude error vs ML probability (colour = true height) — proves calibration",
              "p_k = P(trigger_lidar)", "|z_gnss − z_true| (m)")
    ax6.legend(fontsize=8)
    try:
        from scipy.stats import pearsonr
        r, _ = pearsonr(df.loc[valid, "ml_p"], df.loc[valid, "gnss_err"].clip(0, 50))
        ax6.text(0.97, 0.93, f"Pearson r = {r:.3f}", transform=ax6.transAxes,
                 ha="right", fontsize=9, color=PALETTE["B"],
                 bbox=dict(fc="white", alpha=0.8, pad=3))
    except ImportError:
        pass

    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Results plot: {out_path}")
    plt.close(fig)


def plot_3d(df: pd.DataFrame, bldg_polys: list, waypoints: list, out_path: str):
    """3-D trajectory with building footprints and top-down 2-D map."""
    fig = plt.figure(figsize=(20, 9), facecolor=PALETTE["bg"])
    x, y, zt = df["x"].values, df["y"].values, df["true_z"].values
    ml_p = df["ml_p"].values

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_facecolor(PALETTE["bg"])
    for poly in bldg_polys[:80]:
        px, py = poly[:, 0], poly[:, 1]
        ax1.plot(np.append(px, px[0]), np.append(py, py[0]),
                 np.zeros(len(px) + 1), color=PALETTE["bldg"], lw=0.7, alpha=0.5)
    ax1.plot(x, y, zt, "k-", lw=2.5, label="True path")
    ax1.plot(x, y, df["EKF-A"].values, color=PALETTE["A"], lw=1.3, alpha=0.7, label="EKF-A (χ²)")
    ax1.plot(x, y, df["EKF-B"].values, color=PALETTE["B"], lw=2.0, label="EKF-B (ML-gated)")
    for wp in waypoints:
        ax1.plot([wp[0], wp[0]], [wp[1], wp[1]], [0, wp[2]], ":", color="#aaa", lw=1, alpha=0.6)
    ax1.scatter(*waypoints[0],  s=120, color="green", marker="^", zorder=6)
    ax1.scatter(*waypoints[-1], s=120, color="red",   marker="v", zorder=6)
    ax1.set_xlabel("X Easting (m)", fontsize=9)
    ax1.set_ylabel("Y Northing (m)", fontsize=9)
    ax1.set_zlabel("Altitude (m)", fontsize=9)
    ax1.set_title("3-D Flight Path over Campus Geometry\n"
                  "(buildings from PLY mesh — floor projection)",
                  fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")

    ax2 = fig.add_subplot(122)
    ax2.set_facecolor("#F0EDE8")
    patches = [MplPolygon(poly, closed=True) for poly in bldg_polys]
    if patches:
        pc = PatchCollection(patches, facecolor=PALETTE["bldg"],
                             edgecolor="#8B6914", linewidth=0.7, alpha=0.85)
        ax2.add_collection(pc)
    ax2.plot(x, y, "-", color="#555", lw=1.5, alpha=0.7, label="Ground track")
    lidar_mask = ml_p > ML_THRESH
    if lidar_mask.any():
        ax2.scatter(x[lidar_mask], y[lidar_mask], s=30, color=PALETTE["C"],
                    zorder=5, label="LiDAR fired (EKF-B)")
    if (~lidar_mask).any():
        ax2.scatter(x[~lidar_mask][::30], y[~lidar_mask][::30], s=8,
                    color=PALETTE["B"], alpha=0.4, label="GNSS trusted")
    wps = np.array(waypoints)
    ax2.scatter(wps[:, 0], wps[:, 1], s=80, color="navy", zorder=6,
                marker="D", label="Waypoints")
    for i, wp in enumerate(waypoints):
        ax2.annotate(f"WP{i}\n{wp[2]:.0f} m", (wp[0], wp[1]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color="navy")
    ax2.scatter(x[0],  y[0],  s=150, color="green", marker="*", zorder=7, label="Takeoff")
    ax2.scatter(x[-1], y[-1], s=150, color="red",   marker="X", zorder=7, label="Land")
    ax2.set_xlim(X_MIN * 0.6, X_MAX * 0.6)
    ax2.set_ylim(Y_MIN * 0.6, Y_MAX * 0.6)
    ax2.set_aspect("equal")
    ax2.grid(True, color=PALETTE["grid"], lw=0.5)
    ax2.set_xlabel("X Easting (m)")
    ax2.set_ylabel("Y Northing (m)")
    ax2.set_title("Top-down: Buildings + Flight Path + ML Activations", fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"  Trajectory plot: {out_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(df: pd.DataFrame, ekfs: list, sens_J: dict):
    true_z = df["true_z"].values
    canyon = true_z > 3.5

    print(f"\n{'='*80}")
    print("UAV Altitude Simulation — Results Summary")
    print(f"{'='*80}")
    print(f"{'EKF':<8} | {'MAE-all':>8} | {'MAE-canyon':>10} | {'MAE-open':>9} | "
          f"{'LiDAR':>6} | {'Sens.E (J)':>10} | {'GNSS acc%':>9}")
    print("-" * 80)

    for e in ekfs:
        err   = np.abs(df[e.name] - df["true_z"])
        mae_a = err.mean()
        mae_c = err[canyon].mean() if canyon.any() else float("nan")
        mae_o = err[~canyon].mean() if (~canyon).any() else float("nan")
        gnss_a = e.gnss_accepted / max(e.gnss_total, 1) * 100
        print(f"{e.name:<8} | {mae_a:>7.3f} m | {mae_c:>9.3f} m | {mae_o:>8.3f} m | "
              f"{e.lidar_pulses:>6} | {sens_J[e.name]:>10.2f} | {gnss_a:>8.1f}%")

    print("=" * 80)

    mae_B   = np.abs(df["EKF-B"] - df["true_z"])[canyon].mean()
    mae_C   = np.abs(df["EKF-C"] - df["true_z"])[canyon].mean()
    pB, pC  = ekfs[1].lidar_pulses, ekfs[2].lidar_pulses
    if pC > 0:
        lidar_red  = (1 - pB / pC) * 100
        energy_red = (1 - sens_J["EKF-B"] / sens_J["EKF-C"]) * 100
        energy_sav = sens_J["EKF-C"] - sens_J["EKF-B"]
        print(f"\nKey result:")
        print(f"  Canyon MAE:   EKF-B = {mae_B:.3f} m  vs  EKF-C = {mae_C:.3f} m  "
              f"(delta = {abs(mae_B-mae_C)*1000:.0f} mm)")
        print(f"  LiDAR pulses: EKF-B = {pB}  vs  EKF-C = {pC}  ({lidar_red:.1f}% fewer)")
        print(f"  Energy saved: {energy_sav:.0f} J  ({energy_red:.1f}% of EKF-C sensor energy)")

    gnss_mask = ~np.isnan(df["z_gnss"].values)
    gnss_rows = df[gnss_mask]
    c_rows = gnss_rows[gnss_rows["true_z"] > 3.5]
    o_rows = gnss_rows[gnss_rows["true_z"] <= 2.0]
    if len(c_rows) and len(o_rows):
        c_trig = (c_rows["ml_p"] > ML_THRESH).mean() * 100
        o_trig = (o_rows["ml_p"] > ML_THRESH).mean() * 100
        print(f"\n  ML discrimination:  canyon = {c_trig:.0f}% trigger  "
              f"open-sky = {o_trig:.0f}% trigger")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  INTERACTIVE WAYPOINT PICKER
# ══════════════════════════════════════════════════════════════════════════════

def pick_waypoints(bldg_polys: list) -> list:
    """
    Show 2-D building map and let the user click waypoints.
    Z is auto-assigned: 1 m for first/last, alternating 7/10 m for middle.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.canvas.manager.set_window_title("Click waypoints — close window when done")
    ax.set_facecolor("#F0EDE8")

    patches = [MplPolygon(poly, closed=True) for poly in bldg_polys]
    if patches:
        pc = PatchCollection(patches, facecolor=PALETTE["bldg"],
                             edgecolor="#8B6914", lw=0.8, alpha=0.85)
        ax.add_collection(pc)

    ax.set_xlim(X_MIN * 0.7, X_MAX * 0.7)
    ax.set_ylim(Y_MIN * 0.7, Y_MAX * 0.7)
    ax.set_aspect("equal")
    ax.grid(True, color=PALETTE["grid"])
    ax.set_xlabel("X Easting (m)")
    ax.set_ylabel("Y Northing (m)")
    ax.set_title(
        "Click waypoints (start → end)\n"
        "Near buildings = canyon, open areas = clear-sky\n"
        "Close window when done.",
        fontweight="bold",
    )
    ax.scatter(0, 0, s=200, color="green", marker="*", zorder=5, label="Origin")
    ax.legend()

    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)

    if not pts or len(pts) < 2:
        print("  No waypoints selected — using defaults")
        return WAYPOINTS_IITM

    result = []
    for i, (px, py) in enumerate(pts):
        if i == 0 or i == len(pts) - 1:
            z = 1.0
        elif i % 2 == 1:
            z = 7.0
        else:
            z = 10.0
        result.append([float(px), float(py), z])
        print(f"  WP{i}: ({px:.1f}, {py:.1f}, {z:.1f} m)")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GRAIL UAV Altitude Simulation — 4-EKF comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--feat-csv",    default=DEFAULT_FEAT_CSV,
                    help="Path to GNSS feature CSV (gnss_ml_features.csv)")
    ap.add_argument("--model",       default=DEFAULT_MODEL,
                    help="Path to trained XGBoost classifier (.pkl)")
    ap.add_argument("--mesh-dir",    default=DEFAULT_MESH_DIR,
                    help="Directory containing PLY mesh files for the scene")
    ap.add_argument("--out-log",     default=DEFAULT_OUT_LOG,
                    help="Output flight log CSV path")
    ap.add_argument("--out-results", default=DEFAULT_OUT_RES,
                    help="Output results figure path")
    ap.add_argument("--out-traj",    default=DEFAULT_OUT_TRAJ,
                    help="Output 3-D trajectory figure path")
    ap.add_argument("--duration",    type=float, default=SIM_DURATION,
                    help="Simulation duration in seconds")
    ap.add_argument("--smoke",       action="store_true",
                    help="30-second quick test (overrides --duration)")
    ap.add_argument("--no-interact", action="store_true",
                    help="Skip interactive waypoint picker; use built-in IITM defaults")
    ap.add_argument("--no-live",     action="store_true",
                    help="Disable live altitude plot")
    ap.add_argument("--no-mesh",     action="store_true",
                    help="Skip PLY mesh loading (pure-physics fallback)")
    args = ap.parse_args()

    print("=" * 65)
    print("GRAIL UAV Altitude Simulation — Honest 4-EKF Comparison")
    print("=" * 65)

    # Load scene
    if args.no_mesh:
        scene, bldg_polys = None, []
    else:
        scene, bldg_polys = load_ply_scene(args.mesh_dir)

    # Load ML assets
    model, feat_df, kdtree = load_assets(args.feat_csv, args.model)

    # Waypoints
    if args.no_interact or not bldg_polys:
        waypoints = WAYPOINTS_IITM
        print(f"  Waypoints: {len(waypoints)} IITM defaults")
    else:
        waypoints = pick_waypoints(bldg_polys)

    duration = 30.0 if args.smoke else args.duration

    # Run simulation
    df, ekfs, sens_J = run_simulation(
        scene, model, feat_df, kdtree, waypoints,
        duration=duration,
        live_plot=(not args.no_live and not args.smoke),
    )

    # Save outputs
    df.to_csv(args.out_log, index=False)
    print(f"  Flight log: {args.out_log}")

    plot_results(df, ekfs, sens_J, waypoints, args.out_results)
    plot_3d(df, bldg_polys, waypoints, args.out_traj)
    print_summary(df, ekfs, sens_J)


if __name__ == "__main__":
    main()
