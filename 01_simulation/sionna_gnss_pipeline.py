"""
sionna_gnss_pipeline.py
=======================
GRAIL Phase-1: Physics-accurate GNSS fingerprint dataset generator.

Uses NVIDIA Sionna 2.0 RT to ray-trace GPS L1 signals through a 3-D campus
scene and extract Channel-Impulse-Response (CIR) features at thousands of
receiver locations simultaneously (batched on GPU VRAM).

Pipeline overview
-----------------
  1. Parse SP3 IGS precise-orbit files → ENU satellite geometry per epoch.
  2. Load Sionna scene (BlenderGIS-exported XML + PLY meshes).
  3. For each SP3 epoch × receiver batch:
       a. Place satellites as Sionna Transmitters on a sky dome.
       b. Run PathSolver (LOS + reflections + diffraction).
       c. Extract per-satellite: LOS ratio, multipath error, delay spread, C/N0.
       d. Run WLS solver → estimated z, VDOP, z-error.
       e. Aggregate 22-feature fingerprint per receiver epoch.
  4. Write per-link raw CSV + per-epoch aggregate CSV + analytics plot.

Inputs
------
  --scene        Path to Sionna scene XML file (default: iitm_snippet.xml)
  --sampling-csv Path to smart_sampler output CSV (default: sampling_points.csv)
  --sp3-dir      Directory containing *.SP3 / *.sp3 orbit files (default: sp3_data/)
  --out-agg      Output aggregate CSV (default: gnss_synthetic_agg.csv)
  --out-raw      Output raw per-link CSV (default: gnss_synthetic_raw.csv)
  --batch-size   Max receivers per GPU batch (tune to fit VRAM, default: 5000)
  --max-epochs   Limit number of SP3 epochs processed (default: all)

Outputs
-------
  gnss_synthetic_raw.csv   — one row per satellite-receiver link
  gnss_synthetic_agg.csv   — one row per receiver epoch (22-feature fingerprint)
  gnss_synthetic_analytics.png — floor-level error + C/N0 distribution plots

Requirements
------------
  sionna>=2.0, tensorflow>=2.12, numpy, pandas, matplotlib
  GPU with >=8 GB VRAM recommended (RTX 3080+)
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants & GPS L1 link budget
# ---------------------------------------------------------------------------

C               = 299_792_458.0         # speed of light (m/s)
F_L1            = 1_575_420_000.0       # GPS L1 carrier frequency (Hz)
SKY_DOME_RADIUS = 20_000.0             # virtual sky-dome radius (m)
ELEV_MASK_DEG   = 10.0                 # elevation mask angle (degrees)

GPS_TX_EIRP_DBM      = 47.0
GPS_ORBIT_KM         = 20_200.0
NOISE_DENSITY_DBM_HZ = -174.0
FSPL_DB = 20 * math.log10(4 * math.pi * GPS_ORBIT_KM * 1000 * F_L1 / C)
CN0_FREE_SPACE_DBHZ  = GPS_TX_EIRP_DBM - FSPL_DB - NOISE_DENSITY_DBM_HZ

# ---------------------------------------------------------------------------
# WGS-84 geodetic → ECEF → ENU transforms
# ---------------------------------------------------------------------------

_WGS84_A  = 6_378_137.0
_WGS84_E2 = 0.006_694_379_990_14


def _geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> tuple:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = _WGS84_A / math.sqrt(1.0 - _WGS84_E2 * math.sin(lat) ** 2)
    return (
        (N + alt_m) * math.cos(lat) * math.cos(lon),
        (N + alt_m) * math.cos(lat) * math.sin(lon),
        (N * (1.0 - _WGS84_E2) + alt_m) * math.sin(lat),
    )


def make_enu_converter(lat_deg: float, lon_deg: float, alt_m: float):
    """Return a function that converts ECEF (x,y,z) → local ENU (e,n,u)."""
    org = _geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    slat, clat = math.sin(lat), math.cos(lat)
    slon, clon = math.sin(lon), math.cos(lon)

    def ecef_to_enu(x: float, y: float, z: float) -> tuple:
        dx, dy, dz = x - org[0], y - org[1], z - org[2]
        e = -slon * dx + clon * dy
        n = -slat * clon * dx - slat * slon * dy + clat * dz
        u = clat * clon * dx + clat * slon * dy + slat * dz
        return e, n, u

    return ecef_to_enu


# ---------------------------------------------------------------------------
# SP3 orbit file parser
# ---------------------------------------------------------------------------

def parse_sp3_gps(sp3_dir: str, ecef_to_enu) -> list[dict]:
    """Parse all SP3 files in *sp3_dir* and return a list of epoch dicts.

    Each epoch dict has keys:
        epoch_str    : ISO-8601 UTC timestamp string
        visible_sats : list of satellite dicts with Sionna placement info
    """
    files = sorted(glob.glob(os.path.join(sp3_dir, "*.SP3"))) + \
            sorted(glob.glob(os.path.join(sp3_dir, "*.sp3")))
    if not files:
        raise FileNotFoundError(f"No SP3 files found in {sp3_dir!r}")

    all_epochs: list[dict] = []
    for fpath in files:
        epoch_sats: list[dict] = []
        cur_dt: datetime | None = None
        with open(fpath) as fh:
            for line in fh:
                if line.startswith("*"):
                    if epoch_sats and cur_dt is not None:
                        all_epochs.append({
                            "epoch_str": cur_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                            "visible_sats": list(epoch_sats),
                        })
                    epoch_sats = []
                    try:
                        p = line.split()
                        cur_dt = datetime(int(p[1]), int(p[2]), int(p[3]),
                                          int(p[4]), int(p[5]), 0, tzinfo=timezone.utc)
                    except Exception:
                        cur_dt = None
                    continue
                if line.startswith("EOF"):
                    break
                if not line.startswith("PG"):
                    continue
                try:
                    prn = int(line[2:4])
                    sx, sy, sz = float(line[4:18]) * 1e3, float(line[18:32]) * 1e3, float(line[32:46]) * 1e3
                except Exception:
                    continue

                e, n, u = ecef_to_enu(sx, sy, sz)
                dist = math.sqrt(e * e + n * n + u * u)
                if dist < 1.0:
                    continue
                elev_rad = math.asin(max(-1.0, min(1.0, u / dist)))
                elev_deg = math.degrees(elev_rad)
                if elev_deg < ELEV_MASK_DEG:
                    continue
                az_rad = math.atan2(e, n)
                ce = math.cos(elev_rad)
                epoch_sats.append({
                    "prn": prn,
                    "enu_pos": np.array([e, n, u]),
                    "sionna_pos": [
                        SKY_DOME_RADIUS * ce * math.sin(az_rad),
                        SKY_DOME_RADIUS * ce * math.cos(az_rad),
                        SKY_DOME_RADIUS * math.sin(elev_rad),
                    ],
                    "true_dist_m": dist,
                    "elevation": elev_deg,
                    "azimuth": math.degrees(az_rad) % 360.0,
                })
        if epoch_sats and cur_dt is not None:
            all_epochs.append({
                "epoch_str": cur_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "visible_sats": list(epoch_sats),
            })

    print(f"[SP3] Loaded {len(all_epochs)} epochs from {len(files)} files.")
    return all_epochs


# ---------------------------------------------------------------------------
# WLS position solver
# ---------------------------------------------------------------------------

def solve_wls(rx_init: np.ndarray, sat_enu_list: list, pseudoranges: list,
              elevations: list, max_iter: int = 15, tol: float = 1e-3):
    """Weighted Least Squares GNSS position solver.

    Returns (pos_enu, clock_bias_m, vdop, hdop, gdop) or (None, ...) on failure.
    """
    n = len(pseudoranges)
    if n < 4:
        return None, None, None, None, None
    els = np.clip(np.array(elevations, dtype=float), 1.0, 90.0)
    W = np.diag(np.sin(np.radians(els)) ** 2)
    pos = np.array(rx_init, dtype=np.float64).copy()
    cdt = 0.0

    for _ in range(max_iter):
        rows, res = [], []
        for j in range(n):
            vec = np.asarray(sat_enu_list[j]) - pos
            rng = float(np.linalg.norm(vec))
            rows.append([-vec[0] / rng, -vec[1] / rng, -vec[2] / rng, 1.0])
            res.append(pseudoranges[j] - (rng + cdt))
        H = np.array(rows)
        HtWH = H.T @ W @ H
        if not np.isfinite(np.linalg.cond(HtWH)) or np.linalg.cond(HtWH) > 1e10:
            return None, None, None, None, None
        try:
            dx = np.linalg.solve(HtWH, H.T @ W @ np.array(res))
        except np.linalg.LinAlgError:
            return None, None, None, None, None
        pos += dx[:3]
        cdt += float(dx[3])
        if np.linalg.norm(dx[:3]) < tol:
            break

    try:
        Q = np.linalg.pinv(H.T @ H)
        vdop = float(np.sqrt(max(float(Q[2, 2]), 0.0)))
        hdop = float(np.sqrt(max(float(Q[0, 0]) + float(Q[1, 1]), 0.0)))
        gdop = float(np.sqrt(max(float(np.trace(Q[:3, :3])), 0.0)))
        return pos, cdt, vdop, hdop, gdop
    except Exception:
        return pos, cdt, float("nan"), float("nan"), float("nan")


# ---------------------------------------------------------------------------
# CIR feature extraction
# ---------------------------------------------------------------------------

def extract_cir_features(a_flat: np.ndarray, tau_flat: np.ndarray,
                          true_dist_m: float) -> dict:
    """Compute per-satellite fingerprint features from the Sionna CIR."""
    power = np.abs(a_flat) ** 2
    valid = power > 0
    if not valid.any():
        return {}
    power, tau = power[valid], tau_flat[valid]
    order = np.argsort(tau)
    power, tau = power[order], tau[order]

    total = float(power.sum())
    mean_delay = float(np.average(tau, weights=power))
    delay_spread = float(np.sqrt(np.average((tau - mean_delay) ** 2, weights=power)))
    mp_error_m = float(C * (mean_delay - float(tau[0])))
    return {
        "pseudorange_m":    true_dist_m + mp_error_m,
        "los_ratio":        float(power[0]) / total if total > 0 else 0.0,
        "mp_error_m":       mp_error_m,
        "delay_spread_ns":  delay_spread * 1e9,
        "path_gain_db":     float(10.0 * np.log10(total + 1e-300)),
    }


def estimate_cn0(path_gain_db: float, ref_gain_db: float) -> float:
    return float(np.clip(CN0_FREE_SPACE_DBHZ + (path_gain_db - ref_gain_db), 5.0, 60.0))


# ---------------------------------------------------------------------------
# Sionna scene management
# ---------------------------------------------------------------------------

def build_scene(scene_file: str):
    from sionna.rt import load_scene, PlanarArray, Transmitter, PathSolver
    scene = load_scene(scene_file)
    try:
        scene.frequency = F_L1
    except Exception:
        scene.frequency.assign(F_L1)
    ant = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                      horizontal_spacing=0.5, pattern="dipole", polarization="V")
    scene.tx_array = scene.rx_array = ant
    scene.add(Transmitter(name="sat", position=[0.0, 0.0, float(SKY_DOME_RADIUS)]))
    return scene, PathSolver()


def calibrate_reference_gain(scene, p_solver, orbit_data: list[dict]) -> float:
    import tensorflow as tf
    from sionna.rt import Receiver

    ref_sat = next(
        (sat for epoch in orbit_data[:20] for sat in epoch["visible_sats"] if sat["elevation"] > 45),
        None,
    )
    if not ref_sat:
        return -150.0
    for name in list(scene.receivers.keys()):
        scene.remove(name)
    scene.add(Receiver(name="rx_cal", position=[0.0, 0.0, 1.0]))
    try:
        scene.get("sat").position.assign(tf.constant(ref_sat["sionna_pos"], dtype=tf.float32))
    except Exception:
        scene.get("sat").position = tf.constant(ref_sat["sionna_pos"], dtype=tf.float32)
    try:
        a, _ = p_solver(scene=scene, max_depth=5, los=True, specular_reflection=True,
                        diffraction=True, synthetic_array=True).cir(out_type="numpy")
        ps = (np.abs(a.flatten()) ** 2).sum()
        if ps > 0:
            return float(10.0 * np.log10(ps))
    except Exception:
        pass
    return -150.0


# ---------------------------------------------------------------------------
# Per-epoch batch processor
# ---------------------------------------------------------------------------

def process_epoch(scene, p_solver, rx_positions: np.ndarray, rx_meta: pd.DataFrame,
                  epoch: dict, ref_gain_db: float,
                  batch_size: int = 5000) -> tuple[list, list]:
    import tensorflow as tf
    from sionna.rt import Receiver

    n_rx = len(rx_positions)
    rx_meas: list[list] = [[] for _ in range(n_rx)]

    for chunk_start in range(0, n_rx, batch_size):
        chunk_end = min(chunk_start + batch_size, n_rx)
        chunk_pos = rx_positions[chunk_start:chunk_end]

        for name in list(scene.receivers.keys()):
            scene.remove(name)
        for i, pos in enumerate(chunk_pos):
            scene.add(Receiver(name=f"rx{i}", position=pos.tolist()))

        for sat in epoch["visible_sats"]:
            try:
                scene.get("sat").position.assign(
                    tf.constant(sat["sionna_pos"], dtype=tf.float32)
                )
            except Exception:
                scene.get("sat").position = tf.constant(sat["sionna_pos"], dtype=tf.float32)
            try:
                a, tau = p_solver(
                    scene=scene, max_depth=5, los=True,
                    specular_reflection=True, diffraction=True, synthetic_array=True,
                ).cir(out_type="numpy")
            except Exception:
                continue

            for i in range(chunk_end - chunk_start):
                global_idx = chunk_start + i
                try:
                    a_f = np.asarray(a)[i].flatten()
                    tau_f = np.asarray(tau)[i].flatten()
                except Exception:
                    continue
                feats = extract_cir_features(a_f, tau_f, sat["true_dist_m"])
                if feats:
                    feats.update({
                        "elevation": sat["elevation"],
                        "azimuth":   sat["azimuth"],
                        "prn":       sat["prn"],
                        "sat_enu":   sat["enu_pos"],
                        "cn0_dbhz":  estimate_cn0(feats["path_gain_db"], ref_gain_db),
                    })
                    rx_meas[global_idx].append(feats)

    agg_records: list[dict] = []
    raw_records: list[dict] = []

    for i in range(n_rx):
        meas = rx_meas[i]
        if len(meas) < 4:
            continue
        true_z = float(rx_positions[i][2])
        meta   = rx_meta.iloc[i]

        for m in meas:
            raw_records.append({
                "rx_id":               int(i),
                "epoch_str":           epoch["epoch_str"],
                "Svid":                m["prn"],
                "Cn0DbHz":             round(m["cn0_dbhz"], 2),
                "SvElevationDegrees":  round(m["elevation"], 2),
                "SvAzimuthDegrees":    round(m["azimuth"], 2),
                "PseudorangeMeters":   round(m["pseudorange_m"], 3),
                "MultipathErrorMeters":round(m["mp_error_m"], 3),
                "LosRatio":            round(m["los_ratio"], 3),
                "DelaySpreadNs":       round(m["delay_spread_ns"], 2),
                "true_x":              round(float(rx_positions[i][0]), 3),
                "true_y":              round(float(rx_positions[i][1]), 3),
                "true_z":              true_z,
                "floor":               int(meta.get("floor", round(true_z / 3))),
                "point_type":          str(meta.get("point_type", "unknown")),
            })

        est, _, vdop, _, _ = solve_wls(
            rx_positions[i],
            [m["sat_enu"] for m in meas],
            [m["pseudorange_m"] for m in meas],
            [m["elevation"] for m in meas],
        )
        if est is None:
            continue
        agg_records.append({
            "rx_id":          int(i),
            "true_z":         true_z,
            "estimated_z_m":  round(float(est[2]), 4),
            "z_error_m":      round(abs(float(est[2]) - true_z), 4),
            "floor":          int(meta.get("floor", round(true_z / 3))),
            "point_type":     str(meta.get("point_type", "unknown")),
            "epoch_str":      epoch["epoch_str"],
            "vdop":           round(float(vdop), 4) if vdop else float("nan"),
            "mean_cn0_dbhz":  round(float(np.mean([m["cn0_dbhz"] for m in meas])), 3),
            "n_sats":         len(meas),
        })

    return agg_records, raw_records


# ---------------------------------------------------------------------------
# Analytics plot
# ---------------------------------------------------------------------------

def plot_analytics(agg_csv: str, out_png: str) -> None:
    if not os.path.exists(agg_csv):
        return
    try:
        import matplotlib.pyplot as plt
        df = pd.read_csv(agg_csv)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        floors = sorted(df["floor"].unique())
        data = [df[df["floor"] == f]["z_error_m"].dropna().values for f in floors]
        axes[0].boxplot(data, labels=[f"F{f}" for f in floors], patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", alpha=0.7))
        axes[0].set_title("Altitude WLS Error by Floor")
        axes[0].set_ylabel("Absolute z-error (m)")
        axes[0].grid(True, alpha=0.3)

        for pt in df["point_type"].unique():
            axes[1].hist(df[df["point_type"] == pt]["mean_cn0_dbhz"],
                         bins=30, alpha=0.6, label=pt, density=True)
        axes[1].set_title("Mean C/N₀ by Point Type")
        axes[1].set_xlabel("Mean C/N₀ (dB-Hz)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].scatter(df["vdop"].clip(0, 10), df["z_error_m"].clip(0, 30),
                        s=2, alpha=0.2, c="#E45454")
        axes[2].set_title("VDOP vs Altitude Error")
        axes[2].set_xlabel("VDOP")
        axes[2].set_ylabel("Absolute z-error (m)")
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("GRAIL Synthetic Dataset Analytics", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"[pipeline] Analytics plot → {out_png}")
    except Exception as exc:
        print(f"[pipeline] Could not generate analytics plot: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="GRAIL Sionna GNSS fingerprint dataset generator."
    )
    ap.add_argument("--scene",        default="iitm_snippet.xml",    help="Sionna scene XML file.")
    ap.add_argument("--sampling-csv", default="sampling_points.csv", help="Smart-sampler output CSV.")
    ap.add_argument("--sp3-dir",      default="sp3_data",            help="Directory with *.SP3 orbit files.")
    ap.add_argument("--out-agg",      default="gnss_synthetic_agg.csv",  help="Output aggregate CSV.")
    ap.add_argument("--out-raw",      default="gnss_synthetic_raw.csv",  help="Output raw per-link CSV.")
    ap.add_argument("--batch-size",   type=int, default=5000,        help="GPU batch size (receivers per chunk).")
    ap.add_argument("--max-epochs",   type=int, default=0,           help="Limit SP3 epochs processed (0 = all).")
    ap.add_argument("--lat",          type=float, default=12.990628, help="Scene origin latitude.")
    ap.add_argument("--lon",          type=float, default=80.229689, help="Scene origin longitude.")
    ap.add_argument("--alt",          type=float, default=12.5,      help="Scene origin altitude (m).")
    args = ap.parse_args()

    ecef_to_enu = make_enu_converter(args.lat, args.lon, args.alt)

    pts_df = pd.read_csv(args.sampling_csv)
    rx_positions = pts_df[["x", "y", "z"]].values.astype(np.float64)
    print(f"[pipeline] {len(rx_positions):,} receiver positions loaded from {args.sampling_csv!r}")

    orbit_data = parse_sp3_gps(args.sp3_dir, ecef_to_enu)
    if args.max_epochs > 0:
        orbit_data = orbit_data[: args.max_epochs]

    scene, p_solver = build_scene(args.scene)
    ref_gain_db = calibrate_reference_gain(scene, p_solver, orbit_data)
    print(f"[pipeline] Reference gain calibrated: {ref_gain_db:.2f} dB")

    wrote_header = not os.path.exists(args.out_agg)
    total_links = 0

    for ep_idx, epoch in enumerate(orbit_data):
        if len(epoch["visible_sats"]) < 4:
            continue
        t0 = time.time()
        agg_recs, raw_recs = process_epoch(
            scene, p_solver, rx_positions, pts_df, epoch, ref_gain_db, args.batch_size
        )
        if agg_recs:
            pd.DataFrame(agg_recs).to_csv(args.out_agg, mode="a", header=wrote_header, index=False)
            pd.DataFrame(raw_recs).to_csv(args.out_raw, mode="a", header=wrote_header, index=False)
            wrote_header = False
            total_links += len(raw_recs)
        print(
            f"[pipeline] Epoch {ep_idx + 1}/{len(orbit_data)} | "
            f"{epoch['epoch_str']} | "
            f"{len(epoch['visible_sats'])} sats | "
            f"{len(raw_recs):,} links | "
            f"{time.time() - t0:.1f}s"
        )

    print(f"\n[pipeline] Complete. {total_links:,} total satellite-link observations.")
    out_png = args.out_agg.replace(".csv", "_analytics.png")
    plot_analytics(args.out_agg, out_png)


if __name__ == "__main__":
    main()
