# GNSS-VIM-Sim

**GNSS Vertical Integrity Monitor Simulator** — physics-accurate UAV altitude fusion with ML-gated sensor control.

Part of the [GRAIL](../README.md) (GNSS Ray-tracing for Altitude Inference and Localization) project.

---

## What it does

GNSS-VIM-Sim runs a 4-EKF UAV altitude simulation over an arbitrary 3-D mesh scene.
It compares four sensor fusion strategies:

| EKF | Strategy | LiDAR usage |
|-----|----------|------------|
| A | PX4 chi-squared gate + LiDAR fallback | Reactive (on GNSS rejection) |
| **B** | **GRAIL ML adaptive-R** ← _ours_ | **Proactive (ML triggers)** |
| C | Always-on LiDAR | 100% (energy upper bound) |
| D | Blind GNSS only | 0% (accuracy lower bound) |

The bundled GRAIL XGBoost classifier (AUC ≈ 0.999 on synthetic IITM campus data)
predicts P(GNSS altitude error > 3 m) from the 22-feature fingerprint vector and drives
EKF-B's adaptive measurement noise:

```
R_GNSS,k = R_base × exp(α × p_k)
```

When p_k → 1, Kalman gain → 0 (GNSS silently down-weighted).
When p_k > τ = 0.40, LiDAR is triggered proactively.

**IITM 300-second result:** EKF-B matches always-LiDAR within 12 mm canyon MAE
while firing LiDAR **41.9 % less often**, saving **857 J per flight**.

---

## Installation

```bash
cd gnss_vim_sim
pip install -e ".[full]"          # includes plotly, trimesh, scipy
# or minimal
pip install -e .
```

For the `new-scene` OSM workflow:
```bash
pip install osmnx shapely trimesh
```

---

## Quick start (30 seconds)

```bash
# 1. Create a demo project
gnss-vim-sim init --out my_project
cd my_project

# 2. Inspect the scene
gnss-vim-sim inspect-scene --config demo_config.json

# 3. Open the WebGL mission planner in your browser
gnss-vim-sim studio --config demo_config.json --out runs/studio.html

# 4. Run the 4-EKF simulation (bundled GRAIL classifier loads automatically)
gnss-vim-sim run --config demo_config.json --out runs/demo_run --dashboard
```

---

## Bring Your Own OSM Scene

```bash
# Downloads OpenStreetMap buildings, builds PLY meshes, writes config.json
gnss-vim-sim new-scene \
    --lat  12.9906 \
    --lon  80.2296 \
    --name iitm_campus \
    --radius 800

cd iitm_campus
gnss-vim-sim run --config demo_config.json --out runs/iitm_run
```

Works for **any urban area** on the planet covered by OpenStreetMap.

```
Architecture of new-scene
─────────────────────────────────────────────────────────────────
 Overpass API   ──→  osmnx.features_from_point()
                           │
                    building footprints (GeoJSON polygons)
                           │
               shapely.Polygon + trimesh.extrude_polygon()
                           │
                    demo_mesh/buildings.ply  (binary PLY)
                    demo_mesh/ground.ply     (flat ground box)
                           │
               write demo_config.json  (pre-populated, run-ready)
                           │
         gnss-vim-sim run --config demo_config.json  ✓
─────────────────────────────────────────────────────────────────
```

If the network is unavailable, use `--offline` and place your own PLY files
in `demo_mesh/` manually (e.g. exported from BlenderGIS + Blender).

---

## Manual BlenderGIS / OSM workflow

1. Install [BlenderGIS](https://github.com/domlysz/BlenderGIS) in Blender 3.x+
2. In Blender: `File → Import → OpenStreetMap (.osm)` or use the BlenderGIS panel
3. Select buildings layer → set extrusion from `building:levels` tag
4. `File → Export → Stanford PLY (.ply)` → save to `<project>/demo_mesh/buildings.ply`
5. Repeat for terrain / ground mesh → `demo_mesh/ground.ply`
6. Generate config: `gnss-vim-sim new-scene --lat X --lon Y --name Z --offline`

---

## Architecture

```
gnss_vim_sim/
├── src/gnss_vim_sim/
│   ├── cli.py              # entry point: gnss-vim-sim <command>
│   │
│   ├── core/
│   │   ├── config.py       # SimConfig dataclass (loads JSON)
│   │   ├── coordinates.py  # ENU ↔ ECEF conversion
│   │   ├── state.py        # VehicleState (pos, vel, att, EKF)
│   │   └── validation.py   # config checker with warnings/errors
│   │
│   ├── sensors/
│   │   ├── gnss.py         # GNSS sensor model + 22-feature extractor
│   │   ├── baro.py         # MS5611 barometer model
│   │   ├── imu.py          # MPU-6000 IMU noise model
│   │   └── rangefinder.py  # LiDAR ToF model (VL53L1X / Garmin)
│   │
│   ├── estimators/
│   │   └── vertical_ekf.py # 3-state altitude EKF (all 4 variants)
│   │                        # State: x = [z, v_z, baro_bias]^T
│   │
│   ├── ml/
│   │   ├── runtime.py      # RuntimeModel ABC + PickleRuntimeModel adapter
│   │   └── integrity.py    # GNSS integrity score wrapper
│   │
│   ├── world/
│   │   ├── scene.py        # MeshScene: PLY loader + ray-cast (LiDAR)
│   │   └── osm_builder.py  # OSM → PLY pipeline (new-scene command)
│   │
│   ├── planning/
│   │   ├── mission.py      # Mission: waypoint sequencer
│   │   ├── router.py       # A* path planner (obstacle avoidance)
│   │   └── interactive.py  # Matplotlib waypoint picker
│   │
│   ├── sim/
│   │   ├── runner.py       # SimulationRunner: main loop (4-EKF)
│   │   └── metrics.py      # MAE, RMSE, energy, LiDAR pulse counting
│   │
│   ├── viz/
│   │   ├── plots.py        # Matplotlib: altitude, MAE, energy panels
│   │   ├── dashboard.py    # Plotly: interactive 3-D dashboard
│   │   ├── player.py       # Lightweight HTML5 30fps flight player
│   │   └── planner_html.py # WebGL mission planning page
│   │
│   ├── io/
│   │   ├── init_project.py # gnss-vim-sim init  (demo scaffolding)
│   │   └── logging.py      # flight_log.csv writer
│   │
│   └── assets/
│       ├── demo_config.json        # runnable demo config (4 synthetic buildings)
│       ├── demo_mesh/demo_scene.ply # 4 synthetic building boxes
│       └── grail_classifier.pkl    # pretrained XGBoost (AUC ≈ 0.999)
│
├── configs/
│   ├── iitm_demo.json      # IITM campus config (requires large CSV / PLY)
│   └── mission_config.json # example mission-only config
│
├── docs/
│   ├── architecture.md     # detailed module description
│   ├── model_adapter.md    # how to plug in a custom ML model
│   └── runbook.md          # step-by-step BYO-scene guide
│
└── pyproject.toml
```

---

## CLI reference

```
gnss-vim-sim <command> [options]

Commands:
  new-scene       Download OSM area → PLY meshes → demo_config.json
  init            Scaffold a demo project with 4 synthetic buildings
  validate        Check a config.json for errors and warnings
  inspect-scene   Report mesh bounds, vertex count, building footprints
  studio          Open WebGL mission planner in browser
  plan-mission    Click waypoints on Matplotlib map, save mission JSON
  preview-mission Render waypoint path preview (PNG or window)
  run             Run 4-EKF simulation, save flight_log.csv + plots
  replay          Live-replay a flight_log.csv in Matplotlib
  dashboard       Build interactive Plotly 3-D HTML dashboard
  player          Build lightweight 30fps browser flight player
  wizard          Guided browser → waypoint plan → simulation run
```

### `new-scene`

```
gnss-vim-sim new-scene \
    --lat   <latitude>    # WGS-84 decimal degrees (required)
    --lon   <longitude>   # WGS-84 decimal degrees (required)
    --name  <project>     # project name, also used as output dir name
    --out   <path>        # output directory (default: ./<name>/)
    --radius <metres>     # scene half-width (default: 800 m)
    --offline             # skip OSM download; scaffold config only
```

### `run`

```
gnss-vim-sim run \
    --config          <json>    # simulation config (required)
    --model-checkpoint <pkl>    # custom ML model; default: bundled GRAIL
    --out             <dir>     # output directory (default: runs/run_<stamp>)
    --dashboard                 # also build Plotly dashboard.html
    --require-mesh              # fail if demo_mesh/ is empty
```

---

## Custom ML model

Any sklearn/XGBoost/LightGBM model that exposes `predict_proba(X)` works:

```python
# Model must accept shape (N, 22) array with columns in FEATURE_COLS order.
# predict_proba(X)[:, 1] should return P(GNSS altitude error > 3 m).

from gnss_vim_sim.ml.runtime import FEATURE_COLS  # 22 column names
import pickle, numpy as np

# Train on your own Sionna-generated data
X = np.random.rand(1000, 22)
y = np.random.randint(0, 2, 1000)
from xgboost import XGBClassifier
clf = XGBClassifier().fit(X, y)
pickle.dump(clf, open("my_model.pkl", "wb"))

# Use in simulation
# gnss-vim-sim run --config cfg.json --model-checkpoint my_model.pkl
```

Model priority:
1. `--model-checkpoint` (user-supplied)
2. Bundled `grail_classifier.pkl` (loaded automatically, AUC ≈ 0.999)
3. VDOP/C/N₀ heuristic fallback (no file I/O)

---

## Key results (IITM campus, 300-second mission)

| EKF | MAE all | MAE canyon | MAE open | LiDAR pulses | Sensor energy |
|-----|---------|------------|----------|--------------|---------------|
| A — chi²+LiDAR fallback | 0.185 m | 0.222 m | 0.090 m | 3 / 301 | ~39 J |
| **B — ML-gated** ← _ours_ | **0.098 m** | **0.097 m** | 0.101 m | **175 / 301** | **~1566 J** |
| C — always-LiDAR | 0.075 m | 0.085 m | 0.046 m | 301 / 301 | ~2423 J |
| D — blind GNSS | 0.323 m | 0.317 m | 0.337 m | 0 / 301 | ~15 J |

**EKF-B saves 857 J (35.4 %) vs EKF-C with only 12 mm canyon accuracy penalty.**

---

## Sim-to-Real Transfer

The GRAIL classifier was trained entirely on Sionna-simulated data from the IITM campus.
Applied zero-shot to 660 real Android GNSS epochs:

| Transfer mode | AUC |
|---------------|-----|
| Zero-shot (no real data) | 0.779 |
| Fine-tuned (70% real epochs) | 0.938 |

VDOP and elevation geometry transfer accurately because they depend on
satellite orbital geometry (SP3), not signal amplitude. C/N₀ shows a
−9.3 dB gap (no tree canopy in Sionna scene).

---

## Citation

```bibtex
@misc{grail2026,
  title  = {GRAIL: GNSS Ray-tracing for Altitude Inference and Localization},
  author = {Tharaneeshwaran Rajasekar},
  year   = {2026},
  note   = {BTP Final Report, Dept. of Electrical Engineering, IIT Madras}
}
```

---

## License

MIT © 2026 Tharaneeshwaran Rajasekar, IIT Madras
