# GRAIL

**GNSS Ray-tracing for Altitude Inference and Localization**

B.Tech Final Year Project — Department of Computer Science and Engineering, IIT Madras, 2025–26.
(Tharaneeshwaran V U : CS25E053)

---

## What is GRAIL?

Urban UAVs suffer a fundamental vertical blindness. GNSS altitude errors are
2–5× larger than horizontal errors due to geometric VDOP amplification in
city canyons. LiDAR altimeters solve this but consume 8–30 W continuously.

GRAIL answers: **can we predict when GNSS altitude is untrustworthy from the
GNSS observables themselves, and trigger LiDAR only when needed?**

**Yes.** The key insight is that VDOP, a purely geometric property of the visible
satellite constellation, is the dominant predictor of altitude error, and VDOP
correlates with delay spread and elevation spread in the raw fingerprint.
A NVIDIA's Sionna-trained XGBoost classifier learns this relationship and drives a
3-state EKF's adaptive measurement noise, saving 857 J per flight at 12 mm
accuracy cost versus always-on LiDAR.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GRAIL Pipeline                              │
│                                                                     │
│  OSM Scene (BlenderGIS)                                             │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────┐    SP3 IGS Orbits (7 days, GPS)              │
│  │  Smart Sampler   │────────────────────────────────┐             │
│  │  v6 (7 floors)   │                                │             │
│  └────────┬─────────┘                                │             │
│           │ 338 M satellite-link observations         │             │
│           ▼                                           ▼             │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │           Sionna 2.0 Ray-Tracer (GNSS channel sim)       │      │
│  │  CIR taps → LOS ratio, delay spread, C/N₀, MP error      │      │
│  └─────────────────────────┬────────────────────────────────┘      │
│                            │ 16.7 M aggregated receiver epochs      │
│                            ▼                                        │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │     Feature Engineering  (22-dim fingerprint vector)    │       │
│  │  n_sats, C/N₀ stats, elevation stats, VDOP,             │       │
│  │  LOS ratio stats, delay stats, multipath error stats     │       │
│  └─────────────────────┬───────────────────────────────────┘       │
│                        │                                            │
│           ┌────────────┴─────────────┐                             │
│           │                          │                             │
│           ▼                          ▼                             │
│  ┌─────────────────┐      ┌─────────────────────────┐             │
│  │ XGBoost Classif │      │  XGBoost Regressor       │             │
│  │ GroupKFold CV   │      │  (z-error prediction)    │             │
│  │ AUC ≈ 0.999     │      │  R² ≈ 0.72               │             │
│  └────────┬────────┘      └─────────────────────────┘             │
│           │ p_k = P(|z_err| > 3 m)                                 │
│           ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              3-State Altitude EKF (x = [z, v_z, b_baro])│       │
│  │                                                          │       │
│  │   R_GNSS,k = R_base · exp(α · p_k)   [EKF-B adaptive-R]│       │
│  │   LiDAR fires when p_k > τ = 0.40    [proactive gate]   │       │
│  └─────────────────────────────────────────────────────────┘       │
│           │                                                         │
│           ▼                                                         │
│  ┌────────────────────────────────┐                                │
│  │   Sim-to-Real Transfer         │                                │
│  │   660 real Android GNSS epochs │                                │
│  │   Zero-shot AUC = 0.779        │                                │
│  │   Fine-tuned AUC = 0.938       │                                │
│  └────────────────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Results at a Glance

### Synthetic Dataset (Sionna + IITM campus OSM)

| Metric | Value |
|--------|-------|
| Raw satellite-link observations | 338 million |
| Aggregated receiver epochs | 16.7 million |
| Floor levels (z) | 7 — {1, 4, 7, 10, 13, 16, 19} m |
| Mean GNSS altitude MAE | 12.8 m |
| LiDAR trigger rate (|err| > 3 m) | 86.2 % |
| Majority-class baseline (EKF-D proxy) | 86.2 % accuracy |

### ML Classifier (XGBoost, GroupKFold spatial CV)

| Metric | Value |
|--------|-------|
| AUC (OOF ROC) | ≈ 0.999 |
| Top feature (SHAP) | VDOP |
| CV strategy | GroupKFold, grouped by rx_id |
| Training set | 4.5 M × 22 features |

### UAV Simulation (300 s, 23 waypoints, IITM campus)

| EKF | MAE (all) | MAE (canyon) | MAE (open) | LiDAR pulses | Sensor energy |
|-----|-----------|--------------|------------|--------------|---------------|
| A — chi² + LiDAR fallback | 0.185 m | 0.222 m | 0.090 m | 3 / 301 | ~39 J |
| **B — ML adaptive-R** ← | **0.098 m** | **0.097 m** | 0.101 m | **175 / 301** | **~1566 J** |
| C — always-LiDAR | 0.075 m | 0.085 m | 0.046 m | 301 / 301 | ~2423 J |
| D — blind GNSS | 0.323 m | 0.317 m | 0.337 m | 0 / 301 | ~15 J |

**EKF-B saves 857 J (35.4 %) vs always-LiDAR at only 12 mm canyon accuracy cost.**

### Sim-to-Real Transfer (660 real Android GNSS epochs, 7 sessions)

| Transfer mode | AUC | Notes |
|---------------|-----|-------|
| Zero-shot | 0.779 | Sionna model applied directly, no real training |
| Fine-tuned | 0.938 | 70 % real epochs for training, 30 % test |

Feature shift analysis:

| Feature | Shift (normalised by syn. IQR) | Root cause |
|---------|-------------------------------|------------|
| n_sats | 6.0× | Android satellite selection algorithm |
| mean_cn0 | 2.14× (−9.3 dB) | No vegetation in Sionna scene |
| vdop | 0.94× | Well-aligned (SP3 orbits identical) |
| elev_spread | 0.88× | Well-aligned |

---

## Repository Structure

```
GRAIL/
├── 01_simulation/
│   ├── smart_sampler.py          # geometry-aware sampler (7 floors)
│   └── sionna_gnss_pipeline.py   # Sionna 2.0 GNSS simulation + SP3 orbits
│
├── 02_ml_pipeline/
│   └── train_classifier.py       # Feature engineering +  CV + XGBoost
│
├── 03_uav_sim/
│   └── uav_altitude_sim.py       # 4-EKF UAV simulation with ML-in-the-loop
│
├── 04_sim2real/
│   └── sim2real_transfer.py      # Android GNSS parser + zero-shot/fine-tune
│
├── models/
│   ├── xgboost_classifier_v2.pkl # Trained XGBoost classifier (AUC ≈ 0.999)
│   └── xgboost_regressor.pkl     # Trained XGBoost z-error regressor
│
├── data/
│   ├── sampling_points_v6.csv    # 834 K receiver positions (7 floors)
│   ├── sim_flight_log_v4.csv     # 300-s UAV flight log (30 K ticks)
│   ├── real_gnss_features.csv    # 660 real GNSS epoch features
│   └── real_log_summary.csv      # Per-session statistics
│
├── real_data/
│   └── *.txt                     # 7 Android GNSSLogger log files
│
├── figures/
│   ├── sys_overview.png
│   ├── 01_motivation/            # fig1_motivation.png, sampling maps
│   ├── 02_signal_physics/        # fig2_signal_physics.png
│   ├── 03_ml_results/            # fig4_model_results.png
│   ├── 04_uav_sim/               # sim_results_v4.png, sim_3d_v4.png
│   └── 05_sim2real/              # sim_to_real_v2.png, distribution comparisons
│
└── gnss_vim_sim/                 # Python package: full simulation framework
    └── README.md                 # Detailed usage guide (BYO-scene workflow)
```

---

## Running the Pipeline

### Step 0 — Install dependencies

```bash
pip install numpy pandas scipy scikit-learn xgboost lightgbm shap matplotlib
pip install sionna trimesh          # for simulation steps
pip install osmnx shapely           # for new-scene OSM workflow
```

### Step 1 — Generate sampling points

```bash
python 01_simulation/smart_sampler.py \
    --mesh-dir <path/to/ply_meshes> \
    --out sampling_points.csv \
    --visualise
```

Outputs ~800 K receiver positions at wall-proximal, corner, and open-space
locations across 7 altitude levels.

### Step 2 — Run Sionna GNSS simulation

```bash
python 01_simulation/sionna_gnss_pipeline.py \
    --scene <path/to/scene.xml> \
    --sampling-csv sampling_points.csv \
    --sp3-dir <path/to/sp3_files/> \
    --out-agg  gnss_synthetic_agg.csv \
    --out-raw  gnss_synthetic_raw.csv \
    --lat 12.9906 --lon 80.2296 --alt 12.5
```

Generates per-satellite CIR features (raw) and per-epoch aggregated fingerprints.

### Step 3 — Train ML classifier

```bash
python 02_ml_pipeline/train_classifier.py \
    --raw  gnss_synthetic_raw.csv \
    --agg  gnss_synthetic_agg.csv \
    --out-dir ml_outputs/
```

Runs feature engineering, GroupKFold spatial CV, XGBoost/RF/MLP comparison,
SHAP analysis, and saves trained models.

### Step 4 — Run UAV simulation

```bash
python 03_uav_sim/uav_altitude_sim.py \
    --feat-csv gnss_ml_features.csv \
    --model models/xgboost_classifier_v2.pkl \
    --no-interact
```

Runs 4-EKF comparison over 300 seconds, saves flight_log.csv and result plots.

### Step 5 — Sim-to-real transfer

```bash
python 04_sim2real/sim2real_transfer.py \
    --real-dir real_data/ \
    --model models/xgboost_classifier_v2.pkl \
    --out-png sim_to_real.png
```

Parses real Android GNSS logs, evaluates zero-shot and fine-tuned AUC.

### Step 6 — GNSS-VIM-Sim (any new scene)

```bash
pip install -e gnss_vim_sim
gnss-vim-sim new-scene --lat 12.9906 --lon 80.2296 --name iitm
cd iitm && gnss-vim-sim run --config demo_config.json
```

---

## Key Scientific Finding

> *In urban canyons, even when line-of-sight is maintained, the purely geometric
> property of VDOP is sufficient to predict altitude error quality with ML, and this
> VDOP signal is detectable through delay spread and elevation features without
> requiring raw signal access.*

This finding generalises beyond IITM: any urban environment where buildings
obstruct low-elevation satellites produces the same VDOP-to-error relationship.

---

## Citation

```bibtex
@misc{grail2026,
  title  = {GRAIL: GNSS Ray-tracing for Altitude Inference and Localization},
  author = {Tharaneeshwaran Rajasekar},
  year   = {2026},
  school = {Department of Computer Science and Engineering, IIT Madras},
  note   = {BTP Final Report}
}
```

---

## License

MIT © 2026 Tharaneeshwaran V U, IIT Madras
