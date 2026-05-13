# GNSS-VIM-Sim Runbook

This is the normal workflow for repeatable runs.

## 1. Install

```bash
cd ~/Documents/New_V2_Sionna/gnss_vim_sim
pip install -e ".[mesh,ml,data]"
```

Create a license-clean demo project:

```bash
gnss-vim-sim init --out demo_project
```

## 2. Inspect Mesh

```bash
gnss-vim-sim inspect-scene --config configs/iitm_demo.json
```

You want nonzero `vertex_count`. If it is zero, mesh raycasting and A* routing fall back to simpler modes.

Validate before a run:

```bash
gnss-vim-sim validate --config configs/iitm_demo.json --require-mesh
```

## 3. Plan A Mission

Guided browser planner plus run:

```bash
gnss-vim-sim wizard \
  --config configs/iitm_demo.json
```

The command creates a timestamped workflow folder, opens `mission_planner.html`, asks you to save the generated mission JSON, then runs the flight after you paste the JSON path.

Mission studio only:

```bash
gnss-vim-sim studio --config configs/iitm_demo.json --out runs/mission_studio.html
```

Open `runs/mission_studio.html`, orbit/pan/zoom the OSM mesh, click the ground plane for takeoff/waypoints/landing, edit selected points in the side panel, then download the generated mission JSON into `configs/my_mission.json`.

## 4. Run Flight

Without ML checkpoint:

```bash
gnss-vim-sim run --config configs/my_mission.json
```

With a custom runtime model checkpoint:

```bash
gnss-vim-sim run \
  --config configs/my_mission.json \
  --model-checkpoint path/to/model.pkl
```

If `--out` is omitted, results go to a timestamped folder under `runs/`.

## 5. Open Results

Each run folder contains:

- `flight_player.html`: fast WebGL 30fps browser flight player with OSM mesh, drone icon, chase-45/top/wide views, 1x/2x/4x/8x playback, plots, read-only telemetry, and model/sensor-state route coloring.
- `dashboard.html`: heavier interactive analysis dashboard, only when `--dashboard` is requested.
- `summary_panel.png`, `altitude_fusion.png`, `mission_map.png`, `flight_3d_risk.png`.
- `flight_log.csv`, `summary.json`, `route_plan.csv`, `route_plan.json`.

To regenerate the player from an existing log:

```bash
gnss-vim-sim player \
  --log runs/ml_demo/flight_log.csv \
  --config configs/iitm_demo.json \
  --out runs/ml_demo/flight_player.html \
  --drone-icon ../drone.png
```

To regenerate the dashboard:

```bash
gnss-vim-sim dashboard \
  --log runs/ml_demo/flight_log.csv \
  --config configs/iitm_demo.json \
  --out runs/ml_demo/dashboard.html
```

To generate it during a run:

```bash
gnss-vim-sim run --config configs/my_mission.json --dashboard
```
