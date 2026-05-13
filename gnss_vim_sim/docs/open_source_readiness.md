# Open Source Readiness

The public project should not lead with a specific research novelty. It should be positioned as a general drone simulation and data-generation tool.

Recommended public pitch:

> A lightweight drone-in-mesh simulator for mission planning, sensor logging, energy accounting, ML-in-the-loop experiments, and reproducible flight visualizations.

This keeps the package useful to others while preserving the thesis-specific vertical-integrity application as a private or example-level use case.

## Public Identity

Avoid:

- "GNSS vertical integrity simulator"
- "learned VDOP augmentation"
- "Sionna/SP3/GNSS novelty"
- thesis-specific claims in the first paragraph

Prefer:

- drone-in-mesh simulation,
- sensor data generation,
- WebGL mission planning and replay,
- custom ML model hooks,
- reproducible route, sensor, estimator, and energy artifacts.

The current internal name can stay for now, but a public package could use a neutral name:

- `mesh-uav-lab`
- `uav-mesh-sim`
- `drone-mesh-lab`
- `aerialsim-lite`

## What Works Today

- Local mesh scenes loaded from PLY files through `trimesh`.
- ENU/geodetic coordinate handling.
- WebGL mission studio for waypoint creation and editing.
- A* route planning over mesh-derived obstacle geometry.
- Pose-driven UAV sensor stack: IMU, barometer, GNSS-like receiver, rangefinder.
- Optional checkpoint-based ML scoring during simulation.
- Estimator/policy variants and energy ledger.
- WebGL flight replay, static plots, CSV logs, route artifacts, and JSON summaries.

## What The Tool Is

It is a data generator and experiment runner:

- define a scene,
- plan a path,
- fly a configurable drone model,
- simulate sensors,
- optionally call a custom model,
- log everything,
- replay the flight and inspect plots.

That framing is broad, useful, and defensible.

## What The Tool Is Not

- Not a full physics replacement for Gazebo/Isaac.
- Not a propeller/aerodynamics simulator.
- Not a GNSS-only package.
- Not tied to a single campus mesh or thesis dataset.

## Custom ML Model Contract

Publicly describe the model hook as:

> A runtime model adapter that receives per-epoch sensor/context features and returns a scalar score used by user-configurable policies.

Examples of possible score meanings:

- sensor anomaly probability,
- landing-zone safety,
- navigation risk,
- link-quality risk,
- range-aiding trigger,
- perception confidence,
- mission abort probability.

The current implementation supports pickle checkpoints with `predict_proba`/`predict`. Before public release, generalize names and docs from "integrity model" to "runtime model adapter."

## Public Artifacts

Every run should clearly save:

- `flight_log.csv`: one row per sim tick with truth, sensors, model outputs, estimator states, and route state.
- `summary.json`: metrics, energy, route planner status, scene stats, model metadata.
- `route_plan.csv/json`: the planned path.
- `flight_player.html`: WebGL replay and plots.
- `mission_studio.html`: optional planned mission artifact.
- `plots/`: static PNGs for reports.

Before release, consider moving PNGs into a `plots/` subfolder and keeping root run folders tidy.

## Before PyPI

- Add `gnss-vim-sim init` to create sample config, folders, and a tiny mesh.
- Add strict config validation with readable errors.
- Add public demo assets that are small and license-clean.
- Add tests for coordinate conversion, scene loading, planning, sensor outputs, logging, and summary metrics.
- Add a neutral public README and keep thesis-specific configs/examples separate.
- Add a formal model-adapter interface:
  - pickle/sklearn,
  - ONNX,
  - TorchScript,
  - user Python class path,
  - optional HTTP inference endpoint.
- Add schema docs for `flight_log.csv` and `summary.json`.
- Add a short "bring your own mesh" guide.

## Private Research Positioning

The thesis-specific GNSS vertical-integrity pitch can remain in private notes, papers, and dedicated examples. It does not need to define the public package.
