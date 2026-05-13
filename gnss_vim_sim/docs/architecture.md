# Architecture

GNSS-VIM-Sim is organized as a modular UAV-in-mesh simulation pipeline. The public architecture is intentionally general: mesh scene in, mission route in, simulated flight and sensor logs out.

## Data Flow

1. A config defines a local mesh scene, ENU/geodetic origin, vehicle settings, sensors, estimator settings, energy constants, and mission waypoints.
2. The WebGL mission studio emits editable ENU waypoints in the same coordinate frame as the mesh.
3. The route planner expands requested waypoints into a mesh-aware path when obstacle geometry is available.
4. The vehicle model produces truth pose, velocity, and acceleration along the route.
5. Sensor models derive measurements from truth plus configurable noise, bias, dropout, and scene effects.
6. Optional runtime ML models consume feature dictionaries at supported epochs and emit scalar scores.
7. Estimator/control policies consume sensor data and optional model scores.
8. The logger writes truth, measurements, features, model outputs, estimator states, route state, activations, and energy.
9. Visualizers generate WebGL replay, static plots, and machine-readable summaries.

## Module Boundaries

- `world`: mesh loading, scene statistics, building/obstacle extraction, ray queries, WebGL mesh export.
- `planning`: waypoint missions, multirotor route tracking, A* mesh-aware route expansion.
- `sensors`: pose-driven IMU, barometer, GNSS-like position receiver, rangefinder.
- `ml`: checkpoint adapters and runtime scoring hooks.
- `estimators`: EKF/policy variants that consume sensor measurements and model outputs.
- `sim`: runner, metrics, and artifact writing.
- `viz`: WebGL mission studio, WebGL flight player, static plots, and optional dashboard.

## Drone Model

The current vehicle is a research multirotor abstraction. It is not a full rigid-body aerodynamics simulator. The purpose is reproducible sensor/data generation and estimator comparison, so the model keeps the math inspectable:

- waypoint-following trajectory,
- configurable cruise speed and waypoint acceptance,
- truth state containing position, velocity, and acceleration,
- deterministic random seed for repeatable sensor noise and route outcomes.

This is a good fit for perception, navigation, estimator, data-generation, and ML-in-the-loop experiments where the scene and sensor signals matter more than propeller-level dynamics.

## Sensor Stack

Current built-in sensors:

- IMU: acceleration/gyro-style perturbations and high-rate propagation input.
- Barometer: altitude measurement with noise, slow drift, and scene-proximity disturbance.
- GNSS-like receiver: geodetic position, covariance-like accuracy fields, dropout, satellite-count/C/N0-style quality features.
- Rangefinder/LiDAR: downward mesh raycast when mesh dependencies are available.
- Energy ledger: base vehicle energy plus marginal sensing/compute energy.

The sensor interface is intentionally pose-driven: truth goes in, noisy measurement plus metadata comes out.

## ML Runtime Hook

The current checkpoint hook accepts sklearn/xgboost/lightgbm-style models with `predict_proba` or `predict`. The runner calls the model on per-epoch feature dictionaries and logs the scalar score.

In the default research scenario this score is used as a vertical measurement-risk signal. In a general simulator release, the same hook can represent:

- sensor anomaly risk,
- landing-zone quality,
- link degradation probability,
- terrain or corridor class,
- collision/clearance risk,
- perception confidence,
- mission abort likelihood.

The next public-facing refactor should rename this layer from domain-specific "integrity" wording to a neutral `RuntimeModel` / `ModelAdapter` interface while keeping backward compatibility.

## Estimator And Policy Layer

The current estimator layer compares several policies over the same sensor stream:

- fixed sensor covariance,
- quality-derived covariance,
- range-aided baseline,
- model-adaptive covariance/range policy.

For open-source positioning, describe these as "policy variants" rather than thesis claims. Users can swap in their own policy logic later.

## Coordinate Policy

The simulator assumes a local ENU frame:

- mesh vertices are already aligned to the local scene frame,
- waypoints are expressed in ENU meters,
- a geodetic origin converts truth/sensor outputs to lat/lon/alt,
- logs preserve both ENU and geodetic forms.

This keeps the tool usable across arbitrary campus, factory, warehouse, city-block, or synthetic mesh scenes.

## Public Extension Points

- Mesh import adapters: OBJ/GLB/CityGML/OSM preprocessing into PLY or direct trimesh scenes.
- Sensor adapters: camera, depth, magnetometer, UWB, Wi-Fi/LoRa link quality, battery health.
- Runtime model adapters: pickle, ONNX, TorchScript, HTTP/gRPC model server.
- Policy adapters: estimator covariance policy, flight abort policy, sensor duty-cycling policy.
- Data adapters: convert `flight_log.csv` into ML training datasets for arbitrary downstream tasks.
