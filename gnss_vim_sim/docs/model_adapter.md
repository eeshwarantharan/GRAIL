# Runtime Model Adapter

The public model story should be generic:

> A custom model can run during flight and produce a scalar score that the simulator logs and policies can use.

## Current Adapter

The current implementation loads a pickle checkpoint and supports:

- `predict_proba(x)`: uses class-1 probability as the score,
- `predict(x)`: clips the returned scalar to `[0, 1]`.

The runner calls the model at supported sensor epochs, logs the score as `model_score`, keeps `ml_risk` as a backward-compatible alias, and uses the score in the current model-adaptive estimator policy.

## Public Interpretation

Do not hard-code the public explanation to one domain. The same scalar can mean:

- anomaly probability,
- sensor confidence,
- landing-zone quality,
- link degradation risk,
- perception failure probability,
- collision/clearance risk,
- mission abort probability,
- sensor duty-cycle trigger.

## Next Refactor

For a PyPI-ready release, rename the public interface:

- `IntegrityModel` -> `RuntimeModel`
- `predict_risk(features)` -> `predict_score(features)`
- `ml_risk` log column -> backward-compatible alias for `model_score`

Recommended adapter interface:

```python
class RuntimeModel:
    name: str
    feature_schema: list[str]

    def predict_score(self, features: dict[str, float], context: dict) -> float:
        ...
```

Recommended CLI shape:

```bash
gnss-vim-sim run \
  --config mission.json \
  --model-checkpoint model.pkl \
  --model-schema configs/model_features.json \
  --policy configs/policy.json
```

This keeps the tool useful as a general data generator even when no model is attached.
