from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import pickle
import warnings

from gnss_vim_sim.sensors.gnss import FEATURE_COLS


class RuntimeModel:
    """Neutral runtime model interface for ML-in-the-loop simulation."""

    name = "runtime_model"
    feature_schema = FEATURE_COLS

    def predict_score(self, features: dict[str, float], context: dict | None = None) -> float:
        raise NotImplementedError

    def predict_risk(self, features: dict[str, float]) -> float:
        """Backward-compatible alias for older experiment code."""
        return self.predict_score(features)


@dataclass
class HeuristicRuntimeModel(RuntimeModel):
    """Transparent fallback baseline from observable sensor quality features."""

    name = "heuristic_runtime_model"

    def predict_score(self, features: dict[str, float], context: dict | None = None) -> float:
        vdop = features.get("vdop", 2.0)
        cn0 = features.get("mean_cn0", 38.0)
        n_sats = features.get("n_sats", 8.0)
        lock = features.get("phase_locked_frac", 1.0)
        delay = features.get("mean_delay_ns", 1.0)
        score = 0.9 * (vdop - 2.0) + 0.08 * (35.0 - cn0) + 0.22 * (7.0 - n_sats)
        score += 1.2 * (0.8 - lock) + 0.18 * (delay - 2.0)
        return float(1.0 / (1.0 + math.exp(-score)))


class PickleRuntimeModel(RuntimeModel):
    """Adapter for sklearn/xgboost/lightgbm-style checkpoints."""

    name = "pickle_runtime_model"

    def __init__(self, checkpoint: Path):
        self.checkpoint = checkpoint
        with checkpoint.open("rb") as f:
            self.model = pickle.load(f)

    def predict_score(self, features: dict[str, float], context: dict | None = None) -> float:
        x = [[float(features.get(col, 0.0)) for col in FEATURE_COLS]]
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(x)[0][1])
        if hasattr(self.model, "predict"):
            y = self.model.predict(x)
            return float(max(0.0, min(1.0, y[0])))
        return 0.5


def _bundled_model_path() -> Path | None:
    """Return path to the GRAIL pretrained classifier bundled with the package."""
    try:
        from importlib import resources
        assets = resources.files("gnss_vim_sim.assets")
        with resources.as_file(assets / "grail_classifier.pkl") as p:
            if p.exists():
                return p
    except Exception:
        pass
    return None


def load_runtime_model(checkpoint: Path | None) -> RuntimeModel:
    """
    Load a runtime model in priority order:
      1. User-supplied checkpoint (--model-checkpoint)
      2. Bundled GRAIL XGBoost classifier (grail_classifier.pkl, AUC ≈ 0.999)
      3. VDOP/C/N₀ heuristic fallback (no ML, no file I/O)
    """
    if checkpoint is not None:
        try:
            return PickleRuntimeModel(checkpoint)
        except Exception as exc:
            warnings.warn(
                f"Could not load runtime model {checkpoint}: {exc}. "
                "Trying bundled GRAIL model."
            )

    bundled = _bundled_model_path()
    if bundled is not None:
        try:
            model = PickleRuntimeModel(bundled)
            print(f"  [ML] Loaded bundled GRAIL classifier: {bundled.name}")
            return model
        except Exception as exc:
            warnings.warn(f"Bundled model load failed: {exc}. Falling back to heuristic.")

    print("  [ML] Using VDOP/C/N₀ heuristic (no trained model available)")
    return HeuristicRuntimeModel()


IntegrityModel = RuntimeModel
HeuristicIntegrityModel = HeuristicRuntimeModel
PickleIntegrityModel = PickleRuntimeModel
load_integrity_model = load_runtime_model
