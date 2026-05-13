from __future__ import annotations

import math
import numpy as np


def estimator_metrics(rows: list[dict], key: str) -> dict[str, float]:
    err = np.array([abs(r[key] - r["true_z"]) for r in rows], dtype=float)
    return {
        "mae_m": float(np.mean(err)),
        "rmse_m": float(math.sqrt(np.mean(err * err))),
        "p95_m": float(np.percentile(err, 95)),
        "max_m": float(np.max(err)),
    }


def binary_metrics(rows: list[dict]) -> dict[str, float]:
    y = np.array([bool(r["gnss_bad_truth"]) for r in rows if r["gnss_epoch"]], dtype=bool)
    p = np.array([float(r["ml_risk"]) for r in rows if r["gnss_epoch"]], dtype=float)
    if len(y) == 0:
        return {}
    pred = p >= 0.5
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "bad_epoch_rate": float(np.mean(y)),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "false_negative_rate_at_0_5": float(fn / max(tp + fn, 1)),
        "f1_at_0_5": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def improvement_metrics(rows: list[dict]) -> dict[str, float]:
    fixed = estimator_metrics(rows, "fixed_gnss_z")
    vdop = estimator_metrics(rows, "vdop_chi2_z")
    ml = estimator_metrics(rows, "ml_integrity_z")
    always = estimator_metrics(rows, "always_range_z")
    return {
        "ml_mae_gain_vs_fixed_pct": float((fixed["mae_m"] - ml["mae_m"]) / max(fixed["mae_m"], 1e-12) * 100.0),
        "ml_mae_gain_vs_vdop_pct": float((vdop["mae_m"] - ml["mae_m"]) / max(vdop["mae_m"], 1e-12) * 100.0),
        "ml_mae_gap_vs_always_range_pct": float((ml["mae_m"] - always["mae_m"]) / max(always["mae_m"], 1e-12) * 100.0),
    }
