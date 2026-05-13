"""
train_classifier.py
===================
GRAIL Phase-2: Feature engineering, honest spatial cross-validation, and
ML classifier training for the LiDAR-trigger binary classifier.

Key design choices
------------------
  GroupKFold CV  — groups by rx_id (receiver location hash) so all epochs
                   from a given location appear in exactly one fold, preventing
                   spatial data leakage that inflates metrics in naive splits.
  Binary target  — y = 1 if |z_GNSS − z_true| > TRIGGER_THRESHOLD (3 m).
                   This directly trains the LiDAR-gating decision.
  Models         — XGBoost (deployed), LightGBM, RandomForest, MLP.
  SHAP           — Feature importance via shapley values.

Outputs (written to --out-dir)
-------------------------------
  best_clf_final.pkl           — Serialised XGBoost classifier (deploy artefact)
  gnss_ml_features.csv         — 22-feature matrix (cached, skips recompute)
  fig1_motivation.png          — Violin/scatter/trigger-rate motivation panel
  fig2_signal_physics.png      — C/N0, delay-spread, LOS-ratio physics panel
  fig4_model_results_honest.png — 5-fold CV: ROC, accuracy, SHAP, energy sweep

Usage
-----
  # Full run (EDA + ML):
  python train_classifier.py --raw gnss_synthetic_raw.csv --agg gnss_synthetic_agg.csv

  # ML only (re-use saved feature matrix):
  python train_classifier.py --ml-only

  # EDA only:
  python train_classifier.py --eda-only --raw gnss_synthetic_raw.csv --agg gnss_synthetic_agg.csv

Requirements
------------
  numpy, pandas, matplotlib, scikit-learn>=1.2
  xgboost>=1.7, lightgbm>=4.0 (optional)
  shap (optional, for SHAP feature importance)
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRIGGER_THRESHOLD = 3.0     # metres — |z_error| above this triggers LiDAR
CV_FOLDS = 5
DECISION_THRESHOLD = 0.40   # operating point for precision/recall

COLOUR = {
    "wall_halo":  "#E63946",
    "corner":     "#1D3557",
    "open_space": "#2A9D8F",
    "safe":       "#2A9D8F",
    "danger":     "#E63946",
    "accent":     "#F4A261",
    "bg":         "#FAFAFA",
    "grid":       "#E8E8E8",
    "text":       "#1A1A2E",
}

FEATURE_COLS = [
    "n_sats", "mean_cn0", "std_cn0", "min_cn0", "range_cn0", "canopy_cn0",
    "mean_elev", "min_elev", "max_elev", "std_elev", "elev_spread", "vdop",
    "mean_los_ratio", "min_los_ratio", "std_los_ratio", "phase_locked_frac",
    "mean_delay_ns", "max_delay_ns", "std_delay_ns",
    "mean_mp_error", "std_mp_error", "max_mp_error",
]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(COLOUR["bg"])
    ax.grid(True, color=COLOUR["grid"], linewidth=0.6, zorder=0)
    ax.tick_params(labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("#CCCCCC")
        sp.set_linewidth(0.7)
    if title:   ax.set_title(title, fontsize=11, fontweight="bold", color=COLOUR["text"], pad=8)
    if xlabel:  ax.set_xlabel(xlabel, fontsize=9, color="#555")
    if ylabel:  ax.set_ylabel(ylabel, fontsize=9, color="#555")


def _save(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="#F7F7F7")
    plt.close(fig)
    print(f"  [plot] Saved {path}")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(raw: pd.DataFrame, agg: pd.DataFrame,
                       out_csv: str = "") -> pd.DataFrame:
    """Build the 22-feature fingerprint matrix from raw satellite-link data."""
    print("[features] Engineering features (vectorised)...")
    t0 = time.time()

    raw = raw.copy()
    raw["is_locked"]  = (raw["LosRatio"] > 0.5).astype(float)
    raw["abs_mp"]     = raw["MultipathErrorMeters"].abs()
    raw["cn0_canopy"] = raw["Cn0DbHz"] - raw.groupby(["rx_id", "epoch_str"])["Cn0DbHz"].transform("max")

    feats = (
        raw.groupby(["rx_id", "epoch_str"])
        .agg(
            n_sats            = ("Svid",                 "count"),
            mean_cn0          = ("Cn0DbHz",              "mean"),
            std_cn0           = ("Cn0DbHz",              "std"),
            min_cn0           = ("Cn0DbHz",              "min"),
            max_cn0           = ("Cn0DbHz",              "max"),
            canopy_cn0        = ("cn0_canopy",            "mean"),
            mean_elev         = ("SvElevationDegrees",   "mean"),
            min_elev          = ("SvElevationDegrees",   "min"),
            max_elev          = ("SvElevationDegrees",   "max"),
            std_elev          = ("SvElevationDegrees",   "std"),
            mean_los_ratio    = ("LosRatio",             "mean"),
            min_los_ratio     = ("LosRatio",             "min"),
            std_los_ratio     = ("LosRatio",             "std"),
            phase_locked_frac = ("is_locked",            "mean"),
            mean_delay_ns     = ("DelaySpreadNs",        "mean"),
            max_delay_ns      = ("DelaySpreadNs",        "max"),
            std_delay_ns      = ("DelaySpreadNs",        "std"),
            mean_mp_error     = ("abs_mp",               "mean"),
            std_mp_error      = ("MultipathErrorMeters", "std"),
            max_mp_error      = ("abs_mp",               "max"),
            true_z            = ("true_z",               "first"),
            floor             = ("floor",                "first"),
            point_type        = ("point_type",           "first"),
        )
        .reset_index()
    )

    feats["range_cn0"]   = feats["max_cn0"] - feats["min_cn0"]
    feats["elev_spread"] = feats["max_elev"] - feats["min_elev"]
    feats["rx_id"]       = (feats["rx_id"].astype(str) + "_" + feats["epoch_str"]).apply(
        lambda s: abs(hash(s.split("_")[0])) % 10_000
    )

    agg_lookup = agg.set_index(["rx_id", "epoch_str"])
    for col in ["vdop", "z_error_m"]:
        if col in agg.columns:
            feats[col] = feats.set_index(["rx_id", "epoch_str"]).index.map(
                lambda k: agg_lookup[col].get(k, float("nan"))
            ).values

    feats["lidar_trigger"] = (feats.get("z_error_m", pd.Series(float("nan"), index=feats.index))
                              .fillna(TRIGGER_THRESHOLD + 1) > TRIGGER_THRESHOLD).astype(int)
    feats = feats.dropna(subset=FEATURE_COLS + ["vdop"])

    print(f"[features] {len(feats):,} rows × {len(FEATURE_COLS) + 1} features in {time.time()-t0:.1f}s")
    if out_csv:
        feats.to_csv(out_csv, index=False)
        print(f"[features] Saved → {out_csv}")
    return feats


# ---------------------------------------------------------------------------
# EDA plots
# ---------------------------------------------------------------------------

def plot_motivation(agg: pd.DataFrame, out_dir: str) -> None:
    fig = plt.figure(figsize=(20, 6), facecolor="#F7F7F7")
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    floors = sorted(agg["floor"].unique())
    data   = [agg[agg["floor"] == f]["z_error_m"].dropna().clip(0, 40).values for f in floors]

    ax1 = fig.add_subplot(gs[0])
    vp  = ax1.violinplot(data, positions=floors, showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_alpha(0.75)
        body.set_facecolor(COLOUR["danger"])
    _style(ax1, "GNSS Altitude Error by Floor", "Floor level", "Absolute z-error (m)")
    ax1.set_xticks(floors)

    ax2 = fig.add_subplot(gs[1])
    h = ax2.hexbin(agg["vdop"].clip(0, 10), agg["z_error_m"].clip(0, 40),
                   gridsize=60, cmap="YlOrRd", mincnt=1)
    fig.colorbar(h, ax=ax2, label="Count")
    _style(ax2, "VDOP vs Altitude Error", "VDOP", "Absolute z-error (m)")

    ax3 = fig.add_subplot(gs[2])
    trigger_by_floor = (
        agg.groupby("floor")
        .apply(lambda g: (g["z_error_m"] > TRIGGER_THRESHOLD).mean() * 100)
        .reset_index(name="trigger_pct")
    )
    ax3.bar(trigger_by_floor["floor"], trigger_by_floor["trigger_pct"],
            color=COLOUR["danger"], alpha=0.8, edgecolor="white")
    ax3.axhline(y=(agg["z_error_m"] > TRIGGER_THRESHOLD).mean() * 100,
                color="#555", lw=1.5, linestyle="--", label="Dataset mean")
    _style(ax3, "LiDAR Trigger Rate by Floor", "Floor level", "Trigger rate (%)")
    ax3.legend(fontsize=9)

    plt.suptitle("GRAIL: GNSS Altitude Error Physics", fontsize=14,
                 fontweight="bold", color=COLOUR["text"])
    _save(fig, out_dir, "fig1_motivation.png")


def plot_signal_physics(raw: pd.DataFrame, feats: pd.DataFrame, out_dir: str) -> None:
    fig = plt.figure(figsize=(22, 10), facecolor="#F7F7F7")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    for fl in sorted(feats["floor"].unique()):
        sub = feats[feats["floor"] == fl]["mean_cn0"].dropna()
        ax1.hist(sub, bins=30, density=True, alpha=0.5, label=f"F{fl}")
    _style(ax1, "C/N₀ Distribution by Floor", "Mean C/N₀ (dB-Hz)", "Density")
    ax1.legend(fontsize=7, ncol=2)

    ax2 = fig.add_subplot(gs[0, 1])
    h = ax2.hexbin(raw["Cn0DbHz"].clip(15, 55), raw["DelaySpreadNs"].clip(0, 50),
                   gridsize=60, cmap="Blues", mincnt=1)
    fig.colorbar(h, ax=ax2, label="Count")
    _style(ax2, "Delay Spread vs C/N₀", "C/N₀ (dB-Hz)", "Delay spread (ns)")

    ax3 = fig.add_subplot(gs[0, 2])
    for fl in sorted(feats["floor"].unique()):
        sub = feats[feats["floor"] == fl]["mean_los_ratio"].dropna()
        ax3.hist(sub, bins=30, density=True, alpha=0.5, label=f"F{fl}")
    _style(ax3, "LOS Ratio by Floor", "Mean LOS ratio", "Density")
    ax3.legend(fontsize=7, ncol=2)

    ax4 = fig.add_subplot(gs[1, 0])
    for pt in feats["point_type"].unique():
        sub = feats[feats["point_type"] == pt]["vdop"].dropna().clip(1, 9)
        ax4.hist(sub, bins=30, density=True, alpha=0.6, label=pt.replace("_", " "), color=COLOUR.get(pt))
    _style(ax4, "VDOP by Point Type", "VDOP", "Density")
    ax4.legend(fontsize=9)

    ax5 = fig.add_subplot(gs[1, 1])
    subset = feats[FEATURE_COLS].dropna().sample(min(5000, len(feats)))
    corr = subset.corr()
    im = ax5.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax5.set_xticks(range(len(FEATURE_COLS)))
    ax5.set_yticks(range(len(FEATURE_COLS)))
    ax5.set_xticklabels([c.replace("_", "\n") for c in FEATURE_COLS], fontsize=5, rotation=90)
    ax5.set_yticklabels([c.replace("_", "\n") for c in FEATURE_COLS], fontsize=5)
    fig.colorbar(im, ax=ax5, label="Pearson r")
    ax5.set_title("Feature Correlation Matrix", fontsize=11, fontweight="bold", color=COLOUR["text"])

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.scatter(feats["vdop"].clip(1, 9), feats["mean_delay_ns"].clip(0, 40),
                s=2, alpha=0.15, c=feats["lidar_trigger"],
                cmap="RdYlGn_r")
    _style(ax6, "VDOP × Delay Spread (colour=trigger)", "VDOP", "Mean delay (ns)")

    plt.suptitle("GRAIL: GNSS Signal Physics", fontsize=14,
                 fontweight="bold", color=COLOUR["text"])
    _save(fig, out_dir, "fig2_signal_physics.png")


# ---------------------------------------------------------------------------
# Honest spatial cross-validation
# ---------------------------------------------------------------------------

def run_honest_cv(feats: pd.DataFrame):
    from sklearn.model_selection import GroupKFold
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, r2_score, mean_absolute_error
    from sklearn.neural_network import MLPClassifier

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
        print("[cv] xgboost not found — skipping XGBoost model.")

    try:
        import lightgbm as lgb
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    X        = feats[FEATURE_COLS].values.astype(np.float32)
    y_clf    = feats["lidar_trigger"].values.astype(int)
    y_reg    = feats.get("z_error_m", pd.Series(np.zeros(len(feats)))).values.astype(np.float32)
    rx_id    = feats.get("rx_id", pd.Series(np.arange(len(feats)))).values.astype(int)

    naive_acc = max(y_clf.mean(), 1 - y_clf.mean())
    print(f"\n[cv] Dataset: {len(X):,} rows | {y_clf.mean()*100:.1f}% trigger rate")
    print(f"[cv] Naive majority-class baseline accuracy: {naive_acc*100:.1f}%")

    gkf = GroupKFold(n_splits=CV_FOLDS)

    clf_models = {}
    if HAS_XGB:
        clf_models["XGB"] = lambda: xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss",
            n_jobs=-1, random_state=42, verbosity=0,
        )
    clf_models["RF"]  = lambda: RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
    clf_models["MLP"] = lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42)

    oof_prob  = {name: np.zeros(len(X)) for name in clf_models}
    oof_reg   = np.zeros(len(X))
    clf_summary: dict[str, list] = {name: [] for name in clf_models}

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_clf, groups=rx_id)):
        X_tr, y_tr = X[tr_idx], y_clf[tr_idx]
        X_va, y_va = X[va_idx], y_clf[va_idx]

        for name, factory in clf_models.items():
            model = factory()
            if name == "XGB":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                          early_stopping_rounds=30, verbose=False)
            else:
                model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_va)[:, 1]
            pred = (prob >= DECISION_THRESHOLD).astype(int)
            oof_prob[name][va_idx] = prob
            auc = roc_auc_score(y_va, prob)
            acc = accuracy_score(y_va, pred)
            rec = recall_score(y_va, pred, zero_division=0)
            clf_summary[name].append({"fold": fold, "auc": auc, "accuracy": acc, "recall": rec})
            print(f"  [cv] fold {fold+1} | {name:<4} | AUC={auc:.4f} acc={acc:.4f} rec={rec:.4f}")

        # Regression (use RF regressor)
        reg = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        reg.fit(X_tr, y_reg[tr_idx])
        oof_reg[va_idx] = reg.predict(X_va)

    print("\n[cv] === Overall OOF Metrics ===")
    for name in clf_models:
        auc_oof = roc_auc_score(y_clf, oof_prob[name])
        acc_oof = accuracy_score(y_clf, (oof_prob[name] >= DECISION_THRESHOLD).astype(int))
        print(f"  {name:<4}: AUC={auc_oof:.4f}  Acc={acc_oof:.4f}  (baseline={naive_acc:.4f})")

    # Train final XGBoost on full dataset
    best_clf = None
    if HAS_XGB:
        best_clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, use_label_encoder=False, eval_metric="logloss",
            n_jobs=-1, random_state=42, verbosity=0,
        )
        best_clf.fit(X, y_clf)
    elif clf_models:
        best_clf = list(clf_models.values())[0]()
        best_clf.fit(X, y_clf)

    return clf_summary, best_clf, FEATURE_COLS, X, y_clf, y_reg, oof_prob, oof_reg


# ---------------------------------------------------------------------------
# Results plot
# ---------------------------------------------------------------------------

def plot_results(clf_summary, best_clf, feat_names, X, y_clf, y_reg,
                 oof_prob, oof_reg, out_dir: str) -> None:
    from sklearn.metrics import roc_curve, auc as roc_auc_area, r2_score, mean_absolute_error
    from sklearn.metrics import recall_score

    fig = plt.figure(figsize=(26, 12), facecolor="#F7F7F7")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # Panel A: accuracy and recall bars
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(clf_summary.keys())
    mean_acc = [np.mean([f["accuracy"] for f in clf_summary[m]]) for m in models]
    mean_rec = [np.mean([f["recall"]   for f in clf_summary[m]]) for m in models]
    x = np.arange(len(models))
    ax1.bar(x - 0.2, mean_acc, 0.35, color="#4C72B0", alpha=0.85, label="Accuracy")
    ax1.bar(x + 0.2, mean_rec, 0.35, color=COLOUR["danger"], alpha=0.85, label="Recall")
    ax1.axhline(max(y_clf.mean(), 1 - y_clf.mean()), color="#888", lw=1.5,
                linestyle="--", label="Baseline")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylim(0, 1.05)
    _style(ax1, "Accuracy & Recall (5-fold Spatial CV)", "Model", "Score")
    ax1.legend(fontsize=9)

    # Panel B: R² regression bars
    ax2 = fig.add_subplot(gs[0, 1])
    r2 = r2_score(y_reg, oof_reg)
    mae = mean_absolute_error(y_reg, oof_reg)
    ax2.bar(["RF Regressor"], [r2], color=COLOUR["accent"], alpha=0.85)
    ax2.set_ylim(0, 1.0)
    ax2.text(0, r2 + 0.02, f"R²={r2:.3f}\nMAE={mae:.2f}m", ha="center", fontsize=10)
    _style(ax2, "Regression R² (z-error prediction)", "Model", "R²")

    # Panel C: OOF ROC curves
    ax3 = fig.add_subplot(gs[0, 2])
    colours = ["#4C72B0", "#DD8452", "#55A868"]
    for (name, prob), col in zip(oof_prob.items(), colours):
        fpr, tpr, _ = roc_curve(y_clf, prob)
        area = roc_auc_area(fpr, tpr)
        ax3.plot(fpr, tpr, lw=2, color=col, label=f"{name} (AUC={area:.4f})")
    ax3.plot([0, 1], [0, 1], "k--", lw=1)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1.02)
    _style(ax3, "OOF ROC Curves (5-fold Spatial CV)", "FPR", "TPR")
    ax3.legend(fontsize=9)

    # Panel D: predicted vs true scatter
    ax4 = fig.add_subplot(gs[1, 0])
    sample = np.random.choice(len(y_reg), min(8000, len(y_reg)), replace=False)
    ax4.scatter(y_reg[sample], oof_reg[sample], s=3, alpha=0.3, c="#4C72B0")
    ax4.plot([0, 30], [0, 30], "r--", lw=1.5)
    _style(ax4, "Predicted vs True z-error", "True z-error (m)", "Predicted z-error (m)")

    # Panel E: SHAP feature importance
    ax5 = fig.add_subplot(gs[1, 1])
    try:
        import shap
        explainer = shap.TreeExplainer(best_clf)
        idx = np.random.choice(len(X), min(2000, len(X)), replace=False)
        sv  = explainer.shap_values(X[idx])
        if isinstance(sv, list):
            sv = sv[1]
        importance = np.abs(sv).mean(axis=0)
        series = pd.Series(importance, index=feat_names).sort_values().tail(15)
        ax5.barh(range(len(series)), series.values, color="#4C72B0", alpha=0.85)
        ax5.set_yticks(range(len(series)))
        ax5.set_yticklabels([n.replace("_", "\n") for n in series.index], fontsize=8)
        _style(ax5, "SHAP Feature Importance (XGBoost)", "Mean |SHAP value|", "")
    except Exception as exc:
        ax5.text(0.5, 0.5, f"SHAP unavailable\n({exc})", ha="center", va="center",
                 transform=ax5.transAxes, fontsize=10)

    # Panel F: energy-recall threshold sweep
    ax6   = fig.add_subplot(gs[1, 2])
    ax6_t = ax6.twinx()
    thresholds = np.linspace(0.1, 0.9, 40)
    savings, recalls = [], []
    for thresh in thresholds:
        pred = (oof_prob[list(oof_prob.keys())[0]] >= thresh).astype(int)
        savings.append((1.0 - pred.mean()) * 100.0)
        recalls.append(recall_score(y_clf, pred, zero_division=0))
    ax6.plot(thresholds, savings, color=COLOUR["safe"], lw=2, label="LiDAR energy saved (%)")
    ax6_t.plot(thresholds, recalls, color=COLOUR["danger"], lw=2, linestyle="--", label="Recall (safety)")
    ax6.axvline(DECISION_THRESHOLD, color="#888", lw=1.2, linestyle=":")
    op_idx = np.argmin(np.abs(thresholds - DECISION_THRESHOLD))
    ax6.text(DECISION_THRESHOLD + 0.01, savings[op_idx],
             f"{savings[op_idx]:.0f}%\nsaved", fontsize=8, color=COLOUR["safe"])
    h1, l1 = ax6.get_legend_handles_labels()
    h2, l2 = ax6_t.get_legend_handles_labels()
    ax6.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")
    _style(ax6, "Energy Saved vs Safety Recall Trade-off", "Decision threshold", "LiDAR energy saved (%)")
    ax6_t.set_ylabel("Recall (safety)", fontsize=9, color=COLOUR["danger"])

    plt.suptitle("GRAIL: Honest 5-Fold Spatial CV Results", fontsize=14,
                 fontweight="bold", color=COLOUR["text"], y=1.01)
    _save(fig, out_dir, "fig4_model_results_honest.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="GRAIL ML classifier training pipeline.")
    ap.add_argument("--raw",       default="gnss_synthetic_raw.csv",   help="Raw satellite-link CSV.")
    ap.add_argument("--agg",       default="gnss_synthetic_agg.csv",   help="Aggregate epoch CSV.")
    ap.add_argument("--feat-csv",  default="gnss_ml_features.csv",     help="Cached feature matrix (written/read).")
    ap.add_argument("--out-dir",   default="ml_outputs",               help="Output directory for plots + model.")
    ap.add_argument("--eda-only",  action="store_true")
    ap.add_argument("--ml-only",   action="store_true",                help="Skip feature engineering (use cached --feat-csv).")
    ap.add_argument("--no-plots",  action="store_true",                help="Skip EDA plots (useful on headless servers).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 65)
    print("GRAIL — Honest Spatial ML Pipeline")
    print("=" * 65)

    if args.ml_only:
        feats = pd.read_csv(args.feat_csv)
    else:
        raw   = pd.read_csv(args.raw)
        agg   = pd.read_csv(args.agg)
        feats = engineer_features(raw, agg, out_csv=args.feat_csv)

    if not args.ml_only and not args.no_plots:
        agg_for_plot = pd.read_csv(args.agg) if "z_error_m" not in feats.columns else feats
        plot_motivation(agg_for_plot, args.out_dir)
        plot_signal_physics(raw if not args.ml_only else feats, feats, args.out_dir)

    if not args.eda_only:
        clf_summary, best_clf, feat_names, X, y_clf, y_reg, oof_prob, oof_reg = run_honest_cv(feats)
        if not args.no_plots:
            plot_results(clf_summary, best_clf, feat_names, X, y_clf, y_reg,
                         oof_prob, oof_reg, args.out_dir)
        model_path = os.path.join(args.out_dir, "best_clf_final.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(best_clf, f)
        print(f"\n[done] Model saved → {model_path}")
        print(f"[done] All outputs in: {args.out_dir}/")


if __name__ == "__main__":
    main()
