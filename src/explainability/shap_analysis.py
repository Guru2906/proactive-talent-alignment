"""
src/explainability/shap_analysis.py
======================================
SHAP explanations for the known-role LightGBM forecasting model.

Generates:
  - Global feature importance via SHAP values
  - Per-role top-5 positive/negative drivers
  - SHAP waterfall data for dashboard
"""

import sys
import json
import yaml
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "pipeline_config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)


def load_known_model_and_data():
    """Load the trained LightGBM model and master feature matrix."""
    model_path = BASE / CFG["paths"]["known_model"]
    feat_path  = BASE / CFG["paths"]["master_features"]
    cfg_path   = BASE / CFG["paths"]["known_model_config"]

    model = joblib.load(model_path)
    df    = pd.read_csv(feat_path)

    with open(cfg_path) as f:
        model_cfg = json.load(f)

    features = [c for c in model_cfg["features"] if c in df.columns]
    target   = model_cfg["target"]

    logger.info(f"Loaded model: {model_path.name} | features: {len(features)}")
    return model, df, features, target, model_cfg


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """
    Compute SHAP TreeExplainer values.
    Returns shap_values array (N, F).
    """
    logger.info(f"Computing SHAP values for {len(X)} samples ...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)
    logger.info(f"SHAP values shape: {np.array(shap_vals).shape}")
    return shap_vals, explainer


def get_global_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Global SHAP feature importance (mean |SHAP|).
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({
        "feature":          feature_names,
        "mean_abs_shap":    np.round(mean_abs, 6),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df["importance_rank"] = imp_df.index + 1
    imp_df["pct_importance"] = np.round(imp_df["mean_abs_shap"] / imp_df["mean_abs_shap"].sum() * 100, 2)
    return imp_df


def get_per_role_drivers(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list,
    roles: pd.Series,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    For each unique role: compute average SHAP per feature → top positive/negative drivers.
    Returns: DataFrame with columns [role, driver_rank, feature, shap_direction, shap_value, description]
    """
    rows = []
    unique_roles = roles.unique()

    for role in unique_roles:
        mask = roles == role
        if mask.sum() == 0:
            continue
        role_shap = shap_values[mask].mean(axis=0)

        # Top positive drivers (increase prediction)
        pos_idx = np.argsort(-role_shap)[:top_n]
        for rank, idx in enumerate(pos_idx):
            rows.append({
                "role":          role,
                "driver_rank":   rank + 1,
                "feature":       feature_names[idx],
                "shap_value":    round(float(role_shap[idx]), 6),
                "shap_direction": "positive",
                "avg_feature_val": round(float(X.iloc[mask.values, idx].mean()), 4),
            })

        # Top negative drivers (decrease prediction)
        neg_idx = np.argsort(role_shap)[:top_n]
        for rank, idx in enumerate(neg_idx):
            rows.append({
                "role":          role,
                "driver_rank":   rank + 1,
                "feature":       feature_names[idx],
                "shap_value":    round(float(role_shap[idx]), 6),
                "shap_direction": "negative",
                "avg_feature_val": round(float(X.iloc[mask.values, idx].mean()), 4),
            })

    return pd.DataFrame(rows)


# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    "sk_vel":                "Skill velocity — how fast new skills are being added",
    "lag1":                  "Job posting share 1 month ago",
    "lag2":                  "Job posting share 2 months ago",
    "lag3":                  "Job posting share 3 months ago",
    "lag6":                  "Job posting share 6 months ago",
    "roll6m":                "6-month rolling average share",
    "roll3m":                "3-month rolling average share",
    "slope6m":               "6-month linear trend slope",
    "accel":                 "Trend acceleration (is momentum increasing?)",
    "mom1":                  "1-month momentum",
    "mom3":                  "3-month momentum",
    "mom6":                  "6-month momentum",
    "li_share":              "LinkedIn market share for this role",
    "li_unique_companies":   "Number of companies hiring on LinkedIn",
    "li_frontier_skill_ratio": "Fraction of skills that are frontier/emerging",
    "onet_skill_count":      "Number of skills in O*NET for this occupation",
    "onet_matched":          "Role matched to an O*NET occupation",
    "az_months_active":      "Total months role appeared in Adzuna",
    "li_seniority_entropy":  "Diversity of seniority levels (career maturity signal)",
    "std3m":                 "3-month posting volatility",
}


def generate_role_explanation(
    role: str,
    per_role_shap: pd.DataFrame,
    known_forecast: pd.DataFrame = None,
) -> dict:
    """
    Generate a structured explanation for a known role.
    Returns a dict suitable for the dashboard.
    """
    role_drivers = per_role_shap[per_role_shap["role"] == role]
    top_positive = role_drivers[role_drivers["shap_direction"] == "positive"].head(5)
    top_negative = role_drivers[role_drivers["shap_direction"] == "negative"].head(5)

    explanation = {
        "role": role,
        "top_positive_drivers": [],
        "top_negative_drivers": [],
    }

    for _, r in top_positive.iterrows():
        feat = r["feature"]
        explanation["top_positive_drivers"].append({
            "feature":     feat,
            "shap_value":  r["shap_value"],
            "description": FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ").title()),
            "direction":   "↑ Increases predicted demand",
        })

    for _, r in top_negative.iterrows():
        feat = r["feature"]
        explanation["top_negative_drivers"].append({
            "feature":     feat,
            "shap_value":  r["shap_value"],
            "description": FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ").title()),
            "direction":   "↓ Decreases predicted demand",
        })

    return explanation


def run_full_shap_analysis():
    """
    Full SHAP pipeline: load model → compute values → save outputs.
    """
    model, df, features, target, model_cfg = load_known_model_and_data()

    valid = df[df[target].notna()].copy()
    X = valid[features]
    roles = valid["role"]

    shap_vals, explainer = compute_shap_values(model, X)

    # Global importance
    global_imp = get_global_importance(shap_vals, features)
    out_dir = BASE / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    global_imp.to_csv(out_dir / "shap_global_importance.csv", index=False)
    logger.info(f"Saved global SHAP importance: {len(global_imp)} features")

    # Per-role drivers
    per_role = get_per_role_drivers(shap_vals, X, features, roles)
    per_role.to_csv(out_dir / "shap_per_role_drivers.csv", index=False)
    logger.info(f"Saved per-role SHAP drivers: {len(per_role)} rows")

    # Per-role explanations (JSON)
    explanations = {}
    for role in roles.unique():
        explanations[role] = generate_role_explanation(role, per_role)

    with open(out_dir / "role_explanations.json", "w", encoding="utf-8") as f:
        json.dump(explanations, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved role explanations: {len(explanations)} roles")

    print("\n=== TOP 10 GLOBAL SHAP FEATURES ===")
    print(global_imp.head(10).to_string(index=False))

    return global_imp, per_role, explanations


if __name__ == "__main__":
    run_full_shap_analysis()
