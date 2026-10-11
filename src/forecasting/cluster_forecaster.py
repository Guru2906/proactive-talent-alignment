"""
src/forecasting/cluster_forecaster.py
=======================================
Forecasts future posting volume for discovered emerging clusters.

Two-tier approach:
  - Clusters with ≥3 quarterly observations → LightGBM regression
  - Clusters with 2 observations → Holt damped trend
  - Clusters with 1 observation → Cannot forecast (report as "insufficient data")

Target: predicted_postings_next_quarter
"""

import sys
import yaml
import logging
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
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

FC_CFG = CFG["cluster_forecaster"]


def holt_damped_forecast(series: list[float], horizon: int = 1) -> float:
    """
    Simple Holt damped-trend exponential smoothing for short series.
    Falls back to mean if series is too short.
    """
    if len(series) < 2:
        return float(np.mean(series)) if series else 0.0

    # Parameters
    alpha = 0.4   # level smoothing
    beta  = 0.3   # trend smoothing
    phi   = 0.9   # damping factor

    # Initialize
    L = series[0]
    T = series[1] - series[0]

    for val in series[1:]:
        L_prev, T_prev = L, T
        L = alpha * val + (1 - alpha) * (L_prev + phi * T_prev)
        T = beta * (L - L_prev) + (1 - beta) * phi * T_prev

    # Forecast h steps
    forecast = L
    T_h = T
    for h in range(1, horizon + 1):
        T_h = phi * T_h
        forecast = forecast + T_h

    return max(0.0, float(forecast))


def build_cluster_temporal_features(timeline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build temporal features for each (tracked_cluster_id, quarter) row.
    Features: lag1, lag2, roll2_mean, growth_1q, growth_2q, posting_count
    """
    rows = []
    for tcid, grp in timeline_df.groupby("tracked_cluster_id"):
        grp = grp.sort_values("quarter").reset_index(drop=True)

        for i in range(2, len(grp)):
            curr = grp.iloc[i]
            prev1 = grp.iloc[i - 1]
            prev2 = grp.iloc[i - 2]

            lag1 = float(prev1["posting_count"])
            lag2 = float(prev2["posting_count"])
            roll2 = (lag1 + lag2) / 2.0
            growth_1q = (lag1 - lag2) / (lag2 + 1e-6)
            growth_2q = (lag1 - float(grp.iloc[max(0, i-2)]["posting_count"])) / \
                        (float(grp.iloc[max(0, i-2)]["posting_count"]) + 1e-6)

            companies = float(prev1.get("unique_companies", 1))

            rows.append({
                "tracked_cluster_id": tcid,
                "quarter":            curr["quarter"],
                "lag1":               lag1,
                "lag2":               lag2,
                "roll2":              roll2,
                "growth_1q":          growth_1q,
                "growth_2q":          growth_2q,
                "companies_lag1":     companies,
                "target":             float(curr["posting_count"]),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def forecast_all_clusters(
    persistent_clusters: pd.DataFrame,
    timeline_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forecast next-quarter posting volume for all persistent clusters.
    
    Returns persistent_clusters enriched with:
      - forecast_next_quarter_postings
      - forecast_growth_pct
      - forecast_method (lgb / holt / mean)
    """
    # Sort timeline
    timeline_df = timeline_df.sort_values(["tracked_cluster_id", "quarter"])

    # Build features for LightGBM training
    feat_df = build_cluster_temporal_features(timeline_df)

    lgb_trained = False
    lgb_model = None

    if len(feat_df) >= 20:
        # Train LightGBM on all clusters with ≥3 observations
        FEATS = ["lag1", "lag2", "roll2", "growth_1q", "growth_2q", "companies_lag1"]
        X = feat_df[FEATS].values
        y = feat_df["target"].values

        lgb_model = lgb.LGBMRegressor(
            n_estimators=FC_CFG["n_estimators"],
            learning_rate=FC_CFG["learning_rate"],
            num_leaves=FC_CFG["num_leaves"],
            max_depth=FC_CFG["max_depth"],
            min_child_samples=FC_CFG["min_child_samples"],
            verbose=-1,
            random_state=CFG["random_seed"],
        )
        lgb_model.fit(X, y)
        lgb_trained = True
        logger.info(f"Cluster forecaster LightGBM trained on {len(feat_df)} samples")

    # Generate forecasts per cluster
    forecasts = []
    FEATS = ["lag1", "lag2", "roll2", "growth_1q", "growth_2q", "companies_lag1"]

    for _, row in persistent_clusters.iterrows():
        tcid = row["tracked_cluster_id"]
        ts = timeline_df[timeline_df["tracked_cluster_id"] == tcid].sort_values("quarter")
        posting_series = ts["posting_count"].tolist()
        companies_series = ts.get("unique_companies", pd.Series([1])).tolist()

        n_obs = len(posting_series)
        last_val = posting_series[-1] if posting_series else 0

        if n_obs >= 3 and lgb_trained:
            # LightGBM forecast
            lag1 = posting_series[-1]
            lag2 = posting_series[-2]
            roll2 = (lag1 + lag2) / 2.0
            growth_1q = (lag1 - lag2) / (lag2 + 1e-6)
            growth_2q = (lag1 - posting_series[-3]) / (posting_series[-3] + 1e-6) if n_obs >= 3 else growth_1q
            comp = float(companies_series[-1]) if companies_series else 1.0

            X_pred = np.array([[lag1, lag2, roll2, growth_1q, growth_2q, comp]])
            pred = float(lgb_model.predict(X_pred)[0])
            pred = max(0.0, pred)
            method = "lgb"
        elif n_obs >= 2:
            # Holt damped
            pred = holt_damped_forecast(posting_series)
            method = "holt"
        else:
            pred = float(np.mean(posting_series)) if posting_series else 0.0
            method = "mean"

        growth_pct = (pred - last_val) / (last_val + 1e-6) * 100

        forecasts.append({
            "tracked_cluster_id":              tcid,
            "forecast_next_quarter_postings":  round(pred, 1),
            "forecast_growth_pct":             round(growth_pct, 1),
            "forecast_method":                 method,
            "n_obs_quarters":                  n_obs,
        })

    forecast_df = pd.DataFrame(forecasts)
    result = persistent_clusters.merge(forecast_df, on="tracked_cluster_id", how="left")
    logger.info(f"Cluster forecasts generated: {len(result)} clusters")
    return result


if __name__ == "__main__":
    # Quick self-test
    series = [10, 15, 22, 31, 45, 62]
    pred = holt_damped_forecast(series)
    print(f"Holt test: series={series} → forecast={pred:.1f}")
