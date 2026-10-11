"""
src/ranking/final_ranker.py
=============================
Produces the unified final ranking combining:
  - Known roles (LightGBM forecasted demand)
  - Emerging clusters (emergence score + cluster forecasts)

Final Priority Score (configurable):
  Known role:   demand_score (normalized)
  Emerging:     emergence_score
  Combined:     0.60 * demand + 0.40 * emergence  (weights from config)
"""

import sys
import json
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "pipeline_config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

RANK_CFG = CFG["ranking"]


def load_known_role_forecasts() -> pd.DataFrame:
    """Load known-role 6M/12M forecasts from the existing pipeline."""
    path_6m = BASE / "step1_enhanced" / "outputs" / "top_emerging_roles_6m.csv"
    path_12m = BASE / "step1_enhanced" / "outputs" / "top_emerging_roles_12m.csv"

    df6  = pd.read_csv(path_6m)
    df12 = pd.read_csv(path_12m)

    # Standardize
    df6  = df6.rename(columns={"pred_share_bps_6m":  "pred_demand_6m",
                                 "growth_bps_6m":     "growth_bps",
                                 "growth_pct_6m":     "growth_pct",
                                 "emergence_tier_6m": "tier"})
    df12 = df12.rename(columns={"pred_share_bps_12m": "pred_demand_12m",
                                 "growth_bps_12m":    "growth_bps_12m",
                                 "growth_pct_12m":    "growth_pct_12m",
                                 "emergence_tier_12m":"tier_12m"})

    # Merge
    known = df6.merge(df12[["role", "pred_demand_12m", "growth_bps_12m", "growth_pct_12m"]],
                      on="role", how="left")
    known["source_type"] = "known_role"
    return known


def normalize_demand_score(known_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize predicted demand to [0, 1] for unified ranking."""
    col = "pred_demand_6m"
    if col not in known_df.columns:
        known_df["demand_score"] = 0.5
        return known_df
    mn, mx = known_df[col].min(), known_df[col].max()
    if mx - mn < 1e-9:
        known_df["demand_score"] = 0.5
    else:
        known_df["demand_score"] = (known_df[col] - mn) / (mx - mn)
    return known_df


def build_unified_ranking(
    known_df: pd.DataFrame,
    emerging_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Combine known roles and emerging clusters into one ranked table.
    """
    w_known   = RANK_CFG["known_role_weight"]
    w_emerging = RANK_CFG["emerging_role_weight"]

    # Known roles
    known_df = normalize_demand_score(known_df)
    known_df["final_priority_score"] = w_known * known_df["demand_score"]
    known_df["source_type"] = "known_role"

    if emerging_df is not None and len(emerging_df):
        # Normalize emergence score
        mn = emerging_df["emergence_score"].min()
        mx = emerging_df["emergence_score"].max()
        if mx - mn < 1e-9:
            emerging_df["emergence_score_norm"] = 0.5
        else:
            emerging_df["emergence_score_norm"] = (emerging_df["emergence_score"] - mn) / (mx - mn)

        emerging_df["demand_score"] = emerging_df.get("forecast_growth_pct", 0.0).clip(0, 200) / 200.0
        emerging_df["final_priority_score"] = (
            w_known    * emerging_df["demand_score"] +
            w_emerging * emerging_df["emergence_score_norm"]
        )
        emerging_df["source_type"] = "emerging_cluster"

        # Align columns for concat
        common_cols = [
            "source_type", "role", "demand_score",
            "final_priority_score", "emergence_score",
        ]
        known_out = known_df.copy()
        if "emergence_score" not in known_out.columns:
            known_out["emergence_score"] = 0.0
        if "role" not in known_out.columns and "canonical_role" in known_out.columns:
            known_out["role"] = known_out["canonical_role"]

        emerg_out = emerging_df.copy()
        if "role" not in emerg_out.columns and "candidate_role_name" in emerg_out.columns:
            emerg_out["role"] = emerg_out["candidate_role_name"]

        unified = pd.concat([known_out, emerg_out], ignore_index=True, sort=False)
    else:
        unified = known_df.copy()
        if "emergence_score" not in unified.columns:
            unified["emergence_score"] = 0.0

    unified = unified.sort_values("final_priority_score", ascending=False).reset_index(drop=True)
    unified["final_rank"] = unified.index + 1

    logger.info(f"Unified ranking: {len(unified)} entries "
                f"({(unified['source_type']=='known_role').sum()} known + "
                f"{(unified['source_type']=='emerging_cluster').sum()} emerging)")
    return unified


def format_known_role_output(row: pd.Series, explanations: dict = None) -> dict:
    """
    Format a known role row into the standard final output structure.
    """
    role = row.get("role", "Unknown")
    exp  = explanations.get(role, {}) if explanations else {}

    top_drivers = [d["description"] for d in exp.get("top_positive_drivers", [])[:5]]

    return {
        "type":          "KNOWN ROLE",
        "role":          role,
        "current_demand": row.get("baseline_share_bps", "N/A"),
        "predicted_6m":  row.get("pred_demand_6m", "N/A"),
        "growth_pct":    row.get("growth_pct", "N/A"),
        "tier":          row.get("tier", "N/A"),
        "demand_score":  row.get("demand_score", 0.0),
        "final_rank":    row.get("final_rank", 0),
        "top_drivers":   top_drivers,
    }


def format_emerging_role_output(row: pd.Series) -> dict:
    """Format an emerging cluster row into the standard final output structure."""
    return {
        "type":               "EMERGING ROLE",
        "role":               row.get("candidate_role_name", row.get("role", "Unknown")),
        "emergence_score":    row.get("emergence_score", 0.0),
        "forecast_growth_pct": row.get("forecast_growth_pct", 0.0),
        "persistence_quarters": row.get("persistence_quarters", 0),
        "title_novelty":      row.get("title_novelty", 0.5),
        "market_breadth_score": row.get("market_breadth_score", 0.0),
        "confidence":         row.get("confidence_label", "Medium"),
        "closest_canonical":  row.get("closest_canonical", "N/A"),
        "representative_titles": row.get("representative_titles", ""),
        "final_rank":         row.get("final_rank", 0),
    }


def save_final_outputs(unified_df: pd.DataFrame, out_dir: Path):
    """Save all final output files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    unified_df.to_csv(out_dir / "unified_ranking.csv", index=False)

    known = unified_df[unified_df["source_type"] == "known_role"]
    known.to_csv(out_dir / "known_role_forecasts.csv", index=False)

    emerging = unified_df[unified_df["source_type"] == "emerging_cluster"]
    if len(emerging):
        emerging.to_csv(out_dir / "emerging_role_discoveries.csv", index=False)

    logger.info(f"Final outputs saved to {out_dir}")


if __name__ == "__main__":
    known_df = load_known_role_forecasts()
    unified  = build_unified_ranking(known_df)

    out_dir = BASE / "data" / "outputs"
    save_final_outputs(unified, out_dir)

    print("\n=== FINAL RANKING (Top 20) ===")
    print(unified[["final_rank", "source_type", "role", "demand_score", "final_priority_score"]].head(20).to_string(index=False))
