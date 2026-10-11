"""
src/emerging/emergence_score.py
==================================
Transparent, configurable emergence score for discovered job-role clusters.

Formula (weights from config):
  score = w_growth    * demand_growth_score
        + w_skill_vel * skill_velocity_score
        + w_skill_nov * skill_novelty_score
        + w_title_nov * title_novelty_score
        + w_breadth   * market_breadth_score
        + w_persist   * persistence_score
        + w_tech      * technology_adoption_score

All components are normalized to [0, 1].
Confidence is computed SEPARATELY from emergence score.
"""

import sys
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

W = CFG["emergence_weights"]
PERSIST_MAP = CFG["persistence"]["score_map"]
NOISE_CFG   = CFG["noise_filters"]


# ── Component scorers ─────────────────────────────────────────────────────────

def score_demand_growth(
    posting_counts: pd.Series,
    window_size: int = 2,
) -> float:
    """
    Score based on posting growth trend.
    Uses last vs first window growth, clipped to [0, 1].
    """
    vals = posting_counts.values
    if len(vals) < 2:
        return 0.1
    # Compare recent average vs early average
    mid = max(1, len(vals) // 2)
    early = vals[:mid].mean() + 1e-6
    recent = vals[mid:].mean() + 1e-6
    growth_ratio = (recent - early) / early
    # Scale: 0% growth → 0.0, 100% growth → 0.7, 200%+ → 1.0
    score = min(1.0, max(0.0, growth_ratio / 2.0))
    return round(float(score), 4)


def score_persistence(persistence_quarters: int) -> float:
    """Map persistence_quarters to [0, 1] using config table."""
    max_key = max(int(k) for k in PERSIST_MAP.keys())
    key = min(int(persistence_quarters), max_key)
    # Find closest key
    keys = sorted(int(k) for k in PERSIST_MAP.keys())
    for k in reversed(keys):
        if key >= k:
            return float(PERSIST_MAP[k])
    return 0.05


def score_market_breadth(unique_companies: int, unique_industries: int = 0) -> float:
    """
    Score based on how broadly distributed demand is.
    1 company → very low. 10+ companies → high. 
    """
    comp_score = min(1.0, np.log1p(unique_companies) / np.log1p(30))
    ind_score  = min(1.0, np.log1p(unique_industries) / np.log1p(10))
    # Weighted: companies matter more than industries (industries often missing)
    if unique_industries > 0:
        return round(float(0.7 * comp_score + 0.3 * ind_score), 4)
    else:
        return round(float(comp_score), 4)


def score_skill_velocity(skill_count_trend: list[int]) -> float:
    """
    Score how rapidly new skills are being added.
    skill_count_trend: list of skill counts per quarter.
    """
    if len(skill_count_trend) < 2:
        return 0.3
    x = np.array(skill_count_trend, dtype=float)
    growth = (x[-1] - x[0]) / (x[0] + 1e-6)
    return round(float(min(1.0, max(0.0, growth))), 4)


def normalize_series(s: pd.Series) -> pd.Series:
    """Min-max normalize a pandas series to [0, 1]."""
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def compute_emergence_scores(clusters_df: pd.DataFrame, timeline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function: compute emergence score for all persistent clusters.

    clusters_df must have columns:
      - tracked_cluster_id
      - persistence_quarters
      - total_postings
      - max_unique_companies
      - title_novelty (from cluster_labeling)
      - novelty_category (from cluster_labeling)

    timeline_df must have:
      - tracked_cluster_id
      - quarter
      - posting_count
      - unique_companies

    Returns enriched clusters_df with emergence_score and components.
    """
    logger.info(f"Computing emergence scores for {len(clusters_df)} clusters ...")

    results = []
    for _, row in clusters_df.iterrows():
        tcid = row["tracked_cluster_id"]

        # Get time series for this cluster
        ts = timeline_df[timeline_df["tracked_cluster_id"] == tcid].sort_values("quarter")

        # ── Demand Growth ──
        if len(ts) >= 2:
            demand_growth_score = score_demand_growth(ts["posting_count"])
        else:
            demand_growth_score = 0.1

        # ── Persistence ──
        persist_score = score_persistence(row.get("persistence_quarters", 1))

        # ── Market Breadth ──
        max_companies = row.get("max_unique_companies", 1)
        breadth_score = score_market_breadth(max_companies)

        # ── Title Novelty ──
        title_novelty = row.get("title_novelty", 0.5)

        # ── Skill Velocity (placeholder — needs skill tracking) ──
        # If skill data available in row, use it; otherwise default to 0.3
        skill_vel_score  = row.get("skill_velocity_score", 0.3)
        skill_nov_score  = row.get("skill_novelty_score", title_novelty * 0.8)

        # ── Technology Adoption ──
        tech_score = row.get("technology_adoption_score", demand_growth_score * 0.8)

        # ── Weighted Emergence Score ──
        emergence_score = (
            W["demand_growth"]    * demand_growth_score +
            W["skill_velocity"]   * skill_vel_score +
            W["skill_novelty"]    * skill_nov_score +
            W["title_novelty"]    * title_novelty +
            W["market_breadth"]   * breadth_score +
            W["persistence"]      * persist_score +
            W["technology_adoption"] * tech_score
        )

        results.append({
            "tracked_cluster_id":       tcid,
            "demand_growth_score":      round(demand_growth_score, 4),
            "skill_velocity_score":     round(skill_vel_score, 4),
            "skill_novelty_score":      round(skill_nov_score, 4),
            "title_novelty_score":      round(title_novelty, 4),
            "market_breadth_score":     round(breadth_score, 4),
            "persistence_score":        round(persist_score, 4),
            "technology_adoption_score": round(tech_score, 4),
            "emergence_score":          round(float(emergence_score), 4),
        })

    scores_df = pd.DataFrame(results)

    # Merge back
    out = clusters_df.merge(scores_df, on="tracked_cluster_id", how="left")

    # Rank
    out = out.sort_values("emergence_score", ascending=False).reset_index(drop=True)
    out["emergence_rank"] = out.index + 1

    logger.info(f"Emergence scores computed. Top score: {out['emergence_score'].iloc[0]:.3f}")
    return out


def compute_confidence(
    emergence_score: float,
    persistence_quarters: int,
    total_postings: int,
    max_unique_companies: int,
    novelty_category: str,
) -> tuple[float, str]:
    """
    Compute confidence SEPARATELY from emergence score.

    Confidence reflects: data volume, persistence, breadth, signal agreement.
    Returns: (confidence_score [0,1], confidence_label)
    """
    # Data volume component
    vol_score = min(1.0, np.log1p(total_postings) / np.log1p(500))

    # Persistence component
    persist_score = min(1.0, persistence_quarters / 6.0)

    # Breadth component
    breadth_score = min(1.0, np.log1p(max_unique_companies) / np.log1p(20))

    # Signal agreement: high emergence + high novelty = more confident
    if novelty_category == "high" and emergence_score >= 0.6:
        agreement = 1.0
    elif novelty_category == "medium":
        agreement = 0.7
    else:
        agreement = 0.4  # low novelty = probably evolution of existing role

    confidence = (
        0.30 * vol_score +
        0.30 * persist_score +
        0.20 * breadth_score +
        0.20 * agreement
    )

    if confidence >= CFG["ranking"]["high_confidence_threshold"]:
        label = "High"
    elif confidence >= CFG["ranking"]["medium_confidence_threshold"]:
        label = "Medium"
    else:
        label = "Low"

    return round(float(confidence), 4), label


def apply_noise_filters(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove clusters that fail minimum quality criteria.
    Does NOT remove small-but-growing roles — uses relative criteria.
    """
    n_before = len(clusters_df)

    filtered = clusters_df[
        (clusters_df["persistence_quarters"] >= NOISE_CFG["min_persistence_quarters"]) &
        (clusters_df["max_unique_companies"] >= NOISE_CFG["min_unique_companies"]) &
        (clusters_df["total_postings"] >= NOISE_CFG["min_absolute_postings_per_quarter"])
    ].copy()

    logger.info(f"Noise filter: {n_before} → {len(filtered)} clusters")
    return filtered
