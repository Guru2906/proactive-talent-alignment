"""
src/evaluation/emerging_backtest.py
=====================================
Historical backtesting for the emerging role discovery pipeline.

Methodology:
  1. Set cutoff = T_cutoff (e.g. 2024-Q3, 2024-Q4, 2025-Q1, 2025-Q2)
  2. Only use data available at or before T_cutoff (Zero Lookahead / Leakage)
  3. Run discovery pipeline (SBERT embeddings + HDBSCAN + temporal tracking)
  4. Identify clusters that scored as "emerging" at that historical cutoff
  5. Look forward: did those clusters grow? persist? gain market breadth?
  6. Compute: HitRate@K, Precision@K, Persistence Accuracy, NDCG@K, and Future Growth Rate

Ensures the emergence methodology has true out-of-time empirical predictive value.
"""

import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "pipeline_config.yaml"
with open(CFG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def quarter_to_date(q: str) -> pd.Timestamp:
    """Convert '2024-Q2' → 2024-06-30 (end of quarter)."""
    parts = q.strip().split("-Q")
    year, qtr = parts[0], parts[1]
    month_end = {"1": "03-31", "2": "06-30", "3": "09-30", "4": "12-31"}[qtr]
    return pd.Timestamp(f"{year}-{month_end}")


def backtest_at_cutoff(
    jobs_df: pd.DataFrame,
    embeddings: np.ndarray,
    title_to_idx: dict,
    cutoff_quarter: str,
    lookahead_quarters: int = 2,
    top_k: int = 5,
) -> Dict:
    """
    Run emergence discovery at a historical cutoff and evaluate against future data.
    """
    from src.clustering.hdbscan_discovery import discover_quarterly_clusters
    from src.clustering.cluster_tracking import build_cluster_timelines, compute_persistence, get_persistent_clusters
    from src.emerging.emergence_score import compute_emergence_scores, apply_noise_filters

    cutoff_date = quarter_to_date(cutoff_quarter)
    logger.info(f"\n{'='*60}")
    logger.info(f"BACKTEST CUTOFF: {cutoff_quarter} (<= {cutoff_date.strftime('%Y-%m-%d')})")
    logger.info(f"{'='*60}")

    if "quarter" not in jobs_df.columns:
        logger.error("jobs_df missing 'quarter' column")
        return {}

    all_quarters = sorted(jobs_df["quarter"].dropna().unique())
    hist_quarters = [q for q in all_quarters if quarter_to_date(q) <= cutoff_date]
    future_quarters = [
        q for q in all_quarters
        if quarter_to_date(q) > cutoff_date and quarter_to_date(q) <= cutoff_date + pd.DateOffset(months=lookahead_quarters * 3 + 15)
    ]

    if len(hist_quarters) < 2:
        logger.warning(f"Insufficient historical quarters before {cutoff_quarter} ({len(hist_quarters)})")
        return {}
    if not future_quarters:
        logger.warning(f"No future data after {cutoff_quarter}")
        return {}

    logger.info(f"Historical quarters: {hist_quarters} | Future quarters: {future_quarters}")

    # Historical jobs only (zero leakage)
    hist_jobs = jobs_df[jobs_df["quarter"].isin(hist_quarters)].copy()
    hist_quarterly = discover_quarterly_clusters(hist_jobs, embeddings)

    if not hist_quarterly:
        logger.warning(f"No quarterly clusters found for cutoff {cutoff_quarter}")
        return {}

    timeline_df, tracked_centroids = build_cluster_timelines(hist_quarterly, embeddings)
    timeline_df = compute_persistence(timeline_df)
    persistent = get_persistent_clusters(timeline_df, min_quarters=2)

    if persistent.empty:
        logger.info(f"No persistent clusters at cutoff {cutoff_quarter}")
        return {
            "cutoff_quarter": cutoff_quarter,
            "n_hist_quarters": len(hist_quarters),
            "n_future_quarters": len(future_quarters),
            "n_persistent_clusters": 0,
            "top_k": top_k,
            "hits": 0,
            "hit_rate_at_k": 0.0,
            "persistence_accuracy": 0.0,
            "avg_future_growth": 0.0,
            "ndcg_at_k": 0.0,
        }

    scored = compute_emergence_scores(persistent, timeline_df)
    scored = apply_noise_filters(scored)

    if scored.empty:
        return {}

    # Evaluate against future data
    future_jobs = jobs_df[jobs_df["quarter"].isin(future_quarters)].copy()
    future_quarterly = discover_quarterly_clusters(future_jobs, embeddings)
    future_timeline, _ = build_cluster_timelines(future_quarterly, embeddings)

    # Calculate Top-K emergence predictions
    top_k_df = scored.head(top_k)
    hits = 0
    persisted_in_future = 0
    growth_rates = []
    relevance_scores = []

    for _, row in top_k_df.iterrows():
        cid = row.get("tracked_cluster_id")
        # Check if this cluster's centroid or titles appeared in future_timeline
        matched_future = future_timeline[future_timeline["tracked_cluster_id"] == cid]
        if not matched_future.empty:
            future_posts = matched_future["posting_count"].sum()
            hist_posts = row.get("total_postings", 1)
            growth = (future_posts - (hist_posts / max(1, len(hist_quarters)))) / max(1, (hist_posts / max(1, len(hist_quarters))))
            growth_rates.append(growth)
            persisted_in_future += 1
            if growth >= 0:
                hits += 1
                relevance_scores.append(1.0)
            else:
                relevance_scores.append(0.5)
        else:
            relevance_scores.append(0.0)

    k_actual = len(top_k_df)
    hit_rate = hits / max(1, k_actual)
    persistence_acc = persisted_in_future / max(1, k_actual)
    avg_growth = float(np.mean(growth_rates)) if growth_rates else 0.0

    # Calculate NDCG@K
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(k_actual))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    result = {
        "cutoff_quarter": cutoff_quarter,
        "n_hist_quarters": len(hist_quarters),
        "n_future_quarters": len(future_quarters),
        "n_persistent_clusters": len(persistent),
        "top_k": k_actual,
        "hits": hits,
        "hit_rate_at_k": round(hit_rate, 4),
        "persistence_accuracy": round(persistence_acc, 4),
        "avg_future_growth": round(avg_growth, 4),
        "ndcg_at_k": round(ndcg, 4),
    }

    logger.info(f"Backtest result: Hit@{k_actual}={hit_rate:.2%} | PersistAcc={persistence_acc:.2%} | NDCG@{k_actual}={ndcg:.3f}")
    return result


def run_multi_cutoff_backtest(
    jobs_df: Optional[pd.DataFrame] = None,
    embeddings: Optional[np.ndarray] = None,
    title_to_idx: Optional[dict] = None,
    cutoffs: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Runs multi-cutoff historical backtesting across valid chronological cutoffs.
    """
    if jobs_df is None:
        from src.preprocessing.job_cleaning import load_all_jobs
        jobs_df = load_all_jobs(CFG)

    if embeddings is None or title_to_idx is None:
        from src.embeddings.sentence_embeddings import generate_or_load_embeddings
        unique_titles, embeddings, title_to_idx, jobs_df = generate_or_load_embeddings(jobs_df)

    if cutoffs is None:
        # Valid cutoffs where at least 2 historical quarters exist and future quarters follow
        cutoffs = ["2024-Q3", "2024-Q4", "2025-Q1", "2025-Q2"]

    results = []
    for cutoff in cutoffs:
        try:
            res = backtest_at_cutoff(jobs_df, embeddings, title_to_idx, cutoff)
            if res:
                results.append(res)
        except Exception as e:
            logger.error(f"Backtest failed at {cutoff}: {e}")

    backtest_df = pd.DataFrame(results)
    out_path = BASE / "data" / "outputs" / "emerging_backtest_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(out_path, index=False)
    logger.info(f"Saved emerging role backtest results to {out_path}")
    return backtest_df


if __name__ == "__main__":
    df = run_multi_cutoff_backtest()
    print("\n=== EMERGING ROLE MULTI-CUTOFF BACKTEST RESULTS ===")
    print(df.to_string(index=False))
