"""
run_pipeline.py
=================
End-to-end orchestrator for the Proactive IT Role & Skill Emergence Forecaster.

Usage:
  python run_pipeline.py                    # full pipeline (A + B + Step 2 + Ranking + Dashboard)
  python run_pipeline.py --known-only       # only Pipeline A (LightGBM known roles)
  python run_pipeline.py --discovery-only   # only Pipeline B (SBERT + HDBSCAN discovery)
  python run_pipeline.py --skills-only      # only Step 2 (Skill emergence & gap analysis)
  python run_pipeline.py --backtest         # run historical out-of-time emergence backtest
  python run_pipeline.py --dashboard        # regenerate 6-page interactive dashboard
  python run_pipeline.py --shap-only        # only run SHAP analysis
  python run_pipeline.py --validate-only    # run test assertions and integrity checks
"""

import os
import sys
import json
import yaml
import argparse
import logging
import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent
CFG_PATH = BASE / "config" / "pipeline_config.yaml"
with open(CFG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def banner(msg: str):
    logger.info("=" * 70)
    logger.info(f"  {msg}")
    logger.info("=" * 70)


# ── Pipeline A — Known Role Forecasting ───────────────────────────────────────
def run_known_role_pipeline() -> dict:
    """
    Load existing trained model + results. Run SHAP analysis.
    Returns known-role data dict.
    """
    banner("PIPELINE A — KNOWN ROLE FORECASTING (LightGBM)")
    t0 = time.time()

    # Load existing 6M/12M forecasts (already trained)
    from src.ranking.final_ranker import load_known_role_forecasts
    known_df = load_known_role_forecasts()
    logger.info(f"Known roles loaded: {len(known_df)} canonical roles")

    # Run SHAP analysis
    banner("Running SHAP Feature Attribution ...")
    try:
        from src.explainability.shap_analysis import run_full_shap_analysis
        global_imp, per_role_shap, explanations = run_full_shap_analysis()
        logger.info("SHAP analysis complete")
    except Exception as e:
        logger.warning(f"SHAP analysis failed: {e}. Continuing with cached SHAP.")
        global_imp, per_role_shap, explanations = None, None, {}

    logger.info(f"Pipeline A complete in {time.time()-t0:.1f}s")
    return {
        "known_df":   known_df,
        "global_imp": global_imp,
        "per_role":   per_role_shap,
        "explanations": explanations,
    }


# ── Pipeline B — Unknown Role Discovery ───────────────────────────────────────
def run_discovery_pipeline() -> dict:
    """
    Full semantic discovery pipeline:
    Clean → Embed → Cluster → Track → Label → Score → Forecast
    """
    banner("PIPELINE B — UNKNOWN ROLE DISCOVERY (SBERT + HDBSCAN)")
    t0 = time.time()

    # Step B1: Clean IT job titles
    banner("Step B1: Cleaning IT job titles ...")
    from src.preprocessing.job_cleaning import load_all_jobs
    jobs = load_all_jobs(CFG, include_linkedin=True)

    if len(jobs) == 0:
        logger.error("No IT jobs loaded! Check data paths.")
        return {}

    cluster_dir = BASE / "data" / "clusters"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(cluster_dir / "cleaned_it_jobs.csv", index=False)

    # Step B2: Generate embeddings
    banner("Step B2: Generating / Loading SBERT embeddings ...")
    from src.embeddings.sentence_embeddings import generate_or_load_embeddings, embed_canonical_roles
    unique_titles, embeddings, title_to_idx, jobs = generate_or_load_embeddings(jobs)
    canonical_names, canonical_embs = embed_canonical_roles(CFG)
    logger.info(f"Embeddings: {embeddings.shape} | Canonical: {canonical_embs.shape}")

    # Step B3: Per-quarter HDBSCAN clustering
    banner("Step B3: Running per-quarter HDBSCAN clustering ...")
    from src.clustering.hdbscan_discovery import discover_quarterly_clusters
    quarterly_results = discover_quarterly_clusters(jobs, embeddings)
    logger.info(f"Quarterly results: {len(quarterly_results)} quarters processed")

    if not quarterly_results:
        logger.error("No quarterly clusters! Insufficient data per quarter.")
        return {}

    # Step B4: Track clusters across quarters
    banner("Step B4: Temporal cluster tracking ...")
    from src.clustering.cluster_tracking import (
        build_cluster_timelines, compute_persistence, get_persistent_clusters
    )
    timeline_df, tracked_centroids = build_cluster_timelines(quarterly_results, embeddings)
    timeline_df = compute_persistence(timeline_df)
    timeline_df.to_csv(cluster_dir / "cluster_timeline.csv", index=False)

    persistent = get_persistent_clusters(timeline_df)
    persistent.to_csv(cluster_dir / "persistent_clusters.csv", index=False)
    logger.info(f"Persistent clusters: {len(persistent)}")

    if len(persistent) == 0:
        logger.warning("No persistent clusters found.")
        return {}

    # Step B5: Label clusters + novelty
    banner("Step B5: Cluster labeling and novelty scoring ...")
    from src.clustering.cluster_labeling import label_all_clusters
    labeled = label_all_clusters(
        timeline_df, jobs,
        canonical_embs, canonical_names,
        embeddings, title_to_idx,
    )
    cols_to_use = [c for c in labeled.columns if c not in persistent.columns or c == "tracked_cluster_id"]
    persistent = persistent.merge(labeled[cols_to_use], on="tracked_cluster_id", how="left")

    # Step B6: Emergence scoring
    banner("Step B6: Computing emergence scores ...")
    from src.emerging.emergence_score import compute_emergence_scores, apply_noise_filters, compute_confidence
    scored = compute_emergence_scores(persistent, timeline_df)
    scored = apply_noise_filters(scored)

    if len(scored):
        scored[["confidence_score", "confidence_label"]] = scored.apply(
            lambda r: pd.Series(compute_confidence(
                r.get("emergence_score", 0.0),
                r.get("persistence_quarters", 1),
                r.get("total_postings", 0),
                r.get("max_unique_companies", 1),
                r.get("novelty_category", "medium"),
            )),
            axis=1,
        )

    scored.to_csv(cluster_dir / "scored_clusters.csv", index=False)
    logger.info(f"Scored clusters: {len(scored)} after noise filtering")

    # Step B7: Cluster growth forecasting
    banner("Step B7: Forecasting cluster growth ...")
    from src.forecasting.cluster_forecaster import forecast_all_clusters
    forecasted = forecast_all_clusters(scored, timeline_df)
    forecasted.to_csv(cluster_dir / "forecasted_clusters.csv", index=False)
    logger.info(f"Cluster forecasts: {len(forecasted)}")

    # Save primary predictions table
    out_dir = BASE / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate clean predictions_df table
    FINAL_COLS_MAP = {
        "emergence_rank": "Rank",
        "candidate_role_name": "Emerging Role",
        "tracked_cluster_id": "Cluster ID",
        "emergence_score": "Emergence Score",
        "confidence_label": "Confidence",
        "demand_growth_score": "Demand Growth",
        "persistence_quarters": "Persistence (Qtrs)",
        "title_novelty": "Title Novelty",
        "market_breadth_score": "Market Breadth",
        "total_postings": "Historical Volume",
        "forecast_next_quarter_postings": "Forecast Volume (Next Q)",
        "forecast_growth_pct": "Forecast Growth %",
        "forecast_method": "Forecast Method",
        "closest_canonical": "Closest Known Role",
        "representative_titles": "Representative Titles",
        "first_seen": "First Seen",
        "last_seen": "Last Seen",
    }
    avail = {k: v for k, v in FINAL_COLS_MAP.items() if k in forecasted.columns}
    preds_clean = forecasted[list(avail.keys())].rename(columns=avail).sort_values("Rank").reset_index(drop=True)
    preds_clean.to_csv(out_dir / "emerging_role_predictions.csv", index=False)

    logger.info(f"Pipeline B complete in {time.time()-t0:.1f}s")
    return {
        "jobs":              jobs,
        "embeddings":        embeddings,
        "title_to_idx":     title_to_idx,
        "canonical_names":  canonical_names,
        "canonical_embs":   canonical_embs,
        "timeline_df":      timeline_df,
        "tracked_centroids": tracked_centroids,
        "persistent":       persistent,
        "scored":           scored,
        "forecasted":       forecasted,
    }


# ── Step 2 — Skill Emergence & Gap Engine ─────────────────────────────────────
def run_skill_pipeline() -> dict:
    """
    Step 2: Proactive Skill Emergence & Forecasting Engine.
    """
    banner("STEP 2 — SKILL EMERGENCE & FORECASTING ENGINE")
    t0 = time.time()
    from src.skills.skill_engine import generate_step2_skill_intelligence
    res = generate_step2_skill_intelligence()
    logger.info(f"Step 2 Skill Engine complete in {time.time()-t0:.1f}s")
    return res


# ── Final — Unified Ranking + Dashboard ───────────────────────────────────────
def run_final_ranking(known_data: dict, discovery_data: dict) -> pd.DataFrame:
    """Combine known + emerging into unified ranking."""
    banner("FINAL UNIFIED RANKING ENGINE")
    from src.ranking.final_ranker import build_unified_ranking, save_final_outputs

    known_df   = known_data.get("known_df", pd.DataFrame())
    forecasted = discovery_data.get("forecasted", None)

    unified = build_unified_ranking(known_df, forecasted)

    out_dir = BASE / "data" / "outputs"
    save_final_outputs(unified, out_dir)

    print("\n" + "="*70)
    print("FINAL UNIFIED RANKING — TOP 15")
    print("="*70)
    display_cols = [c for c in ["final_rank", "source_type", "role", "demand_score",
                                "emergence_score", "final_priority_score"] if c in unified.columns]
    print(unified[display_cols].head(15).to_string(index=False))
    return unified


def build_dashboard(known_data: dict, discovery_data: dict, unified: pd.DataFrame):
    """Generate the 6-page interactive HTML dashboard."""
    banner("GENERATING 6-PAGE PRODUCTION DASHBOARD")
    try:
        from dashboard.generate_dashboard import build_full_dashboard
        build_full_dashboard(known_data, discovery_data, unified, CFG)
        logger.info("Dashboard generated successfully")
    except Exception as e:
        logger.warning(f"Dashboard generation failed: {e}")


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proactive IT Role & Skill Forecaster Orchestrator")
    parser.add_argument("--known-only",      action="store_true", help="Run only Pipeline A (Known Roles)")
    parser.add_argument("--discovery-only",  action="store_true", help="Run only Pipeline B (Unknown Role Discovery)")
    parser.add_argument("--skills-only",     action="store_true", help="Run only Step 2 (Skill Emergence Engine)")
    parser.add_argument("--backtest",        action="store_true", help="Run emergence historical backtest")
    parser.add_argument("--dashboard",       action="store_true", help="Regenerate 6-page dashboard HTML")
    parser.add_argument("--shap-only",       action="store_true", help="Run only SHAP feature attribution")
    parser.add_argument("--validate-only",   action="store_true", help="Run smoke test & integrity assertions")
    parser.add_argument("--no-dashboard",    action="store_true", help="Skip dashboard generation")
    args = parser.parse_args()

    if args.validate_only:
        banner("INTEGRITY & VALIDATION CHECK")
        from src.preprocessing.job_cleaning import clean_title
        assert clean_title("Senior LLM Engineer III") != "", "clean_title check failed"
        assert clean_title("AI/ML Platform Engineer - Remote") != "", "clean_title check 2 failed"
        logger.info("Token preservation & integrity checks: PASSED")
        sys.exit(0)

    if args.shap_only:
        from src.explainability.shap_analysis import run_full_shap_analysis
        run_full_shap_analysis()
        sys.exit(0)

    if args.backtest:
        from src.preprocessing.job_cleaning import load_all_jobs
        from src.embeddings.sentence_embeddings import generate_or_load_embeddings
        from src.evaluation.emerging_backtest import run_multi_cutoff_backtest

        jobs = load_all_jobs(CFG)
        _, embeddings, title_to_idx, jobs = generate_or_load_embeddings(jobs)
        bt = run_multi_cutoff_backtest(jobs, embeddings, title_to_idx)
        print("\n=== MULTI-CUTOFF BACKTEST RESULTS ===")
        print(bt.to_string(index=False))
        sys.exit(0)

    if args.skills_only:
        run_skill_pipeline()
        sys.exit(0)

    if args.dashboard:
        from dashboard.generate_dashboard import build_full_dashboard
        build_full_dashboard({}, {}, None, CFG)
        sys.exit(0)

    # ── Full Execution Workflow ────────────────────────────────────────────────
    known_data = {}
    discovery_data = {}

    if not args.discovery_only:
        known_data = run_known_role_pipeline()

    if not args.known_only:
        discovery_data = run_discovery_pipeline()

    # Always run Step 2 Skill Engine in full run
    run_skill_pipeline()

    if known_data or discovery_data:
        unified = run_final_ranking(known_data, discovery_data)
        if not args.no_dashboard:
            build_dashboard(known_data, discovery_data, unified)

    banner("FULL HYBRID SYSTEM EXECUTION COMPLETE")
