"""
src/clustering/cluster_tracking.py
=====================================
Tracks HDBSCAN clusters across quarterly time windows.

Key principle: A single quarter's cluster is NOT evidence of emergence.
Persistence across multiple quarters is required.

Algorithm:
  1. Run HDBSCAN per quarter (from hdbscan_discovery.py)
  2. Compute centroids per (quarter, cluster)
  3. Match clusters across adjacent quarters via cosine similarity
  4. Build cluster identity timeline
  5. Output: cluster_timeline.csv with persistence and growth metrics
"""

import sys
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "pipeline_config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

CT_CFG = CFG["cluster_tracking"]
SIM_THRESH = CT_CFG["centroid_similarity_threshold"]
MIN_PERSIST = CT_CFG["min_persistence_quarters"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two unit-norm vectors."""
    return float(np.dot(a, b))


def match_clusters_across_quarters(
    prev_centroids: dict,    # {local_cid -> centroid}
    curr_centroids: dict,    # {local_cid -> centroid}
    similarity_threshold: float = SIM_THRESH,
) -> dict:
    """
    Greedily match current-quarter clusters to previous-quarter clusters.
    Returns: {curr_local_cid -> prev_local_cid} (or None if no match)
    """
    matches = {}
    prev_ids   = list(prev_centroids.keys())
    prev_vecs  = np.array([prev_centroids[c] for c in prev_ids])
    used_prev  = set()

    for curr_cid, curr_vec in curr_centroids.items():
        if len(prev_ids) == 0:
            matches[curr_cid] = None
            continue
        sims = prev_vecs @ curr_vec   # dot product = cosine sim (normalized)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_prev = prev_ids[best_idx]

        if best_sim >= similarity_threshold and best_prev not in used_prev:
            matches[curr_cid] = best_prev
            used_prev.add(best_prev)
        else:
            matches[curr_cid] = None

    return matches


def build_cluster_timelines(
    quarterly_results: dict,    # {quarter -> (jobs_df, clusterer, embeddings)}
    embeddings_full: np.ndarray,
) -> pd.DataFrame:
    """
    Build a timeline of tracked cluster identities across quarters.
    
    Returns DataFrame with one row per (tracked_cluster_id, quarter) with:
      - tracked_cluster_id: persistent identity
      - quarter: time period
      - posting_count: jobs in this cluster this quarter
      - unique_companies: company breadth
      - centroid: stored separately
      - top_titles: most common titles
    """
    from src.clustering.hdbscan_discovery import compute_cluster_centroids

    quarters = sorted(quarterly_results.keys())
    logger.info(f"Building cluster timelines across {len(quarters)} quarters")

    # tracked_cluster_id counter
    next_tcid = 0
    # Map: (quarter, local_cid) -> tracked_cluster_id
    assignment_map = {}
    # Centroid store: {tracked_cluster_id -> latest_centroid}
    tracked_centroids = {}
    # Timeline records
    timeline_rows = []

    prev_quarter    = None
    prev_centroids  = {}    # {local_cid -> centroid}
    prev_local_to_tracked = {}  # {local_cid -> tracked_cluster_id}

    for quarter in quarters:
        jobs_q, clusterer_q, embs_q = quarterly_results[quarter]

        # Compute cluster centroids for this quarter using full embedding space
        curr_centroids = compute_cluster_centroids(jobs_q, embeddings_full)

        if not curr_centroids:
            prev_quarter = quarter
            prev_centroids = curr_centroids
            prev_local_to_tracked = {}
            continue

        # Match to previous quarter
        if prev_centroids:
            matches = match_clusters_across_quarters(prev_centroids, curr_centroids)
        else:
            matches = {cid: None for cid in curr_centroids}

        curr_local_to_tracked = {}
        for curr_cid, prev_cid in matches.items():
            if prev_cid is not None and prev_cid in prev_local_to_tracked:
                # Matched: same tracked cluster continues
                tcid = prev_local_to_tracked[prev_cid]
            else:
                # New cluster identity
                tcid = next_tcid
                next_tcid += 1

            curr_local_to_tracked[curr_cid] = tcid
            tracked_centroids[tcid] = curr_centroids[curr_cid]
            assignment_map[(quarter, curr_cid)] = tcid

        # Build timeline rows for this quarter
        clustered_q = jobs_q[jobs_q["cluster_id"] >= 0]
        for local_cid, grp in clustered_q.groupby("cluster_id"):
            tcid = curr_local_to_tracked.get(int(local_cid), None)
            if tcid is None:
                continue

            top_titles = (
                grp["clean_title"].value_counts().head(10).index.tolist()
                if "clean_title" in grp.columns else []
            )
            companies = (
                grp["company"].nunique()
                if "company" in grp.columns else 0
            )

            timeline_rows.append({
                "quarter":             quarter,
                "tracked_cluster_id":  tcid,
                "local_cluster_id":    int(local_cid),
                "posting_count":       len(grp),
                "unique_companies":    companies,
                "top_titles":          "|".join(top_titles[:5]),
                "mean_prob":           float(grp["cluster_prob"].mean()) if "cluster_prob" in grp.columns else 0.0,
            })

        prev_quarter          = quarter
        prev_centroids        = curr_centroids
        prev_local_to_tracked = curr_local_to_tracked

    timeline_df = pd.DataFrame(timeline_rows)
    logger.info(f"Timeline built: {len(timeline_df)} rows | {timeline_df['tracked_cluster_id'].nunique()} tracked clusters")
    return timeline_df, tracked_centroids


def compute_persistence(timeline_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add persistence statistics to timeline.
    persistence_quarters = number of distinct quarters the cluster appeared in.
    """
    if len(timeline_df) == 0:
        return timeline_df

    persist = (
        timeline_df.groupby("tracked_cluster_id")["quarter"]
        .nunique()
        .reset_index()
        .rename(columns={"quarter": "persistence_quarters"})
    )
    first_seen = (
        timeline_df.groupby("tracked_cluster_id")["quarter"]
        .min()
        .reset_index()
        .rename(columns={"quarter": "first_seen"})
    )
    last_seen = (
        timeline_df.groupby("tracked_cluster_id")["quarter"]
        .max()
        .reset_index()
        .rename(columns={"quarter": "last_seen"})
    )

    timeline_df = timeline_df.merge(persist, on="tracked_cluster_id", how="left")
    timeline_df = timeline_df.merge(first_seen, on="tracked_cluster_id", how="left")
    timeline_df = timeline_df.merge(last_seen, on="tracked_cluster_id", how="left")
    return timeline_df


def get_persistent_clusters(
    timeline_df: pd.DataFrame,
    min_quarters: int = MIN_PERSIST,
    min_postings: int = None,
) -> pd.DataFrame:
    """
    Filter to only clusters that meet minimum persistence criteria.
    Returns cluster summary with one row per tracked_cluster_id.
    """
    min_posts = min_postings or CT_CFG.get("min_cluster_postings", 20)

    # Aggregate summary per tracked cluster
    agg = timeline_df.groupby("tracked_cluster_id").agg(
        persistence_quarters=("quarter", "nunique"),
        total_postings=("posting_count", "sum"),
        max_quarterly_postings=("posting_count", "max"),
        avg_quarterly_postings=("posting_count", "mean"),
        max_unique_companies=("unique_companies", "max"),
        first_seen=("quarter", "min"),
        last_seen=("quarter", "max"),
        top_titles=("top_titles", lambda x: "|".join(x.dropna().unique()[:3])),
    ).reset_index()

    persistent = agg[
        (agg["persistence_quarters"] >= min_quarters) &
        (agg["total_postings"] >= min_posts)
    ].copy()

    logger.info(f"Persistent clusters (≥{min_quarters}Q, ≥{min_posts} posts): "
                f"{len(persistent)} / {len(agg)} total")
    return persistent


if __name__ == "__main__":
    import pickle
    from src.preprocessing.job_cleaning import load_all_jobs
    from src.embeddings.sentence_embeddings import generate_or_load_embeddings
    from src.clustering.hdbscan_discovery import discover_quarterly_clusters

    jobs = load_all_jobs(CFG)
    unique_titles, embeddings, title_to_idx, jobs = generate_or_load_embeddings(jobs)
    quarterly_results = discover_quarterly_clusters(jobs, embeddings)

    timeline_df, tracked_centroids = build_cluster_timelines(quarterly_results, embeddings)
    timeline_df = compute_persistence(timeline_df)

    out_dir = BASE / "data" / "clusters"
    out_dir.mkdir(parents=True, exist_ok=True)
    timeline_df.to_csv(out_dir / "cluster_timeline.csv", index=False)

    persistent = get_persistent_clusters(timeline_df)
    persistent.to_csv(out_dir / "persistent_clusters.csv", index=False)

    print(f"\nTop 20 most persistent clusters:")
    print(persistent.sort_values("persistence_quarters", ascending=False).head(20).to_string(index=False))
