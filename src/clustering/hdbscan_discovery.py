"""
src/clustering/hdbscan_discovery.py
=====================================
HDBSCAN-based semantic clustering of job title embeddings.

Problem B: Unknown Role Discovery
  - Uses HDBSCAN (not K-Means) because number of clusters is unknown
  - Separate full-data clustering and per-quarter clustering for temporal tracking
  - Stores cluster assignments, probabilities, and outlier scores
"""

import sys
import yaml
import logging
import numpy as np
import pandas as pd
import hdbscan
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "pipeline_config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

H_CFG = CFG["hdbscan"]


def run_hdbscan(
    embeddings: np.ndarray,
    min_cluster_size: int = None,
    min_samples: int = None,
    metric: str = None,
) -> hdbscan.HDBSCAN:
    """
    Run HDBSCAN on embedding matrix.
    Returns fitted HDBSCAN object.
    """
    mcs = min_cluster_size or H_CFG["min_cluster_size"]
    ms  = min_samples      or H_CFG["min_samples"]
    met = metric           or H_CFG["metric"]

    logger.info(f"Running HDBSCAN: n={len(embeddings):,} | "
                f"min_cluster_size={mcs} | min_samples={ms} | metric={met}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        metric=met,
        cluster_selection_method=H_CFG.get("cluster_selection_method", "eom"),
        prediction_data=True,
    )
    clusterer.fit(embeddings)

    n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    n_noise    = np.sum(clusterer.labels_ == -1)
    logger.info(f"HDBSCAN complete: {n_clusters} clusters | {n_noise:,} noise points "
                f"({n_noise/len(embeddings)*100:.1f}%)")
    return clusterer


def assign_clusters(
    jobs_df: pd.DataFrame,
    embeddings: np.ndarray,
    clusterer: hdbscan.HDBSCAN,
) -> pd.DataFrame:
    """
    Add cluster assignments to jobs_df.
    jobs_df must have 'emb_idx' column (from generate_or_load_embeddings).
    """
    jobs = jobs_df.copy()
    labels = clusterer.labels_
    probs  = clusterer.probabilities_
    outlier_scores = clusterer.outlier_scores_

    jobs["cluster_id"] = jobs["emb_idx"].apply(
        lambda idx: int(labels[idx]) if idx >= 0 else -1
    )
    jobs["cluster_prob"] = jobs["emb_idx"].apply(
        lambda idx: float(probs[idx]) if idx >= 0 else 0.0
    )
    jobs["outlier_score"] = jobs["emb_idx"].apply(
        lambda idx: float(outlier_scores[idx]) if idx >= 0 else 1.0
    )

    # Filter out noise
    jobs_clustered = jobs[jobs["cluster_id"] >= 0].copy()
    logger.info(f"Jobs with cluster assignment: {len(jobs_clustered):,}/{len(jobs):,}")
    return jobs


def discover_clusters(jobs_df: pd.DataFrame, embeddings: np.ndarray) -> tuple:
    """
    Full-corpus HDBSCAN discovery. 
    Returns (clustered_jobs_df, clusterer)
    """
    # Build embedding matrix for jobs that have valid emb_idx
    valid = jobs_df[jobs_df["emb_idx"] >= 0].copy()
    valid_embs = embeddings[valid["emb_idx"].values]

    clusterer = run_hdbscan(valid_embs)

    # Assign back to full jobs_df
    valid["cluster_id"]    = clusterer.labels_
    valid["cluster_prob"]  = clusterer.probabilities_
    valid["outlier_score"] = clusterer.outlier_scores_

    # Merge back
    jobs_out = jobs_df.copy()
    jobs_out["cluster_id"]    = -1
    jobs_out["cluster_prob"]  = 0.0
    jobs_out["outlier_score"] = 1.0
    jobs_out.loc[valid.index, "cluster_id"]    = valid["cluster_id"].values
    jobs_out.loc[valid.index, "cluster_prob"]  = valid["cluster_prob"].values
    jobs_out.loc[valid.index, "outlier_score"] = valid["outlier_score"].values

    return jobs_out, clusterer


def discover_quarterly_clusters(
    jobs_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict:
    """
    Run HDBSCAN per quarter for temporal tracking.
    Returns: dict {quarter -> (quarter_jobs_df_with_clusters, clusterer)}
    """
    mcs = H_CFG.get("quarterly_min_cluster_size", 8)
    ms  = H_CFG.get("quarterly_min_samples", 3)

    quarters = sorted(jobs_df["quarter"].dropna().unique())
    logger.info(f"Running per-quarter HDBSCAN over {len(quarters)} quarters")

    quarterly_results = {}
    for q in quarters:
        q_jobs = jobs_df[jobs_df["quarter"] == q].copy()
        valid  = q_jobs[q_jobs["emb_idx"] >= 0]

        if len(valid) < mcs * 2:
            logger.warning(f"Quarter {q}: too few postings ({len(valid)}), skipping")
            continue

        q_embs = embeddings[valid["emb_idx"].values]
        try:
            clusterer = run_hdbscan(q_embs, min_cluster_size=mcs, min_samples=ms)
        except Exception as e:
            logger.warning(f"Quarter {q} HDBSCAN failed: {e}")
            continue

        valid = valid.copy()
        valid["cluster_id"]    = clusterer.labels_
        valid["cluster_prob"]  = clusterer.probabilities_
        valid["outlier_score"] = clusterer.outlier_scores_

        n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        logger.info(f"  {q}: {n_clusters} clusters from {len(valid)} postings")

        quarterly_results[q] = (valid, clusterer, q_embs)

    return quarterly_results


def compute_cluster_centroids(
    cluster_jobs: pd.DataFrame,
    embeddings: np.ndarray,
) -> dict:
    """
    Compute mean embedding centroid per cluster.
    Returns: {cluster_id -> centroid_array}
    """
    centroids = {}
    for cid, grp in cluster_jobs[cluster_jobs["cluster_id"] >= 0].groupby("cluster_id"):
        idxs = grp["emb_idx"].values
        valid_idxs = idxs[idxs >= 0]
        if len(valid_idxs):
            centroid = embeddings[valid_idxs].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[cid] = centroid / norm if norm > 0 else centroid
    return centroids


if __name__ == "__main__":
    from src.preprocessing.job_cleaning import load_all_jobs
    from src.embeddings.sentence_embeddings import generate_or_load_embeddings

    jobs = load_all_jobs(CFG)
    unique_titles, embeddings, title_to_idx, jobs = generate_or_load_embeddings(jobs)

    # Full-corpus clustering
    jobs, clusterer = discover_clusters(jobs, embeddings)
    out = BASE / "data" / "clusters" / "clustered_jobs_full.csv"
    jobs.to_csv(out, index=False)
    print(f"\nSaved full clustering: {out}")
    print(f"Cluster distribution:\n{jobs['cluster_id'].value_counts().head(20)}")
