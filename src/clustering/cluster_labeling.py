"""
src/clustering/cluster_labeling.py
=====================================
Generates human-readable candidate role names from cluster evidence
and assesses semantic novelty vs canonical roles.
"""

import sys
import re
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "pipeline_config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

NOVELTY_CFG = CFG["novelty"]
HIGH_NOVEL_THRESH = NOVELTY_CFG["high_novelty_threshold"]
CANON_SIM_THRESH  = NOVELTY_CFG["canonical_similarity_threshold"]


# Key technical terms that should appear in role names
TECH_TERMS = {
    "ai", "ml", "llm", "rag", "nlp", "mlops", "devops", "sre", "cloud",
    "kubernetes", "docker", "generative", "deep learning", "machine learning",
    "data", "backend", "frontend", "fullstack", "security", "quantum",
    "robotics", "embedded", "platform", "analytics", "engineer", "developer",
    "scientist", "architect", "analyst", "agent", "agentic", "edge", "iot",
}

_NOISE_WORDS = {
    "job", "opportunity", "opening", "position", "role", "opportunity",
    "new", "apply", "now", "hiring", "wanted", "needed", "required",
}


def extract_cluster_representative_titles(
    cluster_jobs: pd.DataFrame,
    top_n: int = 10,
) -> list[str]:
    """Get top N most frequent clean titles for a cluster."""
    if "clean_title" not in cluster_jobs.columns:
        return []
    vc = cluster_jobs["clean_title"].value_counts()
    return vc.head(top_n).index.tolist()


def generate_candidate_role_name(titles: list[str]) -> str:
    """
    Generate a candidate role name from the most frequent titles in a cluster.
    Strategy: find the most informative 2-4 word phrase that appears in top titles.
    """
    if not titles:
        return "Unknown Role"

    # Count word frequencies across top titles
    word_counts = Counter()
    for t in titles:
        words = [w for w in t.lower().split() if len(w) >= 3 and w not in _NOISE_WORDS]
        word_counts.update(words)

    # Score words: prefer technical terms
    scored = {}
    for w, cnt in word_counts.items():
        score = cnt
        if w in TECH_TERMS:
            score *= 2
        scored[w] = score

    # Take top scored words and build a name
    top_words = sorted(scored, key=lambda x: scored[x], reverse=True)[:6]

    # Find the most frequent title that contains the top word
    if top_words:
        top_word = top_words[0]
        for t in titles:
            if top_word in t.lower():
                # Return title as-is if it's short enough
                if 2 <= len(t.split()) <= 6:
                    return t.title()
                # Otherwise take first 4 words
                return " ".join(t.split()[:4]).title()

    # Fallback: return most common title
    return titles[0].title() if titles else "Unknown Role"


def compute_title_novelty(
    candidate_titles: list[str],
    canonical_embeddings: np.ndarray,
    canonical_names: list[str],
    title_embeddings: np.ndarray = None,
    title_to_idx: dict = None,
) -> dict:
    """
    Compute title novelty as 1 - max_cosine_similarity_to_any_canonical_role.
    
    Returns dict with:
      - max_canonical_sim: highest cosine similarity to any canonical role
      - closest_canonical: name of closest canonical role
      - title_novelty: 1 - max_canonical_sim (higher = more novel)
      - novelty_category: "high", "medium", "low"
    """
    if not candidate_titles or title_embeddings is None or title_to_idx is None:
        return {
            "max_canonical_sim": 0.5,
            "closest_canonical": "Unknown",
            "title_novelty": 0.5,
            "novelty_category": "medium",
        }

    # Get embeddings for candidate titles
    cand_embs = []
    for t in candidate_titles:
        idx = title_to_idx.get(t)
        if idx is not None and idx < len(title_embeddings):
            cand_embs.append(title_embeddings[idx])

    if not cand_embs:
        return {
            "max_canonical_sim": 0.5,
            "closest_canonical": "Unknown",
            "title_novelty": 0.5,
            "novelty_category": "medium",
        }

    # Average cluster embedding
    cluster_centroid = np.mean(cand_embs, axis=0)
    norm = np.linalg.norm(cluster_centroid)
    if norm > 0:
        cluster_centroid = cluster_centroid / norm

    # Cosine similarities to all canonical roles
    sims = canonical_embeddings @ cluster_centroid
    best_idx = int(np.argmax(sims))
    max_sim  = float(sims[best_idx])
    closest  = canonical_names[best_idx] if best_idx < len(canonical_names) else "Unknown"

    novelty = 1.0 - max_sim

    if max_sim >= CANON_SIM_THRESH:
        cat = "low"       # Very similar to existing — evolution, not new
    elif novelty >= HIGH_NOVEL_THRESH:
        cat = "high"
    else:
        cat = "medium"

    return {
        "max_canonical_sim":    round(max_sim, 4),
        "closest_canonical":    closest,
        "title_novelty":        round(novelty, 4),
        "novelty_category":     cat,
    }


def label_all_clusters(
    timeline_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    canonical_embeddings: np.ndarray,
    canonical_names: list[str],
    title_embeddings: np.ndarray,
    title_to_idx: dict,
) -> pd.DataFrame:
    """
    For each tracked cluster: generate a candidate name + novelty scores.
    Returns enriched cluster summary DataFrame.
    """
    rows = []
    cluster_ids = timeline_df["tracked_cluster_id"].unique()

    logger.info(f"Labeling {len(cluster_ids)} tracked clusters ...")

    for tcid in cluster_ids:
        # Get all jobs belonging to this cluster (latest quarter top titles)
        q_entries = timeline_df[timeline_df["tracked_cluster_id"] == tcid]
        # Get top titles from timeline
        all_titles_str = "|".join(q_entries["top_titles"].dropna().tolist())
        titles = [t.strip() for t in all_titles_str.split("|") if t.strip()]

        # Deduplicate while preserving order
        seen = set()
        unique_titles = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                unique_titles.append(t)

        candidate_name = generate_candidate_role_name(unique_titles[:20])

        novelty_info = compute_title_novelty(
            unique_titles[:10],
            canonical_embeddings,
            canonical_names,
            title_embeddings,
            title_to_idx,
        )

        persistence = q_entries["quarter"].nunique()
        total_posts = q_entries["posting_count"].sum() if "posting_count" in q_entries.columns else 0

        rows.append({
            "tracked_cluster_id":   tcid,
            "candidate_role_name":  candidate_name,
            "persistence_quarters": persistence,
            "total_postings":       total_posts,
            "representative_titles": "|".join(unique_titles[:5]),
            **novelty_info,
        })

    labeled_df = pd.DataFrame(rows)
    logger.info(f"Labeled clusters: {len(labeled_df)} | "
                f"High novelty: {(labeled_df['novelty_category']=='high').sum()}")
    return labeled_df


if __name__ == "__main__":
    # Quick test
    test_titles = [
        "llm application engineer",
        "rag engineer",
        "generative ai application developer",
        "ai agent engineer",
        "agentic ai developer",
    ]
    name = generate_candidate_role_name(test_titles)
    print(f"Generated name: {name}")
