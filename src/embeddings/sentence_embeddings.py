"""
src/embeddings/sentence_embeddings.py
======================================
Generates and caches SBERT embeddings for IT job titles.

Key design decisions:
  - Only embeds UNIQUE clean titles (deduplicated) — fast and memory efficient
  - Caches to disk so re-runs are instant
  - Also embeds canonical role names for similarity comparison
  - Uses all-MiniLM-L6-v2 by default (384-dim, fast, good quality)
"""

import os
import sys
import yaml
import logging
import hashlib
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

EMB_CFG = CFG["embeddings"]
MODEL_NAME = EMB_CFG["model_name"]
BATCH_SIZE = EMB_CFG.get("batch_size", 512)
CACHE_NPY  = BASE / EMB_CFG["cache_file"]
ID_MAP_CSV = BASE / EMB_CFG["id_map_file"]
CANON_NPY  = BASE / EMB_CFG["canonical_cache"]


def get_model():
    """Load (or re-use cached) SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def embed_titles(titles: list[str], model=None) -> np.ndarray:
    """Embed a list of titles. Returns (N, D) float32 array."""
    if model is None:
        model = get_model()
    logger.info(f"Embedding {len(titles):,} titles with {MODEL_NAME} ...")
    embeddings = model.encode(
        titles,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine distance = 1 - dot product
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def generate_or_load_embeddings(jobs_df: pd.DataFrame, force_recompute: bool = False):
    """
    Main entry point. Returns:
        - unique_titles: list of unique clean_title strings
        - embeddings: np.ndarray (N_unique, D)
        - title_to_idx: dict {clean_title -> row index in embeddings}
        - job_df_with_idx: original df with 'emb_idx' column added
    """
    CACHE_NPY.parent.mkdir(parents=True, exist_ok=True)

    unique_titles = jobs_df["clean_title"].dropna().unique().tolist()
    title_to_idx  = {t: i for i, t in enumerate(unique_titles)}

    cache_ok = (
        not force_recompute
        and CACHE_NPY.exists()
        and ID_MAP_CSV.exists()
    )

    if cache_ok:
        # Verify cache covers current titles
        cached_map = pd.read_csv(ID_MAP_CSV)
        cached_titles = set(cached_map["clean_title"].tolist())
        new_titles = set(unique_titles) - cached_titles
        if not new_titles:
            logger.info(f"Loading cached embeddings from {CACHE_NPY}")
            embeddings = np.load(CACHE_NPY)
            title_to_idx = dict(zip(cached_map["clean_title"], cached_map["emb_idx"]))
            unique_titles = cached_map["clean_title"].tolist()
            cache_ok = True
        else:
            logger.info(f"{len(new_titles)} new titles — recomputing embeddings")
            cache_ok = False

    if not cache_ok:
        model = get_model()
        embeddings = embed_titles(unique_titles, model)
        np.save(CACHE_NPY, embeddings)
        pd.DataFrame({"clean_title": unique_titles,
                      "emb_idx": range(len(unique_titles))}).to_csv(ID_MAP_CSV, index=False)
        logger.info(f"Embeddings saved: {CACHE_NPY} | shape={embeddings.shape}")

    # Add embedding index to jobs_df
    jobs_df = jobs_df.copy()
    jobs_df["emb_idx"] = jobs_df["clean_title"].map(title_to_idx).fillna(-1).astype(int)
    valid = jobs_df[jobs_df["emb_idx"] >= 0]
    logger.info(f"Jobs with valid embedding index: {len(valid):,}/{len(jobs_df):,}")

    return unique_titles, embeddings, title_to_idx, jobs_df


def embed_canonical_roles(cfg: dict = CFG, force_recompute: bool = False) -> tuple:
    """
    Embed canonical role names for novelty comparison.
    Returns: (canonical_names, canonical_embeddings)
    """
    CANON_NPY.parent.mkdir(parents=True, exist_ok=True)
    canon_path = BASE / cfg["paths"]["canonical_roles"]
    canon_df = pd.read_csv(canon_path)
    canon_names = canon_df["canonical_role_name"].tolist()

    if not force_recompute and CANON_NPY.exists():
        canon_embs = np.load(CANON_NPY)
        if len(canon_embs) == len(canon_names):
            logger.info(f"Loaded canonical embeddings ({len(canon_names)} roles)")
            return canon_names, canon_embs

    logger.info(f"Embedding {len(canon_names)} canonical roles ...")
    model = get_model()
    canon_embs = embed_titles(canon_names, model)
    np.save(CANON_NPY, canon_embs)
    logger.info(f"Canonical embeddings saved: {CANON_NPY}")
    return canon_names, canon_embs


if __name__ == "__main__":
    from src.preprocessing.job_cleaning import load_all_jobs
    jobs = load_all_jobs(CFG)
    unique_titles, embeddings, title_to_idx, jobs = generate_or_load_embeddings(jobs)
    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Unique titles: {len(unique_titles)}")

    canon_names, canon_embs = embed_canonical_roles()
    print(f"Canonical embeddings: {canon_embs.shape}")
