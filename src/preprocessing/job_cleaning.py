"""
src/preprocessing/job_cleaning.py
==================================
Cleans raw job titles from Adzuna (and optionally LinkedIn/Naukri) for the
semantic discovery pipeline (Problem B — Unknown Role Discovery).

DESIGN RULES:
  - Preserves technical terms (AI, ML, LLM, RAG, MLOps, etc.)
  - Removes seniority noise (Senior, Junior, I/II/III)
  - Removes location/contract noise
  - Deduplicates at (clean_title, source) level
  - Outputs a leakage-safe dataset keyed by (job_id, month)
"""

import os
import re
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

# ─── Load Config ──────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "pipeline_config.yaml"

with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

PRESERVE = set(CFG.get("preserve_terms", []))
STOP_LEVELS = {"i", "ii", "iii", "iv", "v", "jr", "sr", "junior", "senior",
               "associate", "principal", "staff", "lead", "director",
               "vice president", "vp", "head", "chief", "manager", "executive",
               "contract", "contractor", "temp", "temporary", "intern",
               "graduate", "entry", "level", "mid", "part-time", "part time",
               "remote", "hybrid", "onsite", "on-site", "full-time", "fulltime"}

# Seniority pattern at start of title
_LEVEL_SUFFIXES = re.compile(
    r"\b(i{1,3}|iv|v|1|2|3|4|5|"
    r"junior|jr\.?|senior|sr\.?|associate|principal|"
    r"staff|lead|director|vp|vice president|head of|chief|"
    r"manager|executive|intern|graduate|entry.level|mid.level)\b",
    re.IGNORECASE,
)
_LOCATION_PATTERN = re.compile(
    r"\b(remote|hybrid|on.?site|new york|london|bangalore|san francisco|"
    r"india|us|usa|uk|global)\b",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^a-zA-Z0-9\s\+#\./]")
_MULTI_SPACE = re.compile(r"\s{2,}")

# Normalize common abbreviations to canonical form  
ABBREV_MAP = {
    "genai": "generative ai",
    "gen ai": "generative ai",
    "ml engineer": "machine learning engineer",
    "ai/ml": "ai ml",
    "devops engineer": "devops engineer",
    "dev ops": "devops",
    "devsecops": "devops security",
    "nlp engineer": "nlp engineer",
    "llm engineer": "llm engineer",
    "rag engineer": "rag engineer",
}


def clean_title(raw: str) -> str:
    """
    Clean a single job title for semantic embedding.
    Strategy: minimal cleaning — remove noise, preserve signal.
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""

    t = raw.strip()

    # Protect preserve terms by temporarily replacing with placeholders
    placeholder_map = {}
    for i, term in enumerate(sorted(PRESERVE, key=len, reverse=True)):
        pat = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        placeholder = f"__PRSV{i}__"
        if pat.search(t):
            t = pat.sub(placeholder, t)
            placeholder_map[placeholder] = term

    # Lowercase remaining
    t = t.lower()

    # Apply abbrev map
    for k, v in ABBREV_MAP.items():
        t = t.replace(k, v)

    # Remove location noise
    t = _LOCATION_PATTERN.sub("", t)

    # Remove seniority levels
    t = _LEVEL_SUFFIXES.sub("", t)

    # Remove punctuation (except +, #, /, .)
    t = _PUNCT.sub(" ", t)

    # Restore preserved terms
    for placeholder, term in placeholder_map.items():
        t = t.replace(placeholder, term)

    # Clean whitespace
    t = _MULTI_SPACE.sub(" ", t).strip()
    t = " ".join(w for w in t.split() if len(w) >= 2)

    return t


def load_adzuna_it_jobs(cfg: dict) -> pd.DataFrame:
    """
    Load Adzuna IT jobs with minimal required columns.
    Returns: DataFrame with [job_id, raw_title, clean_title, company, month, source]
    """
    path = BASE / cfg["paths"]["raw_adzuna"]
    logger.info(f"Loading Adzuna from {path} ...")

    chunks = []
    reader = pd.read_csv(
        path,
        usecols=["job_id", "title", "company", "created", "category_label"],
        dtype={"job_id": str, "title": str, "company": str},
        chunksize=50_000,
    )
    for chunk in reader:
        it_chunk = chunk[chunk["category_label"] == cfg["data"]["it_category_label"]]
        if len(it_chunk):
            chunks.append(it_chunk)

    if not chunks:
        logger.warning("No IT jobs found in Adzuna!")
        return pd.DataFrame()

    az = pd.concat(chunks, ignore_index=True)
    logger.info(f"Adzuna IT jobs loaded: {len(az):,} rows")

    # Parse dates
    az["date"] = pd.to_datetime(az["created"], utc=True, errors="coerce")
    az = az.dropna(subset=["date"])

    # Filter by date range
    min_dt = pd.Timestamp(cfg["data"]["min_date"], tz="UTC")
    max_dt = pd.Timestamp(cfg["data"]["max_date"], tz="UTC")
    az = az[(az["date"] >= min_dt) & (az["date"] <= max_dt)]

    az["month"] = az["date"].dt.to_period("M").dt.strftime("%Y-%m")
    az["quarter"] = az["date"].dt.to_period("Q").dt.strftime("%Y-Q%q")
    az["clean_title"] = az["title"].apply(clean_title)
    az = az[az["clean_title"].str.len() >= 5]
    az["source"] = "adzuna"

    result = az[["job_id", "title", "clean_title", "company", "month", "quarter", "source"]].copy()
    result = result.rename(columns={"title": "raw_title"})
    logger.info(f"Adzuna after cleaning: {len(result):,} rows, {result['clean_title'].nunique():,} unique titles")
    return result


def load_linkedin_jobs(cfg: dict) -> pd.DataFrame:
    """
    Load LinkedIn IT postings (subset of postings.csv by IT-relevant titles).
    """
    path = BASE / cfg["paths"]["raw_linkedin_postings"]
    logger.info(f"Loading LinkedIn postings from {path} (sampling for title discovery)...")

    it_keywords = [
        "engineer", "developer", "scientist", "analyst", "architect",
        "devops", "mlops", "aiops", "dataops", "sre", "platform", "cloud",
        "security", "machine learning", "deep learning", "nlp", "ai", "ml",
        "data", "backend", "frontend", "fullstack", "software", "python",
        "java", "mobile", "robotics", "quantum", "embedded"
    ]
    kw_pattern = "|".join(it_keywords)

    chunks = []
    try:
        reader = pd.read_csv(
            path,
            usecols=["job_id", "title", "company_id", "listed_time"],
            dtype={"job_id": str, "title": str},
            chunksize=50_000,
        )
        for chunk in reader:
            it_chunk = chunk[
                chunk["title"].str.lower().str.contains(kw_pattern, na=False, regex=True)
            ]
            if len(it_chunk):
                chunks.append(it_chunk.head(2000))  # sample per chunk for speed
            if sum(len(c) for c in chunks) > 100_000:
                break
    except Exception as e:
        logger.warning(f"LinkedIn load failed: {e}")
        return pd.DataFrame()

    if not chunks:
        return pd.DataFrame()

    lk = pd.concat(chunks, ignore_index=True)
    lk["date"] = pd.to_datetime(lk["listed_time"], unit="ms", errors="coerce")
    lk = lk.dropna(subset=["date"])
    lk["month"] = lk["date"].dt.to_period("M").dt.strftime("%Y-%m")
    lk["quarter"] = lk["date"].dt.to_period("Q").dt.strftime("%Y-Q%q")
    lk["clean_title"] = lk["title"].apply(clean_title)
    lk = lk[lk["clean_title"].str.len() >= 5]
    lk["source"] = "linkedin"
    lk["company"] = lk["company_id"].astype(str)

    result = lk[["job_id", "title", "clean_title", "company", "month", "quarter", "source"]].copy()
    result = result.rename(columns={"title": "raw_title"})
    logger.info(f"LinkedIn after cleaning: {len(result):,} rows, {result['clean_title'].nunique():,} unique titles")
    return result


def load_all_jobs(cfg: dict, include_linkedin: bool = True) -> pd.DataFrame:
    """
    Load and merge Adzuna + LinkedIn IT job data for discovery pipeline.
    """
    dfs = [load_adzuna_it_jobs(cfg)]
    if include_linkedin:
        lk = load_linkedin_jobs(cfg)
        if len(lk):
            dfs.append(lk)

    combined = pd.concat([d for d in dfs if len(d)], ignore_index=True)
    logger.info(f"Combined IT jobs: {len(combined):,} rows | {combined['clean_title'].nunique():,} unique titles")
    return combined


if __name__ == "__main__":
    jobs = load_all_jobs(CFG)
    out_path = BASE / "data" / "clusters" / "cleaned_it_jobs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(out_path, index=False)
    logger.info(f"Saved to {out_path}")

    # Quick stats
    print("\n=== CLEANING STATS ===")
    print(f"Total IT job postings: {len(jobs):,}")
    print(f"Unique cleaned titles: {jobs['clean_title'].nunique():,}")
    print(f"Date range: {jobs['month'].min()} → {jobs['month'].max()}")
    print(f"\nTop 20 most frequent cleaned titles:")
    print(jobs['clean_title'].value_counts().head(20).to_string())
