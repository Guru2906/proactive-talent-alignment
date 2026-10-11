"""
build_notebook.py
------------------
Builds notebooks/emerging_roles_prediction.ipynb using nbformat.
Run from project root:  python notebooks/build_notebook.py
"""
# Fix cp1252 encoding issues for the builder script itself
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import json
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent   # project root
NB_PATH = ROOT / "notebooks" / "emerging_roles_prediction.ipynb"

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.0"
    }
}

def md(text):
    return nbf.v4.new_markdown_cell(text.strip())

def code(src):
    return nbf.v4.new_code_cell(src.strip())

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Project / Pipeline Overview
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("""
# 🔮 Emerging IT Role Discovery — Pipeline B Notebook

**Project:** Proactive IT Role & Skill Emergence Forecaster  
**Pipeline B** discovers *genuinely new* IT job roles that have not yet been codified into the 39 canonical roles tracked by the supervised forecasting pipeline (Pipeline A).

---

## Architecture Overview

```
Raw IT job postings (Adzuna + LinkedIn)
      ↓
Job title cleaning & technical-term preservation
      ↓
SBERT dense semantic embeddings  (all-MiniLM-L6-v2, 384-dim)
      ↓
HDBSCAN variable-density clustering  (no fixed K)
      ↓
Temporal cluster tracking across quarterly windows
      ↓
Cluster labeling — automated role-name generation
      ↓
Multi-component emergence scoring
      ↓
Growth forecasting per persistent cluster
      ↓
★ Final ranked Emerging Role Candidates ★
```

## Pipeline A vs Pipeline B — Key Distinction

| | **Pipeline A** | **Pipeline B** |
|---|---|---|
| Approach | Supervised LightGBM regression | Unsupervised SBERT + HDBSCAN |
| Roles covered | 39 pre-defined canonical roles | Any number of discovered clusters |
| Ground truth | Historical Adzuna market share (bps) | No labelled target — discovery only |
| Output | 6M/12M demand forecast per role | Ranked "Emerging Role Candidate" list |
| Output file | `step1_enhanced/outputs/top_emerging_roles_6m.csv` | `data/outputs/emerging_role_predictions.csv` |

> ⚠️ **Scientific note:** The candidates produced here are **observed semantic clusters** from raw job-market text, ranked by a transparent mathematical emergence score. They are **not guaranteed future job titles** — they are evidence-based candidates that show persistent, growing, and novel demand signals.
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Imports and Configuration
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 2 — Imports and Configuration"))

nb.cells.append(code("""
import sys, os, warnings, json
warnings.filterwarnings("ignore")

# ── Make project root importable from this notebook ──────────────────────────
from pathlib import Path

# Resolve project root: notebooks/ is one level below root
NOTEBOOK_DIR = Path.cwd()
if NOTEBOOK_DIR.name == "notebooks":
    PROJECT_ROOT = NOTEBOOK_DIR.parent
else:
    PROJECT_ROOT = NOTEBOOK_DIR

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"✅ Project root: {PROJECT_ROOT}")
print(f"   Python: {sys.version.split()[0]}")
"""))

nb.cells.append(code("""
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Consistent plot style
plt.rcParams.update({
    "figure.facecolor": "#0f1923",
    "axes.facecolor":   "#0f1923",
    "axes.labelcolor":  "#cbd5e1",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "text.color":       "#f8fafc",
    "grid.color":       "#1e2d40",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})
ACCENT   = "#38bdf8"
FRONTIER = "#c084fc"
SUCCESS  = "#34d399"
WARNING  = "#fbbf24"

pd.set_option("display.max_columns", 40)
pd.set_option("display.max_colwidth", 60)
pd.set_option("display.width", 130)
print("✅ Libraries loaded")
"""))

nb.cells.append(code("""
# ── Load pipeline configuration ───────────────────────────────────────────────
CFG_PATH = PROJECT_ROOT / "config" / "pipeline_config.yaml"
assert CFG_PATH.exists(), f"Config not found: {CFG_PATH}"

with open(CFG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

print("✅ Configuration loaded:", CFG_PATH)
print()
print("┌─────────────────────────────────────────────────────────┐")
print("│  KEY CONFIGURATION PARAMETERS                           │")
print("└─────────────────────────────────────────────────────────┘")
print(f"  Embedding model          : {CFG['embeddings']['model_name']}")
print(f"  Embedding dimensions     : 384  (all-MiniLM-L6-v2 fixed)")
print(f"  HDBSCAN min_cluster_size : {CFG['hdbscan']['min_cluster_size']}")
print(f"  HDBSCAN min_samples      : {CFG['hdbscan']['min_samples']}")
print(f"  HDBSCAN metric           : {CFG['hdbscan']['metric']}")
print(f"  Cluster similarity thresh: {CFG['cluster_tracking']['centroid_similarity_threshold']}")
print(f"  Min persistence (quarters): {CFG['cluster_tracking']['min_persistence_quarters']}")
print(f"  Min cluster postings     : {CFG['cluster_tracking']['min_cluster_postings']}")
print()
print("  Emergence Score Weights:")
for k, v in CFG["emergence_weights"].items():
    print(f"    {k:<30} {v}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Load Raw Job Data
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 3 — Load Raw Job Data"))

nb.cells.append(code("""
# The cleaned dataset is the direct output of Pipeline B preprocessing.
# If it exists (from a prior run) we load it; otherwise we re-run the cleaning step.
CLEANED_PATH = PROJECT_ROOT / "data" / "clusters" / "cleaned_it_jobs.csv"

if CLEANED_PATH.exists():
    jobs_df = pd.read_csv(CLEANED_PATH)
    print(f"✅ Loaded pre-cleaned IT jobs: {CLEANED_PATH.name}")
else:
    print("⚠️  Cleaned dataset not found — will generate in Section 4.")
    jobs_df = None

# Show raw Adzuna basic info regardless
RAW_AZ = PROJECT_ROOT / CFG["paths"]["raw_adzuna"]
if RAW_AZ.exists():
    az_sample = pd.read_csv(RAW_AZ, usecols=["job_id", "title", "created", "category_label"], nrows=5)
    raw_it_count = sum(
        1 for chunk in pd.read_csv(
            RAW_AZ, usecols=["category_label"], chunksize=50_000
        ) if len(chunk[chunk["category_label"] == "IT Jobs"])
        # count IT rows
        for _ in [chunk]  # placeholder
    )
    # Simpler count
    total_raw = pd.read_csv(RAW_AZ, usecols=["category_label"])
    it_rows = (total_raw["category_label"] == CFG["data"]["it_category_label"]).sum()
    print(f"\\n  Raw Adzuna IT postings: {it_rows:,}")
    print(f"  Category filter       : '{CFG['data']['it_category_label']}'")
    print(f"  Date range (config)   : {CFG['data']['min_date']} → {CFG['data']['max_date']}")
else:
    print(f"⚠️  Adzuna file not found at: {RAW_AZ}")

if jobs_df is not None:
    print(f"\\n  Cleaned dataset shape : {jobs_df.shape}")
    print(f"  Columns               : {list(jobs_df.columns)}")
    print(f"  Unique clean titles   : {jobs_df['clean_title'].nunique():,}")
    print(f"  Date range (months)   : {jobs_df['month'].min()} → {jobs_df['month'].max()}")
    print(f"  Data sources          : {jobs_df['source'].value_counts().to_dict()}")
    print(f"  Quarterly windows     : {sorted(jobs_df['quarter'].dropna().unique())}")
    print()
    print("  Sample rows:")
    display(jobs_df[["job_id","raw_title","clean_title","source","month","quarter"]].head(8))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Job Cleaning
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 4 — Job Cleaning / Preprocessing"))

nb.cells.append(code("""
from src.preprocessing.job_cleaning import load_all_jobs, clean_title

# Run if we do not have a cached cleaned file
if jobs_df is None:
    print("Running Pipeline B preprocessing (cleaning raw IT postings)...")
    jobs_df = load_all_jobs(CFG, include_linkedin=True)
    out = PROJECT_ROOT / "data" / "clusters" / "cleaned_it_jobs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    jobs_df.to_csv(out, index=False)
    print(f"Saved: {out}")
else:
    print("✅ Using cached cleaned dataset (no re-run needed)")

print(f"\\n  Total cleaned IT postings : {len(jobs_df):,}")
print(f"  Unique clean titles       : {jobs_df['clean_title'].nunique():,}")

# Verify key technical tokens are preserved
test_cases = [
    ("Senior LLM Engineer III",            "llm engineer"),
    ("AI/ML Platform Engineer - Remote",   "AI"),
    ("MLOps DevOps Infrastructure Lead",   "MLOps"),
    ("Generative AI Research Scientist",   "generative"),
    ("RAG Application Developer (Contract)","RAG"),
]
print("\\n  Token preservation checks (existing clean_title function):")
print(f"  {'Input':<45} {'Output':<45} {'Pass?'}")
print("  " + "-"*100)
for raw, must_contain in test_cases:
    cleaned = clean_title(raw)
    ok = must_contain.lower() in cleaned.lower()
    status = "✅" if ok else "❌"
    print(f"  {raw:<45} {cleaned:<45} {status}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SBERT Embeddings
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 5 — Generate / Load SBERT Embeddings"))

nb.cells.append(code("""
from src.embeddings.sentence_embeddings import generate_or_load_embeddings, embed_canonical_roles

CACHE_NPY  = PROJECT_ROOT / CFG["embeddings"]["cache_file"]
ID_MAP_CSV = PROJECT_ROOT / CFG["embeddings"]["id_map_file"]
CANON_NPY  = PROJECT_ROOT / CFG["embeddings"]["canonical_cache"]

cache_exists = CACHE_NPY.exists() and ID_MAP_CSV.exists()
print(f"  Embedding cache exists: {cache_exists}  ({CACHE_NPY})")

# This call will LOAD from cache if available, re-generate if not
unique_titles, embeddings, title_to_idx, jobs_df = generate_or_load_embeddings(
    jobs_df, force_recompute=False
)
canonical_names, canonical_embs = embed_canonical_roles(CFG)

print(f"\\n  Embedding model         : {CFG['embeddings']['model_name']}")
print(f"  Title embedding matrix  : {embeddings.shape}  (unique_titles × 384-dim)")
print(f"  Canonical embedding     : {canonical_embs.shape}  ({len(canonical_names)} canonical roles × 384-dim)")
print(f"  Jobs with valid emb_idx : {(jobs_df['emb_idx'] >= 0).sum():,}")
print(f"  Cache file              : {CACHE_NPY}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — HDBSCAN Discovery
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 6 — HDBSCAN Discovery"))

nb.cells.append(code("""
from src.clustering.hdbscan_discovery import discover_quarterly_clusters

print("Running per-quarter HDBSCAN (using existing module)...")
quarterly_results = discover_quarterly_clusters(jobs_df, embeddings)

quarters_processed = list(quarterly_results.keys())
print(f"\\n  Quarterly windows successfully clustered: {len(quarters_processed)}")
print(f"  Quarters: {quarters_processed}")

# Summarise per-quarter statistics
q_summary = []
for q, (q_jobs, clusterer, q_embs) in quarterly_results.items():
    n_clustered = q_jobs["cluster_id"].value_counts()
    n_clusters  = int((q_jobs["cluster_id"] >= 0).sum() > 0)  # at least 1 non-noise
    import numpy as np
    labels = clusterer.labels_
    n_cls   = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    q_summary.append({
        "Quarter":       q,
        "Total postings": len(q_jobs),
        "Clusters found": n_cls,
        "Noise points":  n_noise,
        "Noise %":       f"{n_noise/len(q_jobs)*100:.1f}%",
    })

qdf = pd.DataFrame(q_summary)
print()
display(qdf)
"""))

nb.cells.append(code("""
# Show the largest clusters from the largest quarterly window
largest_q = max(quarterly_results, key=lambda q: len(quarterly_results[q][0]))
q_jobs_lg, clusterer_lg, _ = quarterly_results[largest_q]

print(f"\\n  Showing cluster profiles from largest quarter: {largest_q}\\n")

cluster_profiles = []
for cid, grp in q_jobs_lg[q_jobs_lg["cluster_id"] >= 0].groupby("cluster_id"):
    top_titles = grp["clean_title"].value_counts().head(3).index.tolist()
    cluster_profiles.append({
        "Cluster ID":       int(cid),
        "Postings":         len(grp),
        "Companies":        grp["company"].nunique(),
        "Top Representative Titles": " | ".join(top_titles),
    })

cp_df = pd.DataFrame(cluster_profiles).sort_values("Postings", ascending=False).head(20)
display(cp_df.reset_index(drop=True))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Temporal Cluster Tracking
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 7 — Temporal Cluster Tracking"))

nb.cells.append(code("""
from src.clustering.cluster_tracking import (
    build_cluster_timelines, compute_persistence, get_persistent_clusters
)

print("Building temporal cluster timelines across quarterly windows...")
timeline_df, tracked_centroids = build_cluster_timelines(quarterly_results, embeddings)
timeline_df = compute_persistence(timeline_df)

print(f"\\n  Total (quarter, cluster) observations : {len(timeline_df)}")
print(f"  Unique tracked cluster IDs            : {timeline_df['tracked_cluster_id'].nunique()}")
print(f"  Quarters covered                      : {sorted(timeline_df['quarter'].unique())}")
print(f"  Cosine similarity threshold            : {CFG['cluster_tracking']['centroid_similarity_threshold']}")

# Save / reload timeline cache
TIMELINE_PATH = PROJECT_ROOT / "data" / "clusters" / "cluster_timeline.csv"
timeline_df.to_csv(TIMELINE_PATH, index=False)

# Persistence distribution
persist_dist = timeline_df.drop_duplicates("tracked_cluster_id")["persistence_quarters"].value_counts().sort_index()
print("\\n  Persistence distribution (quarters → cluster count):")
for q, cnt in persist_dist.items():
    bar = "█" * cnt
    print(f"    {int(q)} quarter(s): {bar} ({cnt} clusters)")
"""))

nb.cells.append(code("""
# Persistent clusters (meeting min_persistence_quarters and min_cluster_postings)
MIN_Q     = CFG["cluster_tracking"]["min_persistence_quarters"]
MIN_POSTS = CFG["cluster_tracking"]["min_cluster_postings"]

persistent = get_persistent_clusters(timeline_df, min_quarters=MIN_Q, min_postings=MIN_POSTS)
PERSISTENT_PATH = PROJECT_ROOT / "data" / "clusters" / "persistent_clusters.csv"
persistent.to_csv(PERSISTENT_PATH, index=False)

print(f"  Clusters meeting persistence criteria (≥{MIN_Q}Q, ≥{MIN_POSTS} posts): {len(persistent)}")
print()
display(persistent[[
    "tracked_cluster_id","persistence_quarters","total_postings",
    "max_unique_companies","first_seen","last_seen","top_titles"
]].sort_values("persistence_quarters", ascending=False).head(15).reset_index(drop=True))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Cluster Labeling
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 8 — Cluster Labeling & Novelty Scoring"))

nb.cells.append(code("""
from src.clustering.cluster_labeling import label_all_clusters

print("Generating human-readable candidate role names for all tracked clusters...")
labeled = label_all_clusters(
    timeline_df, jobs_df,
    canonical_embs, canonical_names,
    embeddings, title_to_idx,
)

# Merge labels onto persistent clusters
cols_to_use = [c for c in labeled.columns if c not in persistent.columns or c == "tracked_cluster_id"]
persistent_labeled = persistent.merge(labeled[cols_to_use], on="tracked_cluster_id", how="left")

print(f"  Labeled {len(labeled)} total clusters")
print(f"  High novelty clusters  : {(labeled['novelty_category']=='high').sum()}")
print(f"  Medium novelty clusters: {(labeled['novelty_category']=='medium').sum()}")
print(f"  Low novelty clusters   : {(labeled['novelty_category']=='low').sum()}")
print()

# Display human-readable labels for persistent clusters
display_cols = [
    "tracked_cluster_id", "candidate_role_name",
    "persistence_quarters", "total_postings", "max_unique_companies",
    "title_novelty", "novelty_category", "closest_canonical",
    "representative_titles"
]
display_cols = [c for c in display_cols if c in persistent_labeled.columns]
print("  Persistent cluster labels and novelty:")
display(persistent_labeled[display_cols].sort_values("title_novelty", ascending=False).reset_index(drop=True))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Emergence Scoring
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 9 — Emergence Scoring"))

nb.cells.append(code("""
from src.emerging.emergence_score import (
    compute_emergence_scores, apply_noise_filters, compute_confidence
)

print("Computing multi-component emergence scores...")
scored = compute_emergence_scores(persistent_labeled, timeline_df)

# Apply noise filter
scored_filtered = apply_noise_filters(scored)
print(f"  Clusters before noise filter : {len(scored)}")
print(f"  Clusters after noise filter  : {len(scored_filtered)}")

# Add confidence
def add_confidence(row):
    score, label = compute_confidence(
        row.get("emergence_score", 0.0),
        row.get("persistence_quarters", 1),
        row.get("total_postings", 0),
        row.get("max_unique_companies", 1),
        row.get("novelty_category", "medium"),
    )
    return pd.Series({"confidence_score": score, "confidence_label": label})

conf_df = scored_filtered.apply(add_confidence, axis=1)
scored_filtered[["confidence_score","confidence_label"]] = conf_df

# Save scored clusters
SCORED_PATH = PROJECT_ROOT / "data" / "clusters" / "scored_clusters.csv"
scored_filtered.to_csv(SCORED_PATH, index=False)
print(f"  Saved scored clusters: {SCORED_PATH}")

print()
print("  Emergence Score Formula:")
print("  score = Σ weight_i × component_i")
print()
for comp, w in CFG["emergence_weights"].items():
    print(f"    {w:.2f} × {comp}")

print()
score_cols = [
    "tracked_cluster_id","candidate_role_name","emergence_score","confidence_label",
    "demand_growth_score","skill_velocity_score","title_novelty_score",
    "market_breadth_score","persistence_score","technology_adoption_score",
]
score_cols = [c for c in score_cols if c in scored_filtered.columns]
display(scored_filtered[score_cols].sort_values("emergence_score", ascending=False).reset_index(drop=True))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Emerging Role Predictions (MAIN OUTPUT)
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 10 — Emerging Role Predictions (Main Output)"))

nb.cells.append(code("""
from src.forecasting.cluster_forecaster import forecast_all_clusters

print("Forecasting cluster growth for persistent clusters...")
forecasted = forecast_all_clusters(scored_filtered, timeline_df)

print(f"  Cluster forecasts generated: {len(forecasted)}")
print(f"  Forecast methods used: {forecasted['forecast_method'].value_counts().to_dict()}")

# Build the final ranked prediction table
def classify_cluster(row):
    score  = row.get("emergence_score", 0)
    novel  = row.get("novelty_category", "low")
    persist = row.get("persistence_quarters", 1)
    growth  = row.get("forecast_growth_pct", 0)

    if score >= 0.55 and novel in ("high", "medium"):
        return "Emerging"
    elif score >= 0.40 and persist >= 3:
        return "Growing"
    elif persist >= 4:
        return "Persistent"
    elif novel == "high":
        return "Novel"
    elif persist <= 2 and row.get("total_postings", 0) < 40:
        return "Sparse / Low-Evidence"
    else:
        return "Candidate"

forecasted["classification"] = forecasted.apply(classify_cluster, axis=1)

# Final ranked output
FINAL_COLS_MAP = {
    "emergence_rank":               "Rank",
    "candidate_role_name":          "Emerging Role",
    "tracked_cluster_id":           "Cluster ID",
    "emergence_score":              "Emergence Score",
    "confidence_label":             "Confidence",
    "demand_growth_score":          "Demand Growth",
    "persistence_quarters":         "Persistence (Qtrs)",
    "title_novelty":                "Title Novelty",
    "market_breadth_score":         "Market Breadth",
    "total_postings":               "Historical Volume",
    "forecast_next_quarter_postings":"Forecast Volume (Next Q)",
    "forecast_growth_pct":          "Forecast Growth %",
    "forecast_method":              "Forecast Method",
    "classification":               "Classification",
    "closest_canonical":            "Closest Known Role",
    "representative_titles":        "Representative Titles",
    "first_seen":                   "First Seen",
    "last_seen":                    "Last Seen",
}

available_cols = {k: v for k, v in FINAL_COLS_MAP.items() if k in forecasted.columns}
predictions_df = (
    forecasted[list(available_cols.keys())]
    .rename(columns=available_cols)
    .sort_values("Rank")
    .reset_index(drop=True)
)

print(f"\\n  Final predictions shape: {predictions_df.shape}")
display(predictions_df.head(21))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Top Emerging Roles Display
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 11 — Top Emerging Role Candidates"))

nb.cells.append(code("""
print("=" * 80)
print("  TOP 10 EMERGING IT ROLE CANDIDATES  (Pipeline B — Semantic Discovery)")
print("=" * 80)

top10 = predictions_df.head(10)
for _, row in top10.iterrows():
    rank  = int(row.get("Rank", 0))
    role  = row.get("Emerging Role", "Unknown")
    score = float(row.get("Emergence Score", 0))
    conf  = row.get("Confidence", "—")
    cls   = row.get("Classification", "—")
    growth= row.get("Forecast Growth %", 0)
    pers  = row.get("Persistence (Qtrs)", "—")
    novel = row.get("Title Novelty", 0)
    canon = row.get("Closest Known Role", "—")
    titles = row.get("Representative Titles", "—")

    growth_str = f"+{growth:.1f}%" if growth > 0 else f"{growth:.1f}%"

    print(f"\\n  #{rank:02d} {role}")
    print(f"       Emergence Score : {score:.4f}   |  Confidence: {conf}  |  Class: {cls}")
    print(f"       Forecast Growth : {growth_str}   |  Persistence: {pers} quarters")
    print(f"       Title Novelty   : {float(novel):.3f}   |  Closest Known: {canon}")
    print(f"       Sample Titles   : {str(titles)[:90]}")

print()
print("─" * 80)
print()

if len(predictions_df) >= 11:
    print("\\n" + "=" * 80)
    print("  RANKS 11–20")
    print("=" * 80)
    display(predictions_df.iloc[10:20][
        ["Rank","Emerging Role","Emergence Score","Confidence","Classification","Closest Known Role"]
    ].reset_index(drop=True))
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12 — Visualizations
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 12 — Prediction Visualizations"))

nb.cells.append(code("""
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Emerging IT Role Candidates — Pipeline B", color="#f8fafc", fontsize=14, fontweight="bold")

top_plot = predictions_df.head(12).copy()
roles    = top_plot["Emerging Role"].str[:30].tolist()
scores   = top_plot["Emergence Score"].tolist()
colors   = [FRONTIER if c == "Emerging" else ACCENT
            for c in top_plot["Classification"].tolist()]

# ── Plot 1: Emergence scores ─────────────────────────────────────────────────
ax1 = axes[0, 0]
bars = ax1.barh(range(len(roles)), scores, color=colors, edgecolor="#1e2d40", height=0.7)
ax1.set_yticks(range(len(roles)))
ax1.set_yticklabels(roles, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("Emergence Score", fontsize=9)
ax1.set_title("Top Emerging Role Candidates by Score", fontsize=10, color="#38bdf8")
ax1.axvline(0.5, color=WARNING, lw=1, ls="--", alpha=0.7, label="0.5 threshold")
ax1.legend(fontsize=7)
ax1.set_xlim(0, 1.0)

# ── Plot 2: Persistence vs Emergence Score ───────────────────────────────────
ax2 = axes[0, 1]
if "Persistence (Qtrs)" in predictions_df.columns and len(predictions_df):
    x_vals = predictions_df["Persistence (Qtrs)"].fillna(1).astype(float).tolist()
    y_vals = predictions_df["Emergence Score"].fillna(0).tolist()
    conf_map = {"High": FRONTIER, "Medium": ACCENT, "Low": "#64748b"}
    c_vals = [conf_map.get(str(c), "#64748b") for c in predictions_df["Confidence"].tolist()]
    ax2.scatter(x_vals, y_vals, c=c_vals, s=80, alpha=0.85, edgecolors="#1e2d40")
    ax2.set_xlabel("Persistence (quarters)", fontsize=9)
    ax2.set_ylabel("Emergence Score", fontsize=9)
    ax2.set_title("Persistence vs Emergence Score", fontsize=10, color="#38bdf8")
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=FRONTIER, label="High confidence"),
                  Patch(facecolor=ACCENT,   label="Medium confidence"),
                  Patch(facecolor="#64748b",label="Low confidence")]
    ax2.legend(handles=legend_els, fontsize=7)

# ── Plot 3: Novelty vs Emergence Score ───────────────────────────────────────
ax3 = axes[1, 0]
if "Title Novelty" in predictions_df.columns and len(predictions_df):
    nov  = predictions_df["Title Novelty"].fillna(0).astype(float).tolist()
    escore = predictions_df["Emergence Score"].fillna(0).tolist()
    ax3.scatter(nov, escore, c=ACCENT, s=80, alpha=0.85, edgecolors="#1e2d40")
    for _, r in predictions_df.head(5).iterrows():
        ax3.annotate(str(r["Emerging Role"])[:18],
                     (float(r["Title Novelty"]), float(r["Emergence Score"])),
                     fontsize=6.5, color="#f8fafc", alpha=0.9,
                     xytext=(4, 4), textcoords="offset points")
    ax3.axvline(CFG["novelty"]["high_novelty_threshold"],
                color=WARNING, lw=1, ls="--", alpha=0.7, label="High-novelty threshold")
    ax3.set_xlabel("Title Novelty Score", fontsize=9)
    ax3.set_ylabel("Emergence Score", fontsize=9)
    ax3.set_title("Title Novelty vs Emergence Score", fontsize=10, color="#38bdf8")
    ax3.legend(fontsize=7)

# ── Plot 4: Cluster timeline — posting volume heat ───────────────────────────
ax4 = axes[1, 1]
if len(timeline_df):
    # Pivot: rows=tracked_cluster_id (top 10 by total postings), cols=quarter
    top_ids = (
        timeline_df.groupby("tracked_cluster_id")["posting_count"].sum()
        .nlargest(10).index.tolist()
    )
    pivot = (
        timeline_df[timeline_df["tracked_cluster_id"].isin(top_ids)]
        .pivot_table(index="tracked_cluster_id", columns="quarter", values="posting_count", fill_value=0)
    )
    # Label clusters by candidate role name from labeled df
    name_map = dict(zip(labeled["tracked_cluster_id"], labeled["candidate_role_name"]))
    pivot.index = [name_map.get(i, str(i))[:22] for i in pivot.index]
    sns.heatmap(pivot, ax=ax4, cmap="Blues", linewidths=0.3,
                cbar_kws={"label": "Postings"}, annot=True, fmt=".0f", annot_kws={"size": 6})
    ax4.set_title("Cluster Posting Volume Over Time (Top 10)", fontsize=10, color="#38bdf8")
    ax4.set_xlabel("Quarter", fontsize=9)
    ax4.set_ylabel("", fontsize=9)
    ax4.tick_params(axis="x", rotation=45, labelsize=7)
    ax4.tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "data" / "outputs" / "emerging_roles_charts.png",
            dpi=130, bbox_inches="tight", facecolor="#0f1923")
plt.show()
print("✅ Charts saved: data/outputs/emerging_roles_charts.png")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 13 — Save Predictions
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 13 — Save Predictions"))

nb.cells.append(code("""
OUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Primary output (clean summary) ───────────────────────────────────────────
PRIMARY_OUT = OUT_DIR / "emerging_role_predictions.csv"
predictions_df.to_csv(PRIMARY_OUT, index=False)
print(f"✅ Primary predictions saved  : {PRIMARY_OUT}")

# ── Detailed output (all columns from forecasted_clusters) ───────────────────
DETAIL_OUT = OUT_DIR / "emerging_role_predictions_detailed.csv"
forecasted.to_csv(DETAIL_OUT, index=False)
print(f"✅ Detailed predictions saved : {DETAIL_OUT}")

print(f"\\n   Primary shape : {predictions_df.shape}")
print(f"   Detailed shape: {forecasted.shape}")
print()
print("  Primary columns:")
for c in predictions_df.columns:
    print(f"    {c}")
"""))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 14 — Reproducibility / Validation
# ─────────────────────────────────────────────────────────────────────────────
nb.cells.append(md("## Section 14 — Reproducibility & Validation"))

nb.cells.append(code("""
print("=" * 70)
print("  PIPELINE B VALIDATION CHECKS")
print("=" * 70)

issues = []

# 1. No duplicate cluster IDs
dup_check = predictions_df["Cluster ID"].duplicated().sum() if "Cluster ID" in predictions_df.columns else 0
status = "✅ PASS" if dup_check == 0 else f"❌ FAIL ({dup_check} duplicates)"
print(f"  Duplicate cluster IDs    : {status}")
if dup_check > 0:
    issues.append("Duplicate cluster IDs found")

# 2. No invalid emergence scores
invalid_scores = ((predictions_df["Emergence Score"] < 0) | (predictions_df["Emergence Score"] > 1)).sum()
status = "✅ PASS" if invalid_scores == 0 else f"❌ FAIL ({invalid_scores} out-of-range)"
print(f"  Scores in [0,1]          : {status}")
if invalid_scores > 0:
    issues.append("Scores outside [0,1]")

# 3. No missing role labels
missing_labels = predictions_df["Emerging Role"].isna().sum() + (predictions_df["Emerging Role"] == "Unknown Role").sum()
status = "✅ PASS" if missing_labels == 0 else f"⚠️  WARN ({missing_labels} missing)"
print(f"  Role labels populated    : {status}")

# 4. Prediction counts
n_preds     = len(predictions_df)
n_high      = (predictions_df["Confidence"] == "High").sum()
n_medium    = (predictions_df["Confidence"] == "Medium").sum()
n_low       = (predictions_df["Confidence"] == "Low").sum()
n_emerging  = (predictions_df["Classification"] == "Emerging").sum()
n_growing   = (predictions_df["Classification"] == "Growing").sum()
n_persistent = (predictions_df["Classification"] == "Persistent").sum()
n_novel     = (predictions_df["Classification"] == "Novel").sum()
n_sparse    = (predictions_df["Classification"] == "Sparse / Low-Evidence").sum()
status = "✅ PASS" if n_preds > 0 else "❌ FAIL"
print(f"  Total predictions        : {n_preds}  {status}")

# 5. Date range coverage
date_range = f"{jobs_df['month'].min()} → {jobs_df['month'].max()}"
print(f"  Data date range          : {date_range}")

# 6. Cross-reference with existing pipeline output
EXISTING_PATH = PROJECT_ROOT / "data" / "outputs" / "emerging_role_discoveries.csv"
if EXISTING_PATH.exists():
    existing = pd.read_csv(EXISTING_PATH)
    n_exist = len(existing[existing["source_type"] == "emerging_cluster"]) if "source_type" in existing.columns else len(existing)
    match = "✅ MATCH" if n_preds == n_exist else f"ℹ️  DIFFERS  (notebook={n_preds}, prior run={n_exist})"
    print(f"  Cross-ref with prior run : {match}")
    if n_preds != n_exist:
        print(f"     ↳ Difference = {n_preds - n_exist}. Embeddings are cached, HDBSCAN is "
              f"non-deterministic at boundaries — minor count variation is normal.")

print()
print("─" * 70)
print()
print("  Confidence distribution:")
print(f"    High   : {n_high}  clusters")
print(f"    Medium : {n_medium}  clusters")
print(f"    Low    : {n_low}  clusters")
print()
print("  Classification distribution:")
print(f"    Emerging            : {n_emerging}")
print(f"    Growing             : {n_growing}")
print(f"    Persistent          : {n_persistent}")
print(f"    Novel               : {n_novel}")
print(f"    Sparse/Low-Evidence : {n_sparse}")
print()
if issues:
    print("  ⚠️  ISSUES FOUND:")
    for i in issues:
        print(f"    - {i}")
else:
    print("  ✅ All validation checks passed")
"""))

nb.cells.append(code("""
# ── Final Summary ─────────────────────────────────────────────────────────────
top_role = predictions_df.iloc[0]["Emerging Role"] if len(predictions_df) else "N/A"
top_score = float(predictions_df.iloc[0]["Emergence Score"]) if len(predictions_df) else 0.0

print()
print("╔══════════════════════════════════════════════════════════════════╗")
print("║              EMERGING ROLE DISCOVERY COMPLETE                   ║")
print("╠══════════════════════════════════════════════════════════════════╣")
print(f"║  Data source                   : Adzuna + LinkedIn IT postings  ║")
print(f"║  Total IT postings ingested    : {len(jobs_df):>6,}                        ║")
print(f"║  Unique semantic titles        : {len(unique_titles):>6,}                        ║")
print(f"║  Embedding dimensions          : 384 (all-MiniLM-L6-v2)       ║")
print(f"║  Clusters discovered (total)   : {sum(len(quarterly_results[q][0]['cluster_id'].unique())-1 for q in quarterly_results):>6}                        ║")
print(f"║  Temporally tracked clusters   : {timeline_df['tracked_cluster_id'].nunique():>6}                        ║")
print(f"║  Persistent candidates (≥2Q)   : {len(persistent):>6}                        ║")
print(f"║  Noise-filtered predictions    : {len(predictions_df):>6}                        ║")
print(f"║  Top predicted role            : {top_role[:30]:<30}  ║")
print(f"║  Top emergence score           : {top_score:.4f}                          ║")
print(f"╠══════════════════════════════════════════════════════════════════╣")
print(f"║  PRIMARY OUTPUT : data/outputs/emerging_role_predictions.csv   ║")
print(f"║  DETAILED OUTPUT: data/outputs/emerging_role_predictions_detailed.csv ║")
print(f"║  CHARTS         : data/outputs/emerging_roles_charts.png       ║")
print("╚══════════════════════════════════════════════════════════════════╝")
"""))

# Write the notebook
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Notebook written to: {NB_PATH}")
print(f"   Cells: {len(nb.cells)}")
