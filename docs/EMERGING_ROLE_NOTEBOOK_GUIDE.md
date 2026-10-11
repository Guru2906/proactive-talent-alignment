# Emerging Roles Prediction — Notebook Guide

**Notebook:** [`notebooks/emerging_roles_prediction.ipynb`](../notebooks/emerging_roles_prediction.ipynb)  
**Output:** `data/outputs/emerging_role_predictions.csv`

---

## Purpose

This notebook provides a transparent, cell-by-cell execution of **Pipeline B** — the unsupervised semantic discovery component of the Proactive IT Role & Skill Emergence Forecaster.

Pipeline B answers a fundamentally different question from Pipeline A:

> **Pipeline A** asks: *"How much demand will there be for these 39 known IT roles in the next 6–12 months?"*  
> **Pipeline B** asks: *"What NEW IT job roles are appearing in the market that don't fit into the 39 known roles?"*

The notebook is the **primary interface** for inspecting, reproducing, and presenting Pipeline B results.

---

## Quick Start

```bash
# From project root
cd c:\Users\Harshavardhana\Downloads\capstoneee
jupyter notebook notebooks/emerging_roles_prediction.ipynb
```

Or use VS Code's Jupyter extension to open the `.ipynb` file directly.

**Run all cells top-to-bottom** (Kernel → Restart & Run All).

The notebook re-uses cached embeddings (≈8 MB `.npy` file) and cached cluster data by default. A full fresh run from scratch takes approximately 5–15 minutes depending on GPU availability.

---

## Notebook Structure (14 Sections)

| Section | Title | Description |
|---------|-------|-------------|
| 1 | **Overview** | Architecture diagram and Pipeline A vs B comparison |
| 2 | **Imports & Config** | Libraries + `config/pipeline_config.yaml` parameter display |
| 3 | **Load Raw Job Data** | Load Adzuna/LinkedIn IT postings and cleaned dataset |
| 4 | **Job Cleaning** | Preprocessing via `src/preprocessing/job_cleaning.py` with token tests |
| 5 | **SBERT Embeddings** | Load or generate 384-dim sentence embeddings (cached by default) |
| 6 | **HDBSCAN Discovery** | Per-quarter variable-density clustering; shows per-quarter statistics |
| 7 | **Temporal Tracking** | Cross-quarter cluster persistence tracking; saves `cluster_timeline.csv` |
| 8 | **Cluster Labeling** | Automated role-name generation + canonical similarity scoring |
| 9 | **Emergence Scoring** | 7-component weighted emergence score computation |
| 10 | **Predictions** | Build final ranked `predictions_df` table |
| 11 | **Top Candidates** | Human-readable display of Top 10 (and 11–20) with metrics |
| 12 | **Visualizations** | 4-panel chart: scores, persistence, novelty, cluster timeline heatmap |
| 13 | **Save Predictions** | Write `emerging_role_predictions.csv` and detailed CSV |
| 14 | **Validation** | Automated checks + cross-reference with prior pipeline run |

---

## Output Files

| File | Description |
|------|-------------|
| `data/outputs/emerging_role_predictions.csv` | **Primary output** — clean ranked table of role candidates |
| `data/outputs/emerging_role_predictions_detailed.csv` | All columns including all 7 component scores |
| `data/outputs/emerging_roles_charts.png` | 4-panel visualization (saved automatically) |
| `data/clusters/cluster_timeline.csv` | Per-quarter cluster tracking history |
| `data/clusters/persistent_clusters.csv` | Clusters that survived persistence filtering |
| `data/clusters/scored_clusters.csv` | Clusters with all emergence score components |
| `data/clusters/forecasted_clusters.csv` | Scored clusters + next-quarter volume forecasts |

---

## Column Reference — `emerging_role_predictions.csv`

| Column | Type | Description |
|--------|------|-------------|
| `Rank` | int | Final emergence rank (1 = highest) |
| `Emerging Role` | str | Auto-generated candidate role name from SBERT centroids |
| `Cluster ID` | str | Internal tracked cluster identifier |
| `Emergence Score` | float [0,1] | Weighted composite score (see formula below) |
| `Confidence` | str | High / Medium / Low |
| `Demand Growth` | float [0,1] | Component: trend in posting count over time |
| `Persistence (Qtrs)` | int | Number of quarterly windows cluster appeared in |
| `Title Novelty` | float [0,1] | 1 − cosine_similarity(cluster_centroid, nearest_canonical) |
| `Market Breadth` | float [0,1] | Breadth of companies and sectors posting this role |
| `Historical Volume` | int | Total postings across all observed quarters |
| `Forecast Volume (Next Q)` | int | Projected posting count for the next quarter |
| `Forecast Growth %` | float | Projected % growth vs. most recent quarter |
| `Forecast Method` | str | linear / exponential / median (depends on n_obs_quarters) |
| `Classification` | str | Emerging / Growing / Persistent / Novel / Sparse |
| `Closest Known Role` | str | Nearest Pipeline A canonical role (by cosine similarity) |
| `Representative Titles` | str | Top 3 raw job titles that define this cluster |
| `First Seen` | str | Earliest quarter (e.g., 2024-Q3) |
| `Last Seen` | str | Most recent quarter observed |

---

## Emergence Score Formula

```
emergence_score = w_growth    × demand_growth_score
                + w_skill_vel × skill_velocity_score
                + w_skill_nov × skill_novelty_score
                + w_title_nov × title_novelty_score
                + w_breadth   × market_breadth_score
                + w_persist   × persistence_score
                + w_tech      × technology_adoption_score
```

All components are normalized to **[0, 1]**. Weights are defined in `config/pipeline_config.yaml` under `emergence_weights`.

---

## Interpreting Results

### Confidence Labels

| Confidence | Meaning |
|------------|---------|
| **High** | Emergence score ≥ 0.5 AND persistence ≥ 3 quarters AND ≥ 30 unique companies |
| **Medium** | Emergence score ≥ 0.35 OR persistence ≥ 2 quarters |
| **Low** | Fewer data points — treat as exploratory signal |

### Classification Labels

| Classification | Criteria |
|----------------|----------|
| **Emerging** | Emergence score ≥ 0.55 AND novelty in {high, medium} |
| **Growing** | Emergence score ≥ 0.40 AND persistence ≥ 3 quarters |
| **Persistent** | Persistence ≥ 4 quarters (stable, not necessarily novel) |
| **Novel** | High novelty score but shorter persistence |
| **Sparse / Low-Evidence** | Persistence ≤ 2 quarters AND total postings < 40 |
| **Candidate** | Does not meet any stronger classification criteria |

### Title Novelty Score

- Score **> 0.6** → role is semantically distant from all 39 canonical roles → genuinely emerging
- Score **0.3–0.6** → role is a variant / specialization of an existing role
- Score **< 0.3** → role is likely already captured by a canonical role (potentially a duplicate signal)

---

## Caching and Reproducibility

| Component | Cache File | Re-generated If |
|-----------|-----------|-----------------|
| SBERT embeddings | `data/embeddings/title_embeddings.npy` | `force_recompute=True` in Section 5 |
| ID map | `data/embeddings/title_id_map.csv` | With embeddings |
| Canonical embeddings | `data/embeddings/canonical_embeddings.npy` | With embeddings |
| Cleaned jobs | `data/clusters/cleaned_it_jobs.csv` | If missing |
| Cluster timeline | `data/clusters/cluster_timeline.csv` | Every notebook run (from HDBSCAN output) |
| Scored clusters | `data/clusters/scored_clusters.csv` | Every notebook run |
| Forecasted clusters | `data/clusters/forecasted_clusters.csv` | Every notebook run |

> **Note:** HDBSCAN is inherently non-deterministic at cluster boundaries. Minor differences in cluster count (±1–2) between runs are expected and normal. The embedding cache eliminates the most expensive source of variance (transformer inference).

---

## Relationship to Pipeline A

Pipeline B results are **complementary** to, not competitive with, Pipeline A:

- If a Pipeline B cluster has **Title Novelty < 0.3** and `Closest Known Role` is one of the 39 canonical roles → this is a confirmation signal for that canonical role.
- If a Pipeline B cluster has **Title Novelty > 0.6** and **Confidence = High** → this is a genuinely new role that is not covered by Pipeline A.
- Both outputs are merged in `data/outputs/unified_ranking.csv` by `src/ranking/final_ranker.py`.

---

## Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: src` | Run from wrong directory | Always run from project root |
| `FileNotFoundError: config/pipeline_config.yaml` | Wrong working directory | Same fix |
| Kernel dies during SBERT encoding | RAM exhausted | Set `force_recompute=False` in Section 5 |
| Different cluster count vs prior run | HDBSCAN non-determinism | Expected — see Caching note above |
| Heatmap axis labels wrong | Cluster label merge issue | Re-run Section 8 before Section 12 |

---

## Author / Pipeline Reference

- **Project:** Proactive IT Role & Skill Emergence Forecaster
- **Pipeline B Implementation:** `src/` modules (job_cleaning, sentence_embeddings, hdbscan_discovery, cluster_tracking, cluster_labeling, emergence_score, cluster_forecaster)
- **Configuration:** `config/pipeline_config.yaml`
- **Technical Report:** `MASTER_TECHNICAL_REPORT.md`
