# Proactive Talent Alignment for Sustainable Employment

> **An end-to-end ML system that forecasts future IT role demand, discovers emerging career patterns, and delivers personalized career recommendations with skill-gap analysis and learning roadmaps.**

---

## System Architecture

```
RAW DATA (Adzuna + LinkedIn + Naukri + O*NET)
    │
    ├── PIPELINE A: LightGBM Known-Role Forecasting
    │   └── 39 IT roles × 5 feature layers × walk-forward validation
    │
    └── PIPELINE B: SBERT + HDBSCAN Emerging-Role Discovery
        └── 12 quarterly windows → cluster tracking → Holt-Damped growth forecast
                        │
              ROLE CATALOG (60 roles)
                        │
         ┌──────────────┴──────────────┐
         │    RECOMMENDATION ENGINE    │
         │  BAIO questionnaire         │
         │  Multi-factor scoring       │
         │  Skill-gap analysis         │
         │  Learning roadmap           │
         └──────────────┬──────────────┘
                        │
         ┌──────────────┴──────────────┐
         │   SQLITE APPLICATION DB     │
         │  (Users, Sessions, Answers, │
         │   Snapshots & Feedback)     │
         └──────────────┬──────────────┘
                        │
         FastAPI (18 endpoints) ─── Streamlit UI (9 pages)
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Known-Role MAE | 1.13 bps |
| Known-Role NDCG@10 | 0.977 |
| Known-Role P@10 | 0.967 |
| Emergence Detection P@5 | 0.80 (BT_2025Q1) |
| Emergence Lead Time | avg 2.3 quarters |
| Leakage Audit | 0/58 features flagged |
| Application Database | **SQLite + SQLAlchemy ORM** |
| Automated Tests | **32/32 PASS** |
| API Endpoints | 18 |
| Role Catalog | 60 roles |
| Feature Store | 936 × 65 |

---

## Persistent Application Database (SQLite)

The application uses **SQLite** (`data/application/career_recommendation.db`) via **SQLAlchemy ORM** to persist user data, questionnaire progression, assessment histories, recommendation snapshots, and user feedback.

> [!NOTE]
> **ML Data Isolation**: Raw datasets, training matrices, SBERT embeddings, HDBSCAN cluster files, and LightGBM model weights remain strictly file- and artifact-based. SQLite is solely used for user and application state. For full architectural details, see [DATABASE_ARCHITECTURE.md](DATABASE_ARCHITECTURE.md).

### Tables:
- `users`: User profiles, declared background, skills, and interests.
- `assessment_sessions`: Session lifecycle, BAIO scores, and archetype classification.
- `assessment_answers`: Item-level responses and trait weights for historical reconstruction.
- `role_recommendations`: Generated recommendation rankings and match scores.
- `recommendation_snapshots`: Immutable, versioned historical snapshots of recommendations.
- `feedback`: User ratings (1–5) and qualitative feedback.

### Database Initialization:
```bash
python -m backend.init_db
```

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+ required)
pip install -r requirements.txt

# 2. Start both services (Windows)
start_app.bat

# 2. Start both services (Unix/Mac)
bash start_app.sh

# Or manually:
uvicorn backend.main:app --reload --port 8000
streamlit run frontend/app.py --server.port 8501
```

- **API docs**: http://localhost:8000/docs  
- **UI**: http://localhost:8501

---

## Run the ML Pipeline

```bash
# Full pipeline (forecasting + discovery + catalog build)
python run_pipeline.py

# Specific stages
python run_pipeline.py --known-only       # Pipeline A only (LightGBM)
python run_pipeline.py --discovery-only   # Pipeline B only (SBERT+HDBSCAN)
python run_pipeline.py --backtest         # Historical emergence backtest
python run_pipeline.py --validate-only    # Leakage tests + assertions
python run_pipeline.py --shap-only        # SHAP explanations only
python run_pipeline.py --dashboard        # Regenerate HTML dashboard

# Run tests
pytest tests/ -v
```

---

## System Components

### Pipeline A — Known Role Forecasting

| Component | Detail |
|-----------|--------|
| Model | LightGBM (`models/final_model.joblib`) |
| Features | 54 features across 5 layers (Adzuna temporal, LinkedIn cross-sectional, Skill adoption, Era novelty, O\*NET) |
| Training data | `data/features/master_role_month_features.csv` (936 × 65) |
| Validation | Walk-forward: BT_2024Q1, BT_2024Q3, BT_2025Q1 |
| Results | MAE=1.13, RMSE=3.10, P@10=0.967, NDCG@10=0.977 |
| Explainability | SHAP TreeExplainer (global + per-role) |

### Pipeline B — Emerging Role Discovery

| Component | Detail |
|-----------|--------|
| Embedding | SBERT `all-MiniLM-L6-v2` (384-dim, cached in `data/features/title_embeddings.npy`) |
| Clustering | HDBSCAN (min_cluster_size=15, min_samples=5, per quarter) |
| Tracking | Centroid cosine similarity > 0.75 across 12 quarters |
| Scoring | 7-component emergence score (demand_growth=25%, novelty=15%, etc.) |
| Forecasting | Holt-Damped exponential smoothing (statsmodels) |
| Backtest | P@5=0.80, NDCG@10=0.836 at BT_2025Q1 |

### Recommendation Engine

| Component | Detail |
|-----------|--------|
| Assessment | Adaptive 20-question BAIO questionnaire (48-question bank) |
| Profiling | Multi-vector: BAIO + interests + technical + skills + preferences |
| Scoring | 6-component weighted: Personality(20%) + Skill(20%) + Interest(15%) + WorkStyle(15%) + Market(15%) + Emergence(15%) |
| Output | Top 5 roles with match score, score breakdown, explanation |
| Skill Gap | Per-skill classification: already_strong / needs_improvement / missing / priority |
| Roadmap | 6-month phased learning path based on role-skill intelligence |

---

## Project Structure

```
capstoneee/
├── data/
│   ├── raw/                    # Source datasets
│   ├── features/               # Feature store + SBERT embedding cache
│   ├── mappings/               # Role + skill canonical mappings
│   ├── role_catalog/           # Production role catalog (60 roles)
│   └── role_skill_intelligence.csv
├── models/
│   └── final_model.joblib      # Trained LightGBM — DO NOT DELETE
├── src/                        # Production ML modules
│   ├── clustering/             # HDBSCAN discovery + tracking
│   ├── emerging/               # Emergence scoring
│   ├── forecasting/            # Cluster growth forecasting
│   ├── recommendation/         # Profile, matcher, skill-gap, roadmap
│   ├── explainability/         # SHAP analysis
│   └── evaluation/             # Backtest + leakage tests + ablation
├── backend/                    # FastAPI (12 endpoints)
├── frontend/                   # Streamlit (8 pages)
├── outputs/                    # All generated ML outputs
├── docs/                       # Complete documentation
├── tests/                      # 20 automated tests
├── config/pipeline_config.yaml # Central configuration
├── run_pipeline.py             # Master orchestrator
└── requirements.txt
```

See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) for the complete annotated structure.

---

## Documentation Index

| Document | Purpose |
|----------|---------|
| [`AUDIT_BEFORE_10X_UPGRADE.md`](AUDIT_BEFORE_10X_UPGRADE.md) | Ground-truth Phase 0 audit |
| [`FINAL_10_10_PROJECT_REPORT.md`](FINAL_10_10_PROJECT_REPORT.md) | Comprehensive project report |
| [`docs/FINAL_SYSTEM_ARCHITECTURE.md`](docs/FINAL_SYSTEM_ARCHITECTURE.md) | System architecture |
| [`docs/KNOWN_ROLE_FORECASTING_REPORT.md`](docs/KNOWN_ROLE_FORECASTING_REPORT.md) | Pipeline A validation |
| [`docs/EMERGING_ROLE_BACKTEST_REPORT.md`](docs/EMERGING_ROLE_BACKTEST_REPORT.md) | Pipeline B backtest |
| [`docs/DATA_USAGE_GUIDE.md`](docs/DATA_USAGE_GUIDE.md) | Dataset classification |
| [`data/DATASET_REGISTRY.csv`](data/DATASET_REGISTRY.csv) | 25-dataset registry |
| [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md) | Project structure guide |

---

## Key Output Files

| Output | Location | Description |
|--------|----------|-------------|
| Role Catalog | `data/role_catalog/emerging_roles.json` | 60 roles, 39 fields — API source |
| Known Forecasts | `outputs/forecasts/known_role_forecasts.csv` | 39 roles, 6M+12M |
| Emerging Roles | `outputs/emerging_roles/emerging_role_discoveries.csv` | 21 HDBSCAN clusters |
| SHAP Drivers | `outputs/explanations/shap_per_role_drivers.csv` | Per-role LightGBM explanations |
| Model Comparison | `outputs/known_role_model_comparison.csv` | Benchmark results |
| Backtest Results | `outputs/emerging_role_backtest.csv` | Emergence detection metrics |
| Skill Intelligence | `data/role_skill_intelligence.csv` | 2,660 role-skill records |

---

## Validation Checklist

- [x] All datasets documented (`data/DATASET_REGISTRY.csv`)
- [x] Zero temporal leakage (58-feature audit: 100% SAFE/NONE)
- [x] Known-role forecasting validated (walk-forward, 3 cutoffs)
- [x] Emerging-role discovery validated (P@5=0.80, NDCG@10=0.836)
- [x] Historical backtest with no future data
- [x] Emergence score ablation (7 weight configurations tested)
- [x] Discovery method ablation (TF-IDF/KMeans/HDBSCAN compared)
- [x] Role-skill intelligence layer built (2,660 records)
- [x] Questionnaire produces structured user profiles
- [x] Personalized top-5 recommendations working
- [x] Every recommendation has explanation
- [x] Skill gaps generated per recommendation
- [x] Career roadmap generated per recommendation
- [x] UI displays complete 8-page journey
- [x] API separates frontend from ML logic (12 endpoints)
- [x] All outputs versioned with metadata
- [x] 20/20 automated tests passing
- [x] No hardcoded absolute paths
- [x] No fabricated claims or metrics

---

## Scientific Integrity

- **No future data** used during any training or backtesting phase
- **No fabricated metrics** — all numbers derived from actual computation
- **Confidence labels** explicitly communicate uncertainty (HIGH/MEDIUM/LOW)
- **Recommendation weights** documented as policy-based (no ground-truth user outcomes)
- **Leakage evidence** preserved in `step1_enhanced/outputs/audited_feature_leakage_log.csv`
- **Limitations** documented honestly in `FINAL_10_10_PROJECT_REPORT.md § 21`

---

## Technologies

| Component | Technology |
|-----------|-----------|
| Known-Role Forecasting | LightGBM 4.6.0 |
| Job Title Embedding | SBERT `all-MiniLM-L6-v2` (sentence-transformers 5.1.2) |
| Cluster Discovery | HDBSCAN 0.8.44 |
| Cluster Forecasting | Holt-Damped (statsmodels 0.14.6) |
| Explainability | SHAP 0.52.0 |
| API | FastAPI 0.136.3 + Uvicorn 0.49.0 |
| UI | Streamlit 1.58.0 + Plotly 6.3.1 |
| ML Libraries | scikit-learn 1.9.0, pandas 2.3.1, numpy 2.3.5 |
| Config | PyYAML 6.0.3 |
| Testing | pytest 8.4.2 |
