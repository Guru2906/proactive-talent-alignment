# PROJECT STRUCTURE
## Project: Proactive Talent Alignment for Sustainable Employment
**Version:** 1.0.0 | **Date:** 2026-09-01

---

## Final Clean Structure

    capstoneee/
    |
    |-- AUDIT_BEFORE_10X_UPGRADE.md    <- Phase 0 ground-truth audit
    |-- README.md                       <- Professional project overview
    |-- requirements.txt                <- Pinned production dependencies
    |-- run_pipeline.py                 <- Master end-to-end orchestrator
    |-- start_app.bat                   <- Windows: one-command startup
    |-- start_app.sh                    <- Unix/Mac: one-command startup
    |
    |-- config/
    |   |-- pipeline_config.yaml        <- ALL pipeline parameters (no hardcoded paths)
    |   |-- __init__.py
    |
    |-- data/
    |   |-- DATASET_REGISTRY.csv        <- Machine-readable dataset governance registry
    |   |-- raw/                        <- Source datasets (Adzuna, LinkedIn, Naukri, O*NET)
    |   |-- features/                   <- Engineered feature store + embedding caches
    |   |   |-- master_role_month_features.csv  <- 936x65 training feature matrix
    |   |   |-- title_embeddings.npy            <- SBERT embedding cache (~200K titles)
    |   |   |-- canonical_embeddings.npy        <- SBERT embeddings for 74 canonical roles
    |   |   |-- title_id_map.csv
    |   |-- mappings/                   <- Role + skill canonical mappings
    |   |   |-- canonical_roles.csv     <- 74 known IT role names
    |   |   |-- role_mapping.csv        <- 2000+ raw->canonical role mapping
    |   |   |-- skill_mapping.csv       <- 900+ raw->canonical skill mapping
    |   |-- questionnaire/
    |   |   |-- questionnaire.json      <- 48-question adaptive assessment bank
    |   |-- role_catalog/               <- PRODUCTION: consumed by API at inference
    |   |   |-- emerging_roles.json     <- 60-role catalog (39 schema fields)
    |   |   |-- emerging_roles.csv
    |   |-- role_skill_intelligence.csv <- Enriched role-skill matrix (2660 rows)
    |
    |-- models/
    |   |-- final_model.joblib          <- Trained LightGBM (Pipeline A) -- DO NOT DELETE
    |   |-- final_config.json           <- Model metadata (version, cutoff, features)
    |
    |-- src/                            <- PRODUCTION ML source modules
    |   |-- clustering/
    |   |   |-- hdbscan_discovery.py    <- HDBSCAN per-quarter discovery
    |   |   |-- cluster_tracking.py     <- Cross-quarter centroid tracking
    |   |   |-- cluster_labeling.py     <- Cluster naming from top titles
    |   |-- embeddings/
    |   |   |-- sentence_embeddings.py  <- SBERT embedding + caching
    |   |-- emerging/
    |   |   |-- emergence_score.py      <- 7-component emergence scoring
    |   |-- evaluation/
    |   |   |-- leakage_tests.py        <- Temporal + target leakage checks
    |   |   |-- emerging_backtest.py    <- Historical emergence backtest
    |   |   |-- ablation.py             <- Discovery method ablation
    |   |-- explainability/
    |   |   |-- shap_analysis.py        <- SHAP TreeExplainer for LightGBM
    |   |-- forecasting/
    |   |   |-- cluster_forecaster.py   <- Holt-Damped cluster growth forecast
    |   |-- preprocessing/
    |   |   |-- job_cleaning.py         <- Title normalization, deduplication
    |   |-- ranking/
    |   |   |-- final_ranker.py         <- Unified known+emerging ranking
    |   |-- recommendation/
    |   |   |-- profile_engine.py       <- User profile vector construction
    |   |   |-- role_matcher.py         <- Multi-factor recommendation scoring
    |   |   |-- skill_gap.py            <- Skill gap analysis + classification
    |   |   |-- roadmap.py              <- Learning roadmap generation
    |   |-- skills/
    |       |-- skill_engine.py         <- Role-skill matrix builder + velocity
    |
    |-- backend/                        <- PRODUCTION FastAPI API layer
    |   |-- main.py                     <- 12-endpoint REST API
    |   |-- data_loader.py              <- Role catalog loading
    |   |-- questionnaire_engine.py     <- Adaptive questionnaire engine
    |   |-- models.py                   <- Pydantic request/response schemas
    |   |-- questionnaire.json          <- Questionnaire bank (copy for API access)
    |
    |-- frontend/                       <- PRODUCTION Streamlit UI
    |   |-- app.py                      <- 8-page multi-stage UI
    |   |-- components/
    |       |-- charts.py               <- Plotly chart renderers
    |       |-- cards.py                <- Role card components
    |       |-- roadmap_display.py      <- Skill gap + roadmap display
    |
    |-- outputs/                        <- Generated pipeline outputs
    |   |-- forecasts/
    |   |   |-- known_role_forecasts.csv            <- 39 roles, 6M+12M
    |   |   |-- top_emerging_roles_6m.csv           <- Top 20 by 6M growth
    |   |   |-- top_emerging_roles_12m.csv          <- Top 20 by 12M growth
    |   |   |-- unified_ranking.csv                 <- Combined known+emerging
    |   |-- emerging_roles/
    |   |   |-- emerging_roles.json                 <- 60-role catalog (copy)
    |   |   |-- emerging_role_discoveries.csv       <- 21 HDBSCAN clusters
    |   |   |-- emerging_role_predictions.csv
    |   |   |-- emerging_role_predictions_detailed.csv
    |   |   |-- candidate_emerging_roles.csv
    |   |-- explanations/
    |   |   |-- role_explanations.json              <- SHAP-derived text per role
    |   |   |-- shap_global_importance.csv
    |   |   |-- shap_per_role_drivers.csv
    |   |   |-- feature_importance.csv
    |   |-- reports/
    |       |-- role_skill_gap_analysis.csv
    |       |-- ablation_forecasting_models.csv
    |       |-- ablation_discovery_pipeline.csv
    |       |-- leakage_audit_summary.csv
    |       |-- emerging_roles_charts.png
    |   |-- known_role_model_comparison.csv         <- Phase 3 benchmark
    |   |-- known_role_walkforward_splits.csv       <- Walk-forward split details
    |   |-- emerging_discovery_ablation.csv         <- Phase 4B ablation
    |   |-- emerging_role_backtest.csv              <- Phase 5 backtest metrics
    |   |-- emergence_score_ablation.csv            <- Phase 6 weight ablation
    |
    |-- docs/                           <- ALL documentation (single entry point)
    |   |-- FINAL_SYSTEM_ARCHITECTURE.md    <- Phase 1: definitive architecture
    |   |-- DATA_USAGE_GUIDE.md             <- Phase 2: dataset classification
    |   |-- KNOWN_ROLE_FORECASTING_REPORT.md <- Phase 3: model validation
    |   |-- EMERGING_ROLE_BACKTEST_REPORT.md <- Phase 5: emergence backtest
    |   |-- ARCHITECTURE.md                 <- Original architecture summary
    |   |-- DATASET_GUIDE.md                <- Dataset quick reference
    |   |-- MODEL_GUIDE.md                  <- Model quick reference
    |   |-- RECOMMENDATION_GUIDE.md         <- Recommendation engine guide
    |   |-- PROJECT_STRUCTURE.md            <- This file
    |   |-- RECOMMENDATION_ENGINE_SPEC.md   <- Detailed recommendation spec
    |   |-- USER_APPLICATION_ARCHITECTURE.md <- UI flow documentation
    |
    |-- notebooks/                      <- Research + validation notebooks
    |   |-- 01_data_analysis.ipynb      <- Data exploration
    |   |-- 02_role_forecasting.ipynb   <- Pipeline A analysis
    |   |-- 03_emerging_role_discovery.ipynb <- Pipeline B analysis
    |   |-- 04_skill_analysis.ipynb     <- Role-skill intelligence
    |   |-- 05_recommendation_analysis.ipynb <- Recommendation system
    |   |-- build_notebook.py           <- Notebook content generator
    |
    |-- tests/                          <- Automated test suite (20 tests)
    |   |-- test_api.py                 <- FastAPI endpoint tests
    |   |-- test_questionnaire.py       <- Questionnaire engine tests
    |   |-- test_recommendation_engine.py <- Recommender tests
    |   |-- test_role_catalog.py        <- Catalog schema tests
    |
    |-- scripts/                        <- Pipeline utility scripts
    |   |-- build_role_catalog.py       <- PRODUCTION: builds 60-role catalog
    |   |-- train_models.py             <- Model training script
    |   |-- generate_phase1_to_8_artifacts.py <- Documentation generator
    |   |-- ... (audit + prep scripts)
    |
    |-- step1/                          <- Original pipeline (preserved for reference)
    |   |-- scripts/                    <- Original pipeline scripts
    |   |-- data/cleaned/               <- Cleaned source datasets (PRESERVED)
    |   |-- outputs/                    <- Original pipeline outputs
    |   |-- models/                     <- Original baseline model (260KB)
    |   |-- reports/                    <- Original audit reports
    |
    |-- step1_enhanced/                 <- Enhanced pipeline (PRODUCTION TRAINING SOURCE)
    |   |-- scripts/                    <- Enhanced feature engineering scripts
    |   |-- outputs/                    <- Audited outputs: leakage log, backtest, skill matrix
    |
    |-- dashboard/                      <- Interactive HTML dashboard generator
    |-- data_governance/                <- Data lineage + governance documents
    |-- validation/                     <- Additional validation CSVs
    |-- archive/                        <- All legacy + obsolete components
    |   |-- legacy/                     <- Moved during Phase 20 cleanup

---

## Key Principles

1. NEVER modify models/final_model.joblib without documented justification.
2. ALL paths in Python code use Path(__file__).resolve().parents[N] relative paths.
3. ALL pipeline parameters live in config/pipeline_config.yaml.
4. Notebooks are for research/exploration -- the application NEVER executes notebooks.
5. The API (backend/) reads from data/role_catalog/ at startup, not from notebooks.
6. tests/ must pass before any pull request or submission.

---

## How to Run

    # Install dependencies
    pip install -r requirements.txt

    # Option 1: One-command startup (Windows)
    start_app.bat

    # Option 2: One-command startup (Unix/Mac)
    bash start_app.sh

    # Option 3: Manual
    uvicorn backend.main:app --reload --port 8000
    streamlit run frontend/app.py

    # Run full pipeline (regenerate all outputs)
    python run_pipeline.py

    # Run tests
    pytest tests/ -v

    # Run specific pipeline stages
    python run_pipeline.py --known-only       # Pipeline A (LightGBM)
    python run_pipeline.py --discovery-only   # Pipeline B (SBERT+HDBSCAN)
    python run_pipeline.py --backtest         # Emergence backtest
    python run_pipeline.py --validate-only    # Validation + leakage tests
    python run_pipeline.py --shap-only        # SHAP analysis only
