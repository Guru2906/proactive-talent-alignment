# FINAL SYSTEM ARCHITECTURE
## Project: Proactive Talent Alignment for Sustainable Employment
**Version:** 1.0.0 | **Date:** 2026-09-01 | **Status:** Production-Ready

---

## System Overview

The system is a research-grade, end-to-end ML pipeline that:
1. Forecasts demand for 39 known IT roles (Pipeline A — LightGBM)
2. Discovers emerging IT role patterns from job posting semantics (Pipeline B — SBERT+HDBSCAN)
3. Profiles a user through an adaptive questionnaire (BAIO + multi-vector profiling)
4. Recommends the best 5 roles with multi-factor scoring, explanation, skill gap, and roadmap

---

## Component Architecture

### A. ROLE DISCOVERY (Pipeline B)

    RAW JOB POSTINGS (LinkedIn 1.6M postings.csv + Adzuna 17K)
        |
    src/preprocessing/job_cleaning.py
        |-- Title normalization (preserve: AI, ML, LLM, RAG, MLOps, etc.)
        |-- Stop-word removal (contract, temp, remote, hybrid)
        |-- Deduplication
        |
    src/embeddings/sentence_embeddings.py
        |-- Model: SBERT all-MiniLM-L6-v2 (384-dim)
        |-- Batch size: 512
        |-- Cache: data/features/title_embeddings.npy (~200K titles)
        |
    src/clustering/hdbscan_discovery.py
        |-- Per-quarter clustering (2022-Q4 to 2025-Q3, 12 quarters)
        |-- HDBSCAN: min_cluster_size=15, min_samples=5
        |-- Selection method: EOM (Excess of Mass)
        |-- Metric: euclidean on 384-dim SBERT vectors
        |
    src/clustering/cluster_tracking.py
        |-- Centroid cosine similarity > 0.75 to chain across quarters
        |-- Tracks: birth, growth, merge, split, disappearance
        |-- Output: cluster lifecycle metadata per tracked_cluster_id
        |
    src/emerging/emergence_score.py
        |-- 7-component weighted emergence score:
        |     demand_growth:        25%
        |     skill_velocity:       15%
        |     skill_novelty:        15%
        |     title_novelty:        15%
        |     market_breadth:       10%
        |     persistence:          10%
        |     technology_adoption:  10%
        |-- Noise filters: min 5 postings/quarter, min 2 companies, min 2 quarters
        |
    src/forecasting/cluster_forecaster.py
        |-- Holt Damped exponential smoothing (statsmodels)
        |-- Per-cluster growth forecast: 1-2 quarters ahead
        |-- Produces: forecast_next_quarter_postings, forecast_growth_pct

    OUTPUT: outputs/emerging_roles/emerging_role_discoveries.csv (21 filtered clusters)

---

### B. ROLE FORECASTING (Pipeline A)

    FEATURE STORE: data/features/master_role_month_features.csv
        |-- 936 rows x 65 columns (39 roles x 24 months)
        |-- Date range: 2023-11 to 2025-10
        |
        |-- Layer A (16 features): Adzuna temporal signal
        |   lag1, lag2, lag3, lag6, roll3m, roll6m, std3m, max6m, min6m
        |   mom1, mom3, mom6, slope6m, accel, sk_vel, frontier_ratio
        |
        |-- Layer B (16 features): LinkedIn cross-sectional
        |   li_posting_count, li_role_share_bps, li_unique_companies,
        |   li_unique_locations, li_seniority_entropy, li_skill_diversity,
        |   li_frontier_skill_ratio, li_company_breadth, li_unique_industries, ...
        |
        |-- Layer C (9 features): Skill adoption & AI tools
        |   skill_avg_adoption_2024, skill_max_adoption_2024, skill_ai_tool_count,
        |   skill_ai_tool_penetration, skill_cloud_platform_score,
        |   skill_multiyear_growth_rate, skill_novelty_index, ...
        |
        |-- Layer D (2 features): Historical era novelty
        |   era_novelty_score, is_new_era
        |
        |-- Layer E (7 features): O*NET taxonomy
        |   onet_matched, onet_skill_count, onet_tech_density,
        |   onet_top_languages_count, onet_emerging_relevance, onet_match_similarity
        |
    TARGET: target_t6 (demand share in basis points, 6 months ahead)
    SECONDARY: target_t12 (12 months ahead)

    MODEL: LightGBM (models/final_model.joblib)
        |-- n_estimators=200, learning_rate=0.03
        |-- num_leaves=25, max_depth=5
        |-- subsample=0.85, colsample_bytree=0.85
        |-- Walk-forward validated: BT_2024Q1, BT_2024Q3, BT_2025Q1
        |-- MAE=1.13, RMSE=3.10, P@10=0.967, NDCG@10=0.977
        |
    src/explainability/shap_analysis.py
        |-- SHAP TreeExplainer on LightGBM model
        |-- Global feature importance ranking
        |-- Per-role top driver extraction

    OUTPUT: outputs/forecasts/known_role_forecasts.csv (39 roles, 6M+12M)
    OUTPUT: outputs/explanations/shap_per_role_drivers.csv

---

### C. ROLE RANKING

    scripts/build_role_catalog.py
        |-- Merges Pipeline A + Pipeline B outputs
        |-- Applies confidence labeling (HIGH/MEDIUM/LOW)
        |-- Produces unified 60-role catalog with 39 schema fields
        |
    src/ranking/final_ranker.py
        |-- Normalizes demand scores across known + emerging roles
        |-- Computes final_priority_score (demand 60% + emergence 40%)
        |-- Produces unified_ranking.csv

    OUTPUT: data/role_catalog/emerging_roles.json (60 roles, PRODUCTION)
    OUTPUT: outputs/forecasts/unified_ranking.csv

---

### D. USER PROFILING

    backend/questionnaire_engine.py
        |-- 48-question adaptive bank, 20 questions shown per session
        |-- Stage 1 (6): Common archetype questions
        |-- Stage 2 (10): Archetype-targeted branching
        |-- Stage 3 (4): Reconfirmation / disambiguation
        |-- Produces: raw_answers dict keyed by question_id
        |
    src/recommendation/profile_engine.py
        |-- build_user_profile(raw_answers) -> UserProfile
        |-- Outputs multi-dimensional profile vectors:
        |     personality_vector: [Builder, Analyst, Innovator, Operator] (BAIO)
        |     interest_vector: [data, systems, creative, business, research, cloud]
        |     technical_vector: [ml, backend, frontend, devops, data_eng, security]
        |     experience_vector: [level, years_estimate]
        |     preference_vector: [work_style, team_size, autonomy, learning_speed]
        |     skill_vector: [python, sql, ml, cloud, data, devops, security, ...]
        |     profile_confidence: 0.0-1.0

---

### E. PERSONALIZED RECOMMENDATION

    src/recommendation/role_matcher.py
        |-- RoleMatcher.recommend(profile, catalog, top_n=5)
        |-- 6-component weighted scoring:
        |
        |   Component           Weight   Method
        |   ----------------    -------  ----------------------------------
        |   BAIO Personality     20%     Cosine similarity (BAIO vectors)
        |   Interest Alignment   15%     Jaccard + semantic overlap
        |   Work Style Match     15%     Preference key alignment
        |   Skill Alignment      20%     Jaccard (user skills vs role skills)
        |   Future Opportunity   15%     LightGBM forecast score (Pipeline A)
        |   Emergence Signal     15%     Emergence score (Pipeline B)
        |
        |-- NOTE: Weights are policy-based -- no ground-truth user outcomes available
        |-- Returns: Top 5 roles with match_score, component breakdown, explanation text

---

### F. SKILL-GAP ANALYSIS

    src/recommendation/skill_gap.py
        |-- compute_skill_gap(user_skills, role_required_skills, role_skill_matrix)
        |-- Classifies every role-required skill into:
        |     ALREADY_STRONG: user has skill + high proficiency claim
        |     NEEDS_IMPROVEMENT: user has skill but low proficiency
        |     MISSING: user lacks skill entirely
        |     PRIORITY: missing + high role_skill_probability + high skill_velocity
        |-- Produces: gap_score (0-1, lower = smaller gap)
        |
    src/recommendation/roadmap.py
        |-- generate_roadmap(skill_gaps, role_name)
        |-- Groups priority/missing skills into 6-month learning phases
        |-- Phase ordering based on skill dependency heuristics
        |-- Produces: list of {phase, skills, description} dicts

---

### G. EXPLAINABILITY

    Known Roles: SHAP TreeExplainer
        |-- Per-role top 5 SHAP drivers from LightGBM
        |-- Natural-language explanation template in build_role_catalog.py
        |
    Emerging Roles: Evidence-based explanation
        |-- Components: evidence volume, persistence_quarters, market_breadth
        |-- Confidence label (HIGH/MEDIUM/LOW) with reasoning
        |
    Recommendations: Component-score explanation
        |-- "Recommended because: strong BAIO fit + high skill alignment + growing demand"
        |-- Skill gap shown explicitly per recommendation

---

## Data Lineage

    Raw Sources          Cleaned           Features          Models         Outputs
    ---------------      -------           --------          ------         -------
    adzuna_*.csv    -->  adzuna_cleaned -->  master_role   --> LightGBM  --> known_role_forecasts.csv
    postings.csv    -->  linkedin_clean -->  _month_feat   --> SHAP      --> shap_per_role_drivers.csv
    naukri_*.csv    -->  naukri_cleaned -->  ures.csv      --> Holt      --> top_emerging_roles_6m.csv
    occupation.xlsx ---> (static join)  -->              -->            --> emerging_roles.json
    tech_occ_skill  ---> (static join)  -->              -->            --> role_skill_intelligence.csv
    questionnaire   ---> (static ref)   -->              --> RoleMatcher -> recommendations
    .json                                                --> SkillGap    -> skill_gap_analysis.csv
                                                         --> Roadmap     -> career_roadmap

---

## Configuration

All pipeline parameters centralized in config/pipeline_config.yaml:
  - All paths are project-relative (no hardcoded C:\Users\ paths)
  - Embedding model name, batch size, cache paths
  - HDBSCAN parameters (min_cluster_size, min_samples)
  - Cluster tracking threshold (centroid_similarity_threshold=0.75)
  - Emergence weights (7 components, sum=1.0)
  - LightGBM hyperparameters (preserved from training)
  - Forecast horizons (6M primary, 12M secondary)
  - Noise filter thresholds

---

## Separation of Concerns

  Layer                   Responsibility                               Files
  ----------------------  -------------------------------------------  -------------------------------
  Data Layer              Raw + cleaned + feature storage              data/, step1/data/cleaned/
  Discovery Layer         SBERT embedding + HDBSCAN clustering         src/clustering/, src/embeddings/
  Forecasting Layer       LightGBM + Holt-Damped forecasting          src/forecasting/, models/
  Intelligence Layer      Role catalog + skill intelligence            scripts/build_role_catalog.py
  Scoring Layer           Emergence score + unified ranking            src/emerging/, src/ranking/
  Profiling Layer         Questionnaire + user profile engine          backend/questionnaire_engine.py
  Recommendation Layer    Multi-factor scoring + skill gap + roadmap   src/recommendation/
  Explainability Layer    SHAP + evidence + component explanation      src/explainability/
  API Layer               FastAPI REST endpoints                       backend/main.py
  UI Layer                Streamlit multi-page application             frontend/app.py
  Evaluation Layer        Leakage tests + backtests + ablations        src/evaluation/
  Orchestration Layer     End-to-end pipeline runner                  run_pipeline.py

---

## Distinction Between Key Concepts

  DISCOVERED ROLE:    An emerging semantic cluster found by SBERT+HDBSCAN that
                      does not strongly match any known canonical role (title_novelty > 0.3)

  FORECASTED ROLE:    A role (known or discovered) for which a quantitative demand
                      forecast has been generated (LightGBM or Holt-Damped)

  RECOMMENDED ROLE:   A role selected for a specific user based on multi-factor
                      personal + market scoring (not based on forecast alone)

  EMERGENCE SCORE:    A 7-component composite score measuring how strongly a cluster
                      exhibits characteristics of an emerging role (0-1 scale)

  FORECAST:           A quantitative prediction of future demand in basis points
                      (for known roles) or posting count (for clusters)

  CONFIDENCE:         A label (HIGH/MEDIUM/LOW) reflecting evidence volume,
                      persistence, and forecast uncertainty -- NOT the same as
                      emergence score

  PERSONAL MATCH:     A user-specific multi-factor score combining personality,
                      skill, interest, and market opportunity alignment
