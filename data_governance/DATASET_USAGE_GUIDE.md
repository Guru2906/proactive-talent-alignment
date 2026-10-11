# Master Dataset Usage Guide & Operational Protocols
## Proactive IT Role & Skill Emergence Forecaster

---

## 1. Dataset-to-Model Assignment Matrix

| Dataset | Role_Demand_Model | Skill_Emergence_Model | Role_Skill_Model | Employment_Validation | Future_Projection_Model | Feature_Role | Target_Candidate | Validation_Use | Prior_Use | Recommended | Reasoning |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adzuna_cleaned.csv | YES | OPTIONAL | NO | YES | NO | PRIMARY_FEATURE | YES | PRIMARY_TARGET | NO | YES | Continuous temporal demand backbone |
| linkedin_cleaned.csv | YES | YES | YES | NO | NO | STRUCTURAL_PRIOR | NO | NO | YES | YES | Company breadth, industry diversification, seniority entropy |
| naukri_cleaned.csv | YES | NO | NO | NO | NO | ERA_NOVELTY_FEATURE | NO | NO | YES | YES | Historical pre-AI baseline to quantify role novelty |
| emerging_skill_clusters_8yr.csv | YES | YES | YES | NO | NO | SKILL_ADOPTION_FEATURE | NO | VALIDATION | YES | YES | Developer survey skill adoption and AI tool exposure |
| skill_year_percentage.csv | YES | YES | YES | NO | NO | SKILL_VELOCITY_FEATURE | NO | NO | NO | YES | Longitudinal multi-year skill growth velocity |
| current_roles_master_cleaned_v2.csv | OPTIONAL | YES | YES | NO | NO | OCCUPATIONAL_FEATURE | NO | NO | YES | YES | O*NET technological skill density and language counts |
| future_jobs_expanded.csv | NO | YES | YES | NO | NO | VOCABULARY_FEATURE | NO | NO | YES | YES | Candidate role discovery and skill breadth |
| national_M2018_M2025.xlsx | NO | NO | NO | YES | NO | VALIDATION_ONLY | NO | MACRO_VALIDATION | NO | YES | Independent BLS government employment benchmark (2018-2025) |
| 1.2 Future employment growth.xlsx | NO | NO | NO | YES | YES | VALIDATION_ONLY | NO | PROJECTION_VALIDATION | NO | YES | BLS 10-year employment growth validation (2024-2034) |
| 1.10 future job openings.xlsx | NO | NO | NO | YES | YES | VALIDATION_ONLY | NO | OPENINGS_VALIDATION | NO | YES | BLS projected replacement and opening demand |

---

## 2. Temporal Leakage Safeguards

| Dataset | Feature_Family | Feature_Date | Target_Date | Leakage_Status | Mitigation_Reason |
| --- | --- | --- | --- | --- | --- |
| adzuna_cleaned.csv | Time Series Lags | t-1 | t | SAFE | Lagged features strictly computed at or before t-1 |
| adzuna_cleaned.csv | Rolling Statistics (roll3m/6m) | t-1 | t | SAFE | Moving averages computed over past months |
| adzuna_cleaned.csv | Momentum / Slope | t-1 | t | SAFE | First/second derivatives over historical windows |
| linkedin_cleaned.csv | Company Breadth / Seniority Entropy | 2024-04 | Static Prior | SAFE | Static cross-sectional prior for market structure |
| naukri_cleaned.csv | Historical Era Novelty | 2015-2017 | Static Prior | SAFE | Strictly pre-dates all forecasting horizons |
| emerging_skill_clusters_8yr.csv | Skill Adoption 2024 | 2024-12 | Static Prior | SAFE | Independent survey benchmark across 7 clusters |
| skill_year_percentage.csv | Multi-Year Skill Growth | 2015-2025 | Longitudinal | SAFE | Longitudinal compound annual skill growth |
| national_M2018-M2025.xlsx | Annual SOC Employment | 2018-2025 | Annual Historical | SAFE | Government macroeconomic validation series |
| 1.2 Future employment growth.xlsx | 10-Year BLS Projection | 2024-2034 | Future Horizon | VALIDATION_ONLY | Must NEVER enter feature vector during backtesting; reserved for future emergence triangulation |
| future_it_jobs_master.csv | Heuristic Demand Scores | Synthetic | Heuristic Output | EXCLUDED_FROM_LABELS | Model output excluded from training to eliminate circular validation risk |

---

## 3. Signal Redundancy & Double-Counting Prevention

| Signal_Group | Overlapping_Datasets | Nature_of_Overlap | Redundancy_Risk | Selection_Strategy |
| --- | --- | --- | --- | --- |
| Skill Longitudinal Representations | skill_year_matrix.csv vs skill_year_occurrence.csv vs skill_year_percentage.csv | Mathematical Transformations of Same Signal | HIGH | Use skill_year_percentage.csv as primary velocity; keep occurrence for frequency counts only |
| ESCO Future Profiles | future_jobs_expanded.csv vs future_job_profiles_esco_raw.csv | ESCO Occupational Vocabulary Duplication | HIGH | Use future_jobs_expanded.csv for skill matching; use esco_raw for descriptive lookup |
| O*NET Datasets | current_roles_master_cleaned_v2.csv vs tech_occupation_skill_full.csv | O*NET SOC Taxonomy Overlap | MODERATE | Use current_roles_master for tech density; use tech_occupation_skill for detailed task mappings |
| BLS Projections | 1.2 Future growth vs 1.10 Future openings vs occupation.xlsx | BLS 2024-2034 Macro Projections | MODERATE | Growth provides percentage change; Openings provides absolute replacement volume; separate cleanly in validation |

---

## 4. Usage Guidance by Capstone Phase

### Phase 1: Step 1 IT Role Emergence Forecasting
- **Primary Backbone**: `adzuna_cleaned.csv` (temporal lags, rolling shares, momentum).
- **Enrichment Covariates**: `linkedin_cleaned.csv` (structural market breadth), `emerging_skill_clusters_8yr.csv` (AI tool density), `naukri_cleaned.csv` (era novelty).
- **Validation**: Strict chronological walk-forward splits (`BT_2024Q1`, `BT_2024Q3`, `BT_2025Q1`).

### Phase 2: Step 2 Fine-Grained Skill Emergence Forecasting
- **Primary Input**: `role_skill_evidence.csv` + `skill_temporal.csv`.
- **Core Signal**: Longitudinal skill velocity (CAGR) combined with Stack Overflow developer adoption % and LinkedIn co-occurrence probabilities P(skill|role).

### Phase 3: Macroeconomic & Employment Validation
- **Primary Datasets**: `occupation_employment_history.csv` (BLS 2018–2025) and `future_occupation_projection.csv` (BLS 2024–2034).
- **Rule**: BLS 10-year projections are strictly **VALIDATION_ONLY** to test whether our ML 6–12 month emergence forecasts align with macroeconomic employment trends.
