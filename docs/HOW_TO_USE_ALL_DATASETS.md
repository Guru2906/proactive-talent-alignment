# How to Use All Datasets in the Capstone Ecosystem
## Proactive IT Role & Skill Emergence Forecaster

### 1. Which datasets are for Role Demand?
- **Primary Source**: `adzuna_cleaned.csv` (time series demand) + `linkedin_cleaned.csv` (structural features) + `naukri_cleaned.csv` (era novelty).
- **Model-Ready File**: `data/06_model_ready/role_demand_monthly.csv`.

### 2. Which datasets are for Skill Emergence?
- **Primary Source**: `emerging_skill_clusters_8yr.csv` (Stack Overflow 2024 adoption) + `skill_year_percentage.csv` (longitudinal velocity) + `job_skills.csv`.
- **Model-Ready File**: `data/06_model_ready/role_skill_evidence.csv` & `data/06_model_ready/skill_temporal.csv`.

### 3. Which datasets are for Macroeconomic Employment & Validation?
- **Primary Source**: `national_M2018-M2025.xlsx` (BLS OEWS historical employment) + `1.2 Future employment growth.xlsx` (BLS 10-year projections).
- **Model-Ready Files**: `data/06_model_ready/occupation_employment_history.csv` & `data/06_model_ready/future_occupation_projection.csv`.

### 4. Which datasets should NEVER be merged into a single table?
- **NEVER** merge annual BLS employment data with monthly job posting counts.
- **NEVER** concatenate 2015 Naukri records into 2024 Adzuna time series.
- **NEVER** merge 10-year forward BLS projections into short-term historical training feature matrices.

### 5. Safe Integration Strategy:
Combine source-specific signals at **Model Feature Store Time** using the standard canonical keys (`canonical_role_name`, `canonical_skill_name`, `soc_code`).
