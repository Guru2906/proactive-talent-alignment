# End-to-End Data Lineage & Transformation Topology
## Proactive IT Role & Skill Emergence Forecaster

```
LAYER A: JOB DEMAND TIME SERIES
[Adzuna 2023-2025 Raw] ──> [adzuna_cleaned.csv] ──> [Monthly Role Demand Aggregation] ──> [Lags, Momentum, Slope, Acceleration] ──> [Model-Ready: role_demand_monthly.csv] ──> [Step 1 Role Emergence Model]

LAYER B: CROSS-SECTIONAL HIRING & SKILLS
[LinkedIn Postings Raw] ──> [linkedin_cleaned.csv] ──> [Role Taxonomy Classifier] ──> [Company Breadth, Seniority Entropy, Industry Diversity] ──> [Model-Ready: role_demand_monthly.csv] ──> [Step 1 Structural Prior]
[job_skills.csv + skills.csv] ──> [Relational Extraction] ──> [Role-Skill Co-occurrence Counts] ──> [Conditional Probability P(skill|role)] ──> [Model-Ready: role_skill_evidence.csv] ──> [Step 2 Skill Emergence Model]

LAYER C: SKILL ADOPTION & VELOCITY
[Stack Overflow 2024 Survey] ──> [emerging_skill_clusters_8yr.csv] ──> [7 Skill Clusters & AI Tool Tags] ──> [Role AI Tool Exposure] ──> [Feature Store: role_features]
[Processed Skill Longitude] ──> [skill_year_percentage.csv] ──> [Multi-Year CAGR Calculation] ──> [Longitudinal Skill Velocity] ──> [Model-Ready: skill_temporal.csv] ──> [Step 2 Velocity Forecaster]

LAYER D: PRE-AI HISTORICAL BASELINE
[Naukri 2015-2017 Raw] ──> [naukri_cleaned.csv] ──> [Pre-2018 Role Presence & Share] ──> [Role Era Novelty Score] ──> [Feature Store: role_features] ──> [Step 1 Model Feature]

LAYER E: OCCUPATIONAL STANDARDS & TECH DENSITY
[O*NET Occupational DB] ──> [current_roles_master_cleaned_v2.csv] ──> [Tech Density & Core Languages] ──> [Role-Occupation Crosswalk] ──> [Model-Ready: role_occupation_crosswalk.csv]
[ESCO Tech Profiles] ──> [future_jobs_expanded.csv] ──> [Candidate Role Discovery] ──> [Candidate Taxonomy Enrichment]

LAYER F: GOVERNMENT VALIDATION & LONGITUDINAL BENCHMARK
[BLS OEWS National 2018-2025] ──> [national_M2018-M2025_dl.xlsx] ──> [SOC Employment Histories] ──> [Model-Ready: occupation_employment_history.csv] ──> [Macroeconomic Validation Engine]
[BLS 2024-2034 Projections] ──> [1.2 Future growth / 1.10 Openings] ──> [10-Year SOC Projections] ──> [Model-Ready: future_occupation_projection.csv] ──> [Macro Validation Engine]
```

### Transformation Principles:
1. **Separation of Granularities**: Longitudinal monthly time series are kept distinct from static cross-sectional snapshots.
2. **Deterministic Transformations**: All aggregations (rolling means, momentum, Shannon entropy, Herfindahl indices) use deterministic mathematical formulas without random seeds.
3. **Strict Immutability**: Raw files in `data/01_raw/` are untouched. Transformations flow downstream into `02_clean`, `03_standardized`, `05_features`, and `06_model_ready`.
