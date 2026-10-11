# Final Multi-Tier Data Architecture Specification
## Proactive IT Role & Skill Emergence Forecaster

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: IMMUTABLE RAW DATA                         │
│  data/01_raw/                                                             │
│  ├── job_market/ (Adzuna 2023-2025, combined_jobs)                        │
│  ├── linkedin/   (postings.csv, job_skills, job_industries, salaries)     │
│  ├── naukri/     (naukri_com-job_sample)                                  │
│  ├── skills/     (emerging_skill_clusters_8yr, skill_year_percentage)     │
│  ├── onet/       (current_roles_master_cleaned_v2)                        │
│  ├── esco/       (future_jobs_expanded, future_job_profiles_esco_raw)     │
│  ├── employment/ (BLS OEWS National Estimates 2018–2025)                  │
│  └── projections/(BLS 10-Year Growth, Openings, STEM, Skill Projections)  │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        TIER 2: CLEANED & STANDARDIZED                     │
│  data/02_clean/ & data/03_standardized/                                   │
│  ├── canonical_roles.csv       (39 canonical IT roles, 7 families)        │
│  ├── canonical_skills.csv      (91 standard skills, AI/frontier tags)     │
│  ├── canonical_occupations.csv (41 BLS SOC computer occupations)          │
│  └── canonical_industries.csv  (422 LinkedIn industry sectors)            │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        TIER 3: MAPPINGS & CROSSWALKS                      │
│  data/04_mappings/ & data/05_features/                                    │
│  ├── role_occupation_crosswalk.csv (Canonical IT Role <-> SOC Code)       │
│  ├── role_skill_evidence_matrix.csv (P(skill|role) co-occurrences)        │
│  └── enhanced_role_forecasting_features.csv (65 features, 936 observations│
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        TIER 4: MODEL-READY DATASETS                       │
│  data/06_model_ready/                                                     │
│  ├── role_demand_monthly.csv        (Grain: 1 Role × 1 Month)             │
│  ├── role_skill_evidence.csv        (Grain: 1 Role × 1 Skill × 1 Source)  │
│  ├── skill_temporal.csv             (Grain: 1 Skill × 1 Year)             │
│  ├── occupation_employment_history.csv (Grain: 1 SOC × 1 Year, 2018-2025) │
│  └── future_occupation_projection.csv  (Grain: 1 SOC × 10-Yr Projection)  │
└───────────────────────────────────────────────────────────────────────────┘
```
