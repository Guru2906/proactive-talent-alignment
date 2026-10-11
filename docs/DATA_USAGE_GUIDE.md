# DATA USAGE GUIDE
## Project: Proactive Talent Alignment for Sustainable Employment
**Version:** 1.0.0 | **Date:** 2026-09-01

---

## Dataset Classification System

Every dataset in this project is assigned exactly one of the following classifications:

  CORE_TRAINING      Used directly in model training. Must never contain future information
                     relative to the training cutoff. Temporal splits enforced.

  FEATURE_SOURCE     Engineered features derived from raw sources. Used during training
                     and inference. Must be temporally sanitized.

  REFERENCE          External knowledge base used for taxonomy, enrichment, or labeling.
                     Static -- does not change with time. No leakage risk.

  MAPPING            Lookup tables: raw-to-canonical role/skill normalization.
                     Static. No training/inference boundary applies.

  EXTERNAL_KNOWLEDGE Third-party datasets (O*NET, Stack Overflow, ESCO) used for
                     cross-sectional enrichment. Date of data must predate all test splits.

  VALIDATION_ONLY    Generated during validation runs. Never used as model input.
                     Serves as scientific evidence artifacts.

  INFERENCE          Generated outputs consumed by the application API at inference time.
                     Must not be used as training features.

  ARCHIVE            Historical snapshots, legacy artifacts, superseded versions.
                     Retained for reproducibility but not used in production pipeline.

---

## Temporal Leakage Prevention Policy

For all CORE_TRAINING and FEATURE_SOURCE datasets:

1. All temporal features use strictly lagged values (lag1 = t-1 month, never t).
2. Walk-forward validation uses only data available before the cutoff date.
3. Cross-sectional features (LinkedIn, Naukri) are treated as static priors:
   they are available before any training cutoff since they summarize historical state.
4. O*NET / Stack Overflow features are static external datasets that predate all splits.
5. Target variables (target_t6, target_t12) are shifted forward and never included
   as input features.

Audit evidence: step1_enhanced/outputs/audited_feature_leakage_log.csv
All 58 features classified SAFE or NONE (zero HIGH-risk features found).

---

## Source Dataset Registry Summary

See data/DATASET_REGISTRY.csv for the complete machine-readable registry.

### CORE TRAINING DATASETS (3)

  1. adzuna_global_job_listings_2025.csv
     - 17,350 IT job listings from Adzuna global API
     - Date range: 2022-01 to 2025-10
     - Primary temporal signal: monthly posting counts per canonical role
     - Cleaned version: step1/data/cleaned/adzuna_cleaned.csv

  2. postings.csv (LinkedIn)
     - ~1.6M LinkedIn job postings
     - Used for cross-sectional features: company breadth, seniority distribution,
       skill diversity, frontier skill ratio
     - Also used as SBERT embedding source for Pipeline B discovery
     - Cleaned version: step1/data/cleaned/linkedin_cleaned.csv

  3. naukri_com-job_sample.csv
     - 22,000 Naukri (India) job listings
     - Provides: nk_posting_count, nk_share_bps, nk_skill_overlap_ratio
     - Validates demand signals from LinkedIn/Adzuna for Indian market
     - Cleaned version: step1/data/cleaned/naukri_cleaned.csv

### EXTERNAL KNOWLEDGE DATASETS (3)

  4. occupation.xlsx (BLS O*NET)
     - ~7,000 occupation records from O*NET 2018-2025
     - Mapped to canonical roles using fuzzy title matching
     - Provides Layer E features: onet_skill_count, onet_tech_density

  5. tech_occupation_skill_full.csv (O*NET)
     - 8,000 technology skill records per occupation
     - Provides: top programming languages, emerging tech relevance

  6. emerging_skill_clusters_8yr.csv (Stack Overflow Surveys)
     - 50 skill records across 8 years (2016-2024)
     - Provides: skill_multiyear_growth_rate for Layer C features

### MAPPING DATASETS (3)

  7. canonical_roles.csv -- 74 known IT canonical roles
  8. role_mapping.csv -- 2000+ raw title -> canonical role mappings
  9. skill_mapping.csv -- 900+ raw skill -> canonical skill mappings

### FEATURE STORE (3)

  10. master_role_month_features.csv -- 936x65 training feature matrix (CRITICAL)
  11. title_embeddings.npy -- SBERT embeddings cache for ~200K job titles
  12. canonical_embeddings.npy -- SBERT embeddings for 74 canonical roles

### REFERENCE DATASETS (3)

  13. questionnaire.json -- 48-question adaptive assessment bank
  14. future_it_jobs_master.csv -- curated future IT role names
  15. future_jobs_expanded.csv -- WEF/research future role reference

### INFERENCE DATASETS (1)

  16. data/role_catalog/emerging_roles.json -- 60-role production catalog consumed by API

### VALIDATION ONLY (2)

  17. walk_forward_predictions.csv -- backtest predictions (3 cutoffs x 3 models)
  18. audited_feature_leakage_log.csv -- 58-feature leakage audit verdicts

---

## What NOT to Do

  DO NOT use emerging_roles.json as a training feature.
  DO NOT use target_t6 or target_t12 as input features.
  DO NOT use role_monthly_demand.csv at time t as a feature for predicting time t.
  DO NOT use walk_forward_predictions.csv as training data.
  DO NOT mix datasets from different temporal scopes without checking time coverage.

---

## Data Quality Notes

  adzuna: 17,350 rows after deduplication and IT-category filter. Small but clean.
  linkedin: 123,849 rows after cleaning (from 1.6M total; ~8% are IT-relevant).
  naukri: 22,000 rows, all IT. Higher noise in skill extraction due to free-text format.
  O*NET: Static, authoritative. Matched to canonical roles with average similarity 0.42.
  SBERT embeddings: Generated with all-MiniLM-L6-v2, cached for reproducibility.
