# Master Dataset Inventory & Governance Catalog
## Proactive IT Role & Skill Emergence Forecaster

**Status:** Completed & Audited  
**Total Tracked Files:** 132  
**Total Volume:** 1398.09 MB  

---

## 1. Inventory Summary by Functional Data Tier

| Functional_Category | File_Count | Total_Size_MB |
| --- | --- | --- |
| A. HISTORICAL JOB DEMAND | 7 | 141.60399999999998 |
| B. CURRENT JOB-MARKET SIGNAL | 9 | 1040.555 |
| D. SKILL ADOPTION / SKILL VELOCITY | 9 | 0.048 |
| E. ROLE-SKILL RELATIONSHIPS | 7 | 15.939 |
| F. OCCUPATIONAL TAXONOMY | 6 | 1.3359999999999999 |
| G. ACTUAL OCCUPATIONAL EMPLOYMENT | 16 | 4.102 |
| H. FUTURE EMPLOYMENT PROJECTIONS | 2 | 0.788 |
| I. FUTURE JOB OPENINGS | 2 | 0.788 |
| J. FUTURE SKILL PROJECTIONS | 1 | 0.203 |
| K. COMPANY / INDUSTRY CONTEXT | 8 | 50.54 |
| L. SALARY / COMPENSATION | 2 | 4.296 |
| O. DERIVED / GENERATED DATA | 63 | 137.891 |

---

## 2. Core Primary Datasets Detail

| Relative_Path | File_Name | File_Size_MB | MD5_Hash | Row_Count | Column_Count | Functional_Category | Primary_Purpose | Column_Names | Data_Types |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

---

## 3. Storage and Tiering Policy

1. **`data/01_raw/`**: Read-only immutable raw snapshots. Never modify or overwrite in place.
2. **`data/02_clean/`**: Parsed, deduplicated, and typed data tables.
3. **`data/03_standardized/`**: Mappings to standard canonical role, skill, and occupation hierarchies.
4. **`data/04_mappings/`**: Formal crosswalks between source identifiers and canonical entities.
5. **`data/05_features/`**: Feature tables with strict timestamp provenance.
6. **`data/06_model_ready/`**: Strict-grain tables prepared for Model training and validation.
