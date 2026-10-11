"""
generate_dataset_usage_matrix.py
--------------------------------
Generates DATASET_MODEL_USAGE_MATRIX.csv detailing the exact usage, purpose,
join keys, leakage risks, and pipeline roles for every dataset in the repository.
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "notebooks" else Path(__file__).resolve().parent

matrix_data = [
    {
        "dataset": "raw_data/adzuna_global_job_listings_2025.csv",
        "purpose": "Primary time-series posting volume and raw title source for demand forecasting & semantic discovery",
        "time_coverage": "2022-01 to 2025-10 (Monthly)",
        "granularity": "Job-level posting (66,000+ IT postings)",
        "used_by_pipeline": "Pipeline A & Pipeline B",
        "used_as_feature": "Yes",
        "used_as_reference": "No",
        "used_for_validation": "Yes",
        "join_key": "job_id, created (month/quarter), clean_title",
        "leakage_risk": "Low (strictly split by monthly time-window t < target)",
        "reason_if_unused": "N/A - Fully used in both pipelines"
    },
    {
        "dataset": "raw_data/postings.csv (LinkedIn)",
        "purpose": "Modern job titles, company distributions, job descriptions, and skill associations",
        "time_coverage": "2023-2024 (Cross-sectional sample)",
        "granularity": "Job-level posting (33,000+ IT postings)",
        "used_by_pipeline": "Pipeline A & Pipeline B",
        "used_as_feature": "Yes",
        "used_as_reference": "No",
        "used_for_validation": "No",
        "join_key": "job_id, title, company_name",
        "leakage_risk": "Low (treated as cross-sectional structural priors)",
        "reason_if_unused": "N/A - Fully used"
    },
    {
        "dataset": "raw_data/jobs/job_skills.csv",
        "purpose": "Posting-to-skill mapping for co-occurrence, skill velocity, and tool adoption ratios",
        "time_coverage": "2023-2024",
        "granularity": "Posting-skill association (1.29M rows)",
        "used_by_pipeline": "Pipeline A & Step 2 Skill Engine",
        "used_as_feature": "Yes",
        "used_as_reference": "No",
        "used_for_validation": "No",
        "join_key": "job_id, skill_abr",
        "leakage_risk": "Low (co-occurrence matrix aggregated per canonical role)",
        "reason_if_unused": "N/A - Fully used"
    },
    {
        "dataset": "raw_data/naukri_com-job_sample.csv",
        "purpose": "Historical baseline market share and taxonomy validation for Indian & global tech hiring",
        "time_coverage": "2019-2020",
        "granularity": "Job-level posting (22,000 postings)",
        "used_by_pipeline": "Pipeline A (Baseline Era Features)",
        "used_as_feature": "Yes",
        "used_as_reference": "Yes",
        "used_for_validation": "No",
        "join_key": "job_id, jobtitle, skills",
        "leakage_risk": "Zero (strictly historical baseline precedes training window)",
        "reason_if_unused": "N/A - Fully used"
    },
    {
        "dataset": "standardized/canonical_roles.csv",
        "purpose": "Ground-truth taxonomy of 39 standard IT roles with keywords and embeddings",
        "time_coverage": "Standardized baseline",
        "granularity": "Role-level (39 canonical IT roles)",
        "used_by_pipeline": "Pipeline A & Pipeline B",
        "used_as_feature": "No",
        "used_as_reference": "Yes",
        "used_for_validation": "Yes",
        "join_key": "canonical_role",
        "leakage_risk": "Zero (static taxonomic reference)",
        "reason_if_unused": "N/A - Fully used"
    },
    {
        "dataset": "data/features/master_role_month_features.csv",
        "purpose": "Canonical panel feature store for LightGBM supervised demand forecasting",
        "time_coverage": "2022-09 to 2025-10 (38 monthly steps)",
        "granularity": "Role-Month (1,482 rows x 42 features)",
        "used_by_pipeline": "Pipeline A",
        "used_as_feature": "Yes",
        "used_as_reference": "No",
        "used_for_validation": "Yes",
        "join_key": "canonical_role, month",
        "leakage_risk": "Zero (strictly verified chronological walk-forward split)",
        "reason_if_unused": "N/A - Authoritative training dataset"
    },
    {
        "dataset": "raw_data/emerging_skill_clusters_8yr.csv",
        "purpose": "Longitudinal 8-year penetration and emergence trajectory for 91 core tech skills",
        "time_coverage": "2015-2024 (Annual)",
        "granularity": "Skill-Year percentage",
        "used_by_pipeline": "Step 2 Skill Emergence Engine",
        "used_as_feature": "Yes",
        "used_as_reference": "No",
        "used_for_validation": "Yes",
        "join_key": "Skill, Category",
        "leakage_risk": "Zero (multi-year historical trend)",
        "reason_if_unused": "N/A - Fully used"
    },
    {
        "dataset": "raw_data/processed_data/skill_year_percentage.csv",
        "purpose": "Empirical annual percentage share of jobs requiring specific technical skills",
        "time_coverage": "2015-2025",
        "granularity": "Skill-Year percentage (282 rows)",
        "used_by_pipeline": "Step 2 Skill Emergence Engine",
        "used_as_feature": "Yes",
        "used_as_reference": "No",
        "used_for_validation": "No",
        "join_key": "skill, year",
        "leakage_risk": "Zero",
        "reason_if_unused": "N/A - Fully used"
    },
    {
        "dataset": "raw_data/role_skill_mapping.csv",
        "purpose": "Mapping between current occupations and emerging required technical skillsets",
        "time_coverage": "Cross-sectional",
        "granularity": "Role-to-skills list (29 rows)",
        "used_by_pipeline": "Step 2 Skill Emergence Engine",
        "used_as_feature": "No",
        "used_as_reference": "Yes",
        "used_for_validation": "No",
        "join_key": "Current_Role",
        "leakage_risk": "Zero",
        "reason_if_unused": "N/A - Fully used"
    },
    {
        "dataset": "raw_data/tech_occupation_skill_full.csv (ESCO)",
        "purpose": "European Skills/Competences and Occupations technical skill taxonomy graph",
        "time_coverage": "Cross-sectional standard",
        "granularity": "Occupation-Skill URI pairs (6,618 rows)",
        "used_by_pipeline": "Taxonomy & Reference",
        "used_as_feature": "No",
        "used_as_reference": "Yes",
        "used_for_validation": "Yes",
        "join_key": "occupationUri, Skill_Clean",
        "leakage_risk": "Zero",
        "reason_if_unused": "Used as taxonomic reference graph rather than direct regressor"
    },
    {
        "dataset": "raw_data/companies/companies.csv",
        "purpose": "Company entity normalization, industry classification, and headquarters location",
        "time_coverage": "Cross-sectional",
        "granularity": "Company-level (23 MB)",
        "used_by_pipeline": "Market Breadth Scorer (Pipeline B)",
        "used_as_feature": "Yes",
        "used_as_reference": "Yes",
        "used_for_validation": "No",
        "join_key": "company_id, name",
        "leakage_risk": "Zero",
        "reason_if_unused": "N/A - Used for entity resolution and employer breadth scoring"
    },
    {
        "dataset": "raw_data/companies/employee_counts.csv",
        "purpose": "Company headcount size tiering (startup vs mid-market vs enterprise)",
        "time_coverage": "Cross-sectional",
        "granularity": "Company headcount",
        "used_by_pipeline": "Market Breadth Scorer",
        "used_as_feature": "Yes",
        "used_as_reference": "Yes",
        "used_for_validation": "No",
        "join_key": "company_id",
        "leakage_risk": "Zero",
        "reason_if_unused": "N/A - Used in employer tier weighting"
    },
    {
        "dataset": "raw_data/jobs/salaries.csv",
        "purpose": "Compensation data per job listing",
        "time_coverage": "2023-2024",
        "granularity": "Job-level salary",
        "used_by_pipeline": "Auxiliary Reference",
        "used_as_feature": "No",
        "used_as_reference": "Yes",
        "used_for_validation": "No",
        "join_key": "job_id",
        "leakage_risk": "High missingness (>85% nulls across job boards)",
        "reason_if_unused": "Excluded from core ML regressors to prevent systematic selection bias"
    },
    {
        "dataset": "raw_data/jobs/benefits.csv",
        "purpose": "Job perks and fringe benefits per listing",
        "time_coverage": "2023-2024",
        "granularity": "Job-level benefit tags",
        "used_by_pipeline": "Auxiliary Reference",
        "used_as_feature": "No",
        "used_as_reference": "Yes",
        "used_for_validation": "No",
        "join_key": "job_id",
        "leakage_risk": "High sparsity and subjective reporting",
        "reason_if_unused": "Excluded from demand forecasting as benefit mentions do not drive macro demand"
    },
    {
        "dataset": "initial implementation/backend/questionnaire.json",
        "purpose": "BAIO aptitude questionnaire weights and question logic",
        "time_coverage": "Static questionnaire",
        "granularity": "20 psychometric questions",
        "used_by_pipeline": "Auxiliary Student Guidance (Non-ML)",
        "used_as_feature": "No",
        "used_as_reference": "Yes",
        "used_for_validation": "No",
        "join_key": "question_id, archetype",
        "leakage_risk": "Zero (separate functional domain)",
        "reason_if_unused": "Belongs to student questionnaire flow, not macro labor market forecasting"
    }
]

df = pd.DataFrame(matrix_data)
out_csv = ROOT / "DATASET_MODEL_USAGE_MATRIX.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv} ({len(df)} datasets cataloged)")
