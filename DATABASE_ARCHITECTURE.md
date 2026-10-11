# Database Architecture

## 1. System Architecture & Data Layer Separation

The **Proactive IT Role & Skill Emergence Forecaster** maintains a strict architectural separation between **ML Datasets / Model Artifacts** and the **Persistent Application Database (SQLite)**.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                             ML DATA LAYER                                │
│   - Raw Job Listings (Adzuna, LinkedIn, Naukri CSVs)                    │
│   - Cleaned & Standardized Datasets (data/cleaned/, data/standardized/)  │
│   - Feature Matrices (data/features/master_role_month_features.csv)      │
│   - SBERT 384-dim Embeddings Cache (data/embeddings/)                   │
│   - HDBSCAN Clusters & Longitudinal Tracking (data/clusters/)            │
│   - Trained Models (models/final_model.joblib)                           │
│   - Role Catalog & Predictions (data/role_catalog/emerging_roles.json)   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ File-Based Artifacts
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION ENGINE LAYER                         │
│   - Psychometric BAIO Profiling (src/recommendation/profile_engine.py)   │
│   - Multi-Factor Role Matcher (src/recommendation/role_matcher.py)       │
│   - Empirical Skill Gap Analysis (src/recommendation/skill_gap.py)       │
│   - Rule-Based Learning Roadmap (src/recommendation/roadmap.py)          │
└──────────────────────┬────────────────────────────┬──────────────────────┘
                       │ Top Matches                │ Snapshots
                       ▼                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     APPLICATION DATABASE (SQLITE)                        │
│             data/application/career_recommendation.db                    │
│                                                                          │
│   • users                     • assessment_sessions                      │
│   • assessment_answers        • role_recommendations                     │
│   • recommendation_snapshots  • feedback                                 │
└──────────────────────▲────────────────────────────▲──────────────────────┘
                       │ Read / Write               │ Read / Write
        ┌──────────────┴──────────┐   ┌─────────────┴────────────┐
        │     FASTAPI BACKEND     │   │      STREAMLIT UI        │
        │    (REST API Server)    │   │  (Interactive Frontend)  │
        └─────────────────────────┘   └──────────────────────────┘
```

---

## 2. Why SQLite?

1. **Lightweight & Serverless**: Zero configuration or external database server required; ideal for standalone deployments and reproducible research environments.
2. **ACID Compliant**: Guarantees atomic transactions, data integrity, and rollback safety across multi-step questionnaire sessions.
3. **Thread-Safe with FastAPI & Streamlit**: Configured with `check_same_thread=False` and SQLite foreign key constraints (`PRAGMA foreign_keys=ON`).
4. **Isolated Application State**: Clearly delineates transient/user-facing application data from heavy numerical ML computation files.

---

## 3. Relational Schema Overview

```mermaid
erDiagram
    USERS ||--o{ ASSESSMENT_SESSIONS : "has many"
    USERS ||--o{ FEEDBACK : "submits"
    ASSESSMENT_SESSIONS ||--o{ ASSESSMENT_ANSWERS : "records"
    ASSESSMENT_SESSIONS ||--o{ ROLE_RECOMMENDATIONS : "generates"
    ASSESSMENT_SESSIONS ||--o{ RECOMMENDATION_SNAPSHOTS : "freezes"
    ASSESSMENT_SESSIONS ||--o{ FEEDBACK : "receives"
    ROLE_RECOMMENDATIONS ||--o{ RECOMMENDATION_SNAPSHOTS : "snapshots"
    ROLE_RECOMMENDATIONS ||--o{ FEEDBACK : "rated in"

    USERS {
        string id PK
        string name
        string email
        string education
        string degree_stream
        string experience_level
        text declared_skills
        text declared_interests
        datetime created_at
        datetime updated_at
    }

    ASSESSMENT_SESSIONS {
        string id PK
        string user_id FK
        datetime started_at
        datetime completed_at
        string status
        string primary_archetype
        string secondary_archetype
        text primary_archetype_description
        text secondary_archetype_description
        float archetype_confidence
        float profile_confidence
        float B_score
        float A_score
        float I_score
        float O_score
        text raw_vector
        text identified_interests
        text work_style
        text tech_alignment
        integer total_questions
        integer answered_count
    }

    ASSESSMENT_ANSWERS {
        integer id PK
        string session_id FK
        string question_id
        integer selected_option_index
        text selected_option_text
        text trait_weights
        string stage
        datetime answered_at
    }

    ROLE_RECOMMENDATIONS {
        integer id PK
        string session_id FK
        string role_id
        string role_name
        integer rank
        float match_score
        float personal_fit
        float future_potential
        float current_readiness
        string recommendation_type
        string source_type
        string role_family
        string confidence_label
        float emergence_score
        float growth_rate
        string growth_tier
        datetime created_at
    }

    RECOMMENDATION_SNAPSHOTS {
        integer id PK
        string session_id FK
        integer recommendation_id FK
        string role_id
        string role_name
        integer rank
        string recommendation_type
        float personal_fit
        float future_potential
        float current_readiness
        float final_score
        float emergence_score
        float growth_pct_6m
        float growth_pct_12m
        string growth_tier
        string source_type
        string role_family
        string role_category
        string confidence_label
        text description
        text why_this_role
        text challenges
        text required_skills
        text frontier_skills
        text component_scores
        text shap_top_drivers
        string model_version
        string pipeline_version
        datetime generated_at
    }

    FEEDBACK {
        integer id PK
        string user_id FK
        string session_id FK
        integer recommendation_id FK
        integer rating
        text feedback_text
        string category
        datetime created_at
    }
```

---

## 4. Recommendation Snapshots & Immutability

### Problem
ML pipelines are continually updated and retrained on new job market data. If recommendations were dynamically recomputed on every view, a user returning to view their past career assessment could see different recommendations than what was originally presented.

### Solution: Versioned Recommendation Snapshots
When recommendations are generated, the system creates frozen `RecommendationSnapshot` records containing:
- Complete score triad (`personal_fit`, `future_potential`, `current_readiness`, `final_score`)
- Role attributes, growth projections, and SHAP drivers
- Specific skill requirements and why/challenges explanations
- Model and Pipeline version tags (`model_version="v2.0.0"`, `pipeline_version="v2.0.0"`)
- Generation timestamp (`generated_at`)

When a user visits **Assessment History**, the system reads these immutable snapshots, ensuring 100% reproducible historical assessment records.

---

## 5. Database Initializer

Initialize the SQLite schema at any time using:

```bash
python -m backend.init_db
```

This creates `data/application/career_recommendation.db` and all tables with appropriate indexes and foreign key constraints without wiping existing data.
