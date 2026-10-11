# USER APPLICATION ARCHITECTURE: PROACTIVE IT CAREER FORECASTER

**Project:** Proactive IT Role & Skill Emergence Forecaster (Capstone Upgrade)  
**Version:** 2.0 User-Facing Integration  
**Date:** 2026-09-01  

---

## 1. Executive Summary

This document describes the unified architecture integrating the standalone **BAIO Career Recommendation System** prototype (`initial implementation-20260831T100857Z-1-001/`) into the **Proactive IT Role & Skill Emergence Forecaster** (`capstoneee/`).

The result is a coherent, end-to-end user-facing product that transforms offline ML predictions into actionable, personalized career guidance.

---

## 2. Comparison: What Was Preserved vs. What Was Upgraded

| Subsystem | Initial Implementation | Upgraded Unified System |
|---|---|---|
| **Role Sourcing** | Static `Dataset1.csv` (10 hardcoded roles) | Machine-readable `data/role_catalog/emerging_roles.json` generated from Pipeline A (LightGBM) & Pipeline B (HDBSCAN) (60 roles) |
| **Questionnaire Engine** | 20-question, 3-phase adaptive state machine | Preserved 100% of adaptive branching + question weights; extended to output rich multi-dimensional user profile |
| **BAIO Scoring** | $[B, A, I, O]$ weights accumulated per answer | Preserved 100%; $[B, A, I, O]$ vector is integrated as the foundation of personal fit |
| **Matching Algorithm** | Cosine similarity on BAIO traits alone | 6-component transparent scoring engine (BAIO fit, interest, work style, skill overlap, future demand, emergence signal) |
| **Output Metrics** | Single cosine match score | Three distinct scores: **Personal Match %**, **Future Potential %**, **Current Readiness %** |
| **Explainability** | None (black-box match score) | "Why this role?" (strengths) and "What may be challenging?" (skill gaps) |
| **Skill Analysis** | None | 3-bucket gap analysis: *Already Have*, *Need to Learn*, *Optional / Frontier* |
| **Learning Path** | None | Rule-based staged learning roadmap with portfolio projects (explicitly labeled as rule-based) |
| **Frontend UI** | Single-page basic Streamlit quiz | Modern 8-page Streamlit application with interactive navigation, score cards, comparison matrix, and market explorer |
| **Backend API** | 3 basic endpoints (`/start`, `/next`, `/result`) | 10 REST endpoints with full schemas (`/roles/*`, `/assessment/*`, `/recommendations`, `/skill-gap`, `/roadmap`) |

---

## 3. High-Level Data Flow

```
+-------------------------------------------------------------------------+
|                        ML PREDICTION ARTIFACTS                          |
|  • Pipeline A 6M/12M Forecasts (step1_enhanced/outputs/)                |
|  • Pipeline B Discovered Clusters (data/outputs/)                       |
|  • Role-Skill Matrix (step1_enhanced/outputs/role_skill_matrix.csv)     |
|  • SHAP TreeExplainer Local Drivers (data/outputs/role_explanations.json)|
+-------------------------------------------------------------------------+
                                    │
                                    ▼
                 [scripts/build_role_catalog.py]
                                    │
                                    ▼
       [data/role_catalog/emerging_roles.json] (Canonical Catalog)
                                    │
          ┌─────────────────────────┴─────────────────────────┐
          ▼                                                   ▼
[FastAPI Backend / backend/main.py]                 [Streamlit / frontend/app.py]
  • GET  /roles/catalog                               • Page 1: Landing Page
  • GET  /roles/emerging                              • Page 2: Basic Info
  • GET  /roles/{role_id}                             • Page 3: Adaptive Questionnaire
  • POST /assessment/start                            • Page 4: Career Archetype
  • POST /assessment/next                             • Page 5: Top 5 Matches
  • POST /assessment/complete                         • Page 6: Role Details & Gap
  • POST /recommendations                             • Page 7: Role Comparison
  • POST /skill-gap                                   • Page 8: Emerging Explorer
  • POST /roadmap
```

---

## 4. Separation of Scientific Concerns

The system strictly enforces the separation between empirical ML prediction, heuristic personalization, and deterministic roadmapping:

1. **Forecasting (Machine Learning - Offline)**:
   - *Question Answered*: "What roles and skills are growing in the macroeconomic market?"
   - *Mechanism*: LightGBM walk-forward time-series models + SBERT/HDBSCAN quarterly clustering.
   - *Output*: Growth %, emergence scores, confidence intervals, SHAP drivers.

2. **Personalization (Multi-Factor Scoring - Online)**:
   - *Question Answered*: "Which future roles best align with this specific user's traits and passions?"
   - *Mechanism*: Configurable 6-component scoring formula in `config/recommendation_config.yaml`.
   - *Output*: Top 5 personalized roles with Personal Match % and Future Potential %.

3. **Skill Gap Analysis (Deterministic Set Operations - Online)**:
   - *Question Answered*: "What specific competencies does the user already possess vs. lack?"
   - *Mechanism*: Set intersections between declared user skills and empirical role-skill matrix.
   - *Output*: *Already Have*, *Need to Learn*, and *Optional / Frontier*.

4. **Roadmap Generation (Rule-Based Staging - Online)**:
   - *Question Answered*: "In what logical sequence should the user master their skill gaps?"
   - *Mechanism*: Dependency ordering from `roadmap_stage_order` config.
   - *Output*: Staged progression milestones and suggested capstone projects.
