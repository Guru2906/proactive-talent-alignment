# SYSTEM ARCHITECTURE: PROACTIVE TALENT ALIGNMENT

**Project:** Proactive Talent Alignment for Sustainable Employment  
**Version:** 2.0 Unified Production Architecture  

```
                               DATA SOURCES
        (Adzuna 66k+, LinkedIn 33k+, Naukri, O*NET, ESCO, LinkedIn Skills 1.29M)
                                    │
                                    ▼
                         PREPROCESSING & TAXONOMY
               (Technical Token Preservation, Canonical Entity Mappings)
                                    │
                                    ▼
                              FEATURE STORE
        (master_role_month_features.csv, SBERT 384-dim title_embeddings.npy)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
   PIPELINE A: KNOWN ROLES                         PIPELINE B: UNKNOWN ROLES
   (Supervised LightGBM Regressor)                 (Unsupervised SBERT + HDBSCAN)
   • 39 Canonical IT Roles                         • Quarterly Semantic Clustering
   • 6M / 12M Demand Forecasts                     • Temporal Centroid Tracking (>=0.75)
   • Walk-Forward Chronological CV                 • 7-Component Emergence Scoring
   • SHAP TreeExplainer Attribution                • Holt Linear Exponential Growth
            │                                               │
            └───────────────────────┬───────────────────────┘
                                    │
                                    ▼
                      STEP 2: SKILL EMERGENCE ENGINE
               (Role-Skill Matrix, Velocity CAGR, Frontier Ratios)
                                    │
                                    ▼
                         CANONICAL ROLE CATALOG
                     (data/role_catalog/emerging_roles.json)
                                    │
         ┌──────────────────────────┴──────────────────────────┐
         ▼                                                     ▼
[USER TALENT ASSESSMENT]                             [RECOMMENDATION ENGINE]
• 20-Question Adaptive Flow                          • 6-Component Transparent Scoring
• BAIO Vector Accumulation                           • Personal Fit % (0-100%)
• Interest & Work-Style Derivation                   • Future Potential % (0-100%)
• Experience Level Modifiers                         • Current Readiness % (0-100%)
         │                                                     │
         └──────────────────────────┬──────────────────────────┘
                                    │
                                    ▼
                     TOP 5 PERSONALIZED CAREER MATCHES
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
    [WHY THIS ROLE?]       [SKILL GAP ANALYSIS]     [LEARNING ROADMAP]
  • Strengths / Drivers   • Already Have (✓)       • Staged Progression
  • Market Growth Signal  • Need to Learn (→)      • Frontier Skill Milestones
  • Challenges & Gaps     • Optional/Frontier (◦)  • Portfolio Projects
```
