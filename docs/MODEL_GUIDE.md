# MODEL GUIDE: METHODOLOGY & SPECIFICATIONS

**Project:** Proactive Talent Alignment for Sustainable Employment  

## 1. Pipeline A: Known Role Forecasting (Supervised LightGBM)
- **Model**: `LGBMRegressor` (`models/final_model.joblib`)
- **Hyperparameters**: `n_estimators=200`, `learning_rate=0.03`, `num_leaves=25`, `max_depth=5`, `subsample=0.85`, `colsample_bytree=0.85`
- **Validation**: Out-of-time chronological walk-forward cross-validation (zero temporal leakage).
- **Targets**: `target_t6` (6-month demand share in bps), `target_t12` (12-month demand share in bps).
- **Explainability**: SHAP TreeExplainer (`outputs/explanations/role_explanations.json`).

## 2. Pipeline B: Unknown Emerging Role Discovery (Unsupervised)
- **Embeddings**: Sentence-BERT `all-MiniLM-L6-v2` (384 dimensions) with token preservation (`AI`, `LLM`, `RAG`, `MLOps`, `FinOps`).
- **Clustering**: HDBSCAN with Euclidean metric on unit sphere, Excess of Mass (`eom`), `min_cluster_size=8`, `min_samples=3`.
- **Temporal Tracking**: Centroid matching with cosine similarity threshold $\ge 0.75$ across quarterly windows (2022-Q4 to 2025-Q4).
- **Growth Forecasting**: Holt's Linear Exponential Smoothing on quarterly posting counts.
- **7-Component Emergence Score**:
  $$\text{Emergence} = 0.25 \cdot \text{Growth} + 0.15 \cdot \text{SkillVel} + 0.15 \cdot \text{SkillNov} + 0.15 \cdot \text{TitleNov} + 0.10 \cdot \text{Breadth} + 0.10 \cdot \text{Persist} + 0.10 \cdot \text{Tech}$$
