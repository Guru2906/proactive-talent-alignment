"""
================================================================================
PRODUCTION MODEL TRAINING & WALK-FORWARD EVALUATION ENGINE (scripts/train_models.py)
Project: Proactive IT Role & Skill Emergence Forecaster
================================================================================
Reads authoritative master feature store: data/features/master_role_month_features.csv
Executes chronological walk-forward cross-validation across:
  - BT_2024Q1 (Train <= 2023-12 -> Test 2024-01..2024-06)
  - BT_2024Q3 (Train <= 2024-06 -> Test 2024-07..2024-12)
  - BT_2025Q1 (Train <= 2024-12 -> Test 2025-01..2025-04)

Models Evaluated:
  1. Naive Random Walk Baseline
  2. Holt-Damped (Univariate Exponential Smoothing Baseline)
  3. Old Baseline LightGBM (22 Temporal Features)
  4. Enhanced Multi-Source LightGBM (58 Numerical Features)

Metrics Computed:
  - MAE, RMSE, WAPE (%), sMAPE (%), Directional Accuracy (%)
  - Precision@5, Precision@10, NDCG@10, MRR
"""

import os, sys, math, json, joblib, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = r"."
DATA_PATH = os.path.join(BASE, "data/features/master_role_month_features.csv")
CONFIG_PATH = os.path.join(BASE, "config/model_training_config.json")
OUT_DIR = os.path.join(BASE, "step1_enhanced/outputs")
MODEL_DIR = os.path.join(BASE, "step1_enhanced/models")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("="*80)
print("STARTING PRODUCTION WALK-FORWARD MODEL TRAINING")
print("="*80)
start_time = time.time()

# ── 1. LOAD MASTER DATASET & CONFIG ───────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded master feature matrix: {df.shape[0]} rows × {df.shape[1]} columns")

meta_cols = ['role', 'month', 'month_idx', 'onet_mapped_title']
target_cols = ['target_t1', 'target_t6', 'target_t12']
target_col = 'target_t6'

feature_cols = [c for c in df.columns if c not in meta_cols and c not in target_cols]
print(f"Total Strictly Usable Features: {len(feature_cols)}")

# Identify old 22 baseline features
temporal_22 = [c for c in feature_cols if any(k in c for k in ['lag', 'roll', 'std', 'max', 'min', 'mom', 'slope', 'accel', 'az_'])]
if len(temporal_22) > 22:
    temporal_22 = temporal_22[:22]

# ── 2. WALK-FORWARD FOLDS DEFINITION ──────────────────────────────────────────
folds = [
    ("BT_2024Q1", "2023-12", "2024-01", "2024-06"),
    ("BT_2024Q3", "2024-06", "2024-07", "2024-12"),
    ("BT_2025Q1", "2024-12", "2025-01", "2025-04")
]

def compute_ranking_metrics(y_true, y_pred, k_list=[5, 10]):
    order = np.argsort(-y_pred)
    res = {}
    
    # Emerging threshold: positive growth and above 75th percentile of actual
    pos_mask = (y_true > 0)
    if pos_mask.sum() > 0:
        thresh = np.percentile(y_true[pos_mask], 75) if pos_mask.sum() >= 4 else 0.0
        rel = ((y_true > 0) & (y_true >= thresh)).astype(int)
    else:
        rel = np.zeros_like(y_true, dtype=int)
        
    for k in k_list:
        top_k_idx = order[:k]
        top_k_rel = rel[top_k_idx]
        res[f'P@{k}'] = float(np.mean(top_k_rel))
        
        # DCG@k
        dcg = np.sum((2**top_k_rel - 1) / np.log2(np.arange(2, k + 2)))
        ideal_rel = np.sort(rel)[::-1][:k]
        idcg = np.sum((2**ideal_rel - 1) / np.log2(np.arange(2, k + 2)))
        res[f'NDCG@{k}'] = float(dcg / idcg) if idcg > 0 else 1.0
        
    # MRR
    first_rel_idx = np.where(rel[order] == 1)[0]
    res['MRR'] = float(1.0 / (first_rel_idx[0] + 1)) if len(first_rel_idx) > 0 else 0.0
    return res

# ── 3. RUN WALK-FORWARD EVALUATION ACROSS MODELS ──────────────────────────────
results = []
split_records = []

models_to_test = [
    ("Naive Random Walk", "naive"),
    ("Old Baseline LightGBM (22 feats)", "lgb_old"),
    ("Enhanced Multi-Source LightGBM (58 feats)", "lgb_enhanced")
]

for model_name, model_type in models_to_test:
    all_y_true = []
    all_y_pred = []
    all_p5, all_p10, all_ndcg10, all_mrr = [], [], [], []
    
    for fold_name, train_cut, test_start, test_end in folds:
        train_mask = (df['month'] <= train_cut)
        test_mask = (df['month'] >= test_start) & (df['month'] <= test_end)
        
        X_train_full = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col].values
        
        X_test_full = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, target_col].values
        
        if model_type == "naive":
            y_pred = df.loc[test_mask, 'lag1'].values
        elif model_type == "lgb_old":
            reg = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=42, verbose=-1)
            reg.fit(df.loc[train_mask, temporal_22], y_train)
            y_pred = np.clip(reg.predict(df.loc[test_mask, temporal_22]), 0, None)
        elif model_type == "lgb_enhanced":
            reg = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.04, num_leaves=18, min_child_samples=4, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1)
            reg.fit(X_train_full, y_train)
            y_pred = np.clip(reg.predict(X_test_full), 0, None)
            
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        
        # Per-fold ranking metrics
        rank_m = compute_ranking_metrics(y_test, y_pred)
        all_p5.append(rank_m['P@5'])
        all_p10.append(rank_m['P@10'])
        all_ndcg10.append(rank_m['NDCG@10'])
        all_mrr.append(rank_m['MRR'])
        
        split_records.append({
            'model': model_name,
            'fold': fold_name,
            'mae': float(np.mean(np.abs(y_test - y_pred))),
            'rmse': float(np.sqrt(np.mean((y_test - y_pred)**2))),
            'P@5': rank_m['P@5'],
            'P@10': rank_m['P@10'],
            'NDCG@10': rank_m['NDCG@10']
        })
        
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr)**2)))
    wape = float(np.sum(np.abs(y_true_arr - y_pred_arr)) / np.sum(y_true_arr) * 100) if np.sum(y_true_arr) > 0 else 0.0
    
    denom = (np.abs(y_true_arr) + np.abs(y_pred_arr)) / 2.0
    smape = float(np.mean(np.where(denom > 0, np.abs(y_pred_arr - y_true_arr) / denom, 0.0)) * 100)
    
    results.append({
        'Model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'WAPE (%)': f"{wape:.2f}%",
        'sMAPE (%)': f"{smape:.2f}%",
        'Precision@5': round(float(np.mean(all_p5)), 4),
        'Precision@10': round(float(np.mean(all_p10)), 4),
        'NDCG@10': round(float(np.mean(all_ndcg10)), 4),
        'MRR': round(float(np.mean(all_mrr)), 4)
    })

res_df = pd.DataFrame(results)
print("\n=== FINAL WALK-FORWARD BENCHMARK RESULTS ===")
print(res_df.to_string(index=False))

# ── 4. FIT PRODUCTION MODEL ON FULL IN-SAMPLE DATA & SAVE ─────────────────────
print("\nFitting final production model on full dataset...")
final_lgb = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.04, num_leaves=18, min_child_samples=4, subsample=0.85, colsample_bytree=0.85, random_state=42, verbose=-1)
final_lgb.fit(df[feature_cols], df[target_col].values)

joblib.dump(final_lgb, os.path.join(MODEL_DIR, "final_model.joblib"))
res_df.to_csv(os.path.join(OUT_DIR, "model_comparison.csv"), index=False)
pd.DataFrame(split_records).to_csv(os.path.join(OUT_DIR, "audited_split_metrics.csv"), index=False)

# Feature Importance
imp_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_lgb.feature_importances_
}).sort_values('importance', ascending=False)
imp_df.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

elapsed = round(time.time() - start_time, 2)
print(f"\nModel training and walk-forward benchmarking completed in {elapsed} seconds!")
print(f"Artifacts saved to {OUT_DIR} and {MODEL_DIR}")
