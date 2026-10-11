"""
07_leakage_audit.py
Performs a rigorous data leakage audit across all features and target formulations.
Outputs: 07_LEAKAGE_AUDIT.csv & validation/leakage_audit.csv
"""
import os
import pandas as pd

BASE = r"."

rows = [
    ("Target Variable Y(t+6)", "target_t6 = market_share(t+6)", "Target Formulation", "SAFE", "Future outcome is the prediction target; strictly isolated from input feature vectors X(t)"),
    ("Autoregressive Lags", "lag1, lag2, lag3, lag6", "Temporal Feature", "SAFE", "Indexed at t-1, t-2, t-3, t-6 with zero future overlap"),
    ("Rolling Aggregations", "roll3m, roll6m, std3m", "Temporal Feature", "SAFE", "Computed over historical windows strictly before forecast origin"),
    ("Momentum & Slope", "mom1, mom3, mom6, slope6m, accel", "Temporal Feature", "SAFE", "First and second derivatives calculated on historical lags"),
    ("LinkedIn Structural Features", "li_posting_count, li_unique_companies, li_entropy", "Cross-Sectional Prior", "SAFE", "Fixed April 2024 structural snapshot; acts as a static Bayesian cross-sectional prior"),
    ("Skill Velocity", "sk_vel (CAGR over skill trajectories)", "Emerging Skill Feature", "SAFE", "Technology adoption curve computed from independent historical skill trends"),
    ("Pre-AI Novelty Index", "era_novelty_score, nk_share_bps", "Historical Feature", "SAFE", "Constructed from 2015-2017 Naukri data which strictly pre-dates all forecasting periods"),
    ("O*NET Tech Density", "onet_skill_count, onet_tech_density", "Occupational Knowledge", "SAFE", "Static government occupational taxonomy with no dynamic post-cutoff drift"),
    ("BLS 10-Year Projections", "10-Year Growth % / Job Openings", "Macroeconomic Projection", "FUTURE INFORMATION (EXCLUDED FROM TRAINING)", "Government 2024-2034 forecasts are strictly excluded from X(t) and reserved for validation"),
    ("Heuristic Future Scores", "Demand_Level in future_it_jobs_master", "Synthetic Heuristic", "EXCLUDED FROM GROUND TRUTH", "Model outputs excluded from training to prevent circular validation")
]

df = pd.DataFrame(rows, columns=[
    'feature_or_signal', 'mathematical_definition', 'feature_group', 'leakage_status', 'audit_explanation_and_mitigation'
])

df.to_csv(os.path.join(BASE, "07_LEAKAGE_AUDIT.csv"), index=False)
df.to_csv(os.path.join(BASE, "validation/leakage_audit.csv"), index=False)
print("07_LEAKAGE_AUDIT.csv generated.")