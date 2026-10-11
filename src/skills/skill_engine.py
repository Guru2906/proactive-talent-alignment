"""
src/skills/skill_engine.py
==========================
Step 2 — Proactive Skill Emergence & Forecasting Engine.

Workflow:
  Role (Known & Emerging)
    ↓
  Required / Associated Skills
    ↓
  Current Skill Adoption (%)
    ↓
  Longitudinal Skill Velocity (CAGR & Acceleration)
    ↓
  Future Skill Demand Index
    ↓
  Skill Gap & Frontier Analysis

Integrates:
  - raw_data/emerging_skill_clusters_8yr.csv
  - raw_data/processed_data/skill_year_percentage.csv
  - raw_data/role_skill_mapping.csv
  - step1_enhanced/data/linkedin_role_skill_matrix.csv
  - data/outputs/emerging_role_predictions.csv
  - step1_enhanced/outputs/top_emerging_roles_6m.csv
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parents[2]


# ─── 1. Skill Velocity & Growth Metrics ───────────────────────────────────────

def compute_skill_velocity(
    skill_pct_path: Optional[Path] = None,
    skill_clusters_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Computes longitudinal growth velocity and acceleration metrics for technical skills
    using empirical multi-year posting percentages.
    """
    if skill_pct_path is None:
        skill_pct_path = BASE / "raw_data" / "processed_data" / "skill_year_percentage.csv"
    if skill_clusters_path is None:
        skill_clusters_path = BASE / "raw_data" / "emerging_skill_clusters_8yr.csv"

    if not skill_pct_path.exists():
        logger.warning(f"Skill year percentage file missing: {skill_pct_path}")
        return pd.DataFrame()

    df_pct = pd.read_csv(skill_pct_path)
    
    # Calculate latest penetration and multi-year velocity
    records = []
    for skill_name, grp in df_pct.groupby("skill"):
        grp_sorted = grp.sort_values("year")
        years = grp_sorted["year"].tolist()
        pcts = grp_sorted["percentage_of_jobs"].tolist()
        
        latest_year = years[-1]
        latest_pct = pcts[-1]
        
        # Velocity: recent vs early change
        if len(pcts) >= 2:
            early_pct = pcts[0] + 1e-5
            growth_pct = ((latest_pct - early_pct) / early_pct) * 100.0
            annual_velocity = (latest_pct - early_pct) / max(1, (years[-1] - years[0]))
        else:
            growth_pct = 0.0
            annual_velocity = 0.0
            
        records.append({
            "skill": skill_name,
            "latest_year": latest_year,
            "current_adoption_pct": round(latest_pct * 100.0, 3),
            "historical_growth_pct": round(growth_pct, 1),
            "annual_velocity_bps": round(annual_velocity * 10000.0, 2),
        })

    vel_df = pd.DataFrame(records)

    # Merge with 8-year cluster categories if available
    if skill_clusters_path.exists():
        clust_df = pd.read_csv(skill_clusters_path)
        # Normalize skill names
        clust_df["skill_clean"] = clust_df["Skill"].str.strip().str.lower()
        vel_df["skill_clean"] = vel_df["skill"].str.strip().str.lower()
        
        vel_df = vel_df.merge(
            clust_df[["skill_clean", "Category", "Cluster", "2024"]],
            on="skill_clean",
            how="left"
        ).drop(columns=["skill_clean"])
        
        vel_df["Category"] = vel_df["Category"].fillna("General Technical")
        vel_df["Cluster"] = vel_df["Cluster"].fillna(0).astype(int)
    else:
        vel_df["Category"] = "General Technical"
        vel_df["Cluster"] = 0

    # Classify tier
    def classify_skill(row):
        growth = row.get("historical_growth_pct", 0)
        adopt = row.get("current_adoption_pct", 0)
        cat = str(row.get("Category", "")).lower()
        name = str(row.get("skill", "")).lower()
        
        frontier_keywords = ["ai", "llm", "rag", "mlops", "generative", "transformer", "pytorch", "vector", "cloud"]
        is_frontier = any(k in name or k in cat for k in frontier_keywords)
        
        if is_frontier and growth > 20:
            return "Frontier / High Acceleration"
        elif growth > 50:
            return "Rapidly Emerging"
        elif adopt > 15:
            return "Established Core"
        elif growth > 0:
            return "Steady Growth"
        else:
            return "Matured / Declining"

    vel_df["growth_tier"] = vel_df.apply(classify_skill, axis=1)
    return vel_df.sort_values("annual_velocity_bps", ascending=False).reset_index(drop=True)


# ─── 2. Role-to-Skill Association & Demand Projection ────────────────────────

def build_role_skill_matrix(
    known_forecast_path: Optional[Path] = None,
    emerging_pred_path: Optional[Path] = None,
    role_skill_matrix_path: Optional[Path] = None,
    role_mapping_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Connects forecasted roles (Known & Emerging) with their required skillsets,
    computing a future skill demand index based on projected role growth.
    """
    if role_skill_matrix_path is None:
        role_skill_matrix_path = BASE / "step1_enhanced" / "data" / "linkedin_role_skill_matrix.csv"
    if known_forecast_path is None:
        known_forecast_path = BASE / "step1_enhanced" / "outputs" / "top_emerging_roles_6m.csv"
    if emerging_pred_path is None:
        emerging_pred_path = BASE / "data" / "outputs" / "emerging_role_predictions.csv"
    if role_mapping_path is None:
        role_mapping_path = BASE / "raw_data" / "role_skill_mapping.csv"

    # Base skill relationships
    if role_skill_matrix_path.exists():
        rs_df = pd.read_csv(role_skill_matrix_path)
    else:
        logger.warning(f"Role-skill matrix not found at {role_skill_matrix_path}")
        rs_df = pd.DataFrame(columns=["canonical_role", "skill", "role_skill_probability", "is_frontier"])

    # Load known role growth forecasts
    role_growth_map = {}
    if known_forecast_path.exists():
        kdf = pd.read_csv(known_forecast_path)
        for _, r in kdf.iterrows():
            role_growth_map[str(r["role"]).strip()] = float(r.get("growth_pct_6m", 0.0))

    # Load emerging role growth forecasts
    if emerging_pred_path.exists():
        edf = pd.read_csv(emerging_pred_path)
        for _, r in edf.iterrows():
            role_growth_map[str(r["Emerging Role"]).strip()] = float(r.get("Forecast Growth %", 0.0))

    # Augment role-skill matrix with projected future demand impact
    records = []
    for _, row in rs_df.iterrows():
        role = str(row["canonical_role"]).strip()
        skill = str(row["skill"]).strip()
        prob = float(row.get("role_skill_probability", 0.1))
        is_front = int(row.get("is_frontier", 0))
        
        growth = role_growth_map.get(role, 5.0)  # default 5% growth if unmapped
        
        # Skill demand index formula = probability * (1 + max(-0.5, growth/100.0))
        demand_idx = prob * (1.0 + max(-0.5, growth / 100.0))
        
        records.append({
            "role": role,
            "skill": skill,
            "role_skill_probability": round(prob, 4),
            "role_projected_growth_pct": round(growth, 1),
            "skill_demand_index": round(demand_idx, 4),
            "is_frontier": is_front,
        })

    res_df = pd.DataFrame(records)
    return res_df


# ─── 3. Comprehensive Step 2 Aggregation & Skill Summary ─────────────────────

def generate_step2_skill_intelligence(output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Executes the entire Step 2 Skill Emergence Pipeline and saves authoritative outputs:
      1. data/outputs/emerging_skills_forecast.csv
      2. data/outputs/role_skill_gap_analysis.csv
    """
    if output_dir is None:
        output_dir = BASE / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Executing Step 2: Skill Emergence & Forecasting Engine...")

    # 1. Compute longitudinal skill velocity
    vel_df = compute_skill_velocity()

    # 2. Build role-to-skill projections
    rs_proj = build_role_skill_matrix()

    # 3. Aggregate future skill importance across all roles
    if not rs_proj.empty:
        skill_agg = rs_proj.groupby("skill").agg(
            associated_roles_count=("role", "nunique"),
            associated_roles=("role", lambda s: " | ".join(list(s)[:4])),
            mean_role_growth=("role_projected_growth_pct", "mean"),
            future_demand_importance=("skill_demand_index", "sum"),
            is_frontier=("is_frontier", "max")
        ).reset_index()
    else:
        skill_agg = pd.DataFrame()

    # 4. Merge velocity with role-driven future demand
    if not skill_agg.empty and not vel_df.empty:
        merged_skills = vel_df.merge(skill_agg, on="skill", how="outer")
        merged_skills["future_demand_importance"] = merged_skills["future_demand_importance"].fillna(
            merged_skills["current_adoption_pct"] / 100.0
        )
        merged_skills["associated_roles_count"] = merged_skills["associated_roles_count"].fillna(1).astype(int)
        merged_skills["mean_role_growth"] = merged_skills["mean_role_growth"].fillna(0.0)
        merged_skills["is_frontier"] = merged_skills["is_frontier"].fillna(0).astype(int)
        merged_skills["growth_tier"] = merged_skills["growth_tier"].fillna("General Growth")
    elif not vel_df.empty:
        merged_skills = vel_df
        merged_skills["future_demand_importance"] = merged_skills["current_adoption_pct"] / 100.0
        merged_skills["associated_roles_count"] = 1
        merged_skills["associated_roles"] = "Broad IT Occupations"
        merged_skills["mean_role_growth"] = 10.0
        merged_skills["is_frontier"] = 0
    else:
        merged_skills = pd.DataFrame()

    # Sort and rank
    if not merged_skills.empty:
        merged_skills = merged_skills.sort_values("future_demand_importance", ascending=False).reset_index(drop=True)
        merged_skills["skill_rank"] = range(1, len(merged_skills) + 1)

    # 5. Build Role-Skill Gap Analysis
    role_gap_path = BASE / "raw_data" / "role_skill_mapping.csv"
    if role_gap_path.exists():
        gap_raw = pd.read_csv(role_gap_path)
        gap_records = []
        for _, r in gap_raw.iterrows():
            gap_records.append({
                "role": r["Current_Role"],
                "core_required_skills": r.get("Core_Skills", "N/A"),
                "emerging_frontier_skills": r.get("Emerging_Skills", "N/A"),
                "skill_gap_summary": r.get("Skill_Gap", "Requires modern cloud/AI upskilling"),
                "curriculum_gap_status": "Empirical Market Gap Identified (Formal Curriculum Gap requires syllabus dataset input)",
            })
        gap_df = pd.DataFrame(gap_records)
    else:
        gap_df = pd.DataFrame()

    # Save to data/outputs
    out_skill_path = output_dir / "emerging_skills_forecast.csv"
    out_gap_path = output_dir / "role_skill_gap_analysis.csv"

    merged_skills.to_csv(out_skill_path, index=False)
    gap_df.to_csv(out_gap_path, index=False)

    logger.info(f"Saved Step 2 Skill Forecast: {out_skill_path} ({len(merged_skills)} skills)")
    logger.info(f"Saved Role Skill Gap Analysis: {out_gap_path} ({len(gap_df)} roles)")

    return {
        "skills_forecast": merged_skills,
        "role_skill_gap": gap_df,
        "role_skill_matrix": rs_proj,
    }


if __name__ == "__main__":
    generate_step2_skill_intelligence()
