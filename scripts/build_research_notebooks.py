"""
scripts/build_research_notebooks.py
====================================
Generates 5 clean, well-structured Jupyter notebooks for research, evaluation, and visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"


def make_cell(cell_type: str, source: str) -> dict:
    lines = [line + "\n" for line in source.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        "outputs": [] if cell_type == "code" else None,
        "execution_count": None if cell_type == "code" else None,
    }


def save_notebook(filename: str, cells: list[dict]):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.13.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path = NB_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"Generated notebook: {path}")


def build_all_notebooks():
    # ── 01_data_analysis.ipynb ────────────────────────────────────────────────
    cells_01 = [
        make_cell("markdown", "# Notebook 01: Multi-Source Dataset Ingestion & EDA\n**Project:** Proactive Talent Alignment for Sustainable Employment\n\nThis notebook profiles the raw job postings and taxonomy datasets across Adzuna, LinkedIn, Naukri, O*NET, and ESCO."),
        make_cell("code", """import pandas as pd
from pathlib import Path

ROOT = Path('..')
canonical_roles = pd.read_csv(ROOT / 'data' / 'mappings' / 'canonical_roles.csv')
print(f'Total Canonical Roles: {len(canonical_roles)}')
canonical_roles.head(10)"""),
        make_cell("code", """master_features = pd.read_csv(ROOT / 'data' / 'features' / 'master_role_month_features.csv')
print(f'Master Role-Month Features Shape: {master_features.shape}')
master_features.describe()"""),
    ]
    save_notebook("01_data_analysis.ipynb", cells_01)

    # ── 02_role_forecasting.ipynb ─────────────────────────────────────────────
    cells_02 = [
        make_cell("markdown", "# Notebook 02: Known Role Demand Forecasting (Pipeline A)\n**Project:** Proactive Talent Alignment for Sustainable Employment\n\nEvaluates the Supervised LightGBM 6-Month and 12-Month macroeconomic demand forecasting models and SHAP attribution."),
        make_cell("code", """import pandas as pd
import json
from pathlib import Path

ROOT = Path('..')
f6m = pd.read_csv(ROOT / 'outputs' / 'forecasts' / 'top_emerging_roles_6m.csv')
print(f'Top 10 High-Growth Canonical Roles (6M Forecast):')
f6m.head(10)"""),
        make_cell("code", """with open(ROOT / 'outputs' / 'explanations' / 'role_explanations.json', 'r') as f:
    explanations = json.load(f)

first_role = list(explanations.keys())[0]
print(f'SHAP Drivers for: {first_role}')
explanations[first_role]['top_positive_drivers'][:5]"""),
    ]
    save_notebook("02_role_forecasting.ipynb", cells_02)

    # ── 03_emerging_role_discovery.ipynb ──────────────────────────────────────
    cells_03 = [
        make_cell("markdown", "# Notebook 03: Emerging Role Discovery & Semantic Tracking (Pipeline B)\n**Project:** Proactive Talent Alignment for Sustainable Employment\n\nInspects SBERT dense embeddings, quarterly HDBSCAN clustering, temporal centroid tracking, and 7-component emergence scoring."),
        make_cell("code", """import pandas as pd
from pathlib import Path

ROOT = Path('..')
emerging_detailed = pd.read_csv(ROOT / 'outputs' / 'emerging_roles' / 'emerging_role_predictions_detailed.csv')
print(f'Total Discovered Clusters: {len(emerging_detailed)}')
emerging_detailed[['candidate_role_name', 'emergence_score', 'confidence_label', 'persistence_quarters', 'forecast_growth_pct', 'closest_canonical']].head(10)"""),
    ]
    save_notebook("03_emerging_role_discovery.ipynb", cells_03)

    # ── 04_skill_analysis.ipynb ───────────────────────────────────────────────
    cells_04 = [
        make_cell("markdown", "# Notebook 04: Skill Velocity & Frontier Emergence (Step 2)\n**Project:** Proactive Talent Alignment for Sustainable Employment\n\nAnalyzes longitudinal skill growth velocities, frontier adoption ratios, and empirical role-skill association matrices."),
        make_cell("code", """import pandas as pd
from pathlib import Path

ROOT = Path('..')
skills_df = pd.read_csv(ROOT / 'outputs' / 'reports' / 'emerging_skills_forecast.csv')
print(f'Analyzed Skills: {len(skills_df)}')
skills_df.head(10)"""),
        make_cell("code", """role_skill = pd.read_csv(ROOT / 'step1_enhanced' / 'outputs' / 'role_skill_matrix.csv')
print(f'Total Role-Skill Associations: {len(role_skill)}')
role_skill.head(10)"""),
    ]
    save_notebook("04_skill_analysis.ipynb", cells_04)

    # ── 05_recommendation_analysis.ipynb ──────────────────────────────────────
    cells_05 = [
        make_cell("markdown", "# Notebook 05: Talent Assessment & Personalized Recommendations\n**Project:** Proactive Talent Alignment for Sustainable Employment\n\nValidates the 20-question BAIO profiling engine, 6-component multi-factor role recommendation, skill gap breakdown, and learning roadmap generator."),
        make_cell("code", """import numpy as np
import pandas as pd
import json
from pathlib import Path
from src.recommendation.profile_engine import build_user_profile
from src.recommendation.role_matcher import RoleMatcher
from src.recommendation.skill_gap import compute_skill_gap
from src.recommendation.roadmap import generate_roadmap

matcher = RoleMatcher()
user_vec = np.array([8.0, 2.0, 1.0, 1.0])
profile = build_user_profile(user_vec, answers=[], declared_skills=['Python', 'FastAPI', 'SQL'], declared_interests=['AI / Machine Learning'])
top5, sec = matcher.rank(profile, top_n=5)

print(f"User Archetype: {profile['primary_archetype']}")
for r in top5:
    print(f"#{r['rank']} {r['role_name']}: Personal Fit {r['personal_fit']*100:.0f}%, Future Potential {r['future_potential']*100:.0f}%, Readiness {r['current_readiness']*100:.0f}%")
"""),
    ]
    save_notebook("05_recommendation_analysis.ipynb", cells_05)


if __name__ == "__main__":
    build_all_notebooks()
