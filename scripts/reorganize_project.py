"""
scripts/reorganize_project.py
==============================
Performs the canonical file migration and path sanitization.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def safe_copy(src: Path, dst: Path):
    if not src.exists():
        return
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"Copied: {src.name} -> {dst.relative_to(ROOT)}")
    except Exception as e:
        print(f"Warning copying {src} to {dst}: {e}")


def safe_move(src: Path, dst: Path):
    if not src.exists():
        return
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
        print(f"Moved: {src.name} -> {dst.relative_to(ROOT)}")
    except Exception as e:
        print(f"Warning moving {src} to {dst}: {e}")


def main():
    print("Beginning project reorganization...")

    # 1. Models -> models/
    safe_copy(ROOT / "step1_enhanced" / "models" / "final_model.joblib", ROOT / "models" / "final_model.joblib")
    safe_copy(ROOT / "step1_enhanced" / "models" / "final_config.json", ROOT / "models" / "final_config.json")

    # 2. Questionnaire -> data/questionnaire/
    safe_copy(ROOT / "backend" / "questionnaire.json", ROOT / "data" / "questionnaire" / "questionnaire.json")

    # 3. Canonical Mappings -> data/mappings/
    for f in (ROOT / "standardized").glob("*.csv"):
        safe_copy(f, ROOT / "data" / "mappings" / f.name)
    for f in (ROOT / "mappings").glob("*.csv"):
        safe_copy(f, ROOT / "data" / "mappings" / f.name)

    # 4. Raw Data -> data/raw/
    if (ROOT / "raw_data").exists():
        for item in (ROOT / "raw_data").iterdir():
            safe_copy(item, ROOT / "data" / "raw" / item.name)

    # 5. Master Features & Embeddings -> data/features/
    if (ROOT / "data" / "embeddings").exists():
        safe_copy(ROOT / "data" / "embeddings", ROOT / "data" / "features" / "embeddings")

    # 6. Structured Outputs -> outputs/
    # Forecasts
    for fn in ["known_role_forecasts.csv", "top_emerging_roles_6m.csv", "top_emerging_roles_12m.csv", "unified_ranking.csv"]:
        for loc in [ROOT / "data" / "outputs" / fn, ROOT / "step1_enhanced" / "outputs" / fn]:
            if loc.exists():
                safe_copy(loc, ROOT / "outputs" / "forecasts" / fn)

    # Emerging Roles
    for fn in ["emerging_role_discoveries.csv", "emerging_role_predictions.csv", "emerging_role_predictions_detailed.csv", "candidate_emerging_roles.csv"]:
        for loc in [ROOT / "data" / "outputs" / fn, ROOT / "step1_enhanced" / "outputs" / fn]:
            if loc.exists():
                safe_copy(loc, ROOT / "outputs" / "emerging_roles" / fn)

    # Copy Role Catalog to outputs/emerging_roles & outputs/recommendations
    if (ROOT / "data" / "role_catalog" / "emerging_roles.json").exists():
        safe_copy(ROOT / "data" / "role_catalog" / "emerging_roles.json", ROOT / "outputs" / "emerging_roles" / "emerging_roles.json")
        safe_copy(ROOT / "data" / "role_catalog" / "emerging_roles.csv", ROOT / "outputs" / "emerging_roles" / "emerging_roles.csv")
        safe_copy(ROOT / "data" / "role_catalog" / "emerging_roles.json", ROOT / "outputs" / "recommendations" / "emerging_roles.json")

    # Explanations
    for fn in ["role_explanations.json", "shap_global_importance.csv", "shap_per_role_drivers.csv", "feature_importance.csv"]:
        for loc in [ROOT / "data" / "outputs" / fn, ROOT / "step1_enhanced" / "outputs" / fn]:
            if loc.exists():
                safe_copy(loc, ROOT / "outputs" / "explanations" / fn)

    # Reports
    for fn in ["ablation_discovery_pipeline.csv", "ablation_forecasting_models.csv", "emerging_roles_charts.png", "leakage_audit_summary.csv", "role_skill_gap_analysis.csv"]:
        for loc in [ROOT / "data" / "outputs" / fn, ROOT / "step1_enhanced" / "outputs" / fn]:
            if loc.exists():
                safe_copy(loc, ROOT / "outputs" / "reports" / fn)

    # 7. Copy Root Audit CSVs to data/validation/
    for f in ROOT.glob("*.csv"):
        if f.name.startswith(("0", "1", "DATASET_MODEL_USAGE")):
            safe_copy(f, ROOT / "data" / "validation" / f.name)

    # 8. App Backend & Frontend -> app/
    safe_copy(ROOT / "backend", ROOT / "app" / "backend")
    safe_copy(ROOT / "frontend", ROOT / "app" / "frontend")

    # 9. Archive Legacy Implementation and old step1 snapshots
    if (ROOT / "initial implementation-20260831T100857Z-1-001").exists():
        safe_move(ROOT / "initial implementation-20260831T100857Z-1-001", ROOT / "archive" / "legacy" / "initial_implementation_raw")

    if (ROOT / "step1").exists():
        safe_copy(ROOT / "step1", ROOT / "archive" / "legacy" / "step1_prototype")

    if (ROOT / "step1_output").exists():
        safe_copy(ROOT / "step1_output", ROOT / "archive" / "legacy" / "step1_output")

    print("Reorganization step complete.")


if __name__ == "__main__":
    main()
