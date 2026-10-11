"""
scripts/build_role_catalog.py
==============================
One-shot script to generate the canonical role catalog artifact.

Reads EXISTING ML output files (no ML retraining):
  - step1_enhanced/outputs/top_emerging_roles_6m.csv
  - step1_enhanced/outputs/top_emerging_roles_12m.csv
  - data/outputs/emerging_role_predictions_detailed.csv
  - data/outputs/role_explanations.json
  - step1_enhanced/outputs/role_skill_matrix.csv
  - data/outputs/emerging_skills_forecast.csv
  - standardized/canonical_roles.csv

Outputs:
  - data/role_catalog/emerging_roles.json
  - data/role_catalog/emerging_roles.csv

Run with:
  python scripts/build_role_catalog.py

Safe to re-run. Overwrites previous catalog.
Missing fields are set to null — never fabricated.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "data" / "role_catalog"


# ── Helper ────────────────────────────────────────────────────────────────────

def _nv(val: Any) -> Any:
    """Return None instead of NaN / empty string."""
    if val is None:
        return None
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if isinstance(val, str) and val.strip() in ("", "nan", "NaN", "N/A", "None"):
        return None
    return val


def _safe_float(val: Any, decimals: int = 4) -> Optional[float]:
    v = _nv(val)
    if v is None:
        return None
    try:
        return round(float(v), decimals)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    v = _nv(val)
    if v is None:
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _skills_from_str(s: Any) -> List[str]:
    """Parse pipe-separated or comma-separated skill strings."""
    if not _nv(s):
        return []
    raw = str(s)
    sep = "|" if "|" in raw else ","
    return [sk.strip() for sk in raw.split(sep) if sk.strip()]


# ── Loaders ───────────────────────────────────────────────────────────────────

def _find_path(*candidates: Path) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_canonical_roles() -> pd.DataFrame:
    path = _find_path(BASE / "data" / "mappings" / "canonical_roles.csv", BASE / "standardized" / "canonical_roles.csv")
    if not path:
        logger.warning("Missing canonical_roles.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} canonical roles from {path}")
    return df


def load_6m_forecasts() -> pd.DataFrame:
    path = _find_path(BASE / "outputs" / "forecasts" / "top_emerging_roles_6m.csv", BASE / "step1_enhanced" / "outputs" / "top_emerging_roles_6m.csv")
    if not path:
        logger.warning("Missing top_emerging_roles_6m.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        "pred_share_bps_6m": "forecast_6m_bps",
        "growth_bps_6m": "growth_bps_6m",
        "growth_pct_6m": "growth_pct_6m",
        "emergence_tier_6m": "growth_tier",
    })
    logger.info(f"Loaded {len(df)} 6M forecasts from {path}")
    return df


def load_12m_forecasts() -> pd.DataFrame:
    path = _find_path(BASE / "outputs" / "forecasts" / "top_emerging_roles_12m.csv", BASE / "step1_enhanced" / "outputs" / "top_emerging_roles_12m.csv")
    if not path:
        logger.warning("Missing top_emerging_roles_12m.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={
        "pred_share_bps_12m": "forecast_12m_bps",
        "growth_bps_12m": "growth_bps_12m",
        "growth_pct_12m": "growth_pct_12m",
        "emergence_tier_12m": "growth_tier_12m",
    })
    logger.info(f"Loaded {len(df)} 12M forecasts from {path}")
    return df


def load_emerging_detailed() -> pd.DataFrame:
    path = _find_path(BASE / "outputs" / "emerging_roles" / "emerging_role_predictions_detailed.csv", BASE / "data" / "outputs" / "emerging_role_predictions_detailed.csv")
    if not path:
        logger.warning("Missing emerging_role_predictions_detailed.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} emerging cluster rows from {path}")
    return df


def load_role_explanations() -> Dict[str, dict]:
    path = _find_path(BASE / "outputs" / "explanations" / "role_explanations.json", BASE / "data" / "outputs" / "role_explanations.json")
    if not path:
        logger.warning("Missing role_explanations.json")
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded SHAP explanations for {len(data)} roles from {path}")
    return data


def load_skill_matrix() -> pd.DataFrame:
    path = _find_path(BASE / "step1_enhanced" / "outputs" / "role_skill_matrix.csv", BASE / "outputs" / "forecasts" / "role_skill_matrix.csv", BASE / "data" / "features" / "role_skill_matrix.csv")
    if not path:
        logger.warning("Missing role_skill_matrix.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded skill matrix: {len(df)} role-skill pairs from {path}")
    return df


def load_skills_forecast() -> pd.DataFrame:
    path = _find_path(BASE / "outputs" / "reports" / "emerging_skills_forecast.csv", BASE / "data" / "outputs" / "emerging_skills_forecast.csv")
    if not path:
        logger.warning("Missing emerging_skills_forecast.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} skill forecasts from {path}")
    return df


# ── Builders ──────────────────────────────────────────────────────────────────

def build_role_skill_lookup(skill_matrix: pd.DataFrame) -> Dict[str, Dict]:
    """Build per-role skill lists from the skill matrix."""
    lookup: Dict[str, Dict] = {}
    if skill_matrix.empty:
        return lookup

    for role, grp in skill_matrix.groupby("canonical_role"):
        # All skills for this role
        all_skills = grp["skill"].tolist()

        # Core required = highest penetration
        top = grp.nlargest(20, "role_skill_probability") if "role_skill_probability" in grp.columns else grp.head(20)
        required = top["skill"].tolist()

        # Frontier skills
        frontier_col = "is_frontier" if "is_frontier" in grp.columns else None
        frontier_skills: List[str] = []
        if frontier_col:
            frontier_skills = grp[grp[frontier_col] == 1]["skill"].tolist()

        lookup[role] = {
            "required_skills": required,
            "all_skills": all_skills,
            "frontier_skills": frontier_skills,
        }
    return lookup


def build_frontier_skills_lookup(skills_df: pd.DataFrame) -> Dict[str, dict]:
    """Build per-skill metadata from the skills forecast."""
    lookup: Dict[str, dict] = {}
    if skills_df.empty:
        return lookup

    for _, row in skills_df.iterrows():
        skill = str(row.get("skill", "")).strip()
        if not skill:
            continue
        lookup[skill.lower()] = {
            "growth_tier": _nv(row.get("growth_tier")),
            "is_frontier": bool(row.get("is_frontier", 0)),
            "future_demand_importance": _safe_float(row.get("future_demand_importance")),
            "annual_velocity_bps": _safe_float(row.get("annual_velocity_bps")),
        }
    return lookup


def get_shap_drivers(role_name: str, explanations: Dict[str, dict]) -> List[str]:
    """Extract top SHAP driver descriptions for a role."""
    exp = explanations.get(role_name, {})
    drivers = []
    for d in exp.get("top_positive_drivers", [])[:5]:
        desc = _nv(d.get("description"))
        if desc:
            drivers.append(str(desc))
    return drivers


# ── Known Role Entry ──────────────────────────────────────────────────────────

def build_known_role_entry(
    canonical_row: pd.Series,
    f6m: Optional[pd.Series],
    f12m: Optional[pd.Series],
    skill_lookup: Dict[str, Dict],
    explanations: Dict[str, dict],
) -> Dict[str, Any]:
    role_name = str(canonical_row.get("canonical_role_name", "")).strip()
    role_id = str(canonical_row.get("canonical_role_id", "")).strip()
    role_family = _nv(canonical_row.get("role_family"))
    role_category = _nv(canonical_row.get("role_category"))
    emergence_tier = _nv(canonical_row.get("emergence_tier"))

    skills_data = skill_lookup.get(role_name, {})
    required_skills = skills_data.get("required_skills", [])
    frontier_skills = skills_data.get("frontier_skills", [])

    # SHAP-based evidence
    shap_drivers = get_shap_drivers(role_name, explanations)

    entry: Dict[str, Any] = {
        "role_id": role_id,
        "role_name": role_name,
        "classification": "known_role",
        "source_type": "known_role",
        "forecast_method": "LightGBM (Pipeline A)",
        "role_family": role_family,
        "role_category": role_category,
        "emergence_tier": emergence_tier,

        # Scores from Pipeline A
        "emergence_score": None,  # not applicable for known roles
        "confidence": _safe_float(canonical_row.get("mapping_confidence")),
        "confidence_label": None,

        # Demand forecasts
        "current_demand_bps": _safe_float(f6m.get("baseline_share_bps") if f6m is not None else None),
        "forecast_6m_bps": _safe_float(f6m.get("forecast_6m_bps") if f6m is not None else None),
        "forecast_12m_bps": _safe_float(f12m.get("forecast_12m_bps") if f12m is not None else None),
        "growth_pct_6m": _safe_float(f6m.get("growth_pct_6m") if f6m is not None else None),
        "growth_pct_12m": _safe_float(f12m.get("growth_pct_12m") if f12m is not None else None),
        "growth_tier": _nv(f6m.get("growth_tier") if f6m is not None else None),
        "final_priority_score": None,

        # LinkedIn signals
        "li_posting_count": _safe_int(f6m.get("li_posting_count") if f6m is not None else None),
        "li_unique_companies": _safe_int(f6m.get("li_unique_companies") if f6m is not None else None),
        "li_frontier_skill_ratio": _safe_float(f6m.get("li_frontier_skill_ratio") if f6m is not None else None),
        "era_novelty_score": _safe_float(f6m.get("era_novelty_score") if f6m is not None else None),

        # Skills
        "required_skills": required_skills,
        "frontier_skills": frontier_skills,
        "emerging_skills": frontier_skills,  # alias

        # Cluster fields (N/A for known roles)
        "first_seen": None,
        "last_seen": None,
        "persistence_quarters": None,
        "market_breadth_score": None,
        "title_novelty": None,
        "novelty_category": None,
        "representative_titles": None,
        "closest_known_role": None,
        "forecast_next_quarter_postings": None,
        "forecast_growth_pct_cluster": None,

        # Explainability
        "shap_top_drivers": shap_drivers,
        "evidence": _nv(canonical_row.get("source_coverage", "Adzuna, LinkedIn, Naukri, O*NET, ESCO")),
        "role_description": None,  # not available from current artifacts
        "related_roles": [],
    }
    return entry


# ── Emerging Cluster Entry ─────────────────────────────────────────────────────

def build_emerging_role_entry(
    cluster_row: pd.Series,
    skill_lookup: Dict[str, Dict],
) -> Dict[str, Any]:
    role_name = _nv(cluster_row.get("candidate_role_name")) or _nv(cluster_row.get("role_name", "Unknown"))
    cluster_id = _safe_int(cluster_row.get("tracked_cluster_id"))

    # Map cluster name to skills via lookup (best-effort)
    skills_data = skill_lookup.get(role_name, {})
    required_skills = skills_data.get("required_skills", [])
    frontier_skills = skills_data.get("frontier_skills", [])

    # Representative titles as evidence
    rep_titles = _skills_from_str(cluster_row.get("representative_titles"))
    evidence_str = " | ".join(rep_titles[:5]) if rep_titles else None

    entry: Dict[str, Any] = {
        "role_id": f"CLUSTER_{cluster_id}" if cluster_id else f"CLUSTER_{role_name[:8].upper().replace(' ', '_')}",
        "role_name": role_name,
        "classification": _nv(cluster_row.get("classification", "Emerging")),
        "source_type": "emerging_cluster",
        "forecast_method": "SBERT + HDBSCAN (Pipeline B) + Holt Forecast",
        "role_family": None,
        "role_category": None,
        "emergence_tier": None,

        # Emergence scores
        "emergence_score": _safe_float(cluster_row.get("emergence_score")),
        "confidence": _safe_float(cluster_row.get("confidence_score")),
        "confidence_label": _nv(cluster_row.get("confidence_label")),

        # Demand forecasts (cluster-based, not LightGBM)
        "current_demand_bps": None,
        "forecast_6m_bps": None,
        "forecast_12m_bps": None,
        "growth_pct_6m": None,
        "growth_pct_12m": None,
        "growth_tier": None,
        "final_priority_score": _safe_float(cluster_row.get("final_priority_score")),

        # LinkedIn signals
        "li_posting_count": None,
        "li_unique_companies": None,
        "li_frontier_skill_ratio": None,
        "era_novelty_score": None,

        # Skills
        "required_skills": required_skills,
        "frontier_skills": frontier_skills,
        "emerging_skills": frontier_skills,

        # Cluster metadata
        "first_seen": _nv(cluster_row.get("first_seen")),
        "last_seen": _nv(cluster_row.get("last_seen")),
        "persistence_quarters": _safe_int(cluster_row.get("persistence_quarters")),
        "total_postings": _safe_int(cluster_row.get("total_postings")),
        "max_unique_companies": _safe_int(cluster_row.get("max_unique_companies")),
        "market_breadth_score": _safe_float(cluster_row.get("market_breadth_score")),
        "title_novelty": _safe_float(cluster_row.get("title_novelty")),
        "novelty_category": _nv(cluster_row.get("novelty_category")),
        "representative_titles": rep_titles,
        "closest_known_role": _nv(cluster_row.get("closest_canonical")),
        "forecast_next_quarter_postings": _safe_float(cluster_row.get("forecast_next_quarter_postings")),
        "forecast_growth_pct_cluster": _safe_float(cluster_row.get("forecast_growth_pct")),

        # Emergence component scores
        "demand_growth_score": _safe_float(cluster_row.get("demand_growth_score")),
        "skill_velocity_score": _safe_float(cluster_row.get("skill_velocity_score")),
        "persistence_score": _safe_float(cluster_row.get("persistence_score")),
        "technology_adoption_score": _safe_float(cluster_row.get("technology_adoption_score")),

        # Explainability
        "shap_top_drivers": [],
        "evidence": evidence_str,
        "role_description": None,
        "related_roles": [],
    }
    return entry


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 60)
    logger.info("  BUILD ROLE CATALOG")
    logger.info("=" * 60)

    # Load all sources
    canonical_df = load_canonical_roles()
    f6m_df = load_6m_forecasts()
    f12m_df = load_12m_forecasts()
    emerging_df = load_emerging_detailed()
    explanations = load_role_explanations()
    skill_matrix = load_skill_matrix()
    skills_df = load_skills_forecast()

    # Build lookup indices
    skill_lookup = build_role_skill_lookup(skill_matrix)
    logger.info(f"Built skill lookup for {len(skill_lookup)} roles")

    # Index forecasts by role name
    f6m_idx = f6m_df.set_index("role").to_dict("index") if not f6m_df.empty and "role" in f6m_df.columns else {}
    f12m_idx = f12m_df.set_index("role").to_dict("index") if not f12m_df.empty and "role" in f12m_df.columns else {}

    catalog: List[Dict[str, Any]] = []
    seen_roles: set = set()

    # ── 1. Known roles (Pipeline A) ───────────────────────────────────────────
    if not canonical_df.empty:
        for _, row in canonical_df.iterrows():
            role_name = str(row.get("canonical_role_name", "")).strip()
            if not role_name or role_name in seen_roles:
                continue

            f6m = f6m_idx.get(role_name)
            f12m = f12m_idx.get(role_name)

            entry = build_known_role_entry(
                canonical_row=row,
                f6m=pd.Series(f6m) if f6m else None,
                f12m=pd.Series(f12m) if f12m else None,
                skill_lookup=skill_lookup,
                explanations=explanations,
            )
            catalog.append(entry)
            seen_roles.add(role_name)

    logger.info(f"Built {len(catalog)} known-role entries")

    # ── 2. Emerging cluster roles (Pipeline B) ────────────────────────────────
    n_emerging_before = len(catalog)
    if not emerging_df.empty:
        for _, row in emerging_df.iterrows():
            role_name = _nv(row.get("candidate_role_name")) or _nv(row.get("role_name"))
            if not role_name:
                continue
            # Skip if essentially duplicate of a known role (very close canonical sim)
            max_sim = _safe_float(row.get("max_canonical_sim"))
            if max_sim is not None and max_sim >= 0.95 and role_name in seen_roles:
                continue
            if role_name in seen_roles:
                # Mark as emerging cluster version of a known role
                role_name = f"{role_name} (Emerging Cluster)"
            entry = build_emerging_role_entry(row, skill_lookup)
            catalog.append(entry)
            seen_roles.add(role_name)

    logger.info(f"Added {len(catalog) - n_emerging_before} emerging-cluster entries")
    logger.info(f"Total catalog entries: {len(catalog)}")

    # ── Output ────────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUT_DIR / "emerging_roles.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON catalog written: {json_path}")

    # CSV export (flatten list fields)
    flat_rows = []
    for entry in catalog:
        flat = dict(entry)
        flat["required_skills"] = "|".join(entry.get("required_skills") or [])
        flat["frontier_skills"] = "|".join(entry.get("frontier_skills") or [])
        flat["emerging_skills"] = "|".join(entry.get("emerging_skills") or [])
        flat["representative_titles"] = "|".join(entry.get("representative_titles") or [])
        flat["shap_top_drivers"] = "|".join(entry.get("shap_top_drivers") or [])
        flat["related_roles"] = "|".join(entry.get("related_roles") or [])
        flat_rows.append(flat)

    csv_df = pd.DataFrame(flat_rows)
    csv_path = OUT_DIR / "emerging_roles.csv"
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"CSV catalog written: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    known_count = sum(1 for e in catalog if e["source_type"] == "known_role")
    emerging_count = sum(1 for e in catalog if e["source_type"] == "emerging_cluster")
    logger.info("=" * 60)
    logger.info(f"  CATALOG SUMMARY")
    logger.info(f"  Known roles (Pipeline A):      {known_count}")
    logger.info(f"  Emerging clusters (Pipeline B): {emerging_count}")
    logger.info(f"  Total entries:                  {len(catalog)}")
    logger.info(f"  Output: {json_path}")
    logger.info("=" * 60)
    logger.info("  Catalog build complete. Ready for backend and UI.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
