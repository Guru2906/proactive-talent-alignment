"""
src/recommendation/role_matcher.py
=====================================
Multi-factor role matching engine.

Takes:
  - UserProfile dict (from profile_engine)
  - Role catalog list (from data/role_catalog/emerging_roles.json)
  - Config weights (from config/recommendation_config.yaml)

Computes 6 scoring components separately, then produces:
  - personal_fit      (BAIO fit + interest + work_style)
  - future_potential  (demand forecast + emergence signal)
  - current_readiness (skill alignment)
  - final_score       (weighted combination of all 6)
  - explanation       (why this role matches)
  - challenges        (what may be difficult)

IMPORTANT: The emergence_signal is ONE of SIX components.
A high emergence score alone does NOT make a role the #1 recommendation.
Different user profiles will produce different top-5 orderings.
"""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "recommendation_config.yaml"
CATALOG_PATH = BASE / "data" / "role_catalog" / "emerging_roles.json"

with open(CFG_PATH, encoding="utf-8") as f:
    REC_CFG = yaml.safe_load(f)


# ── Catalog loading ────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_catalog() -> List[Dict[str, Any]]:
    if not CATALOG_PATH.exists():
        logger.warning(f"Role catalog not found at {CATALOG_PATH}. Run scripts/build_role_catalog.py first.")
        return []
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    logger.info(f"Loaded role catalog: {len(catalog)} entries")
    return catalog


# ── Score helpers ──────────────────────────────────────────────────────────────

def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        return f if not (np.isnan(f) or np.isinf(f)) else default
    except (TypeError, ValueError):
        return default


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.5
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _get_role_baio_vector(role: Dict[str, Any]) -> np.ndarray:
    """
    Get the BAIO vector for a role.
    Uses family-level defaults from config since role-level BAIO profiles
    are not available in the current ML artifacts.
    """
    family = role.get("role_family") or "default"
    family_profiles = REC_CFG.get("role_family_baio_profiles", {})
    profile = family_profiles.get(family, family_profiles.get("default", {}))
    return np.array([
        profile.get("builder", 0.25),
        profile.get("analyst", 0.25),
        profile.get("innovator", 0.25),
        profile.get("operator", 0.25),
    ], dtype=float)


def _score_baio_fit(user_profile: Dict[str, Any], role: Dict[str, Any]) -> float:
    """Cosine similarity between user BAIO vector and role BAIO profile."""
    user_vec = np.array(user_profile.get("raw_vector", [0.25, 0.25, 0.25, 0.25]), dtype=float)
    role_vec = _get_role_baio_vector(role)
    return max(0.0, min(1.0, (_cosine_sim(user_vec, role_vec) + 1.0) / 2.0))


def _score_interest_alignment(user_profile: Dict[str, Any], role: Dict[str, Any]) -> float:
    """How well user interests align with this role's category/family."""
    user_interests = set((user_profile.get("interests") or []))
    if not user_interests:
        return 0.5  # neutral when no interests declared

    interest_map = REC_CFG.get("interest_category_map", {})
    role_family = role.get("role_family", "")
    role_category = role.get("role_category", "")

    score = 0.0
    matches = 0
    for interest in user_interests:
        mapped_families = interest_map.get(interest, [])
        if role_family in mapped_families or role_category in mapped_families:
            score += 1.0
            matches += 1
        elif any(mf in (role_family or "") for mf in mapped_families):
            score += 0.5
            matches += 0.5

    return min(1.0, score / max(1, len(user_interests)))


def _score_work_style(user_profile: Dict[str, Any], role: Dict[str, Any]) -> float:
    """How well user work-style preferences align with this role."""
    user_styles = set(user_profile.get("work_style") or [])
    if not user_styles:
        return 0.5

    work_style_map = REC_CFG.get("work_style_map", {})
    role_family = role.get("role_family", "")

    total_boost = 0.0
    for style in user_styles:
        cfg = work_style_map.get(style, {})
        role_families = cfg.get("role_families", [])
        boost = cfg.get("boost", 0.0)
        if role_family in role_families:
            total_boost += boost

    return min(1.0, 0.5 + total_boost)


def _score_skill_alignment(user_profile: Dict[str, Any], role: Dict[str, Any]) -> float:
    """Jaccard similarity between user skills and role required skills."""
    user_skills = {s.lower().strip() for s in (user_profile.get("declared_skills") or [])}
    role_skills = {s.lower().strip() for s in (role.get("required_skills") or [])}

    if not role_skills:
        return 0.5  # no skill data → neutral

    if not user_skills:
        return 0.2  # no declared skills → low readiness

    return _jaccard(user_skills, role_skills)


def _score_future_opportunity(role: Dict[str, Any]) -> float:
    """
    Normalized future opportunity from Pipeline A LightGBM forecasts.
    Uses 6M growth % when available, then 12M, then emergence cluster forecast.
    """
    growth_6m = _safe_float(role.get("growth_pct_6m"))
    growth_12m = _safe_float(role.get("growth_pct_12m"))
    cluster_growth = _safe_float(role.get("forecast_growth_pct_cluster"))

    # Use the best available signal
    growth = None
    if role.get("source_type") == "known_role":
        # Prefer 12M for known roles (more stable)
        growth = growth_12m if growth_12m != 0.0 else growth_6m
    else:
        growth = cluster_growth

    if growth is None or growth == 0.0:
        # Fall back to growth tier as proxy
        tier = str(role.get("growth_tier") or "")
        tier_scores = {
            "very high": 0.85,
            "high": 0.65,
            "solid expansion": 0.60,
            "emerging leader": 0.85,
            "frontier": 0.75,
            "moderate": 0.40,
            "contracting": 0.15,
            "plateau": 0.25,
        }
        tier_lower = tier.lower()
        for keyword, score in tier_scores.items():
            if keyword in tier_lower:
                return score
        return 0.40

    # Scale: negative growth→0, 0% growth→0.3, 50%→0.65, 100%+→1.0
    if growth <= 0:
        return max(0.0, 0.3 + growth / 200.0)
    return min(1.0, 0.3 + (growth / 100.0) * 0.7)


def _score_emergence_signal(role: Dict[str, Any]) -> float:
    """
    Emergence signal (ONE of SIX components, capped weight of 0.05).
    For known roles: derived from emergence_tier.
    For emerging clusters: direct emergence_score.
    """
    if role.get("source_type") == "emerging_cluster":
        return _safe_float(role.get("emergence_score"), 0.3)

    # Known role: map emergence_tier to score
    tier = str(role.get("emergence_tier") or "")
    tier_lower = tier.lower()
    if "frontier" in tier_lower or "sparse emerging" in tier_lower:
        return 0.85
    if "growing" in tier_lower:
        return 0.65
    if "established" in tier_lower:
        return 0.35
    return 0.40


def _build_explanation(
    user_profile: Dict[str, Any],
    role: Dict[str, Any],
    component_scores: Dict[str, float],
) -> Tuple[List[str], List[str]]:
    """
    Build human-readable explanation strings.
    Returns (strengths, challenges).
    """
    strengths = []
    challenges = []

    primary = user_profile.get("primary_archetype", "")
    secondary = user_profile.get("secondary_archetype", "")
    interests = user_profile.get("interests", [])
    work_styles = user_profile.get("work_style", [])
    role_family = role.get("role_family", "")
    required_skills = set(s.lower() for s in (role.get("required_skills") or []))
    user_skills = set(s.lower() for s in (user_profile.get("declared_skills") or []))
    frontier_skills = [s for s in (role.get("frontier_skills") or [])]

    # BAIO alignment
    baio_score = component_scores.get("baio_fit", 0)
    if baio_score >= 0.70:
        strengths.append(f"Strong {primary} archetype alignment for {role['role_name']}")
    elif baio_score >= 0.55:
        strengths.append(f"Good {primary}/{secondary} alignment for this role")

    # Interest alignment
    interest_score = component_scores.get("interest_alignment", 0)
    if interest_score >= 0.70:
        for interest in interests[:2]:
            if any(k in (role_family or "").lower() for k in interest.lower().split()):
                strengths.append(f"Strong {interest} interest matches this domain")

    # Work style
    ws_score = component_scores.get("work_style_alignment", 0)
    if ws_score >= 0.65:
        for ws in work_styles[:1]:
            strengths.append(f"Work style preference ({ws}) aligns well")

    # Skill matches
    matched_skills = user_skills & required_skills
    if matched_skills:
        top_matched = sorted(matched_skills)[:3]
        strengths.append(f"Existing skills: {', '.join(s.title() for s in top_matched)}")

    # Future potential
    fp_score = component_scores.get("future_opportunity", 0)
    growth_pct = _safe_float(role.get("growth_pct_12m") or role.get("growth_pct_6m") or role.get("forecast_growth_pct_cluster"))
    if fp_score >= 0.65:
        if growth_pct > 0:
            strengths.append(f"Strong market demand signal (+{growth_pct:.0f}% projected growth)")
        else:
            strengths.append("Strong future demand signal from ML forecasts")

    # Emergence
    em_score = component_scores.get("emergence_signal", 0)
    confidence_label = role.get("confidence_label", "")
    if em_score >= 0.7 and confidence_label in ("High", "Medium"):
        strengths.append(f"Emerging role with {confidence_label} confidence signal")

    # Skill gaps → challenges
    skill_gap = required_skills - user_skills
    if len(skill_gap) > len(required_skills) * 0.6:
        challenges.append("Significant skill development required for this role")
    
    if frontier_skills:
        # Frontier skills user doesn't have
        missing_frontier = [s for s in frontier_skills if s.lower() not in user_skills][:3]
        if missing_frontier:
            challenges.append(f"Frontier skills to develop: {', '.join(missing_frontier)}")

    readiness_score = component_scores.get("skill_alignment", 0)
    if readiness_score < 0.3:
        challenges.append("Limited skill overlap — this is a stretch / aspirational role")
    elif readiness_score < 0.5:
        challenges.append("Some foundational skills still to be developed")

    # Ensure at least one strength and one challenge
    if not strengths:
        strengths.append(f"Matches your {primary} work preferences")
    if not challenges:
        challenges.append("Continuous learning expected as this domain evolves rapidly")

    return strengths[:5], challenges[:4]


def _apply_experience_modifier(readiness_score: float, experience_level: str) -> float:
    """Scale readiness score based on experience level."""
    modifiers = REC_CFG.get("experience_modifiers", {})
    mod = modifiers.get(experience_level, {})
    scale = mod.get("readiness_scale", 1.0)
    return min(1.0, readiness_score * scale)


# ── Main Matcher Class ─────────────────────────────────────────────────────────

class RoleMatcher:
    """
    Multi-factor role recommendation engine.

    Usage:
        matcher = RoleMatcher()
        results = matcher.rank(user_profile, top_n=5)
    """

    def __init__(self, catalog: Optional[List[Dict[str, Any]]] = None):
        self.catalog = catalog if catalog is not None else load_catalog()
        self.weights = REC_CFG.get("weights", {})
        self.output_cfg = REC_CFG.get("output_scores", {})
        self.rec_cfg = REC_CFG.get("recommendation", {})

    def _score_one_role(
        self,
        user_profile: Dict[str, Any],
        role: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Score a single role against the user profile."""
        # Compute 6 components
        baio_fit = _score_baio_fit(user_profile, role)
        interest_alignment = _score_interest_alignment(user_profile, role)
        work_style_alignment = _score_work_style(user_profile, role)
        skill_alignment = _score_skill_alignment(user_profile, role)
        future_opportunity = _score_future_opportunity(role)
        emergence_signal = _score_emergence_signal(role)

        # Apply experience modifier to readiness
        exp_level = user_profile.get("experience_level", "1-3 years")
        skill_alignment_adj = _apply_experience_modifier(skill_alignment, exp_level)

        component_scores = {
            "baio_fit": baio_fit,
            "interest_alignment": interest_alignment,
            "work_style_alignment": work_style_alignment,
            "skill_alignment": skill_alignment_adj,
            "future_opportunity": future_opportunity,
            "emergence_signal": emergence_signal,
        }

        # Weighted final score
        w = self.weights
        final_score = (
            w.get("baio_fit", 0.30) * baio_fit
            + w.get("interest_alignment", 0.25) * interest_alignment
            + w.get("work_style_alignment", 0.15) * work_style_alignment
            + w.get("skill_alignment", 0.15) * skill_alignment_adj
            + w.get("future_opportunity", 0.10) * future_opportunity
            + w.get("emergence_signal", 0.05) * emergence_signal
        )

        # Compute the three user-facing output scores
        pf_cfg = self.output_cfg.get("personal_fit_components", {})
        personal_fit = (
            pf_cfg.get("baio_fit", 0.50) * baio_fit
            + pf_cfg.get("interest_alignment", 0.30) * interest_alignment
            + pf_cfg.get("work_style_alignment", 0.20) * work_style_alignment
        )

        fp_cfg = self.output_cfg.get("future_potential_components", {})
        future_potential = (
            fp_cfg.get("future_opportunity", 0.70) * future_opportunity
            + fp_cfg.get("emergence_signal", 0.30) * emergence_signal
        )

        current_readiness = skill_alignment_adj  # 100% from skill alignment

        # Build explanations
        strengths, challenges = _build_explanation(user_profile, role, component_scores)

        # Minimum threshold filter
        min_threshold = self.rec_cfg.get("min_personal_fit_threshold", 0.10)
        if personal_fit < min_threshold:
            return None

        # Confidence from role or default
        confidence_label = role.get("confidence_label", "Medium")
        if confidence_label is None:
            # Known roles: infer from growth tier
            tier = str(role.get("growth_tier") or "")
            if "very high" in tier.lower() or "emerging leader" in tier.lower():
                confidence_label = "High"
            elif "high" in tier.lower() or "solid" in tier.lower():
                confidence_label = "Medium"
            else:
                confidence_label = "Medium"

        return {
            "role_id": role.get("role_id"),
            "role_name": role.get("role_name"),
            "source_type": role.get("source_type"),
            "role_family": role.get("role_family"),
            "role_category": role.get("role_category"),
            "emergence_tier": role.get("emergence_tier"),
            "confidence_label": confidence_label,

            # The 3 output scores (0-1, shown as %)
            "personal_fit": round(personal_fit, 4),
            "future_potential": round(future_potential, 4),
            "current_readiness": round(current_readiness, 4),

            # Final ranking score (internal)
            "final_score": round(final_score, 4),

            # Component breakdown (for transparency)
            "component_scores": {k: round(v, 4) for k, v in component_scores.items()},

            # Explanation
            "why_this_role": strengths,
            "challenges": challenges,

            # Role data passthrough
            "required_skills": role.get("required_skills") or [],
            "frontier_skills": role.get("frontier_skills") or [],
            "growth_pct_6m": role.get("growth_pct_6m"),
            "growth_pct_12m": role.get("growth_pct_12m"),
            "forecast_growth_pct_cluster": role.get("forecast_growth_pct_cluster"),
            "emergence_score": role.get("emergence_score"),
            "growth_tier": role.get("growth_tier"),
            "shap_top_drivers": role.get("shap_top_drivers") or [],
        }

    def rank(
        self,
        user_profile: Dict[str, Any],
        top_n: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Rank all catalog roles against user profile.

        Returns
        -------
        top_recommendations : list of top_n dicts (primary recommendations)
        secondary_recommendations : list of secondary_n dicts
        """
        if not self.catalog:
            return [], []

        top_n = top_n or self.rec_cfg.get("top_n", 5)
        secondary_n = self.rec_cfg.get("secondary_n", 5)

        scored: List[Dict[str, Any]] = []
        for role in self.catalog:
            result = self._score_one_role(user_profile, role)
            if result is not None:
                scored.append(result)

        # Sort by final_score descending
        scored.sort(key=lambda x: -x["final_score"])

        # Deduplicate by role name (take highest scorer)
        seen_names: set = set()
        deduped: List[Dict[str, Any]] = []
        for r in scored:
            name = r["role_name"]
            if name not in seen_names:
                seen_names.add(name)
                deduped.append(r)

        top = deduped[:top_n]
        secondary = deduped[top_n:top_n + secondary_n]

        # Add rank numbers
        for i, r in enumerate(top):
            r["rank"] = i + 1
        for i, r in enumerate(secondary):
            r["rank"] = top_n + i + 1

        return top, secondary
