"""
src/recommendation/skill_gap.py
=================================
Skill gap analysis: compares user's declared skills vs role's required skills.

Produces three buckets:
  - already_have   : skills user already has
  - need_to_learn  : core required skills user lacks
  - optional_advanced : frontier/advanced skills (optional upskilling)

Input:
  - role: dict from role catalog
  - user_skills: list of skill strings

Output:
  - SkillGapResult dict

NOTE: This module uses ONLY the role catalog data.
It does not invent skills or fabricate data.
If a role has no skill data, all buckets are empty with a data_unavailable flag.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _normalize_skill(s: str) -> str:
    return s.lower().strip()


def compute_skill_gap(
    role: Dict[str, Any],
    user_skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute skill gap for a specific role vs user's current skills.

    Parameters
    ----------
    role : dict
        Role entry from the role catalog.
    user_skills : list of str or None
        User's declared skills. If None/empty, all required skills go to need_to_learn.

    Returns
    -------
    dict with keys:
        role_name, already_have, need_to_learn, optional_advanced,
        skill_overlap_pct, data_available
    """
    user_skills_set = {_normalize_skill(s) for s in (user_skills or [])}

    required_skills = role.get("required_skills") or []
    frontier_skills = role.get("frontier_skills") or []
    all_role_skills = role.get("all_skills") or required_skills

    if not required_skills:
        return {
            "role_name": role.get("role_name", "Unknown"),
            "role_id": role.get("role_id"),
            "already_have": [],
            "need_to_learn": [],
            "optional_advanced": [],
            "skill_overlap_pct": None,
            "data_available": False,
            "message": "Skill data not available for this role from current ML artifacts.",
        }

    # Normalize role skills
    required_normalized = [_normalize_skill(s) for s in required_skills]
    frontier_normalized = {_normalize_skill(s) for s in frontier_skills}

    # Categorize
    already_have: List[str] = []
    need_to_learn: List[str] = []

    for skill, norm_skill in zip(required_skills, required_normalized):
        if norm_skill in user_skills_set:
            already_have.append(skill)
        else:
            need_to_learn.append(skill)

    # Optional/advanced = frontier skills the user doesn't have,
    # that aren't already in need_to_learn
    need_to_learn_set = {_normalize_skill(s) for s in need_to_learn}
    optional_advanced: List[str] = []
    for skill in frontier_skills:
        norm = _normalize_skill(skill)
        if norm not in user_skills_set and norm not in need_to_learn_set:
            optional_advanced.append(skill)

    # Skill overlap percentage
    total_required = len(required_skills)
    overlap_pct = round(len(already_have) / total_required * 100, 1) if total_required > 0 else 0.0

    return {
        "role_name": role.get("role_name", "Unknown"),
        "role_id": role.get("role_id"),
        "already_have": already_have,
        "need_to_learn": need_to_learn,
        "optional_advanced": optional_advanced[:8],  # cap to avoid overwhelming
        "skill_overlap_pct": overlap_pct,
        "data_available": True,
        "message": None,
    }


def format_gap_for_display(gap: Dict[str, Any]) -> str:
    """Return a formatted text summary of the skill gap."""
    lines = [f"### {gap['role_name']} — Skill Gap"]

    if not gap.get("data_available"):
        lines.append("_Skill data unavailable for this role._")
        return "\n".join(lines)

    lines.append(f"**Skill Overlap: {gap['skill_overlap_pct']}%**\n")

    if gap["already_have"]:
        lines.append("**✓ Already Have:**")
        for s in gap["already_have"]:
            lines.append(f"  ✓ {s}")
        lines.append("")

    if gap["need_to_learn"]:
        lines.append("**→ Need to Learn (Core):**")
        for s in gap["need_to_learn"]:
            lines.append(f"  → {s}")
        lines.append("")

    if gap["optional_advanced"]:
        lines.append("**◦ Optional / Advanced (Frontier Skills):**")
        for s in gap["optional_advanced"]:
            lines.append(f"  ◦ {s}")

    return "\n".join(lines)
