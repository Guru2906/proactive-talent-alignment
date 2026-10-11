"""
src/recommendation/profile_engine.py
======================================
Converts raw questionnaire session state → rich UserProfile dict.

Input:
  - user_vector: np.ndarray [B, A, I, O] accumulated from answers
  - answers: list of answer dicts (question_id, answer_index, answer_text)
  - declared_skills: list of skill strings from profile page
  - experience_level: str
  - education: str
  - declared_interests: list of interest strings

Output:
  - UserProfile dict (fully typed)

IMPORTANT: This module performs ONLY profiling / aggregation.
It does NOT match against roles (that is RoleMatcher's job).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

ARCHETYPE_KEYS = ["B", "A", "I", "O"]
ARCHETYPE_NAMES = {
    "B": "Builder",
    "A": "Analyst",
    "I": "Innovator",
    "O": "Operator",
}
ARCHETYPE_DESCRIPTIONS = {
    "Builder": "You love building, creating and engineering systems from the ground up.",
    "Analyst": "You excel at finding patterns, analyzing data, and deriving insights.",
    "Innovator": "You thrive in research, experimentation and pushing boundaries.",
    "Operator": "You excel at reliability, operations, and keeping systems running at scale.",
}

# ── Interest inference from answer patterns ───────────────────────────────────
# Maps question_id prefix → interest tag
QUESTION_INTEREST_MAP: Dict[str, str] = {
    "ai": "AI / Machine Learning",
    "ml": "AI / Machine Learning",
    "data": "Data Science / Analytics",
    "cloud": "Cloud / Infrastructure",
    "infra": "Cloud / Infrastructure",
    "security": "Cybersecurity",
    "sec": "Cybersecurity",
    "dev": "Software Engineering",
    "mobile": "Software Engineering",
    "devops": "DevOps / SRE",
    "sre": "DevOps / SRE",
    "product": "Product Management",
    "robotics": "Robotics / Hardware",
    "embedded": "Robotics / Hardware",
    "web": "Full Stack / Web",
    "fullstack": "Full Stack / Web",
}

# Work style keywords in answer text → tag
WORK_STYLE_ANSWER_MAP: List[tuple[str, str]] = [
    ("build", "Build and create systems"),
    ("create", "Build and create systems"),
    ("engineer", "Build and create systems"),
    ("analy", "Analyze data and find patterns"),
    ("data", "Analyze data and find patterns"),
    ("pattern", "Analyze data and find patterns"),
    ("research", "Research and experiment"),
    ("experiment", "Research and experiment"),
    ("innovat", "Research and experiment"),
    ("operat", "Operate and maintain reliability"),
    ("maintain", "Operate and maintain reliability"),
    ("reliab", "Operate and maintain reliability"),
    ("secur", "Solve security and compliance"),
    ("strateg", "Lead and strategize"),
    ("lead", "Lead and strategize"),
    ("manag", "Lead and strategize"),
]

# Technology preference keywords
TECH_PREFERENCE_MAP: List[tuple[str, str]] = [
    ("ai", "Artificial Intelligence"),
    ("machine learn", "Machine Learning"),
    ("deep learn", "Deep Learning"),
    ("cloud", "Cloud Computing"),
    ("devops", "DevOps / CI-CD"),
    ("kubernetes", "DevOps / CI-CD"),
    ("security", "Cybersecurity"),
    ("data", "Data Engineering"),
    ("mobile", "Mobile Development"),
    ("web", "Web Development"),
    ("robot", "Robotics"),
    ("embed", "Embedded Systems"),
    ("quantum", "Quantum Computing"),
    ("blockchain", "Blockchain"),
]


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0:
        return np.full(4, 0.25, dtype=float)
    return vector / total


def _derive_interests_from_answers(answers: List[Dict[str, Any]]) -> List[str]:
    """Infer interest tags from question_id prefixes."""
    interests: Dict[str, int] = {}
    for ans in answers:
        qid = str(ans.get("question_id", "")).lower()
        for prefix, tag in QUESTION_INTEREST_MAP.items():
            if qid.startswith(prefix):
                interests[tag] = interests.get(tag, 0) + 1
                break
    # Return interests that appeared in ≥2 questions, sorted by frequency
    result = [t for t, cnt in sorted(interests.items(), key=lambda x: -x[1]) if cnt >= 1]
    return result[:5]  # top 5


def _derive_work_style(answers: List[Dict[str, Any]]) -> List[str]:
    """Infer work-style tags from answer text content."""
    style_counts: Dict[str, int] = {}
    for ans in answers:
        text = str(ans.get("answer_text", "")).lower()
        for keyword, tag in WORK_STYLE_ANSWER_MAP:
            if keyword in text:
                style_counts[tag] = style_counts.get(tag, 0) + 1
    result = [t for t, cnt in sorted(style_counts.items(), key=lambda x: -x[1]) if cnt >= 1]
    return result[:3]


def _derive_tech_preferences(answers: List[Dict[str, Any]]) -> List[str]:
    """Infer technology preferences from answer text."""
    tech_counts: Dict[str, int] = {}
    for ans in answers:
        text = str(ans.get("answer_text", "")).lower()
        for keyword, tag in TECH_PREFERENCE_MAP:
            if keyword in text:
                tech_counts[tag] = tech_counts.get(tag, 0) + 1
    result = [t for t, cnt in sorted(tech_counts.items(), key=lambda x: -x[1]) if cnt >= 1]
    return result[:4]


def _compute_profile_confidence(
    normalized: np.ndarray,
    n_answers: int,
    total_questions: int,
) -> float:
    """
    Confidence in profile determination.
    Based on: vector spread (how dominant the primary archetype is)
    and completion ratio.
    """
    completion = n_answers / max(total_questions, 1)
    primary_dominance = float(normalized.max()) - float(normalized.mean())
    # Scale: 0 = completely flat vector, ~0.25 = max possible dominance
    dominance_score = min(1.0, primary_dominance / 0.3)
    confidence = 0.6 * completion + 0.4 * dominance_score
    return round(float(confidence), 4)


def build_user_profile(
    user_vector: np.ndarray,
    answers: List[Dict[str, Any]],
    declared_skills: Optional[List[str]] = None,
    experience_level: str = "1-3 years",
    education: str = "",
    degree_stream: str = "",
    declared_interests: Optional[List[str]] = None,
    total_questions: int = 20,
) -> Dict[str, Any]:
    """
    Build a rich UserProfile from raw questionnaire session data.

    Parameters
    ----------
    user_vector : np.ndarray
        Accumulated [B, A, I, O] float array from questionnaire answers.
    answers : list of dicts
        Full answer history: [{question_id, answer_index, answer_text}, ...]
    declared_skills : list of str
        Skills the user declared on the profile page.
    experience_level : str
        E.g. "Student / Fresher", "0-1 years", "1-3 years", "3-5 years", "5+ years"
    education : str
        E.g. "Bachelor's", "Master's"
    degree_stream : str
        E.g. "Computer Science", "Electronics", "Mathematics"
    declared_interests : list of str
        Interests selected on the profile page.
    total_questions : int
        Total questions in the questionnaire.

    Returns
    -------
    dict
        Rich UserProfile dict.
    """
    normalized = _normalize_vector(user_vector)
    sorted_indices = sorted(range(4), key=lambda i: (-float(normalized[i]), i))

    primary_key = ARCHETYPE_KEYS[sorted_indices[0]]
    secondary_key = ARCHETYPE_KEYS[sorted_indices[1]]
    primary = ARCHETYPE_NAMES[primary_key]
    secondary = ARCHETYPE_NAMES[secondary_key]

    baio_dict = {
        ARCHETYPE_NAMES[k]: round(float(normalized[i]), 4)
        for i, k in enumerate(ARCHETYPE_KEYS)
    }

    # Derive additional profile dimensions from answers
    inferred_interests = _derive_interests_from_answers(answers)
    work_style = _derive_work_style(answers)
    tech_preferences = _derive_tech_preferences(answers)

    # Merge declared and inferred interests (declared take priority)
    all_interests = list(declared_interests or [])
    for t in inferred_interests:
        if t not in all_interests:
            all_interests.append(t)
    all_interests = all_interests[:6]

    profile_confidence = _compute_profile_confidence(
        normalized, len(answers), total_questions
    )

    return {
        "BAIO": baio_dict,
        "primary_archetype": primary,
        "secondary_archetype": secondary,
        "primary_archetype_description": ARCHETYPE_DESCRIPTIONS.get(primary, ""),
        "secondary_archetype_description": ARCHETYPE_DESCRIPTIONS.get(secondary, ""),
        "raw_vector": [round(float(v), 4) for v in normalized],
        "archetype_confidence": round(float(normalized[sorted_indices[0]] - normalized[sorted_indices[1]]), 4),
        "interests": all_interests,
        "work_style": work_style,
        "technology_interest": tech_preferences,
        "preferred_problem_types": work_style,  # derived from same source
        "learning_preferences": [],  # could be extended with specific learning-style questions
        "career_preferences": [],
        "experience_level": experience_level,
        "education": education,
        "degree_stream": degree_stream,
        "declared_skills": declared_skills or [],
        "profile_confidence": profile_confidence,
        "n_answers": len(answers),
        "total_questions": total_questions,
    }
