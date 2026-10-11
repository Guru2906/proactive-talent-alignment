"""
tests/test_recommendation_engine.py
====================================
Tests for recommendation engine, profile engine, skill gap, and roadmap.
"""

from __future__ import annotations

import numpy as np
import pytest
from src.recommendation.profile_engine import build_user_profile
from src.recommendation.role_matcher import RoleMatcher
from src.recommendation.skill_gap import compute_skill_gap
from src.recommendation.roadmap import generate_roadmap
from backend.data_loader import load_role_catalog, get_role_by_name


def test_profile_engine_output():
    # Builder vector
    builder_vector = np.array([10.0, 2.0, 1.0, 1.0])
    answers = [
        {"question_id": "ai_q1", "answer_index": 0, "answer_text": "I like building scalable systems."},
        {"question_id": "data_q2", "answer_index": 1, "answer_text": "I like designing software architecture."},
    ]
    profile = build_user_profile(
        user_vector=builder_vector,
        answers=answers,
        declared_skills=["Python", "Docker", "Git"],
        declared_interests=["AI / Machine Learning", "Software Engineering"],
    )

    assert profile["primary_archetype"] == "Builder"
    assert profile["profile_confidence"] > 0.0
    assert "Builder" in profile["BAIO"]
    assert len(profile["interests"]) >= 2
    assert "Python" in profile["declared_skills"]


def test_role_matcher_top5():
    matcher = RoleMatcher()
    builder_vector = np.array([8.0, 2.0, 1.0, 1.0])
    profile = build_user_profile(
        user_vector=builder_vector,
        answers=[],
        declared_skills=["Python", "FastAPI"],
        declared_interests=["Software Engineering"],
    )

    top5, secondary = matcher.rank(profile, top_n=5)
    assert len(top5) == 5, f"Expected 5 recommendations, got {len(top5)}"
    assert len(secondary) >= 1

    for rec in top5:
        assert 0.0 <= rec["personal_fit"] <= 1.0
        assert 0.0 <= rec["future_potential"] <= 1.0
        assert 0.0 <= rec["current_readiness"] <= 1.0
        assert len(rec["why_this_role"]) >= 1
        assert len(rec["challenges"]) >= 1


def test_emergence_does_not_dominate_all_profiles():
    matcher = RoleMatcher()

    # Analyst Profile
    analyst_vector = np.array([1.0, 10.0, 2.0, 1.0])
    analyst_profile = build_user_profile(
        user_vector=analyst_vector,
        answers=[],
        declared_skills=["SQL", "Tableau", "Pandas"],
        declared_interests=["Data Science / Analytics"],
    )
    analyst_top5, _ = matcher.rank(analyst_profile, top_n=5)

    # Operator / Security Profile
    operator_vector = np.array([1.0, 2.0, 1.0, 10.0])
    operator_profile = build_user_profile(
        user_vector=operator_vector,
        answers=[],
        declared_skills=["Linux", "Kubernetes", "Docker", "Terraform"],
        declared_interests=["Cloud / Infrastructure", "DevOps / SRE"],
    )
    operator_top5, _ = matcher.rank(operator_profile, top_n=5)

    analyst_top_names = [r["role_name"] for r in analyst_top5]
    operator_top_names = [r["role_name"] for r in operator_top5]

    # Different profiles MUST generate different top recommendations
    assert analyst_top_names != operator_top_names, "Expected different top-5 for Analyst vs Operator"


def test_skill_gap_buckets():
    catalog = load_role_catalog()
    role = next((r for r in catalog if r.get("required_skills")), None)
    assert role is not None, "At least one role must have required skills"

    # Give user the first required skill
    first_skill = role["required_skills"][0]
    gap = compute_skill_gap(role, user_skills=[first_skill])

    assert gap["data_available"] is True
    assert first_skill in gap["already_have"]
    assert first_skill not in gap["need_to_learn"]


def test_roadmap_generation():
    catalog = load_role_catalog()
    role = next((r for r in catalog if r.get("required_skills")), None)
    assert role is not None

    gap = compute_skill_gap(role, user_skills=["Python"])
    roadmap = generate_roadmap(role, gap)

    assert "method_note" in roadmap
    assert "rule-based" in roadmap["method_note"].lower()
    assert len(roadmap["projects"]) >= 1
