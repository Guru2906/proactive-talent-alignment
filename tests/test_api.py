"""
tests/test_api.py
==================
Tests for FastAPI backend REST endpoints, including SQLite persistence,
user management, assessment sessions, snapshots, and feedback.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["database_connected"] is True
    assert data["catalog_size"] >= 50


def test_roles_catalog_endpoint():
    res = client.get("/roles/catalog")
    assert res.status_code == 200
    catalog = res.json()
    assert len(catalog) >= 50


def test_roles_emerging_and_known_endpoints():
    res_e = client.get("/roles/emerging")
    assert res_e.status_code == 200
    assert len(res_e.json()) >= 15

    res_k = client.get("/roles/known")
    assert res_k.status_code == 200
    assert len(res_k.json()) >= 30


def test_users_endpoints():
    # Create user
    create_res = client.post("/users", json={
        "name": "Jordan Lee",
        "email": "jordan@example.com",
        "education": "Bachelor's",
        "declared_skills": ["Python", "SQL"],
        "declared_interests": ["AI / Machine Learning"],
    })
    assert create_res.status_code == 201
    user_data = create_res.json()
    user_id = user_data["id"]
    assert user_data["name"] == "Jordan Lee"

    # Get user
    get_res = client.get(f"/users/{user_id}")
    assert get_res.status_code == 200
    assert get_res.json()["email"] == "jordan@example.com"


def test_full_assessment_and_recommendation_flow():
    # 1. Start Assessment with User Info
    start_res = client.post("/assessment/start", json={
        "name": "Sam Taylor",
        "education": "Bachelor's",
        "experience_level": "1-3 years",
        "declared_skills": ["Python", "FastAPI"],
    })
    assert start_res.status_code == 200
    start_data = start_res.json()
    session_id = start_data["session_id"]
    user_id = start_data.get("user_id")
    assert "question" in start_data

    # 2. Answer 20 questions
    for _ in range(20):
        next_res = client.post("/assessment/next", json={"session_id": session_id, "answer_index": 0})
        assert next_res.status_code == 200
        next_data = next_res.json()
        if next_data["completed"]:
            break

    # 3. Complete Assessment
    comp_res = client.post("/assessment/complete", json={"session_id": session_id})
    assert comp_res.status_code == 200
    profile_data = comp_res.json()
    assert "primary_archetype" in profile_data
    assert "BAIO" in profile_data

    # 4. Get Assessment Result Endpoint
    result_res = client.get(f"/assessment/{session_id}/result")
    assert result_res.status_code == 200
    res_data = result_res.json()
    assert res_data["status"] == "completed"
    assert res_data["answers_count"] == 20

    # 5. Get Recommendations (persists snapshots)
    rec_res = client.post("/recommendations", json={"session_id": session_id})
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert len(rec_data["top_recommendations"]) == 5

    top1 = rec_data["top_recommendations"][0]
    assert 0.0 <= top1["personal_fit"] <= 1.0
    assert 0.0 <= top1["future_potential"] <= 1.0
    assert 0.0 <= top1["current_readiness"] <= 1.0

    # 6. Retrieve Snapshots
    snap_res = client.get(f"/assessment/{session_id}/snapshots")
    assert snap_res.status_code == 200
    snapshots = snap_res.json()
    assert len(snapshots) >= 5
    assert snapshots[0]["model_version"] == "v2.0.0"

    # 7. User Assessment History
    if user_id:
        hist_res = client.get(f"/users/{user_id}/assessments")
        assert hist_res.status_code == 200
        assert len(hist_res.json()) >= 1

    # 8. Submit Feedback
    fb_res = client.post("/feedback", json={
        "user_id": user_id,
        "session_id": session_id,
        "rating": 5,
        "feedback_text": "Great recommendation match!",
    })
    assert fb_res.status_code == 201
    assert fb_res.json()["rating"] == 5


def test_skill_gap_and_roadmap_endpoints():
    catalog_res = client.get("/roles/catalog")
    first_role = catalog_res.json()[0]
    role_id = first_role.get("role_id") or first_role["role_name"]

    # Skill gap
    gap_res = client.post("/skill-gap", json={"role_id": role_id, "user_skills": ["Python"]})
    assert gap_res.status_code == 200
    assert "already_have" in gap_res.json()

    # Roadmap
    road_res = client.post("/roadmap", json={"role_id": role_id, "user_skills": ["Python"]})
    assert road_res.status_code == 200
    road_data = road_res.json()
    assert "stages" in road_data
    assert "rule-based" in road_data["method_note"].lower()
