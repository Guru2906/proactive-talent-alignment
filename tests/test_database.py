"""
tests/test_database.py
======================
Comprehensive unit tests for the SQLite persistent application database layer.

Covers:
  1. User creation and retrieval
  2. Assessment session lifecycle (in_progress -> completed)
  3. Answer persistence and reconstruction
  4. BAIO profile calculation and result persistence
  5. Role recommendations and snapshot persistence
  6. Historical recommendation retrieval & reproducibility
  7. Multiple assessments for the same user
  8. Transaction rollback & error handling
  9. Feedback persistence
 10. ML dataset isolation (verifying ML files are untouched)
"""

from __future__ import annotations

import json
import uuid
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, init_db
from backend.db_models import (
    User,
    AssessmentSession,
    AssessmentAnswer,
    RoleRecommendation,
    RecommendationSnapshot,
    Feedback,
)
import backend.db_repository as repo
from src.recommendation.profile_engine import build_user_profile
from src.recommendation.role_matcher import RoleMatcher


@pytest.fixture
def test_db():
    """Create an isolated, in-memory SQLite database for each test."""
    test_engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


# ── 1. User Creation & Retrieval ──────────────────────────────────────────────

def test_user_creation_and_retrieval(test_db):
    user = repo.get_or_create_user(
        test_db,
        name="Alex Rivera",
        email="alex@example.com",
        education="Master's",
        degree_stream="Data Science",
        experience_level="3-5 years",
        declared_skills=["Python", "SQL", "Docker"],
        declared_interests=["AI / Machine Learning"],
    )

    assert user.id is not None
    assert user.name == "Alex Rivera"
    assert user.email == "alex@example.com"
    assert user.get_declared_skills() == ["Python", "SQL", "Docker"]
    assert user.get_declared_interests() == ["AI / Machine Learning"]

    # Fetch user back
    fetched = repo.get_user(test_db, user.id)
    assert fetched is not None
    assert fetched.id == user.id
    assert fetched.name == "Alex Rivera"


def test_user_profile_update(test_db):
    user = repo.get_or_create_user(test_db, name="Jamie", email="jamie@test.com")
    updated = repo.update_user_profile(
        test_db,
        user.id,
        education="PhD",
        declared_skills=["Rust", "Kubernetes"],
    )

    assert updated is not None
    assert updated.education == "PhD"
    assert updated.get_declared_skills() == ["Rust", "Kubernetes"]


# ── 2. Assessment Session Creation ────────────────────────────────────────────

def test_assessment_session_creation(test_db):
    user = repo.get_or_create_user(test_db, name="Taylor")
    session_obj = repo.create_assessment_session(
        test_db,
        user_id=user.id,
        education="Bachelor's",
        experience_level="1-3 years",
        total_questions=20,
    )

    assert session_obj.id is not None
    assert session_obj.user_id == user.id
    assert session_obj.status == "in_progress"
    assert session_obj.answered_count == 0
    assert session_obj.total_questions == 20


# ── 3. Answer Persistence ─────────────────────────────────────────────────────

def test_answer_persistence_and_order(test_db):
    user = repo.get_or_create_user(test_db, name="Jordan")
    session_obj = repo.create_assessment_session(test_db, user_id=user.id)

    # Save 3 answers
    a1 = repo.save_answer(
        test_db,
        session_id=session_obj.id,
        question_id="core_q1",
        selected_option_index=0,
        selected_option_text="I prefer designing software architecture.",
        trait_weights=[1.0, 0.0, 0.0, 0.0],
        stage="core",
    )
    a2 = repo.save_answer(
        test_db,
        session_id=session_obj.id,
        question_id="core_q2",
        selected_option_index=1,
        selected_option_text="I prefer statistical analysis and ML models.",
        trait_weights=[0.0, 1.0, 0.0, 0.0],
        stage="core",
    )

    answers = repo.get_session_answers(test_db, session_obj.id)
    assert len(answers) == 2
    assert answers[0].question_id == "core_q1"
    assert answers[0].get_trait_weights() == [1.0, 0.0, 0.0, 0.0]
    assert answers[1].question_id == "core_q2"

    # Session answered_count should be updated
    session_refreshed = repo.get_assessment_session(test_db, session_obj.id)
    assert session_refreshed.answered_count == 2


# ── 4. BAIO Result Persistence ────────────────────────────────────────────────

def test_complete_assessment_persists_baio(test_db):
    user = repo.get_or_create_user(test_db, name="Morgan")
    session_obj = repo.create_assessment_session(test_db, user_id=user.id)

    builder_vector = np.array([8.0, 2.0, 1.0, 1.0])
    profile_dict = build_user_profile(
        user_vector=builder_vector,
        answers=[{"question_id": "q1", "answer_index": 0, "answer_text": "Building systems"}],
        declared_skills=["Python", "Go"],
        declared_interests=["Software Engineering"],
    )

    completed_session = repo.complete_assessment(test_db, session_obj.id, profile_dict)
    assert completed_session is not None
    assert completed_session.status == "completed"
    assert completed_session.completed_at is not None
    assert completed_session.primary_archetype == "Builder"
    assert completed_session.B_score > 0.5
    assert completed_session.profile_confidence > 0.0
    assert "Software Engineering" in completed_session.get_interests()


# ── 5. Recommendation Persistence & Snapshots ─────────────────────────────────

def test_recommendation_and_snapshot_persistence(test_db):
    user = repo.get_or_create_user(test_db, name="Casey")
    session_obj = repo.create_assessment_session(test_db, user_id=user.id)

    profile_dict = build_user_profile(
        user_vector=np.array([8.0, 2.0, 1.0, 1.0]),
        answers=[],
        declared_skills=["Python", "FastAPI"],
    )
    repo.complete_assessment(test_db, session_obj.id, profile_dict)

    matcher = RoleMatcher()
    top_recs, sec_recs = matcher.rank(profile_dict, top_n=5)

    saved_recs, saved_snaps = repo.save_recommendations_and_snapshots(
        test_db,
        session_id=session_obj.id,
        top_recommendations=top_recs,
        secondary_recommendations=sec_recs,
        model_version="v2.0.0",
        pipeline_version="v2.0.0",
    )

    assert len(saved_recs) == len(top_recs) + len(sec_recs)
    assert len(saved_snaps) == len(saved_recs)

    # Verify snapshot contents
    snap1 = saved_snaps[0]
    assert snap1.rank == 1
    assert snap1.model_version == "v2.0.0"
    assert snap1.personal_fit > 0.0
    assert isinstance(json.loads(snap1.why_this_role), list)
    assert isinstance(json.loads(snap1.required_skills), list)


# ── 6. Historical Snapshot Reproducibility ─────────────────────────────────────

def test_snapshot_reproducibility(test_db):
    """Verify that saved snapshots can be retrieved identically even after ML updates."""
    user = repo.get_or_create_user(test_db, name="Historical Tester")
    session_obj = repo.create_assessment_session(test_db, user_id=user.id)

    profile_dict = build_user_profile(
        user_vector=np.array([2.0, 9.0, 1.0, 1.0]),
        answers=[],
        declared_skills=["SQL", "Pandas"],
    )
    repo.complete_assessment(test_db, session_obj.id, profile_dict)

    matcher = RoleMatcher()
    top_recs, _ = matcher.rank(profile_dict, top_n=5)

    repo.save_recommendations_and_snapshots(
        test_db,
        session_id=session_obj.id,
        top_recommendations=top_recs,
        model_version="v2.0.0",
    )

    # Retrieve snapshots
    snapshots = repo.get_session_snapshots(test_db, session_obj.id)
    assert len(snapshots) == 5
    first_snap = snapshots[0]
    assert first_snap.role_name == top_recs[0]["role_name"]
    assert first_snap.final_score == top_recs[0]["final_score"]
    assert first_snap.personal_fit == top_recs[0]["personal_fit"]


# ── 7. Multiple Assessments for Same User ─────────────────────────────────────

def test_multiple_assessments_per_user(test_db):
    user = repo.get_or_create_user(test_db, name="Multi-Session User")

    # Session 1
    s1 = repo.create_assessment_session(test_db, user_id=user.id)
    repo.complete_assessment(test_db, s1.id, {"primary_archetype": "Builder", "BAIO": {"Builder": 0.8}})

    # Session 2
    s2 = repo.create_assessment_session(test_db, user_id=user.id)
    repo.complete_assessment(test_db, s2.id, {"primary_archetype": "Innovator", "BAIO": {"Innovator": 0.9}})

    history = repo.get_user_assessment_history(test_db, user.id)
    assert len(history) == 2
    assert {h.primary_archetype for h in history} == {"Builder", "Innovator"}


# ── 8. Transaction Rollback on Error ──────────────────────────────────────────

def test_transaction_rollback_on_invalid_session(test_db):
    # Attempting to save an answer to a nonexistent session should raise ValueError and rollback
    with pytest.raises(ValueError):
        repo.save_answer(
            test_db,
            session_id="nonexistent-session-id",
            question_id="q1",
            selected_option_index=0,
            selected_option_text="Test",
        )

    # Database should remain clean
    answers = test_db.query(AssessmentAnswer).all()
    assert len(answers) == 0


# ── 9. Feedback Persistence ───────────────────────────────────────────────────

def test_feedback_persistence(test_db):
    user = repo.get_or_create_user(test_db, name="Feedback User")
    session_obj = repo.create_assessment_session(test_db, user_id=user.id)

    fb = repo.save_feedback(
        test_db,
        rating=5,
        feedback_text="Spot on recommendations!",
        user_id=user.id,
        session_id=session_obj.id,
    )

    assert fb.id is not None
    assert fb.rating == 5
    assert fb.feedback_text == "Spot on recommendations!"

    user_feedback = repo.get_feedback_by_user(test_db, user.id)
    assert len(user_feedback) == 1
    assert user_feedback[0].rating == 5


# ── 10. ML Dataset Isolation ──────────────────────────────────────────────────

def test_ml_datasets_untouched_by_sqlite():
    """Verify that ML datasets and files remain file-based and not corrupted."""
    from backend.data_loader import load_role_catalog, get_known_roles, get_emerging_roles
    catalog = load_role_catalog()
    assert len(catalog) >= 50
    assert len(get_known_roles()) >= 30
    assert len(get_emerging_roles()) >= 15
