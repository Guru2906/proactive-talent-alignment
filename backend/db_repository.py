"""
backend/db_repository.py
========================
Data Access Object (DAO) and Repository Layer for the SQLite application database.

Provides transactional, error-safe database operations for:
  - Users
  - Assessment Sessions & Questionnaire Answers
  - Role Recommendations & Historical Snapshots
  - User Feedback
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.db_models import (
    AssessmentAnswer,
    AssessmentSession,
    Feedback,
    RecommendationSnapshot,
    RoleRecommendation,
    User,
    utc_now,
)

logger = logging.getLogger(__name__)


# ── User Repository ────────────────────────────────────────────────────────────

def get_or_create_user(
    db: Session,
    user_id: Optional[str] = None,
    name: str = "Anonymous",
    email: Optional[str] = None,
    education: str = "",
    degree_stream: str = "",
    experience_level: str = "1-3 years",
    declared_skills: Optional[List[str]] = None,
    declared_interests: Optional[List[str]] = None,
) -> User:
    """Retrieve existing user by ID or create a new user profile."""
    try:
        user = None
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()
        elif email:
            user = db.query(User).filter(User.email == email).first()

        if user:
            # Update optional fields if newly provided
            if name and name != "Anonymous" and user.name == "Anonymous":
                user.name = name
            if education and not user.education:
                user.education = education
            if degree_stream and not user.degree_stream:
                user.degree_stream = degree_stream
            if experience_level and not user.experience_level:
                user.experience_level = experience_level
            if declared_skills:
                user.set_declared_skills(declared_skills)
            if declared_interests:
                user.set_declared_interests(declared_interests)
            db.commit()
            db.refresh(user)
            return user

        # Create new user
        new_id = user_id or str(uuid.uuid4())
        user = User(
            id=new_id,
            name=name or "Anonymous",
            email=email,
            education=education or "",
            degree_stream=degree_stream or "",
            experience_level=experience_level or "1-3 years",
            declared_skills=json.dumps(declared_skills or []),
            declared_interests=json.dumps(declared_interests or []),
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created new user profile: {user.id} ({user.name})")
        return user
    except Exception as e:
        db.rollback()
        logger.error(f"Error in get_or_create_user: {e}", exc_info=True)
        raise


def get_user(db: Session, user_id: str) -> Optional[User]:
    """Find user by UUID."""
    return db.query(User).filter(User.id == user_id).first()


def update_user_profile(
    db: Session,
    user_id: str,
    education: Optional[str] = None,
    degree_stream: Optional[str] = None,
    experience_level: Optional[str] = None,
    declared_skills: Optional[List[str]] = None,
    declared_interests: Optional[List[str]] = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
) -> Optional[User]:
    """Update an existing user's profile metadata."""
    try:
        user = get_user(db, user_id)
        if not user:
            return None

        if name is not None:
            user.name = name
        if email is not None:
            user.email = email
        if education is not None:
            user.education = education
        if degree_stream is not None:
            user.degree_stream = degree_stream
        if experience_level is not None:
            user.experience_level = experience_level
        if declared_skills is not None:
            user.set_declared_skills(declared_skills)
        if declared_interests is not None:
            user.set_declared_interests(declared_interests)

        user.updated_at = utc_now()
        db.commit()
        db.refresh(user)
        return user
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user profile {user_id}: {e}", exc_info=True)
        raise


def list_users(db: Session, limit: int = 50, offset: int = 0) -> List[User]:
    """List users ordered by creation date descending."""
    return db.query(User).order_by(desc(User.created_at)).offset(offset).limit(limit).all()


# ── Assessment Session Repository ──────────────────────────────────────────────

def create_assessment_session(
    db: Session,
    user_id: str,
    session_id: Optional[str] = None,
    education: str = "",
    degree_stream: str = "",
    experience_level: str = "1-3 years",
    declared_skills: Optional[List[str]] = None,
    declared_interests: Optional[List[str]] = None,
    total_questions: int = 20,
) -> AssessmentSession:
    """Create a new in-progress assessment session."""
    try:
        # Ensure user exists
        user = get_user(db, user_id)
        if not user:
            user = get_or_create_user(
                db,
                user_id=user_id,
                education=education,
                degree_stream=degree_stream,
                experience_level=experience_level,
                declared_skills=declared_skills,
                declared_interests=declared_interests,
            )

        sid = session_id or str(uuid.uuid4())
        session_obj = AssessmentSession(
            id=sid,
            user_id=user.id,
            started_at=utc_now(),
            status="in_progress",
            education=education or user.education,
            degree_stream=degree_stream or user.degree_stream,
            experience_level=experience_level or user.experience_level,
            declared_skills=json.dumps(declared_skills or user.get_declared_skills()),
            declared_interests=json.dumps(declared_interests or user.get_declared_interests()),
            total_questions=total_questions,
            answered_count=0,
        )
        db.add(session_obj)
        db.commit()
        db.refresh(session_obj)
        logger.info(f"Created assessment session {session_obj.id} for user {user.id}")
        return session_obj
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating assessment session: {e}", exc_info=True)
        raise


def get_assessment_session(db: Session, session_id: str) -> Optional[AssessmentSession]:
    """Retrieve an assessment session with all its relationships."""
    return db.query(AssessmentSession).filter(AssessmentSession.id == session_id).first()


def save_answer(
    db: Session,
    session_id: str,
    question_id: str,
    selected_option_index: int,
    selected_option_text: str,
    trait_weights: Optional[List[float]] = None,
    stage: Optional[str] = None,
) -> AssessmentAnswer:
    """Persist a single question answer to the database."""
    try:
        session_obj = get_assessment_session(db, session_id)
        if not session_obj:
            raise ValueError(f"Session '{session_id}' not found.")

        answer = AssessmentAnswer(
            session_id=session_id,
            question_id=question_id,
            selected_option_index=selected_option_index,
            selected_option_text=selected_option_text,
            trait_weights=json.dumps(trait_weights or []),
            stage=stage,
            answered_at=utc_now(),
        )
        db.add(answer)

        # Update session progress
        session_obj.answered_count = db.query(AssessmentAnswer).filter(
            AssessmentAnswer.session_id == session_id
        ).count() + 1
        db.commit()
        db.refresh(answer)
        return answer
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving answer for session {session_id}: {e}", exc_info=True)
        raise


def get_session_answers(db: Session, session_id: str) -> List[AssessmentAnswer]:
    """Return all recorded answers for a session."""
    return (
        db.query(AssessmentAnswer)
        .filter(AssessmentAnswer.session_id == session_id)
        .order_by(AssessmentAnswer.id)
        .all()
    )


def complete_assessment(
    db: Session,
    session_id: str,
    profile_dict: Dict[str, Any],
) -> Optional[AssessmentSession]:
    """
    Mark assessment completed and persist the computed BAIO scores,
    archetypes, and qualitative tags into the session record.
    """
    try:
        session_obj = get_assessment_session(db, session_id)
        if not session_obj:
            logger.warning(f"Session {session_id} not found when completing assessment.")
            return None

        baio = profile_dict.get("BAIO", {})
        session_obj.status = "completed"
        session_obj.completed_at = utc_now()
        session_obj.primary_archetype = profile_dict.get("primary_archetype")
        session_obj.secondary_archetype = profile_dict.get("secondary_archetype")
        session_obj.primary_archetype_description = profile_dict.get("primary_archetype_description", "")
        session_obj.secondary_archetype_description = profile_dict.get("secondary_archetype_description", "")
        session_obj.archetype_confidence = float(profile_dict.get("archetype_confidence", 0.0))
        session_obj.profile_confidence = float(profile_dict.get("profile_confidence", 0.0))

        session_obj.B_score = float(baio.get("Builder", 0.0))
        session_obj.A_score = float(baio.get("Analyst", 0.0))
        session_obj.I_score = float(baio.get("Innovator", 0.0))
        session_obj.O_score = float(baio.get("Operator", 0.0))
        session_obj.raw_vector = json.dumps(profile_dict.get("raw_vector", [0.0, 0.0, 0.0, 0.0]))

        session_obj.identified_interests = json.dumps(profile_dict.get("interests", []))
        session_obj.work_style = json.dumps(profile_dict.get("work_style", []))
        session_obj.tech_alignment = json.dumps(profile_dict.get("technology_interest", []))

        # Context updates if present
        if "education" in profile_dict and profile_dict["education"]:
            session_obj.education = profile_dict["education"]
        if "degree_stream" in profile_dict and profile_dict["degree_stream"]:
            session_obj.degree_stream = profile_dict["degree_stream"]
        if "experience_level" in profile_dict and profile_dict["experience_level"]:
            session_obj.experience_level = profile_dict["experience_level"]
        if "declared_skills" in profile_dict:
            session_obj.declared_skills = json.dumps(profile_dict["declared_skills"])
        if "declared_interests" in profile_dict:
            session_obj.declared_interests = json.dumps(profile_dict["declared_interests"])

        db.commit()
        db.refresh(session_obj)
        logger.info(f"Completed assessment session {session_id} - Archetype: {session_obj.primary_archetype}")
        return session_obj
    except Exception as e:
        db.rollback()
        logger.error(f"Error completing assessment session {session_id}: {e}", exc_info=True)
        raise


# ── Recommendation & Snapshot Repository ───────────────────────────────────────

def save_recommendations_and_snapshots(
    db: Session,
    session_id: str,
    top_recommendations: List[Dict[str, Any]],
    secondary_recommendations: Optional[List[Dict[str, Any]]] = None,
    model_version: str = "v2.0.0",
    pipeline_version: str = "v2.0.0",
) -> Tuple[List[RoleRecommendation], List[RecommendationSnapshot]]:
    """
    Persist generated role recommendations and their immutable historical snapshots.
    Replaces any prior recommendation records for this session while guaranteeing
    that snapshot data is preserved.
    """
    try:
        session_obj = get_assessment_session(db, session_id)
        if not session_obj:
            raise ValueError(f"Assessment session '{session_id}' not found.")

        # Clean existing recommendations for this session (if regenerating)
        db.query(RoleRecommendation).filter(RoleRecommendation.session_id == session_id).delete()
        db.flush()

        saved_recs: List[RoleRecommendation] = []
        saved_snapshots: List[RecommendationSnapshot] = []

        all_items: List[Tuple[str, Dict[str, Any]]] = [("top", r) for r in top_recommendations]
        if secondary_recommendations:
            all_items.extend([("secondary", r) for r in secondary_recommendations])

        for rec_type, r in all_items:
            # 1. RoleRecommendation record
            rec_obj = RoleRecommendation(
                session_id=session_id,
                role_id=r.get("role_id"),
                role_name=r.get("role_name", ""),
                rank=int(r.get("rank", 0)),
                match_score=float(r.get("final_score", r.get("personal_fit", 0.0))),
                personal_fit=float(r.get("personal_fit", 0.0)),
                future_potential=float(r.get("future_potential", 0.0)),
                current_readiness=float(r.get("current_readiness", 0.0)),
                recommendation_type=rec_type,
                source_type=r.get("source_type"),
                role_family=r.get("role_family"),
                confidence_label=r.get("confidence_label"),
                emergence_score=r.get("emergence_score"),
                growth_rate=r.get("growth_pct_6m") or r.get("growth_pct_12m"),
                growth_tier=r.get("growth_tier"),
                created_at=utc_now(),
            )
            db.add(rec_obj)
            db.flush()  # Generate rec_obj.id

            # 2. Immutable RecommendationSnapshot record
            cs = r.get("component_scores", {})
            if hasattr(cs, "dict"):
                cs = cs.dict()
            elif not isinstance(cs, dict):
                cs = {}

            snapshot_obj = RecommendationSnapshot(
                session_id=session_id,
                recommendation_id=rec_obj.id,
                role_id=r.get("role_id"),
                role_name=r.get("role_name", ""),
                rank=int(r.get("rank", 0)),
                recommendation_type=rec_type,
                personal_fit=float(r.get("personal_fit", 0.0)),
                future_potential=float(r.get("future_potential", 0.0)),
                current_readiness=float(r.get("current_readiness", 0.0)),
                final_score=float(r.get("final_score", 0.0)),
                emergence_score=r.get("emergence_score"),
                growth_pct_6m=r.get("growth_pct_6m"),
                growth_pct_12m=r.get("growth_pct_12m"),
                growth_tier=r.get("growth_tier"),
                source_type=r.get("source_type"),
                role_family=r.get("role_family"),
                role_category=r.get("role_category"),
                confidence_label=r.get("confidence_label"),
                description=r.get("description", ""),
                why_this_role=json.dumps(r.get("why_this_role", [])),
                challenges=json.dumps(r.get("challenges", [])),
                required_skills=json.dumps(r.get("required_skills", [])),
                frontier_skills=json.dumps(r.get("frontier_skills", [])),
                component_scores=json.dumps(cs),
                shap_top_drivers=json.dumps(r.get("shap_top_drivers", [])),
                model_version=model_version,
                pipeline_version=pipeline_version,
                generated_at=utc_now(),
            )
            db.add(snapshot_obj)

            saved_recs.append(rec_obj)
            saved_snapshots.append(snapshot_obj)

        db.commit()
        for rec in saved_recs:
            db.refresh(rec)
        for snap in saved_snapshots:
            db.refresh(snap)

        logger.info(f"Saved {len(saved_recs)} recommendations and {len(saved_snapshots)} snapshots for session {session_id}")
        return saved_recs, saved_snapshots
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving recommendations for session {session_id}: {e}", exc_info=True)
        raise


def get_session_recommendations(db: Session, session_id: str) -> List[RoleRecommendation]:
    """Retrieve all recommendations for a given session."""
    return (
        db.query(RoleRecommendation)
        .filter(RoleRecommendation.session_id == session_id)
        .order_by(RoleRecommendation.rank)
        .all()
    )


def get_session_snapshots(db: Session, session_id: str) -> List[RecommendationSnapshot]:
    """Retrieve all historical recommendation snapshots for a given session."""
    return (
        db.query(RecommendationSnapshot)
        .filter(RecommendationSnapshot.session_id == session_id)
        .order_by(RecommendationSnapshot.rank)
        .all()
    )


def get_user_assessment_history(db: Session, user_id: str, limit: int = 20) -> List[AssessmentSession]:
    """Retrieve all assessment sessions for a specific user ordered by start date."""
    return (
        db.query(AssessmentSession)
        .filter(AssessmentSession.user_id == user_id)
        .order_by(desc(AssessmentSession.started_at))
        .limit(limit)
        .all()
    )


def get_user_recommendations(db: Session, user_id: str, limit: int = 50) -> List[RoleRecommendation]:
    """Retrieve all historical recommendations across all sessions for a user."""
    return (
        db.query(RoleRecommendation)
        .join(AssessmentSession, RoleRecommendation.session_id == AssessmentSession.id)
        .filter(AssessmentSession.user_id == user_id)
        .order_by(desc(RoleRecommendation.created_at), RoleRecommendation.rank)
        .limit(limit)
        .all()
    )


# ── Feedback Repository ────────────────────────────────────────────────────────

def save_feedback(
    db: Session,
    rating: int,
    feedback_text: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    recommendation_id: Optional[int] = None,
    category: str = "general",
) -> Feedback:
    """Save user rating and optional qualitative feedback."""
    try:
        fb = Feedback(
            user_id=user_id,
            session_id=session_id,
            recommendation_id=recommendation_id,
            rating=rating,
            feedback_text=feedback_text or "",
            category=category,
            created_at=utc_now(),
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        logger.info(f"Saved feedback ID {fb.id} (Rating: {rating}/5)")
        return fb
    except Exception as e:
        db.rollback()
        logger.error(f"Error saving feedback: {e}", exc_info=True)
        raise


def get_feedback_by_session(db: Session, session_id: str) -> List[Feedback]:
    """Retrieve feedback submitted for a specific assessment session."""
    return db.query(Feedback).filter(Feedback.session_id == session_id).all()


def get_feedback_by_user(db: Session, user_id: str) -> List[Feedback]:
    """Retrieve feedback submitted by a specific user."""
    return db.query(Feedback).filter(Feedback.user_id == user_id).order_by(desc(Feedback.created_at)).all()
