"""
backend/db_models.py
====================
SQLAlchemy ORM models defining the relational database schema for the
Proactive IT Role & Skill Emergence Forecaster application.

Tables:
  - users: User profiles and metadata
  - assessment_sessions: Individual questionnaire sessions and resulting BAIO profile
  - assessment_answers: Question-by-question response history
  - role_recommendations: High-level generated role recommendations
  - recommendation_snapshots: Frozen historical snapshots of recommendation details (versioned)
  - feedback: Optional user ratings and qualitative feedback
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Index,
)
from sqlalchemy.orm import relationship

from backend.database import Base


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def generate_uuid() -> str:
    """Generate a standard UUID4 hex string."""
    return str(uuid.uuid4())


class User(Base):
    """Stores user identity, basic background, and metadata."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, default="Anonymous")
    email = Column(String(255), nullable=True, index=True)
    education = Column(String(100), nullable=True, default="")
    degree_stream = Column(String(255), nullable=True, default="")
    experience_level = Column(String(50), nullable=True, default="1-3 years")
    declared_skills = Column(Text, nullable=True, default="[]")       # JSON array
    declared_interests = Column(Text, nullable=True, default="[]")    # JSON array
    created_at = Column(DateTime, default=utc_now, nullable=False)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    assessment_sessions = relationship(
        "AssessmentSession",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(AssessmentSession.started_at)",
    )
    feedbacks = relationship(
        "Feedback",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def get_declared_skills(self) -> List[str]:
        if not self.declared_skills:
            return []
        try:
            return json.loads(self.declared_skills)
        except Exception:
            return []

    def set_declared_skills(self, skills: List[str]) -> None:
        self.declared_skills = json.dumps(skills or [])

    def get_declared_interests(self) -> List[str]:
        if not self.declared_interests:
            return []
        try:
            return json.loads(self.declared_interests)
        except Exception:
            return []

    def set_declared_interests(self, interests: List[str]) -> None:
        self.declared_interests = json.dumps(interests or [])


class AssessmentSession(Base):
    """
    Stores an individual assessment attempt with its progress,
    final BAIO scores, archetype classification, and profile confidence.
    """
    __tablename__ = "assessment_sessions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    started_at = Column(DateTime, default=utc_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(30), default="in_progress", nullable=False, index=True)  # in_progress, completed, abandoned

    # Archetype & BAIO results
    primary_archetype = Column(String(50), nullable=True)
    secondary_archetype = Column(String(50), nullable=True)
    primary_archetype_description = Column(Text, nullable=True)
    secondary_archetype_description = Column(Text, nullable=True)
    archetype_confidence = Column(Float, nullable=True, default=0.0)
    profile_confidence = Column(Float, nullable=True, default=0.0)

    # Individual BAIO component scores
    B_score = Column(Float, nullable=True, default=0.0)
    A_score = Column(Float, nullable=True, default=0.0)
    I_score = Column(Float, nullable=True, default=0.0)
    O_score = Column(Float, nullable=True, default=0.0)
    raw_vector = Column(Text, nullable=True, default="[0.0, 0.0, 0.0, 0.0]")  # JSON float array

    # Qualitative profile tags
    identified_interests = Column(Text, nullable=True, default="[]")  # JSON array
    work_style = Column(Text, nullable=True, default="[]")            # JSON array
    tech_alignment = Column(Text, nullable=True, default="[]")        # JSON array

    # Contextual background inputs at time of assessment
    education = Column(String(100), nullable=True, default="")
    degree_stream = Column(String(255), nullable=True, default="")
    experience_level = Column(String(50), nullable=True, default="")
    declared_skills = Column(Text, nullable=True, default="[]")
    declared_interests = Column(Text, nullable=True, default="[]")

    total_questions = Column(Integer, default=20, nullable=False)
    answered_count = Column(Integer, default=0, nullable=False)

    # Relationships
    user = relationship("User", back_populates="assessment_sessions")
    answers = relationship(
        "AssessmentAnswer",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AssessmentAnswer.id",
    )
    recommendations = relationship(
        "RoleRecommendation",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="RoleRecommendation.rank",
    )
    snapshots = relationship(
        "RecommendationSnapshot",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="RecommendationSnapshot.rank",
    )
    feedbacks = relationship(
        "Feedback",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def get_raw_vector(self) -> List[float]:
        try:
            return json.loads(self.raw_vector or "[0.0, 0.0, 0.0, 0.0]")
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

    def set_raw_vector(self, vec: List[float]) -> None:
        self.raw_vector = json.dumps([float(x) for x in vec])

    def get_interests(self) -> List[str]:
        try:
            return json.loads(self.identified_interests or "[]")
        except Exception:
            return []

    def get_work_style(self) -> List[str]:
        try:
            return json.loads(self.work_style or "[]")
        except Exception:
            return []

    def get_tech_alignment(self) -> List[str]:
        try:
            return json.loads(self.tech_alignment or "[]")
        except Exception:
            return []


class AssessmentAnswer(Base):
    """
    Stores an individual answer to an assessment question,
    enabling complete historical reconstruction and auditing.
    """
    __tablename__ = "assessment_answers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("assessment_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    question_id = Column(String(100), nullable=False, index=True)
    selected_option_index = Column(Integer, nullable=False)
    selected_option_text = Column(Text, nullable=False)
    trait_weights = Column(Text, nullable=True, default="[]")  # JSON weights
    stage = Column(String(50), nullable=True)                  # core, adaptive, reconfirmation
    answered_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    session = relationship("AssessmentSession", back_populates="answers")

    def get_trait_weights(self) -> List[float]:
        try:
            return json.loads(self.trait_weights or "[]")
        except Exception:
            return []

    def set_trait_weights(self, weights: List[float]) -> None:
        self.trait_weights = json.dumps([float(w) for w in weights])


class RoleRecommendation(Base):
    """
    Stores a recommendation entry generated for a session.
    """
    __tablename__ = "role_recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("assessment_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role_id = Column(String(100), nullable=True, index=True)
    role_name = Column(String(255), nullable=False, index=True)
    rank = Column(Integer, nullable=False)
    match_score = Column(Float, nullable=False)                # final_score
    personal_fit = Column(Float, nullable=False, default=0.0)
    future_potential = Column(Float, nullable=False, default=0.0)
    current_readiness = Column(Float, nullable=False, default=0.0)
    recommendation_type = Column(String(50), nullable=False, default="top")  # top, secondary, emerging_role, known_role
    source_type = Column(String(50), nullable=True)            # known_role, emerging_cluster
    role_family = Column(String(100), nullable=True)
    confidence_label = Column(String(50), nullable=True)
    emergence_score = Column(Float, nullable=True)
    growth_rate = Column(Float, nullable=True)
    growth_tier = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)

    # Relationships
    session = relationship("AssessmentSession", back_populates="recommendations")
    snapshots = relationship(
        "RecommendationSnapshot",
        back_populates="recommendation",
        cascade="all, delete-orphan",
    )
    feedbacks = relationship("Feedback", back_populates="recommendation")


class RecommendationSnapshot(Base):
    """
    Preserves an immutable snapshot of all details displayed to the user at recommendation time.
    Guarantees historical reproducibility even if ML models are retrained.
    """
    __tablename__ = "recommendation_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("assessment_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    recommendation_id = Column(Integer, ForeignKey("role_recommendations.id", ondelete="SET NULL"), nullable=True, index=True)
    role_id = Column(String(100), nullable=True)
    role_name = Column(String(255), nullable=False)
    rank = Column(Integer, nullable=False)
    recommendation_type = Column(String(50), nullable=False, default="top")

    # Metrics
    personal_fit = Column(Float, nullable=False)
    future_potential = Column(Float, nullable=False)
    current_readiness = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    emergence_score = Column(Float, nullable=True)
    growth_pct_6m = Column(Float, nullable=True)
    growth_pct_12m = Column(Float, nullable=True)
    growth_tier = Column(String(50), nullable=True)

    # Categorization
    source_type = Column(String(50), nullable=True)
    role_family = Column(String(100), nullable=True)
    role_category = Column(String(100), nullable=True)
    confidence_label = Column(String(50), nullable=True)
    description = Column(Text, nullable=True)

    # Structured explanations (JSON serialized)
    why_this_role = Column(Text, nullable=True, default="[]")      # JSON array
    challenges = Column(Text, nullable=True, default="[]")         # JSON array
    required_skills = Column(Text, nullable=True, default="[]")    # JSON array
    frontier_skills = Column(Text, nullable=True, default="[]")    # JSON array
    component_scores = Column(Text, nullable=True, default="{}")   # JSON object
    shap_top_drivers = Column(Text, nullable=True, default="[]")   # JSON array

    # Versioning
    model_version = Column(String(50), nullable=False, default="v2.0.0")
    pipeline_version = Column(String(50), nullable=False, default="v2.0.0")
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    session = relationship("AssessmentSession", back_populates="snapshots")
    recommendation = relationship("RoleRecommendation", back_populates="snapshots")

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot record to dictionary representation."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role_id": self.role_id,
            "role_name": self.role_name,
            "rank": self.rank,
            "recommendation_type": self.recommendation_type,
            "personal_fit": self.personal_fit,
            "future_potential": self.future_potential,
            "current_readiness": self.current_readiness,
            "final_score": self.final_score,
            "emergence_score": self.emergence_score,
            "growth_pct_6m": self.growth_pct_6m,
            "growth_pct_12m": self.growth_pct_12m,
            "growth_tier": self.growth_tier,
            "source_type": self.source_type,
            "role_family": self.role_family,
            "role_category": self.role_category,
            "confidence_label": self.confidence_label,
            "description": self.description,
            "why_this_role": json.loads(self.why_this_role or "[]"),
            "challenges": json.loads(self.challenges or "[]"),
            "required_skills": json.loads(self.required_skills or "[]"),
            "frontier_skills": json.loads(self.frontier_skills or "[]"),
            "component_scores": json.loads(self.component_scores or "{}"),
            "shap_top_drivers": json.loads(self.shap_top_drivers or "[]"),
            "model_version": self.model_version,
            "pipeline_version": self.pipeline_version,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


class Feedback(Base):
    """
    Stores optional qualitative feedback and 1-5 rating from users.
    """
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    session_id = Column(String(36), ForeignKey("assessment_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    recommendation_id = Column(Integer, ForeignKey("role_recommendations.id", ondelete="SET NULL"), nullable=True, index=True)
    rating = Column(Integer, nullable=False)  # 1 to 5
    feedback_text = Column(Text, nullable=True, default="")
    category = Column(String(100), nullable=True, default="general")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="feedbacks")
    session = relationship("AssessmentSession", back_populates="feedbacks")
    recommendation = relationship("RoleRecommendation", back_populates="feedbacks")


# Indexes for high performance querying
Index("idx_user_created_at", User.created_at)
Index("idx_session_user_status", AssessmentSession.user_id, AssessmentSession.status)
Index("idx_session_started_at", AssessmentSession.started_at)
Index("idx_answers_session_qid", AssessmentAnswer.session_id, AssessmentAnswer.question_id)
Index("idx_recs_session_rank", RoleRecommendation.session_id, RoleRecommendation.rank)
Index("idx_snapshots_session_rank", RecommendationSnapshot.session_id, RecommendationSnapshot.rank)
