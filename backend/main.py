"""
backend/main.py
=================
FastAPI backend for the Proactive IT Role & Skill Emergence Forecaster.

Integrates:
  - SQLite persistent application database (via SQLAlchemy ORM & db_repository)
  - Role Catalog (data/role_catalog/emerging_roles.json)
  - LightGBM 6M/12M forecasting artifacts (Pipeline A)
  - SBERT + HDBSCAN emergence discovery (Pipeline B)
  - Adaptive 3-Phase BAIO Questionnaire Engine
  - Multi-factor Role Matching & Recommendation Engine
  - Rule-based Skill Gap Analysis & Learning Roadmap Generator

Endpoints:
  - Users: POST /users, GET /users/{user_id}, GET /users/{user_id}/assessments, GET /users/{user_id}/recommendations
  - Assessment: POST /assessment/start, POST /assessment/profile, POST /assessment/next, POST /assessment/complete, GET /assessment/{session_id}/result
  - Recommendations: POST /recommendations, GET /recommendations/{session_id}, GET /assessment/{session_id}/snapshots
  - Roles: GET /roles/catalog, GET /roles/emerging, GET /roles/known, GET /roles/{role_id}
  - Skills: POST /skill-gap, POST /roadmap, GET /roadmap/{role_id}
  - Feedback: POST /feedback, GET /feedback
  - Health: GET /health
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Internal database & repository imports
from backend.database import get_db, init_db
from backend import db_repository as repo
from backend.data_loader import (
    build_role_index,
    get_emerging_roles,
    get_known_roles,
    get_role_by_id,
    get_role_by_name,
    load_role_catalog,
)
from backend.models import (
    AssessmentCompleteRequest,
    AssessmentResultResponse,
    AssessmentStartRequest,
    AssessmentSummary,
    BAIOScores,
    ComponentScores,
    FeedbackCreate,
    FeedbackResponse,
    NextRequest,
    NextResponse,
    QuestionPayload,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationSnapshotItem,
    RoadmapRequest,
    RoadmapResponse,
    RoadmapStage,
    RoleCard,
    RoleDetail,
    RoleSummary,
    SkillGapRequest,
    SkillGapResponse,
    StartResponse,
    UserCreate,
    UserProfile,
    UserProfileInput,
    UserResponse,
    UserUpdate,
)
from backend.questionnaire_engine import ARCHETYPE_NAMES, QuestionnaireEngine
from src.recommendation.profile_engine import build_user_profile
from src.recommendation.role_matcher import RoleMatcher
from src.recommendation.skill_gap import compute_skill_gap
from src.recommendation.roadmap import generate_roadmap

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOTAL_QUESTIONS = 20
STAGE_1_COUNT = 6
STAGE_2_COUNT = 9
STAGE_3_COUNT = 5
ARCHETYPE_KEYS = ["B", "A", "I", "O"]

MODEL_VERSION = "v2.0.0"
PIPELINE_VERSION = "v2.0.0"


# ── Session State (In-Memory Fast Cache) ───────────────────────────────────────

@dataclass
class SessionState:
    session_id: str
    user_id: str = ""
    user_vector: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    asked_question_ids: List[str] = field(default_factory=list)
    answers: List[dict] = field(default_factory=list)
    pending_question_ids: List[str] = field(default_factory=list)
    current_question_id: Optional[str] = None
    completed: bool = False
    # Extended profile fields (from UserProfileInput)
    education: str = ""
    degree_stream: str = ""
    experience_level: str = "1-3 years"
    declared_skills: List[str] = field(default_factory=list)
    declared_interests: List[str] = field(default_factory=list)
    # Cached computed objects
    user_profile: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None


# ── App Setup ─────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        logger.info("SQLite application database initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize database on startup: {e}", exc_info=True)
    yield


app = FastAPI(
    title="Proactive IT Role & Skill Emergence Forecaster",
    description=(
        "Production-grade API for personalized IT career recommendations "
        "powered by ML-based demand forecasting (Pipeline A: LightGBM) "
        "and emerging role discovery (Pipeline B: SBERT + HDBSCAN), "
        "with persistent SQLite storage for user assessments and snapshots."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load at startup
QUESTIONNAIRE_PATH = Path(__file__).resolve().parent / "questionnaire.json"
engine = QuestionnaireEngine(QUESTIONNAIRE_PATH)
matcher = RoleMatcher()
sessions: Dict[str, SessionState] = {}
session_lock = asyncio.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    total = float(vector.sum())
    if total <= 0:
        return np.full(4, 0.25, dtype=float)
    return vector / total


def _phase_label(answered_count: int) -> str:
    if answered_count < STAGE_1_COUNT:
        return "initial"
    if answered_count < STAGE_1_COUNT + STAGE_2_COUNT:
        return "adaptive"
    if answered_count < TOTAL_QUESTIONS:
        return "reconfirmation"
    return "completed"


def _public_question(state: SessionState) -> QuestionPayload:
    if not state.current_question_id:
        raise HTTPException(status_code=404, detail="No active question for this session.")
    payload = engine.to_public_question(
        question_id=state.current_question_id,
        question_number=len(state.asked_question_ids) + 1,
        total_questions=TOTAL_QUESTIONS,
    )
    return QuestionPayload(**payload)


def _build_user_profile(state: SessionState) -> Dict[str, Any]:
    """Build rich UserProfile from session state."""
    return build_user_profile(
        user_vector=state.user_vector,
        answers=state.answers,
        declared_skills=state.declared_skills,
        experience_level=state.experience_level,
        education=state.education,
        degree_stream=state.degree_stream,
        declared_interests=state.declared_interests,
        total_questions=TOTAL_QUESTIONS,
    )


def _profile_to_model(profile_dict: Dict[str, Any]) -> UserProfile:
    baio = profile_dict.get("BAIO", {})
    return UserProfile(
        BAIO=BAIOScores(
            Builder=baio.get("Builder", 0.25),
            Analyst=baio.get("Analyst", 0.25),
            Innovator=baio.get("Innovator", 0.25),
            Operator=baio.get("Operator", 0.25),
        ),
        primary_archetype=profile_dict.get("primary_archetype", "Builder"),
        secondary_archetype=profile_dict.get("secondary_archetype", "Analyst"),
        primary_archetype_description=profile_dict.get("primary_archetype_description", ""),
        secondary_archetype_description=profile_dict.get("secondary_archetype_description", ""),
        raw_vector=profile_dict.get("raw_vector", [0.25, 0.25, 0.25, 0.25]),
        archetype_confidence=profile_dict.get("archetype_confidence", 0.0),
        interests=profile_dict.get("interests", []),
        work_style=profile_dict.get("work_style", []),
        technology_interest=profile_dict.get("technology_interest", []),
        experience_level=profile_dict.get("experience_level", "1-3 years"),
        education=profile_dict.get("education", ""),
        degree_stream=profile_dict.get("degree_stream", ""),
        declared_skills=profile_dict.get("declared_skills", []),
        profile_confidence=profile_dict.get("profile_confidence", 0.0),
        n_answers=profile_dict.get("n_answers", 0),
        total_questions=profile_dict.get("total_questions", TOTAL_QUESTIONS),
    )


def _role_dict_to_card(r: Dict[str, Any]) -> RoleCard:
    cs = r.get("component_scores", {})
    if hasattr(cs, "dict"):
        cs = cs.dict()
    elif not isinstance(cs, dict):
        cs = {}
    return RoleCard(
        rank=r.get("rank", 0),
        role_id=r.get("role_id"),
        role_name=r.get("role_name", ""),
        source_type=r.get("source_type"),
        role_family=r.get("role_family"),
        role_category=r.get("role_category"),
        confidence_label=r.get("confidence_label"),
        personal_fit=r.get("personal_fit", 0.0),
        future_potential=r.get("future_potential", 0.0),
        current_readiness=r.get("current_readiness", 0.0),
        final_score=r.get("final_score", 0.0),
        component_scores=ComponentScores(
            baio_fit=cs.get("baio_fit", 0.0),
            interest_alignment=cs.get("interest_alignment", 0.0),
            work_style_alignment=cs.get("work_style_alignment", 0.0),
            skill_alignment=cs.get("skill_alignment", 0.0),
            future_opportunity=cs.get("future_opportunity", 0.0),
            emergence_signal=cs.get("emergence_signal", 0.0),
        ),
        why_this_role=r.get("why_this_role", []),
        challenges=r.get("challenges", []),
        required_skills=r.get("required_skills", []),
        frontier_skills=r.get("frontier_skills", []),
        growth_pct_6m=r.get("growth_pct_6m"),
        growth_pct_12m=r.get("growth_pct_12m"),
        growth_tier=r.get("growth_tier"),
        emergence_score=r.get("emergence_score"),
        shap_top_drivers=r.get("shap_top_drivers", []),
    )


def _snapshot_to_card(snap: Any) -> RoleCard:
    """Convert a database RecommendationSnapshot to RoleCard."""
    cs_dict = json.loads(snap.component_scores or "{}")
    return RoleCard(
        rank=snap.rank,
        role_id=snap.role_id,
        role_name=snap.role_name,
        source_type=snap.source_type,
        role_family=snap.role_family,
        role_category=snap.role_category,
        confidence_label=snap.confidence_label,
        personal_fit=snap.personal_fit,
        future_potential=snap.future_potential,
        current_readiness=snap.current_readiness,
        final_score=snap.final_score,
        component_scores=ComponentScores(
            baio_fit=cs_dict.get("baio_fit", 0.0),
            interest_alignment=cs_dict.get("interest_alignment", 0.0),
            work_style_alignment=cs_dict.get("work_style_alignment", 0.0),
            skill_alignment=cs_dict.get("skill_alignment", 0.0),
            future_opportunity=cs_dict.get("future_opportunity", 0.0),
            emergence_signal=cs_dict.get("emergence_signal", 0.0),
        ),
        why_this_role=json.loads(snap.why_this_role or "[]"),
        challenges=json.loads(snap.challenges or "[]"),
        required_skills=json.loads(snap.required_skills or "[]"),
        frontier_skills=json.loads(snap.frontier_skills or "[]"),
        growth_pct_6m=snap.growth_pct_6m,
        growth_pct_12m=snap.growth_pct_12m,
        growth_tier=snap.growth_tier,
        emergence_score=snap.emergence_score,
        shap_top_drivers=json.loads(snap.shap_top_drivers or "[]"),
    )


# ── Utility Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health(db: Session = Depends(get_db)) -> dict:
    catalog_count = len(load_role_catalog())
    db_connected = True
    user_count = 0
    try:
        user_count = len(repo.list_users(db, limit=1))
    except Exception:
        db_connected = False

    return {
        "status": "ok",
        "version": "2.0.0",
        "model_version": MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "database_connected": db_connected,
        "catalog_size": catalog_count,
        "questionnaire_questions": len(engine.questions),
    }


# ── User Management Endpoints ──────────────────────────────────────────────────

@app.post("/users", response_model=UserResponse, tags=["Users"], status_code=status.HTTP_201_CREATED)
async def create_user(request: UserCreate, db: Session = Depends(get_db)) -> UserResponse:
    """Register or create a persistent user profile."""
    user = repo.get_or_create_user(
        db,
        name=request.name,
        email=request.email,
        education=request.education,
        degree_stream=request.degree_stream,
        experience_level=request.experience_level,
        declared_skills=request.declared_skills,
        declared_interests=request.declared_interests,
    )
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        education=user.education,
        degree_stream=user.degree_stream,
        experience_level=user.experience_level,
        declared_skills=user.get_declared_skills(),
        declared_interests=user.get_declared_interests(),
        created_at=user.created_at.isoformat() if user.created_at else None,
        updated_at=user.updated_at.isoformat() if user.updated_at else None,
    )


@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"])
async def get_user_profile(user_id: str, db: Session = Depends(get_db)) -> UserResponse:
    """Retrieve user profile by ID."""
    user = repo.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found.")
    return UserResponse(
        id=user.id,
        name=user.name,
        email=user.email,
        education=user.education,
        degree_stream=user.degree_stream,
        experience_level=user.experience_level,
        declared_skills=user.get_declared_skills(),
        declared_interests=user.get_declared_interests(),
        created_at=user.created_at.isoformat() if user.created_at else None,
        updated_at=user.updated_at.isoformat() if user.updated_at else None,
    )


@app.get("/users/{user_id}/assessments", response_model=List[AssessmentSummary], tags=["Users"])
async def get_user_assessments(user_id: str, db: Session = Depends(get_db)) -> List[AssessmentSummary]:
    """Retrieve historical assessment sessions for a user."""
    sessions_list = repo.get_user_assessment_history(db, user_id)
    return [
        AssessmentSummary(
            session_id=s.id,
            user_id=s.user_id,
            started_at=s.started_at.isoformat() if s.started_at else "",
            completed_at=s.completed_at.isoformat() if s.completed_at else None,
            status=s.status,
            primary_archetype=s.primary_archetype,
            secondary_archetype=s.secondary_archetype,
            archetype_confidence=s.archetype_confidence,
            profile_confidence=s.profile_confidence,
            B_score=s.B_score,
            A_score=s.A_score,
            I_score=s.I_score,
            O_score=s.O_score,
            answered_count=s.answered_count,
            total_questions=s.total_questions,
        )
        for s in sessions_list
    ]


@app.get("/users/{user_id}/recommendations", response_model=List[RoleCard], tags=["Users"])
async def get_user_all_recommendations(user_id: str, db: Session = Depends(get_db)) -> List[RoleCard]:
    """Retrieve all historical role recommendations across sessions for a user."""
    recs = repo.get_user_recommendations(db, user_id)
    # Convert RoleRecommendation records to RoleCard
    results: List[RoleCard] = []
    for r in recs:
        # Check if snapshot exists
        if r.snapshots:
            results.append(_snapshot_to_card(r.snapshots[0]))
        else:
            results.append(
                RoleCard(
                    rank=r.rank,
                    role_id=r.role_id,
                    role_name=r.role_name,
                    source_type=r.source_type,
                    role_family=r.role_family,
                    confidence_label=r.confidence_label,
                    personal_fit=r.personal_fit,
                    future_potential=r.future_potential,
                    current_readiness=r.current_readiness,
                    final_score=r.match_score,
                    component_scores=ComponentScores(
                        baio_fit=0.0,
                        interest_alignment=0.0,
                        work_style_alignment=0.0,
                        skill_alignment=0.0,
                        future_opportunity=0.0,
                        emergence_signal=r.emergence_score or 0.0,
                    ),
                    why_this_role=[],
                    challenges=[],
                    required_skills=[],
                    frontier_skills=[],
                    growth_pct_6m=r.growth_rate,
                    growth_tier=r.growth_tier,
                    emergence_score=r.emergence_score,
                )
            )
    return results


# ── Assessment / Questionnaire Endpoints ──────────────────────────────────────

@app.post("/assessment/start", response_model=StartResponse, tags=["Assessment"])
async def start_assessment(
    request: Optional[AssessmentStartRequest] = None,
    db: Session = Depends(get_db),
) -> StartResponse:
    """
    Start a new questionnaire session and persist user and session records in SQLite.
    Backward-compatible: can be called with an empty body.
    """
    session_id = str(uuid.uuid4())
    stage1_ids = engine.stage1_question_ids(count=STAGE_1_COUNT)
    if not stage1_ids:
        raise HTTPException(status_code=500, detail="Questionnaire bank not configured correctly.")

    req = request or AssessmentStartRequest()
    user = repo.get_or_create_user(
        db,
        user_id=req.user_id,
        name=req.name or "Anonymous",
        email=req.email,
        education=req.education or "",
        degree_stream=req.degree_stream or "",
        experience_level=req.experience_level or "1-3 years",
        declared_skills=req.declared_skills,
        declared_interests=req.declared_interests,
    )

    # Persist session in SQLite
    repo.create_assessment_session(
        db,
        user_id=user.id,
        session_id=session_id,
        education=req.education or user.education,
        degree_stream=req.degree_stream or user.degree_stream,
        experience_level=req.experience_level or user.experience_level,
        declared_skills=req.declared_skills or user.get_declared_skills(),
        declared_interests=req.declared_interests or user.get_declared_interests(),
        total_questions=TOTAL_QUESTIONS,
    )

    # Create in-memory state
    state = SessionState(
        session_id=session_id,
        user_id=user.id,
        current_question_id=stage1_ids[0],
        pending_question_ids=stage1_ids[1:],
        education=req.education or user.education,
        degree_stream=req.degree_stream or user.degree_stream,
        experience_level=req.experience_level or user.experience_level,
        declared_skills=req.declared_skills or user.get_declared_skills(),
        declared_interests=req.declared_interests or user.get_declared_interests(),
    )
    async with session_lock:
        sessions[session_id] = state

    return StartResponse(session_id=session_id, question=_public_question(state), user_id=user.id)


@app.post("/assessment/profile", tags=["Assessment"])
async def update_profile(
    request: UserProfileInput,
    db: Session = Depends(get_db),
) -> dict:
    """
    Save user's profile-page inputs into in-memory state and SQLite database.
    """
    async with session_lock:
        state = sessions.get(request.session_id)
        if not state:
            # Check DB if not in memory
            db_session = repo.get_assessment_session(db, request.session_id)
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found.")
            state = SessionState(
                session_id=request.session_id,
                user_id=db_session.user_id,
            )
            sessions[request.session_id] = state

        state.education = request.education
        state.degree_stream = request.degree_stream
        state.experience_level = request.experience_level
        state.declared_skills = request.declared_skills
        state.declared_interests = request.declared_interests

        # Update in database
        repo.update_user_profile(
            db,
            user_id=state.user_id,
            education=request.education,
            degree_stream=request.degree_stream,
            experience_level=request.experience_level,
            declared_skills=request.declared_skills,
            declared_interests=request.declared_interests,
        )

    return {"status": "ok", "session_id": request.session_id}


@app.post("/assessment/next", response_model=NextResponse, tags=["Assessment"])
async def next_question(
    request: NextRequest,
    db: Session = Depends(get_db),
) -> NextResponse:
    """
    Submit an answer, persist to SQLite assessment_answers, and receive the next question.
    """
    async with session_lock:
        state = sessions.get(request.session_id)
        if not state:
            # Try to restore from database
            db_session = repo.get_assessment_session(db, request.session_id)
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found.")
            state = SessionState(session_id=request.session_id, user_id=db_session.user_id)
            sessions[request.session_id] = state

        if state.completed:
            raise HTTPException(status_code=400, detail="Questionnaire already completed.")
        if not state.current_question_id:
            raise HTTPException(status_code=400, detail="No active question in this session.")

        question = engine.get_question(state.current_question_id)
        options = question.get("options", [])
        if request.answer_index < 0 or request.answer_index >= len(options):
            raise HTTPException(status_code=400, detail="Invalid answer index.")

        selected_option = options[request.answer_index]
        weights = selected_option.get("weights", [0.0, 0.0, 0.0, 0.0])
        state.user_vector += np.array(weights, dtype=float)
        state.asked_question_ids.append(question["id"])
        state.answers.append({
            "question_id": question["id"],
            "answer_index": request.answer_index,
            "answer_text": selected_option["text"],
        })

        # Persist answer into SQLite
        repo.save_answer(
            db,
            session_id=request.session_id,
            question_id=question["id"],
            selected_option_index=request.answer_index,
            selected_option_text=selected_option["text"],
            trait_weights=weights,
            stage=question.get("stage"),
        )

        state.current_question_id = None
        answered_count = len(state.asked_question_ids)
        asked_set = set(state.asked_question_ids)

        # Adaptive branching
        if answered_count == STAGE_1_COUNT:
            top_indices = engine.top_archetype_indices(_normalize_vector(state.user_vector), top_k=3)
            state.pending_question_ids.extend(
                engine.stage2_question_ids(top_indices=top_indices, asked=asked_set, total_count=STAGE_2_COUNT)
            )
        elif answered_count == STAGE_1_COUNT + STAGE_2_COUNT:
            top_indices = engine.top_archetype_indices(_normalize_vector(state.user_vector), top_k=3)
            state.pending_question_ids.extend(
                engine.stage3_question_ids(top_indices=top_indices, asked=asked_set, total_count=STAGE_3_COUNT)
            )

        if answered_count >= TOTAL_QUESTIONS:
            state.completed = True
            profile_dict = _build_user_profile(state)
            state.user_profile = profile_dict
            repo.complete_assessment(db, request.session_id, profile_dict)
            return NextResponse(
                completed=True,
                answered_count=answered_count,
                total_questions=TOTAL_QUESTIONS,
                phase="completed",
                question=None,
            )

        if not state.pending_question_ids:
            needed = TOTAL_QUESTIONS - answered_count
            state.pending_question_ids.extend(engine.fallback_questions(asked=asked_set, needed=needed))

        if not state.pending_question_ids:
            state.completed = True
            profile_dict = _build_user_profile(state)
            state.user_profile = profile_dict
            repo.complete_assessment(db, request.session_id, profile_dict)
            return NextResponse(
                completed=True,
                answered_count=answered_count,
                total_questions=TOTAL_QUESTIONS,
                phase="completed",
                question=None,
            )

        state.current_question_id = state.pending_question_ids.pop(0)
        return NextResponse(
            completed=False,
            answered_count=answered_count,
            total_questions=TOTAL_QUESTIONS,
            phase=_phase_label(answered_count),
            question=_public_question(state),
        )


@app.post("/assessment/complete", response_model=UserProfile, tags=["Assessment"])
async def complete_assessment(
    request: AssessmentCompleteRequest,
    db: Session = Depends(get_db),
) -> UserProfile:
    """
    Mark assessment complete, calculate BAIO profile, and persist to SQLite.
    """
    async with session_lock:
        state = sessions.get(request.session_id)
        if not state:
            db_session = repo.get_assessment_session(db, request.session_id)
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found.")
            state = SessionState(session_id=request.session_id, user_id=db_session.user_id)
            sessions[request.session_id] = state

        # Update profile fields if provided
        if request.profile_input:
            pi = request.profile_input
            state.education = pi.education
            state.degree_stream = pi.degree_stream
            state.experience_level = pi.experience_level
            state.declared_skills = pi.declared_skills
            state.declared_interests = pi.declared_interests

        state.completed = True
        profile_dict = _build_user_profile(state)
        state.user_profile = profile_dict

        # Persist completion into SQLite
        repo.complete_assessment(db, request.session_id, profile_dict)

    return _profile_to_model(profile_dict)


@app.get("/assessment/{session_id}/result", response_model=AssessmentResultResponse, tags=["Assessment"])
async def get_assessment_result(
    session_id: str,
    db: Session = Depends(get_db),
) -> AssessmentResultResponse:
    """Retrieve full result details for an assessment session from SQLite."""
    session_obj = repo.get_assessment_session(db, session_id)
    if not session_obj:
        raise HTTPException(status_code=404, detail=f"Assessment session '{session_id}' not found.")

    profile = None
    if session_obj.status == "completed" and session_obj.primary_archetype:
        profile = UserProfile(
            BAIO=BAIOScores(
                Builder=session_obj.B_score or 0.0,
                Analyst=session_obj.A_score or 0.0,
                Innovator=session_obj.I_score or 0.0,
                Operator=session_obj.O_score or 0.0,
            ),
            primary_archetype=session_obj.primary_archetype,
            secondary_archetype=session_obj.secondary_archetype or "",
            primary_archetype_description=session_obj.primary_archetype_description or "",
            secondary_archetype_description=session_obj.secondary_archetype_description or "",
            raw_vector=session_obj.get_raw_vector(),
            archetype_confidence=session_obj.archetype_confidence or 0.0,
            interests=session_obj.get_interests(),
            work_style=session_obj.get_work_style(),
            technology_interest=session_obj.get_tech_alignment(),
            experience_level=session_obj.experience_level or "",
            education=session_obj.education or "",
            degree_stream=session_obj.degree_stream or "",
            declared_skills=json.loads(session_obj.declared_skills or "[]"),
            profile_confidence=session_obj.profile_confidence or 0.0,
            n_answers=len(session_obj.answers),
            total_questions=session_obj.total_questions,
        )

    return AssessmentResultResponse(
        session_id=session_obj.id,
        user_id=session_obj.user_id,
        status=session_obj.status,
        started_at=session_obj.started_at.isoformat() if session_obj.started_at else "",
        completed_at=session_obj.completed_at.isoformat() if session_obj.completed_at else None,
        user_profile=profile,
        answers_count=len(session_obj.answers),
        recommendations_count=len(session_obj.recommendations),
    )


# ── Recommendation Endpoints ───────────────────────────────────────────────────

@app.post("/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: RecommendationRequest,
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    """
    Generate personalized top-5 role recommendations and save immutable snapshots into SQLite.
    """
    async with session_lock:
        state = sessions.get(request.session_id)
        if not state:
            db_session = repo.get_assessment_session(db, request.session_id)
            if not db_session:
                raise HTTPException(status_code=404, detail="Session not found.")
            state = SessionState(session_id=request.session_id, user_id=db_session.user_id)
            sessions[request.session_id] = state

        if not state.completed:
            # Check if completed in DB
            db_session = repo.get_assessment_session(db, request.session_id)
            if db_session and db_session.status == "completed":
                state.completed = True
            else:
                raise HTTPException(status_code=400, detail="Assessment not completed yet.")

        # Build or reuse profile
        if state.user_profile is None:
            state.user_profile = _build_user_profile(state)
        profile_dict = state.user_profile

    # Run matcher (pure computation)
    top_recs, secondary_recs = matcher.rank(profile_dict)

    # Persist recommendations and immutable snapshots to SQLite
    repo.save_recommendations_and_snapshots(
        db,
        session_id=request.session_id,
        top_recommendations=top_recs,
        secondary_recommendations=secondary_recs,
        model_version=MODEL_VERSION,
        pipeline_version=PIPELINE_VERSION,
    )

    # Cache recommendations in memory
    async with session_lock:
        if state := sessions.get(request.session_id):
            state.recommendations = {
                "top": top_recs,
                "secondary": secondary_recs,
            }

    return RecommendationResponse(
        session_id=request.session_id,
        user_profile=_profile_to_model(profile_dict),
        top_recommendations=[_role_dict_to_card(r) for r in top_recs],
        secondary_recommendations=[_role_dict_to_card(r) for r in secondary_recs],
        model_version=MODEL_VERSION,
        pipeline_version=PIPELINE_VERSION,
    )


@app.get("/recommendations/{session_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_cached_recommendations(
    session_id: str,
    db: Session = Depends(get_db),
) -> RecommendationResponse:
    """Retrieve saved recommendations or historical snapshots for a session."""
    # First check memory
    async with session_lock:
        state = sessions.get(session_id)
        if state and state.recommendations and state.user_profile:
            return RecommendationResponse(
                session_id=session_id,
                user_profile=_profile_to_model(state.user_profile),
                top_recommendations=[_role_dict_to_card(r) for r in state.recommendations["top"]],
                secondary_recommendations=[_role_dict_to_card(r) for r in state.recommendations["secondary"]],
                model_version=MODEL_VERSION,
                pipeline_version=PIPELINE_VERSION,
            )

    # Fallback to database snapshots
    snapshots = repo.get_session_snapshots(db, session_id)
    if not snapshots:
        raise HTTPException(
            status_code=404,
            detail="No recommendations found for this session. Call POST /recommendations first."
        )

    db_session = repo.get_assessment_session(db, session_id)
    user_profile_model = None
    if db_session and db_session.primary_archetype:
        user_profile_model = UserProfile(
            BAIO=BAIOScores(
                Builder=db_session.B_score or 0.0,
                Analyst=db_session.A_score or 0.0,
                Innovator=db_session.I_score or 0.0,
                Operator=db_session.O_score or 0.0,
            ),
            primary_archetype=db_session.primary_archetype,
            secondary_archetype=db_session.secondary_archetype or "",
            primary_archetype_description=db_session.primary_archetype_description or "",
            secondary_archetype_description=db_session.secondary_archetype_description or "",
            raw_vector=db_session.get_raw_vector(),
            archetype_confidence=db_session.archetype_confidence or 0.0,
            interests=db_session.get_interests(),
            work_style=db_session.get_work_style(),
            technology_interest=db_session.get_tech_alignment(),
            experience_level=db_session.experience_level or "",
            education=db_session.education or "",
            degree_stream=db_session.degree_stream or "",
            declared_skills=json.loads(db_session.declared_skills or "[]"),
            profile_confidence=db_session.profile_confidence or 0.0,
            n_answers=len(db_session.answers),
            total_questions=db_session.total_questions,
        )
    else:
        user_profile_model = _profile_to_model({})

    top_cards = [_snapshot_to_card(s) for s in snapshots if s.recommendation_type == "top"]
    sec_cards = [_snapshot_to_card(s) for s in snapshots if s.recommendation_type == "secondary"]

    return RecommendationResponse(
        session_id=session_id,
        user_profile=user_profile_model,
        top_recommendations=top_cards,
        secondary_recommendations=sec_cards,
        model_version=snapshots[0].model_version if snapshots else MODEL_VERSION,
        pipeline_version=snapshots[0].pipeline_version if snapshots else PIPELINE_VERSION,
    )


@app.get("/assessment/{session_id}/snapshots", response_model=List[RecommendationSnapshotItem], tags=["Recommendations"])
async def get_snapshots_endpoint(
    session_id: str,
    db: Session = Depends(get_db),
) -> List[RecommendationSnapshotItem]:
    """Retrieve raw historical snapshot records for audit and compliance."""
    snapshots = repo.get_session_snapshots(db, session_id)
    if not snapshots:
        raise HTTPException(status_code=404, detail="No snapshots found for this session.")
    return [RecommendationSnapshotItem(**s.to_dict()) for s in snapshots]


# ── Feedback Endpoints ─────────────────────────────────────────────────────────

@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"], status_code=status.HTTP_201_CREATED)
async def submit_feedback(
    request: FeedbackCreate,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """Submit optional user feedback or star rating on recommendations."""
    fb = repo.save_feedback(
        db,
        rating=request.rating,
        feedback_text=request.feedback_text,
        user_id=request.user_id,
        session_id=request.session_id,
        recommendation_id=request.recommendation_id,
        category=request.category or "general",
    )
    return FeedbackResponse(
        id=fb.id,
        user_id=fb.user_id,
        session_id=fb.session_id,
        recommendation_id=fb.recommendation_id,
        rating=fb.rating,
        feedback_text=fb.feedback_text,
        category=fb.category,
        created_at=fb.created_at.isoformat() if fb.created_at else None,
    )


@app.get("/feedback", response_model=List[FeedbackResponse], tags=["Feedback"])
async def list_feedback_endpoint(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    db: Session = Depends(get_db),
) -> List[FeedbackResponse]:
    """List feedback filtered by session or user."""
    if session_id:
        items = repo.get_feedback_by_session(db, session_id)
    elif user_id:
        items = repo.get_feedback_by_user(db, user_id)
    else:
        items = db.query(repo.Feedback).order_by(desc(repo.Feedback.created_at)).limit(50).all()

    return [
        FeedbackResponse(
            id=fb.id,
            user_id=fb.user_id,
            session_id=fb.session_id,
            recommendation_id=fb.recommendation_id,
            rating=fb.rating,
            feedback_text=fb.feedback_text,
            category=fb.category,
            created_at=fb.created_at.isoformat() if fb.created_at else None,
        )
        for fb in items
    ]


# ── Role Catalog Endpoints ─────────────────────────────────────────────────────

@app.get("/roles/catalog", response_model=List[RoleSummary], tags=["Roles"])
async def get_full_catalog() -> List[RoleSummary]:
    """Return all roles in the catalog (known + emerging)."""
    catalog = load_role_catalog()
    return [
        RoleSummary(
            role_id=r.get("role_id"),
            role_name=r.get("role_name", ""),
            source_type=r.get("source_type", "unknown"),
            role_family=r.get("role_family"),
            role_category=r.get("role_category"),
            emergence_tier=r.get("emergence_tier"),
            confidence_label=r.get("confidence_label"),
            emergence_score=r.get("emergence_score"),
            forecast_6m_bps=r.get("forecast_6m_bps"),
            forecast_12m_bps=r.get("forecast_12m_bps"),
            growth_pct_6m=r.get("growth_pct_6m"),
            growth_pct_12m=r.get("growth_pct_12m"),
            growth_tier=r.get("growth_tier"),
        )
        for r in catalog
    ]


@app.get("/roles/emerging", response_model=List[RoleSummary], tags=["Roles"])
async def get_emerging_roles_endpoint() -> List[RoleSummary]:
    """Return only Pipeline B emerging cluster roles."""
    roles = get_emerging_roles()
    return [
        RoleSummary(
            role_id=r.get("role_id"),
            role_name=r.get("role_name", ""),
            source_type="emerging_cluster",
            role_family=r.get("role_family"),
            role_category=r.get("role_category"),
            emergence_tier=r.get("emergence_tier"),
            confidence_label=r.get("confidence_label"),
            emergence_score=r.get("emergence_score"),
            forecast_6m_bps=r.get("forecast_6m_bps"),
            forecast_12m_bps=r.get("forecast_12m_bps"),
            growth_pct_6m=r.get("growth_pct_6m"),
            growth_pct_12m=r.get("growth_pct_12m"),
            growth_tier=r.get("growth_tier"),
        )
        for r in roles
    ]


@app.get("/roles/known", response_model=List[RoleSummary], tags=["Roles"])
async def get_known_roles_endpoint() -> List[RoleSummary]:
    """Return only Pipeline A LightGBM-forecasted known roles."""
    roles = get_known_roles()
    return [
        RoleSummary(
            role_id=r.get("role_id"),
            role_name=r.get("role_name", ""),
            source_type="known_role",
            role_family=r.get("role_family"),
            role_category=r.get("role_category"),
            emergence_tier=r.get("emergence_tier"),
            confidence_label=r.get("confidence_label"),
            emergence_score=r.get("emergence_score"),
            forecast_6m_bps=r.get("forecast_6m_bps"),
            forecast_12m_bps=r.get("forecast_12m_bps"),
            growth_pct_6m=r.get("growth_pct_6m"),
            growth_pct_12m=r.get("growth_pct_12m"),
            growth_tier=r.get("growth_tier"),
        )
        for r in roles
    ]


@app.get("/roles/{role_id}", response_model=RoleDetail, tags=["Roles"])
async def get_role_detail(role_id: str) -> RoleDetail:
    """Get full details for a specific role by ID or name."""
    role = get_role_by_id(role_id)
    if role is None:
        role = get_role_by_name(role_id)
    if role is None:
        raise HTTPException(status_code=404, detail=f"Role '{role_id}' not found in catalog.")
    return RoleDetail(
        role_id=role.get("role_id"),
        role_name=role.get("role_name", ""),
        classification=role.get("classification"),
        source_type=role.get("source_type", "unknown"),
        forecast_method=role.get("forecast_method"),
        role_family=role.get("role_family"),
        role_category=role.get("role_category"),
        emergence_tier=role.get("emergence_tier"),
        emergence_score=role.get("emergence_score"),
        confidence=role.get("confidence"),
        confidence_label=role.get("confidence_label"),
        current_demand_bps=role.get("current_demand_bps"),
        forecast_6m_bps=role.get("forecast_6m_bps"),
        forecast_12m_bps=role.get("forecast_12m_bps"),
        growth_pct_6m=role.get("growth_pct_6m"),
        growth_pct_12m=role.get("growth_pct_12m"),
        growth_tier=role.get("growth_tier"),
        li_posting_count=role.get("li_posting_count"),
        li_unique_companies=role.get("li_unique_companies"),
        required_skills=role.get("required_skills") or [],
        frontier_skills=role.get("frontier_skills") or [],
        first_seen=role.get("first_seen"),
        last_seen=role.get("last_seen"),
        persistence_quarters=role.get("persistence_quarters"),
        market_breadth_score=role.get("market_breadth_score"),
        representative_titles=role.get("representative_titles"),
        closest_known_role=role.get("closest_known_role"),
        shap_top_drivers=role.get("shap_top_drivers") or [],
        evidence=role.get("evidence"),
    )


# ── Skill Gap Endpoint ─────────────────────────────────────────────────────────

@app.post("/skill-gap", response_model=SkillGapResponse, tags=["Analysis"])
async def get_skill_gap(request: SkillGapRequest) -> SkillGapResponse:
    """
    Compute skill gap for a role vs user's skills.
    Skills come from the ML role catalog — NOT fabricated.
    """
    role = get_role_by_id(request.role_id)
    if role is None:
        role = get_role_by_name(request.role_id)
    if role is None:
        raise HTTPException(status_code=404, detail=f"Role '{request.role_id}' not found.")

    gap = compute_skill_gap(role, request.user_skills)
    return SkillGapResponse(**gap)


# ── Roadmap Endpoint ───────────────────────────────────────────────────────────

@app.post("/roadmap", response_model=RoadmapResponse, tags=["Analysis"])
async def get_roadmap(request: RoadmapRequest) -> RoadmapResponse:
    """
    Generate a rule-based learning roadmap for a role.
    Method note is included in the response documenting it is rule-based.
    """
    role = get_role_by_id(request.role_id)
    if role is None:
        role = get_role_by_name(request.role_id)
    if role is None:
        raise HTTPException(status_code=404, detail=f"Role '{request.role_id}' not found.")

    gap = compute_skill_gap(role, request.user_skills)
    roadmap = generate_roadmap(role, gap)

    stages = [RoadmapStage(**s) for s in roadmap.get("stages", [])]
    return RoadmapResponse(
        role_name=roadmap["role_name"],
        method_note=roadmap["method_note"],
        already_have=roadmap.get("already_have", []),
        stages=stages,
        projects=roadmap.get("projects", []),
        total_stages=roadmap.get("total_stages", 0),
        data_available=roadmap.get("data_available", False),
        role_family=roadmap.get("role_family"),
    )


# ── Legacy endpoint (preserved for backward compatibility) ─────────────────────

@app.get("/roadmap/{role_id}", response_model=RoadmapResponse, tags=["Analysis"])
async def get_roadmap_by_id(role_id: str, user_skills: str = "") -> RoadmapResponse:
    """GET version of roadmap endpoint for direct browser access."""
    skills = [s.strip() for s in user_skills.split(",") if s.strip()] if user_skills else []
    req = RoadmapRequest(role_id=role_id, user_skills=skills)
    return await get_roadmap(req)
