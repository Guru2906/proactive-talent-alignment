"""
backend/models.py
==================
Pydantic request/response schemas for the upgraded FastAPI backend.

Extends the original initial-implementation models with:
- User Management schemas (UserCreate, UserResponse, UserUpdate)
- Assessment Lifecycle & Result schemas (AssessmentStartRequest, AssessmentSummary, AssessmentResultResponse)
- Recommendation Snapshot schemas (RecommendationSnapshotItem)
- Feedback schemas (FeedbackCreate, FeedbackResponse)
- Rich UserProfile
- RoleCard (recommendation result per role)
- RecommendationResult (full top-5 response)
- SkillGap schema
- LearningRoadmap schema
- Role catalog schemas

All original models preserved for backward compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ── Original models (preserved from initial implementation) ───────────────────

QuestionType = Literal["mcq", "likert", "binary"]


class QuestionOption(BaseModel):
    text: str


class QuestionPayload(BaseModel):
    id: str
    text: str
    type: QuestionType
    stage: Literal["common", "builder", "analyst", "innovator", "operator"]
    options: List[QuestionOption]
    question_number: int
    total_questions: int


class StartResponse(BaseModel):
    session_id: str
    question: QuestionPayload
    user_id: Optional[str] = None


class NextRequest(BaseModel):
    session_id: str
    answer_index: int = Field(ge=0)


class NextResponse(BaseModel):
    completed: bool
    answered_count: int
    total_questions: int
    phase: Literal["initial", "adaptive", "reconfirmation", "completed"]
    question: Optional[QuestionPayload] = None


# ── User Schemas ──────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    name: str = "Anonymous"
    email: Optional[str] = None
    education: Optional[str] = ""
    degree_stream: Optional[str] = ""
    experience_level: Optional[str] = "1-3 years"
    declared_skills: List[str] = Field(default_factory=list)
    declared_interests: List[str] = Field(default_factory=list)


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    education: Optional[str] = None
    degree_stream: Optional[str] = None
    experience_level: Optional[str] = None
    declared_skills: Optional[List[str]] = None
    declared_interests: Optional[List[str]] = None


class UserResponse(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    education: Optional[str] = ""
    degree_stream: Optional[str] = ""
    experience_level: Optional[str] = ""
    declared_skills: List[str] = Field(default_factory=list)
    declared_interests: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ── Assessment Start & Profile Schemas ────────────────────────────────────────

class AssessmentStartRequest(BaseModel):
    """Optional payload when starting an assessment."""
    user_id: Optional[str] = None
    name: Optional[str] = "Anonymous"
    email: Optional[str] = None
    education: Optional[str] = ""
    degree_stream: Optional[str] = ""
    experience_level: Optional[str] = "1-3 years"
    declared_skills: List[str] = Field(default_factory=list)
    declared_interests: List[str] = Field(default_factory=list)


class UserProfileInput(BaseModel):
    """Submitted from the Profile / Basic Info page before questionnaire."""
    session_id: str
    education: str = ""
    degree_stream: str = ""
    experience_level: str = "1-3 years"
    declared_skills: List[str] = Field(default_factory=list)
    declared_interests: List[str] = Field(default_factory=list)


# ── Rich User Profile Schemas ─────────────────────────────────────────────────

class BAIOScores(BaseModel):
    Builder: float
    Analyst: float
    Innovator: float
    Operator: float


class UserProfile(BaseModel):
    """Rich multi-dimensional user profile, output after questionnaire."""
    BAIO: BAIOScores
    primary_archetype: str
    secondary_archetype: str
    primary_archetype_description: str
    secondary_archetype_description: str
    raw_vector: List[float]
    archetype_confidence: float
    interests: List[str]
    work_style: List[str]
    technology_interest: List[str]
    experience_level: str
    education: str
    degree_stream: str
    declared_skills: List[str]
    profile_confidence: float
    n_answers: int
    total_questions: int


class AssessmentCompleteRequest(BaseModel):
    session_id: str
    profile_input: Optional[UserProfileInput] = None


class AssessmentSummary(BaseModel):
    """Compact summary of a historical assessment session."""
    session_id: str
    user_id: str
    started_at: str
    completed_at: Optional[str] = None
    status: str
    primary_archetype: Optional[str] = None
    secondary_archetype: Optional[str] = None
    archetype_confidence: Optional[float] = 0.0
    profile_confidence: Optional[float] = 0.0
    B_score: Optional[float] = 0.0
    A_score: Optional[float] = 0.0
    I_score: Optional[float] = 0.0
    O_score: Optional[float] = 0.0
    answered_count: int
    total_questions: int


class AssessmentResultResponse(BaseModel):
    """Full detail of an assessment session result."""
    session_id: str
    user_id: str
    status: str
    started_at: str
    completed_at: Optional[str] = None
    user_profile: Optional[UserProfile] = None
    answers_count: int
    recommendations_count: int


# ── Recommendation Schemas ───────────────────────────────────────────────────

class ComponentScores(BaseModel):
    baio_fit: float
    interest_alignment: float
    work_style_alignment: float
    skill_alignment: float
    future_opportunity: float
    emergence_signal: float


class RoleCard(BaseModel):
    """One recommended role in the top-5 result."""
    rank: int
    role_id: Optional[str] = None
    role_name: str
    source_type: Optional[str] = None
    role_family: Optional[str] = None
    role_category: Optional[str] = None
    confidence_label: Optional[str] = None

    # The three user-facing output scores (shown as percentages in UI)
    personal_fit: float        # 0.0–1.0 → shown as X%
    future_potential: float    # 0.0–1.0 → shown as X%
    current_readiness: float   # 0.0–1.0 → shown as X%
    final_score: float         # internal ranking score

    # Component transparency
    component_scores: ComponentScores

    # Explanations
    why_this_role: List[str]
    challenges: List[str]

    # Skills
    required_skills: List[str]
    frontier_skills: List[str]

    # Growth data (may be None if unavailable)
    growth_pct_6m: Optional[float] = None
    growth_pct_12m: Optional[float] = None
    growth_tier: Optional[str] = None
    emergence_score: Optional[float] = None
    shap_top_drivers: List[str] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    session_id: str


class RecommendationResponse(BaseModel):
    session_id: str
    user_profile: UserProfile
    top_recommendations: List[RoleCard]
    secondary_recommendations: List[RoleCard]
    model_version: Optional[str] = "v2.0.0"
    pipeline_version: Optional[str] = "v2.0.0"


class RecommendationSnapshotItem(BaseModel):
    """Frozen snapshot of a historical recommendation."""
    id: int
    session_id: str
    role_id: Optional[str] = None
    role_name: str
    rank: int
    recommendation_type: str
    personal_fit: float
    future_potential: float
    current_readiness: float
    final_score: float
    emergence_score: Optional[float] = None
    growth_pct_6m: Optional[float] = None
    growth_pct_12m: Optional[float] = None
    growth_tier: Optional[str] = None
    source_type: Optional[str] = None
    role_family: Optional[str] = None
    role_category: Optional[str] = None
    confidence_label: Optional[str] = None
    description: Optional[str] = None
    why_this_role: List[str] = Field(default_factory=list)
    challenges: List[str] = Field(default_factory=list)
    required_skills: List[str] = Field(default_factory=list)
    frontier_skills: List[str] = Field(default_factory=list)
    component_scores: Dict[str, Any] = Field(default_factory=dict)
    shap_top_drivers: List[str] = Field(default_factory=list)
    model_version: str
    pipeline_version: str
    generated_at: Optional[str] = None


# ── Feedback Schemas ──────────────────────────────────────────────────────────

class FeedbackCreate(BaseModel):
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    recommendation_id: Optional[int] = None
    rating: int = Field(ge=1, le=5)
    feedback_text: Optional[str] = ""
    category: Optional[str] = "general"


class FeedbackResponse(BaseModel):
    id: int
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    recommendation_id: Optional[int] = None
    rating: int
    feedback_text: Optional[str] = ""
    category: Optional[str] = "general"
    created_at: Optional[str] = None


# ── Skill Gap schemas ─────────────────────────────────────────────────────────

class SkillGapRequest(BaseModel):
    role_id: str
    user_skills: List[str] = Field(default_factory=list)


class SkillGapResponse(BaseModel):
    role_name: str
    role_id: Optional[str] = None
    already_have: List[str]
    need_to_learn: List[str]
    optional_advanced: List[str]
    skill_overlap_pct: Optional[float] = None
    data_available: bool
    message: Optional[str] = None


# ── Roadmap schemas ───────────────────────────────────────────────────────────

class RoadmapStage(BaseModel):
    stage_number: int
    stage_name: str
    skills: List[str]
    is_foundation: bool
    is_advanced: bool


class RoadmapRequest(BaseModel):
    role_id: str
    user_skills: List[str] = Field(default_factory=list)


class RoadmapResponse(BaseModel):
    role_name: str
    method_note: str   # Always documents that this is rule-based, not ML
    already_have: List[str]
    stages: List[RoadmapStage]
    projects: List[str]
    total_stages: int
    data_available: bool
    role_family: Optional[str] = None


# ── Role Catalog schemas ──────────────────────────────────────────────────────

class RoleSummary(BaseModel):
    """Lightweight role summary for listings."""
    role_id: Optional[str] = None
    role_name: str
    source_type: str
    role_family: Optional[str] = None
    role_category: Optional[str] = None
    emergence_tier: Optional[str] = None
    confidence_label: Optional[str] = None
    emergence_score: Optional[float] = None
    forecast_6m_bps: Optional[float] = None
    forecast_12m_bps: Optional[float] = None
    growth_pct_6m: Optional[float] = None
    growth_pct_12m: Optional[float] = None
    growth_tier: Optional[str] = None


class RoleDetail(BaseModel):
    """Full role details for the Role Details page."""
    role_id: Optional[str] = None
    role_name: str
    classification: Optional[str] = None
    source_type: str
    forecast_method: Optional[str] = None
    role_family: Optional[str] = None
    role_category: Optional[str] = None
    emergence_tier: Optional[str] = None
    emergence_score: Optional[float] = None
    confidence: Optional[float] = None
    confidence_label: Optional[str] = None
    current_demand_bps: Optional[float] = None
    forecast_6m_bps: Optional[float] = None
    forecast_12m_bps: Optional[float] = None
    growth_pct_6m: Optional[float] = None
    growth_pct_12m: Optional[float] = None
    growth_tier: Optional[str] = None
    li_posting_count: Optional[int] = None
    li_unique_companies: Optional[int] = None
    required_skills: List[str] = Field(default_factory=list)
    frontier_skills: List[str] = Field(default_factory=list)
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    persistence_quarters: Optional[int] = None
    market_breadth_score: Optional[float] = None
    representative_titles: Optional[List[str]] = None
    closest_known_role: Optional[str] = None
    shap_top_drivers: List[str] = Field(default_factory=list)
    evidence: Optional[str] = None


# ── Legacy models (kept for backward compatibility) ────────────────────────────

class CareerMatch(BaseModel):
    role: str
    title: str
    score: float


class ResultRequest(BaseModel):
    session_id: str


class ResultResponse(BaseModel):
    primary_archetype: str
    secondary_archetype: str
    baio_scores: List[float]
    confidence: float
    top_career_matches: List[CareerMatch]
    secondary_suggestions: List[str]
