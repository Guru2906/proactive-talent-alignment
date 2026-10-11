"""
frontend/app.py
================
Streamlit user-facing application for the:
"Proactive IT Role & Skill Emergence Forecaster"

Combines:
  1. Landing Page
  2. Profile / Basic Information
  3. Adaptive Questionnaire (20 questions, 3-phase branching)
  4. Your Career Archetype & Profile (BAIO vector + interests + work style)
  5. Top-5 Personalized Career Recommendations + Feedback Submission
  6. Role Deep Dive & Skill Gap Analysis
  7. Role Comparison Matrix
  8. Emerging Careers Explorer (General Market ML View)
  9. Assessment History & Snapshots (Persistent SQLite storage)
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import streamlit as st

# Add workspace root to sys.path
_this_file = Path(__file__).resolve()
BASE_DIR = _this_file.parents[2] if "app" in _this_file.parts else _this_file.parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Local database & ML imports
try:
    from backend.database import get_session, init_db
    from backend import db_repository as db_repo
    from backend.db_models import User, AssessmentSession, RecommendationSnapshot
except ImportError:
    from app.backend.database import get_session, init_db
    from app.backend import db_repository as db_repo
    from app.backend.db_models import User, AssessmentSession, RecommendationSnapshot

try:
    from backend.data_loader import (
        load_role_catalog,
        get_role_by_id,
        get_role_by_name,
        get_emerging_roles,
        get_known_roles,
    )
    from backend.questionnaire_engine import QuestionnaireEngine
except ImportError:
    from app.backend.data_loader import (
        load_role_catalog,
        get_role_by_id,
        get_role_by_name,
        get_emerging_roles,
        get_known_roles,
    )
    from app.backend.questionnaire_engine import QuestionnaireEngine

from src.recommendation.profile_engine import build_user_profile
from src.recommendation.role_matcher import RoleMatcher
from src.recommendation.skill_gap import compute_skill_gap
from src.recommendation.roadmap import generate_roadmap

try:
    from frontend.components.charts import (
        render_baio_bar_chart,
        render_score_triad,
        render_component_breakdown_chart,
        render_comparison_chart,
    )
    from frontend.components.cards import render_role_card, render_catalog_role_card
    from frontend.components.roadmap_display import render_skill_gap_section, render_learning_roadmap
except ImportError:
    from app.frontend.components.charts import (
        render_baio_bar_chart,
        render_score_triad,
        render_component_breakdown_chart,
        render_comparison_chart,
    )
    from app.frontend.components.cards import render_role_card, render_catalog_role_card
    from app.frontend.components.roadmap_display import render_skill_gap_section, render_learning_roadmap

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
_q_candidates = [
    BASE_DIR / "data" / "questionnaire" / "questionnaire.json",
    BASE_DIR / "backend" / "questionnaire.json",
    BASE_DIR / "app" / "backend" / "questionnaire.json",
]
QUESTIONNAIRE_PATH = next((p for p in _q_candidates if p.exists()), _q_candidates[0])
TOTAL_QUESTIONS = 20
STAGE_1_COUNT = 6
STAGE_2_COUNT = 9
STAGE_3_COUNT = 5

COMMON_SKILLS_LIST = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "SQL", "HTML/CSS", "React", "Node.js",
    "AWS", "Microsoft Azure", "Google Cloud", "Docker", "Kubernetes", "Git", "Linux",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP", "Computer Vision",
    "LLMs / Generative AI", "LangChain", "RAG", "Prompt Engineering", "Vector Databases",
    "Data Analysis", "Pandas", "Spark", "Data Modeling", "ETL", "Tableau", "Power BI",
    "DevOps", "CI/CD", "Terraform", "Ansible", "Cybersecurity", "Penetration Testing",
    "Network Security", "Agile / Scrum", "Product Management", "System Architecture"
]

INTEREST_DOMAINS = [
    "AI / Machine Learning", "Data Science / Analytics", "Cloud / Infrastructure",
    "Cybersecurity", "Software Engineering", "DevOps / SRE", "Product Management",
    "Robotics / Hardware", "Full Stack / Web"
]


# ── State Initialization ──────────────────────────────────────────────────────

def init_session():
    init_db()  # Ensure SQLite tables exist
    defaults = {
        "current_page": "Landing",
        "user_id": None,
        "user_name": "Student / Professional",
        "user_email": "",
        "session_id": None,
        "education": "Bachelor's",
        "degree_stream": "Computer Science",
        "experience_level": "1-3 years",
        "declared_skills": ["Python", "SQL", "Git"],
        "declared_interests": ["AI / Machine Learning", "Software Engineering"],
        "user_vector": np.zeros(4, dtype=float),
        "asked_question_ids": [],
        "answers": [],
        "pending_question_ids": [],
        "current_question_id": None,
        "assessment_completed": False,
        "user_profile": None,
        "top_recommendations": [],
        "secondary_recommendations": [],
        "selected_role_id": None,
        "compare_role_ids": [],
        "feedback_submitted": False,
        "engine": None,
        "matcher": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if st.session_state["engine"] is None and QUESTIONNAIRE_PATH.exists():
        st.session_state["engine"] = QuestionnaireEngine(QUESTIONNAIRE_PATH)
    if st.session_state["matcher"] is None:
        st.session_state["matcher"] = RoleMatcher()

    # Generate persistent anonymous user ID if none
    if not st.session_state.get("user_id"):
        st.session_state["user_id"] = str(uuid.uuid4())


# ── Assessment Engine Controller ──────────────────────────────────────────────

def start_assessment_flow():
    engine: QuestionnaireEngine = st.session_state["engine"]
    if engine is None:
        st.error("Questionnaire database not loaded.")
        return

    stage1_ids = engine.stage1_question_ids(count=STAGE_1_COUNT)
    session_id = str(uuid.uuid4())
    user_id = st.session_state.get("user_id") or str(uuid.uuid4())

    st.session_state["session_id"] = session_id
    st.session_state["user_id"] = user_id
    st.session_state["user_vector"] = np.zeros(4, dtype=float)
    st.session_state["asked_question_ids"] = []
    st.session_state["answers"] = []
    st.session_state["current_question_id"] = stage1_ids[0]
    st.session_state["pending_question_ids"] = stage1_ids[1:]
    st.session_state["assessment_completed"] = False
    st.session_state["user_profile"] = None
    st.session_state["top_recommendations"] = []
    st.session_state["secondary_recommendations"] = []
    st.session_state["feedback_submitted"] = False
    st.session_state["current_page"] = "Questionnaire"

    # Persist in SQLite
    try:
        db = get_session()
        try:
            db_repo.get_or_create_user(
                db,
                user_id=user_id,
                name=st.session_state.get("user_name", "Anonymous"),
                email=st.session_state.get("user_email") or None,
                education=st.session_state.get("education", ""),
                degree_stream=st.session_state.get("degree_stream", ""),
                experience_level=st.session_state.get("experience_level", ""),
                declared_skills=st.session_state.get("declared_skills", []),
                declared_interests=st.session_state.get("declared_interests", []),
            )
            db_repo.create_assessment_session(
                db,
                user_id=user_id,
                session_id=session_id,
                education=st.session_state.get("education", ""),
                degree_stream=st.session_state.get("degree_stream", ""),
                experience_level=st.session_state.get("experience_level", ""),
                declared_skills=st.session_state.get("declared_skills", []),
                declared_interests=st.session_state.get("declared_interests", []),
                total_questions=TOTAL_QUESTIONS,
            )
        finally:
            db.close()
    except Exception as e:
        print(f"Warning: Could not persist assessment start to SQLite: {e}")


def submit_questionnaire_answer(answer_index: int):
    engine: QuestionnaireEngine = st.session_state["engine"]
    qid = st.session_state["current_question_id"]
    question = engine.get_question(qid)
    option = question["options"][answer_index]
    weights = option.get("weights", [0.0, 0.0, 0.0, 0.0])

    # Accumulate BAIO weights
    st.session_state["user_vector"] += np.array(weights, dtype=float)
    st.session_state["asked_question_ids"].append(qid)
    st.session_state["answers"].append({
        "question_id": qid,
        "answer_index": answer_index,
        "answer_text": option["text"],
    })

    # Persist answer into SQLite
    session_id = st.session_state.get("session_id")
    if session_id:
        try:
            db = get_session()
            try:
                db_repo.save_answer(
                    db,
                    session_id=session_id,
                    question_id=qid,
                    selected_option_index=answer_index,
                    selected_option_text=option["text"],
                    trait_weights=weights,
                    stage=question.get("stage"),
                )
            finally:
                db.close()
        except Exception as e:
            print(f"Warning: Could not persist answer to SQLite: {e}")

    answered_count = len(st.session_state["asked_question_ids"])
    asked_set = set(st.session_state["asked_question_ids"])

    # Adaptive Branching logic
    if answered_count == STAGE_1_COUNT:
        norm_v = st.session_state["user_vector"] / max(st.session_state["user_vector"].sum(), 1e-6)
        top_indices = engine.top_archetype_indices(norm_v, top_k=3)
        st.session_state["pending_question_ids"].extend(
            engine.stage2_question_ids(top_indices=top_indices, asked=asked_set, total_count=STAGE_2_COUNT)
        )
    elif answered_count == STAGE_1_COUNT + STAGE_2_COUNT:
        norm_v = st.session_state["user_vector"] / max(st.session_state["user_vector"].sum(), 1e-6)
        top_indices = engine.top_archetype_indices(norm_v, top_k=3)
        st.session_state["pending_question_ids"].extend(
            engine.stage3_question_ids(top_indices=top_indices, asked=asked_set, total_count=STAGE_3_COUNT)
        )

    if answered_count >= TOTAL_QUESTIONS or not st.session_state["pending_question_ids"]:
        finish_assessment_flow()
    else:
        st.session_state["current_question_id"] = st.session_state["pending_question_ids"].pop(0)


def finish_assessment_flow():
    st.session_state["assessment_completed"] = True
    profile = build_user_profile(
        user_vector=st.session_state["user_vector"],
        answers=st.session_state["answers"],
        declared_skills=st.session_state["declared_skills"],
        experience_level=st.session_state["experience_level"],
        education=st.session_state["education"],
        degree_stream=st.session_state["degree_stream"],
        declared_interests=st.session_state["declared_interests"],
        total_questions=TOTAL_QUESTIONS,
    )
    st.session_state["user_profile"] = profile

    # Generate Recommendations
    matcher: RoleMatcher = st.session_state["matcher"]
    top_recs, sec_recs = matcher.rank(profile, top_n=5)
    st.session_state["top_recommendations"] = top_recs
    st.session_state["secondary_recommendations"] = sec_recs
    st.session_state["current_page"] = "Profile Summary"

    # Persist completion and snapshots to SQLite
    session_id = st.session_state.get("session_id")
    if session_id:
        try:
            db = get_session()
            try:
                db_repo.complete_assessment(db, session_id, profile)
                db_repo.save_recommendations_and_snapshots(
                    db,
                    session_id=session_id,
                    top_recommendations=top_recs,
                    secondary_recommendations=sec_recs,
                )
            finally:
                db.close()
        except Exception as e:
            print(f"Warning: Could not persist recommendations to SQLite: {e}")


# ── Page 1: Landing Page ──────────────────────────────────────────────────────

def render_landing_page():
    st.markdown("# 🚀 Discover the IT Careers of Tomorrow")
    st.markdown("#### *Understand your strengths, explore emerging careers, and build your personalized learning path.*")
    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Why This Platform?
        The technology landscape is shifting rapidly. Traditional job roles are being redefined by AI, cloud automation, and frontier technologies. 

        This intelligent system pairs **machine learning market emergence forecasts** with **psychometric work-style profiling** to guide your career evolution.
        """)

        with st.container(border=True):
            st.markdown("#### How It Works in 4 Steps:")
            st.markdown("1. **Understand Yourself** — Answer 20 adaptive scenarios to discover your BAIO archetype.")
            st.markdown("2. **Discover Future Careers** — Sourced from empirical forecasts across 60,000+ IT postings.")
            st.markdown("3. **Find Your Best Match** — Multi-factor scoring balancing passion, skill readiness, and market growth.")
            st.markdown("4. **Build Your Roadmap** — Staged, actionable skill milestones with targeted portfolio projects.")

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            if st.button("🌟 Start Career Assessment", type="primary", use_container_width=True):
                st.session_state["current_page"] = "Basic Info"
                st.rerun()
        with col_b2:
            if st.button("📜 View Past Assessments", use_container_width=True):
                st.session_state["current_page"] = "Assessment History"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 📊 Market Intelligence")
            st.metric("Analyzed Roles Catalog", "60 Roles", help="39 Canonical Known + 21 Discovered Emerging Clusters")
            st.metric("Historical Job Postings", "66,000+", help="Adzuna + LinkedIn + Naukri Longitudinal Dataset")
            st.metric("Forecasting Models", "LightGBM + SBERT + HDBSCAN")
            st.metric("Persistent Storage", "SQLite Relational Database", help="Preserves user sessions, answers, and immutable recommendation snapshots")

        if st.button("🔍 Explore Emerging Careers First", use_container_width=True):
            st.session_state["current_page"] = "Emerging Explorer"
            st.rerun()


# ── Page 2: Basic Info Page ───────────────────────────────────────────────────

def render_basic_info_page():
    st.markdown("## 👤 Step 1: Tell Us About Your Background")
    st.caption("We use this to tailor your skill readiness scores and domain recommendations.")
    st.divider()

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        user_name = st.text_input("Your Name / Alias (Optional):", value=st.session_state.get("user_name", "Student / Professional"))
        st.session_state["user_name"] = user_name
    with col_u2:
        user_email = st.text_input("Email Address (Optional):", value=st.session_state.get("user_email", ""))
        st.session_state["user_email"] = user_email

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        education = st.selectbox(
            "Highest Education Level",
            ["Bachelor's", "Master's", "PhD", "Associate Degree", "High School / Self-Taught"],
            index=0,
        )
        degree = st.text_input("Field / Stream of Study", value=st.session_state["degree_stream"])
        exp = st.selectbox(
            "Professional Experience Level",
            ["Student / Fresher", "0-1 years", "1-3 years", "3-5 years", "5+ years"],
            index=2,
        )

    with col2:
        interests = st.multiselect(
            "Primary Technology Interests",
            INTEREST_DOMAINS,
            default=st.session_state["declared_interests"],
        )
        skills = st.multiselect(
            "Select Your Current Technical Skills",
            COMMON_SKILLS_LIST,
            default=st.session_state["declared_skills"],
        )

    st.session_state["education"] = education
    st.session_state["degree_stream"] = degree
    st.session_state["experience_level"] = exp
    st.session_state["declared_interests"] = interests
    st.session_state["declared_skills"] = skills

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Continue to Assessment →", type="primary"):
            start_assessment_flow()
            st.rerun()
    with col_btn2:
        if st.button("← Back to Landing"):
            st.session_state["current_page"] = "Landing"
            st.rerun()


# ── Page 3: Questionnaire Page ────────────────────────────────────────────────

def render_questionnaire_page():
    engine: QuestionnaireEngine = st.session_state["engine"]
    qid = st.session_state["current_question_id"]
    if not qid:
        st.warning("Assessment session ended or not started.")
        if st.button("Restart Assessment"):
            start_assessment_flow()
            st.rerun()
        return

    q = engine.get_question(qid)
    answered_count = len(st.session_state["asked_question_ids"])
    current_q_num = answered_count + 1

    # Progress bar
    progress = answered_count / TOTAL_QUESTIONS
    st.progress(min(progress, 1.0))
    st.caption(f"**Question {current_q_num} of {TOTAL_QUESTIONS}** &nbsp;•&nbsp; Phase: `{q.get('stage', 'Core').title()}`")

    st.markdown(f"### {q['text']}")

    options = [opt["text"] for opt in q["options"]]
    choice = st.radio("Choose the option that best reflects your natural inclination:", options, key=f"q_radio_{qid}")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("Submit Answer →", type="primary"):
            idx = options.index(choice)
            submit_questionnaire_answer(idx)
            st.rerun()


# ── Page 4: Profile Summary Page ──────────────────────────────────────────────

def render_profile_summary_page():
    profile = st.session_state["user_profile"]
    if not profile:
        st.warning("No profile data found. Please complete the assessment.")
        return

    st.markdown("## 🧬 Your Career Archetype Profile")
    st.divider()

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown(f"### Primary: **{profile['primary_archetype']}**")
        st.info(profile["primary_archetype_description"])
        st.markdown(f"### Secondary: **{profile['secondary_archetype']}**")
        st.caption(profile["secondary_archetype_description"])
        st.metric("Profile Confidence", f"{profile['profile_confidence'] * 100:.0f}%")

    with col2:
        st.markdown("#### BAIO Score Breakdown")
        render_baio_bar_chart(profile["BAIO"])

    st.divider()

    # Derived tags
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("#### 🎯 Identified Interests")
        for item in profile.get("interests", []):
            st.markdown(f"- {item}")
    with col_t2:
        st.markdown("#### ⚙️ Work-Style Tendencies")
        for item in profile.get("work_style", []):
            st.markdown(f"- {item}")
    with col_t3:
        st.markdown("#### 💻 Tech Alignment")
        for item in profile.get("technology_interest", []):
            st.markdown(f"- {item}")

    st.divider()
    if st.button("🔮 View Your Personalized Career Matches →", type="primary", use_container_width=True):
        st.session_state["current_page"] = "Recommendations"
        st.rerun()


# ── Page 5: Career Matches / Recommendations ──────────────────────────────────

def render_recommendations_page():
    recs = st.session_state["top_recommendations"]
    sec_recs = st.session_state["secondary_recommendations"]

    st.markdown("## 🎯 Your Top 5 Personalized Career Recommendations")
    st.caption("Ranked by our multi-factor scoring engine balancing your personal fit, current readiness, and future market growth.")
    st.divider()

    if not recs:
        st.warning("No recommendations generated yet.")
        if st.button("Start Assessment"):
            start_assessment_flow()
            st.rerun()
        return

    for rec in recs:
        render_role_card(rec, key_prefix="top_rec")

    if sec_recs:
        with st.expander("📚 Secondary Career Paths to Consider", expanded=False):
            for s in sec_recs:
                render_role_card(s, key_prefix="sec_rec")

    # Feedback section
    st.divider()
    with st.expander("💬 Rate These Recommendations & Provide Feedback", expanded=not st.session_state.get("feedback_submitted")):
        if st.session_state.get("feedback_submitted"):
            st.success("Thank you! Your feedback has been saved to the database.")
        else:
            rating = st.slider("How accurate/helpful were these recommendations? (1 = Poor, 5 = Excellent)", 1, 5, 5)
            feedback_note = st.text_area("Any thoughts or suggestions on the recommendation accuracy?")
            if st.button("Submit Feedback", type="primary"):
                try:
                    db = get_session()
                    try:
                        db_repo.save_feedback(
                            db,
                            rating=rating,
                            feedback_text=feedback_note,
                            user_id=st.session_state.get("user_id"),
                            session_id=st.session_state.get("session_id"),
                        )
                        st.session_state["feedback_submitted"] = True
                        st.rerun()
                    finally:
                        db.close()
                except Exception as e:
                    st.error(f"Could not submit feedback: {e}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚖️ Compare Recommended Roles", use_container_width=True):
            st.session_state["compare_role_ids"] = [r["role_name"] for r in recs[:3]]
            st.session_state["current_page"] = "Compare Roles"
            st.rerun()
    with col2:
        if st.button("🔄 Retake Assessment", use_container_width=True):
            start_assessment_flow()
            st.rerun()


# ── Page 6: Role Details & Roadmap ────────────────────────────────────────────

def render_role_details_page():
    role_id = st.session_state.get("selected_role_id")
    if not role_id:
        st.warning("No role selected. Please choose a role from recommendations or the explorer.")
        if st.button("Back to Recommendations"):
            st.session_state["current_page"] = "Recommendations"
            st.rerun()
        return

    role = get_role_by_id(role_id) or get_role_by_name(role_id)
    if not role:
        st.error(f"Role '{role_id}' not found in catalog.")
        return

    role_name = role.get("role_name", "Role Details")
    source_type = role.get("source_type", "known_role")
    family = role.get("role_family") or role.get("role_category") or "General IT"
    growth_pct = role.get("growth_pct_6m") or role.get("forecast_growth_pct_cluster")

    st.markdown(f"## 🔎 {role_name}")
    st.caption(f"**Domain:** {family} | **Method:** `{role.get('forecast_method', 'ML Forecaster')}`")
    st.divider()

    # Metric Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Source Type", "Known Canonical" if source_type == "known_role" else "Discovered Emerging")
    with col2:
        st.metric("Forecast Horizon", "6M / 12M Demand" if source_type == "known_role" else "Quarterly Cluster")
    with col3:
        if growth_pct is not None:
            st.metric("Projected Growth", f"{growth_pct:+.1f}%")
        else:
            st.metric("Growth Tier", role.get("growth_tier") or "N/A")
    with col4:
        st.metric("Confidence", role.get("confidence_label") or "Medium")

    # SHAP & Market Evidence
    drivers = role.get("shap_top_drivers") or []
    rep_titles = role.get("representative_titles") or []
    evidence = role.get("evidence")

    if drivers or rep_titles or evidence:
        with st.container(border=True):
            st.markdown("### 📊 Market & Feature Attribution Evidence")
            if drivers:
                st.markdown("**Top Demand Drivers (SHAP TreeExplainer):**")
                for d in drivers:
                    st.markdown(f"- {d}")
            if rep_titles:
                st.markdown(f"**Discovered Job Titles Sample:** `{', '.join(rep_titles[:6])}`")
            if evidence and not drivers:
                st.markdown(f"**Data Sources:** `{evidence}`")

    st.divider()

    # Skill Gap Section
    st.markdown("### 🎯 Skill Gap Analysis")
    user_skills = st.session_state.get("declared_skills", [])
    gap = compute_skill_gap(role, user_skills)
    render_skill_gap_section(gap)

    st.divider()

    # Roadmap Section
    roadmap = generate_roadmap(role, gap)
    render_learning_roadmap(roadmap)

    st.divider()
    if st.button("← Back to Recommendations"):
        st.session_state["current_page"] = "Recommendations"
        st.rerun()


# ── Page 7: Role Comparison Matrix ────────────────────────────────────────────

def render_compare_roles_page():
    catalog = load_role_catalog()
    all_role_names = sorted([r["role_name"] for r in catalog])

    st.markdown("## ⚖️ Compare Career Paths")
    st.caption("Compare future growth, personal match scores, and readiness side-by-side.")
    st.divider()

    default_selected = st.session_state.get("compare_role_ids") or all_role_names[:3]
    selected_names = st.multiselect(
        "Select up to 3 roles to compare:",
        all_role_names,
        default=[n for n in default_selected if n in all_role_names][:3],
        max_selections=3,
    )

    if not selected_names:
        st.info("Select at least 1 role to view details.")
        return

    matcher: RoleMatcher = st.session_state["matcher"]
    user_profile = st.session_state.get("user_profile")
    if not user_profile:
        # Fallback profile if assessment skipped
        user_profile = build_user_profile(
            user_vector=np.array([0.25, 0.25, 0.25, 0.25]),
            answers=[],
            declared_skills=st.session_state.get("declared_skills", []),
        )

    comparison_metrics = []
    cols = st.columns(len(selected_names))

    for idx, name in enumerate(selected_names):
        role = get_role_by_name(name)
        if not role:
            continue

        scored = matcher._score_one_role(user_profile, role) or {}
        pf = scored.get("personal_fit", 0.0)
        fp = scored.get("future_potential", 0.0)
        cr = scored.get("current_readiness", 0.0)
        growth = role.get("growth_pct_6m") or role.get("forecast_growth_pct_cluster") or 0.0

        comparison_metrics.append({
            "Role": name,
            "Personal Match": round(pf * 100, 1),
            "Future Potential": round(fp * 100, 1),
            "Current Readiness": round(cr * 100, 1),
        })

        with cols[idx]:
            with st.container(border=True):
                st.markdown(f"### {name}")
                st.metric("Personal Match", f"{pf * 100:.0f}%")
                st.metric("Future Potential", f"{fp * 100:.0f}%")
                st.metric("Readiness", f"{cr * 100:.0f}%")
                st.markdown(f"**Domain:** {role.get('role_family') or 'IT'}")
                st.markdown(f"**Growth:** `{growth:+.0f}%`")
                if st.button("View Details", key=f"cmp_{name}"):
                    st.session_state["selected_role_id"] = name
                    st.session_state["current_page"] = "Role Details"
                    st.rerun()

    st.markdown("### 📊 Side-by-Side Score Comparison")
    render_comparison_chart(comparison_metrics)


# ── Page 8: Emerging Careers Explorer ─────────────────────────────────────────

def render_explorer_page():
    catalog = load_role_catalog()

    st.markdown("## 🌐 Explore Emerging IT Careers (Market View)")
    st.caption("Browse all 60 empirical roles from Pipeline A (LightGBM Forecasts) and Pipeline B (HDBSCAN Semantic Discovery).")
    st.divider()

    # Filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        search_query = st.text_input("🔍 Search roles by title or keyword:", "")
    with col2:
        source_filter = st.selectbox(
            "Filter by Origin Pipeline:",
            ["All Sources", "Known Canonical (Pipeline A)", "Discovered Emerging (Pipeline B)"]
        )
    with col3:
        sort_by = st.selectbox(
            "Sort by:",
            ["Role Name (A-Z)", "Emergence Score (High to Low)", "Projected Growth (High to Low)"]
        )

    # Filtering logic
    filtered = catalog.copy()
    if search_query:
        q = search_query.lower().strip()
        filtered = [
            r for r in filtered
            if q in r.get("role_name", "").lower()
            or q in (r.get("role_family") or "").lower()
            or any(q in s.lower() for s in r.get("required_skills", []))
        ]

    if source_filter == "Known Canonical (Pipeline A)":
        filtered = [r for r in filtered if r.get("source_type") == "known_role"]
    elif source_filter == "Discovered Emerging (Pipeline B)":
        filtered = [r for r in filtered if r.get("source_type") == "emerging_cluster"]

    # Sorting
    if sort_by == "Role Name (A-Z)":
        filtered.sort(key=lambda x: x.get("role_name", ""))
    elif sort_by == "Emergence Score (High to Low)":
        filtered.sort(key=lambda x: -(x.get("emergence_score") or 0.0))
    elif sort_by == "Projected Growth (High to Low)":
        filtered.sort(key=lambda x: -(x.get("growth_pct_6m") or x.get("forecast_growth_pct_cluster") or -999.0))

    st.markdown(f"**Showing {len(filtered)} roles:**")

    for r in filtered:
        render_catalog_role_card(r, key_prefix="exp")


# ── Page 9: Assessment History & Snapshots ─────────────────────────────────────

def render_assessment_history_page():
    st.markdown("## 📜 Assessment History & Snapshots")
    st.caption("Retrieve past psychometric profiles and historical recommendation snapshots preserved in SQLite.")
    st.divider()

    user_id = st.session_state.get("user_id")
    col1, col2 = st.columns([3, 1])
    with col1:
        search_id = st.text_input("Your User Profile ID:", value=user_id or "")
    with col2:
        if st.button("Refresh History", use_container_width=True):
            st.session_state["user_id"] = search_id.strip()
            st.rerun()

    lookup_id = search_id.strip() or user_id
    if not lookup_id:
        st.info("No User ID specified. Please enter a User ID or take an assessment.")
        return

    try:
        db = get_session()
        try:
            history = db_repo.get_user_assessment_history(db, lookup_id)
            user_obj = db_repo.get_user(db, lookup_id)
        finally:
            db.close()
    except Exception as e:
        st.error(f"Could not load assessment history from database: {e}")
        return

    if user_obj:
        st.markdown(f"**User:** `{user_obj.name}` &nbsp;|&nbsp; **Created:** `{user_obj.created_at.strftime('%Y-%m-%d %H:%M') if user_obj.created_at else 'N/A'}`")

    if not history:
        st.info("No assessment sessions found for this user ID. Take an assessment to record your career profile.")
        if st.button("Start New Assessment", type="primary"):
            st.session_state["current_page"] = "Basic Info"
            st.rerun()
        return

    st.markdown(f"### Found {len(history)} Assessment Session(s):")

    for idx, s in enumerate(history):
        started_str = s.started_at.strftime("%Y-%m-%d %H:%M UTC") if s.started_at else "Unknown date"
        with st.container(border=True):
            hcol1, hcol2, hcol3, hcol4 = st.columns([2, 2, 2, 2])
            with hcol1:
                st.markdown(f"**Session #{len(history) - idx}**")
                st.caption(f"ID: `{s.id[:8]}...`")
                st.caption(f"📅 {started_str}")
            with hcol2:
                st.markdown(f"**Status:** `{s.status.title()}`")
                st.markdown(f"**Primary:** {s.primary_archetype or 'N/A'}")
                st.caption(f"Secondary: {s.secondary_archetype or 'N/A'}")
            with hcol3:
                st.markdown("**BAIO Profile:**")
                st.caption(f"B: {s.B_score:.2f} | A: {s.A_score:.2f}")
                st.caption(f"I: {s.I_score:.2f} | O: {s.O_score:.2f}")
            with hcol4:
                if s.status == "completed":
                    if st.button(f"📂 Load Results", key=f"load_s_{s.id}", use_container_width=True):
                        _load_historical_session(s.id)
                        st.session_state["current_page"] = "Recommendations"
                        st.rerun()
                else:
                    st.caption(f"{s.answered_count}/{s.total_questions} answered")


def _load_historical_session(session_id: str) -> None:
    """Load a historical assessment session from SQLite snapshots into the UI."""
    try:
        db = get_session()
        try:
            s = db_repo.get_assessment_session(db, session_id)
            snapshots = db_repo.get_session_snapshots(db, session_id)
            if not s:
                return

            st.session_state["session_id"] = s.id
            st.session_state["user_id"] = s.user_id
            st.session_state["education"] = s.education or ""
            st.session_state["degree_stream"] = s.degree_stream or ""
            st.session_state["experience_level"] = s.experience_level or ""
            st.session_state["declared_skills"] = json.loads(s.declared_skills or "[]")
            st.session_state["declared_interests"] = json.loads(s.declared_interests or "[]")
            st.session_state["assessment_completed"] = True

            # Reconstruct profile
            st.session_state["user_profile"] = {
                "BAIO": {
                    "Builder": s.B_score or 0.0,
                    "Analyst": s.A_score or 0.0,
                    "Innovator": s.I_score or 0.0,
                    "Operator": s.O_score or 0.0,
                },
                "primary_archetype": s.primary_archetype or "Builder",
                "secondary_archetype": s.secondary_archetype or "Analyst",
                "primary_archetype_description": s.primary_archetype_description or "",
                "secondary_archetype_description": s.secondary_archetype_description or "",
                "raw_vector": s.get_raw_vector(),
                "archetype_confidence": s.archetype_confidence or 0.0,
                "interests": s.get_interests(),
                "work_style": s.get_work_style(),
                "technology_interest": s.get_tech_alignment(),
                "experience_level": s.experience_level or "",
                "education": s.education or "",
                "degree_stream": s.degree_stream or "",
                "declared_skills": json.loads(s.declared_skills or "[]"),
                "profile_confidence": s.profile_confidence or 0.0,
                "n_answers": len(s.answers),
                "total_questions": s.total_questions,
            }

            # Reconstruct recommendations from immutable snapshots
            top_recs = []
            sec_recs = []
            for snap in snapshots:
                card_dict = {
                    "rank": snap.rank,
                    "role_id": snap.role_id,
                    "role_name": snap.role_name,
                    "source_type": snap.source_type,
                    "role_family": snap.role_family,
                    "role_category": snap.role_category,
                    "confidence_label": snap.confidence_label,
                    "personal_fit": snap.personal_fit,
                    "future_potential": snap.future_potential,
                    "current_readiness": snap.current_readiness,
                    "final_score": snap.final_score,
                    "component_scores": json.loads(snap.component_scores or "{}"),
                    "why_this_role": json.loads(snap.why_this_role or "[]"),
                    "challenges": json.loads(snap.challenges or "[]"),
                    "required_skills": json.loads(snap.required_skills or "[]"),
                    "frontier_skills": json.loads(snap.frontier_skills or "[]"),
                    "growth_pct_6m": snap.growth_pct_6m,
                    "growth_pct_12m": snap.growth_pct_12m,
                    "growth_tier": snap.growth_tier,
                    "emergence_score": snap.emergence_score,
                    "shap_top_drivers": json.loads(snap.shap_top_drivers or "[]"),
                }
                if snap.recommendation_type == "top":
                    top_recs.append(card_dict)
                else:
                    sec_recs.append(card_dict)

            st.session_state["top_recommendations"] = top_recs
            st.session_state["secondary_recommendations"] = sec_recs
        finally:
            db.close()
    except Exception as e:
        st.error(f"Failed to restore historical session: {e}")


# ── Main Application Router ───────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Proactive IT Career Forecaster",
        page_icon="🔮",
        layout="wide",
    )
    init_session()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🔮 Navigation")
        pages = [
            "Landing",
            "Basic Info",
            "Questionnaire",
            "Profile Summary",
            "Recommendations",
            "Role Details",
            "Compare Roles",
            "Emerging Explorer",
            "Assessment History",
        ]
        curr_idx = pages.index(st.session_state["current_page"]) if st.session_state["current_page"] in pages else 0
        selected = st.radio("Go to:", pages, index=curr_idx)
        if selected != st.session_state["current_page"]:
            st.session_state["current_page"] = selected
            st.rerun()

        st.divider()
        st.caption(f"👤 **User ID:** `{str(st.session_state.get('user_id', ''))[:8]}...`")
        st.caption("🎓 **Capstone Project**")
        st.caption("Proactive IT Role & Skill Emergence Forecaster")

    # Routing
    page = st.session_state["current_page"]
    if page == "Landing":
        render_landing_page()
    elif page == "Basic Info":
        render_basic_info_page()
    elif page == "Questionnaire":
        render_questionnaire_page()
    elif page == "Profile Summary":
        render_profile_summary_page()
    elif page == "Recommendations":
        render_recommendations_page()
    elif page == "Role Details":
        render_role_details_page()
    elif page == "Compare Roles":
        render_compare_roles_page()
    elif page == "Emerging Explorer":
        render_explorer_page()
    elif page == "Assessment History":
        render_assessment_history_page()


if __name__ == "__main__":
    main()
