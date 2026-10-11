"""
frontend/components/roadmap_display.py
=======================================
Visual learning roadmap and skill gap display component.
"""

from __future__ import annotations

from typing import Any, Dict, List
import streamlit as st


def render_skill_gap_section(gap: Dict[str, Any]) -> None:
    """Render the 3-bucket skill gap analysis view."""
    if not gap.get("data_available"):
        st.info("Skill data unavailable for this role in current empirical artifacts.")
        return

    overlap_pct = gap.get("skill_overlap_pct", 0.0)
    st.markdown(f"**Current Skill Overlap:** `{overlap_pct}%`")
    st.progress(min(max(overlap_pct / 100.0, 0.0), 1.0))

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("#### ✓ Already Have")
            skills = gap.get("already_have") or []
            if skills:
                for s in skills:
                    st.success(f"✓ {s}")
            else:
                st.caption("No declared skills match this role's core requirements.")

    with col2:
        with st.container(border=True):
            st.markdown("#### → Need to Learn")
            skills = gap.get("need_to_learn") or []
            if skills:
                for s in skills:
                    st.warning(f"→ {s}")
            else:
                st.caption("You have all identified core required skills!")

    with col3:
        with st.container(border=True):
            st.markdown("#### ◦ Optional / Frontier")
            skills = gap.get("optional_advanced") or []
            if skills:
                for s in skills:
                    st.info(f"◦ {s}")
            else:
                st.caption("No additional frontier skills listed.")


def render_learning_roadmap(roadmap: Dict[str, Any]) -> None:
    """Render structured roadmap stages and project suggestions."""
    method_note = roadmap.get("method_note")
    if method_note:
        st.caption(f"ℹ️ **Methodology Note:** {method_note}")

    stages: List[Dict[str, Any]] = roadmap.get("stages") or []
    if not stages:
        st.info("No roadmap stages available for this role.")
        return

    st.markdown("### 🗺️ Step-by-Step Learning Progression")

    for stage in stages:
        stage_num = stage.get("stage_number", 1)
        stage_name = stage.get("stage_name", "Core Stage")
        skills = stage.get("skills", [])
        is_foundation = stage.get("is_foundation", False)
        is_advanced = stage.get("is_advanced", False)

        badge = "🌱 Foundation" if is_foundation else ("🚀 Advanced / Frontier" if is_advanced else "⚙️ Core Mastery")

        with st.container(border=True):
            st.markdown(f"#### Stage {stage_num}: {stage_name} &nbsp; `{badge}`")
            skill_pills = " &nbsp;•&nbsp; ".join([f"**{s}**" for s in skills]) if skills else "Skills in progress"
            st.markdown(f"**Target Skills:** {skill_pills}")

    # Suggested Capstones & Projects
    projects: List[str] = roadmap.get("projects") or []
    if projects:
        st.markdown("### 🛠️ Portfolio Projects to Build")
        for idx, p in enumerate(projects, 1):
            st.markdown(f"**Project {idx}:** {p}")
