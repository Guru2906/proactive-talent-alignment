"""
frontend/components/cards.py
==============================
Reusable card rendering components for recommendation results and role catalogs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import streamlit as st


def render_role_card(
    role_card: Dict[str, Any],
    on_select_detail: Optional[Callable[[str], None]] = None,
    key_prefix: str = "rec",
) -> None:
    """Render a structured, explainable card for a recommended career role."""
    role_name = role_card.get("role_name", "Unknown Role")
    role_id = role_card.get("role_id") or role_name
    rank = role_card.get("rank", 1)
    personal_fit = role_card.get("personal_fit", 0.0)
    future_potential = role_card.get("future_potential", 0.0)
    current_readiness = role_card.get("current_readiness", 0.0)
    confidence_label = role_card.get("confidence_label", "Medium")
    source_type = role_card.get("source_type", "known_role")
    family = role_card.get("role_family") or role_card.get("role_category") or "IT Domain"

    with st.container(border=True):
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown(f"### #{rank} {role_name}")
            source_badge = "🔵 Forecasted Known Role (Pipeline A)" if source_type == "known_role" else "🟣 Discovered Emerging Role (Pipeline B)"
            st.caption(f"**Domain:** {family} | {source_badge}")

        with header_col2:
            st.markdown(f"**Confidence:** `{confidence_label}`")

        # Metric triad
        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 Personal Match", f"{personal_fit * 100:.0f}%")
        col2.metric("📈 Future Potential", f"{future_potential * 100:.0f}%")
        col3.metric("⚡ Current Readiness", f"{current_readiness * 100:.0f}%")

        # Explainable section
        with st.expander("💡 Why this role? (Explainability & Challenges)", expanded=False):
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("#### ✓ Key Matching Factors")
                strengths = role_card.get("why_this_role") or []
                if strengths:
                    for s in strengths:
                        st.markdown(f"- {s}")
                else:
                    st.markdown("- Strong alignment with your core profile attributes.")

            with exp_col2:
                st.markdown("#### ⚠ What May Be Challenging")
                challenges = role_card.get("challenges") or []
                if challenges:
                    for c in challenges:
                        st.markdown(f"- {c}")
                else:
                    st.markdown("- Continuous learning required as market technologies evolve.")

        # Action Buttons
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            if st.button("Explore Details →", key=f"{key_prefix}_detail_{role_id}_{rank}", type="primary"):
                if on_select_detail:
                    on_select_detail(role_id)
                st.session_state["selected_role_id"] = role_id
                st.session_state["current_page"] = "Role Details"
                st.rerun()


def render_catalog_role_card(
    role: Dict[str, Any],
    on_select_detail: Optional[Callable[[str], None]] = None,
    key_prefix: str = "cat",
) -> None:
    """Render a card for the general emerging/known role explorer."""
    role_name = role.get("role_name", "Unknown Role")
    role_id = role.get("role_id") or role_name
    source_type = role.get("source_type", "unknown")
    emergence_score = role.get("emergence_score")
    forecast_6m = role.get("forecast_6m_bps")
    growth_pct_6m = role.get("growth_pct_6m")
    confidence_label = role.get("confidence_label") or "Medium"
    family = role.get("role_family") or role.get("role_category") or "General IT"

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"#### {role_name}")
            st.caption(f"**Domain:** {family} | **Source:** `{source_type.replace('_', ' ').title()}`")
        with col2:
            if emergence_score is not None:
                st.metric("Emergence", f"{emergence_score:.2f}")
            elif growth_pct_6m is not None:
                st.metric("6M Growth", f"{growth_pct_6m:+.0f}%")
            else:
                st.markdown(f"`{confidence_label} Confidence`")

        if st.button("View Full Role Breakdown →", key=f"{key_prefix}_{role_id}"):
            st.session_state["selected_role_id"] = role_id
            st.session_state["current_page"] = "Role Details"
            st.rerun()
