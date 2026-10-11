"""
frontend/components/charts.py
==============================
Visual chart helpers for the Streamlit UI.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import streamlit as st


def render_baio_bar_chart(baio_scores: Dict[str, float]) -> None:
    """Render a clean horizontal or vertical bar chart for BAIO archetype scores."""
    df = pd.DataFrame(
        {
            "Archetype": list(baio_scores.keys()),
            "Score (%)": [round(v * 100, 1) for v in baio_scores.values()],
        }
    ).set_index("Archetype")
    
    st.bar_chart(df, color="#4F46E5")


def render_score_triad(personal_fit: float, future_potential: float, current_readiness: float) -> None:
    """Render three prominent metric score cards."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="🎯 Personal Match",
            value=f"{personal_fit * 100:.0f}%",
            help="How well your archetype, interests & work-style align with this role",
        )
    with col2:
        st.metric(
            label="📈 Future Potential",
            value=f"{future_potential * 100:.0f}%",
            help="Projected market demand growth from ML forecasts (Pipeline A / B)",
        )
    with col3:
        st.metric(
            label="⚡ Current Readiness",
            value=f"{current_readiness * 100:.0f}%",
            help="Skill alignment based on your declared technical skills",
        )


def render_component_breakdown_chart(component_scores: Dict[str, float]) -> None:
    """Render a breakdown of the 6 recommendation scoring components."""
    labels = {
        "baio_fit": "BAIO Archetype Fit",
        "interest_alignment": "Interest Alignment",
        "work_style_alignment": "Work-Style Alignment",
        "skill_alignment": "Skill Alignment",
        "future_opportunity": "Future Demand Growth",
        "emergence_signal": "Emergence Signal",
    }
    df = pd.DataFrame(
        {
            "Component": [labels.get(k, k.replace("_", " ").title()) for k in component_scores.keys()],
            "Score (%)": [round(v * 100, 1) for v in component_scores.values()],
        }
    ).set_index("Component")

    st.bar_chart(df, color="#06B6D4")


def render_comparison_chart(roles_data: List[Dict[str, float]]) -> None:
    """
    Render a comparison bar chart across 2-3 roles.
    roles_data = [{"Role": "Role A", "Personal Fit": 90, "Future Potential": 85, "Readiness": 60}, ...]
    """
    if not roles_data:
        return
    df = pd.DataFrame(roles_data).set_index("Role")
    st.bar_chart(df)
