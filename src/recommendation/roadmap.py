"""
src/recommendation/roadmap.py
================================
Rule-based learning roadmap generator.

IMPORTANT: This module is RULE-BASED, not ML-generated.
The ML pipeline produces role demand forecasts and skill associations.
This module uses those skills to produce a staged learning path.
This distinction is preserved in all output and documentation.

Input:
  - role: dict from role catalog
  - gap: dict from skill_gap.compute_skill_gap()
  - user_profile: dict from profile_engine (optional, for personalization)

Output:
  - LearningRoadmap dict with stages

Stage ordering is based on config/recommendation_config.yaml roadmap_stage_order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

BASE = Path(__file__).resolve().parents[2]
CFG_PATH = BASE / "config" / "recommendation_config.yaml"

with open(CFG_PATH, encoding="utf-8") as f:
    REC_CFG = yaml.safe_load(f)

STAGE_ORDER = REC_CFG.get("roadmap_stage_order", [])

# Suggested projects per role family (rule-based, not ML-generated)
ROLE_FAMILY_PROJECTS: Dict[str, List[str]] = {
    "Artificial Intelligence & Machine Learning": [
        "Train a classification model on a public dataset (e.g., Kaggle)",
        "Build a RAG chatbot with LangChain + a vector database",
        "Deploy an ML model as a REST API using FastAPI",
        "Fine-tune an open-source LLM on domain-specific data",
        "Create an AI agent that uses tools/APIs autonomously",
    ],
    "Data & Analytics": [
        "Build an end-to-end ETL pipeline (raw → cleaned → dashboard)",
        "Perform exploratory data analysis on a real-world dataset",
        "Create a SQL + Python data warehouse prototype",
        "Design and build a business analytics dashboard (Power BI / Tableau)",
        "Write a data quality monitoring script with automated alerts",
    ],
    "Cloud & Infrastructure": [
        "Deploy a 3-tier web app on AWS/GCP/Azure using IaC (Terraform)",
        "Set up a Kubernetes cluster and deploy a containerized app",
        "Design a multi-region, highly available architecture",
        "Implement cost monitoring and auto-scaling policies",
        "Build a CI/CD pipeline with GitHub Actions",
    ],
    "DevOps & Platform Engineering": [
        "Build a full CI/CD pipeline for a microservices app",
        "Set up observability: metrics, logging, alerting (Prometheus + Grafana)",
        "Automate infrastructure provisioning with Ansible or Terraform",
        "Implement blue-green deployment for zero-downtime releases",
        "Create a self-service developer platform using Backstage",
    ],
    "Security": [
        "Perform a vulnerability assessment on a test application",
        "Set up a SIEM system and write custom detection rules",
        "Build an automated security scanning pipeline for CI/CD",
        "Conduct a threat modeling exercise on a sample architecture",
        "Create an IAM policy framework for a cloud environment",
    ],
    "Software Engineering": [
        "Build a full-stack web application with REST APIs",
        "Contribute to an open-source project on GitHub",
        "Implement a microservices architecture with service discovery",
        "Write comprehensive unit and integration tests (>80% coverage)",
        "Build and publish an npm/PyPI package",
    ],
    "Embedded & Hardware": [
        "Build a sensor data logger using Raspberry Pi / Arduino",
        "Implement a real-time OS (RTOS) task on embedded hardware",
        "Develop firmware for a custom protocol (I2C / SPI / UART)",
        "Create a robotics simulation in ROS (Robot Operating System)",
        "Design a PCB and test a custom hardware prototype",
    ],
    "Product & Management": [
        "Write a complete PRD (Product Requirements Document) for a feature",
        "Conduct user research and synthesize findings into personas",
        "Build and prioritize a product roadmap using a scoring framework",
        "Run a sprint as Scrum Master with retrospective",
        "Define and track OKRs for a product initiative",
    ],
}

DEFAULT_PROJECTS = [
    "Build a portfolio project demonstrating core skills in this domain",
    "Contribute to open-source projects in this technology area",
    "Complete a capstone project integrating multiple skills",
    "Write a technical blog post documenting what you built",
]


def _classify_skill_to_stage(skill: str, stage_order: List[Dict]) -> int:
    """Return the stage index for a skill based on keyword matching."""
    skill_lower = skill.lower()
    for i, stage in enumerate(stage_order):
        for keyword in stage.get("keywords", []):
            if keyword in skill_lower:
                return i
    # Default to middle of the list if no match
    return len(stage_order) // 2


def generate_roadmap(
    role: Dict[str, Any],
    gap: Optional[Dict[str, Any]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a structured learning roadmap for a given role.

    This is a RULE-BASED generator. It does NOT use ML to produce the roadmap.
    The skills come from the ML artifact (role catalog), but the staging
    and ordering is determined by config/recommendation_config.yaml.

    Parameters
    ----------
    role : dict
        Role entry from the role catalog.
    gap : dict or None
        Output from compute_skill_gap(). If None, roadmap uses all required skills.
    user_profile : dict or None
        UserProfile dict. Used to skip skills the user already has.

    Returns
    -------
    dict with keys:
        role_name, method_note, stages, projects, total_stages
    """
    role_name = role.get("role_name", "Unknown Role")
    role_family = role.get("role_family") or ""

    # Skills to learn (priority = need_to_learn, then optional_advanced)
    if gap and gap.get("data_available"):
        skills_to_learn = gap.get("need_to_learn", [])
        advanced_skills = gap.get("optional_advanced", [])
        already_have = gap.get("already_have", [])
    else:
        # No gap computed — use all required skills
        skills_to_learn = role.get("required_skills") or []
        advanced_skills = role.get("frontier_skills") or []
        already_have = []

    if not skills_to_learn and not advanced_skills:
        return {
            "role_name": role_name,
            "method_note": "Rule-based roadmap generator (not ML-generated). Skill data unavailable for this role.",
            "stages": [],
            "projects": DEFAULT_PROJECTS[:3],
            "total_stages": 0,
            "data_available": False,
        }

    # Assign each skill to a stage
    skill_stage_map: Dict[int, List[str]] = {}
    for skill in skills_to_learn:
        idx = _classify_skill_to_stage(skill, STAGE_ORDER)
        skill_stage_map.setdefault(idx, []).append(skill)

    # Put advanced skills in the last custom stage
    if advanced_skills:
        last_idx = len(STAGE_ORDER) - 1
        skill_stage_map.setdefault(last_idx, []).extend(advanced_skills)

    # Build ordered stage list (only non-empty stages)
    stages: List[Dict[str, Any]] = []
    for i, stage_cfg in enumerate(STAGE_ORDER):
        skills_in_stage = skill_stage_map.get(i, [])
        if not skills_in_stage:
            continue
        stages.append({
            "stage_number": len(stages) + 1,
            "stage_name": stage_cfg["name"],
            "skills": skills_in_stage,
            "is_foundation": i < len(STAGE_ORDER) // 3,
            "is_advanced": i >= len(STAGE_ORDER) * 2 // 3,
        })

    # If no stages were created (skills didn't match any keyword), create a generic one
    if not stages and skills_to_learn:
        stages.append({
            "stage_number": 1,
            "stage_name": "Core Skills",
            "skills": skills_to_learn[:10],
            "is_foundation": True,
            "is_advanced": False,
        })

    # Projects
    family_projects = ROLE_FAMILY_PROJECTS.get(role_family, DEFAULT_PROJECTS)
    selected_projects = family_projects[:4]

    return {
        "role_name": role_name,
        "source_type": role.get("source_type", "unknown"),
        "method_note": (
            "This roadmap is rule-based (not ML-generated). "
            "Skills are sourced from ML prediction artifacts (Pipeline A/B). "
            "Stage ordering is derived from config/recommendation_config.yaml."
        ),
        "already_have": already_have,
        "stages": stages,
        "projects": selected_projects,
        "total_stages": len(stages),
        "data_available": True,
        "role_family": role_family,
    }
