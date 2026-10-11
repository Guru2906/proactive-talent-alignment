"""
tests/test_questionnaire.py
============================
Tests for questionnaire engine, staging, and adaptive branching.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from backend.questionnaire_engine import QuestionnaireEngine

QUESTIONNAIRE_PATH = Path(__file__).resolve().parents[1] / "backend" / "questionnaire.json"


def test_questionnaire_file_exists():
    assert QUESTIONNAIRE_PATH.exists()


def test_questionnaire_engine_load():
    engine = QuestionnaireEngine(QUESTIONNAIRE_PATH)
    assert len(engine.questions) > 0
    assert 1 in engine.questions_by_phase
    assert 2 in engine.questions_by_phase
    assert 3 in engine.questions_by_phase


def test_stage1_selection():
    engine = QuestionnaireEngine(QUESTIONNAIRE_PATH)
    stage1_ids = engine.stage1_question_ids(count=6)
    assert len(stage1_ids) == 6


def test_adaptive_stage2_branching():
    engine = QuestionnaireEngine(QUESTIONNAIRE_PATH)
    stage1_ids = engine.stage1_question_ids(count=6)
    asked = set(stage1_ids)

    # Builder top index = 0
    stage2_builder = engine.stage2_question_ids(top_indices=[0, 1, 2], asked=asked, total_count=9)
    assert len(stage2_builder) == 9
    assert len(set(stage2_builder) & asked) == 0  # No duplicates with stage 1


def test_to_public_question():
    engine = QuestionnaireEngine(QUESTIONNAIRE_PATH)
    first_id = engine.stage1_question_ids(count=1)[0]
    public_q = engine.to_public_question(first_id, question_number=1, total_questions=20)

    assert "id" in public_q
    assert "text" in public_q
    assert "options" in public_q
    assert len(public_q["options"]) >= 2
    # Ensure raw weights are not leaked in public question payload
    for opt in public_q["options"]:
        assert "weights" not in opt
