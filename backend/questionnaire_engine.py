from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Set

import numpy as np

ARCHETYPE_KEYS = ["B", "A", "I", "O"]
ARCHETYPE_NAMES = ["Builder", "Analyst", "Innovator", "Operator"]
INDEX_TO_STAGE = {
    0: "builder",
    1: "analyst",
    2: "innovator",
    3: "operator",
}


class QuestionnaireEngine:
    def __init__(self, question_bank_path: Path):
        raw_questions = json.loads(question_bank_path.read_text(encoding="utf-8"))
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("Questionnaire JSON must be a non-empty list.")

        self.questions: Dict[str, dict] = {}
        self.questions_by_phase: Dict[int, List[dict]] = {1: [], 2: [], 3: []}
        for question in raw_questions:
            qid = question["id"]
            if qid in self.questions:
                raise ValueError(f"Duplicate question id found: {qid}")
            self.questions[qid] = question
            self.questions_by_phase[question["phase"]].append(question)

        for phase in self.questions_by_phase:
            self.questions_by_phase[phase].sort(key=lambda q: q["order"])

    def get_question(self, question_id: str) -> dict:
        return self.questions[question_id]

    def stage1_question_ids(self, count: int = 6) -> List[str]:
        phase_questions = [q["id"] for q in self.questions_by_phase[1]]
        if len(phase_questions) < count:
            raise ValueError("Not enough phase-1 questions to start the questionnaire.")
        return phase_questions[:count]

    def stage2_question_ids(
        self,
        top_indices: Sequence[int],
        asked: Set[str],
        total_count: int = 9,
        strong_count: int = 5,
        targeted_count: int = 4,
    ) -> List[str]:
        phase_pool = [q for q in self.questions_by_phase[2] if q["id"] not in asked]
        strong_pool = [
            q
            for q in phase_pool
            if q.get("category") == "strong" and q.get("type") == "mcq"
        ]
        targeted_pool = [q for q in phase_pool if q.get("category") == "targeted"]

        strong_selected = self._ranked_pick(strong_pool, top_indices, strong_count)
        targeted_selected = self._ranked_pick(
            [q for q in targeted_pool if q["id"] not in set(strong_selected)],
            top_indices,
            targeted_count,
            targeted=True,
        )

        selected = strong_selected + targeted_selected
        if len(selected) < total_count:
            selected_set = set(selected)
            fallback_pool = [q for q in phase_pool if q["id"] not in selected_set]
            selected.extend(
                self._ranked_pick(
                    fallback_pool,
                    top_indices,
                    total_count - len(selected),
                )
            )

        return selected[:total_count]

    def stage3_question_ids(
        self,
        top_indices: Sequence[int],
        asked: Set[str],
        total_count: int = 5,
    ) -> List[str]:
        phase_pool = [q for q in self.questions_by_phase[3] if q["id"] not in asked]
        return self._ranked_pick(phase_pool, top_indices, total_count, targeted=True)

    def fallback_questions(
        self,
        asked: Set[str],
        needed: int,
    ) -> List[str]:
        all_questions = (
            self.questions_by_phase[1] + self.questions_by_phase[2] + self.questions_by_phase[3]
        )
        remaining = [q["id"] for q in all_questions if q["id"] not in asked]
        return remaining[:needed]

    def to_public_question(
        self,
        question_id: str,
        question_number: int,
        total_questions: int,
    ) -> dict:
        q = self.get_question(question_id)
        return {
            "id": q["id"],
            "text": q["text"],
            "type": q["type"],
            "stage": q["stage"],
            "options": [{"text": option["text"]} for option in q["options"]],
            "question_number": question_number,
            "total_questions": total_questions,
        }

    def top_archetype_indices(self, vector: np.ndarray, top_k: int = 3) -> List[int]:
        ranked = sorted(range(len(vector)), key=lambda idx: (-float(vector[idx]), idx))
        return ranked[:top_k]

    def _ranked_pick(
        self,
        pool: Sequence[dict],
        top_indices: Sequence[int],
        count: int,
        targeted: bool = False,
    ) -> List[str]:
        scored = sorted(
            pool,
            key=lambda q: (
                self._question_relevance_score(q, top_indices, targeted),
                -q["order"],
            ),
            reverse=True,
        )
        return [q["id"] for q in scored[:count]]

    def _question_relevance_score(
        self,
        question: dict,
        top_indices: Sequence[int],
        targeted: bool,
    ) -> float:
        option_matrix = np.array([opt["weights"] for opt in question["options"]], dtype=float)
        if option_matrix.size == 0:
            return 0.0

        max_by_dim = option_matrix.max(axis=0)
        base = float(sum(max_by_dim[i] for i in top_indices))

        stage_bonus = 0.0
        focus_stages = {INDEX_TO_STAGE[i] for i in top_indices}
        if question["stage"] in focus_stages:
            stage_bonus = 0.35

        specificity_bonus = 0.0
        if targeted:
            active_dims = np.where(max_by_dim > 0.0)[0]
            if len(active_dims) <= 2:
                specificity_bonus = 0.25

        return base + stage_bonus + specificity_bonus
