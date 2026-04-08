"""Deterministic grading and dense reward logic for the environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import AgentAction, EnvReward
from .tasks import PhaseSpec, SlotRule, TaskDefinition


GENERIC_HARMFUL_PHRASES = (
    "ignore the user",
    "doesn't matter",
    "do whatever",
    "password",
    "threaten",
)


@dataclass(frozen=True)
class GradeResult:
    score: float
    matched_slots: list[str]
    missing_slots: list[str]
    penalties: list[str]
    off_topic: bool


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _word_count(text: str) -> int:
    return len([word for word in text.replace("\n", " ").split(" ") if word.strip()])


def _slot_match(text: str, slot: SlotRule) -> bool:
    normalized = normalize_text(text)
    if slot.match_mode == "max_words":
        if slot.max_words is None:
            return False
        return _word_count(text) <= slot.max_words
    return any(keyword in normalized for keyword in slot.keywords)


def _slot_triggers(text: str, slot: SlotRule) -> bool:
    normalized = normalize_text(text)
    return any(trigger in normalized for trigger in slot.question_triggers)


def _is_off_topic(text: str, domain_keywords: Iterable[str]) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return True
    if any(keyword in normalized for keyword in domain_keywords):
        return False
    return _word_count(normalized) > 2


def grade_solution(text: str, task: TaskDefinition, phase: PhaseSpec) -> GradeResult:
    """Return a deterministic score from 0.0 to 1.0 for the active hidden intent."""

    matched_slots: list[str] = []
    missing_slots: list[str] = []
    penalties: list[str] = []

    total_weight = sum(slot.weight for slot in phase.slots)
    matched_weight = 0.0
    normalized = normalize_text(text)

    for slot in phase.slots:
        if _slot_match(text, slot):
            matched_slots.append(slot.name)
            matched_weight += slot.weight
        else:
            missing_slots.append(slot.name)

    penalty_value = 0.0
    for phrase in phase.forbidden_phrases:
        if phrase in normalized:
            penalties.append(f"forbidden:{phrase}")
            penalty_value += 0.25

    for phrase in GENERIC_HARMFUL_PHRASES:
        if phrase in normalized:
            penalties.append(f"harmful:{phrase}")
            penalty_value += 0.5

    base_score = matched_weight / total_weight if total_weight else 0.0
    score = max(0.0, min(1.0, base_score - penalty_value))
    off_topic = _is_off_topic(text, task.domain_keywords)

    return GradeResult(
        score=score,
        matched_slots=matched_slots,
        missing_slots=missing_slots,
        penalties=penalties,
        off_topic=off_topic,
    )


def detect_useful_clarifications(
    question: str,
    phase: PhaseSpec,
    unresolved_slots: set[str],
    already_asked: set[str],
) -> tuple[list[str], list[str]]:
    """Detect whether a clarification question targets unresolved slots."""

    useful: list[str] = []
    repeated: list[str] = []
    normalized = normalize_text(question)

    for slot in phase.slots:
        if slot.name not in unresolved_slots and slot.name in already_asked:
            if _slot_triggers(question, slot):
                repeated.append(slot.name)
            continue

        if slot.name in unresolved_slots and _slot_triggers(question, slot):
            if slot.name in already_asked:
                repeated.append(slot.name)
            else:
                useful.append(slot.name)

    if not useful and any(
        phrase in normalized
        for phrase in ("anything else", "what am i missing", "what changed", "what should i include")
    ):
        for slot in phase.slots:
            if slot.name in unresolved_slots and slot.name not in already_asked:
                useful.append(slot.name)
                break

    return useful, repeated


def reward_action(
    action: AgentAction,
    task: TaskDefinition,
    phase: PhaseSpec,
    resolved_slots: set[str],
    asked_slots: set[str],
) -> EnvReward:
    """Compute dense step reward for every action."""

    unresolved_slots = {slot.name for slot in phase.slots if slot.name not in resolved_slots}

    if action.action_type == "ask_clarification":
        useful, repeated = detect_useful_clarifications(
            question=action.content,
            phase=phase,
            unresolved_slots=unresolved_slots,
            already_asked=asked_slots,
        )
        if useful:
            bonus = min(0.3 + 0.05 * (len(useful) - 1), 0.4)
            return EnvReward(
                value=bonus,
                task_score=0.0,
                matched_slots=[],
                missing_slots=sorted(unresolved_slots),
                useful_clarifications=useful,
                penalties=[],
                notes=["Useful clarification question."],
                success=False,
            )
        if _is_off_topic(action.content, task.domain_keywords):
            return EnvReward(
                value=-1.0,
                task_score=0.0,
                matched_slots=[],
                missing_slots=sorted(unresolved_slots),
                useful_clarifications=[],
                penalties=["irrelevant_question"],
                notes=["Clarification request was irrelevant to the task."],
                success=False,
            )
        penalties = ["repeated_question"] if repeated else ["vague_question"]
        return EnvReward(
            value=-0.2,
            task_score=0.0,
            matched_slots=[],
            missing_slots=sorted(unresolved_slots),
            useful_clarifications=[],
            penalties=penalties,
            notes=["Clarification did not reduce ambiguity."],
            success=False,
        )

    grade = grade_solution(action.content, task=task, phase=phase)

    if action.action_type == "respond":
        if grade.off_topic:
            return EnvReward(
                value=-1.0,
                task_score=grade.score,
                matched_slots=grade.matched_slots,
                missing_slots=grade.missing_slots,
                useful_clarifications=[],
                penalties=grade.penalties + ["irrelevant_response"],
                notes=["Response was harmful or irrelevant."],
                success=False,
            )
        if grade.score >= task.partial_threshold:
            return EnvReward(
                value=0.5,
                task_score=grade.score,
                matched_slots=grade.matched_slots,
                missing_slots=grade.missing_slots,
                useful_clarifications=[],
                penalties=grade.penalties,
                notes=["Response shows partial understanding but is not the final proposal."],
                success=False,
            )
        return EnvReward(
            value=-0.5,
            task_score=grade.score,
            matched_slots=grade.matched_slots,
            missing_slots=grade.missing_slots,
            useful_clarifications=[],
            penalties=grade.penalties + ["premature_assumption"],
            notes=["Response made unsupported assumptions."],
            success=False,
        )

    if grade.off_topic:
        return EnvReward(
            value=-1.0,
            task_score=grade.score,
            matched_slots=grade.matched_slots,
            missing_slots=grade.missing_slots,
            useful_clarifications=[],
            penalties=grade.penalties + ["irrelevant_solution"],
            notes=["Solution does not address the user's intent."],
            success=False,
        )

    success = grade.score >= task.success_threshold and not grade.penalties
    if success:
        return EnvReward(
            value=1.0,
            task_score=grade.score,
            matched_slots=grade.matched_slots,
            missing_slots=grade.missing_slots,
            useful_clarifications=[],
            penalties=[],
            notes=["Final solution matches the current hidden intent."],
            success=True,
        )

    if grade.score >= task.partial_threshold:
        return EnvReward(
            value=0.5,
            task_score=grade.score,
            matched_slots=grade.matched_slots,
            missing_slots=grade.missing_slots,
            useful_clarifications=[],
            penalties=grade.penalties,
            notes=["Solution is partially correct but misses active constraints."],
            success=False,
        )

    return EnvReward(
        value=-0.5,
        task_score=grade.score,
        matched_slots=grade.matched_slots,
        missing_slots=grade.missing_slots,
        useful_clarifications=[],
        penalties=grade.penalties + ["wrong_solution"],
        notes=["Final proposal does not yet fit the hidden intent."],
        success=False,
    )
