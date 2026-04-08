"""Baseline inference script for all three tasks."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from my_env.env import HumanConfusionGoalDriftEnv
from my_env.models import AgentAction, EnvObservation
from my_env.tasks import TASK_ORDER


GLOBAL_SEED = 17
ENV_NAME = "Human Confusion & Goal Drift Environment"


def compact(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


@dataclass
class PolicyResult:
    action: AgentAction
    error: str | None = None


class LLMBaselinePolicy:
    """OpenAI-compatible policy with deterministic heuristic fallback."""

    def __init__(self) -> None:
        self.api_base_url = os.getenv("API_BASE_URL", "")
        self.model_name = os.getenv("MODEL_NAME", "")
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self._rng = random.Random(GLOBAL_SEED)

        self._client: OpenAI | None = None
        api_key = self.hf_token or self.openai_api_key
        if self.model_name and (api_key or self.api_base_url.startswith("http://localhost")):
            self._client = OpenAI(
                base_url=self.api_base_url or None,
                api_key=api_key or "local-token",
            )

    @property
    def label(self) -> str:
        return self.model_name or "heuristic-fallback"

    def next_action(self, task_id: str, observation: EnvObservation) -> PolicyResult:
        if self._client is not None:
            try:
                return PolicyResult(action=self._llm_action(task_id, observation))
            except Exception as exc:  # pragma: no cover - runtime dependent
                fallback = self._heuristic_action(task_id, observation)
                return PolicyResult(action=fallback, error=f"llm_fallback:{type(exc).__name__}")
        return PolicyResult(action=self._heuristic_action(task_id, observation))

    def _llm_action(self, task_id: str, observation: EnvObservation) -> AgentAction:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are solving a reinforcement-learning environment where the user is ambiguous and may drift.\n"
                    "Choose exactly one action.\n"
                    "Return JSON with keys action_type and content.\n"
                    "Valid action_type values: respond, ask_clarification, propose_solution.\n"
                    "Prefer clarification when essential constraints are missing.\n"
                    "When proposing a final solution, make it concise and directly usable."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_id": task_id,
                        "latest_user_message": observation.latest_user_message,
                        "known_constraints": observation.known_constraints,
                        "hints": observation.hints,
                        "remaining_steps": observation.remaining_steps,
                        "conversation_history": [turn.model_dump() for turn in observation.conversation_history],
                    },
                    ensure_ascii=True,
                ),
            },
        ]

        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=0,
            top_p=1,
            messages=messages,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        return AgentAction(
            action_type=payload["action_type"],
            content=payload["content"],
        )

    def _heuristic_action(self, task_id: str, observation: EnvObservation) -> AgentAction:
        history_text = " ".join(turn.text.lower() for turn in observation.conversation_history)
        known = " ".join(observation.known_constraints).lower()
        latest = observation.latest_user_message.lower()
        step_index = len([turn for turn in observation.conversation_history if turn.speaker == "assistant"]) + 1

        if task_id == "easy_meeting_note":
            if "4 pm" not in history_text and "4 pm" not in known and step_index <= 2:
                return AgentAction.ask_clarification("What time should the sync move to?")
            if "sunflower" not in history_text and observation.remaining_steps >= 4 and ("location" in latest or step_index >= 3):
                return AgentAction.ask_clarification("Which room should I mention?")
            if "team" not in history_text and "team" not in known:
                return AgentAction.ask_clarification("Who is this note going to?")
            solution = (
                "Short note to the product team: Tomorrow's sync is moving to 4 PM in Sunflower room. Thanks for adjusting."
                if "sunflower" in history_text
                else "Short note to the product team: Tomorrow's sync is moving to 4 PM. Thanks for adjusting."
            )
            return AgentAction.propose_solution(solution)

        if task_id == "medium_client_lunch":
            if "vegetarian" not in history_text and "vegetarian" not in known and step_index <= 2:
                return AgentAction.ask_clarification("Do I need to account for any dietary preferences?")
            if "10-minute walk" not in history_text and "walk" not in history_text and step_index <= 3:
                return AgentAction.ask_clarification("How close should the lunch spot be?")
            if "$35" not in history_text and "$25" not in history_text and step_index <= 3:
                return AgentAction.ask_clarification("What budget should I optimize for per person?")
            if "investor" in history_text and "polished" not in history_text:
                return AgentAction.ask_clarification("Do you want the final plan to feel more polished than casual?")
            solution = (
                "Plan a polished-but-not-formal vegetarian-friendly lunch for two tomorrow around 12:15, within a 10-minute walk, with a budget of about $35 per person."
                if "$35" in history_text
                else "Plan a casual vegetarian-friendly lunch for two tomorrow around 12:15, within a 10-minute walk, around $25 per person."
            )
            return AgentAction.propose_solution(solution)

        if "next wednesday" not in history_text and step_index <= 2:
            return AgentAction.ask_clarification("When should the update say access is opening?")
        if "validation" not in history_text and step_index <= 3:
            return AgentAction.ask_clarification("How should I explain the delay without sharing internal details?")
        if "outside the company" not in history_text and "external" not in history_text and step_index <= 3:
            return AgentAction.ask_clarification("Should I treat this as something that might be forwarded externally?")
        solution = (
            "Ready to send now: Customer-safe update: We are taking a few extra validation steps before opening access next Wednesday. We appreciate your patience while we finish those final checks."
        )
        return AgentAction.propose_solution(solution)


def run_all_tasks() -> None:
    policy = LLMBaselinePolicy()

    for task_index, task_id in enumerate(TASK_ORDER, start=1):
        env = HumanConfusionGoalDriftEnv(task_id=task_id, default_seed=GLOBAL_SEED + task_index)
        observation = env.reset(seed=GLOBAL_SEED + task_index, task_id=task_id)
        rewards: list[float] = []
        success = False

        print(f"[START] task={task_id} env={ENV_NAME} model={policy.label}")

        for step_number in range(1, env.state().max_steps + 1):
            result = policy.next_action(task_id, observation)
            error = result.error or "none"
            try:
                observation, reward, done, info = env.step(result.action)
            except Exception as exc:  # pragma: no cover - unexpected runtime errors
                print(
                    f"[STEP] step={step_number} action={compact(result.action.model_dump())} "
                    f"reward=0.0 done=false error={type(exc).__name__}"
                )
                break

            rewards.append(reward)
            success = bool(info.get("reward_model", {}).get("success", False))
            print(
                f"[STEP] step={step_number} action={compact(result.action.model_dump())} "
                f"reward={reward:.3f} done={str(done).lower()} error={error}"
            )
            if done:
                break

        print(
    f"[END] success={str(success).lower()} steps={len(rewards)} "
    f"rewards={','.join([f'{item:.2f}' for item in rewards])}"
    )
      

if __name__ == "__main__":
    random.seed(GLOBAL_SEED)
    run_all_tasks()
