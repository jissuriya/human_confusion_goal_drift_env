"""Environment implementation and OpenEnv adapter."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from .models import AgentAction, ConversationTurn, EnvObservation, EnvReward, EnvState
from .reward import reward_action
from .tasks import PhaseSpec, SlotRule, TASK_ORDER, TaskDefinition, get_task

try:
    from openenv.core.env_server import Environment as OpenEnvEnvironment
    from openenv.core.env_server import create_app
except Exception:  # pragma: no cover - fallback for local editing without OpenEnv
    OpenEnvEnvironment = object  # type: ignore[assignment]
    create_app = None


class ResetRequest(BaseModel):
    seed: int | None = Field(default=None)
    task_id: str | None = Field(default=None)
    episode_id: str | None = Field(default=None)


@dataclass
class EpisodeRuntime:
    task: TaskDefinition
    rng: random.Random
    seed: int
    episode_id: str = field(default_factory=lambda: str(uuid4()))
    step_count: int = 0
    cumulative_reward: float = 0.0
    latest_user_message: str = ""
    history: list[ConversationTurn] = field(default_factory=list)
    resolved_slots: set[str] = field(default_factory=set)
    asked_slots: set[str] = field(default_factory=set)
    success: bool = False


class HumanConfusionGoalDriftEnv:
    """
    Core environment using the strict tuple step API:
    step(action) -> (observation, reward, done, info)
    """

    def __init__(
        self,
        task_id: str | None = None,
        default_seed: int = 13,
    ) -> None:
        self.default_seed = default_seed
        self.task_id = task_id or TASK_ORDER[0]
        self._episode: EpisodeRuntime | None = None

    def available_tasks(self) -> tuple[str, ...]:
        return TASK_ORDER

    def reset(self, seed: int | None = None, task_id: str | None = None) -> EnvObservation:
        chosen_task = get_task(task_id or self.task_id)
        actual_seed = self.default_seed if seed is None else seed
        rng = random.Random(actual_seed)

        opening_message = rng.choice(chosen_task.phases[0].intro_variants)
        episode = EpisodeRuntime(
            task=chosen_task,
            rng=rng,
            seed=actual_seed,
            latest_user_message=opening_message,
        )
        episode.history.append(ConversationTurn(speaker="user", text=opening_message))
        self._episode = episode
        return self._build_observation()

    def state(self) -> EnvState:
        if self._episode is None:
            return EnvState()
        phase = self._current_phase()
        return EnvState(
            episode_id=self._episode.episode_id,
            step_count=self._episode.step_count,
            max_steps=self._episode.task.max_steps,
            task_id=self._episode.task.task_id,
            task_level=self._episode.task.level,  # type: ignore[arg-type]
            current_phase_id=phase.phase_id,
            goal_drift_active=self._episode.step_count >= self._episode.task.phase_starts[1]
            if len(self._episode.task.phase_starts) > 1
            else False,
            resolved_slots=sorted(self._episode.resolved_slots),
            asked_clarifications=sorted(self._episode.asked_slots),
            cumulative_reward=round(self._episode.cumulative_reward, 3),
            success=self._episode.success,
        )

    def step(self, action: AgentAction) -> tuple[EnvObservation, float, bool, dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("Environment must be reset() before step().")

        if self._episode.success or self._episode.step_count >= self._episode.task.max_steps:
            observation = self._build_observation(done=True, reward=0.0)
            return observation, 0.0, True, {"error": "episode_already_finished"}

        previous_phase_index = self._episode.task.phase_index_for_step(self._episode.step_count)
        phase = self._episode.task.phase_for_step(self._episode.step_count)

        self._episode.history.append(
            ConversationTurn(
                speaker="assistant",
                text=action.content,
                action_type=action.action_type,
            )
        )

        reward_model = reward_action(
            action=action,
            task=self._episode.task,
            phase=phase,
            resolved_slots=self._episode.resolved_slots,
            asked_slots=self._episode.asked_slots,
        )

        if action.action_type == "ask_clarification":
            self._episode.asked_slots.update(reward_model.useful_clarifications)
            self._episode.resolved_slots.update(reward_model.useful_clarifications)
        else:
            self._episode.resolved_slots.update(reward_model.matched_slots)

        self._episode.step_count += 1
        self._episode.cumulative_reward += reward_model.value
        self._episode.success = reward_model.success

        current_phase_index = self._episode.task.phase_index_for_step(self._episode.step_count)
        phase_changed = current_phase_index != previous_phase_index
        active_phase = self._episode.task.phase_for_step(self._episode.step_count)
        done = reward_model.success or self._episode.step_count >= self._episode.task.max_steps

        if done and reward_model.success and not active_phase.no_explicit_confirmation:
            acknowledgment = self._choice(active_phase.ack_variants) or "That works."
            self._episode.latest_user_message = acknowledgment
            self._episode.history.append(ConversationTurn(speaker="user", text=acknowledgment))
        elif not done:
            user_reply = self._compose_user_reply(
                action=action,
                reward_model=reward_model,
                phase_before=phase,
                phase_after=active_phase,
                phase_changed=phase_changed,
            )
            self._episode.latest_user_message = user_reply
            self._episode.history.append(ConversationTurn(speaker="user", text=user_reply))

        observation = self._build_observation(done=done, reward=reward_model.value)
        info = {
            "task_id": self._episode.task.task_id,
            "task_level": self._episode.task.level,
            "phase_id": active_phase.phase_id,
            "score": reward_model.task_score,
            "reward_model": reward_model.model_dump(),
            "max_steps_reached": self._episode.step_count >= self._episode.task.max_steps and not reward_model.success,
        }
        return observation, reward_model.value, done, info

    def _choice(self, options: tuple[str, ...]) -> str:
        if not options:
            return ""
        if self._episode is None:
            return options[0]
        return self._episode.rng.choice(list(options))

    def _current_phase(self) -> PhaseSpec:
        if self._episode is None:
            raise RuntimeError("Environment is not active.")
        return self._episode.task.phase_for_step(self._episode.step_count)

    def _slot_by_name(self, phase: PhaseSpec, slot_name: str) -> SlotRule | None:
        for slot in phase.slots:
            if slot.name == slot_name:
                return slot
        return None

    def _compose_user_reply(
        self,
        action: AgentAction,
        reward_model: EnvReward,
        phase_before: PhaseSpec,
        phase_after: PhaseSpec,
        phase_changed: bool,
    ) -> str:
        if self._episode is None:
            raise RuntimeError("Environment is not active.")

        parts: list[str] = []

        if phase_changed and phase_after.drift_variants:
            parts.append(self._choice(phase_after.drift_variants))

        if action.action_type == "ask_clarification" and reward_model.useful_clarifications:
            answer_chunks: list[str] = []
            for slot_name in reward_model.useful_clarifications:
                slot = self._slot_by_name(phase_before, slot_name) or self._slot_by_name(phase_after, slot_name)
                if slot:
                    answer_chunks.append(self._choice(slot.clarification_answers))
            if answer_chunks:
                parts.append(" ".join(answer_chunks))

        if not parts and reward_model.value < 0:
            slot = self._pick_missing_slot(phase_after, reward_model.missing_slots)
            if slot is not None:
                parts.append(self._choice(slot.nudge_variants))
            else:
                parts.append(self._choice(phase_after.generic_redirects) or "That is not quite what I meant.")

        if not parts and reward_model.value >= 0.5 and reward_model.missing_slots:
            slot = self._pick_missing_slot(phase_after, reward_model.missing_slots)
            if slot is not None:
                parts.append(self._choice(slot.nudge_variants))

        if not parts:
            parts.append(self._choice(phase_after.generic_redirects) or "Keep going.")

        return " ".join(part for part in parts if part).strip()

    def _pick_missing_slot(self, phase: PhaseSpec, missing_slots: list[str]) -> SlotRule | None:
        for slot_name in missing_slots:
            slot = self._slot_by_name(phase, slot_name)
            if slot is not None:
                return slot
        return None

    def _build_observation(self, done: bool = False, reward: float | None = None) -> EnvObservation:
        if self._episode is None:
            raise RuntimeError("Environment is not active.")

        phase = self._episode.task.phase_for_step(self._episode.step_count)
        resolved_descriptions = []
        for slot in phase.slots:
            if slot.name in self._episode.resolved_slots:
                resolved_descriptions.append(slot.description)

        hints = list(phase.hints) if phase.hints else None
        history_window = self._episode.task.history_window

        return EnvObservation(
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._episode.episode_id,
                "seed": self._episode.seed,
                "phase_id": phase.phase_id,
            },
            latest_user_message=self._episode.latest_user_message,
            conversation_history=self._episode.history[-history_window:],
            hints=hints,
            task_id=self._episode.task.task_id,
            task_level=self._episode.task.level,  # type: ignore[arg-type]
            remaining_steps=max(self._episode.task.max_steps - self._episode.step_count, 0),
            known_constraints=resolved_descriptions,
        )


class HumanConfusionGoalDriftOpenEnv(OpenEnvEnvironment):  # type: ignore[misc]
    """Thin OpenEnv adapter that wraps the tuple-based core environment."""

    def __init__(self, task_id: str | None = None) -> None:
        super().__init__()  # type: ignore[misc]
        self.core_env = HumanConfusionGoalDriftEnv(task_id=task_id)
        self._state = EnvState()

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> EnvObservation:
        observation = self.core_env.reset(seed=seed, task_id=task_id)
        state = self.core_env.state()
        if episode_id:
            state.episode_id = episode_id
        self._state = state
        return observation

    def step(
        self,
        action: AgentAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> EnvObservation:
        del timeout_s
        observation, _, _, _ = self.core_env.step(action)
        self._state = self.core_env.state()
        return observation

    @property
    def state(self) -> EnvState:
        return self._state


app = (
    create_app(
        HumanConfusionGoalDriftOpenEnv,
        AgentAction,
        EnvObservation,
        env_name="human_confusion_goal_drift_env",
    )
    if create_app is not None
    else None
)


if app is None:  # pragma: no cover - exercised in lightweight local installs
    from fastapi import FastAPI, HTTPException

    fallback_env = HumanConfusionGoalDriftEnv()
    fallback_app = FastAPI(
        title="Human Confusion Goal Drift Environment",
        version="1.0.0",
        description="Fallback FastAPI app for the Human Confusion & Goal Drift environment.",
    )

    @fallback_app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @fallback_app.get("/metadata")
    def metadata() -> dict[str, str]:
        return {
            "name": "human_confusion_goal_drift_env",
            "description": "RL environment for ambiguous human instructions and hidden goal drift.",
        }

    @fallback_app.get("/schema")
    def schema() -> dict[str, Any]:
        return {
            "action": AgentAction.model_json_schema(),
            "observation": EnvObservation.model_json_schema(),
            "state": EnvState.model_json_schema(),
        }

    @fallback_app.post("/reset", response_model=EnvObservation)
    def reset_endpoint(request: ResetRequest | None = None) -> EnvObservation:
        payload = request or ResetRequest()
        return fallback_env.reset(seed=payload.seed, task_id=payload.task_id)

    @fallback_app.post("/step", response_model=EnvObservation)
    def step_endpoint(action: AgentAction) -> EnvObservation:
        try:
            observation, _, _, _ = fallback_env.step(action)
            return observation
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @fallback_app.get("/state", response_model=EnvState)
    def state_endpoint() -> EnvState:
        return fallback_env.state()

    @fallback_app.post("/mcp")
    def mcp_endpoint() -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "error": {"code": -32601, "message": "MCP not implemented in fallback app"},
            "id": None,
        }

    app = fallback_app
