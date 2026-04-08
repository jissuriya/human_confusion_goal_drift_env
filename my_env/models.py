"""Typed models used by the environment."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server import Action as OpenEnvActionBase
    from openenv.core.env_server import Observation as OpenEnvObservationBase
    from openenv.core.env_server import State as OpenEnvStateBase
except ImportError:  # pragma: no cover - fallback for local editing without OpenEnv
    class OpenEnvActionBase(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        metadata: dict[str, Any] = Field(default_factory=dict)

    class OpenEnvObservationBase(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)
        done: bool = False
        reward: float | int | bool | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class OpenEnvStateBase(BaseModel):
        model_config = ConfigDict(extra="allow", validate_assignment=True)
        episode_id: Optional[str] = None
        step_count: int = 0


ActionType = Literal["respond", "ask_clarification", "propose_solution"]
SpeakerType = Literal["user", "assistant"]


class ConversationTurn(BaseModel):
    """A single turn in the human-agent conversation."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    speaker: SpeakerType
    text: str = Field(min_length=1)
    action_type: Optional[ActionType] = None


class AgentAction(OpenEnvActionBase):
    """Single typed action model with helper constructors for the action space."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    action_type: ActionType
    content: str = Field(min_length=1, max_length=4000)

    @classmethod
    def respond(cls, text: str) -> "AgentAction":
        return cls(action_type="respond", content=text)

    @classmethod
    def ask_clarification(cls, question: str) -> "AgentAction":
        return cls(action_type="ask_clarification", content=question)

    @classmethod
    def propose_solution(cls, solution: str) -> "AgentAction":
        return cls(action_type="propose_solution", content=solution)


class EnvReward(BaseModel):
    """Dense reward breakdown returned in the info dictionary and observation metadata."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    value: float = Field(ge=-1.0, le=1.0)
    task_score: float = Field(ge=0.0, le=1.0)
    matched_slots: list[str] = Field(default_factory=list)
    missing_slots: list[str] = Field(default_factory=list)
    useful_clarifications: list[str] = Field(default_factory=list)
    penalties: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    success: bool = False


class EnvObservation(OpenEnvObservationBase):
    """Observation exposed to the agent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    latest_user_message: str = Field(default="", description="Most recent user utterance.")
    conversation_history: list[ConversationTurn] = Field(default_factory=list)
    hints: Optional[list[str]] = Field(default=None)
    task_id: str
    task_level: Literal["easy", "medium", "hard"]
    remaining_steps: int = Field(ge=0)
    known_constraints: list[str] = Field(default_factory=list)


class EnvState(OpenEnvStateBase):
    """Public environment state without revealing hidden intent."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    episode_id: Optional[str] = None
    step_count: int = Field(default=0, ge=0)
    max_steps: int = Field(default=8, ge=1)
    task_id: str = Field(default="")
    task_level: Literal["easy", "medium", "hard"] = "easy"
    current_phase_id: str = Field(default="phase_0")
    goal_drift_active: bool = False
    resolved_slots: list[str] = Field(default_factory=list)
    asked_clarifications: list[str] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    success: bool = False
