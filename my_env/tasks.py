"""Task definitions for the Human Confusion & Goal Drift environment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotRule:
    """A single intent slot used for grading and clarification detection."""

    name: str
    description: str
    keywords: tuple[str, ...]
    question_triggers: tuple[str, ...]
    clarification_answers: tuple[str, ...]
    nudge_variants: tuple[str, ...]
    weight: float = 1.0
    match_mode: str = "keyword"
    max_words: int | None = None


@dataclass(frozen=True)
class PhaseSpec:
    """A task phase with its current hidden intent and user phrasing."""

    phase_id: str
    intro_variants: tuple[str, ...]
    slots: tuple[SlotRule, ...]
    hints: tuple[str, ...] = ()
    drift_variants: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    generic_redirects: tuple[str, ...] = ()
    ack_variants: tuple[str, ...] = ()
    no_explicit_confirmation: bool = False


@dataclass(frozen=True)
class TaskDefinition:
    """A complete environment task with one or more hidden-intent phases."""

    task_id: str
    title: str
    level: str
    description: str
    max_steps: int
    history_window: int
    success_threshold: float
    partial_threshold: float
    phase_starts: tuple[int, ...]
    phases: tuple[PhaseSpec, ...]
    domain_keywords: tuple[str, ...]

    def phase_index_for_step(self, step_count: int) -> int:
        index = 0
        for candidate, start in enumerate(self.phase_starts):
            if step_count >= start:
                index = candidate
        return min(index, len(self.phases) - 1)

    def phase_for_step(self, step_count: int) -> PhaseSpec:
        return self.phases[self.phase_index_for_step(step_count)]


EASY_TASK = TaskDefinition(
    task_id="easy_meeting_note",
    title="Meeting Note Rewrite",
    level="easy",
    description="Draft a short meeting note from a slightly ambiguous request with only mild drift.",
    max_steps=8,
    history_window=6,
    success_threshold=0.88,
    partial_threshold=0.55,
    phase_starts=(0, 2),
    domain_keywords=("team", "sync", "meeting", "tomorrow", "room", "message"),
    phases=(
        PhaseSpec(
            phase_id="easy_initial",
            intro_variants=(
                "Can you write a quick note to the product team about tomorrow's sync? I need to send it soon.",
                "Help me draft a short message for the team about tomorrow's check-in.",
                "I need a brief update for the team about tomorrow's sync, but I'm blanking on the wording.",
            ),
            slots=(
                SlotRule(
                    name="audience_team",
                    description="address the full product team",
                    keywords=("team", "everyone", "folks", "product team"),
                    question_triggers=("team", "audience", "who", "send"),
                    clarification_answers=(
                        "It is for the whole product team.",
                        "Send it to the product team, not just a subgroup.",
                    ),
                    nudge_variants=(
                        "Keep it aimed at the whole team.",
                        "The note should clearly be for the product team.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="meeting_sync",
                    description="refer to tomorrow's sync",
                    keywords=("sync", "check-in", "standup", "meeting"),
                    question_triggers=("meeting", "sync", "standup", "check-in", "what"),
                    clarification_answers=(
                        "It is about tomorrow's sync.",
                        "The note is for tomorrow's team sync.",
                    ),
                    nudge_variants=(
                        "Make it obvious the note is about tomorrow's sync.",
                        "The message still needs to mention the sync itself.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="day_tomorrow",
                    description="say it happens tomorrow",
                    keywords=("tomorrow",),
                    question_triggers=("when", "tomorrow", "date", "day"),
                    clarification_answers=(
                        "Yes, it is for tomorrow.",
                        "Please say tomorrow so nobody misses the timing.",
                    ),
                    nudge_variants=(
                        "You still need the day in there.",
                        "Make sure people can tell this is for tomorrow.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="time_4pm",
                    description="mention the new time at 4 PM",
                    keywords=("4 pm", "4pm", "4:00", "16:00"),
                    question_triggers=("time", "when", "schedule", "moved"),
                    clarification_answers=(
                        "The sync is moving to 4 PM.",
                        "Please mention that the new time is 4 PM.",
                    ),
                    nudge_variants=(
                        "The timing still is not specific enough.",
                        "People need the new time, not just a vague note.",
                    ),
                    weight=1.3,
                ),
                SlotRule(
                    name="tone_brief",
                    description="keep the message short",
                    keywords=(),
                    question_triggers=("tone", "short", "brief", "length"),
                    clarification_answers=(
                        "Keep it short and friendly.",
                        "I only need a concise note, not a long explanation.",
                    ),
                    nudge_variants=(
                        "Trim it down a little.",
                        "The note should stay brief.",
                    ),
                    weight=0.7,
                    match_mode="max_words",
                    max_words=55,
                ),
            ),
            hints=(
                "If the schedule is unclear, ask directly about timing.",
                "A short, sendable message is better than a long explanation.",
            ),
            generic_redirects=(
                "You're filling in details I never gave you.",
                "Stay concrete instead of guessing.",
            ),
            ack_variants=(
                "Yes, that covers it.",
                "That works. I can send that.",
            ),
        ),
        PhaseSpec(
            phase_id="easy_location_addition",
            intro_variants=(),
            drift_variants=(
                "One more thing: mention that it is in Sunflower room.",
                "I forgot the location. It is in Sunflower room, so weave that in.",
            ),
            slots=(
                SlotRule(
                    name="audience_team",
                    description="address the full product team",
                    keywords=("team", "everyone", "folks", "product team"),
                    question_triggers=("team", "audience", "who", "send"),
                    clarification_answers=(
                        "It is still for the whole product team.",
                        "Same audience: the product team.",
                    ),
                    nudge_variants=(
                        "Keep it aimed at the whole team.",
                        "The audience should still be obvious.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="meeting_sync",
                    description="refer to tomorrow's sync",
                    keywords=("sync", "check-in", "standup", "meeting"),
                    question_triggers=("meeting", "sync", "standup", "check-in", "what"),
                    clarification_answers=(
                        "Still tomorrow's sync.",
                        "Same event: tomorrow's sync.",
                    ),
                    nudge_variants=(
                        "The message still needs the sync called out.",
                        "Keep the sync explicit.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="day_tomorrow",
                    description="say it happens tomorrow",
                    keywords=("tomorrow",),
                    question_triggers=("when", "tomorrow", "date", "day"),
                    clarification_answers=(
                        "Yes, still tomorrow.",
                        "Tomorrow is still important to include.",
                    ),
                    nudge_variants=(
                        "Do not drop the day.",
                        "It still needs to say tomorrow.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="time_4pm",
                    description="mention the new time at 4 PM",
                    keywords=("4 pm", "4pm", "4:00", "16:00"),
                    question_triggers=("time", "when", "schedule", "moved"),
                    clarification_answers=(
                        "Still 4 PM.",
                        "Same time: 4 PM.",
                    ),
                    nudge_variants=(
                        "Keep the new time in there.",
                        "The note still needs 4 PM.",
                    ),
                    weight=1.3,
                ),
                SlotRule(
                    name="room_sunflower",
                    description="include Sunflower room",
                    keywords=("sunflower room", "sunflower"),
                    question_triggers=("room", "location", "where"),
                    clarification_answers=(
                        "The room is Sunflower room.",
                        "Please say Sunflower room.",
                    ),
                    nudge_variants=(
                        "You have the schedule, but not the room yet.",
                        "Do not forget the Sunflower room mention.",
                    ),
                    weight=1.2,
                ),
                SlotRule(
                    name="tone_brief",
                    description="keep the message short",
                    keywords=(),
                    question_triggers=("tone", "short", "brief", "length"),
                    clarification_answers=(
                        "Still short and friendly.",
                        "Keep it concise.",
                    ),
                    nudge_variants=(
                        "Keep it brief.",
                        "The sendable version should stay short.",
                    ),
                    weight=0.7,
                    match_mode="max_words",
                    max_words=60,
                ),
            ),
            hints=(
                "The location becomes important after the first couple of turns.",
            ),
            generic_redirects=(
                "You are close, but one operational detail is still missing.",
                "This needs one more concrete detail.",
            ),
            ack_variants=(
                "Perfect. I can send that now.",
                "That version is ready to go.",
            ),
        ),
    ),
)

MEDIUM_TASK = TaskDefinition(
    task_id="medium_client_lunch",
    title="Client Lunch Planning",
    level="medium",
    description="Resolve multiple ambiguities and contradictions in a lunch-planning request.",
    max_steps=8,
    history_window=6,
    success_threshold=0.84,
    partial_threshold=0.5,
    phase_starts=(0, 3),
    domain_keywords=("lunch", "client", "vegetarian", "walk", "budget", "reservation"),
    phases=(
        PhaseSpec(
            phase_id="medium_initial",
            intro_variants=(
                "Can you help me shape a lunch plan for tomorrow with a client? Nothing too fancy, but I do not want it to feel cheap either.",
                "I need a lunch plan for a client tomorrow. Keep it casual, but not sloppy.",
                "Help me pin down a client lunch for tomorrow. I want it comfortable, not over the top.",
            ),
            slots=(
                SlotRule(
                    name="day_tomorrow",
                    description="make it for tomorrow",
                    keywords=("tomorrow",),
                    question_triggers=("when", "tomorrow", "date", "day"),
                    clarification_answers=(
                        "Yes, this is for tomorrow.",
                        "Tomorrow is the day.",
                    ),
                    nudge_variants=(
                        "Make sure the timing is pinned to tomorrow.",
                        "The plan still needs the day.",
                    ),
                    weight=0.8,
                ),
                SlotRule(
                    name="time_lunch",
                    description="set the meal around noon",
                    keywords=("lunch", "noon", "12:00", "12 pm", "12:15"),
                    question_triggers=("time", "noon", "lunch", "when"),
                    clarification_answers=(
                        "Aim for around 12:15.",
                        "Lunch around noon is right.",
                    ),
                    nudge_variants=(
                        "The time should feel like lunch, not a vague meetup.",
                        "It still needs a lunch-time target.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="party_two",
                    description="reserve for two people",
                    keywords=("two people", "2 people", "for two", "me and one client"),
                    question_triggers=("how many", "party", "people", "size"),
                    clarification_answers=(
                        "It is just me and one client, so two people.",
                        "Make it a table for two.",
                    ),
                    nudge_variants=(
                        "You have not said how many people this is for.",
                        "The party size still matters here.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="vegetarian",
                    description="include vegetarian-friendly options",
                    keywords=("vegetarian", "veg-friendly", "meat-free"),
                    question_triggers=("diet", "vegetarian", "food", "menu"),
                    clarification_answers=(
                        "It needs to be vegetarian-friendly.",
                        "Please make sure there are solid vegetarian options.",
                    ),
                    nudge_variants=(
                        "The food constraints are still too vague.",
                        "There is a dietary preference you have not covered yet.",
                    ),
                    weight=1.2,
                ),
                SlotRule(
                    name="walking_distance",
                    description="keep it within a 10-minute walk",
                    keywords=("walking distance", "10-minute walk", "nearby", "walkable"),
                    question_triggers=("distance", "walk", "nearby", "location"),
                    clarification_answers=(
                        "It should be within about a 10-minute walk.",
                        "Keep it walkable from the office.",
                    ),
                    nudge_variants=(
                        "The location should be easy to walk to.",
                        "Please keep the plan nearby.",
                    ),
                    weight=1.1,
                ),
                SlotRule(
                    name="budget_25",
                    description="keep it close to $25 per person",
                    keywords=("$25", "25 per person", "under 25", "around 25"),
                    question_triggers=("budget", "price", "cost"),
                    clarification_answers=(
                        "At first I was thinking around $25 per person.",
                        "Around $25 each felt right at first.",
                    ),
                    nudge_variants=(
                        "You still need a budget read.",
                        "Cost is part of the decision here.",
                    ),
                    weight=0.9,
                ),
            ),
            generic_redirects=(
                "You are still making too many assumptions.",
                "Pause on venue ideas until the constraints are cleaner.",
            ),
            ack_variants=(
                "That plan feels usable.",
                "That is coherent enough to act on.",
            ),
        ),
        PhaseSpec(
            phase_id="medium_polished_shift",
            intro_variants=(),
            drift_variants=(
                "Actually, it is an investor I have not met in person, so make it a little more polished. Budget can stretch to about $35 each.",
                "Small correction: this client is really an investor, so go a touch nicer. We can do around $35 a person.",
            ),
            slots=(
                SlotRule(
                    name="day_tomorrow",
                    description="make it for tomorrow",
                    keywords=("tomorrow",),
                    question_triggers=("when", "tomorrow", "date", "day"),
                    clarification_answers=(
                        "Still tomorrow.",
                        "The day stays tomorrow.",
                    ),
                    nudge_variants=(
                        "Keep tomorrow explicit.",
                        "Do not drop the day.",
                    ),
                    weight=0.8,
                ),
                SlotRule(
                    name="time_lunch",
                    description="set the meal around noon",
                    keywords=("lunch", "noon", "12:00", "12 pm", "12:15"),
                    question_triggers=("time", "noon", "lunch", "when"),
                    clarification_answers=(
                        "Keep it around 12:15.",
                        "Still a noon lunch.",
                    ),
                    nudge_variants=(
                        "The timing should still feel like lunch.",
                        "Keep a specific lunch-time target.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="party_two",
                    description="reserve for two people",
                    keywords=("two people", "2 people", "for two", "me and one investor"),
                    question_triggers=("how many", "party", "people", "size"),
                    clarification_answers=(
                        "Still just two people.",
                        "It remains a table for two.",
                    ),
                    nudge_variants=(
                        "The party size still matters.",
                        "Keep it scoped to two people.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="vegetarian",
                    description="include vegetarian-friendly options",
                    keywords=("vegetarian", "veg-friendly", "meat-free"),
                    question_triggers=("diet", "vegetarian", "food", "menu"),
                    clarification_answers=(
                        "Vegetarian-friendly still matters.",
                        "Please keep the vegetarian options requirement.",
                    ),
                    nudge_variants=(
                        "The dietary preference still applies.",
                        "The meal still needs good vegetarian options.",
                    ),
                    weight=1.2,
                ),
                SlotRule(
                    name="walking_distance",
                    description="keep it within a 10-minute walk",
                    keywords=("walking distance", "10-minute walk", "nearby", "walkable"),
                    question_triggers=("distance", "walk", "nearby", "location"),
                    clarification_answers=(
                        "Still within a 10-minute walk.",
                        "Nearby is still important.",
                    ),
                    nudge_variants=(
                        "Please keep it nearby.",
                        "Location convenience still matters.",
                    ),
                    weight=1.1,
                ),
                SlotRule(
                    name="budget_35",
                    description="budget around $35 per person",
                    keywords=("$35", "35 each", "35 per person", "around 35"),
                    question_triggers=("budget", "price", "cost"),
                    clarification_answers=(
                        "Budget can stretch to about $35 each now.",
                        "Use roughly $35 per person as the cap.",
                    ),
                    nudge_variants=(
                        "The updated budget is part of the change.",
                        "Do not leave out the newer budget range.",
                    ),
                    weight=1.0,
                ),
                SlotRule(
                    name="tone_polished",
                    description="make it polished but not formal",
                    keywords=("polished", "professional", "a little nicer", "not formal"),
                    question_triggers=("tone", "style", "formal", "polished"),
                    clarification_answers=(
                        "Think polished, but not white-tablecloth formal.",
                        "It should feel a bit nicer without being stiff.",
                    ),
                    nudge_variants=(
                        "The tone shifted a bit after I clarified the audience.",
                        "Make the plan feel polished, not just casual.",
                    ),
                    weight=1.0,
                ),
            ),
            generic_redirects=(
                "This still sounds like you locked onto the old version of the brief.",
                "Update the plan to reflect the newer constraints.",
            ),
            ack_variants=(
                "That is the version I needed.",
                "That captures the updated brief.",
            ),
        ),
    ),
)

HARD_TASK = TaskDefinition(
    task_id="hard_release_update",
    title="Launch Update Drift",
    level="hard",
    description="Handle strong goal drift in a status update that quietly shifts from internal to external-safe messaging.",
    max_steps=8,
    history_window=8,
    success_threshold=0.83,
    partial_threshold=0.48,
    phase_starts=(0, 2, 4),
    domain_keywords=("launch", "beta", "update", "validation", "access", "customers"),
    phases=(
        PhaseSpec(
            phase_id="hard_internal_brief",
            intro_variants=(
                "I need a quick update for the launch thread. Keep it calm.",
                "Can you help me word a short launch update? I do not want it to sound panicked.",
                "I need a calm status note for the launch thread, fast.",
            ),
            slots=(
                SlotRule(
                    name="status_update",
                    description="make it a status update",
                    keywords=("update", "status", "quick note"),
                    question_triggers=("what", "update", "status"),
                    clarification_answers=(
                        "Yes, it needs to read like a status update.",
                        "Keep it in update form, not a long memo.",
                    ),
                    nudge_variants=(
                        "Shape it like an update, not a brainstorm.",
                        "This should read like a status note.",
                    ),
                    weight=0.8,
                ),
                SlotRule(
                    name="launch_thread",
                    description="assume it is for the launch thread",
                    keywords=("launch thread", "launch team", "team thread"),
                    question_triggers=("audience", "thread", "who"),
                    clarification_answers=(
                        "At this point it is for the launch thread.",
                        "Right now assume the launch thread is the audience.",
                    ),
                    nudge_variants=(
                        "The audience matters more than you are showing.",
                        "Who sees this is part of the brief.",
                    ),
                    weight=0.8,
                ),
                SlotRule(
                    name="calm_tone",
                    description="keep it calm and measured",
                    keywords=("calm", "measured", "steady"),
                    question_triggers=("tone", "calm", "style"),
                    clarification_answers=(
                        "Yes, calm and measured.",
                        "Please keep the tone steady.",
                    ),
                    nudge_variants=(
                        "The tone should stay calm.",
                        "Do not make it sound alarmist.",
                    ),
                    weight=0.6,
                ),
            ),
            generic_redirects=(
                "That jumps too quickly into specifics I have not aligned yet.",
                "Stay calm, but do not invent a full narrative.",
            ),
            ack_variants=(),
            no_explicit_confirmation=True,
        ),
        PhaseSpec(
            phase_id="hard_external_shift",
            intro_variants=(),
            drift_variants=(
                "This might get forwarded outside the company, so do not mention the rollback or database issue. Frame it as extra validation before access opens.",
                "Actually, assume it could leave the team. Keep the internal mechanics out and say we are doing extra validation before opening access.",
            ),
            slots=(
                SlotRule(
                    name="external_safe",
                    description="make it safe for an outside audience",
                    keywords=("outside the company", "customer-safe", "external", "broader audience"),
                    question_triggers=("audience", "external", "outside", "who"),
                    clarification_answers=(
                        "Yes, write it so an external audience could see it.",
                        "Assume it needs to be safe if it gets forwarded.",
                    ),
                    nudge_variants=(
                        "The audience changed more than the wording suggests.",
                        "This needs to sound safe beyond the internal team.",
                    ),
                    weight=1.1,
                ),
                SlotRule(
                    name="reason_validation",
                    description="say the delay is due to extra validation",
                    keywords=("extra validation", "additional validation", "quality checks", "final checks"),
                    question_triggers=("why", "reason", "validation", "checks"),
                    clarification_answers=(
                        "Use extra validation as the reason.",
                        "Frame it around additional quality checks.",
                    ),
                    nudge_variants=(
                        "The rationale should be framed as validation work.",
                        "Focus on extra validation, not the internal cause.",
                    ),
                    weight=1.3,
                ),
                SlotRule(
                    name="launch_date_next_wednesday",
                    description="say access opens next Wednesday",
                    keywords=("next wednesday", "one week", "next week"),
                    question_triggers=("when", "date", "open", "launch"),
                    clarification_answers=(
                        "Access is opening next Wednesday instead.",
                        "Please say next Wednesday.",
                    ),
                    nudge_variants=(
                        "The revised access timing still is not clear.",
                        "You need the updated timing in the message.",
                    ),
                    weight=1.3,
                ),
                SlotRule(
                    name="calm_confident",
                    description="keep the tone calm and confident",
                    keywords=("thanks for your patience", "appreciate your patience", "we are taking"),
                    question_triggers=("tone", "style", "sound"),
                    clarification_answers=(
                        "Calm and confident is right.",
                        "It should sound steady, not apologetic or evasive.",
                    ),
                    nudge_variants=(
                        "The tone should feel composed.",
                        "Do not make it sound defensive.",
                    ),
                    weight=0.8,
                ),
            ),
            forbidden_phrases=("rollback", "database", "outage", "internal issue"),
            generic_redirects=(
                "You are still leaning on internal details that should stay hidden.",
                "That version does not fully reflect the audience shift.",
            ),
            ack_variants=(),
            no_explicit_confirmation=True,
        ),
        PhaseSpec(
            phase_id="hard_ready_to_send",
            intro_variants=(),
            drift_variants=(
                "Also, do not ask people to wait for another confirmation. I need something they can use as-is.",
                "One last change: make it ready to send now, without promising a later follow-up note.",
            ),
            slots=(
                SlotRule(
                    name="reason_validation",
                    description="say the delay is due to extra validation",
                    keywords=("extra validation", "additional validation", "quality checks", "final checks"),
                    question_triggers=("why", "reason", "validation", "checks"),
                    clarification_answers=(
                        "Keep the explanation focused on extra validation.",
                        "Additional quality checks is still the right framing.",
                    ),
                    nudge_variants=(
                        "The message should still frame this as validation work.",
                        "Keep the explanation at the validation level.",
                    ),
                    weight=1.3,
                ),
                SlotRule(
                    name="launch_date_next_wednesday",
                    description="say access opens next Wednesday",
                    keywords=("next wednesday", "one week", "next week"),
                    question_triggers=("when", "date", "open", "launch"),
                    clarification_answers=(
                        "Still next Wednesday.",
                        "The target remains next Wednesday.",
                    ),
                    nudge_variants=(
                        "Do not lose the revised date.",
                        "The next-Wednesday timing still needs to be explicit.",
                    ),
                    weight=1.3,
                ),
                SlotRule(
                    name="external_safe",
                    description="make it safe for an outside audience",
                    keywords=("forwarded", "customer-safe", "external", "broader audience"),
                    question_triggers=("audience", "external", "outside", "who"),
                    clarification_answers=(
                        "Yes, still external-safe.",
                        "Keep it safe if forwarded.",
                    ),
                    nudge_variants=(
                        "The audience constraint still applies.",
                        "This still needs to be safe for a forwarded message.",
                    ),
                    weight=1.1,
                ),
                SlotRule(
                    name="ready_to_send",
                    description="make it ready to send now",
                    keywords=("ready to send", "use as-is", "sharing now", "sending now"),
                    question_triggers=("follow-up", "send", "ready", "confirmation"),
                    clarification_answers=(
                        "Yes, it should be ready to send now.",
                        "Do not promise another update before using it.",
                    ),
                    nudge_variants=(
                        "I need a version people can use now.",
                        "Do not leave this sounding unfinished.",
                    ),
                    weight=1.1,
                ),
                SlotRule(
                    name="calm_confident",
                    description="keep the tone calm and confident",
                    keywords=("thanks for your patience", "appreciate your patience", "we are taking"),
                    question_triggers=("tone", "style", "sound"),
                    clarification_answers=(
                        "Still calm and confident.",
                        "Keep it measured and usable.",
                    ),
                    nudge_variants=(
                        "The tone should stay composed.",
                        "Keep it steady and sendable.",
                    ),
                    weight=0.8,
                ),
            ),
            forbidden_phrases=("rollback", "database", "outage", "internal issue", "more details soon", "follow up later"),
            generic_redirects=(
                "That still sounds like a draft waiting for a later note.",
                "This should stand on its own right now.",
            ),
            ack_variants=(),
            no_explicit_confirmation=True,
        ),
    ),
)


TASKS: dict[str, TaskDefinition] = {
    EASY_TASK.task_id: EASY_TASK,
    MEDIUM_TASK.task_id: MEDIUM_TASK,
    HARD_TASK.task_id: HARD_TASK,
}

TASK_ORDER: tuple[str, ...] = (
    EASY_TASK.task_id,
    MEDIUM_TASK.task_id,
    HARD_TASK.task_id,
)


def get_task(task_id: str) -> TaskDefinition:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc
