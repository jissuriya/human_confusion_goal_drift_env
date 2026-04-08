# Human Confusion & Goal Drift Environment

## Environment description

This project implements an OpenEnv-compatible reinforcement learning environment where an agent has to work through messy human instructions. The user can be vague, incomplete, contradictory, and prone to changing goals mid-episode. The environment rewards agents that ask useful clarifying questions, avoid premature assumptions, and adapt when the hidden intent drifts.

The project includes:

- A strict tuple-based core environment: `step(action) -> (observation, reward, done, info)`
- A thin OpenEnv adapter and FastAPI app for `openenv validate` and container serving
- Typed Pydantic models for observations, actions, rewards, and state
- Three graded tasks covering easy, medium, and hard levels
- A baseline inference script that uses an OpenAI-compatible client and falls back to deterministic heuristics

## Real-world motivation

Many real AI failures do not come from raw capability limits. They come from misreading unclear human intent, failing to ask follow-up questions, or sticking to an outdated goal after the user quietly changes direction. This environment targets that failure mode directly.

## Action space

The environment exposes one typed action model, `AgentAction`, with three valid `action_type` values:

- `respond(text)`
- `ask_clarification(question)`
- `propose_solution(solution)`

Helper constructors are available:

```python
from my_env.models import AgentAction

AgentAction.respond("I think you want a short internal update.")
AgentAction.ask_clarification("Should I assume this might be forwarded externally?")
AgentAction.propose_solution("Ready-to-send update: ...")
```

## Observation space

Each observation includes:

- `latest_user_message`
- `conversation_history` for the last N turns
- `hints` for easier phases or tasks
- `known_constraints` derived from previously resolved details
- `remaining_steps`
- `task_id` and `task_level`

## Task descriptions

### Task 1: Easy

- ID: `easy_meeting_note`
- Slight ambiguity, minimal drift
- Goal: draft a short team note about tomorrow's sync
- Drift: after early turns, the user adds the missing room
- Success criteria: the final solution should mention the audience, tomorrow, the 4 PM time, and the Sunflower room while staying brief

### Task 2: Medium

- ID: `medium_client_lunch`
- Multiple ambiguities and contradictions
- Goal: shape a lunch plan for a client
- Drift: the user later reveals the client is really an investor, which changes tone and budget
- Success criteria: the final plan must reflect vegetarian needs, distance, timing, party size, updated budget, and the more polished tone

### Task 3: Hard

- ID: `hard_release_update`
- Strong goal drift with no explicit confirmation
- Goal: produce a launch update that quietly shifts from internal-facing to external-safe
- Drift: the user first frames the update as internal, then reveals it may be forwarded outside the company, then requires a ready-to-send version with no later follow-up
- Success criteria: the final message must avoid internal details, explain the delay as extra validation, include the next-Wednesday timing, and be ready to send now

## Reward design

The reward function is dense and deterministic:

- `+1.0` for a correct final solution that matches the active hidden intent
- `+0.5` for partial understanding in a non-final or incomplete answer
- `+0.3` for a useful clarification
- `-0.5` for unsupported assumptions or wrong solutions
- `-1.0` for harmful or irrelevant behavior

Each step also returns a structured reward breakdown in `info["reward_model"]`.

## Project structure

```text
human_confusion_goal_drift_env/
├── my_env/
│   ├── __init__.py
│   ├── env.py
│   ├── models.py
│   ├── reward.py
│   └── tasks.py
├── server/
│   ├── __init__.py
│   └── app.py
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Setup instructions

### Local Python setup

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

### Validate as an OpenEnv project

```bash
openenv validate
```

### Run the environment server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Example usage

### Core tuple API

```python
from my_env.env import HumanConfusionGoalDriftEnv
from my_env.models import AgentAction

env = HumanConfusionGoalDriftEnv(task_id="easy_meeting_note")
obs = env.reset(seed=13)
obs, reward, done, info = env.step(AgentAction.ask_clarification("What time should I mention?"))
print(obs.latest_user_message)
print(reward, done, info["reward_model"]["task_score"])
```

### Baseline inference

The script reads:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

It also accepts `OPENAI_API_KEY` as an alternative key source. If no LLM endpoint is configured, it falls back to a deterministic heuristic policy so the script remains runnable for smoke tests.

```bash
python inference.py
```

Expected log format:

```text
[START] task=easy_meeting_note env=Human Confusion & Goal Drift Environment model=heuristic-fallback
[STEP] step=1 action={"action_type":"ask_clarification","content":"What time should the sync move to?","metadata":{}} reward=0.300 done=false error=none
[END] success=true steps=3 rewards=[0.3,0.3,1.0]
```

## Docker

Build:

```bash
docker build -t human-confusion-goal-drift-env .
```

Run:

```bash
docker run --rm -p 8000:8000 human-confusion-goal-drift-env
```

## Baseline results

The included heuristic fallback is intended as a deterministic smoke-test baseline. In local verification it solves all three tasks within the eight-step limit, with stronger results possible when connected to an OpenAI-compatible model endpoint.

## Design notes

- Hidden intent is stored internally and never returned directly in observations or state
- Phrasing randomness is deterministic under a fixed seed
- Task grading is fully deterministic and always returns a `0.0` to `1.0` score
- Episodes terminate on success or after `max_steps=8`
