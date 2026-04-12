---
title: Data Cleaning Env
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_file: app.py
short_description: RL env for cleaning messy datasets
tags:
  - openenv
  - reinforcement-learning
  - data-cleaning
  - real-world
  - tabular
  - rl
  - adversarial
suggested_hardware: cpu-basic
pinned: false
---

# 🧹 Data Cleaning OpenEnv

An OpenEnv-compatible reinforcement learning environment where AI agents learn to
clean messy, real-world datasets through structured actions and dense reward signals.

**What makes this unique:**
- 4 tasks from Easy to Expert — including `deceptive_clean` with intentional traps
- Strategic adversarial mode — the adversary targets your best column
- Catastrophic penalties for destroying critical data (-1.0, episode ends)
- Per-step action cost rewards efficient agents over brute-force ones
- Hints system reveals dataset structure without giving away the answer
- Per-session isolation — safe for concurrent RL training agents

---

## Motivation

Data cleaning consumes up to 80% of a data scientist's time. This environment trains
RL agents to learn systematic, context-aware cleaning strategies across progressively
harder datasets. Unlike a simple cleaning pipeline, an intelligent agent must read
hints, identify protected columns, and decide which nulls are intentional — making
this a genuine reasoning challenge for language models.

No other OpenEnv environment tackles tabular data cleaning.

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `fill_nulls` | `column`, `strategy` (mean/median/mode/drop/ffill) | Fill missing values |
| `drop_duplicates` | — | Remove duplicate rows |
| `cast_column` | `column`, `dtype` (int/float/str/datetime/bool) | Fix data types |
| `remove_outliers` | `column`, `method` (iqr/zscore) | Remove statistical outliers |
| `normalize_text` | `column` | Strip whitespace and lowercase |
| `finish` | — | Signal episode completion |

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Active task name |
| `step` | int | Current step number |
| `session_id` | string | Unique session ID for this episode |
| `columns` | list | Per-column stats: dtype, null_count, null_pct, unique_count, sample_values |
| `total_rows` | int | Current row count |
| `duplicate_rows` | int | Number of duplicate rows |
| `outlier_counts` | dict | Outlier count per numeric column |
| `score_so_far` | float | Cleaning quality score in [0, 1] |
| `done` | bool | Whether episode has ended |
| `message` | string | Human-readable feedback from last action |
| `metadata` | dict | hints, protected_nulls, valid_outlier_cols, critical_cols, action_cost, adversary_active, catastrophic_termination |

---

## Tasks

### Task 1: `fix_nulls` — Easy
- **Dataset**: 100 rows, 4 columns with 10–20% missing values
- **Goal**: Fill all null values using correct strategies
- **Max steps**: 10
- **Scoring**: Fraction of originally-null cells that are filled
- **Optimal strategy**: fill_nulls with mean/median for numeric, mode for categorical

### Task 2: `fix_types` — Medium
- **Dataset**: 135 rows (incl. 15 duplicates), 6 columns with wrong dtypes and nulls
- **Goal**: Fix data types, remove duplicates, fill nulls
- **Max steps**: 15
- **Scoring**: `0.5 * null_score + 0.5 * duplicate_score`
- **Trap**: `join_date` nulls are intentional — filling is penalised (-0.35)

### Task 3: `full_clean` — Hard
- **Dataset**: 170 rows (incl. 20 duplicates), 7 columns with nulls + outliers + messy text
- **Goal**: Full pipeline — nulls, duplicates, outliers, text normalisation
- **Max steps**: 20
- **Scoring**: `0.30*null + 0.25*duplicate + 0.25*outlier + 0.20*text`
- **Traps**: `email` nulls are protected. `rating=5.0` values are valid. Salary is right-skewed (use median).

### Task 4: `deceptive_clean` — Expert
- **Dataset**: 140 rows, 6 columns with conflicting signals everywhere
- **Goal**: Clean without destroying real data or triggering catastrophic penalties
- **Max steps**: 15
- **Scoring**: `0.30*null + 0.25*outlier + 0.25*text + 0.20*duplicate`
- **Traps**:
  - High revenue rows look like outliers — they are real high earners (removing = catastrophic)
  - `notes` nulls are intentional (no note = no issue) — filling is penalised (-0.35)
  - ~10 near-duplicate rows have different `customer_id` (repeat purchases, not true duplicates)
  - Dropping `customer_id` = instant catastrophic termination (-1.0)

---

## Reward Function

```
reward = score_after_action - score_before_action - action_cost + penalty
```

- `action_cost = 0.04` per step — rewards efficient agents over brute-force ones
- Dense incremental reward at every step
- Negative if adversary re-corrupts faster than the agent cleans
- Range: approximately [-1.0, 1.0]

---

## Penalty System

| Mistake | Penalty | Notes |
|---------|---------|-------|
| Any action | -0.04 | Action cost — applies always |
| Wrong fill strategy | -0.20 | e.g. mean on a categorical column |
| Filling protected null | -0.35 | e.g. `email`, `notes`, `join_date` |
| Mean on skewed column | -0.20 | Use median for salary/revenue |
| Removing valid outliers | **-1.00** | CATASTROPHIC — episode ends |
| Dropping critical column | **-1.00** | CATASTROPHIC — episode ends |

---

## Adversarial Mode

Set `adversarial: true` in the reset request to activate. After every agent step,
a strategic adversary finds the column where the agent made the most progress
and corrupts it — injecting nulls, adding duplicates, or inserting extreme outliers.

```json
POST /reset
{
  "task_id": "full_clean",
  "adversarial": true,
  "adversary_difficulty": 0.4,
  "seed": 42
}
```

Difficulty controls attack frequency (0.0 = inactive, 1.0 = maximum, 3 attacks/step).

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check, returns active session count |
| `/reset` | POST | Start new episode, returns `session_id` |
| `/step` | POST | Execute action (requires `session_id`) |
| `/state` | GET | Get episode metadata (requires `session_id`) |
| `/tasks` | GET | List all 4 tasks with descriptions |
| `/docs` | GET | Interactive Swagger UI |
| `/web` | GET | Interactive browser demo |

---

## Setup and Usage

### Local (Uvicorn)
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Docker
```bash
docker build -t data-cleaning-env .
docker run -p 7860:7860 data-cleaning-env
```

### Python Client
```python
from client import DataCleaningEnv
from models import DataCleaningAction

with DataCleaningEnv(base_url="http://localhost:7860", task_id="fix_nulls") as env:
    obs = env.reset()
    print(f"Score: {obs.score_so_far}")

    result = env.step(DataCleaningAction(
        action_type="fill_nulls",
        column="age",
        strategy="mean"
    ))
    print(f"Reward: {result.reward:.4f}, Score: {result.observation.score_so_far:.4f}")
```

### Raw HTTP
```bash
# Start episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "fix_nulls"}'

# Take a step (use session_id from reset response)
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id": "<id>", "action_type": "fill_nulls", "column": "age", "strategy": "mean"}'
```

---

## Running Inference

```bash
export HF_TOKEN=your_hf_token
export ENV_BASE_URL=http://localhost:7860
# Optional:
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

---

## Baseline Scores

Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router  
Safety filter enabled (blocks catastrophic/penalised actions at inference time)

| Task | Difficulty | Score | Steps | Success |
|------|------------|-------|-------|---------|
| fix_nulls | Easy | 0.990 | 4 | ✓ |
| fix_types | Medium | 0.990 | 4 | ✓ |
| full_clean | Hard | 0.990 | 8 | ✓ |
| deceptive_clean | Expert | 0.750 | 6 | ✓ |
| **Average** | | **0.938** | **5.5** | **4/4** |

> **Note on `deceptive_clean`:** Score of 0.750 reflects intentional design — the
> near-duplicate rows are repeat purchases (different `customer_id`) and must not
> be removed. An agent that blindly calls `drop_duplicates` would destroy real data.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API key for LLM calls |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `7860` | Server port |
| `WORKERS` | `2` | Uvicorn worker count |

---


## Decision Pressure Mechanics

This is what separates this environment from a simple cleaning pipeline:

### 1. Action Cost
Every action costs -0.04 reward. The agent must prioritize high-impact actions. Brute-force cleaning is penalized heavily.

### 2. Catastrophic Penalties

| Mistake | Penalty | Consequence |
|---------|---------|-------------|
| Drop critical column (user_id, emp_id) | -1.0 | Episode ends immediately |
| Remove valid outliers (rating, revenue) | -1.0 | Episode ends immediately |
| Fill protected null column | -0.35 | Severe penalty |
| Wrong fill strategy | -0.20 | Moderate penalty |
| Mean on skewed column | -0.20 | Moderate penalty |

### 3. Hidden Traps

| Trap | Task | Correct behavior |
|------|------|-----------------|
| join_date nulls | fix_types | Do NOT fill — null means new user |
| email nulls | full_clean | Do NOT fill — null means opted out |
| rating=5.0 | full_clean | Do NOT remove — valid perfect score |
| revenue high values | deceptive_clean | Do NOT remove — real high earners |

### 4. The Core Decision
A rule-based pipeline fills every null and removes every outlier. An intelligent agent reads the hints, checks context, and decides some nulls are better left alone — and some outliers are real data worth keeping.

## Project Structure

```
data-cleaning-env/
├── models.py              # Pydantic typed models (Action, Observation, State)
├── client.py              # HTTP client with session_id support
├── inference.py           # Baseline inference script
├── app.py                 # Root entrypoint for HF Spaces
├── openenv.yaml           # OpenEnv manifest (v4.0)
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Package config
├── Dockerfile             # Container definition
├── README.md              # This file
└── server/
    ├── app.py             # FastAPI server — all 4 tasks, session isolation
    └── environment.py     # Environment logic, graders, adversary, penalty engine
```
