# Ops Workbench OpenEnv

Ops Workbench is a single-task OpenEnv benchmark focused on realistic data cleaning work. The environment simulates a structured cleaning session where an agent must inspect a dirty customer dataset, fix issues with targeted tools, and submit a clean analysis-ready table.

## Environment Overview

The benchmark contains one task:

- `data_cleaning` (`hard`): inspect a messy tabular dataset, resolve nulls, correct data types, remove duplicates, standardize inconsistent values, and rename columns to snake_case.

The environment implements typed Pydantic models for actions, observations, and state, and exposes:

- `reset()` to initialize the dataset
- `step(action)` to apply one cleaning tool call
- `state()` to retrieve the current internal state
- `openenv.yaml` metadata for OpenEnv validation and deployment

## Action Space

`WorkbenchAction`

- `inspect`
- `fill_nulls`
- `rename_column`
- `cast_type`
- `remove_duplicates`
- `fix_value`
- `normalize`
- `drop_column`
- `submit`

Each step contains one structured tool call with typed arguments such as `column`, `strategy`, `dtype`, `old_name`, `new_name`, `find`, and `replace`.

## Observation Space

`WorkbenchObservation`

- task metadata and objective
- available tools
- grading criteria
- dataset columns
- dataset preview
- current inferred dtypes
- null counts
- duplicate row count
- last tool result
- inspected columns
- progress score
- last action error

## Task Details

### Hard: Data Cleaning

The loaded dataset intentionally includes:

- non-snake-case column names
- null values in numeric and text columns
- exact duplicate rows
- inconsistent categorical values like `usa`, `USA`, and `U.S.A`
- string-typed numeric and date columns

The agent must inspect first, then clean using one tool call per step before submitting.

## Reward Function

The reward is dense and deterministic:

- positive reward for improving the weighted grading score
- penalties for invalid actions
- a penalty if the first action is not `inspect(column=None)`
- a penalty for rushing submission before step 3
- a penalty for exceeding 25 steps
- a thrashing penalty after more than 5 consecutive errors

The final score is a deterministic weighted grade in `(0, 1)` over:

- null resolution: 30%
- correct data types: 25%
- no duplicate rows: 15%
- value consistency: 20%
- snake_case column names: 10%

## Setup

```bash
python -m pip install -e .
```

## Run Locally

```bash
python -m server.app
```

## Validate

```bash
openenv validate
```

## Inference

The required root-level `inference.py` uses the OpenAI client and reads:

- `API_BASE_URL` with default `https://api.openai.com/v1`
- `MODEL_NAME` with default `gpt-4.1-mini`
- `HF_TOKEN` as a required secret

Run:

```bash
HF_TOKEN=your_token python inference.py
```

The script emits the required `[START]`, `[STEP]`, and `[END]` lines and writes the final result to `results.json`.

## Docker

```bash
docker build -t ops-workbench .
docker run --rm -p 8000:8000 ops-workbench
```

## Hugging Face Spaces

The root `Dockerfile`, FastAPI app, and `openenv.yaml` are included so the project can be deployed as a containerized Hugging Face Space tagged with `openenv`.
