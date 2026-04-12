"""
inference.py — Baseline inference script for the Data Cleaning OpenEnv environment.

Environment variables:
    API_BASE_URL   API endpoint for the LLM          (default: https://router.huggingface.co/v1)
    MODEL_NAME     Model identifier used for inference (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       Hugging Face API token              (required — no default)
    ENV_BASE_URL   Running environment server URL      (default: http://localhost:7860)

Stdout format (strict — hackathon validator):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

IMPORTANT: The hackathon validator requires every score and every value in the
rewards list to be STRICTLY between 0 and 1 (i.e. > 0.0 and < 1.0).
Values of exactly 0.0 or 1.0 are rejected. Negative values are also rejected.
"""

import json
import os
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# API_BASE_URL and MODEL_NAME MUST have defaults (hackathon rule)
# HF_TOKEN is mandatory — raise immediately if missing
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required but not set.")

# Initialize OpenAI client (required by hackathon rules — must use OpenAI SDK)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SUCCESS_THRESHOLD  = 0.5
MAX_STEPS_PER_TASK = 20
TEMPERATURE        = 0.2
MAX_TOKENS         = 512
TASKS              = ["fix_nulls", "fix_types", "full_clean", "deceptive_clean"]

# Safe bounds: strictly (0, 1) as required by hackathon validator.
# Using 0.05/0.95 keeps well away from both boundaries even after rounding.
_SCORE_MIN = 0.01
_SCORE_MAX = 0.99


def _clamp(value: float) -> float:
    """Clamp any float to be strictly between 0 and 1 (exclusive).
    The hackathon validator requires every value in results.json (scores AND rewards)
    to be strictly > 0 and < 1. Negative rewards from the environment are converted
    to the minimum positive value (0.01) before logging.
    Bounds: (0.01, 0.99)."""
    v = float(value)
    if v >= _SCORE_MAX:
        return _SCORE_MAX
    if v <= _SCORE_MIN:
        return _SCORE_MIN
    return max(_SCORE_MIN, min(_SCORE_MAX, v))


# ---------------------------------------------------------------------------
# Strict stdout log format — field names and order must not change
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val  = error if error else "null"
    action_str = str(action).replace("\n", " ").replace("\r", "")
    r = _clamp(reward)
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={r:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """
    [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
    Rewards must be strictly between -1 and 1 (exclusive) and never exactly 0.
    """
    safe_rewards = [_clamp(r) for r in rewards] if rewards else [_SCORE_MIN]
    rewards_str  = ",".join(f"{r:.2f}" for r in safe_rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert data scientist. You receive a description of a messy dataset
    and must clean it by choosing one action per step.

    Available actions — respond with ONLY valid JSON, no markdown, no explanation:

    {"action_type": "fill_nulls",      "column": "<col>", "strategy": "<mean|median|mode|drop|ffill>"}
    {"action_type": "drop_duplicates"}
    {"action_type": "cast_column",     "column": "<col>", "dtype": "<int|float|str|datetime|bool>"}
    {"action_type": "remove_outliers", "column": "<col>", "method": "<iqr|zscore>"}
    {"action_type": "normalize_text",  "column": "<col>"}
    {"action_type": "finish"}

    Rules:
    - Output ONLY a single JSON object, nothing else.
    - Use mean/median for numeric nulls; mode for categorical.
    - Remove outliers only from numeric columns.
    - ALWAYS read the hints — some nulls are protected, some outliers are real data.
    - Every action costs reward. Be efficient — use as few steps as possible.
    - Call finish when the dataset is clean or no more useful actions remain.
""").strip()


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(task_id: str, obs: dict, step: int) -> str:
    cols = obs.get("columns", [])
    col_lines = [
        f"  - {c['name']} (dtype={c['dtype']}, nulls={c['null_count']} "
        f"({c['null_pct']*100:.1f}%), unique={c['unique_count']}, "
        f"sample={c['sample_values'][:3]})"
        for c in cols
    ]
    outliers    = obs.get("outlier_counts", {})
    outlier_str = ", ".join(f"{k}:{v}" for k, v in outliers.items() if v > 0) or "none"
    meta        = obs.get("metadata", {})
    hints       = meta.get("hints", [])
    protected   = meta.get("protected_nulls", [])
    valid_out   = meta.get("valid_outlier_cols", [])
    critical    = meta.get("critical_cols", [])
    hints_str   = "\n".join(f"  ! {h}" for h in hints) if hints else "  (none)"

    return textwrap.dedent(f"""
        Task: {task_id} | Step: {step}
        Score so far: {obs.get('score_so_far', 0):.4f}
        Total rows: {obs.get('total_rows')} | Duplicate rows: {obs.get('duplicate_rows')}
        Outliers by column: {outlier_str}
        Protected null columns (DO NOT fill): {protected or 'none'}
        Valid outlier columns (DO NOT remove): {valid_out or 'none'}
        Critical columns (DO NOT drop): {critical or 'none'}
        Last message: {obs.get('message', '')}

        Hints:
        {hints_str}

        Columns:
        {chr(10).join(col_lines)}

        Choose your next action (JSON only):
    """).strip()


# ---------------------------------------------------------------------------
# LLM call — uses OpenAI client (required by hackathon rules)
# ---------------------------------------------------------------------------

def get_action(task_id: str, obs: dict, step: int) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_prompt(task_id, obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {"action_type": "finish"}


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> dict:
    http        = httpx.Client(timeout=60)
    rewards: List[float] = []
    steps_taken = 0
    score       = _SCORE_MIN   # safe default — never 0.0
    success     = False

    log_start(task=task_id, env="data-cleaning-env", model=MODEL_NAME)

    try:
        resp = http.post(
            f"{ENV_BASE_URL}/reset",
            json={"task_id": task_id, "adversarial": False, "seed": 42},
        )
        resp.raise_for_status()
        obs        = resp.json()
        session_id = obs.get("session_id")

        # Seed score from reset observation so it's never 0.0
        score = _clamp(float(obs.get("score_so_far", _SCORE_MIN)))

        for step in range(1, MAX_STEPS_PER_TASK + 1):
            if obs.get("done", False):
                break

            action_dict = get_action(task_id, obs, step)
            action_str  = json.dumps(action_dict)
            error_msg   = None

            try:
                payload   = {**action_dict, "session_id": session_id, "task_id": task_id}
                step_resp = http.post(f"{ENV_BASE_URL}/step", json=payload)
                step_resp.raise_for_status()
                step_data = step_resp.json()
            except Exception as e:
                error_msg = str(e)
                step_data = {"done": True, "reward": _SCORE_MIN, "score_so_far": score}

            # score_so_far is always a positive score in (0,1)
            # reward can be negative — clamp preserves sign for rewards
            raw_reward = float(step_data.get("reward", _SCORE_MIN))
            reward = _clamp(raw_reward)
            done   = bool(step_data.get("done", False))
            # score_so_far must be a positive value in (0,1)
            raw_score = float(step_data.get("score_so_far", score))
            score  = max(_SCORE_MIN, min(_SCORE_MAX, raw_score)) if raw_score > 0 else _SCORE_MIN

            rewards.append(reward)
            steps_taken = step
            obs         = step_data

            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error_msg)

            if done:
                break

        success = score > SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        http.close()
        if not rewards:
            rewards = [_SCORE_MIN]
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    results = []
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        results.append(run_task(task_id))

    print(f"\n{'='*60}", flush=True)
    print("[SUMMARY]", flush=True)
    for r in results:
        status = "pass" if r["success"] else "fail"
        print(
            f"  [{status}] {r['task_id']:16s}  score={r['score']:.3f}  steps={r['steps']}",
            flush=True,
        )
    avg = _clamp(sum(r["score"] for r in results) / len(results))
    print(f"  Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
