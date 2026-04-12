"""Baseline OpenAI-powered inference runner for the data cleaning environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from openai import OpenAI

from environment import OpsWorkbenchEnv
from models import WorkbenchAction, WorkbenchObservation

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# NOTE: client is initialized lazily inside run_task() so that importing this
# module without HF_TOKEN set (e.g. during validator dry-runs) does not crash.
client: "OpenAI | None" = None


def _get_client() -> "OpenAI":
    global client
    if client is None:
        if HF_TOKEN is None:
            raise ValueError(
                "HF_TOKEN environment variable is required to run inference. "
                "Set it before calling run_task() or main()."
            )
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    return client


def format_action(action: WorkbenchAction) -> str:
    fields = {
        "action_type": action.action_type,
        "column": action.column,
        "strategy": action.strategy,
        "old_name": action.old_name,
        "new_name": action.new_name,
        "dtype": action.dtype,
        "find": action.find,
        "replace": action.replace,
    }
    payload = {key: value for key, value in fields.items() if value is not None}
    return json.dumps(payload, separators=(",", ":"))


def build_prompt(observation: WorkbenchObservation) -> str:
    return (
        "You are an expert data cleaning agent operating inside a structured environment.\n"
        "Return exactly one JSON object describing the next tool call.\n"
        "Allowed action_type values are inspect, fill_nulls, rename_column, cast_type, remove_duplicates, "
        "fix_value, normalize, drop_column, submit.\n"
        "Your first action must be inspect with no column.\n"
        "Inspect each column before modifying it.\n"
        "Do not batch actions.\n\n"
        f"Task: {observation.task_name}\n"
        f"Difficulty: {observation.difficulty}\n"
        f"Objective: {observation.objective}\n"
        f"Available tools: {json.dumps(observation.tools)}\n"
        f"Grading criteria: {json.dumps(observation.grading_criteria, indent=2)}\n"
        f"Dataset columns: {json.dumps(observation.dataset_columns)}\n"
        f"Current dtypes: {json.dumps(observation.current_dtypes, indent=2)}\n"
        f"Null counts: {json.dumps(observation.null_counts, indent=2)}\n"
        f"Duplicate rows: {observation.duplicate_rows}\n"
        f"Dataset preview: {json.dumps(observation.dataset_preview, indent=2)}\n"
        f"Tool result from last step: {json.dumps(observation.tool_result, indent=2)}\n"
        f"Inspected columns: {json.dumps(observation.inspected_columns)}\n"
        f"Last action error: {json.dumps(observation.last_action_error)}\n"
        f"Progress: {observation.progress}\n"
        "If all issues are resolved, return submit.\n"
    )


def parse_action(text: str) -> WorkbenchAction:
    payload = text.strip()
    if "```" in payload:
        chunks = [chunk.strip() for chunk in payload.split("```") if chunk.strip()]
        payload = chunks[-1]
        if payload.lower().startswith("json"):
            payload = payload[4:].strip()
    data = json.loads(payload)
    return WorkbenchAction(**data)


def choose_action(observation: WorkbenchObservation) -> WorkbenchAction:
    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[{"role": "user", "content": build_prompt(observation)}],
    )
    content = response.choices[0].message.content or '{"action_type":"submit"}'
    try:
        return parse_action(content)
    except Exception:
        return WorkbenchAction(action_type="submit")


def run_task(task_name: str = "data_cleaning") -> Dict[str, Any]:
    env = OpsWorkbenchEnv(default_task=task_name)
    observation = env.reset(task_name=task_name)
    rewards: List[float] = []
    steps = 0
    success = False

    print(f"[START] task={task_name} env=ops_workbench model={MODEL_NAME}")
    try:
        while True:
            action = choose_action(observation)
            observation, reward, done, _info = env.step(action)
            rewards.append(reward)
            steps += 1
            error = observation.last_action_error if observation.last_action_error is not None else "null"
            print(
                f"[STEP] step={steps} action={format_action(action)} reward={reward:.2f} "
                f"done={'true' if done else 'false'} error={error}"
            )
            if done:
                success = observation.progress > 0.95
                break
    finally:
        env.close()
        reward_str = ",".join(f"{value:.2f}" for value in rewards)
        print(f"[END] success={'true' if success else 'false'} steps={steps} rewards={reward_str}")

    return {
        "task_name": task_name,
        "success": success,
        "steps": steps,
        "final_score": observation.progress,
        "rewards": rewards,
    }


def main() -> None:
    result = run_task("data_cleaning")
    aggregate = {
        "model": MODEL_NAME,
        "benchmark": "ops_workbench",
        "tasks": [result],
"mean_score": round(max(0.01, min(0.98, mean(task["final_score"] for task in aggregate["tasks"]))), 4),
    }
    Path("results.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
